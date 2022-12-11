from model_list import MODEL
from utils import get_metrics, get_logger, get_name, count_parameters_in_MB
from torch.utils.tensorboard import SummaryWriter
from torch.nn.utils import clip_grad_norm_
import torch
import time
import numpy as np
from hyperopt import fmin, tpe, hp, Trials, partial, STATUS_OK, rand, space_eval
from os import mkdir, makedirs
from os.path import exists
from models.genotypes_new import NA_PRIMITIVES, LC_PRIMITIVES, LF_PRIMITIVES, SEQ_PRIMITIVES
# DEBUG
from hyperopt.pyll.stochastic import sample
from pprint import pprint
from itertools import product
from sortedcontainers import SortedDict

EPOCH_TEST = {"icews14/": 30,
              "icews05-15/": 10,
              "gdelt/": 30,
              "wikidata11k/": 50}


class Trainer(object):
    cnt_tune = 0

    def __init__(self, args, dataset_info_dict, train_loader, evaluate_loader, device):
        self.args = args
        self.device = device
        self.dataset_info_dict = dataset_info_dict
        self.train_loader = train_loader
        self.evaluate_loader = evaluate_loader
        self.optimizer = None
        self.scheduler = None
        self.search_space = None
        self.logger = None

    def train(self):
        name = get_name(self.args)
        log_dir = f'{self.args.log_dir}{self.args.dataset}{self.args.train_mode}/'
        if not exists(log_dir):
            mkdir(log_dir)
        self.logger = get_logger(name, log_dir)
        self.logger.info(self.args)
        writer = SummaryWriter(self.args.tensorboard_dir + self.args.dataset + name)
        model = MODEL[self.args.encoder](self.args, self.dataset_info_dict, self.device)
        model = model.cuda()
        self.logger.info("Parameter size = %fMB", count_parameters_in_MB(model))
        self.optimizer = torch.optim.Adam(model.parameters(), lr=self.args.learning_rate, weight_decay=0.0001)

        best_val_mrr, best_test_mrr = 0.0, 0.0
        early_stop_cnt = 0
        for epoch in range(1, self.args.max_epoch + 1):
            training_loss = self.train_epoch(epoch, model, architect=None, lr=None, mode="train")
            valid_mrr = self.evaluate_epoch(epoch, model, split="valid")
            if valid_mrr > best_val_mrr:
                early_stop_cnt = 0
                best_val_mrr = valid_mrr
                test_mrr = self.evaluate_epoch(epoch, model, split="test")
                if test_mrr > best_test_mrr:
                    best_test_mrr = test_mrr
                    self.logger.info("Success")
                    # torch.save(model.state_dict(), f'{args.saved_model_dir}{name}.pth')
            else:
                early_stop_cnt += 1
            if early_stop_cnt > 10:
                self.logger.info("Early stop!")
                self.logger.info(best_test_mrr)
                break
            writer.add_scalar('Loss/train', training_loss, epoch)
            writer.add_scalar('MRR/test', best_test_mrr, epoch)

    def random_bayesian_search(self):
        genotype_space = []

        for i in range(self.args.gnn_layer_num):
            genotype_space.append(hp.choice("G" + str(i), NA_PRIMITIVES))
            genotype_space.append(hp.choice("SEQ" + str(i), SEQ_PRIMITIVES))
            if i != self.args.gnn_layer_num - 1:
                genotype_space.append(hp.choice("LC" + str(i), LC_PRIMITIVES))
            else:
                genotype_space.append(hp.choice("LA" + str(i), LF_PRIMITIVES))
        trials = Trials()
        search_time = 0.0
        t_start = time.time()
        if self.args.search_mode == "random":
            best = fmin(self.train_parameter, genotype_space, algo=rand.suggest,
                        max_evals=self.args.baseline_sample_num,
                        trials=trials)
        elif self.args.search_mode == "bayesian":
            best = fmin(self.train_parameter, genotype_space,
                        algo=partial(tpe.suggest, n_startup_jobs=int(self.args.baseline_sample_num) / 5),
                        max_evals=self.args.baseline_sample_num,
                        trials=trials)
        else:
            raise NotImplementedError
        best_genotype = space_eval(genotype_space, best)
        t_end = time.time()
        search_time += (t_end - t_start)
        return "||".join(best_genotype)

    def evaluate_epoch(self, current_epoch, model, split="valid", evaluate_ws=False, mode=None):
        rank_list = []
        loss_list = []
        model.eval()
        with torch.no_grad():
            for batch_idx, timestamps in enumerate(self.evaluate_loader):
                if mode == "spos_train":
                    model.ent_encoder.ops = model.ent_encoder.generate_single_path()
                rank, loss = model.evaluate(timestamps, split, evaluate_ws=evaluate_ws)
                rank_list.append(rank)
                if split == 'valid' or split == 'train':
                    loss_list.append(loss.item())
                else:
                    loss_list.append(loss)
            if split == "train":
                self.logger.info(
                    '[Epoch:{} | {}]: Loss:{:.4}'.format(
                        current_epoch, split.capitalize() + ('_WS' if evaluate_ws else ""), np.mean(loss_list)))
                return np.mean(loss_list)
            else:
                all_ranks = torch.cat(rank_list)
                mrr, hit_1, hit_3, hit_10 = get_metrics(all_ranks)
                metrics_dict = {'mrr': mrr, 'hit_10': hit_10, 'hit_3': hit_3, 'hit_1': hit_1}
                metrics_result = {k: v.item() for k, v in metrics_dict.items()}
                # self.logger.info(
                #     '[Epoch:{} | {}]: {} Loss:{:.4}'.format(current_epoch, split.capitalize(), split.capitalize(), np.mean(loss_list)))
                self.logger.info('[Epoch:{} | {}]: Loss:{:.4}, MRR:{:.3}, Hits@10:{:.3}, Hits@3:{:.3}, Hits@1:{:.3}'.format(
                    current_epoch, split.capitalize() + ('_WS' if evaluate_ws else ""), np.mean(loss_list),
                    metrics_result['mrr'], metrics_result['hit_10'],
                    metrics_result['hit_3'],
                    metrics_result['hit_1']))
                return metrics_result['mrr'], np.mean(loss_list)

    def train_epoch(self, current_epoch, model, architect=None, lr=None, mode='NONE'):
        train_loss_list = []
        for batch_idx, train_timestamps in enumerate(self.train_loader):
            if mode == "spos_search":
                train_loss = model(train_timestamps)
                train_loss_list.append(train_loss.item())
            else:
                model.train()
                if mode == "spos_train":
                    model.ent_encoder.ops = model.ent_encoder.generate_single_path()
                self.optimizer.zero_grad()
                train_loss = model(train_timestamps)
                train_loss_list.append(train_loss.item())
                train_loss.backward()
                self.optimizer.step()
        self.logger.info('[Epoch:{} | {}]: Train Loss:{:.4}'.format(current_epoch, self.args.train_mode.capitalize(),
                                                                    np.mean(train_loss_list)))
        return np.mean(train_loss_list)

    def fine_tuning(self, genotype):
        hyper_space = {
            'weight_decay': hp.uniform("wr", -5, -3),
            'seq_head_num': hp.choice('seq_head_num', [2, 4, 8]),
            'head_num': hp.choice('head_num', [2, 4, 8]),
        }
        self.args.genotype = genotype
        trials = Trials()
        best = fmin(self.train_parameter, hyper_space,
                    algo=partial(tpe.suggest, n_startup_jobs=int(self.args.tune_sample_num) / 5),
                    max_evals=self.args.tune_sample_num,
                    trials=trials)
        space = space_eval(hyper_space, best)
        for k, v in space.items():
            setattr(self.args, k, v)
        best_val_mrr, best_test_mrr = 0.0, 0.0
        for d in trials.results:
            if -d["loss"] > best_val_mrr:
                best_val_mrr = -d["loss"]
                best_test_mrr = d["test_mrr"]
        with open(self.args.tune_res_dir + self.args.dataset + self.args.genotype, "w") as f1:
            f1.write(str(vars(self.args)) + "\n")
            f1.write(str(best_test_mrr))

    def train_parameter(self, parameter):
        Trainer.cnt_tune += 1
        self.args.index = Trainer.cnt_tune
        if self.args.train_mode == "search":
            self.args.genotype = "||".join(parameter)
        else:
            parameter['weight_decay'] = 10 ** parameter['weight_decay']
            for k, v in parameter.items():
                setattr(self.args, k, v)
        name = get_name(self.args)
        search_res_dir = self.args.search_res_file.split('/')[-1].split('.')[0]
        log_dir = f'{self.args.log_dir}{self.args.dataset}{self.args.train_mode}/{self.args.search_mode}/{self.args.encoder}/' \
                  f'{search_res_dir}/'
        if not exists(log_dir):
            makedirs(log_dir)
        self.logger = get_logger(name, log_dir)
        self.logger.info(self.args)
        model = MODEL[self.args.encoder](self.args, self.dataset_info_dict, self.device, self.args.genotype)
        model = model.cuda()
        self.logger.info("Parameter size = %fMB", count_parameters_in_MB(model))
        if self.args.optimizer == 'adam':
            self.optimizer = torch.optim.Adam(
                model.parameters(),
                self.args.learning_rate,
                weight_decay=self.args.weight_decay
            )
        elif self.args.optimizer == 'adagrad':
            self.optimizer = torch.optim.Adagrad(
                model.parameters(),
                self.args.learning_rate,
                weight_decay=self.args.weight_decay
            )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='max', factor=0.2, patience=10, verbose=True)
        best_valid_mrr, best_test_mrr = 0.0, 0.0
        early_stop_cnt = 0
        for epoch in range(1, self.args.max_epoch + 1):
            loss = self.train_epoch(epoch, model, mode="tune")
            valid_mrr, _ = self.evaluate_epoch(epoch, model, split="valid")
            if valid_mrr > best_valid_mrr:
                early_stop_cnt = 0
                best_valid_mrr = valid_mrr
                if self.args.train_mode == "tune" and epoch > EPOCH_TEST[self.args.dataset]:
                    test_mrr, _ = self.evaluate_epoch(epoch, model, split="test")
                    if test_mrr > best_test_mrr:
                        best_test_mrr = test_mrr
                        self.logger.info("Success")
            else:
                early_stop_cnt += 1
            if early_stop_cnt > 25 or epoch == self.args.max_epoch:
                self.logger.info("Early stop!")
                self.logger.info(f'{best_valid_mrr} {self.args.genotype}')
                break
            self.scheduler.step(best_valid_mrr)
        return {'loss': -best_valid_mrr, 'status': STATUS_OK} if self.args.train_mode == "search" else {'loss': -best_valid_mrr, 'test_mrr':best_test_mrr,'status': STATUS_OK}

    def debug(self, genotype):
        Trainer.cnt_tune += 1
        self.args.genotype = genotype
        name = get_name(self.args)
        log_dir = f'{self.args.log_dir}{self.args.dataset}{self.args.train_mode}/{self.args.encoder}/' \
                  f'{self.args.time_log_dir}_{self.args.random_seed}/'
        if not exists(log_dir):
            makedirs(log_dir)
        self.logger = get_logger(name, log_dir)
        self.logger.info(self.args)
        writer = SummaryWriter(
            f'{self.args.tensorboard_dir}{self.args.dataset}{self.args.train_mode}/{self.args.encoder}/{self.args.time_log_dir}_{self.args.random_seed}/')
        model = MODEL[self.args.encoder](self.args, self.dataset_info_dict, self.device, self.args.genotype)
        model = model.cuda()
        self.logger.info("Parameter size = %fMB", count_parameters_in_MB(model))
        self.optimizer = torch.optim.Adam(model.parameters(), lr=self.args.learning_rate,
                                          weight_decay=self.args.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='max', factor=0.2, patience=10, verbose=True)

        best_valid_mrr, best_test_mrr = 0.0, 0.0
        early_stop_cnt = 0
        for epoch in range(1, self.args.max_epoch + 1):
            train_loss = self.train_epoch(epoch, model, mode="tune")
            valid_mrr, valid_loss = self.evaluate_epoch(epoch, model, split="valid")
            if valid_mrr > best_valid_mrr:
                early_stop_cnt = 0
                best_valid_mrr = valid_mrr
                test_mrr, _ = self.evaluate_epoch(epoch, model, split="test")
                if test_mrr > best_test_mrr:
                    best_test_mrr = test_mrr
            else:
                early_stop_cnt += 1
            if early_stop_cnt > 25 or epoch == self.args.max_epoch:
                self.logger.info("Early stop!")
                self.logger.info(f'{best_valid_mrr} {self.args.genotype}')
                break
            self.scheduler.step(best_valid_mrr)
            # writer.add_scalar('Loss/train', train_loss, epoch)
            # writer.add_scalar('Loss/valid', valid_loss, epoch)
            writer.add_scalar('MRR/test', best_test_mrr, epoch)

    def spos_train_supernet(self):
        name = get_name(self.args)
        log_dir = f'{self.args.log_dir}{self.args.dataset}{self.args.train_mode}/{self.args.search_mode}/{self.args.encoder}/' \
                  f'{self.args.time_log_dir}_{self.args.random_seed}/'
        if not exists(log_dir):
            makedirs(log_dir)
        weights_dir = f'weights/{self.args.dataset}{self.args.train_mode}/{self.args.search_mode}/{self.args.encoder}/{self.args.time_log_dir}_{self.args.random_seed}/'
        if not exists(weights_dir):
            makedirs(weights_dir)
        self.logger = get_logger(name, log_dir)
        self.logger.info(self.args)
        self.logger.info(f'Log file is saved in {log_dir}')
        self.logger.info(f'Weight file is saved in {weights_dir}')
        writer = SummaryWriter(
            f'{self.args.tensorboard_dir}{self.args.dataset}{self.args.train_mode}/{self.args.search_mode}/{self.args.encoder}/{self.args.time_log_dir}/{self.args.random_seed}')
        model = MODEL[self.args.encoder](self.args, self.dataset_info_dict, self.device)
        model = model.cuda()
        self.optimizer = torch.optim.Adam(model.parameters(), lr=self.args.learning_rate,
                                          weight_decay=self.args.weight_decay)
        search_time = 0.0
        best_val_mrr, best_test_mrr = 0.0, 0.0
        for epoch in range(1, self.args.search_max_epoch + 1):
            t_start = time.time()
            train_loss = self.train_epoch(epoch, model, mode="spos_train")
            valid_mrr, valid_loss = self.evaluate_epoch(epoch, model, split="valid", evaluate_ws=False,
                                                        mode="spos_train")
            t_end = time.time()
            search_time += (t_end - t_start)
            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.add_scalar('Loss/valid', valid_loss, epoch)
            writer.add_scalar('MRR/valid', valid_mrr, epoch)
        torch.save(model.state_dict(), f'{weights_dir}epoch_{self.args.search_max_epoch}.pt')
        search_time = search_time / 3600
        self.logger.info(f'The search process costs {search_time:.2f}h.')

        return None

    def spos_arch_search(self):
        name = '_search_'+str(self.args.random_seed)
        log_dir = self.args.weight_path.replace('weights', 'logs', 1).split('.')[0]
        if not exists(log_dir):
            makedirs(log_dir)
        self.logger = get_logger(name, log_dir)
        self.logger.info(f'Log file is saved in {log_dir}')
        model = MODEL[self.args.encoder](self.args, self.dataset_info_dict, self.device)
        model = model.cuda()
        model.load_state_dict(torch.load(self.args.weight_path))
        self.logger.info(f'Finish loading checkpoint from {self.args.weight_path}')
        search_time = 0.0
        valid_mrr_searched_arch_res = SortedDict()
        for epoch in range(1, self.args.arch_sample_num + 1):
            model.ent_encoder.ops = model.ent_encoder.generate_single_path()
            arch = "||".join(model.ent_encoder.ops)
            t_start = time.time()
            valid_mrr, valid_loss = self.evaluate_epoch(epoch, model, split="valid", evaluate_ws=False)
            valid_mrr_searched_arch_res.setdefault(valid_mrr, arch)
            self.logger.info('[Epoch:{} | {}]: Path:{}'.format(epoch, self.args.arch_sample_num, arch))
            t_end = time.time()
            search_time += (t_end - t_start)
        search_time = search_time / 3600
        self.logger.info(f'The search process costs {search_time:.2f}h.')
        import csv

        with open(log_dir+'_search_valid_mrr_'+str(self.args.random_seed)+'_res.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['valid mrr', 'arch'])
            res = valid_mrr_searched_arch_res.items()
            for i in range(500):
                writer.writerow([res[-1-i][0], res[-1-i][1]])

        return res[-1][1]
