import torch
import dgl
from torch.nn import Module
import torch.nn as nn
import numpy as np
from utils import compute_norm, node_norm_to_edge_norm, filter_none
import torch.nn.functional as F
from pprint import pprint


class DynamicBaseModel(Module):
    def __init__(self, args, dataset_info_dict, device):
        super(DynamicBaseModel, self).__init__()
        self.args = args
        self.dataset_info_dict = dataset_info_dict
        self.dataset_graph = dataset_info_dict['time2graph_dict']
        self.dataset_fact = dataset_info_dict['time2fact_dict']
        self.device = device
        self.train_sr2o = dataset_info_dict['negative_sr2o_dict']
        self.sr2o = dataset_info_dict['filter_sr2o_dict']
        self.ent_num = dataset_info_dict['entity_num']
        self.rel_num = dataset_info_dict['relation_num']
        self.ent_embeds = nn.Parameter(torch.Tensor(self.ent_num, self.args.embed_size))
        self.rel_embeds = nn.Parameter(torch.Tensor(self.rel_num, self.args.embed_size))
        self.get_true_hear_and_tail()
        self.get_true_head_and_tail_all()

        nn.init.xavier_uniform_(self.ent_embeds, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.rel_embeds, gain=nn.init.calculate_gain('relu'))

        self.train_seq_len = args.train_seq_len
        self.test_seq_len = args.train_seq_len

        self.ent_encoder = None

    def forward(self, timestamps):
        total_loss = 0.0
        g_batched_list, time_batched_list = self.get_batch_graph_list(timestamps, self.train_seq_len,
                                                                      self.dataset_graph, split="train")
        hist_embeddings, start_time_tensor, hist_embeddings_transformer, attn_mask = self.pre_forward(g_batched_list,
                                                                                                      time_batched_list)
        train_graphs, time_batched_list_t = g_batched_list[-1], time_batched_list[-1]
        prev_graph_embeds_list, time_diff_tensor, prev_graph_embeds_transformer_list, local_attn_mask = self.get_prev_embeddings(
            train_graphs,
            hist_embeddings,
            start_time_tensor,
            self.train_seq_len - 1, hist_embeddings_transformer, attn_mask)
        node_sizes = [len(g.nodes()) for g in train_graphs]
        _, per_graph_ent_embeds = self.get_per_graph_ent_embeds(train_graphs, time_batched_list_t, node_sizes,
                                                                   time_diff_tensor, prev_graph_embeds_list,
                                                                   prev_graph_embeds_transformer_list, local_attn_mask,
                                                                   full=False)
        i = 0
        for t, g, ent_embed in zip(time_batched_list_t, train_graphs, per_graph_ent_embeds):
            triplets, neg_tail_samples, neg_head_samples, labels = self.single_graph_negative_sampling(t, g,
                                                                                                       self.ent_num)
            all_embeds_g = self.get_all_embeds_Gt(ent_embed, g, t, hist_embeddings[i],
                                                  self.train_seq_len - 1 - start_time_tensor[i])
            loss_tail = self.train_link_prediction(ent_embed, triplets, neg_tail_samples, labels, all_embeds_g,
                                                   corrupt_tail=True)
            loss_head = self.train_link_prediction(ent_embed, triplets, neg_head_samples, labels, all_embeds_g,
                                                   corrupt_tail=False)
            total_loss += loss_tail + loss_head
            i += 1
        return total_loss

    def evaluate(self, t_list, split='valid', evaluate_ws=False):
        # hist_embeddings = self.ent_embeds.new_zeros(bsz, 2, self.num_ents, self.embed_size)
        # start_time_tensor = self.ent_embeds.new_zeros(bsz, self.num_ents)
        #
        # for cur_t in range(self.test_seq_len - 1):
        #     g_batched_list_t, node_sizes = self.get_val_vars(g_train_batched_list, cur_t)
        #     if len(g_batched_list_t) == 0: continue
        #
        #     first_prev_graph_embeds, second_prev_graph_embeds, time_diff_tensor = self.get_prev_embeddings(g_batched_list_t, hist_embeddings, start_time_tensor, cur_t)
        #     first_per_graph_ent_embeds, second_per_graph_ent_embeds = self.get_per_graph_ent_embeds(g_batched_list_t, time_list[cur_t], node_sizes, time_diff_tensor, first_prev_graph_embeds, second_prev_graph_embeds, full=True)
        #     hist_embeddings = self.update_time_diff_hist_embeddings(first_per_graph_ent_embeds, second_per_graph_ent_embeds, start_time_tensor, g_batched_list_t, cur_t, bsz)
        per_graph_ent_embeds, test_graphs, time_list, hist_embeddings, start_time_tensor = self.evaluate_embed(t_list,
                                                                                                               split,
                                                                                                               evaluate_ws)
        return self.calc_metrics(per_graph_ent_embeds, test_graphs, time_list[-1], hist_embeddings, start_time_tensor,
                                 self.test_seq_len - 1, split)

    def evaluate_embed(self, t_list, split='valid', evaluate_ws=False):
        # graph_dict = self.graph_dict_val if val else self.graph_dict_test
        g_train_batched_list, time_list = self.get_batch_graph_list(t_list, self.train_seq_len, self.dataset_graph,
                                                                    split="train")
        if split == 'valid':
            g_val_batched_list, val_time_list = self.get_batch_graph_list(t_list, 1, self.dataset_graph, split="valid")
        elif split == 'test':
            g_val_batched_list, val_time_list = self.get_batch_graph_list(t_list, 1, self.dataset_graph, split="test")

        hist_embeddings, start_time_tensor, hist_embeddings_transformer, attn_mask = self.pre_forward(
            g_train_batched_list, time_list, val=True)
        # test_graphs, _ = self.get_val_vars(g_val_batched_list, -1)
        train_graphs = g_train_batched_list[-1]

        prev_graph_embeds_list, time_diff_tensor, prev_graph_embeds_transformer_list, local_attn_mask = self.get_prev_embeddings(
            train_graphs,
            hist_embeddings,
            start_time_tensor,
            self.test_seq_len - 1, hist_embeddings_transformer, attn_mask)
        node_sizes = [len(g.nodes()) for g in train_graphs]
        _, per_graph_ent_embeds = self.get_per_graph_ent_embeds(train_graphs, time_list[-1], node_sizes,
                                                                   time_diff_tensor, prev_graph_embeds_list,
                                                                   prev_graph_embeds_transformer_list, local_attn_mask,
                                                                   full=True,
                                                                   evaluate_ws=evaluate_ws)
        if split == "train":
            graphs = train_graphs
        else:
            test_graphs, _ = self.get_val_vars(g_val_batched_list, -1)
            graphs = test_graphs
        return per_graph_ent_embeds, graphs, time_list, hist_embeddings, start_time_tensor

    def calc_metrics(self, per_graph_ent_embeds, g_list, t_list, hist_embeddings, start_time_tensor, cur_t, split='valid'):
        mrrs, hit_1s, hit_3s, hit_10s, losses = [], [], [], [], []
        total_loss = 0.0
        ranks = []
        i = 0
        for g, t, ent_embed in zip(g_list, t_list, per_graph_ent_embeds):
            if split == 'train':
                triplets, neg_tail_samples, neg_head_samples, labels = self.single_graph_negative_sampling(t, g,
                                                                                                               self.ent_num)
                all_embeds_g = self.get_all_embeds_Gt(ent_embed, g, t, hist_embeddings[i],
                                                          self.train_seq_len - 1 - start_time_tensor[i])
                loss_tail = self.train_link_prediction(ent_embed, triplets, neg_tail_samples, labels, all_embeds_g,
                                                           corrupt_tail=True)
                loss_head = self.train_link_prediction(ent_embed, triplets, neg_head_samples, labels, all_embeds_g,
                                                           corrupt_tail=False)
                total_loss += loss_tail + loss_head
                i += 1
            else:
                if split == 'valid':
                    triplets, neg_tail_samples, neg_head_samples, labels = self.single_graph_negative_sampling(t, g,
                                                                                                               self.ent_num, mode='valid')
                time_diff_tensor = cur_t - start_time_tensor[i]
                all_embeds_g = self.get_all_embeds_Gt(ent_embed, g, t, hist_embeddings[i],
                                                      time_diff_tensor)

                index_sample = torch.stack([g.edges()[0], g.edata['id'], g.edges()[1]]).transpose(0, 1)
                label = torch.ones(index_sample.shape[0])
                index_sample = index_sample.cuda()
                label = label.cuda()
                if index_sample.shape[0] == 0: continue
                rank = self.calc_metrics_single_graph(ent_embed, self.rel_embeds, all_embeds_g, index_sample, g, t)
                # loss = self.valid_link_prediction(ent_embed, self.rel_embeds, index_sample, label)
                if split == 'valid':
                    loss_tail = self.valid_link_prediction_new(ent_embed, self.rel_embeds, triplets, neg_tail_samples, labels, all_embeds_g,
                                                           corrupt_tail=True)
                    loss_head = self.valid_link_prediction_new(ent_embed, self.rel_embeds, triplets, neg_head_samples, labels, all_embeds_g,
                                                           corrupt_tail=False)
                    total_loss += loss_tail + loss_head
                ranks.append(rank)
                # total_loss += loss_tail + loss_head
                # losses.append(loss.item())
                i += 1
        try:
            ranks = torch.cat(ranks)
        except:
            ranks = torch.tensor([]).long().cuda()

        return ranks, total_loss

    def valid_link_prediction(self, ent_embed, rel_embeds, triplets, labels):
        pprint(triplets)
        pprint(labels.shape)
        s = ent_embed[triplets[:, 0]]
        r = rel_embeds[triplets[:, 1]]
        o = ent_embed[triplets[:, 2]]
        score = self.decoder(s, r, o)
        pprint(score)
        pprint(labels)
        predict_loss = F.cross_entropy(score, labels)
        # predict_loss = F.binary_cross_entropy_with_logits(score, labels)
        # pprint(predict_loss)
        # predict_loss = torch.where(torch.isnan(predict_loss), torch.full_like(predict_loss, 0), predict_loss)
        return predict_loss

    def valid_link_prediction_new(self, ent_embed, rel_embeds, triplets, neg_samples, labels, all_embeds_g, corrupt_tail=True):
        # pprint(triplets)
        # pprint(triplets.shape)
        # pprint(labels)
        # pprint(labels.shape)
        # pprint(neg_samples)
        # pprint(neg_samples.shape)
        # exit(0)
        r = rel_embeds[triplets[:, 1]]
        if corrupt_tail:
            s = ent_embed[triplets[:, 0]]
            neg_o = all_embeds_g[neg_samples]
            score = self.decoder(s, r, neg_o, score_function=self.args.score_function, mode='tail')
        else:
            neg_s = all_embeds_g[neg_samples]
            o = ent_embed[triplets[:, 2]]
            score = self.decoder(neg_s, r, o, score_function=self.args.score_function, mode='head')
        predict_loss = F.cross_entropy(score, labels)
        # predict_loss = F.binary_cross_entropy_with_logits(score, labels)
        predict_loss = torch.where(torch.isnan(predict_loss), torch.full_like(predict_loss, 0), predict_loss)
        return predict_loss

    def calc_metrics_single_graph(self, ent_mean, rel_enc_means, all_ent_embeds, samples, graph, time, eval_bz=100):
        with torch.no_grad():
            s = samples[:, 0]
            r = samples[:, 1]
            o = samples[:, 2]
            test_size = samples.shape[0]
            num_ent = all_ent_embeds.shape[0]
            # pdb.set_trace()
            o_mask = self.mask_eval_set(samples, test_size, num_ent, time, graph, mode="tail")
            s_mask = self.mask_eval_set(samples, test_size, num_ent, time, graph, mode="head")
            # perturb object
            ranks_o = self.perturb_and_get_rank(ent_mean, rel_enc_means, all_ent_embeds, s, r, o, test_size, o_mask,
                                                graph, eval_bz, mode='tail')
            # perturb subject
            ranks_s = self.perturb_and_get_rank(ent_mean, rel_enc_means, all_ent_embeds, s, r, o, test_size, s_mask,
                                                graph, eval_bz, mode='head')
            ranks = torch.cat([ranks_s, ranks_o])
            ranks += 1  # change to 1-indexed
            # print("Graph {} mean ranks {}".format(time.item(), ranks.float().mean().item()))
        return ranks

    def mask_eval_set(self, test_triplets, test_size, num_ent, time, graph, mode='tail'):
        time = int(time)
        mask = test_triplets.new_zeros(test_size, num_ent)
        for i in range(test_size):
            h, r, t = test_triplets[i]
            h, r, t = h.item(), r.item(), t.item()
            if mode == 'tail':
                tails = self.true_tails[time][(h, r)]
                tail_idx = np.array(list(map(lambda x: graph.ids[x], tails)))
                mask[i][tail_idx] = 1
                mask[i][graph.ids[t]] = 0
            elif mode == 'head':
                heads = self.true_heads[time][(r, t)]
                head_idx = np.array(list(map(lambda x: graph.ids[x], heads)))
                mask[i][head_idx] = 1
                mask[i][graph.ids[h]] = 0
        return mask > 0

    def perturb_and_get_rank(self, ent_mean, rel_enc_means, all_ent_embeds, s, r, o, test_size, mask, graph,
                             batch_size=100, mode='tail'):
        """ Perturb one element in the triplets
        """
        n_batch = (test_size + batch_size - 1) // batch_size
        ranks = []
        for idx in range(n_batch):
            batch_start = idx * batch_size
            batch_end = min(test_size, (idx + 1) * batch_size)
            batch_r = rel_enc_means[r[batch_start: batch_end]]
            if mode == 'tail':
                batch_s = ent_mean[s[batch_start: batch_end]]
                batch_o = all_ent_embeds
                target = o[batch_start: batch_end]
            else:
                batch_s = all_ent_embeds
                batch_o = ent_mean[o[batch_start: batch_end]]
                target = s[batch_start: batch_end]
            target = torch.tensor([graph.ids[i.item()] for i in target])

            target = target.cuda()

            unmasked_score = self.decoder(batch_s, batch_r, batch_o, mode=mode)
            masked_score = torch.where(mask[batch_start: batch_end],
                                       -10e6 * unmasked_score.new_ones(unmasked_score.shape), unmasked_score)
            score = torch.sigmoid(masked_score)  # bsz, n_ent
            ranks.append(self.sort_and_rank(score, target))
        return torch.cat(ranks)

    def sort_and_rank(self, score, target):
        # pdb.set_trace()
        _, indices = torch.sort(score, dim=1, descending=True)
        indices = torch.nonzero(indices == target.view(-1, 1))
        indices = indices[:, 1].view(-1)
        return indices

    def train_link_prediction(self, ent_embed, triplets, neg_samples, labels, all_embeds_g, corrupt_tail=True):
        # pprint(triplets)
        # pprint(triplets.shape)
        # pprint(labels)
        # pprint(labels.shape)
        r = self.rel_embeds[triplets[:, 1]]
        if corrupt_tail:
            s = ent_embed[triplets[:, 0]]
            neg_o = all_embeds_g[neg_samples]
            score = self.decoder(s, r, neg_o, score_function=self.args.score_function, mode='tail')
        else:
            neg_s = all_embeds_g[neg_samples]
            o = ent_embed[triplets[:, 2]]
            score = self.decoder(neg_s, r, o, score_function=self.args.score_function, mode='head')
        # pprint(score.shape)
        # exit(0)
        predict_loss = F.cross_entropy(score, labels)
        # predict_loss = F.binary_cross_entropy_with_logits(score, labels)
        predict_loss = torch.where(torch.isnan(predict_loss), torch.full_like(predict_loss, 0), predict_loss)
        return predict_loss

    def decoder(self, head, relation, tail, score_function='complex', mode="single"):
        if score_function == 'distmult':
            if mode == 'tail':
                return torch.sum((head * relation).unsqueeze(1) * tail, dim=-1)
            elif mode == 'head':
                return torch.sum(head * (relation * tail).unsqueeze(1), dim=-1)
            else:
                return torch.sum(head * relation * tail, dim=-1)
        elif score_function == 'complex':
            re_head, im_head = torch.chunk(head, 2, dim=-1)
            re_relation, im_relation = torch.chunk(relation, 2, dim=-1)
            re_tail, im_tail = torch.chunk(tail, 2, dim=-1)
            if mode == 'tail':
                re_score = re_head * re_relation - im_head * im_relation
                im_score = re_head * im_relation + im_head * re_relation
                score = re_score.unsqueeze(1) * re_tail + im_score.unsqueeze(1) * im_tail
            elif mode == 'head':
                re_score = re_relation * re_tail + im_relation * im_tail
                im_score = re_relation * im_tail - im_relation * re_tail
                score = re_head * re_score.unsqueeze(1) + im_head * im_score.unsqueeze(1)
            else:
                re_score = re_head * re_relation - im_head * im_relation
                im_score = re_head * im_relation + im_head * re_relation
                score = re_score * re_tail + im_score * im_tail

            return score.sum(dim=-1)

    def get_all_embeds_Gt(self, convoluted_embeds, g, t, prev_graph_embeds_list,
                          time_diff_tensor):
        all_embeds_g = self.ent_embeds.new_zeros(self.ent_embeds.shape)
        # pprint(all_embeds_g.shape)
        # pprint(convoluted_embeds.shape)
        if not self.args.isolated_change:
            all_embeds_g[:] = self.ent_embeds[:]
        else:
            all_embeds_g = self.ent_encoder.forward_isolated(self.ent_embeds, prev_graph_embeds_list,
                                                             time_diff_tensor.unsqueeze(-1))
        for k, v in g.ids.items():
            all_embeds_g[v] = convoluted_embeds[k]
        return all_embeds_g

    @staticmethod
    def get_batch_graph_list(t_list, seq_len, graph_dict, split):
        times = list(graph_dict.keys())
        times.sort()
        # time_unit = times[1] - times[0]  # compute time unit
        time_list = []
        t_list = t_list.sort(descending=True)[0]
        g_list = []
        # s_lst = [t/15 for t in times]
        # print(s_lst)
        for tim in t_list:
            # length = int(tim / time_unit) + 1
            # cur_seq_len = seq_len if seq_len <= length else length
            length = times.index(tim) + 1
            time_seq = times[length - seq_len:length] if seq_len <= length else times[:length]
            time_list.append(([None] * (seq_len - len(time_seq))) + time_seq)
            g_list.append(
                ([None] * (seq_len - len(time_seq))) + [graph_dict[t].__getattribute__(split) for t in time_seq])
        t_batched_list = [list(x) for x in zip(*time_list)]
        g_batched_list = [list(x) for x in zip(*g_list)]
        return g_batched_list, t_batched_list

    def pre_forward(self, g_batched_list, time_batched_list, val=False):
        seq_len = self.test_seq_len if val else self.train_seq_len
        bsz = len(g_batched_list[0])
        target_time_batched_list = time_batched_list[-1]
        hist_embeddings = self.ent_embeds.new_zeros(bsz, self.args.gnn_layer_num, self.ent_num, self.args.embed_size)
        start_time_tensor = self.ent_embeds.new_zeros(bsz, self.ent_num)
        hist_embeddings_transformer = self.ent_embeds.new_zeros(seq_len - 1, bsz, self.args.gnn_layer_num, self.ent_num,
                                                                self.args.embed_size)
        attn_mask = self.ent_embeds.new_zeros(seq_len, bsz, self.ent_num) - 10e9
        attn_mask[-1] = 0
        full = val
        self.edge_dropout = False
        for cur_t in range(seq_len - 1):
            g_batched_list_t, node_sizes = self.get_val_vars(g_batched_list, cur_t)
            if len(g_batched_list_t) == 0: continue
            prev_graph_embeds_list, time_diff_tensor, prev_graph_embeds_transformer_list, local_attn_mask = self.get_prev_embeddings(
                g_batched_list_t, hist_embeddings, start_time_tensor, cur_t, hist_embeddings_transformer, attn_mask)
            if self.edge_dropout and not val:
                first_per_graph_ent_embeds, second_per_graph_ent_embeds = self.get_per_graph_ent_dropout_embeds(
                    time_batched_list[cur_t], target_time_batched_list, node_sizes,
                    time_diff_tensor, prev_graph_embeds_list)
            else:
                per_graph_ent_embeds_list, _ = self.get_per_graph_ent_embeds(
                    g_batched_list_t, time_batched_list[cur_t], node_sizes,
                    time_diff_tensor, prev_graph_embeds_list, prev_graph_embeds_transformer_list, local_attn_mask,
                    full=full, rate=0.8)
                per_graph_ent_embeds_transformer_list = self.get_per_graph_ent_embeds_transformer(
                    g_batched_list_t, time_batched_list[cur_t], node_sizes,
                    full=full, rate=0.8)
            hist_embeddings = self.update_time_diff_hist_embeddings(per_graph_ent_embeds_list, start_time_tensor,
                                                                    g_batched_list_t, cur_t, bsz)
            hist_embeddings_transformer, attn_mask = self.update_time_diff_hist_embeddings_transformer(
                per_graph_ent_embeds_transformer_list,
                hist_embeddings_transformer,
                g_batched_list_t, cur_t,
                attn_mask, bsz)

        return hist_embeddings, start_time_tensor, hist_embeddings_transformer, attn_mask

    def get_per_graph_ent_embeds(self, g_batched_list_t, time_batched_list_t, node_sizes, time_diff_tensor,
                                 prev_graph_embeds_list, prev_graph_embeds_transformer_list, local_attn_mask, full,
                                 rate=0.5, evaluate_ws=False, pre=True):
        batched_graph = self.get_batch_graph_embeds(g_batched_list_t, full, rate)
        batched_graph = batched_graph.to(self.device)
        # if self.use_cuda:
        #     move_dgl_to_cuda(batched_graph)
        if evaluate_ws:
            layer_embeds_list, final_embed = self.ent_encoder(batched_graph, prev_graph_embeds_list, time_diff_tensor,
                                                              prev_graph_embeds_transformer_list, local_attn_mask,
                                                              mode='evaluate_single_path')
        else:
            layer_embeds_list, final_embed = self.ent_encoder(batched_graph, prev_graph_embeds_list, time_diff_tensor,
                                                              prev_graph_embeds_transformer_list, local_attn_mask,
                                                              mode=None)
        per_graph_ent_embeds_list = []
        for layer_idx in range(self.args.gnn_layer_num):
            per_graph_ent_embeds_list.append(layer_embeds_list[layer_idx].split(node_sizes))
        return per_graph_ent_embeds_list, final_embed.split(node_sizes)
        # return first_layer_embeds.split(node_sizes), second_layer_embeds.split(node_sizes)

    def get_per_graph_ent_embeds_transformer(self, g_batched_list_t, time_batched_list_t, node_sizes, full, rate=0.5, evaluate_ws=False):
        batched_graph = self.get_batch_graph_embeds(g_batched_list_t, full, rate)
        batched_graph = batched_graph.to(self.device)
        # if self.use_cuda:
        #     move_dgl_to_cuda(batched_graph)
        if evaluate_ws:
            layer_embeds_transformer_list = self.ent_encoder.forward_pre(batched_graph, mode='evaluate_single_path')
        else:
            layer_embeds_transformer_list = self.ent_encoder.forward_pre(batched_graph, mode=None)
        per_graph_ent_embeds_transformer_list = []
        for layer_idx in range(self.args.gnn_layer_num):
            per_graph_ent_embeds_transformer_list.append(layer_embeds_transformer_list[layer_idx].split(node_sizes))
        return per_graph_ent_embeds_transformer_list

    def get_batch_graph_embeds(self, g_batched_list_t, full, rate):
        if full:
            sampled_graph_list = g_batched_list_t
        else:
            sampled_graph_list = []
            for g in g_batched_list_t:
                src, rel, dst = g.edges()[0], g.edata['id'], g.edges()[1]
                total_idx = np.random.choice(np.arange(src.shape[0]), size=int(rate * src.shape[0]), replace=False)
                sg = g.edge_subgraph(total_idx, preserve_nodes=True)
                norm = compute_norm(sg)
                sg.ndata.update({'id': g.ndata['id'], 'norm': norm.view(-1, 1)})
                # sg.edata['norm'] = node_norm_to_edge_norm(sg, torch.from_numpy(node_norm).view(-1, 1))
                sg.edata['id'] = rel[total_idx]
                sg.ids = g.ids
                sampled_graph_list.append(sg)

        batched_graph = dgl.batch(sampled_graph_list)
        batched_graph = batched_graph.to(self.device)
        batched_graph.ndata['ft'] = self.ent_embeds[batched_graph.ndata['id']].view(-1, self.args.embed_size)
        batched_graph.edata['ft'] = self.rel_embeds[batched_graph.edata['id']].view(-1, self.args.embed_size)
        return batched_graph

    def update_time_diff_hist_embeddings(self, per_graph_ent_embeds_list,
                                         start_time_tensor, g_batched_list_t, cur_t, bsz):
        res = start_time_tensor.new_zeros(bsz, self.args.gnn_layer_num, self.ent_num, self.args.embed_size)
        for i in range(len(per_graph_ent_embeds_list[0])):
            idx = g_batched_list_t[i].ndata['id'].squeeze()
            for layer_idx in range(self.args.gnn_layer_num):
                res[i][layer_idx][idx] = per_graph_ent_embeds_list[layer_idx][i]
                # res[i][1][idx] = second_per_graph_ent_embeds[i]
            start_time_tensor[i][idx] = cur_t
        return res

    def update_time_diff_hist_embeddings_transformer(self, per_graph_ent_embeds_list,
                                                     hist_embeddings_transformer, g_batched_list_t, cur_t, attn_mask,
                                                     bsz):
        # res = start_time_tensor.new_zeros(bsz, 2, self.ent_num, self.args.embed_size)
        for i in range(len(per_graph_ent_embeds_list[0])):
            idx = g_batched_list_t[i].ndata['id'].squeeze()
            attn_mask[cur_t][i][idx] = 0
            for layer_idx in range(self.args.gnn_layer_num):
                hist_embeddings_transformer[cur_t][i][layer_idx][idx] = per_graph_ent_embeds_list[layer_idx][i]
                # hist_embeddings_transformer[cur_t][i][1][idx] = second_per_graph_ent_embeds[i]
            # res[i][0][idx] = first_per_graph_ent_embeds[i]
            # res[i][1][idx] = second_per_graph_ent_embeds[i]
            # start_time_tensor[i][idx] = cur_t
        return hist_embeddings_transformer, attn_mask

    def get_prev_embeddings(self, g_batched_list_t, history_embeddings, start_time_tensor, cur_t,
                            hist_embeddings_transformer, attn_mask):
        # first_layer_prev_embeddings = []
        # second_layer_prev_embeddings = []
        layer_prev_embeddings_list = []
        layer_prev_embeddings_transformer_list = []
        prev_graph_embeds_list = []
        prev_graph_embeds_transformer_list = []
        for layer_idx in range(self.args.gnn_layer_num):
            layer_prev_embeddings_list.append([])
            layer_prev_embeddings_transformer_list.append([])
        time_diff_tensor = []
        local_attn_mask = []
        for i, graph in enumerate(g_batched_list_t):
            node_idx = graph.ndata['id']
            # pprint(node_idx.shape)
            # pprint(hist_embeddings_transformer[:][i][2][node_idx].shape)
            for layer_idx in range(self.args.gnn_layer_num):
                # first_layer_prev_embeddings.append(history_embeddings[i][0][node_idx].view(-1, self.args.embed_size))
                # second_layer_prev_embeddings.append(history_embeddings[i][1][node_idx].view(-1, self.args.embed_size))
                layer_prev_embeddings_list[layer_idx].append(
                    history_embeddings[i][layer_idx][node_idx].view(-1, self.args.embed_size))
                layer_prev_embeddings_transformer_list[layer_idx].append(
                    hist_embeddings_transformer[:, i, layer_idx, node_idx.squeeze()])
            time_diff_tensor.append(cur_t - start_time_tensor[i][node_idx])
            local_attn_mask.append(attn_mask[:, i, node_idx.squeeze()])
        for layer_idx in range(self.args.gnn_layer_num):
            prev_graph_embeds_list.append(torch.cat(layer_prev_embeddings_list[layer_idx]))
            # pprint(layer_prev_embeddings_list[layer_idx].shape)
            prev_graph_embeds_transformer_list.append(
                torch.cat(layer_prev_embeddings_transformer_list[layer_idx], dim=1).transpose(0, 1))
        return prev_graph_embeds_list, torch.cat(time_diff_tensor), prev_graph_embeds_transformer_list, torch.cat(
            local_attn_mask, dim=1).transpose(0, 1)
        # return torch.cat(first_layer_prev_embeddings), torch.cat(second_layer_prev_embeddings), torch.cat(
        #     time_diff_tensor)

    def get_true_hear_and_tail(self):
        self.true_heads_train = dict()
        self.true_tails_train = dict()
        self.true_heads_valid = dict()
        self.true_tails_valid = dict()
        for t in list(self.dataset_graph.keys()):
            g_train = self.dataset_graph[t].train
            triples_train = torch.stack([g_train.edges()[0], g_train.edata['id'], g_train.edges()[1]]).transpose(0, 1)
            true_head_train, true_tail_train = self.get_true_head_and_tail_per_graph(triples_train)
            self.true_heads_train[t] = true_head_train
            self.true_tails_train[t] = true_tail_train

            g_valid = self.dataset_graph[t].valid
            triples_valid = torch.stack([g_valid.edges()[0], g_valid.edata['id'], g_valid.edges()[1]]).transpose(0, 1)
            true_head_valid, true_tail_valid = self.get_true_head_and_tail_per_graph(triples_valid)
            self.true_heads_valid[t] = true_head_valid
            self.true_tails_valid[t] = true_tail_valid

    def get_true_head_and_tail_all(self):
        self.true_heads = dict()
        self.true_tails = dict()
        times = list(self.dataset_graph.keys())
        for t in times:
            triples = []
            for g in self.dataset_graph[t].train, self.dataset_graph[t].valid, self.dataset_graph[t].test:
                triples.append(torch.stack([g.edges()[0], g.edata['id'], g.edges()[1]]).transpose(0, 1))
            triples = torch.cat(triples, dim=0)
            true_head, true_tail = self.get_true_head_and_tail_per_graph(triples)
            self.true_heads[t] = true_head
            self.true_tails[t] = true_tail

    @staticmethod
    def get_true_head_and_tail_per_graph(triples):
        true_head = {}
        true_tail = {}
        for head, relation, tail in triples:
            head, relation, tail = head.item(), relation.item(), tail.item()
            if (head, relation) not in true_tail:
                true_tail[(head, relation)] = []
            true_tail[(head, relation)].append(tail)
            if (relation, tail) not in true_head:
                true_head[(relation, tail)] = []
            true_head[(relation, tail)].append(head)

        # this is correct
        for relation, tail in true_head:
            true_head[(relation, tail)] = np.array(list(set(true_head[(relation, tail)])))
        for head, relation in true_tail:
            true_tail[(head, relation)] = np.array(list(set(true_tail[(head, relation)])))

        return true_head, true_tail

    def get_val_vars(self, g_batched_list, cur_t):
        g_batched_list_t = filter_none(g_batched_list[cur_t])
        # run RGCN on graph to get encoded ent_embeddings and rel_embeddings in G_t
        node_sizes = [len(g.nodes()) for g in g_batched_list_t]
        return g_batched_list_t, node_sizes

    def single_graph_negative_sampling(self, t, g, num_ents, mode='train'):
        triples = torch.stack([g.edges()[0], g.edata['id'], g.edges()[1]]).transpose(0, 1)
        if mode == 'train':
            sample, neg_tail_sample, neg_head_sample, label = self.negative_sampling(self.true_heads_train[t],
                                                                                 self.true_tails_train[t], triples,
                                                                                 num_ents, g, mode='train')
        else:
            sample, neg_tail_sample, neg_head_sample, label = self.negative_sampling(self.true_heads_valid[t],
                                                                                 self.true_tails_valid[t], triples,
                                                                                 num_ents, g, mode=mode)
        neg_tail_sample, neg_head_sample, label = torch.from_numpy(neg_tail_sample), torch.from_numpy(
            neg_head_sample), torch.from_numpy(label)
        sample, neg_tail_sample, neg_head_sample, label = sample.cuda(), neg_tail_sample.cuda(), neg_head_sample.cuda(), label.cuda()
        return sample, neg_tail_sample, neg_head_sample, label

    def negative_sampling(self, true_head, true_tail, triples, num_entities, g, mode='train'):
        if mode=='train':
            negative_sampling_num = self.args.negative_sampling_num
            size_of_batch = min(triples.shape[0], self.args.positive_fact_num)
            if self.args.positive_fact_num < triples.shape[0]:
                rand_idx = torch.randperm(triples.shape[0])
                triples = triples[rand_idx[:self.args.positive_fact_num]]
        else:
            negative_sampling_num = self.args.negative_sampling_num
            size_of_batch = min(triples.shape[0], self.args.positive_fact_num)
            if self.args.positive_fact_num < triples.shape[0]:
                rand_idx = torch.randperm(triples.shape[0])
                triples = triples[rand_idx[:self.args.positive_fact_num]]
        neg_tail_samples = np.zeros((size_of_batch, 1 + negative_sampling_num), dtype=int)
        neg_head_samples = np.zeros((size_of_batch, 1 + negative_sampling_num), dtype=int)
        neg_tail_samples[:, 0] = triples[:, 2]
        neg_head_samples[:, 0] = triples[:, 0]

        labels = np.zeros(size_of_batch, dtype=int)

        for i in range(size_of_batch):
            h, r, t = triples[i]
            h, r, t = h.item(), r.item(), t.item()
            tail_samples = self.corrupt_triple(h, r, t, true_head, true_tail, num_entities, g, negative_sampling_num, True)
            head_samples = self.corrupt_triple(h, r, t, true_head, true_tail, num_entities, g, negative_sampling_num, False)

            neg_tail_samples[i][0] = g.ids[triples[i][2].item()]
            neg_head_samples[i][0] = g.ids[triples[i][0].item()]
            neg_tail_samples[i, 1:] = tail_samples
            neg_head_samples[i, 1:] = head_samples

        return triples, neg_tail_samples, neg_head_samples, labels

    def corrupt_triple(self, h, r, t, true_head, true_tail, num_entities, g, negative_sampling_num, tail=True):
        negative_sample_list = []
        negative_sample_size = 0

        # while negative_sample_size < self.args.negative_sampling_num:
        #     negative_sample = np.random.randint(self.ent_num, size=self.args.negative_sampling_num)
        #
        #     if tail:
        #         mask = np.in1d(
        #             negative_sample,
        #             [g.ids[i.item()] for i in true_tail[(h, r)]],
        #             assume_unique=True,
        #             invert=True
        #         )
        #     else:
        #         mask = np.in1d(
        #             negative_sample,
        #             [g.ids[i.item()] for i in true_head[(r, t)]],
        #             assume_unique=True,
        #             invert=True
        #         )
        #     negative_sample = negative_sample[mask]
        #     negative_sample_list.append(negative_sample)
        #     negative_sample_size += negative_sample.size
        # return np.concatenate(negative_sample_list)[:self.args.negative_sampling_num]
        while negative_sample_size < negative_sampling_num:
            negative_sample = np.random.randint(self.ent_num, size=negative_sampling_num)

            if tail:
                mask = np.in1d(
                    negative_sample,
                    [g.ids[i.item()] for i in true_tail[(h, r)]],
                    assume_unique=True,
                    invert=True
                )
            else:
                mask = np.in1d(
                    negative_sample,
                    [g.ids[i.item()] for i in true_head[(r, t)]],
                    assume_unique=True,
                    invert=True
                )
            negative_sample = negative_sample[mask]
            negative_sample_list.append(negative_sample)
            negative_sample_size += negative_sample.size
        return np.concatenate(negative_sample_list)[:negative_sampling_num]
