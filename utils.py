import json
import logging
import logging.config
import random

import numpy
import torch


def get_metrics(ranks):
    mrr = torch.mean(1.0 / ranks.float())
    hit_1 = torch.mean((ranks <= 1).float())
    hit_3 = torch.mean((ranks <= 3).float())
    hit_10 = torch.mean((ranks <= 10).float())
    return mrr, hit_1, hit_3, hit_10


def initialize_seed(seed: int):
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False


def get_logger(name, log_dir):
    config_dict = json.load(open('./config/' + 'log_config.json'))
    config_dict['handlers']['file_handler']['filename'] = log_dir + name + '.log'
    logging.config.dictConfig(config_dict)
    logger = logging.getLogger(name)
    return logger


def get_name(args):
    if "GAT" in args.encoder:
        encoder_config = args.head_num
    elif "Comp" in args.encoder:
        encoder_config = args.comp_op
    else:
        if args.train_mode == "train":
            encoder_config = args.train_seq_len
    sampled = "sampled" if args.sampled_dataset else "all"
    if args.train_mode == "train":
        return f'{args.encoder}_{encoder_config}_{args.random_seed}_{sampled}'
    elif args.train_mode == "debug":
        return f'{args.encoder}_{args.genotype}_{args.random_seed}_{args.train_seq_len}_{args.score_function}_{args.inv_temperature}'
    return f'{args.index}'


def compute_norm(g):
    g = g.local_var()
    in_deg = g.in_degrees().float()
    norm = torch.pow(in_deg, -0.5)
    norm[torch.isinf(norm)] = 0
    return norm


def node_norm_to_edge_norm(g, norm):
    g = g.local_var()
    # convert to edge norm
    g.ndata['norm'] = norm
    g.apply_edges(lambda edges: {'norm': edges.dst['norm']})
    return g.edata['norm']


def com_mult(a, b):
    r1, i1 = a[..., 0], a[..., 1]
    r2, i2 = b[..., 0], b[..., 1]
    return torch.stack([r1 * r2 - i1 * i2, r1 * i2 + i1 * r2], dim=-1)


def conj(a):
    a[..., 1] = -a[..., 1]
    return a


def cconv(a, b):
    return torch.fft.irfft(com_mult(torch.fft.rfft(a, dim=1), torch.fft.rfft(b, dim=1)), 1, signal_sizes=(a.shape[-1],))


def ccorr(a, b):
    aa = torch.view_as_real(torch.fft.rfft(a, dim=1))
    bb = torch.view_as_real(torch.fft.rfft(b, dim=1))
    cc = torch.view_as_complex(com_mult(conj(aa), bb))
    return torch.fft.irfft(cc, dim=1)


def rotate(h, r):
    # re: first half, im: second half
    # assume embedding dim is the last dimension
    d = h.shape[-1]
    h_re, h_im = torch.split(h, d // 2, -1)
    r_re, r_im = torch.split(r, d // 2, -1)
    return torch.cat([h_re * r_re - h_im * r_im,
                        h_re * r_im + h_im * r_re], dim=-1)

def filter_none(l):
    return list(filter(lambda x: x is not None, l))


def count_parameters_in_MB(model):
    return numpy.sum(numpy.prod(v.size()) for name, v in model.named_parameters() if "auxiliary" not in name) / 1e6
