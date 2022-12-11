from torch.nn import Module, Linear, ModuleList, Parameter, Dropout, BatchNorm1d
from torch.nn.init import xavier_uniform_, xavier_normal_, zeros_, calculate_gain
from torch.nn.functional import relu, elu
import dgl.function as fn
import torch
from utils import ccorr
from pprint import pprint
from models.rgcn import RGCNLayer
from models.rgat import RGATLayer, MultiHeadRGATLayer
from models.compgcn import CompGCNLayer
from models.jknet import JKNet
from models.self_attention import SelfAttention

NA_OPS = {
    "rgcn": lambda input_dim, output_dim, rel_num, base_num, dropout, head_num: NaAggregator(input_dim, output_dim,
                                                                                             rel_num,
                                                                                             base_num, dropout,
                                                                                             head_num,
                                                                                             'rgcn'),
    "rgat_vanilla": lambda input_dim, output_dim, rel_num, base_num, dropout, head_num: NaAggregator(input_dim,
                                                                                                     output_dim,
                                                                                                     rel_num,
                                                                                                     base_num, dropout,
                                                                                                     head_num,
                                                                                                     'rgat_vanilla'),
    "rgat_sym": lambda input_dim, output_dim, rel_num, base_num, dropout, head_num: NaAggregator(input_dim, output_dim,
                                                                                                 rel_num,
                                                                                                 base_num, dropout,
                                                                                                 head_num,
                                                                                                 'rgat_sym'),
    "rgat_cos": lambda input_dim, output_dim, rel_num, base_num, dropout, head_num: NaAggregator(input_dim, output_dim,
                                                                                                 rel_num,
                                                                                                 base_num, dropout,
                                                                                                 head_num,
                                                                                                 'rgat_cos'),
    "rgat_linear": lambda input_dim, output_dim, rel_num, base_num, dropout, head_num: NaAggregator(input_dim,
                                                                                                    output_dim,
                                                                                                    rel_num,
                                                                                                    base_num, dropout,
                                                                                                    head_num,
                                                                                                    'rgat_linear'),
    "rgat_gene-linear": lambda input_dim, output_dim, rel_num, base_num, dropout, head_num: NaAggregator(input_dim,
                                                                                                         output_dim,
                                                                                                         rel_num,
                                                                                                         base_num,
                                                                                                         dropout,
                                                                                                         head_num,
                                                                                                         'rgat_gene'
                                                                                                         '-linear'),
    "compgcn_corr": lambda input_dim, output_dim, rel_num, base_num, dropout, head_num: NaAggregator(input_dim,
                                                                                                     output_dim,
                                                                                                     rel_num,
                                                                                                     base_num, dropout,
                                                                                                     head_num,
                                                                                                     'compgcn_corr'),
    "compgcn_rotate": lambda input_dim, output_dim, rel_num, base_num, dropout, head_num: NaAggregator(input_dim,
                                                                                                     output_dim,
                                                                                                     rel_num,
                                                                                                     base_num, dropout,
                                                                                                     head_num,
                                                                                                     'compgcn_rotate'),
}

SC_OPS = {
    "none": lambda: Zero(),
    "skip": lambda: Identity()
}

LA_OPS = {
    'l_max': lambda hidden_dim, layer_num: LaAggregator('max', hidden_dim, layer_num),
    'l_concat': lambda hidden_dim, layer_num: LaAggregator('cat', hidden_dim, layer_num),
    'l_lstm': lambda hidden_dim, layer_num: LaAggregator('lstm', hidden_dim, layer_num),
    'l_mean': lambda hidden_dim, layer_num: LaAggregator('mean', hidden_dim, layer_num),
    'l_sum': lambda hidden_dim, layer_num: LaAggregator('sum', hidden_dim, layer_num),
}

SEQ_OPS = {
    '1': lambda input_dim, seq_head_num: SeqEncoder('1', input_dim, seq_head_num),
    '0': lambda input_dim, seq_head_num: SeqEncoder('0', input_dim, seq_head_num)
}


class NaAggregator(Module):
    def __init__(self, input_dim, output_dim, rel_num, base_num, dropout, head_num, aggregator):
        super(NaAggregator, self).__init__()
        if aggregator == "rgcn":
            self._op = RGCNLayer(input_dim, output_dim, rel_num, base_num, dropout=dropout)
        elif aggregator == "rgat_vanilla":
            self._op = MultiHeadRGATLayer(input_dim, output_dim // head_num, rel_num, base_num, dropout=dropout,
                                          head_num=head_num,
                                          att_type="vanilla")
        elif aggregator == "rgat_sym":
            self._op = MultiHeadRGATLayer(input_dim, output_dim // head_num, rel_num, base_num, dropout=dropout,
                                          head_num=head_num,
                                          att_type="sym")
        elif aggregator == "rgat_cos":
            self._op = MultiHeadRGATLayer(input_dim, output_dim // head_num, rel_num, base_num, dropout=dropout,
                                          head_num=head_num,
                                          att_type="cos")
        elif aggregator == "rgat_linear":
            self._op = MultiHeadRGATLayer(input_dim, output_dim // head_num, rel_num, base_num, dropout=dropout,
                                          head_num=head_num,
                                          att_type="linear")
        elif aggregator == "rgat_gene-linear":
            self._op = MultiHeadRGATLayer(input_dim, output_dim // head_num, rel_num, base_num, dropout=dropout,
                                          head_num=head_num,
                                          att_type="gene-linear")
        elif aggregator == "compgcn_add":
            self._op = CompGCNLayer(input_dim, output_dim, rel_num, base_num, dropout=dropout, comp_op='add')
        elif aggregator == "compgcn_sub":
            self._op = CompGCNLayer(input_dim, output_dim, rel_num, base_num, dropout=dropout, comp_op='sub')
        elif aggregator == "compgcn_mult":
            self._op = CompGCNLayer(input_dim, output_dim, rel_num, base_num, dropout=dropout, comp_op='mult')
        elif aggregator == "compgcn_corr":
            self._op = CompGCNLayer(input_dim, output_dim, rel_num, base_num, dropout=dropout, comp_op='corr')
        elif aggregator == "compgcn_rotate":
            self._op = CompGCNLayer(input_dim, output_dim, rel_num, base_num, dropout=dropout, comp_op='rotate')

    def forward(self, graph):
        return self._op(graph)

    def forward_isolated(self, graph):
        return self._op.forward_isolated(graph)


class LaAggregator(Module):
    def __init__(self, mode, hidden_dim, layer_num):
        super(LaAggregator, self).__init__()
        self.mode = mode
        if self.mode in ['lstm', 'max', 'cat']:
            self.jump = JKNet(mode, channels=hidden_dim, layer_num=layer_num)
        if self.mode == 'cat':
            self.lin = Linear(hidden_dim * layer_num, hidden_dim)
        else:
            self.lin = Linear(hidden_dim, hidden_dim)

    def forward(self, fts):
        if self.mode in ['lstm', 'max', 'cat']:
            return self.lin((self.jump(fts)))
        elif self.mode == 'sum':
            return torch.stack(fts, dim=-1).sum(dim=-1)
        elif self.mode == 'mean':
            return torch.stack(fts, dim=-1).mean(dim=-1)
        # return self.lin(elu(self.jump(fts)))


class SeqEncoder(Module):
    def __init__(self, mode, input_dim, seq_head_num, layer_num=1):
        super(SeqEncoder, self).__init__()
        if mode == '1':
            self.seq = SelfAttention(input_dim, seq_head_num)
        else:
            self.seq = lambda x, y, z: x

    def forward(self, current_embed, previous_embed, local_attn_mask):
        return self.seq(current_embed, previous_embed, local_attn_mask)


class Identity(Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, embed):
        return embed


class Zero(Module):
    def __init__(self):
        super(Zero, self).__init__()

    def forward(self, embed):
        embed = embed.mul(0.)
        return embed
