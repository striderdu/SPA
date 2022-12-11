from torch.nn import Module, ModuleList, Parameter, Dropout, BatchNorm1d
from torch.nn.init import xavier_uniform_, xavier_normal_, zeros_, calculate_gain
from torch.nn.functional import relu
import dgl.function as fn
import torch
from utils import ccorr, rotate
from pprint import pprint


class CompGCNLayer(Module):
    def __init__(self, input_size, output_size, rel_num, base_num, activation=None, dropout=0.0, comp_op="corr",
                 bias=True):
        super(CompGCNLayer, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.base_num = base_num
        self.rel_num = rel_num
        self.activation = torch.tanh
        self.dropout = dropout
        self.comp_op = comp_op
        # self.rel = None
        # CompGCN loop weight
        self.w_loop = Parameter(torch.Tensor(input_size, output_size))
        # CompGCN weights
        self.w_in = Parameter(torch.Tensor(input_size, output_size))
        # self.w_rel = Parameter(torch.Tensor(input_size, output_size))  # transform embedding of relations to next layer
        self.loop_rel = Parameter(torch.Tensor(1, input_size))  # self-loop embedding
        # if base_num > 0:
        #     self.rel_wt = Parameter(torch.Tensor(rel_num, base_num))
        # else:
        # self.rel_wt = None
        # CompGCN dropout
        if dropout:
            self.dropout = Dropout(dropout)
        # CompGCN bias
        if bias:
            self.w_bias = Parameter(torch.Tensor(self.output_size))
        else:
            self.register_parameter("w_bias", None)
        self.bn = BatchNorm1d(self.output_size)

        self.reset_parameters()

    def reset_parameters(self):
        gain = calculate_gain('relu')
        xavier_normal_(self.w_loop, gain=gain)
        xavier_normal_(self.w_in, gain=gain)
        # xavier_normal_(self.w_rel, gain=gain)
        xavier_normal_(self.loop_rel, gain=gain)
        # if self.base_num > 0:
        #     xavier_normal_(self.rel_wt, gain=gain)
        if self.w_bias is not None:
            zeros_(self.w_bias)

    def forward(self, graph):
        graph = graph.local_var()
        # if self.rel_wt is None:
        # else:
        #     self.rel = torch.mm(self.rel_wt, rel)  # [num_rel*2, num_base] @ [num_base, in_c]
        loop_message = torch.mm(self.rel_transform(graph.ndata['ft'], self.loop_rel), self.w_loop)  # loop_message
        graph.update_all(self.get_message, fn.sum(msg='msg', out='ft'), self.apply_func)
        node_repr = graph.ndata['ft']
        # node_repr = node_repr + loop_message
        # if self.dropout:
        #     node_repr = self.dropout(node_repr)
        if self.w_bias is not None:
            node_repr = node_repr + self.w_bias
        node_repr = node_repr + loop_message
        node_repr = self.bn(node_repr)
        if self.activation:
            node_repr = self.activation(node_repr)
        if self.dropout:
            node_repr = self.dropout(node_repr)
        graph.ndata['ft'] = node_repr
        return graph

    def forward_isolated(self, ent_embed, rel=None):
        # loop_message = torch.mm(ent_embed, self.w_loop)
        # ent_embed = ent_embed + loop_message
        # if self.bias:
        #     ent_embed = ent_embed + self.bias
        # ent_embed = self.bn(ent_embed)
        # if self.activation:
        #     ent_embed = self.activation(ent_embed)
        # if self.dropout:
        #     ent_embed = self.dropout(ent_embed)
        # return ent_embed
        loop_message = torch.mm(ent_embed, self.w_loop)
        if self.dropout:
            loop_message = self.dropout(loop_message)
        if self.w_bias is not None:
            ent_embed = ent_embed + self.w_bias
        ent_embed = ent_embed + loop_message
        # ent_embed = self.bn(ent_embed)
        if self.activation:
            ent_embed = self.activation(ent_embed)
        # if self.dropout:
        #     loop_message = self.dropout(loop_message)
        # if self.bias is not None:
        #     ent_embed = ent_embed + self.bias
        # ent_embed = ent_embed + loop_message
        # ent_embed = self.bn(ent_embed)
        # if self.activation:
        #     ent_embed = self.activation(ent_embed)
        return ent_embed

    def get_message(self, edges):
        if self.comp_op is not None:
            w_rel = self.w_in
            node = edges.src['ft']
            edge = edges.data['ft']
            # edge = self.rel[edges.data['id']]
            ft = self.rel_transform(node, edge, op=self.comp_op)
            msg = torch.mm(ft, w_rel).view(-1, self.output_size)

        # else:
        #     if self.base_num > 0:
        #         w_rel = torch.einsum("rb, bio -> rio", (self.w_rel, self.w_bases)).index_select(0, edges.data['id'])
        #     else:
        #         w_rel = self.w_rel.index_select(0, edges.data['id'])
        #     ft = edges.src['ft'].view(-1, 1, self.input_size)
        #     msg = torch.bmm(ft, w_rel).view(-1, self.output_size)
        return {'msg': msg}

    def apply_func(self, nodes):
        return {'ft': nodes.data['ft'] * nodes.data['norm']}

    def rel_transform(self, ent_embed, rel_embed, op='corr'):
        if op == 'corr':
            trans_embed = ccorr(ent_embed, rel_embed)
        elif op == 'rotate':
            trans_embed = rotate(ent_embed, rel_embed)
        elif op == 'add':
            trans_embed = ent_embed + rel_embed
        elif op == 'sub':
            trans_embed = ent_embed - rel_embed
        elif op == 'mult':
            trans_embed = ent_embed * rel_embed
        else:
            raise NotImplementedError

        return trans_embed


class CompGCN(Module):
    def __init__(self, args, input_size, hidden_size, output_size, rel_num):
        super(CompGCN, self).__init__()
        self.args = args
        self.layers = ModuleList()
        self.layer_num = self.args.gnn_layer_num
        self.activation = torch.tanh
        self.rel_num = rel_num
        for idx in range(self.layer_num):
            if idx == 0:
                self.layers.append(CompGCNLayer(
                    input_size, hidden_size, self.args.base_num, self.rel_num, self.activation, self.args.dropout,
                    self.args.comp_op
                ))
            else:
                if idx == self.layer_num - 1:
                    self.layers.append(CompGCNLayer(
                        hidden_size, output_size, self.args.base_num, self.rel_num, self.activation, self.args.dropout,
                        self.args.comp_op
                    ))
                else:
                    self.layers.append(CompGCNLayer(
                        hidden_size, hidden_size, self.args.base_num, self.rel_num, self.activation, self.args.dropout,
                        self.args.comp_op
                    ))

    def forward(self, graph):
        for i, layer in enumerate(self.layers):
            graph = layer(graph)
        return graph

    def forward_isolated(self, ent_embed):
        for i, layer in enumerate(self.layers):
            ent_embed = layer.forward_isolated(ent_embed)
        return ent_embed
