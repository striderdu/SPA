from torch.nn import Module, ModuleList, Parameter, Dropout, BatchNorm1d
from torch.nn.init import xavier_uniform_, xavier_normal_, calculate_gain, zeros_
from torch.nn.functional import relu
import dgl.function as fn
import torch
from pprint import pprint


class RGCNLayer(Module):
    def __init__(self, input_size, output_size, rel_num, base_num, activation=relu, dropout=0.0, bias=True):
        super(RGCNLayer, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.base_num = base_num
        self.rel_num = rel_num
        self.activation = activation
        self.dropout = dropout
        # R-GCN loop weight
        self.w_loop = Parameter(torch.Tensor(input_size, output_size))
        # R-GCN weights
        if base_num > 0:
            self.w_bases = Parameter(
                torch.Tensor(self.base_num, self.input_size, self.output_size)
            )
            self.w_rel = Parameter(torch.Tensor(self.rel_num, self.base_num))
        else:
            self.w_rel = Parameter(
                torch.Tensor(self.rel_num, self.input_size, self.output_size)
            )
        # R-GCN dropout
        if dropout:
            self.dropout = Dropout(dropout)
        # R-GCN bias
        if bias:
            self.w_bias = Parameter(torch.Tensor(self.output_size))
        else:
            self.register_parameter("w_bias", None)
        self.bn = BatchNorm1d(self.output_size)

        self.reset_parameters()

    def reset_parameters(self):
        gain = calculate_gain('relu')
        xavier_uniform_(self.w_loop, gain=gain)
        if self.base_num > 0:
            xavier_uniform_(self.w_bases, gain=gain)
            xavier_uniform_(self.w_rel, gain=gain)
        else:
            xavier_uniform_(self.w_rel, gain=gain)
        if self.w_bias is not None:
            zeros_(self.w_bias)

    def forward(self, graph):
        graph = graph.local_var()
        loop_message = torch.mm(graph.ndata['ft'], self.w_loop)
        graph.update_all(self.get_message, fn.sum(msg='msg', out='ft'), self.apply_func)
        node_repr = graph.ndata['ft']
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

    def forward_isolated(self, ent_embed):
        loop_message = torch.mm(ent_embed, self.w_loop)
        if self.dropout:
            loop_message = self.dropout(loop_message)
        ent_embed = ent_embed + loop_message
        if self.w_bias is not None:
            ent_embed = ent_embed + self.w_bias
        if self.activation:
            ent_embed = self.activation(ent_embed)
        return ent_embed

    def get_message(self, edges):
        if self.base_num > 0:
            w_rel = torch.einsum("rb, bio -> rio", (self.w_rel, self.w_bases)).index_select(0, edges.data['id'])
        else:
            w_rel = self.w_rel.index_select(0, edges.data['id'])
        ft = edges.src['ft'].view(-1, 1, self.input_size)
        msg = torch.bmm(ft, w_rel).view(-1, self.output_size)
        return {'msg': msg}

    def apply_func(self, nodes):
        return {'ft': nodes.data['ft'] * nodes.data['norm']}


class RGCN(Module):
    def __init__(self, args, input_size, hidden_size, output_size, rel_num):
        super(RGCN, self).__init__()
        self.args = args
        self.layers = ModuleList()
        self.layer_num = self.args.gnn_layer_num
        self.activation = relu
        self.rel_num = rel_num
        for idx in range(self.layer_num):
            if idx == 0:
                self.layers.append(RGCNLayer(
                    input_size, hidden_size, self.rel_num, self.args.base_num, None, self.args.dropout
                ))
            else:
                if idx == self.layer_num - 1:
                    self.layers.append(RGCNLayer(
                        hidden_size, output_size, self.rel_num, self.args.base_num, self.activation, self.args.dropout
                    ))
                else:
                    self.layers.append(RGCNLayer(
                        hidden_size, hidden_size, self.rel_num, self.args.base_num, self.activation, self.args.dropout
                    ))

    def forward(self, graph):
        for i, layer in enumerate(self.layers):
            graph = layer(graph)
        return graph

    def forward_isolated(self, ent_embed):
        for i, layer in enumerate(self.layers):
            ent_embed = layer.forward_isolated(ent_embed)
        return ent_embed
