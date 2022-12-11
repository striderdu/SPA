from torch.nn import Module, ModuleList, Parameter, Dropout, BatchNorm1d, Linear
from torch.nn.init import xavier_uniform_, xavier_normal_, calculate_gain, zeros_
from torch.nn.functional import relu, softmax, leaky_relu
import dgl.function as fn
import torch
from pprint import pprint


class RGATLayer(Module):
    def __init__(self, input_dim, output_dim, rel_num, base_num, activation=None, dropout=0.0, bias=True,
                 att_type="vanilla"):
        super(RGATLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.base_num = base_num
        self.rel_num = rel_num
        self.activation = relu
        self.dropout = dropout
        self.att_type = att_type
        self.attention_dim = 1
        # R-GAT loop weight
        self.w_loop = Parameter(torch.Tensor(self.input_dim, self.output_dim))
        # R-GAT weights
        if base_num > 0:
            self.w_bases = Parameter(
                torch.Tensor(self.base_num, self.input_dim, self.output_dim)
            )
            self.w_rel = Parameter(torch.Tensor(self.rel_num, self.base_num))
        else:
            self.w_rel = Parameter(
                torch.Tensor(self.rel_num, self.input_dim, self.output_dim)
            )
        # R-GAT Attention
        if self.att_type == "gene-linear":
            self.attention_dim = self.output_dim
            self.general_att_layer = Linear(self.output_dim, 1, bias=False)
        if self.att_type in ["vanilla", "sym", "linear", "cos", "gene-linear"]:
            self.attention_l = Linear(self.output_dim, self.attention_dim, bias=False)
        if self.att_type in ["sym", "linear"]:
            self.attention_r = Linear(self.output_dim, self.attention_dim, bias=False)
        # R-GAT dropout
        if dropout:
            self.dropout = Dropout(dropout)
        # R-GAT bias
        if bias:
            self.w_bias = Parameter(torch.Tensor(self.output_dim))
        else:
            self.register_parameter("w_bias", None)
        self.bn = BatchNorm1d(self.output_dim)

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
        xavier_uniform_(self.attention_l.weight.data, gain=gain)
        if self.att_type in ["sym", "linear"]:
            xavier_uniform_(self.attention_r.weight.data, gain=gain)

    def forward(self, graph):
        graph = graph.local_var()
        loop_message = torch.mm(graph.ndata['ft'], self.w_loop)
        graph.apply_edges(self.attention_prepare)
        graph.update_all(self.get_message, self.attention_reduce, self.apply_func)
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
        ent_embed = ent_embed + loop_message
        if self.bias:
            ent_embed = ent_embed + self.bias
        if self.activation:
            ent_embed = self.activation(ent_embed)
        if self.dropout:
            node_repr = self.dropout(ent_embed)
        return ent_embed

    def get_message(self, edges):
        # pprint(edges.data)
        # if 'norm' in edges.src:
        #     msg_rgcn = edges.data["src_node_r"] * edges.src['norm']
        #     return
        # else:
        #     ft = edges.src['ft'].view(-1, 1, self.input_dim)
        # msg = torch.bmm(ft, w_rel).view(-1, self.output_dim)
        # return {'msg': msg}
        if self.att_type in ["sym", "linear"]:
            return {"attention_l_src": edges.data["attention_l_src"], "attention_l_dst": edges.data["attention_l_dst"],
                    "attention_r_src": edges.data["attention_r_src"], "attention_r_dst": edges.data["attention_r_dst"],
                    "src_node_r": edges.data["src_node_r"], "dst_node_r": edges.data["dst_node_r"]}
        else:
            return {"attention_l_src": edges.data["attention_l_src"], "attention_l_dst": edges.data["attention_l_dst"],
                    "src_node_r": edges.data["src_node_r"], "dst_node_r": edges.data["dst_node_r"]}

    def apply_func(self, nodes):
        # pprint(nodes.data.keys())
        return {'ft': nodes.data['accum'] * nodes.data['norm']}
            # return {'ft': self.attention_l(nodes.data['ft'])}
            # pprint('Over')
            # exit(0)

    def attention_prepare(self, edges):
        if self.base_num > 0:
            w_rel = torch.einsum("rb, bio -> rio", (self.w_rel, self.w_bases)).index_select(0, edges.data['id'])
        else:
            w_rel = self.w_rel.index_select(0, edges.data['id'])
        src_node = edges.src['ft'].view(-1, 1, self.input_dim)
        dst_node = edges.dst['ft'].view(-1, 1, self.input_dim)
        src_node_r = torch.bmm(src_node, w_rel).view(-1, self.output_dim)
        dst_node_r = torch.bmm(dst_node, w_rel).view(-1, self.output_dim)
        attention_l_src = self.attention_l(src_node_r)
        attention_l_dst = self.attention_l(dst_node_r)
        if self.att_type in ["sym", "linear"]:
            attention_r_src = self.attention_r(src_node_r)
            attention_r_dst = self.attention_r(dst_node_r)
            return {"attention_l_src": attention_l_src, "attention_l_dst": attention_l_dst,
                    "attention_r_src": attention_r_src, "attention_r_dst": attention_r_dst,
                    "src_node_r": src_node_r, "dst_node_r": dst_node_r}
        else:
            return {"attention_l_src": attention_l_src, "attention_l_dst": attention_l_dst,
                    "src_node_r": src_node_r, "dst_node_r": dst_node_r}

    def attention_reduce(self, nodes):
        if self.att_type == "vanilla":
            attention_l_src = nodes.mailbox["attention_l_src"]
            attention_l_dst = nodes.mailbox["attention_l_dst"]
            attention_l = attention_l_src + attention_l_dst
            # attention_l = attention_l.sum(-1, keepdim=True)  # Just in case the dimension is not zero
            e = softmax(leaky_relu(attention_l), dim=1)
        elif self.att_type == "sym":
            attention_l_src = nodes.mailbox["attention_l_src"]
            attention_l_dst = nodes.mailbox["attention_l_dst"]
            attention_l = attention_l_src + attention_l_dst
            attention_r_src = nodes.mailbox["attention_r_src"]
            attention_r_dst = nodes.mailbox["attention_r_dst"]
            attention_r = attention_r_src + attention_r_dst
            # attention_l = attention_l.sum(-1, keepdim=True)  # Just in case the dimension is not zero
            e = softmax(leaky_relu(attention_l) + leaky_relu(attention_r), dim=1)
        elif self.att_type == "linear":
            attention_l_src = nodes.mailbox["attention_l_src"]
            attention_r_src = nodes.mailbox["attention_r_src"]
            e = softmax(torch.tanh(attention_l_src + attention_r_src), dim=1)
        elif self.att_type == "cos":
            attention_l_src = nodes.mailbox["attention_l_src"]
            attention_l_dst = nodes.mailbox["attention_l_dst"]
            attention_l = attention_l_src * attention_l_dst
            e = softmax(leaky_relu(attention_l), dim=1)
        elif self.att_type == "gene-linear":
            attention_l_src = nodes.mailbox["attention_l_src"]
            attention_l_dst = nodes.mailbox["attention_l_dst"]
            attention_l = attention_l_src + attention_l_dst
            e = softmax(self.general_att_layer(torch.tanh(attention_l)), dim=1)
        else:
            raise Exception("Wrong attention type:", self.att_type)
        return {"accum": torch.sum(e * nodes.mailbox["src_node_r"], dim=1)}


class MultiHeadRGATLayer(Module):
    def __init__(self, input_dim, output_dim, rel_num, base_num, dropout,
                 bias=False, head_num=1, att_type="vanilla", merge='cat'):
        super(MultiHeadRGATLayer, self).__init__()
        self.head_list = ModuleList()
        for i in range(head_num):
            self.head_list.append(
                RGATLayer(input_dim, output_dim, rel_num, base_num, activation=None, dropout=dropout,
                          att_type=att_type))
        self.merge = merge

    def forward(self, graph):
        graph = graph.local_var()
        batch_graph = [head(graph) for head in self.head_list]
        multi_head = [tmp_graph.ndata['ft'] for tmp_graph in batch_graph]
        if self.merge == 'cat':
            # concat on the output feature dimension (dim=1)
            graph.ndata['ft'] = torch.cat(multi_head, dim=1)
            return graph
        else:
            # merge using average
            graph.ndata['ft'] = torch.mean(torch.stack(multi_head))
            return graph


class RGAT(Module):
    def __init__(self, args, input_dim, hidden_dim, output_dim, rel_num):
        super(RGAT, self).__init__()
        self.args = args
        self.layers = ModuleList()
        self.layer_num = self.args.gnn_layer_num
        self.activation = relu
        self.rel_num = rel_num
        self.head_num = self.args.head_num
        for idx in range(self.layer_num):
            if idx == 0:
                self.layers.append(MultiHeadRGATLayer(
                    input_dim, hidden_dim, self.rel_num, self.args.base_num, dropout=self.args.dropout,
                    head_num=self.head_num
                ))
            else:
                if idx == self.layer_num - 1:
                    self.layers.append(MultiHeadRGATLayer(
                        hidden_dim * (self.head_num ** idx), output_dim, self.rel_num, self.args.base_num,
                        dropout=self.args.dropout, head_num=1
                    ))
                else:
                    self.layers.append(MultiHeadRGATLayer(
                        hidden_dim * (self.head_num ** idx), hidden_dim * (self.head_num ** idx),
                        self.rel_num, self.args.base_num,
                        dropout=self.args.dropout, head_num=self.head_num
                    ))
        # if self.head_num == 1:
        #     for idx in range(self.layer_num):
        #         if idx == 0:
        #             self.layers.append(RGATLayer(
        #                 input_dim, hidden_dim, self.args.base_num, self.rel_num, self.activation, self.args.dropout
        #             ))
        #         else:
        #             if idx == self.layer_num - 1:
        #                 self.layers.append(RGATLayer(
        #                     hidden_dim, output_dim, self.args.base_num, self.rel_num, self.activation,
        #                     self.args.dropout
        #                 ))
        #             else:
        #                 self.layers.append(RGATLayer(
        #                     hidden_dim, hidden_dim, self.args.base_num, self.rel_num, self.activation,
        #                     self.args.dropout
        #                 ))
        # else:
        #     for idx in range(self.layer_num):
        #         if idx == 0:
        #             self.layers.append(MultiHeadRGATLayer(
        #                 input_dim, hidden_dim, self.args.base_num, self.rel_num, self.activation, self.args.dropout,
        #                 head_num=self.head_num
        #             ))
        #         else:
        #             if idx == self.layer_num - 1:
        #                 self.layers.append(MultiHeadRGATLayer(
        #                     hidden_dim * (self.head_num ** idx), output_dim, self.args.base_num, self.rel_num,
        #                     self.activation,
        #                     self.args.dropout, head_num=1
        #                 ))
        #             else:
        #                 self.layers.append(MultiHeadRGATLayer(
        #                     hidden_dim * (self.head_num ** idx), hidden_dim * (self.head_num ** idx),
        #                     self.args.base_num, self.rel_num, self.activation,
        #                     self.args.dropout, head_num=self.head_num
        #                 ))

    def forward(self, graph):
        for i, layer in enumerate(self.layers):
            graph = layer(graph)
        return graph

    def forward_isolated(self, ent_embed):
        for i, layer in enumerate(self.layers):
            ent_embed = layer.forward_isolated(ent_embed)
        return ent_embed
