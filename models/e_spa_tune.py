from torch.nn import Module, ModuleList
import torch
from models.operations_new import G_OPS, LAYER_FUSION_OPS, LAYER_CONNECT_OPS, T_OPS

class NaOp(Module):
    def __init__(self, primitive, input_dim, output_dim, rel_num, base_num, dropout, head_num):
        super(NaOp, self).__init__()
        self._op = G_OPS[primitive](input_dim, output_dim, rel_num, base_num, dropout, head_num)

    def forward(self, graph):
        return self._op(graph)

    def forward_isolated(self, graph):
        return self._op.forward_isolated(graph)


class LcOp(Module):
    def __init__(self, primitive, hidden_dim):
        super(LcOp, self).__init__()
        self._op = LAYER_CONNECT_OPS[primitive](hidden_dim)

    def forward(self, fts):
        return self._op(fts)


class LfOp(Module):
    def __init__(self, primitive, hidden_dim, layer_num):
        super(LfOp, self).__init__()
        self._op = LAYER_FUSION_OPS[primitive](hidden_dim, layer_num)

    def forward(self, fts):
        return self._op(fts)

class TOp(Module):
    def __init__(self, primitive, input_dim, hidden_dim, seq_head_num):
        super(TOp, self).__init__()
        self._op = T_OPS[primitive](input_dim, hidden_dim, seq_head_num)

    def forward(self, current_embed, previous_embed, previous_embed_transformer, local_attn_mask):
        return self._op(current_embed, previous_embed, previous_embed_transformer, local_attn_mask)


class ESPATune(Module):
    def __init__(self, genotype, args, input_dim, hidden_dim, output_dim, rel_num):
        super(ESPATune, self).__init__()
        self.genotype = genotype
        self.args = args
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.rel_num = rel_num
        self.layer_num = self.args.gnn_layer_num
        self.na_layers = ModuleList()
        self.lc_layers = ModuleList()
        self.lf_layers = ModuleList()
        # self.sc_layers = ModuleList()
        self.seq_layers = ModuleList()
        self.inv_temperature = 0.1
        ops = genotype.split('||')
        for idx in range(self.layer_num):
            if idx == 0:
                self.na_layers.append(
                    NaOp(ops[3 * idx], self.input_dim, self.hidden_dim, self.rel_num, self.args.base_num,
                         self.args.dropout,
                         self.args.head_num))
                self.seq_layers.append(
                    TOp(ops[3 * idx + 1], self.input_dim, self.hidden_dim, self.args.seq_head_num))
                self.lc_layers.append(LcOp(ops[3 * idx + 2], self.hidden_dim))
            else:
                if idx == self.layer_num - 1:
                    self.na_layers.append(
                        NaOp(ops[3 * idx], self.hidden_dim, self.output_dim, self.rel_num,
                             self.args.base_num,
                             self.args.dropout, 1))
                    self.seq_layers.append(
                        TOp(ops[idx * 3 + 1], self.hidden_dim, self.output_dim, self.args.seq_head_num))
                    self.lf_layer = LfOp(ops[-1], self.hidden_dim, self.layer_num)
                else:
                    self.na_layers.append(
                        NaOp(ops[3 * idx], self.hidden_dim,
                             self.hidden_dim, self.rel_num, self.args.base_num,
                             self.args.dropout, self.args.head_num))
                    self.seq_layers.append(
                        TOp(ops[3 * idx + 1], self.hidden_dim, self.hidden_dim, self.args.seq_head_num))
                    self.lc_layers.append(LcOp(ops[3 * idx + 2], self.hidden_dim))

    def forward_pre(self, graph, mode=None):
        lf_list = []
        for i, layer in enumerate(self.na_layers):
            lc_list = []
            lc_list.append(graph.ndata["ft"])
            graph = layer(graph)
            lf_list.append(graph.ndata["ft"])
            lc_list.append(graph.ndata["ft"])
            if i != self.layer_num - 1:
                graph.ndata["ft"] = self.lc_layers[i](lc_list)
        return lf_list

    def forward(self, graph, prev_graph_embeds_list, time_diff_tensor, prev_graph_embeds_transformer_list,
                local_attn_mask, mode=None):
        lf_list = []
        for i, layer in enumerate(self.na_layers):
            lc_list = []
            lc_list.append(graph.ndata["ft"])
            graph = layer(graph)
            adjusted_prev_graph_embeds = prev_graph_embeds_list[i] * torch.exp(-time_diff_tensor * self.inv_temperature)
            lf_list.append(self.seq_layers[i](graph.ndata["ft"].unsqueeze(0),
                                              adjusted_prev_graph_embeds.expand(self.args.rnn_layer_num,
                                                                                *prev_graph_embeds_list[i].shape),
                                              prev_graph_embeds_transformer_list[i],
                                              local_attn_mask))
            lc_list.append(graph.ndata["ft"])
            if i != self.layer_num - 1:
                graph.ndata["ft"] = self.lc_layers[i](lc_list)
            # fts.append(graph.ndata["ft"])
        final_embed = self.lf_layer(lf_list)
        return lf_list, final_embed

    def forward_isolated(self, graph, prev_graph_embeds_list, time_diff_tensor):
        lf_list = []
        for i, layer in enumerate(self.na_layers):
            lc_list = []
            lc_list.append(graph.ndata["ft"])
            graph = layer.forward_isolated(graph)
            adjusted_prev_graph_embeds = prev_graph_embeds_list[i] * torch.exp(-time_diff_tensor * self.inv_temperature)
            lf_list.append(self.seq_layers[i](graph.ndata["ft"].unsqueeze(0),
                                              adjusted_prev_graph_embeds.expand(self.args.rnn_layer_num,
                                                                                *prev_graph_embeds_list[i].shape)))
            lc_list.append(graph.ndata["ft"])
            if i != self.layer_num - 1:
                graph.ndata["ft"] = self.lc_layers[i](lc_list)
            # fts.append(graph.ndata["ft"])
        final_embed = self.lf_layer(lf_list)
        return final_embed
