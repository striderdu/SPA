from torch.nn import Module, ModuleList
import torch
from models.operations_new import G_OPS, LAYER_FUSION_OPS, LAYER_CONNECT_OPS, T_OPS
from models.genotypes_new import NA_PRIMITIVES, LC_PRIMITIVES, LF_PRIMITIVES, SEQ_PRIMITIVES
from numpy.random import choice

class GOpBlock(Module):
    def __init__(self, input_dim, output_dim, rel_num, base_num, dropout, head_num):
        super(GOpBlock, self).__init__()
        self._ops = ModuleList()
        for primitive in NA_PRIMITIVES:
            self._op = G_OPS[primitive](input_dim, output_dim, rel_num, base_num, dropout, head_num)
            self._ops.append(self._op)

    def forward(self, graph, primitive):
        return self._ops[NA_PRIMITIVES.index(primitive)](graph)

class TOpBlock(Module):
    def __init__(self, input_dim, hidden_dim, seq_head_num):
        super(TOpBlock, self).__init__()
        self._ops = ModuleList()
        for primitive in SEQ_PRIMITIVES:
            self._op = T_OPS[primitive](input_dim, hidden_dim, seq_head_num)
            self._ops.append(self._op)

    def forward(self, current_embed, previous_embed, previous_embed_transformer, local_attn_mask, primitive):
        return self._ops[SEQ_PRIMITIVES.index(primitive)](current_embed, previous_embed, previous_embed_transformer,
                                                          local_attn_mask)

class LcOpBlock(Module):
    def __init__(self, hidden_dim):
        super(LcOpBlock, self).__init__()
        self._ops = ModuleList()
        for primitive in LC_PRIMITIVES:
            self._op = LAYER_CONNECT_OPS[primitive](hidden_dim)
            self._ops.append(self._op)

    def forward(self, fts, primitive):
        return self._ops[LC_PRIMITIVES.index(primitive)](fts)

class LfOpBlock(Module):
    def __init__(self, hidden_dim, layer_num):
        super(LfOpBlock, self).__init__()
        self._ops = ModuleList()
        for primitive in LF_PRIMITIVES:
            self._op = LAYER_FUSION_OPS[primitive](hidden_dim, layer_num)
            self._ops.append(self._op)

    def forward(self, fts, primitive):
        return self._ops[LF_PRIMITIVES.index(primitive)](fts)

class ESPASPOSSearch(Module):
    def __init__(self, args, input_dim, hidden_dim, output_dim, rel_num):
        super(ESPASPOSSearch, self).__init__()
        self.args = args
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.rel_num = rel_num
        self.layer_num = self.args.gnn_layer_num
        self.g_layers = ModuleList()
        self.lc_layers = ModuleList()
        self.lf_layers = ModuleList()
        self.t_layers = ModuleList()
        self.inv_temperature = 0.1
        self.ops = None
        for idx in range(self.layer_num):
            if idx == 0:
                self.g_layers.append(
                    GOpBlock(self.input_dim, self.hidden_dim, self.rel_num, self.args.base_num,
                             self.args.dropout,
                             self.args.head_num))
                self.t_layers.append(
                    TOpBlock(self.input_dim, self.hidden_dim, self.args.seq_head_num))
                self.lc_layers.append(LcOpBlock(self.hidden_dim))
            else:
                if idx == self.layer_num - 1:
                    self.g_layers.append(
                        GOpBlock(self.hidden_dim, self.output_dim, self.rel_num,
                                 self.args.base_num,
                                 self.args.dropout, 1))
                    self.t_layers.append(
                        TOpBlock(self.hidden_dim, self.output_dim, self.args.seq_head_num))
                    self.lf_layer = LfOpBlock(self.hidden_dim, self.layer_num)
                else:
                    self.g_layers.append(
                        GOpBlock(self.hidden_dim,
                                 self.hidden_dim, self.rel_num, self.args.base_num,
                                 self.args.dropout, self.args.head_num))
                    self.t_layers.append(
                        TOpBlock(self.hidden_dim, self.hidden_dim, self.args.seq_head_num))
                    self.lc_layers.append(LcOpBlock(self.hidden_dim))

    def generate_single_path(self, op_subsupernet=''):
        single_path_list = []
        for i in range(self.layer_num):
            single_path_list.append(choice(NA_PRIMITIVES))
            single_path_list.append(choice(SEQ_PRIMITIVES))
            if i != self.layer_num - 1:
                single_path_list.append(choice(LC_PRIMITIVES))
            else:
                if op_subsupernet != '':
                    single_path_list.append(op_subsupernet)
                else:
                    single_path_list.append(choice(LF_PRIMITIVES))
        return single_path_list

    def forward_pre(self, graph, mode=None):
        lf_list = []
        for i, layer in enumerate(self.g_layers):
            lc_list = []
            lc_list.append(graph.ndata["ft"])
            graph = layer(graph, self.ops[3 * i])
            lf_list.append(graph.ndata["ft"])
            lc_list.append(graph.ndata["ft"])
            if i != self.layer_num - 1:
                graph.ndata["ft"] = self.lc_layers[i](lc_list, self.ops[3 * i + 2])
        return lf_list

    def forward(self, graph, prev_graph_embeds_list, time_diff_tensor, prev_graph_embeds_transformer_list,
                local_attn_mask, mode=None):
        lf_list = []
        for i, layer in enumerate(self.g_layers):
            lc_list = []
            lc_list.append(graph.ndata["ft"])
            graph = layer(graph, self.ops[3 * i])
            adjusted_prev_graph_embeds = prev_graph_embeds_list[i] * torch.exp(-time_diff_tensor * self.inv_temperature)
            lf_list.append(self.t_layers[i](graph.ndata["ft"].unsqueeze(0),
                                            adjusted_prev_graph_embeds.expand(self.args.rnn_layer_num,
                                                                              *prev_graph_embeds_list[i].shape),
                                            prev_graph_embeds_transformer_list[i],
                                            local_attn_mask, self.ops[3 * i + 1]))
            lc_list.append(graph.ndata["ft"])
            if i != self.layer_num - 1:
                graph.ndata["ft"] = self.lc_layers[i](lc_list, self.ops[3 * i + 2])
            # fts.append(graph.ndata["ft"])
        final_embed = self.lf_layer(lf_list, self.ops[-1])
        return lf_list, final_embed