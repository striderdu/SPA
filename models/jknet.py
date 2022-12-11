from torch.nn import Module, Linear, LSTM
import torch


class JKNet(Module):
    def __init__(self, mode, channels=None, layer_num=1):
        super(JKNet, self).__init__()
        self.mode = mode
        if mode == 'lstm':
            assert channels is not None, 'channels cannot be None for lstm'
            assert layer_num is not None, 'num_layers cannot be None for lstm'
            self.lstm = LSTM(
                channels, (layer_num * channels) // 2,
                bidirectional=True,
                batch_first=True)
            self.att = Linear(2 * ((layer_num * channels) // 2), 1)
        self.reset_parameters()

    def reset_parameters(self):
        if hasattr(self, 'lstm'):
            self.lstm.reset_parameters()
        if hasattr(self, 'att'):
            self.att.reset_parameters()

    def forward(self, fts):
        if self.mode == "cat":
            return torch.cat(fts, dim=-1)
        elif self.mode == "max":
            return torch.stack(fts, dim=-1).max(dim=-1)[0]
        elif self.mode == "lstm":
            x = torch.stack(fts, dim=1)  # [num_nodes, num_layers, num_channels]
            alpha, _ = self.lstm(x)
            alpha = self.att(alpha).squeeze(-1)  # [num_nodes, num_layers]
            alpha = torch.softmax(alpha, dim=-1)
            return (x * alpha.unsqueeze(-1)).sum(dim=1)
