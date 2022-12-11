from torch.nn import Module, Linear
from torch.nn.functional import softmax
import torch
import math


class SelfAttention(Module):
    def __init__(self, input_dim, head, bias=True):
        super(SelfAttention, self).__init__()
        self.input_dim = input_dim
        self.head = head
        self.d_k = self.input_dim // self.head
        self.q_linear = Linear(input_dim, input_dim, bias=bias)
        self.k_linear = Linear(input_dim, input_dim, bias=bias)
        self.v_linear = Linear(input_dim, input_dim, bias=bias)

    def forward(self, current_embed, previous_embed, local_attn_mask):
        decay_weight = 0
        all_time_embed = torch.cat([previous_embed, current_embed.unsqueeze(1)], dim=1)
        bs = all_time_embed.shape[0]
        q = self.q_linear(current_embed).unsqueeze(1).view(bs, -1, self.head, self.d_k).transpose(1, 2)
        k = self.k_linear(all_time_embed).view(bs, -1, self.head, self.d_k).transpose(1, 2)
        v = self.v_linear(all_time_embed).view(bs, -1, self.head, self.d_k).transpose(1, 2)
        scores = self.attention(q, k, v, local_attn_mask, decay_weight)
        return scores.transpose(1, 2).contiguous().view(bs, self.input_dim)

    def attention(self, q, k, v, local_attn_mask, decay_weight):
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        norm = softmax(scores.squeeze() + local_attn_mask.unsqueeze(1) + decay_weight, dim=-1)
        return torch.matmul(norm.unsqueeze(2), v).squeeze()
