import torch
from torch import nn
from torch.nn import functional as F

from src.models.modules.components import RMSNorm
from src.models.modules.attention import MultiHeadAttention
from src.models.modules.ffn import SwiGluFeedForward

class Block(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.0):
        super(Block, self).__init__()
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.ffn = SwiGluFeedForward(d_model, d_ff)
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attn_output = self.attention(self.norm1(x))
        x = x + self.dropout(attn_output)

        ffn_output = self.ffn(self.norm2(x))
        x = x + self.dropout(ffn_output)

        return x