import torch
from torch import nn

from src.models.modules.components import QuickGELU

class SwiGluFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(SwiGluFeedForward, self).__init__()
        self.gate_linear = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=False),
            nn.SiLU()
        )
        self.up_linear = nn.Linear(d_model, d_ff, bias=False)
        self.down_linear = nn.Linear(d_ff, d_model, bias=False)
    
    def forward(self, x):
        gate_output = self.gate_linear(x)
        up_output = self.up_linear(x)
        ff_output = gate_output * up_output
        output = self.down_linear(ff_output)
        return output

class PositionwiseFeedForward(nn.Module):
    """Position-wise Feed-Forward Network (FFN) used in pre-LLaMa transformer architectures."""
    def __init__(self, d_model, d_ff):
        super(PositionwiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.activation = QuickGELU() # You can also change to nn.GELU()
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.linear2(self.activation(self.linear1(x)))