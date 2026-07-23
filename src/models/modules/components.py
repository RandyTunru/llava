import torch
from torch import nn

class QuickGELU(nn.Module):
    """Implementation of the QuickGELU activation function.
    
    QuickGELU is an approximation of the GELU activation function, which is usually used 
    Place of GELU in transformer architectures for faster computation while maintaining similar performance.
    """
    def forward(self, x):
        return x * torch.sigmoid(1.702 * x)

class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-8):
        super(RMSNorm, self).__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        norm_x = x / torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return norm_x * self.weight