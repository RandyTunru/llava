import torch
from torch import nn
from torch.nn import functional as F
    
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.linear_q = nn.Linear(d_model, d_model, bias=False)
        self.linear_k = nn.Linear(d_model, d_model, bias=False)
        self.linear_v = nn.Linear(d_model, d_model, bias=False)
        self.linear_out = nn.Linear(d_model, d_model, bias=False)

    def _apply_rope(self, x, freqs_cis):
        # x: (batch_size, num_heads, seq_len, d_k)
        # freqs_cis: (seq_len, d_k) or (1, 1, seq_len, d_k)
        seq_len = x.size(2)

        freqs_cis = freqs_cis[:seq_len].to(x.device)

        x_reshaped = x.float().view(*x.shape[:-1], self.d_k // 2, 2)
        x_complex = torch.view_as_complex(x_reshaped)

        x_rotated = x_complex * freqs_cis

        x_out = torch.view_as_real(x_rotated).flatten(-2)
        return x_out.type_as(x)
        
    def forward(self, x, mask=None, freqs_cis=None):
        batch_size = x.size(0)
        
        # Linear projections
        q = self.linear_q(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        k = self.linear_k(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        v = self.linear_v(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        if freqs_cis is not None:
            q = self._apply_rope(q, freqs_cis)
            k = self._apply_rope(k, freqs_cis)

        # Scaled dot-product attention (Manual implementation commented out for optimization)
        # scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)

        # if mask is not None:
        #     scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # attn_weights = F.softmax(scores, dim=-1)
        # attn_output = torch.matmul(attn_weights, v)

        # Optimized attention computation using PyTorch's built-in function
        attn_output = F.scaled_dot_product_attention(q, k, v, is_causal=False, attn_mask=mask)
        
        # Concatenate heads and pass through final linear layer
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.linear_out(attn_output)
        
        return output
    


