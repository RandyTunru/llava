import torch
from torch import nn

from src.models.clip.modules.block import Block

class TextTransformer(nn.Module):
    """Text transformer model for CLIP.
    
    In the original CLIP implementation, it uses the GPT-1 version of transformer, which uses a causal attention.
    Explicitly quoted "The text encoder is a Transformer (Vaswani et al., 2017) with the architecture modifications described in Radford et al. (2019)."
    """
    def __init__(self, vocab_size, d_model=512, d_ff=2048, num_heads=8, num_layers=6, max_seq_len=1024, dropout=0.1):
        super().__init__()
        self.d_model = d_model

        # Token embedding
        self.token_embedding = nn.Embedding(vocab_size, d_model)

        # Positional encoding
        self.positional_encoding = nn.Parameter(torch.zeros(1, max_seq_len, d_model))

        # Transformer blocks
        self.blocks = nn.ModuleList([
            Block(d_model=d_model, num_heads=num_heads, d_ff=d_ff, dropout=dropout)
            for _ in range(num_layers)
        ])

        # Layer normalization
        self.layer_norm = nn.LayerNorm(d_model)

        # Causal mask 
        mask = torch.tril(torch.ones(max_seq_len, max_seq_len)).bool().unsqueeze(0).unsqueeze(0)
        self.register_buffer("causal_mask", mask, persistent=False)

    def forward(self, x):
        # x shape: (batch_size, seq_len)
        batch_size, seq_len = x.shape
        assert seq_len <= self.positional_encoding.size(1), "Sequence length exceeds max_seq_len."

        # Token embedding
        x = self.token_embedding(x)  # (batch_size, seq_len, d_model)

        # Add positional encoding
        x = x + self.positional_encoding[:, :seq_len, :]

        # Pass through transformer blocks
        for block in self.blocks:
            x = block(x, self.causal_mask[:, :, :seq_len, :seq_len])

        # Final normalization
        x = self.layer_norm(x)

        return x  # (batch_size, seq_len, d_model)