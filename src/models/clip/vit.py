import torch
from torch import nn

from src.models.clip.modules.block import Block

class VisionTransformer(nn.Module):
    def __init__(self, in_channels=3, patch_size=16, d_model=768, d_ff=3072, num_heads=12, num_layers=12, max_seq_len=36, dropout=0.0):
        super().__init__()
        self.patch_size = patch_size

        # Patch embedding
        self.patch_embedding = nn.Conv2d(in_channels, d_model, kernel_size=patch_size, stride=patch_size)

        # Positional encoding
        self.positional_encoding = nn.Parameter(torch.zeros(1, max_seq_len, d_model))

        # Transformer blocks
        self.blocks = nn.ModuleList([
            Block(d_model=d_model, num_heads=num_heads, d_ff=d_ff, dropout=dropout)
            for _ in range(num_layers)
        ])

        # Layer normalization
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        # x shape: (batch_size, in_channels, height, width)
        batch_size, _, height, width = x.shape

        # Patch embedding
        x = self.patch_embedding(x)  # (batch_size, d_model, height/patch_size, width/patch_size)
        x = x.flatten(2).transpose(1, 2)  # (batch_size, num_patches, d_model)

        # Add positional encoding
        seq_len = x.size(1)
        x = x + self.positional_encoding[:, :seq_len, :]

        # Pass through transformer blocks
        for block in self.blocks:
            x = block(x)

        # Final normalization
        x = self.layer_norm(x)

        return x  # (batch_size, num_patches, d_model)