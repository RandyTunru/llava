import torch
from torch import nn

from src.models.llama.modules.block import Block

class LLaMA(nn.Module):
    def __init__(self, vocab_size, d_model=768, num_layers=12, num_heads=12, d_ff=3072, max_seq_len=512, original_seq_len=512, dropout=0.0):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.original_seq_len = original_seq_len if original_seq_len is not None else max_seq_len

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList([
            Block(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.output_layer = nn.Linear(d_model, vocab_size)

        # RoPE Frequencies 
        head_dim = self.d_model // self.num_heads
        freqs = torch.arange(0, head_dim, 2) / head_dim # (head_dim/2,)
        freqs = 1 / (10000 ** freqs)

        # Scale factor to adjust the RoPE frequencies based on the original sequence length
        # This is a simple implementation for CPT so we don't have to worry the model can't extrapolate to longer sequence lengths.
        # For better extrapolation, can look into methods like YaRN, LongRoPE, or NTK-aware Interpolation.
        scale_factor = self.max_seq_len / self.original_seq_len

        # Divide by scale_factor to adjust the effective sequence length for RoPE
        t = torch.arange(self.max_seq_len) / scale_factor # (max_seq_len,)

        angles = torch.outer(t, freqs) # (max_seq_len, head_dim/2)
        freqs_cis = torch.polar(torch.ones_like(angles), angles) 
        self.register_buffer("freqs_cis", freqs_cis, persistent=False)

        # Causal Mask
        mask = torch.tril(torch.ones(max_seq_len, max_seq_len)).bool().unsqueeze(0).unsqueeze(0)
        self.register_buffer("causal_mask", mask, persistent=False)

    def forward(self, input_ids, attention_mask=None):
        batch_size, seq_len = input_ids.shape
        assert seq_len <= self.max_seq_len, "Sequence length exceeds max_seq_len."

        x = self.embedding(input_ids)

        batch_mask = self.causal_mask[:, :, :seq_len, :seq_len]

        if attention_mask is not None:
            batch_mask = batch_mask & attention_mask

        for block in self.blocks:
            x = block(x, mask=batch_mask)

        x = self.norm(x)
        logits = self.output_layer(x)

        return logits