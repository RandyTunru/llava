import torch
from torch import nn
from torch.nn import functional as F

from src.models.clip.vit import VisionTransformer
from src.models.clip.text_transformer import TextTransformer

class CLIP(nn.Module):
    def __init__(self, vision_config: dict, text_config: dict, d_embed=512):
        super().__init__()
        self.vision_transformer = VisionTransformer(**vision_config)
        self.text_transformer = TextTransformer(**text_config)

        self.vision_proj = nn.Linear(vision_config['d_model'], d_embed)
        self.text_proj = nn.Linear(text_config['d_model'], d_embed)

        self.temperature = nn.Parameter(torch.tensor(1.0))

    def forward(self, image, text):
        # image shape: (batch_size, in_channels, height, width)
        # text shape: (batch_size, seq_len)
        vision_features = self.vision_transformer(image)  # (batch_size, num_patches, d_model)
        text_features = self.text_transformer(text)  # (batch_size, seq_len, d_model)

        vision_features = torch.mean(vision_features, dim=1)  # (batch_size, d_model)
        text_features = torch.mean(text_features, dim=1)  # (batch_size, d_model)

        vision_features = self.vision_proj(vision_features)  # (batch_size, d_embed)
        text_features = self.text_proj(text_features)  # (batch_size, d_embed)

        # L2 normalization
        vision_features = F.normalize(vision_features, p=2, dim=-1)
        text_features = F.normalize(text_features, p=2, dim=-1)

        logits = torch.matmul(vision_features, text_features.t()) * torch.exp(self.temperature)  # (batch_size, batch_size)

        return logits
        