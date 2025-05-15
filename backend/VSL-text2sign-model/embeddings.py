import math

import torch
from torch import nn, Tensor


class MaskedNorm(nn.Module):
    def __init__(self, norm_type, num_groups, num_features):
        super(MaskedNorm, self).__init__()
        self.norm_type = norm_type
        if self.norm_type == 'batch':
            self.norm = nn.BatchNorm1d(num_features = num_features)
        elif self.norm_type == 'group':
            self.norm = nn.GroupNorm(num_groups = num_groups, num_channels = num_features)
        elif self.num_type == 'layer':
            self.norm = nn.LayerNorm(num_features)
        else:
            raise ValueError(f"Unsupported norm type: {self.norm_type}")
        self.num_features = num_features

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        if self.training:
            reshaped = x.reshape([-1, self.num_features])
            reshaped_mask = mask.reshape([-1, 1]) > 0
            selected = torch.masked_select(reshaped, reshaped_mask).reshape(
                [-1, self.num_features]
            )

            batch_normed = self.norm(selected)
            scattered = reshaped.masked_scatter(reshaped_mask, batch_normed)

            return scattered.reshape([x.shape[0], -1, self.num_features])
        else:
            reshaped = x.reshape([-1, self.num_features])
            batched_normed = self.norm(reshaped)
            return batched_normed.reshape([x.shape[0], -1, self.num_features])

class Embedding(nn.Module):
    def __init__(self,
                embedding_dim: int = 512,
                vocab_size: int = 0,
                scale: bool = False,
                max_length: int = 512,
                padding_idx: int = 1,
                freeze: bool = False,
                **kwargs):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        self.scale = scale
        self.max_length = max_length
        self.padding_idx = padding_idx
        self.lut = nn.Embedding(
            vocab_size, embedding_dim, padding_idx=padding_idx)

        if freeze:
            self.lut.weight.requires_grad = False

    def forward(self, x: Tensor) -> Tensor:
        """
        performs embedding lookup for the input indices
        :param x: index of words in the vocabulary
        :return: embedded words
        """
        if self.scale:
            x = self.lut(x) * math.sqrt(self.embedding_dim)
        return self.lut(x)

    def repre(self) -> str:
        """
        Returns a string representation of the model
        :return: string representation of the model
        """
        return f"InputEmbedding(embedding_dim={self.embedding_dim}, vocab_size={self.vocab_size}, max_length={self.max_length}, padding_idx={self.padding_idx})"