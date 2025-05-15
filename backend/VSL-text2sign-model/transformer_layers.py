
import math
import torch
import torch.nn as nn
from torch import Tensor
from constants import TARGET_PAD
import logging


class PositionalEncoding(nn.Module):
    """
    Positional Encoding class
    """

    def __init__(self,
                 d_model: int,
                 max_len: int = 200000,
                 mask_count=False):
        """
        Initializes the Positional Encoding.
        :param d_model: the dimension of the model
        :param dropout: dropout probability for Transformer layers
        :param max_len: maximum length of the input sequence
        """
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) # (1, max_len, d_model)

        self.register_buffer('pe', pe)
        self.dim = d_model
        self.mask_count = mask_count

    def forward(self, emb: Tensor) -> Tensor:
        return emb + self.pe[:, :emb.size(1), :]

class MultiHeadedAttention(nn.Module):
    """
    Multi-Headed Attention class
    """
    def __init__(self, num_heads: int, embedding_dim: int, dropout: float = 0.1):
        super().__init__()
        assert embedding_dim % num_heads == 0, "Embedding dimension must be divisible by number of heads"

        self.num_heads = num_heads
        self.d_k = embedding_dim // num_heads
        self.d_model = embedding_dim

        self.w_q = nn.Linear(embedding_dim, embedding_dim)
        self.w_k = nn.Linear(embedding_dim, embedding_dim)
        self.w_v = nn.Linear(embedding_dim, embedding_dim)

        self.w_o = nn.Linear(embedding_dim, embedding_dim)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q: Tensor, k: Tensor, v: Tensor, mask: Tensor = None, padding_mask: Tensor = None):

        batch_size = q.size(0)
        num_heads = self.num_heads

        # project the queries (q), keys (k), and values (v)
        q = self.w_q(q) # (batch_size, seq_len, d_model)
        k = self.w_k(k) # (batch_size, seq_len, d_model)
        v = self.w_v(v) # (batch_size, seq_len, d_model)

        # reshape to (batch_size, num_heads, seq_len, d_k)
        q = q.view(batch_size, -1, num_heads, self.d_k).transpose(1, 2)
        k = k.view(batch_size, -1, num_heads, self.d_k).transpose(1, 2)
        v = v.view(batch_size, -1, num_heads, self.d_k).transpose(1, 2)

        # calculate the attention scores
        attention_scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            # Reshape mask to match attention_scores dimensions
            # mask shape: (batch_size, 1, seq_len) -> (batch_size, 1, 1, seq_len)
            mask = mask.unsqueeze(1)
            attention_scores = attention_scores.masked_fill(mask == 0, float('-inf'))

        attention = self.softmax(attention_scores)
        attention = self.dropout(attention)

        if padding_mask is not None:

            if padding_mask.size(-1) != attention.size(-1):
                logging.warning(f"Padding mask size {padding_mask.size()} doesn't match attention size {attention.size()}") 
                # Either resize or skip applying this mask

            attention = attention.masked_fill(padding_mask == 0, 0.0)

        context= attention @ v # (batch_size, num_heads, seq_len, d_k)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, num_heads * self.d_k)

        output = self.w_o(context)

        return output

class PositionWiseFeedForward(nn.Module):
    """
    Position-wise Feed Forward class
    """
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.pwff_layer = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x: Tensor) -> Tensor:
        x_norm = self.layer_norm(x)
        return x + self.pwff_layer(x_norm)

class TransformerEncoderLayer(nn.Module):
    """
    Transformer Encoder Layer class
    """
    def __init__(self, d_model: int, ff_size: int = 2048, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.self_attn = MultiHeadedAttention(num_heads, d_model, dropout)
        self.feed_forward = PositionWiseFeedForward(d_model, ff_size, dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        x_norm = self.layer_norm(x)

        attn_output = self.self_attn(x_norm, x_norm, x_norm, mask)

        attn_output = self.dropout(attn_output) + x

        o = self.feed_forward(attn_output)

        return o

class TransformerDecoderLayer(nn.Module):
    """
    Transformer Decoder Layer class
    """
    def __init__(self,
                 d_model: int,
                 ff_size: int,
                 num_heads: int,
                 dropout: float = 0.1,
                 decoder_trg_trg: bool = True):
        super().__init__()
        self.d_model = d_model

        self.trg_trg_att = MultiHeadedAttention(num_heads, d_model, dropout=dropout)

        self.src_trg_att = MultiHeadedAttention(num_heads, d_model, dropout=dropout)

        self.feed_forward = PositionWiseFeedForward(d_model, ff_size, dropout)

        self.x_layer_norm = nn.LayerNorm(d_model, eps=1e-6)

        self.dec_layer_norm = nn.LayerNorm(d_model, eps=1e-6)

        self.dropout = nn.Dropout(dropout)

        self.decoder_trg_trg = decoder_trg_trg


    def forward(self,
                x: Tensor ,
                enc_output: Tensor,
                src_mask: Tensor = None,
                trg_mask: Tensor = None,
                padding_mask: Tensor = None) -> Tensor:
        # pre-LN 1
        x_norm = self.x_layer_norm(x)

        h1 = x_norm
        # Target-Target Self Attention
        if self.decoder_trg_trg:
            h1 = self.trg_trg_att(x_norm, x_norm, x_norm, trg_mask, padding_mask = padding_mask)
        h1 = self.dropout(h1) + x

        # pre-LN 2
        h1_norm = self.dec_layer_norm(h1)

        # Source-Target Self Attention
        h2 = self.src_trg_att(h1_norm, enc_output, enc_output, mask = src_mask)

        # final position-wise feed-forward layer
        o = self.feed_forward(self.dropout(h2) + h1)

        return o





