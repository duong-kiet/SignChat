import torch.nn as nn
from torch import Tensor
from transformer_layers import TransformerEncoderLayer, PositionalEncoding

from embeddings import MaskedNorm
from helpers import freeze_params


class Encoder(nn.Module):
    """
    Base encoder class
    """
    @property
    def output_size(self):
        return self._output_size

class TransformerEncoder(Encoder):
    """
    Transformer Encoder class
    """
    def __init__(self,
                 hidden_size: int = 512,
                 ff_size: int = 2048,
                 num_heads: int = 8,
                 num_layers: int = 8,
                 dropout: float = 0.1,
                 emb_dropout: float = 0.1,
                 freeze: bool = False,
                 **kwargs):
        """
        Initializes the Transformer.
        :param hidden_size: hidden size and size of embeddings
        :param ff_size: position-wise feed-forward layer size.
          (Typically this is 2*hidden_size.)
        :param num_layers: number of layers
        :param num_heads: number of heads for multi-headed attention
        :param dropout: dropout probability for Transformer layers
        :param emb_dropout: Is applied to the input (word embeddings).
        :param freeze: freeze the parameters of the encoder during training
        :param kwargs:
        """
        super().__init__()

        self.enc_layers = nn.ModuleList([
            TransformerEncoderLayer(d_model = hidden_size, ff_size = ff_size,
                                    num_heads = num_heads, dropout =dropout)
            for _ in range(num_layers)])

        self.layer_norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.pe = PositionalEncoding(hidden_size)
        self.emb_dropout = nn.Dropout(emb_dropout)
        self._output_size = hidden_size

        if freeze:
            freeze_params(self)

    def forward(self,
                embed_src: Tensor,
                src_length: Tensor,
                mask: Tensor) ->(Tensor, Tensor):

        """
        Pass the input (and mask) through each layer in turn.
        Applies a Transformer encoder to sequence of embeddings x.
        The input mini-batch x needs to be sorted by src length.
        x and mask should have the same dimensions [batch, time, dim].

        :param embed_src: embedded src inputs,
            shape (batch_size, src_len, embed_size)
        :param src_length: length of src inputs
            (counting tokens before padding), shape (batch_size)
        :param mask: indicates padding areas (zeros where padding), shape
            (batch_size, src_len, embed_size)
        :return:
            - output: hidden states with
                shape (batch_size, max_length, directions*hidden),
            - hidden_concat: last hidden state with
                shape (batch_size, directions*hidden)
        """

        x = embed_src

        x = self.pe(x)

        x = self.emb_dropout(x)

        for layer in self.enc_layers:
            x = layer(x, mask)

        return self.layer_norm(x), None
