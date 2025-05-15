
from torch import Tensor
import torch.nn as nn

from helpers import freeze_params, subsequent_mask
from transformer_layers import TransformerDecoderLayer, \
    PositionalEncoding

class Decoder(nn.Module):
    """
    Base decoder class
    """

    @property
    def output_size(self):
        """
        Return the output size (size of the target vocabulary)

        :return:
        """
        return self._output_size

class TransformerDecoder(Decoder):

    def __init__(self,
                 num_layers: int = 4,
                 num_heads: int = 8,
                 hidden_size: int = 512,
                 ff_size: int = 2048,
                 dropout: float = 0.1,
                 emb_dropout: float = 0.1,
                 freeze: bool = False,
                 trg_size: int = 97,
                 decoder_trg_trg: bool = True,
                 **kwargs):
        super().__init__()

        self.hidden_size = hidden_size
        self._output_size = trg_size

        self.layers = nn.ModuleList([TransformerDecoderLayer(
                d_model=hidden_size, ff_size=ff_size, num_heads=num_heads,
                dropout=dropout, decoder_trg_trg=decoder_trg_trg) for _ in range(num_layers)])

        self.pe = PositionalEncoding(hidden_size, mask_count=True)

        self.layer_norm = nn.LayerNorm(hidden_size, eps=1e-6)

        self.emb_dropout = nn.Dropout(p = emb_dropout)

        self.output_layer = nn.Linear(hidden_size, trg_size, bias=False)

        if freeze:
            freeze_params(self)

    def forward(self,
                trg_embed: Tensor = None,
                trg_length: Tensor = None,
                enc_output: Tensor = None,
                src_mask: Tensor = None,
                trg_mask: Tensor = None,
                **kwargs) -> (Tensor, Tensor):
        """
        Forward pass for the decoder.

        :param trg_embed: embedded targets
        :param trg_length: length of the target sequence
        :param enc_output: output from the encoder
        :param src_mask: mask for the encoder output
        :param trg_mask: mask for the decoder input
        :return:

        Args:
            trg_embed:
        """
        assert trg_mask is not None, "trg_mask required for Transformer"

        x = self.pe(trg_embed)

        x = self.emb_dropout(x)

        padding_mask = trg_mask
        # Create subsequent mask for decoding
        sub_mask = subsequent_mask(
            trg_embed.size(1)).type_as(trg_mask)

        # Apply each layer to the input
        for layer in self.layers:
            x = layer(x=x, enc_output=enc_output,
                      src_mask=src_mask, trg_mask=sub_mask, padding_mask=padding_mask)

        # Apply a layer normalisation
        x = self.layer_norm(x)

        # Apply the output layer
        output = self.output_layer(x)

        return output, x, None, None
        