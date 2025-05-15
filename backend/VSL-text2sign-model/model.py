"""
Module to represents whole models
"""
from typing import Any

import numpy as np
import torch.nn as nn
from torch import Tensor
import torch

from constants import BOS_TOKEN, PAD_TOKEN, EOS_TOKEN, TARGET_PAD
from vocabulary import Vocabulary
from embeddings import Embedding
from encoder import Encoder, TransformerEncoder
from decoder import Decoder, TransformerDecoder
from initialization import initialize_model
from search import greedy
from batch import Batch


class Model(nn.Module):
    """
    Base class for all models
    """

    def __init__(self,
                 encoder: Encoder,
                 decoder: Decoder,
                 src_embed: Embedding,
                 trg_embed: nn.Module,
                 src_vocab: Vocabulary,
                 cfg: dict,
                 in_trg_size: int,
                 out_trg_size: int) -> None:
        """
        Initializes the model
        :param encoder: Encoder object
        :param decoder: Decoder object
        :param cfg: configuration dictionary
        :param in_trg_size: input target size
        :param out_trg_size: output target size
        """
        super(Model, self).__init__()

        model_cfg = cfg["model"]

        self.src_embed = src_embed
        self.trg_embed = trg_embed

        self.encoder = encoder
        self.decoder = decoder

        self.src_vocab = src_vocab
        
        self.bos_index = self.src_vocab.stoi[BOS_TOKEN]
        self.eos_index = self.src_vocab.stoi[EOS_TOKEN]
        self.pad_index = self.src_vocab.stoi[PAD_TOKEN]
        self.target_pad = TARGET_PAD

        self.use_cuda = cfg["training"]["use_cuda"]

        self.in_trg_size = in_trg_size
        self.out_trg_size = out_trg_size

        self.count_in = model_cfg.get("count_in",True)
        # Just Counter
        self.just_count_in = model_cfg.get("just_count_in",False)
        # Gaussian Noise
        self.gaussian_noise = model_cfg.get("gaussian_noise",False)
        # Gaussian Noise
        if self.gaussian_noise:
            self.noise_rate = model_cfg.get("noise_rate", 1.0)

        # Future Prediction - predict for this many frames in the future
        self.future_prediction = model_cfg.get("future_prediction", 0)
        self.out_stds = None

    def forward(self,
                src: Tensor,
                trg_input: Tensor,
                src_mask: Tensor,
                src_lengths: Tensor,
                trg_mask: Tensor = None) -> (Tensor, Tensor):

        """
        First encodes the source sentence.

        :param src: source input
        :param trg_input: target input
        :param src_mask: source mask
        :param src_lengths: length of source inputs
        :param trg_mask: target mask
        :return: decoder outputs
        """

        # Encode the source sequence
        encoder_output, encoder_hidden = self.encode(src=src,
                                                     src_length=src_lengths,
                                                     src_mask=src_mask)

        unroll_steps = trg_input.size(1) # Time step (num frames)

        # Add gaussian noise to the target inputs, if in training
        if self.gaussian_noise and self.training and self.out_stds is not None:
            # Create a normal distribution of random numbers between 0 and 1
            noise = torch.randn_like(trg_input)
            # Zero out the noise over the counter
            noise[:, :, -1] = torch.zeros_like(noise[:, :, -1])

            if self.future_prediction != 0:
                self.out_stds = torch.cat((self.out_stds, torch.zeros_like(self.out_stds)))[:trg_input.shape[-1]]
            
            # Multiply by the standard deviations
            noise = noise * self.out_stds

            # Add to trg_input multiplied by the noise rate
            trg_input = trg_input + self.noise_rate * noise

        skel_out, dec_hidden, _, _ = self.decode(encoder_output = encoder_output,
                                                 src_mask=src_mask,
                                                 trg_input=trg_input,
                                                 trg_mask=trg_mask)

        return skel_out

    def encode(self, src: Tensor, src_length: Tensor, src_mask: Tensor) -> (Tensor, Tensor):
        return self.encoder(self.src_embed(src), src_length, src_mask)

    def decode(self,
               encoder_output: Tensor,
               src_mask: Tensor,
               trg_input: Tensor,
               trg_mask: Tensor = None) -> (Tensor, Tensor, Tensor, Tensor):
        
        """
        Decode, given an encoded source sentence.

        :param encoder_output: encoder states for attention computation
        :param encoder_hidden: last encoder state for decoder initialization
        :param src_mask: source mask, 1 at valid tokens
        :param trg_input: target inputs
        :param unroll_steps: number of steps to unrol the decoder for
        :param decoder_hidden: decoder hidden state (optional)
        :param trg_mask: mask for target steps
        :return: decoder outputs (outputs, hidden, att_probs, att_vectors)
        """

        trg_embed = self.trg_embed(trg_input)
        
        decoder_output = self.decoder(trg_embed=trg_embed,
                                     enc_output=encoder_output,
                                     src_mask=src_mask,
                                     trg_mask=trg_mask)
                                    
        return decoder_output

    def get_loss_for_batch(self, batch: Batch, loss_function: nn.Module) -> (Tensor, Tensor):
        """
        Compute non-normalized loss and number of tokens for a batch

        :param batch: batch to compute loss for
        :param loss_function: loss function, computes for input and target
            a scalar loss for the complete batch
        :return: batch_loss: sum of losses over non-pad elements in the batch
        """

        skel_out = self.forward(src=batch.src,
                                trg_input=batch.trg_input,
                                src_mask=batch.src_mask,
                                src_lengths=batch.src_lengths,
                                trg_mask=batch.trg_mask)

        batch_loss = loss_function(skel_out, batch.trg)

        # Xác định thuộc tính gaussian_noise và future_prediction từ batch hoặc từ model
        gaussian_noise = batch.gaussian_noise if hasattr(batch, 'gaussian_noise') else self.gaussian_noise
        future_prediction = batch.future_prediction if hasattr(batch, 'future_prediction') else self.future_prediction

        if gaussian_noise:
            with torch.no_grad():
                noise = skel_out.detach() - batch.trg.detach()
            if future_prediction != 0:
                noise = noise[:, :, :noise.shape[2] // future_prediction]
        else:
            noise = None

        return batch_loss, noise

    def run_batch(self, batch: Batch) -> (np.array, np.array):

        """
        Get outputs and attentions scores for a given batch

        :param batch: batch to generate hypotheses for
        :param beam_size: size of the beam for beam search, if 0 use greedy
        :param beam_alpha: alpha value for beam search
        :return: stacked_output: hypotheses for batch,
            stacked_attention_scores: attention scores for batch
        """

        encoder_output, encoder_hidden = self.encode(batch.src,
                                                     batch.src_lengths,
                                                     batch.src_mask)

        stacked_output, stacked_attention_scores = greedy(
            enc_output=encoder_output,
            src_mask=batch.src_mask,
            embed=self.trg_embed,
            decoder=self.decoder,
            trg_input=batch.trg_input,
            model=self
        )

        return stacked_output, stacked_attention_scores

    def __repr__(self) -> str:
        return "%s(\n" \
               "\tencoder=%s,\n" \
               "\tdecoder=%s,\n" \
               "\tsrc_embed=%s,\n" \
               "\ttrg_embed=%s)" % (self.__class__.__name__, self.encoder,
                                        self.decoder, self.src_embed, self.trg_embed)

def build_model(cfg: dict = None,
                src_vocab: Vocabulary = None,
                trg_vocab: Vocabulary = None) -> Model:

    """
    Build and initialize the model according to the configuration.

    :param cfg: dictionary configuration containing model specifications
    :param src_vocab: source vocabulary
    :param trg_vocab: target vocabulary
    :return: built and initialized model
    """

    full_cfg = cfg
    cfg = cfg["model"]

    src_padding_idx = src_vocab.stoi[PAD_TOKEN]
    trg_padding_idx = 0

    # Input target size is the joint vector length plus one for counter
    in_trg_size = cfg["trg_size"] + 1
    # Output target size is the joint vector length plus one for counter
    out_trg_size = cfg["trg_size"] + 1

    just_count_in = cfg.get("just_count_in", False)
    future_prediction = cfg.get("future_prediction", 0)

    #  Just count in limits the in target size to 1
    if just_count_in:
        in_trg_size = 1

    # Future Prediction increases the ouput tagret size
    if future_prediction != 0:
        # Times the trg_size (minus counter) by amount of predicted frames, and then add back counter
        out_trg_size = (out_trg_size - 1) * future_prediction + 1

    # Define source embedding
    src_embed = Embedding(
        **cfg["encoder"]["embeddings"], vocab_size=len(src_vocab),
        padding_idx=src_padding_idx
    )

    # Define target linear
    # Linear layer replaces an embedding layer - as this takes in the joints size as opposed to a token
    trg_linear = nn.Linear(in_trg_size, cfg["decoder"]["embeddings"]["embedding_dim"])

    ## Encoder --
    enc_dropout = cfg["encoder"].get("dropout", 0.)
    enc_emb_dropout = cfg["encoder"]["embeddings"].get("dropout", enc_dropout)
    assert cfg["encoder"]["embeddings"]["embedding_dim"] == cfg["encoder"]["hidden_size"]

    encoder = TransformerEncoder(**cfg["encoder"],
                                 emb_dropout=enc_emb_dropout)

    ## Decoder --
    dec_dropout = cfg["decoder"].get("dropout", 0.)
    dec_emb_dropout = cfg["decoder"]["embeddings"].get("dropout", dec_dropout)
    decoder_trg_trg = cfg["decoder"].get("decoder_trg_trg", True)

    decoder = TransformerDecoder(**cfg["decoder"],
                             encoder=encoder,
                             emb_dropout=dec_emb_dropout,
                             trg_size=out_trg_size,
                             decoder_trg_trg_=decoder_trg_trg)

    ## Define the model
    model = Model(encoder=encoder,
                  decoder=decoder,
                  src_embed=src_embed,
                  trg_embed=trg_linear,
                  src_vocab=src_vocab,
                  cfg=full_cfg,
                  in_trg_size=in_trg_size,
                  out_trg_size=out_trg_size)

    initialize_model(model, cfg, src_padding_idx, trg_padding_idx)

    return model