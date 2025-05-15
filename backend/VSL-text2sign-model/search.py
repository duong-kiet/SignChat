import torch
import torch.nn.functional as F
from torch import Tensor, nn
import numpy as np

from decoder import Decoder, TransformerDecoder


def greedy(
        src_mask: Tensor,
        embed: nn.Module,
        decoder: Decoder,
        enc_output: Tensor,
        trg_input: Tensor,
        model
        ) -> (np.array, np.array):
    """
    Special greedy function for transformer, since it works differently.
    The transformer remembers all previous states and attends to them.

    :param src_mask: mask for source inputs, 0 for positions after </s>
    :param embed: target embedding
    :param decoder: decoder to use for greedy decoding
    :param enc_output: encoder hidden states for attention
    :return:
        - stacked_output: output hypotheses (2d array of indices),
        - stacked_attention_scores: attention scores (3d array)
    """

    # Initialise the input
    # Extract just the BOS first frame from the target
    ys = trg_input[:,:1,:].float()

    # If the counter is coming into the decoder or not
    ys_out = ys

    # Set the target mask, by finding the padded rows
    trg_mask = trg_input != 0.0
    trg_mask = trg_mask.unsqueeze(1)

    # Find the maximum output length for this batch
    max_output_length = trg_input.shape[1]

    # If just count in, input is just the counter
    if model.just_count_in:
        ys = ys[:,:,-1:]

    for i in range(max_output_length):
        # ys here is the input
        # Driven the timing by giving the GT counter - add the counter to the input
        if model.count_in:
            # If just counter, drive the input using the GT counter
            ys[:, -1] = trg_input[:, i, -1:]
        else:
            # Give the GT counter for timing
            ys[:, -1, -1:] = trg_input[:, i, -1:]
        
        # Embed the tagret input
        trg_embed = embed(ys)

        # Cut padding mask to required size (of the size of the input)
        padding_mask = trg_mask[:, :, :i+1, :i+1]
        # Pad the mask (If required) (To make it square, and used later on correctly)
        pad_amount = padding_mask.shape[2] - padding_mask.shape[3]
        padding_mask = (F.pad(input=padding_mask.double(), pad=(pad_amount, 0, 0, 0), mode='replicate') == 1.0)

        # Pass the embedded input and the encoder output into the decoder
        with torch.no_grad():
            out, _, _, _ = decoder(
                trg_embed = trg_embed,
                enc_output = enc_output,
                src_mask = src_mask,
                trg_mask = padding_mask
            )

            if model.future_prediction != 0:
                # Cut to only the first frame prediction
                out = torch.cat((out[:, :, :out.shape[2] // (model.future_prediction)], out[:,:,-1:]), dim=2)             

            if model.just_count_in:
                # If just couter in trg_input, concatenate counters of output
                ys = torch.cat([ys, out[:, -1:, -1:]], dim=1)

            # Add this frame prediction to the overall prediction
            ys = torch.cat([ys, out[:, -1:, :]], dim=1)

            # All frame output
            ys_out = torch.cat([ys, out[:, -1:, :]], dim=1)
        
    return ys_out, None
