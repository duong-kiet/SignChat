"""
Implementation of a mini-batch.
"""

import torch
from typing import Optional
import torch.nn.functional as F

from vocabulary import Vocabulary
from constants import TARGET_PAD, PAD_TOKEN


class Batch:
    """Object for holding a batch of data with masks during training.
    Input is a batch dictionary from a DataLoader.
    """

    def __init__(
        self,
        batch: dict,
        src_vocab: Vocabulary,
        trg_size: int,
        model,
        device: torch.device
    ):

        """
        Create a new batch from a batch dictionary.

        :param batch: Dictionary containing 'src', 'trg', and 'file_paths' from DataLoader
        :param src_vocab: Source vocabulary for padding index
        :param trg_size: Size of target frame (including joints + counter)
        :param model: Model object containing configuration (e.g., just_count_in, future_prediction)
        :param device: Device to place tensors on (e.g., 'cpu' or 'cuda')
        """

        self.src = batch['src']  # Shape: (batch_size, src_length)
        self.src_lengths = torch.sum(self.src != src_vocab.stoi[PAD_TOKEN], dim=1)  # Length of each source sequence
        self.src_mask = (self.src != src_vocab.stoi[PAD_TOKEN]).unsqueeze(1)  # Shape: (batch_size, 1, src_length)
        self.nseqs = self.src.size(0)  # Batch size
        self.file_paths = batch['file_paths']
        self.device = device
        self.target_pad = TARGET_PAD
        
        # Truy cập các thuộc tính của model, hỗ trợ DataParallel
        if hasattr(model, 'module'):  # DataParallel wraps model in module attribute
            self.just_count_in = model.module.just_count_in
            self.future_prediction = model.module.future_prediction
            self.gaussian_noise = model.module.gaussian_noise
        else:
            self.just_count_in = model.just_count_in
            self.future_prediction = model.future_prediction
            self.gaussian_noise = model.gaussian_noise

        # Initialize target-related attributes
        self.trg_input: Optional[torch.Tensor] = None
        self.trg: Optional[torch.Tensor] = None
        self.trg_mask: Optional[torch.Tensor] = None
        self.trg_lengths: Optional[int] = None
        self.ntokens: Optional[int] = None

        if 'trg' in batch:
            trg = batch['trg']  #Shape: (batch_size, trg_length, trg_size)
            self.trg_lengths = trg.size(1)

            # trg_input is used for teacher forcing, excluding the last frame
            self.trg_input = trg[:, :-1, :].clone()

            # trg is used for loss computation, shifted by one (exclude BOS)
            self.trg = trg[:, 1:, :].clone()

            # Handle just_count_in mode
            if self.just_count_in:
                self.trg_input = self.trg_input[:, :, -1:]  # Keep only the counter

            # Handle future_prediction mode
            if self.future_prediction != 0:
                future_trg = []
                for i in range(self.future_prediction):
                    # Concatenate shifted frames (excluding counter)
                    future_trg.append(self.trg[:, i:-(self.future_prediction - i), :-1].clone())
                future_trg = torch.cat(future_trg, dim=2)
                # Combine with counter from original trg
                self.trg = torch.cat((future_trg, self.trg[:, :-self.future_prediction, -1:]), dim=2)
                self.trg_input = self.trg_input[:, :-self.future_prediction, :]

            # Create target mask to exclude padded areas
            # Shape: (batch_size, 1, 1, trg_length)
            self.trg_mask = (self.trg_input[:, :, 0] != self.target_pad).unsqueeze(1).unsqueeze(2)

            # Calculate number of non-padded tokens in target
            self.ntokens = (self.trg != self.target_pad).sum().item()

        # Move tensors to the specified device
        self._to_device()

    def _to_device(self):
        """ Move all tensors in the batch to the specified device. """

        self.src = self.src.to(self.device)
        self.src_lengths = self.src_lengths.to(self.device)
        self.src_mask = self.src_mask.to(self.device)

        if self.trg_input is not None:
            self.trg_input = self.trg_input.to(self.device)
            self.trg = self.trg.to(self.device)
            self.trg_mask = self.trg_mask.to(self.device)
        
