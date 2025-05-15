"""
Validation function for evaluating a model on a dataset.
"""

import numpy as np
import math
import torch
from torch.utils.data import Dataset
from helpers import bpe_postprocess, load_config, get_latest_checkpoint, \
    load_checkpoint, calculate_dtw
from model import build_model, Model
from batch import Batch
from data import load_data, make_data_iter

def validate_on_data(model: Model,
                     data: Dataset,
                     batch_size: int,
                     max_output_length: int,
                     eval_metric: str,
                     loss_function: torch.nn.Module = None,
                     type: str = "val",
                     BT_model=None):
    """
    Validate the model on a given dataset.

    Args:
        model (Model): The model to evaluate.
        data (Dataset): The dataset for validation.
        batch_size (int): Batch size for processing.
        max_output_length (int): Maximum output length for generation.
        eval_metric (str): Evaluation metric to use.
        loss_function (torch.nn.Module, optional): Loss function for computing loss.
        type (str): Type of evaluation ("val" or other).
        BT_model: Optional auxiliary model (e.g., for back-translation).

    Returns:
        tuple: Contains current_valid_score, valid_loss, valid_references,
               valid_hypotheses, valid_inputs, all_dtw_scores, file_paths.
    """
    # Create an iterator for the validation dataset
    valid_iter = make_data_iter(
        dataset=data,
        batch_size=batch_size,
        shuffle=False
    )

    # Set model to evaluation mode (disable dropout)
    model.eval()

    # Get the actual model from DataParallel if needed
    model_module = model.module if hasattr(model, 'module') else model
    src_vocab = model_module.src_vocab
    out_trg_size = model_module.out_trg_size
    use_cuda = model_module.use_cuda
    just_count_in = model_module.just_count_in
    future_prediction = model_module.future_prediction

    # Disable gradient computation during validation
    with torch.no_grad():
        # Lists to store hypotheses, references, inputs, file paths, and DTW scores
        valid_hypotheses = []
        valid_references = []
        valid_inputs = []
        file_paths = []
        all_dtw_scores = []

        # Initialize loss and counters
        valid_loss = 0
        total_ntokens = 0
        total_nseqs = 0

        # Counter for batches processed
        batches = 0

        # Iterate over validation batches
        for valid_batch in valid_iter:
            # Create a Batch object from the raw batch data
            batch = Batch(
                batch=valid_batch,  # Updated to pass dictionary directly
                src_vocab=src_vocab,
                trg_size=out_trg_size,
                model=model,
                device=torch.device("cuda" if use_cuda else "cpu")
            )

            targets = batch.trg  # Extract target sequences

            # Compute loss if loss function and targets are provided
            if loss_function is not None and batch.trg is not None:
                # Truy cập get_loss_for_batch qua model hoặc model.module
                if hasattr(model, 'module'):
                    batch_loss, _ = model.module.get_loss_for_batch(
                        batch, loss_function=loss_function)
                else:
                    batch_loss, _ = model.get_loss_for_batch(
                        batch, loss_function=loss_function)
                valid_loss += batch_loss.item()
                total_ntokens += batch.ntokens
                total_nseqs += batch.nseqs

            # Run inference to generate outputs if not just counting
            if not just_count_in:
                # Truy cập run_batch qua model hoặc model.module
                if hasattr(model, 'module'):
                    output, attention_scores = model.module.run_batch(batch=batch)
                else:
                    output, attention_scores = model.run_batch(batch=batch)

            # Handle future prediction if enabled
            if future_prediction != 0:
                # Truncate targets to keep only the first frame and counter
                targets = torch.cat(
                    (targets[:, :, :targets.shape[2] // future_prediction],
                     targets[:, :, -1:]),
                    dim=2
                )

            # If only counting, use targets as output
            if just_count_in:
                output = targets  # Fix: Use targets instead of undefined train_output

            # Collect references, hypotheses, file paths, and source inputs
            valid_references.extend(targets)
            valid_hypotheses.extend(output)
            file_paths.extend(batch.file_paths)
            valid_inputs.extend([
                [src_vocab.itos[batch.src[i][j]] for j in range(len(batch.src[i]))]
                for i in range(len(batch.src))
            ])

            # Calculate Dynamic Time Warping (DTW) scores for evaluation
            dtw_score = calculate_dtw(targets, output)
            all_dtw_scores.extend(dtw_score)

            # Limit the number of batches processed (optional)
            if batches == math.ceil(20 / batch_size):
                break
            batches += 1

        # Compute average DTW score for validation
        current_valid_score = np.mean(all_dtw_scores)

    # Return validation results
    return (current_valid_score, valid_loss, valid_references, valid_hypotheses,
            valid_inputs, all_dtw_scores, file_paths)