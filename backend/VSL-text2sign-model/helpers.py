"""
Collection of helper functions
"""
import copy
import glob
import os
import errno
import shutil
import random
import logging
from logging import Logger
from typing import Callable, Optional, List, Dict, Any
import numpy as np
import torch
from torch import nn, Tensor
from torch.utils.tensorboard import SummaryWriter
import yaml
from dtw import dtw

class ConfigurationError(Exception):
    """Custom exception for misspecifications of configuration"""

def make_model_dir(model_dir: str, overwrite: bool = False, model_continue: bool = False) -> str:
    """
    Create a new directory for the model.

    :param model_dir: Path to model directory
    :param overwrite: Whether to overwrite an existing directory
    :param model_continue: Whether to continue from a checkpoint
    :return: Path to model directory
    """
    if os.path.isdir(model_dir):
        if model_continue:
            return model_dir
        if not overwrite:
            raise FileExistsError("Model directory exists and overwriting is disabled.")
        shutil.rmtree(model_dir, ignore_errors=True)
    os.makedirs(model_dir, exist_ok=True)
    return model_dir

def make_logger(model_dir: str, log_file: str = "train.log") -> Logger:
    """
    Create a logger for logging the training process.

    :param model_dir: Path to logging directory
    :param log_file: Path to logging file
    :return: Logger object
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(os.path.join(model_dir, log_file))
    fh.setLevel(logging.DEBUG)
    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(message)s')
    fh.setFormatter(formatter)
    sh.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(sh)
    logger.info("Progressive Transformers for End-to-End SLP")
    return logger

def log_cfg(cfg: Dict[str, Any], logger: Logger, prefix: str = "cfg") -> None:
    """
    Write configuration to log.

    :param cfg: Configuration to log
    :param logger: Logger that defines where log is written to
    :param prefix: Prefix for logging
    """
    for k, v in cfg.items():
        if isinstance(v, dict):
            p = f"{prefix}.{k}"
            log_cfg(v, logger, prefix=p)
        else:
            p = f"{prefix}.{k}"
            logger.info(f"{p:34s} : {v}")

def clones(module: nn.Module, n: int) -> nn.ModuleList:
    """
    Produce N identical layers. Transformer helper function.

    :param module: The module to clone
    :param n: Clone this many times
    :return: Cloned modules
    """
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])

def uneven_subsequent_mask(x_size: int, y_size: int) -> Tensor:
    """
    Mask out subsequent positions for uneven sizes.

    :param x_size: Size of mask (2nd dim)
    :param y_size: Size of mask (3rd dim)
    :return: Tensor with dtype=torch.bool and shape (1, x_size, y_size)
    """
    mask = torch.triu(torch.ones((1, x_size, y_size), dtype=torch.bool), diagonal=1)
    return ~mask

def subsequent_mask(size: int) -> Tensor:
    """
    Mask out subsequent positions (to prevent attending to future positions).

    :param size: size of mask (2nd and 3rd dim)
    :return: Tensor with dtype=torch.bool and shape (1, size, size)
    """
    mask = torch.triu(torch.ones((1, size, size), dtype=torch.bool), diagonal=1)
    return ~mask

def set_seed(seed: int) -> None:
    """
    Set the random seed for modules torch, numpy, and random.

    :param seed: Random seed
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_config(path: str = "configs/default.yaml") -> Dict[str, Any]:
    """
    Loads and parses a YAML configuration file.

    :param path: Path to YAML configuration file
    :return: Configuration dictionary
    """
    with open(path, 'r') as ymlfile:
        cfg = yaml.safe_load(ymlfile)
    return cfg

def bpe_postprocess(string: str) -> str:
    """
    Post-processor for BPE output. Recombines BPE-split tokens.

    :param string: String to process
    :return: Post-processed string
    """
    return string.replace("@@ ", "")

def get_latest_checkpoint(ckpt_dir: str, post_fix: str = "_every") -> Optional[str]:
    """
    Returns the latest checkpoint (by time) from the given directory.

    :param ckpt_dir: Directory of checkpoint
    :param post_fix: Type of checkpoint, either "_every" or "_best"
    :return: Latest checkpoint file or None
    """
    list_of_files = glob.glob(os.path.join(ckpt_dir, f"*{post_fix}.ckpt"))
    return max(list_of_files, key=os.path.getctime) if list_of_files else None

def load_checkpoint(path: str, use_cuda: bool = True) -> Dict[str, Any]:
    """
    Load model from saved checkpoint.

    :param path: Path to checkpoint
    :param use_cuda: Using CUDA or not
    :return: Checkpoint (dict)
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Checkpoint {path} not found")
    map_location = 'cuda' if use_cuda and torch.cuda.is_available() else 'cpu'
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    return checkpoint

def symlink_update(target: str, link_name: str) -> None:
    """
    Update a symbolic link.

    :param target: Target of the link
    :param link_name: Name of the link
    """
    try:
        os.symlink(target, link_name)
    except OSError as e:
        if e.errno == errno.EEXIST:
            os.remove(link_name)
            os.symlink(target, link_name)
        else:
            raise

def freeze_params(module: nn.Module) -> None:
    """
    Freeze the parameters of this module,
    i.e. do not update them during training

    :param module: freeze parameters of this module
    """
    for _, p in module.named_parameters():
        p.requires_grad = False

def calculate_dtw(references: List[Tensor], hypotheses: List[Tensor]) -> List[float]:
    """
    Calculate the DTW costs between a list of references and hypotheses.

    :param references: List of reference sequences
    :param hypotheses: List of hypothesis sequences
    :return: List of DTW costs
    """
    euclidean_norm = lambda x, y: np.sum(np.abs(x - y))
    dtw_scores = []

    for ref, hyp in zip(references, hypotheses):
        hyp = hyp[1:]  # Remove BOS frame
        # Cut the reference down to the max count value
        _, ref_max_idx = torch.max(ref[:, -1], dim=0)
        ref_max_idx = ref_max_idx.item() if ref_max_idx > 0 else 1
        # Cut down frames by to the max counter value, and chop off counter from joints
        ref_count = ref[:ref_max_idx, :-1].cpu().numpy()

        _, hyp_max_idx = torch.max(hyp[:, -1], 0)
        hyp_max_idx = hyp_max_idx.item() if hyp_max_idx > 0 else 1
        hyp_count = hyp[:hyp_max_idx, :-1].cpu().numpy()

        # Calculate DTW of the reference and hypothesis, using euclidean norm
        d, _, acc_cost_matrix, _ = dtw(ref_count, hyp_count, dist=euclidean_norm)
        # Normalise the dtw cost by sequence length
        d /= acc_cost_matrix.shape[0]
        dtw_scores.append(d)

    return dtw_scores