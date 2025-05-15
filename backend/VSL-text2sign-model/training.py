import argparse
import time
import shutil
import os
import queue
import numpy as np

import torch
from torch import Tensor
from torch.utils.tensorboard.writer import SummaryWriter
from torch.utils.data import Dataset

from model import build_model
from batch import Batch
from helpers import load_config, log_cfg, load_checkpoint, make_model_dir, \
    make_logger, set_seed, symlink_update, ConfigurationError, get_latest_checkpoint
from model import Model
from prediction import validate_on_data
from loss import RegLoss
from data import load_data, make_data_iter
from builders import build_optimizer, build_scheduler, \
    build_gradient_clipper
from constants import TARGET_PAD
from plot_videos import plot_video, alter_DTW_timing

class TrainManager:
    """Manages training process, including model training, validation, checkpointing, and logging."""
    
    def __init__(self, model: Model, config: dict, test=False) -> None:
        """
        Initialize the training manager with model, configuration, and optional test mode.
        
        :param model: The model to train.
        :param config: Configuration dictionary for training.
        :param test: If True, sets up for testing mode (forces model continuation).
        """
        train_config = config["training"]
        model_dir = train_config["model_dir"]
        
        # Determine if continuing from a checkpoint
        model_continue = train_config.get("continue", True)
        if not os.path.isdir(model_dir):
            model_continue = False
        if test:
            model_continue = True

        # Set up directories for logging and storing checkpoints
        self.model_dir = make_model_dir(train_config["model_dir"],
                                        overwrite=train_config.get("overwrite", False),
                                        model_continue=model_continue)
        
        # Initialize logger
        self.logger = make_logger(model_dir=self.model_dir)
        self.logging_freq = train_config.get("logging_freq", 100)
        
        # Set up validation report and TensorBoard writer
        self.valid_report_file = f"{self.model_dir}/validations.txt"
        self.tb_writer = SummaryWriter(log_dir=f"{self.model_dir}/tensorboard/")

        # Store model and related indices
        self.model = model
        self.pad_index = self.model.pad_index
        self.bos_index = self.model.bos_index
        self._log_parameters_list()  # Log model parameters
        self.target_pad = TARGET_PAD

        # Initialize regression loss based on config
        self.loss = RegLoss(cfg=config, target_pad=self.target_pad)

        # Set loss normalization mode
        self.normalization = "batch"

        # Optimization settings
        self.learning_rate_min = train_config.get("learning_rate_min", 1.0e-8)
        self.clip_grad_fun = build_gradient_clipper(config=train_config)
        self.optimizer = build_optimizer(config=train_config, parameters=model.parameters())

        # Validation and early stopping settings
        self.validation_freq = train_config.get("validation_freq", 1000)
        self.ckpt_best_queue = queue.Queue(maxsize=train_config.get("keep_last_ckpts", 1))
        self.ckpt_queue = queue.Queue(maxsize=1)
        self.val_on_train = config["data"].get("val_on_train", False)

        # Evaluation metric for validation
        self.eval_metric = train_config.get("eval_metric", "dtw").lower()
        if self.eval_metric not in ['dtw']:
            raise ConfigurationError("Invalid setting for 'eval_metric', valid options: 'dtw'")
        self.early_stopping_metric = train_config.get("early_stopping_metric", "eval_metric")

        # Determine whether to minimize or maximize the early stopping metric
        if self.early_stopping_metric in ["loss", "dtw"]:
            self.minimize_metric = True
        else:
            raise ConfigurationError("Invalid setting for 'early_stopping_metric', "
                                     "valid options: 'loss', 'dtw'.")

        # Learning rate scheduler setup
        self.scheduler, self.scheduler_step_at = build_scheduler(
            config=train_config,
            scheduler_mode="min" if self.minimize_metric else "max",
            optimizer=self.optimizer,
            hidden_size=config["model"]["encoder"]["hidden_size"])

        # Data and batch handling settings
        self.level = "word"
        self.shuffle = train_config.get("shuffle", True)
        self.epochs = train_config["epochs"]
        self.batch_size = train_config["batch_size"]
        self.eval_batch_size = train_config.get("eval_batch_size", self.batch_size)
        self.batch_multiplier = train_config.get("batch_multiplier", 1)

        # Generation settings
        self.max_output_length = train_config.get("max_output_length", None)

        # Device setup (CPU/GPU)
        self.use_cuda = train_config["use_cuda"]
        # For more than 1 GPU
        if self.use_cuda and torch.cuda.device_count() > 1:
            self.model = torch.nn.DataParallel(self.model)
        if self.use_cuda:
            self.model = self.model.to('cuda')
            self.loss = self.loss.to('cuda')

        # Initialize training statistics
        self.steps = 0
        self.stop = False
        self.total_tokens = 0
        self.best_ckpt_iteration = 0
        self.best_ckpt_score = np.inf if self.minimize_metric else -np.inf
        self.is_best = lambda score: score < self.best_ckpt_score \
            if self.minimize_metric else score > self.best_ckpt_score

        # Resume from checkpoint if continuing
        if model_continue:
            ckpt = get_latest_checkpoint(model_dir)
            if ckpt is None:
                self.logger.info("Can't find checkpoint in directory %s", ckpt)
            else:
                self.logger.info("Continuing model from %s", ckpt)
                self.init_from_checkpoint(ckpt)

        # Frame skipping for video data
        self.skip_frames = config["data"].get("skip_frames", 1)

        # Data augmentation settings
        self.just_count_in = config["model"].get("just_count_in", False)
        self.gaussian_noise = config["model"].get("gaussian_noise", False)
        if self.gaussian_noise:
            self.noise_rate = config["model"].get("noise_rate", 1.0)

        # Validate configuration
        if self.just_count_in and self.gaussian_noise:
            raise ConfigurationError("Can't have both just_count_in and gaussian_noise as True")

        # Future prediction setup
        self.future_prediction = config["model"].get("future_prediction", 0)
        if self.future_prediction != 0:
            frames_predicted = [i for i in range(self.future_prediction)]
            self.logger.info("Future prediction. Frames predicted: %s", frames_predicted)

    def _log_parameters_list(self) -> None:
        """
        Log all model parameters (names and shapes) to the logger.
        """
        model_parameters = filter(lambda p: p.requires_grad, self.model.parameters())
        n_params = sum([np.prod(p.size()) for p in model_parameters])
        self.logger.info("Total params: %d", n_params)
        trainable_params = [n for (n, p) in self.model.named_parameters() if p.requires_grad]
        self.logger.info("Trainable parameters: %s", sorted(trainable_params))
        assert trainable_params

    def _save_checkpoint(self, type="every") -> None:
        """
        Save a model checkpoint.
        
        :param type: Type of checkpoint ('best' or 'every').
        """
        model_path = f"{self.model_dir}/{self.steps}_{type}.ckpt"
        
        # Use module.state_dict() if model is wrapped by DataParallel
        model_state = self.model.module.state_dict() if hasattr(self.model, 'module') else self.model.state_dict()
        
        state = {
            "steps": self.steps,
            "total_tokens": self.total_tokens,
            "best_ckpt_score": self.best_ckpt_score,
            "best_ckpt_iteration": self.best_ckpt_iteration,
            "model_state": model_state,
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict() if self.scheduler is not None else None,
        }
        
        # Save checkpoint
        try:
            torch.save(state, model_path)
            self.logger.info(f"Saved checkpoint: {model_path}")
        except Exception as e:
            self.logger.error(f"Failed to save checkpoint {model_path}: {str(e)}")
            return

        # Handle best checkpoint queue
        if type == "best":
            if self.ckpt_best_queue.full():
                to_delete = self.ckpt_best_queue.get()
                try:
                    os.remove(to_delete)
                    self.logger.info(f"Deleted old checkpoint: {to_delete}")
                except FileNotFoundError:
                    self.logger.warning(f"Wanted to delete old checkpoint {to_delete} but file does not exist.")
            self.ckpt_best_queue.put(model_path)
            
            best_path = f"{self.model_dir}/best.ckpt"
            try:
                # Create or update symbolic link for best checkpoint
                symlink_update(f"{self.steps}_best.ckpt", best_path)
                self.logger.info(f"Updated best checkpoint link: {best_path}")
            except OSError as e:
                self.logger.warning(f"Failed to update symlink for {best_path}: {str(e)}. Saving directly.")
                torch.save(state, best_path)

        # Handle every checkpoint queue
        elif type == "every":
            if self.ckpt_queue.full():
                to_delete = self.ckpt_queue.get()
                try:
                    os.remove(to_delete)
                    self.logger.info(f"Deleted old checkpoint: {to_delete}")
                except FileNotFoundError:
                    self.logger.warning(f"Wanted to delete old checkpoint {to_delete} but file does not exist.")
            self.ckpt_queue.put(model_path)
            
            every_path = f"{self.model_dir}/every.ckpt"
            try:
                # Create or update symbolic link for every checkpoint
                symlink_update(f"{self.steps}_every.ckpt", every_path)  # Fixed symlink name
                self.logger.info(f"Updated every checkpoint link: {every_path}")
            except OSError as e:
                self.logger.warning(f"Failed to update symlink for {every_path}: {str(e)}. Saving directly.")
                torch.save(state, every_path)

    def init_from_checkpoint(self, path: str) -> None:
        """
        Initialize model, optimizer, and scheduler from a checkpoint.
        
        :param path: Path to the checkpoint file.
        """
        model_checkpoint = load_checkpoint(path=path, use_cuda=self.use_cuda)
        state_dict = model_checkpoint["model_state"]
        try:
            if hasattr(self.model, 'module'):
                self.model.module.load_state_dict(state_dict)
                self.logger.info("Loaded model state into DataParallel-wrapped model.")
            else:
                self.model.load_state_dict(state_dict)
                self.logger.info("Loaded model state into non-wrapped model.")
        except Exception as e:
            self.logger.error(f"Failed to load model state dict: {e}")
        self.optimizer.load_state_dict(model_checkpoint["optimizer_state"])
        if model_checkpoint["scheduler_state"] is not None and self.scheduler is not None:
            self.scheduler.load_state_dict(model_checkpoint["scheduler_state"])
        self.steps = model_checkpoint["steps"]
        self.total_tokens = model_checkpoint["total_tokens"]
        self.best_ckpt_score = model_checkpoint["best_ckpt_score"]
        self.best_ckpt_iteration = model_checkpoint["best_ckpt_iteration"]
        if self.use_cuda:
            self.model.cuda()

    def train_and_validate(self, train_data: Dataset, valid_data: Dataset) -> None:
        """
        Train the model and validate on the validation set periodically.
        
        :param train_data: Training dataset.
        :param valid_data: Validation dataset.
        """
        train_iter = make_data_iter(
            dataset=train_data,
            batch_size=self.batch_size,
            shuffle=self.shuffle
        )
        val_step = 0
        if self.gaussian_noise:
            all_epoch_noise = []
        
        # Loop through epochs
        for epoch_no in range(self.epochs):
            self.logger.info("EPOCH %d", epoch_no + 1)
            
            # Update scheduler at epoch level if specified
            if self.scheduler is not None and self.scheduler_step_at == "epoch":
                self.scheduler.step(epoch=epoch_no)
            
            self.model.train()
            start = time.time()
            total_valid_duration = 0
            start_tokens = self.total_tokens
            count = self.batch_multiplier - 1
            epoch_loss = 0
            
            # Handle Gaussian noise for data augmentation
            if self.gaussian_noise:
                if len(all_epoch_noise) != 0:
                    # Xử lý out_stds cho mô hình trong DataParallel
                    if hasattr(self.model, 'module'):
                        self.model.module.out_stds = torch.mean(torch.stack(([noise.std(dim=[0]) for noise in all_epoch_noise])), dim=-2)
                    else:
                        self.model.out_stds = torch.mean(torch.stack(([noise.std(dim=[0]) for noise in all_epoch_noise])), dim=-2)
                else:
                    if hasattr(self.model, 'module'):
                        self.model.module.out_stds = None
                    else:
                        self.model.out_stds = None
                all_epoch_noise = []
            
            # Iterate over training batches
            for batch in iter(train_iter):
                self.model.train()
                batch = Batch(
                    batch=batch,
                    src_vocab=train_data.src_vocab,
                    trg_size=train_data.trg_size,
                    model=self.model,
                    device = torch.device('cuda' if self.use_cuda else 'cpu')
                )
                update = count == 0
                batch_loss, noise = self._train_batch(batch, update=update)
                
                # Collect noise for Gaussian noise augmentation
                if self.gaussian_noise:
                    if self.future_prediction != 0:
                        # Truy cập out_trg_size qua model.module nếu là DataParallel
                        if hasattr(self.model, 'module'):
                            out_trg_size = self.model.module.out_trg_size
                        else:
                            out_trg_size = self.model.out_trg_size
                        all_epoch_noise.append(noise.reshape(-1, out_trg_size // self.future_prediction))
                    else:
                        if hasattr(self.model, 'module'):
                            out_trg_size = self.model.module.out_trg_size
                        else:
                            out_trg_size = self.model.out_trg_size
                        all_epoch_noise.append(noise.reshape(-1, out_trg_size))
                
                # Log batch loss to TensorBoard
                self.tb_writer.add_scalar("train/train_batch_loss", batch_loss, self.steps)
                count = self.batch_multiplier if update else count
                count -= 1
                epoch_loss += batch_loss.detach().cpu().numpy()
                
                # Update scheduler at step level if specified
                if self.scheduler is not None and self.scheduler_step_at == "step" and update:
                    self.scheduler.step()
                
                # Log training progress
                if self.steps % self.logging_freq == 0 and update:
                    elapsed = time.time() - start - total_valid_duration
                    elapsed_tokens = self.total_tokens - start_tokens
                    self.logger.info(
                        "Epoch %3d Step: %8d Batch Loss: %12.6f Tokens per Sec: %8.0f, Lr: %.6f",
                        epoch_no + 1, self.steps, batch_loss, elapsed_tokens / elapsed,
                        self.optimizer.param_groups[0]["lr"])
                    start = time.time()
                    total_valid_duration = 0
                    start_tokens = self.total_tokens
                
                # Validate periodically
                if self.steps % self.validation_freq == 0 and update:
                    valid_start_time = time.time()
                    valid_score, valid_loss, valid_references, valid_hypotheses, \
                        valid_inputs, all_dtw_scores, valid_file_paths = \
                        validate_on_data(
                            batch_size=self.eval_batch_size,
                            data=valid_data,
                            eval_metric=self.eval_metric,
                            model=self.model,
                            max_output_length=self.max_output_length,
                            loss_function=self.loss,
                            type="val",
                        )
                    val_step += 1
                    
                    # Log validation metrics to TensorBoard
                    self.tb_writer.add_scalar("valid/valid_loss", valid_loss, self.steps)
                    self.tb_writer.add_scalar("valid/valid_score", valid_score, self.steps)
                    
                    # Determine checkpoint score for early stopping
                    if self.early_stopping_metric == "loss":
                        ckpt_score = valid_loss
                    elif self.early_stopping_metric == "dtw":
                        ckpt_score = valid_score
                    else:
                        ckpt_score = valid_score
                    
                    new_best = False
                    self.best = False
                    if self.is_best(ckpt_score):
                        self.best = True
                        self.best_ckpt_score = ckpt_score
                        self.best_ckpt_iteration = self.steps
                        self.logger.info('Hooray! New best validation result [%s]!', self.early_stopping_metric)
                        if self.ckpt_queue.maxsize > 0:
                            self.logger.info("Saving new checkpoint.")
                            new_best = True
                            self._save_checkpoint(type="best")
                        
                        # Select sequences for video generation
                        display = list(range(0, len(valid_hypotheses), int(np.ceil(len(valid_hypotheses) / 13.15))))
                        self.produce_validation_video(
                            output_joints=valid_hypotheses,
                            references=valid_references,
                            model_dir=self.model_dir,
                            steps=self.steps,
                            display=display,
                            type="val_inf",
                            file_paths=valid_file_paths,
                        )
                    
                    # Save checkpoint for every validation
                    self._save_checkpoint(type="every")
                    
                    # Update scheduler based on validation score
                    if self.scheduler is not None and self.scheduler_step_at == "validation":
                        self.scheduler.step(ckpt_score)
                    
                    # Append validation results to report
                    self._add_report(
                        valid_score=valid_score, valid_loss=valid_loss,
                        eval_metric=self.eval_metric,
                        new_best=new_best, report_type="val")
                    
                    valid_duration = time.time() - valid_start_time
                    total_valid_duration += valid_duration
                    self.logger.info(
                        'Validation result at epoch %3d, step %8d: Val DTW Score: %6.2f, '
                        'loss: %8.4f, duration: %.4fs',
                        epoch_no + 1, self.steps, valid_score, valid_loss, valid_duration)
                
                if self.stop:
                    break
            if self.stop:
                self.logger.info('Training ended since minimum lr %f was reached.', self.learning_rate_min)
                break
            
            self.logger.info('Epoch %3d: total training loss %.5f', epoch_no + 1, epoch_loss)
        else:
            self.logger.info('Training ended after %3d epochs.', epoch_no + 1)
        self.logger.info('Best validation result at step %8d: %6.2f %s.',
                         self.best_ckpt_iteration, self.best_ckpt_score, self.early_stopping_metric)
        self.tb_writer.close()

    def produce_validation_video(self, output_joints, references, display, model_dir, type, steps=None, file_paths=None):
        """
        Generate videos for validation or test sequences.
        
        :param output_joints: Predicted joint sequences.
        :param references: Reference joint sequences.
        :param display: Indices of sequences to visualize.
        :param model_dir: Directory to save videos.
        :param type: Type of run ('val_inf' or 'test').
        :param steps: Training step number (for naming).
        :param file_paths: File paths for sequence IDs.
        """
        # Set video output directory
        if type != "test":
            dir_name = f"{model_dir}/videos/Step_{steps}/"
            os.makedirs(f"{model_dir}/videos/", exist_ok=True)
        else:
            dir_name = f"{model_dir}/test_videos/"
        os.makedirs(dir_name, exist_ok=True)

        # Generate videos for selected sequences
        for i in display:
            seq = output_joints[i]
            ref_seq = references[i]
        
            # Adjust timing using DTW and compute score
            timing_hyp_seq, ref_seq_count, dtw_score = alter_DTW_timing(seq, ref_seq)
            video_ext = f"{dtw_score:.2f}".replace(".", "_") + ".mp4"
            
            # Use sequence ID if provided
            sequence_ID = file_paths[i] if file_paths is not None else None
            
            # Generate video if filename is valid
            if "<" not in video_ext:
                plot_video(
                    joints=timing_hyp_seq,
                    file_path=dir_name,
                    video_name=video_ext,
                    references=ref_seq_count,
                    skip_frames=self.skip_frames,
                    sequence_ID=sequence_ID
                )

    def _train_batch(self, batch: Batch, update: bool = True) -> Tensor:
        """
        Train the model on a single batch.
        
        :param batch: Batch object containing input and target data.
        :param update: If True, update model parameters.
        :return: Normalized batch loss and noise (if applicable).
        """
        # Truy cập get_loss_for_batch bằng cách kiểm tra DataParallel
        if hasattr(self.model, 'module'):
            batch_loss, noise = self.model.module.get_loss_for_batch(batch=batch, loss_function=self.loss)
        else:
            batch_loss, noise = self.model.get_loss_for_batch(batch=batch, loss_function=self.loss)
        
        # Normalize loss
        if self.normalization == "batch":
            normalizer = batch.nseqs
        elif self.normalization == "tokens":
            normalizer = batch.ntokens
        else:
            raise NotImplementedError("Only normalize by 'batch' or 'tokens'")
        norm_batch_loss = batch_loss / normalizer
        norm_batch_multiply = norm_batch_loss / self.batch_multiplier
        
        # Backpropagate gradients
        norm_batch_multiply.backward()
        
        # Clip gradients if specified
        if self.clip_grad_fun is not None:
            self.clip_grad_fun(params=self.model.parameters())
        
        # Update model parameters
        if update:
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.steps += 1
        
        self.total_tokens += batch.ntokens
        return norm_batch_loss, noise

    def _add_report(self, valid_score: float, valid_loss: float, eval_metric: str,
                    new_best: bool = False, report_type: str = "val") -> None:
        """
        Append validation results to the report file.
        
        :param valid_score: Validation score (e.g., DTW).
        :param valid_loss: Validation loss.
        :param eval_metric: Evaluation metric used.
        :param new_best: If True, mark as new best result.
        :param report_type: Type of report ('val' or others).
        """
        current_lr = -1
        for param_group in self.optimizer.param_groups:
            current_lr = param_group['lr']
        if current_lr < self.learning_rate_min:
            self.stop = True
        if report_type == "val":
            with open(self.valid_report_file, 'a') as opened_file:
                opened_file.write(
                    f"Steps: {self.steps} Loss: {valid_loss:.5f}| DTW: {valid_score:.3f}| "
                    f"LR: {current_lr:.6f} {'*' if new_best else ''}\n"
                )

def train(cfg_file: str, ckpt=None) -> None:
    """
    Train a model using the specified configuration.
    
    :param cfg_file: Path to the configuration file (YAML).
    :param ckpt: Path to a checkpoint to resume from (optional).
    """
    cfg = load_config(cfg_file)
    set_seed(seed=cfg["training"].get("random_seed", 42))
    train_data, dev_data, test_data, src_vocab, trg_vocab = load_data(cfg=cfg)
    model = build_model(cfg, src_vocab=src_vocab, trg_vocab=trg_vocab)
    
    # Load checkpoint if provided
    if ckpt is not None:
        use_cuda = cfg["training"].get("use_cuda", False)
        model_checkpoint = load_checkpoint(ckpt, use_cuda=use_cuda)
        model.load_state_dict(model_checkpoint["model_state"])
    
    # Initialize training manager
    trainer = TrainManager(model=model, config=cfg)
    
    # Save config and log details
    shutil.copy2(cfg_file, trainer.model_dir + "/config.yaml")
    log_cfg(cfg, trainer.logger)
    
    # Start training and validation
    trainer.train_and_validate(train_data=train_data, valid_data=dev_data)
    
    # Run test after training
    test(cfg_file)

def test(cfg_file, ckpt: str = None) -> None:
    """
    Test a model using the specified configuration and checkpoint.
    
    :param cfg_file: Path to the configuration file (YAML).
    :param ckpt: Path to the checkpoint to test (optional, uses latest if None).
    """
    cfg = load_config(cfg_file)
    model_dir = cfg["training"]["model_dir"]
    
    # Load latest checkpoint if not specified
    if ckpt is None:
        ckpt = get_latest_checkpoint(model_dir, post_fix="_best")
        if ckpt is None:
            raise FileNotFoundError(f"No checkpoint found in directory {model_dir}.")
    
    batch_size = cfg["training"].get("eval_batch_size", cfg["training"]["batch_size"])
    use_cuda = cfg["training"].get("use_cuda", False)
    eval_metric = cfg["training"]["eval_metric"]
    max_output_length = cfg["training"].get("max_output_length", None)
    
    # Load data
    train_data, dev_data, test_data, src_vocab, trg_vocab = load_data(cfg=cfg)
    data_to_predict = {"test": test_data}
    
    # Load model and checkpoint
    model_checkpoint = load_checkpoint(ckpt, use_cuda=use_cuda)
    model = build_model(cfg, src_vocab=src_vocab, trg_vocab=trg_vocab)
    model.load_state_dict(model_checkpoint["model_state"])
    if use_cuda:
        model.cuda()
    
    # Initialize trainer for testing
    trainer = TrainManager(model=model, config=cfg, test=True)
    
    # Generate predictions and videos for test data
    for data_set_name, data_set in data_to_predict.items():
        score, loss, references, hypotheses, inputs, all_dtw_scores, file_paths = \
            validate_on_data(
                model=model,
                data=data_set,
                batch_size=batch_size,
                max_output_length=max_output_length,
                eval_metric=eval_metric,
                loss_function=None,
                type="val" if data_set_name != "train" else "train_inf"
            )
        print(hypotheses)
        display = list(range(len(hypotheses)))
        trainer.produce_validation_video(
            output_joints=hypotheses,
            references=references,
            model_dir=model_dir,
            display=display,
            type="test",
            file_paths=file_paths,
        )

if __name__ == "__main__":
    """Parse command-line arguments and start training."""
    parser = argparse.ArgumentParser("Progressive Transformers")
    parser.add_argument("config", default="Configs/Base.yaml", type=str,
                        help="Training configuration file (yaml).")
    args = parser.parse_args()
    train(cfg_file=args.config)