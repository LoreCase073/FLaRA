import torch
import numpy as np
import argparse
import os
import csv
import yaml

from torch.utils.tensorboard import SummaryWriter

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

def set_random_seed(seed: int) -> None:
    """
    Set random seeds for reproducible results across different libraries.

    Args:
        seed (int): Random seed value
    """
    import random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    # Required for deterministic CUBLAS operations
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # Enable PyTorch deterministic algorithms (warn_only=False raises errors for unsupported ops)
    torch.use_deterministic_algorithms(True, warn_only=False)



def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "1", "y"):
        return True
    if v.lower() in ("no", "false", "f", "0", "n"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def prepare_directories(args):
    """
    Create necessary output and logging directories for training.

    This function creates the base output and log directories, then creates
    experiment-specific subdirectories and adds them to the args namespace.

    Args:
        args (argparse.Namespace): Parsed arguments containing:
            - output_path: Base directory for model outputs
            - log_path: Base directory for logs
            - experiment_name: Name for this specific experiment

    Returns:
        argparse.Namespace: Updated args with additional directory paths:
            - output_dir: Full path to experiment output directory
            - log_dir: Full path to experiment log directory
    """
    # Create base directories
    os.makedirs(args.output_path, exist_ok=True)
    os.makedirs(args.log_path, exist_ok=True)

    # Create experiment-specific directories
    args.output_dir = os.path.join(args.output_path, args.experiment_name)
    args.log_dir = os.path.join(args.log_path, args.experiment_name)

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    # Print directory information for verification
    print(f"Output directory: {args.output_dir}")
    print(f"Log directory: {args.log_dir}")

    return args


def save_metrics_to_csv(filepath, epoch, metrics_dict):
    """
    Save metrics to a CSV file. Creates the file with headers if it doesn't exist,
    otherwise appends a new row.
    
    Args:
        filepath: Path to the CSV file
        epoch: Current epoch number
        metrics_dict: Dictionary containing all metrics to save
    """
    # Add epoch to metrics
    full_metrics = {'epoch': epoch}
    full_metrics.update(metrics_dict)
    
    # Check if file exists to determine if we need to write headers
    file_exists = os.path.isfile(filepath)
    
    with open(filepath, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=full_metrics.keys())
        
        # Write header if file is new
        if not file_exists:
            writer.writeheader()
        
        # Write the metrics row
        writer.writerow(full_metrics)



def parse_args():
    """
    Parse command line arguments for V-JEPA2 training.

    Returns:
        argparse.Namespace: Parsed arguments containing all configuration parameters
    """
    parser = argparse.ArgumentParser(
        description="Train V-JEPA2 model for video classification",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # =====================================
    # Model Configuration
    # =====================================
    model_group = parser.add_argument_group('Model Parameters')
    model_group.add_argument(
        "--model_name",
        type=str,
        default="facebook/vjepa2-vitl-fpc16-256-ssv2",
        help="Pretrained model name from Hugging Face (e.g., 'facebook/vjepa2-vitl-fpc16-256-ssv2' for VJEPA2)"
    )
    model_group.add_argument(
        "--num_classes",
        type=int,
        default=2,
        help="Number of classes for classification task"
    )
    model_group.add_argument(
        "--trainable_parameters_configuration",
        type=str,
        default="last_block+predictor+pool+head",
        help="Which model parameters to fine-tune: 'pooler+head', 'last_block+pool+head', 'last_block+predictor+pool+head'"
    )
    model_group.add_argument(
        "--pooling_mode",
        type=str,
        default="attentive",
        help="Pooling method for classifier (VJEPA2): 'attentive', 'mean'"
    )
    model_group.add_argument(
        "--predict_future_temporal_steps",
        type=int,
        default=8,
        help="(VJEPA2) Number of future temporal positions to predict with the predictor. "
             "Each step covers tubelet_size frames (typically 2). E.g., for 1 second at 16fps "
             "with tubelet_size=2, use 8. Default 8."
    )
    model_group.add_argument(
        "--classify_on_predicted_only",
        type=str2bool,
        default=True,
        help="(VJEPA2) If True, the pooler and classifier receive only the predictor's output "
             "(predicted future tokens), not the encoder context features. "
             "Requires predict_future_temporal_steps > 0 or prediction_future_frames > 0."
    )

    # =====================================
    # Data Configuration
    # =====================================
    data_group = parser.add_argument_group('Data Parameters')
    data_group.add_argument(
        "--dataset_name",
        type=str,
        default="nexar",
        help="Name of the dataset. Only 'nexar' is supported for training."
    )
    data_group.add_argument(
        "--train_csv",
        type=str,
        default="",
        help="Path to training CSV file"
    )
    data_group.add_argument(
        "--val_csv",
        type=str,
        default="",
        help="Path to validation CSV file"
    )
    data_group.add_argument(
         "--test_csv",
         type=str,
         default="",
         help="Path to test CSV file"
    )
    data_group.add_argument(
        "--data_root",
        type=str,
        default="data/video",
        help="Root directory containing video files"
    )
    data_group.add_argument(
        "--label_mapping_path",
        type=str,
        default="dataset/nexar_label_mapping.json",
        help="Path to JSON file containing label mappings"
    )
    # =====================================
    # Video Processing Configuration
    # =====================================
    video_group = parser.add_argument_group('Video Processing Parameters')
    video_group.add_argument(
        "--anticipation_offset_min",
        type=float,
        default=0.0,
        help="Minimum anticipation offset for 'anticipation' frame selection method (in seconds, range [0, 1])"
    )
    video_group.add_argument(
        "--anticipation_offset_max",
        type=float,
        default=1.0,
        help="Maximum anticipation offset for 'anticipation' frame selection method (in seconds, range [0, 1])"
    )
    video_group.add_argument(
        "--use_time_of_alert_offset",
        type=str2bool,
        default=False,
        help="Use per-sample (time_of_event - time_of_alert) from CSV as anticipation offset max. "
             "Offset range becomes [anticipation_offset_min, max(time_of_event - time_of_alert, anticipation_offset_max)]. "
             "Only applies to positive (crash) samples; negative samples use the fixed anticipation_offset_range."
    )
    video_group.add_argument(
        "--num_frames",
        type=int,
        default=16,
        help="Number of frames to extract per video sample"
    )
    video_group.add_argument(
        "--fps",
        type=float,
        default=8.0,
        help="Target sampling FPS for video frames (default: 8.0)"
    )
    video_group.add_argument(
        "--frame_size",
        type=int,
        default=256,
        help="Target size for frame height and width (square resize)"
    )
    video_group.add_argument(
        "--duration",
        type=float,
        default=2.0,
        help="Duration of video segment to extract in seconds (default: 2.0)"
    )
    video_group.add_argument(
        "--use_fixed_fps",
        type=str2bool,
        default=True,
        help="If True, use a fixed predefined FPS value (DEFAULT_FPS) instead of extracting FPS from each video"
    )

    # =====================================
    # Training Configuration
    # =====================================
    training_group = parser.add_argument_group('Training Parameters')
    training_group.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size for training and validation"
    )
    training_group.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of gradient accumulation steps. Effective batch size = batch_size * gradient_accumulation_steps"
    )
    training_group.add_argument(
        "--epochs",
        type=int,
        default=30,
        help="Total number of training epochs"
    )
    training_group.add_argument(
        "--num_workers",
        type=int,
        default=-1,
        help="Number of DataLoader worker processes"
    )

    # =====================================
    # Optimization Configuration
    # =====================================
    optim_group = parser.add_argument_group('Optimization Parameters')
    optim_group.add_argument(
        "--learning_rate",
        type=float,
        default=2.5e-5,
        help="Initial learning rate for optimizer, general parameters"
    )
    optim_group.add_argument(
        "--weight_decay",
        type=float,
        default=0.01,
        help="Weight decay (L2 regularization) for optimizer"
    )
    optim_group.add_argument(
        "--lr_scheduler",
        type=str,
        default="cosine",
        choices=["cosine", "constant"],
        help="Learning rate scheduler type: 'cosine' for warmup + cosine annealing, 'constant' for fixed learning rate"
    )


    # =====================================
    # Loss Function Configuration
    # =====================================
    loss_group = parser.add_argument_group('Loss Function Parameters')
    loss_group.add_argument(
        "--lambda_cls",
        type=float,
        default=1.0,
        help="Weight coefficient for classification loss component"
    )
    loss_group.add_argument(
        "--lambda_prediction",
        type=float,
        default=10.0,
        help="Weight for auxiliary prediction loss (predictor regularization). "
             "Masks random temporal positions and trains the predictor to reconstruct them."
    )
    loss_group.add_argument(
        "--prediction_future_frames",
        type=int,
        default=16,
        help="Number of actual future frames to load per sample for prediction target reconstruction. "
             "The dataset loads num_frames + prediction_future_frames total; the model splits them "
             "internally, classifying from context only (default: 16)."
    )

    # =====================================
    # System Configuration
    # =====================================
    system_group = parser.add_argument_group('System Parameters')
    system_group.add_argument(
        "--gpu_id",
        type=str,
        default="0",
        help="GPU device ID to use for training (e.g., '0', '1', 'cpu')"
    )
    system_group.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible results"
    )

    # =====================================
    # Output Configuration
    # =====================================
    output_group = parser.add_argument_group('Output Parameters')
    output_group.add_argument(
        "--experiment_name",
        type=str,
        default="exp1",
        help="Name identifier for this experiment run"
    )
    output_group.add_argument(
        "--output_path",
        type=str,
        default="output",
        help="Base directory for saving trained models and checkpoints"
    )
    output_group.add_argument(
        "--log_path",
        type=str,
        default="logs",
        help="Base directory for saving TensorBoard logs and metrics"
    )
    output_group.add_argument(
        "--debug",
        type=str2bool,
        default=False,
        help="If true, disable wandb logging",
    )

    args = parser.parse_args()
    return args



class UnifiedLogger:
    """
    Wrapper class that logs to both TensorBoard and Weights & Biases.
    Automatically initializes wandb if config is provided.
    """
    def __init__(self, log_dir, wandb_config=None, experiment_name=None, 
                 experiment_config=None, output_dir=None, debug=False):
        """
        Args:
            log_dir: Directory for TensorBoard logs
            wandb_config: Dictionary with wandb configuration (project, entity, etc.)
                         If None or empty, wandb will not be initialized
            experiment_name: Name for the experiment (used in wandb run name)
            experiment_config: Dictionary of experiment hyperparameters to log
            output_dir: Output directory for wandb files
            debug: If True, disable wandb logging
        """
        # Initialize TensorBoard writer
        self.writer = SummaryWriter(log_dir=log_dir)
        self.log_dir = log_dir
        
        # Initialize wandb if config is provided and wandb is available
        self.use_wandb = False
        if WANDB_AVAILABLE and wandb_config and wandb_config != {}:
            try:
                # Patch tensorboard to sync with wandb
                # wandb.tensorboard.patch(root_logdir=log_dir)
                
                # Initialize wandb
                if debug:
                    print("Debug mode enabled, skipping wandb initialization.")
                    self.use_wandb = False
                else:
                    wandb.init(
                        project=wandb_config.get('project', 'flara'),
                        entity=wandb_config.get('entity', None),
                        name=experiment_name,
                        config=experiment_config or {},
                        dir=output_dir or log_dir,
                    )
                    self.use_wandb = True
                    print("Initialized Weights & Biases logging.")
            except Exception as e:
                print(f"Warning: Failed to initialize wandb: {e}. Continuing with TensorBoard only.")
                self.use_wandb = False
        else:
            if not WANDB_AVAILABLE:
                print("wandb not available. Using TensorBoard only.")
            elif not wandb_config or wandb_config == {}:
                print("No wandb_config provided, do not initialize wandb.")
        
    def add_scalar(self, tag, scalar_value, global_step=None, walltime=None):
        """Log scalar value to both TensorBoard and wandb."""
        # Log to TensorBoard
        self.writer.add_scalar(tag, scalar_value, global_step, walltime)
        
        # Log to wandb if available
        if self.use_wandb:
            wandb_key = tag
            wandb.log({wandb_key: scalar_value}, step=global_step)
    
    def add_scalars(self, main_tag, tag_scalar_dict, global_step=None, walltime=None):
        """Log multiple scalar values to both TensorBoard and wandb."""
        # Log to TensorBoard
        self.writer.add_scalars(main_tag, tag_scalar_dict, global_step, walltime)
        
        # Log to wandb if available
        if self.use_wandb:
            wandb_dict = {}
            for tag, value in tag_scalar_dict.items():
                wandb_key = f"{main_tag}/{tag}"
                wandb_dict[wandb_key] = value
            wandb.log(wandb_dict, step=global_step)
    
    def add_text(self, tag, text_string, global_step=None, walltime=None):
        """Log text to TensorBoard (wandb doesn't support text logging the same way)."""
        self.writer.add_text(tag, text_string, global_step, walltime)
        
        # Optionally log to wandb as config or note
        if self.use_wandb and global_step == 0:
            # Log initial text (like config) to wandb
            wandb.config.update({tag: text_string}, allow_val_change=True)
    
    def add_hparams(self, hparam_dict, metric_dict=None, hparam_domain_discrete=None, 
                     run_name=None, global_step=None):
        """Log hyperparameters to TensorBoard and wandb config."""
        # Log to TensorBoard
        self.writer.add_hparams(hparam_dict, metric_dict or {}, 
                                hparam_domain_discrete, run_name, global_step)
        
        # Log to wandb config if available
        if self.use_wandb:
            wandb.config.update(hparam_dict, allow_val_change=True)
    
    def add_image(self, tag, img_tensor, global_step=None, walltime=None, dataformats='CHW'):
        """Log image to both TensorBoard and wandb."""
        # Log to TensorBoard
        self.writer.add_image(tag, img_tensor, global_step, walltime, dataformats)
        
        # Log to wandb if available
        if self.use_wandb:
            import numpy as np
            if isinstance(img_tensor, torch.Tensor):
                img_array = img_tensor.cpu().numpy()
            else:
                img_array = np.array(img_tensor)
            
            # Convert to HWC format for wandb if needed
            if dataformats == 'CHW':
                img_array = np.transpose(img_array, (1, 2, 0))
            
            wandb.log({tag: wandb.Image(img_array)}, step=global_step)
    
    def add_histogram(self, tag, values, global_step=None, bins='tensorflow', 
                       walltime=None, max_bins=None):
        """Log histogram to both TensorBoard and wandb."""
        # Log to TensorBoard
        self.writer.add_histogram(tag, values, global_step, bins, walltime, max_bins)
        
        # Log to wandb if available
        if self.use_wandb:
            import numpy as np
            if isinstance(values, torch.Tensor):
                values_array = values.cpu().numpy()
            else:
                values_array = np.array(values)
            
            wandb.log({tag: wandb.Histogram(values_array)}, step=global_step)
    
    def close(self):
        """Close both TensorBoard writer and wandb run."""
        self.writer.close()
        if self.use_wandb:
            wandb.finish()
    
    def flush(self):
        """Flush TensorBoard writer."""
        self.writer.flush()


# Helper function to load wandb config
def load_wandb_config(config_path='wandb_config.yaml'):
    """Load wandb configuration from file if it exists."""
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    return {}