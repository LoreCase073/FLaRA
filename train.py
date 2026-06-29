import os
import multiprocessing
import json

import torch

import time

from utils.utils import UnifiedLogger, set_random_seed, parse_args, prepare_directories, load_wandb_config, save_metrics_to_csv
from utils.train_utils import (
    ModelManager, DatasetManager, TrainingEvaluator, ExperimentManager
)


def setup_experiment(args) -> tuple:
    """
    Setup the experiment environment including device, logging, and directories.

    Args:
        args: Training arguments

    Returns:
        Tuple of (device, writer)
    """
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")
    print(f"Number of workers: {args.num_workers}")

    # Set random seed for reproducibility
    set_random_seed(args.seed)
    print(f"Random seed set to: {args.seed}")

    # Setup logging and wandb
    wandb_config = load_wandb_config()
    writer = UnifiedLogger(
        log_dir=args.log_dir,
        wandb_config=wandb_config,
        experiment_name=args.experiment_name,
        experiment_config=vars(args),
        output_dir=args.output_dir,
        debug=args.debug
    )
    print("Initialized unified logging.")

    return device, writer


def setup_hyperparameters_logging(args, writer) -> None:
    """
    Setup hyperparameter logging and save configuration.

    Args:
        args: Training arguments
        writer: Unified logger
    """
    hparams = vars(args)

    # Save complete configuration as JSON
    with open(os.path.join(args.output_dir, 'config.json'), 'w') as f:
        json.dump(hparams, f, indent=4)

    # Log configuration to tensorboard/wandb
    args_text = "```\n" + "\n".join([f"{k}: {v}" for k, v in hparams.items()]) + "\n```"
    writer.add_text("Config/Arguments", args_text, 0)

    # Log numeric hyperparameters for comparison
    hparams_filtered = {k: v for k, v in hparams.items()
                       if isinstance(v, (int, float, bool))}
    writer.add_hparams(hparams_filtered, {})


def setup_training_components(args, device) -> tuple:
    model = ModelManager.create_model(args)
    model.to(device)

    train_dataset, val_dataset = DatasetManager.create_datasets(args)
    train_loader, val_loader = DatasetManager.create_data_loaders(
        train_dataset, val_dataset, args.batch_size,
        args.num_workers, seed=args.seed
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    criterion_cls = torch.nn.CrossEntropyLoss()

    scheduler = ExperimentManager.setup_scheduler(
        optimizer, args.lr_scheduler, args.learning_rate, args.epochs
    )

    return model, train_loader, val_loader, optimizer, criterion_cls, scheduler


def run_training_loop(args, model, train_loader, val_loader, optimizer, criterion_cls,
                     scheduler, device, writer) -> str:
    """
    Run the main training loop.

    Args:
        args: Training arguments
        model: Model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        optimizer: Optimizer
        criterion_cls: Classification loss function
        scheduler: Learning rate scheduler
        device: Training device
        writer: Unified logger

    Returns:
        Path to best model
    """
    # Initialize training evaluator
    evaluator = TrainingEvaluator(
        model, device, criterion_cls,
        args.lambda_cls,
        args.lambda_prediction,
        args.gradient_accumulation_steps,
        args.num_frames
    )

    # Training state
    best_val_ap = 0.0
    best_model_path = os.path.join(args.output_dir, "best_model.pth")
    best_validation_metrics_csv = os.path.join(args.output_dir, 'best_validation_metrics.csv')


    # Start global training timer
    training_start_time = time.time()

    # Training loop
    for epoch in range(args.epochs):
        print(f"Starting epoch {epoch+1}/{args.epochs}")

        # Start epoch timer
        epoch_start_time = time.time()

        # Training step
        train_metrics = evaluator.train_epoch(train_loader, optimizer)

        # Update learning rate
        if scheduler is not None:
            scheduler.step()
            writer.add_scalar('Train/Learning_Rate', scheduler.get_last_lr()[0], epoch)

        # Log training metrics
        writer.add_scalar('Train/Epoch_Loss', train_metrics['loss'], epoch)

        writer.add_scalar('Train/Epoch_Cls_Loss', train_metrics['cls_loss'], epoch)
        if args.lambda_prediction > 0:
            writer.add_scalar('Train/Epoch_Prediction_Loss', train_metrics['prediction_loss'], epoch)

        # Print epoch summary
        loss_parts = [f"Loss: {train_metrics['loss']:.4f}",
                      f"Cls: {train_metrics['cls_loss']:.4f}"]
        if args.lambda_prediction > 0:
            loss_parts.append(f"Prediction: {train_metrics['prediction_loss']:.4f}")
        print(f"Epoch [{epoch+1}/{args.epochs}], " + ", ".join(loss_parts))

        # Validation step
        print("Starting validation")
        val_metrics, _, _ = evaluator.evaluate(
            val_loader, args.num_classes
        )

        # Log validation metrics
        writer.add_scalar('Val/Loss', val_metrics['loss'], epoch)

        # Prepare epoch metrics for CSV logging
        epoch_metrics = {
            'train_loss': train_metrics['loss'],
            'val_loss': val_metrics['loss'],
            'train_cls_loss': train_metrics['cls_loss'],
        }

        # Log classification metrics
        writer.add_scalar('Val/Classification_Accuracy', val_metrics['cls_accuracy'], epoch)

        if args.num_classes == 2:
            writer.add_scalar('Val/Classification_AP', val_metrics['cls_ap'], epoch)
            epoch_metrics['val_cls_ap'] = val_metrics['cls_ap']
            for i in range(args.num_classes):
                if f'cls_ap_class_{i}' in val_metrics:
                    writer.add_scalar(f'Val/Class_{i}_AP', val_metrics[f'cls_ap_class_{i}'], epoch)
                    epoch_metrics[f'val_cls_ap_class_{i}'] = val_metrics[f'cls_ap_class_{i}']
        else:
            writer.add_scalar('Val/Classification_AP', val_metrics['cls_ap_macro'], epoch)
            epoch_metrics['val_cls_ap_macro'] = val_metrics['cls_ap_macro']
            for i in range(args.num_classes):
                if f'cls_ap_class_{i}' in val_metrics:
                    writer.add_scalar(f'Val/Class_{i}_AP', val_metrics[f'cls_ap_class_{i}'], epoch)
                    epoch_metrics[f'val_cls_ap_class_{i}'] = val_metrics[f'cls_ap_class_{i}']

        epoch_metrics['val_cls_accuracy'] = val_metrics['cls_accuracy']

        # Print validation results
        print(f"Validation Loss: {val_metrics['loss']:.4f}")
        if model.classifier is not None:
            main_ap_key = 'cls_ap' if args.num_classes == 2 else 'cls_ap_macro'
            print(f"Validation Classification AP: {val_metrics[main_ap_key]:.4f}")

        # Check for best model
        is_best = False
        if model.classifier is not None:
            main_ap_key = 'cls_ap' if args.num_classes == 2 else 'cls_ap_macro'
            if val_metrics[main_ap_key] > best_val_ap:
                best_val_ap = val_metrics[main_ap_key]
                is_best = True
                print(f"New best model (AP): {best_val_ap:.4f}")

        if is_best:
            ModelManager.save_model(model, best_model_path)
            save_metrics_to_csv(best_validation_metrics_csv, epoch + 1, epoch_metrics)
            print(f"Best model (AP) saved to {best_model_path}")


        # End of epoch timing
        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        total_training_time = epoch_end_time - training_start_time

        avg_epoch_time = total_training_time / (epoch + 1)
        remaining_epochs = args.epochs - (epoch + 1)
        eta_seconds = avg_epoch_time * remaining_epochs

        print(f"Epoch {epoch+1} duration: {epoch_duration:.2f} seconds")
        print(f"Total training time so far: {total_training_time/60:.2f} minutes")
        print(f"ETA for completion: {eta_seconds/60:.2f} minutes")

        print(f"Epoch {epoch+1} completed.\n")

    return best_model_path


def main():
    """Main training function with clean separation of concerns."""
    # Parse arguments and prepare directories
    args = parse_args()
    args = prepare_directories(args)

    # Resolve num_workers once: -1 means use all available CPUs
    if args.num_workers < -1:
        raise ValueError(f"--num_workers must be >= -1, got {args.num_workers}")
    if args.num_workers == -1:
        args.num_workers = multiprocessing.cpu_count()

    # Setup experiment environment
    device, writer = setup_experiment(args)

    # Setup hyperparameter logging
    setup_hyperparameters_logging(args, writer)

    # Setup training components
    model, train_loader, val_loader, optimizer, criterion_cls, scheduler = setup_training_components(args, device)

    # Run training loop
    run_training_loop(
        args, model, train_loader, val_loader, optimizer, criterion_cls,
        scheduler, device, writer
    )

    # Save final model
    final_model_path = os.path.join(args.output_dir, "last_epoch_model.pth")
    ModelManager.save_model(model, final_model_path)

    # Cleanup
    writer.close()
    print("Training finished.")
    print(f"To view TensorBoard logs, run: tensorboard --logdir={args.log_dir}")


if __name__ == "__main__":
    main()