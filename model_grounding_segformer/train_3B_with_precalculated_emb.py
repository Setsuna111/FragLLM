"""
Training script for ProteinSAM model with pre-computed ESM embeddings.
Single GPU training for protein functional region grounding.

This script supports:
1. Pre-computed ESM embeddings to reduce training time computation
2. TensorBoard logging for loss and metrics tracking
3. External ESM embeddings without BOS/EOS tokens
"""
import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from tqdm import tqdm
import json
from typing import Dict, Any, Optional
from datetime import datetime
import logging
from torch.utils.tensorboard import SummaryWriter

from protein_sam import ProteinSAM
from dataset import get_datasets_and_collator, get_datasets_and_collator_with_esm_cache


def set_seed(seed: int = 42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_logging(log_dir: str, log_level: str = "INFO") -> logging.Logger:
    """Setup logging configuration."""
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"training_{timestamp}.log")
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger(__name__)
    return logger


def calculate_mask_accuracy(mask_preds: torch.Tensor, mask_labels: torch.Tensor, attention_mask: torch.Tensor = None) -> float:
    """Calculate binary mask accuracy."""
    if attention_mask is not None:
        mask = attention_mask.bool()
        mask_preds = mask_preds[mask]
        mask_labels = mask_labels[mask]
    
    correct = (mask_preds == mask_labels).float()
    return correct.mean().item()


def calculate_mask_iou(mask_preds: torch.Tensor, mask_labels: torch.Tensor, attention_mask: torch.Tensor = None) -> float:
    """Calculate IoU for binary masks."""
    batch_size = mask_preds.shape[0]
    total_iou = 0.0
    
    for b in range(batch_size):
        pred = mask_preds[b]
        true = mask_labels[b]
        
        if attention_mask is not None:
            mask = attention_mask[b].bool()
            pred = pred[mask]
            true = true[mask]
        
        # Calculate IoU
        intersection = (pred & true).float().sum()
        union = (pred | true).float().sum()
        
        if union > 0:
            iou = intersection / union
        else:
            iou = 1.0 if intersection == 0 else 0.0
        
        total_iou += iou
    
    return total_iou / batch_size


def train_epoch(
    model: ProteinSAM,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
    logger: logging.Logger,
    epoch: int,
    writer: Optional[SummaryWriter] = None,
    global_step: int = 0,
    use_external_esm: bool = False
) -> Dict[str, Any]:
    """
    Train for one epoch.

    Args:
        model: ProteinSAM model
        train_loader: Training data loader
        optimizer: Optimizer
        device: Device to use
        logger: Logger
        epoch: Current epoch number
        writer: TensorBoard writer for logging
        global_step: Global step counter for TensorBoard
        use_external_esm: Whether using external ESM embeddings

    Returns:
        Dictionary containing metrics and updated global_step
    """
    model.train()
    total_loss = 0.0
    total_dice_loss = 0.0
    total_ce_loss = 0.0
    total_acc = 0.0
    total_iou = 0.0
    num_batches = len(train_loader)

    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch} Training")

    for batch_idx, batch in enumerate(progress_bar):
        # Move batch to device
        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(device)

        # Handle point positions
        point_mask = batch["point_mask"]
        point_positions = batch["point_positions"] if torch.any(point_mask) else None

        # Forward pass
        optimizer.zero_grad()

        if use_external_esm:
            # Use external ESM embeddings (without BOS/EOS tokens)
            outputs = model(
                protein_input_ids=None,
                protein_attention_mask=batch["protein_attention_mask"],
                text_input_ids=batch.get("text_input_ids"),
                text_attention_mask=batch.get("text_attention_mask"),
                categories=batch["categories"],
                point_positions=point_positions,
                residue_labels=batch["residue_labels"],
                external_esm_embeddings=batch["external_esm_embeddings"]
            )
        else:
            # Use internal ESM encoder
            outputs = model(
                protein_input_ids=batch["protein_input_ids"],
                protein_attention_mask=batch["protein_attention_mask"],
                text_input_ids=batch.get("text_input_ids"),
                text_attention_mask=batch.get("text_attention_mask"),
                categories=batch["categories"],
                point_positions=point_positions,
                residue_labels=batch["residue_labels"]
            )

        # Attention mask includes BOS/EOS positions, slice to get actual sequence mask
        adjusted_mask = batch["protein_attention_mask"][:, 1:-1]

        loss = outputs["loss"]
        dice_loss = outputs["dice_loss"]
        ce_loss = outputs["ce_loss"]

        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # Calculate metrics using unified residue labels
        mask_acc = calculate_mask_accuracy(
            outputs["mask_predictions"],
            batch["residue_labels"],
            adjusted_mask
        )
        mask_iou = calculate_mask_iou(
            outputs["mask_predictions"],
            batch["residue_labels"],
            adjusted_mask
        )

        # Update running averages
        total_loss += loss.item()
        total_dice_loss += dice_loss.item()
        total_ce_loss += ce_loss.item()
        total_acc += mask_acc
        total_iou += mask_iou

        # Log to TensorBoard (per step)
        if writer is not None:
            writer.add_scalar("Train/Loss_Step", loss.item(), global_step)
            writer.add_scalar("Train/DiceLoss_Step", dice_loss.item(), global_step)
            writer.add_scalar("Train/CELoss_Step", ce_loss.item(), global_step)
            writer.add_scalar("Train/MaskAcc_Step", mask_acc, global_step)
            writer.add_scalar("Train/MaskIoU_Step", mask_iou, global_step)

        global_step += 1

        # Update progress bar
        avg_loss = total_loss / (batch_idx + 1)
        avg_loss_dice = total_dice_loss / (batch_idx + 1)
        avg_loss_ce = total_ce_loss / (batch_idx + 1)
        avg_mask_acc = total_acc / (batch_idx + 1)
        avg_mask_iou = total_iou / (batch_idx + 1)

        progress_bar.set_postfix({
            "loss": f"{avg_loss:.4f}",
            "loss_dice": f"{avg_loss_dice:.4f}",
            "loss_ce": f"{avg_loss_ce:.4f}",
            "mask_acc": f"{avg_mask_acc:.4f}",
            "mask_iou": f"{avg_mask_iou:.4f}"
        })

    return {
        "loss": total_loss / num_batches,
        "dice_loss": total_dice_loss / num_batches,
        "ce_loss": total_ce_loss / num_batches,
        "mask_accuracy": total_acc / num_batches,
        "mask_iou": total_iou / num_batches,
        "global_step": global_step
    }


def evaluate(
    model: ProteinSAM,
    eval_loader: DataLoader,
    device: str,
    logger: logging.Logger,
    use_external_esm: bool = False
) -> Dict[str, float]:
    """
    Evaluate the model.

    Args:
        model: ProteinSAM model
        eval_loader: Evaluation data loader
        device: Device to use
        logger: Logger
        use_external_esm: Whether using external ESM embeddings

    Returns:
        Dictionary containing evaluation metrics
    """
    model.eval()
    total_loss = 0.0
    total_dice_loss = 0.0
    total_ce_loss = 0.0
    total_acc = 0.0
    total_iou = 0.0
    num_batches = len(eval_loader)

    progress_bar = tqdm(eval_loader, desc="Evaluating")

    with torch.no_grad():
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(device)

            # Handle point positions
            point_mask = batch["point_mask"]
            point_positions = batch["point_positions"] if torch.any(point_mask) else None

            # Forward pass
            if use_external_esm:
                # Use external ESM embeddings (without BOS/EOS tokens)
                outputs = model(
                    protein_input_ids=None,
                    protein_attention_mask=batch["protein_attention_mask"],
                    text_input_ids=batch.get("text_input_ids"),
                    text_attention_mask=batch.get("text_attention_mask"),
                    categories=batch["categories"],
                    point_positions=point_positions,
                    residue_labels=batch["residue_labels"],
                    external_esm_embeddings=batch["external_esm_embeddings"]
                )
            else:
                # Use internal ESM encoder
                outputs = model(
                    protein_input_ids=batch["protein_input_ids"],
                    protein_attention_mask=batch["protein_attention_mask"],
                    text_input_ids=batch.get("text_input_ids"),
                    text_attention_mask=batch.get("text_attention_mask"),
                    categories=batch["categories"],
                    point_positions=point_positions,
                    residue_labels=batch["residue_labels"]
                )

            # Attention mask includes BOS/EOS positions, slice to get actual sequence mask
            adjusted_mask = batch["protein_attention_mask"][:, 1:-1]

            loss = outputs["loss"]
            dice_loss = outputs["dice_loss"]
            ce_loss = outputs["ce_loss"]

            # Calculate metrics using unified residue labels
            mask_acc = calculate_mask_accuracy(
                outputs["mask_predictions"],
                batch["residue_labels"],
                adjusted_mask
            )
            mask_iou = calculate_mask_iou(
                outputs["mask_predictions"],
                batch["residue_labels"],
                adjusted_mask
            )

            # Update running averages
            total_loss += loss.item()
            total_dice_loss += dice_loss.item()
            total_ce_loss += ce_loss.item()
            total_acc += mask_acc
            total_iou += mask_iou

            # Update progress bar
            avg_loss = total_loss / (batch_idx + 1)
            avg_loss_dice = total_dice_loss / (batch_idx + 1)
            avg_loss_ce = total_ce_loss / (batch_idx + 1)
            avg_mask_acc = total_acc / (batch_idx + 1)
            avg_mask_iou = total_iou / (batch_idx + 1)

            progress_bar.set_postfix({
                "loss": f"{avg_loss:.4f}",
                "loss_dice": f"{avg_loss_dice:.4f}",
                "loss_ce": f"{avg_loss_ce:.4f}",
                "mask_acc": f"{avg_mask_acc:.4f}",
                "mask_iou": f"{avg_mask_iou:.4f}"
            })

    return {
        "loss": total_loss / num_batches,
        "dice_loss": total_dice_loss / num_batches,
        "ce_loss": total_ce_loss / num_batches,
        "mask_accuracy": total_acc / num_batches,
        "mask_iou": total_iou / num_batches
    }


def main():
    parser = argparse.ArgumentParser(description="Train ProteinSAM model with pre-computed ESM embeddings")

    # Model arguments
    parser.add_argument("--esm_model_path", type=str,
                       default="/home/dataset-local/projects_dir/pretrained_model/esm2_t36_3B_UR50D",
                       help="Path to ESM model")
    parser.add_argument("--llama_model_path", type=str,
                       default="/home/dataset-local/projects_dir/pretrained_model/Llama-3.1-8B-Instruct",
                       help="Path to Llama model")
    parser.add_argument("--output_llama_layer", type=int, default=16,
                       help="Which Llama layer to use for text encoding")

    # Data arguments
    # parser.add_argument("--data_root", type=str, default="../data_70",
    #                    help="Root directory for datasets")
    # parser.add_argument("--data_root", type=str, default="../data_30",
    #                    help="Root directory for datasets")
    parser.add_argument("--data_root", type=str, default="../data_frag_50",
                       help="Root directory for datasets")
    parser.add_argument("--data_root_for_data", type=str, default="../data_frag_50",
                       help="为了切换data_30时不在重新编码一遍全数据集，故用这个字段将esm embedding的输出目录命名为data_70")
    parser.add_argument("--data_name", type=str, default="VenusX_Dom||VenusX_Act||VenusX_BindI||VenusX_Motif||VenusX_Evo",
                       help="Dataset name(s). Single: 'VenusX_Dom' or Multiple: 'VenusX_Dom||VenusX_Act||VenusX_BindI'")
    parser.add_argument("--max_sequence_length", type=int, default=1021,
                       help="Maximum protein sequence length")
    parser.add_argument("--max_text_length", type=int, default=128,
                       help="Maximum text length")
    parser.add_argument("--use_category_cache", default=True,
                       help="Use pre-computed category embeddings cache")
    parser.add_argument("--category_embeddings_base_dir", type=str, default=".",
                       help="category_embeddings/<data>/category_embeddings.pt")
    
    # ESM embeddings cache arguments
    parser.add_argument("--use_esm_cache", default=True,
                       help="Use pre-computed ESM embeddings cache (saves GPU memory and computation)")
    parser.add_argument("--esm_embeddings_base_dir", type=str, default=".",
                       help="Base directory under which esm_embeddings/<model>/<data>/ was created by preprocess_esm_3B.py")

    # Training arguments
    parser.add_argument("--batch_size", type=int, default=32,
                       help="Training batch size")
    parser.add_argument("--eval_batch_size", type=int, default=16,
                       help="Evaluation batch size")
    parser.add_argument("--learning_rate", type=float, default=5e-5,
                       help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=50,
                       help="Number of training epochs")
    parser.add_argument("--warmup_steps", type=int, default=500,
                       help="Number of warmup steps")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                       help="Weight decay")
    parser.add_argument("--scheduler_type", type=str, default="cosine",
                       choices=["cosine", "step"], help="Scheduler type")

    # Model architecture arguments
    parser.add_argument("--decoder_num_heads", type=int, default=8,
                       help="Number of attention heads in decoder")
    parser.add_argument("--decoder_num_self_attention_heads", type=int, default=8,
                       help="Number of self-attention heads in decoder (if None, uses same as decoder_num_heads)")
    parser.add_argument("--decoder_num_layers", type=int, default=2,
                       help="Number of decoder layers")
    parser.add_argument("--decoder_intermediate_size", type=int, default=1280,
                       help="Decoder intermediate size")  # 3B default
    parser.add_argument("--dropout_rate", type=float, default=0.1,
                       help="Dropout rate")
    parser.add_argument("--use_sigmoid_head", action="store_true", default=True,
                       help="Use N*1 sigmoid output head instead of default N*2 softmax head")

    # Dataset-specific arguments
    parser.add_argument("--null_position_prob", type=float, default=0.0,
                       help="Probability to set position prompt to null")
    parser.add_argument("--random_position_prob", type=float, default=0.0,
                       help="Probability to set random position")
    parser.add_argument("--position_noise_std", type=float, default=20,
                       help="Standard deviation for position noise")

    # Other arguments
    parser.add_argument("--output_dir", type=str, default="./checkpoints_grounding_3B_cluster_frag_50_point_only",
                       help="Output directory for model checkpoints")
    parser.add_argument("--log_dir", type=str, default="./logs",
                       help="Directory for logs")
    parser.add_argument("--save_every", type=int, default=1,
                       help="Save checkpoint every N epochs")
    parser.add_argument("--eval_every", type=int, default=1,
                       help="Evaluate every N epochs")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to use for training")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    parser.add_argument("--num_workers", type=int, default=4,
                       help="Number of data loader workers")

    args = parser.parse_args()

    # Set seed
    set_seed(args.seed)

    # Setup logging
    logger = setup_logging(args.log_dir)
    logger.info("Starting ProteinSAM training")
    logger.info(f"Arguments: {vars(args)}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Setup TensorBoard writer
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tensorboard_log_dir = os.path.join('tensorboard/logs', f"train_{timestamp}")
    writer = SummaryWriter(log_dir=tensorboard_log_dir)
    logger.info(f"TensorBoard logs will be saved to: {tensorboard_log_dir}")

    # Load datasets and collator
    logger.info("Loading datasets...")

    if args.use_esm_cache:
        # Auto-derive embeddings directory from model and data names
        esm_model_name = os.path.basename(os.path.normpath(args.esm_model_path))
        data_root_name = os.path.basename(os.path.normpath(args.data_root))
        data_root_name_for_esm = os.path.basename(os.path.normpath(args.data_root_for_data))
        esm_embeddings_dir = os.path.join(
            args.esm_embeddings_base_dir, "esm_embeddings", esm_model_name, data_root_name_for_esm
        )
        # Use pre-computed ESM embeddings
        logger.info(f"Using pre-computed ESM embeddings from {esm_embeddings_dir}")
        logger.info("Note: ESM embeddings do NOT include BOS/EOS tokens")
        datasets, collator = get_datasets_and_collator_with_esm_cache(
            root_dir=args.data_root,
            data_name=args.data_name,
            esm_model_path=args.esm_model_path,
            esm_embeddings_dir=esm_embeddings_dir,
            llama_model_path=args.llama_model_path if not args.use_category_cache else None,
            max_sequence_length=args.max_sequence_length,
            max_text_length=args.max_text_length,
            use_category_cache=args.use_category_cache,
            null_position_prob=args.null_position_prob,
            random_position_prob=args.random_position_prob,
            position_noise_std=args.position_noise_std
        )
    else:
        raise NotImplementedError("Training without ESM cache is not supported in this script.")

    # Create data loaders
    train_loader = DataLoader(
        datasets["train"],
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=args.num_workers,
        pin_memory=True
    )

    eval_loader = None
    if "test" in datasets:
        eval_loader = DataLoader(
            datasets["test"],
            batch_size=args.eval_batch_size,
            shuffle=False,
            collate_fn=collator,
            num_workers=args.num_workers,
            pin_memory=True
        )

    # Initialize model
    logger.info("Initializing model...")

    # category_embeddings_path = os.path.join(args.category_embeddings_base_dir, "category_embeddings", os.path.basename(os.path.normpath(args.data_root)), "category_embeddings.pt")
    category_embeddings_path = os.path.join(args.category_embeddings_base_dir, "category_embeddings", os.path.basename(os.path.normpath(args.data_root_for_data)), "category_embeddings.pt")

    # Save ProteinSAM initialization parameters to JSON file
    protein_sam_init_params = {
        "esm_model_path": args.esm_model_path,
        "llama_model_path": args.llama_model_path,
        "output_llama_layer": args.output_llama_layer,
        "decoder_num_heads": args.decoder_num_heads,
        "decoder_num_self_attention_heads": args.decoder_num_self_attention_heads,
        "decoder_num_layers": args.decoder_num_layers,
        "decoder_intermediate_size": args.decoder_intermediate_size,
        "max_sequence_length": args.max_sequence_length,
        "dropout_rate": args.dropout_rate,
        "device": args.device,
        "use_category_cache": args.use_category_cache,
        "category_embeddings_path": category_embeddings_path if args.use_category_cache else None,
        "use_sigmoid_head": args.use_sigmoid_head
    }

    # Save parameters to output directory
    params_file = os.path.join(args.output_dir, "protein_sam_init_params.json")
    with open(params_file, 'w') as f:
        json.dump(protein_sam_init_params, f, indent=2)
    logger.info(f"Saved ProteinSAM initialization parameters to {params_file}")

    model = ProteinSAM(
        esm_model_path=args.esm_model_path,
        llama_model_path=args.llama_model_path,
        output_llama_layer=args.output_llama_layer,
        decoder_num_heads=args.decoder_num_heads,
        decoder_num_layers=args.decoder_num_layers,
        decoder_intermediate_size=args.decoder_intermediate_size,
        decoder_num_self_attention_heads=args.decoder_num_self_attention_heads,
        max_sequence_length=args.max_sequence_length,
        dropout_rate=args.dropout_rate,
        device=args.device,
        use_category_cache=args.use_category_cache,
        category_embeddings_path=category_embeddings_path if args.use_category_cache else None,
        use_external_esm=args.use_esm_cache,  # Skip loading ESM model if using cache
        use_sigmoid_head=args.use_sigmoid_head
    )

    model = model.to(args.device)

    # Log model info
    param_info = model.get_trainable_parameters()
    logger.info(f"Model parameters: {param_info}")

    # Log hyperparameters to TensorBoard
    writer.add_hparams(
        {
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "num_epochs": args.num_epochs,
            "decoder_num_heads": args.decoder_num_heads,
            "decoder_num_layers": args.decoder_num_layers,
            "dropout_rate": args.dropout_rate,
            "use_esm_cache": args.use_esm_cache,
        },
        {}
    )

    # Setup optimizer and scheduler
    optimizer = AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )

    if args.scheduler_type == "cosine":
        scheduler = CosineAnnealingLR(optimizer, T_max=args.num_epochs)
    else:
        scheduler = StepLR(optimizer, step_size=args.num_epochs // 3, gamma=0.1)

    # Training loop
    best_mask_iou = 0.0
    global_step = 0

    for epoch in range(1, args.num_epochs + 1):
        logger.info(f"Starting epoch {epoch}/{args.num_epochs}")

        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, args.device, logger, epoch,
            writer=writer, global_step=global_step, use_external_esm=args.use_esm_cache
        )
        global_step = train_metrics["global_step"]
        scheduler.step()

        # Log training metrics to TensorBoard (per epoch)
        writer.add_scalar("Train/Loss_Epoch", train_metrics["loss"], epoch)
        writer.add_scalar("Train/DiceLoss_Epoch", train_metrics["dice_loss"], epoch)
        writer.add_scalar("Train/CELoss_Epoch", train_metrics["ce_loss"], epoch)
        writer.add_scalar("Train/MaskAcc_Epoch", train_metrics["mask_accuracy"], epoch)
        writer.add_scalar("Train/MaskIoU_Epoch", train_metrics["mask_iou"], epoch)
        writer.add_scalar("Train/LearningRate", scheduler.get_last_lr()[0], epoch)

        # Log training metrics
        logger.info(f"Epoch {epoch} Training - Loss: {train_metrics['loss']:.4f}, "
                   f"Dice: {train_metrics['dice_loss']:.4f}, CE: {train_metrics['ce_loss']:.4f}, "
                   f"Mask Acc: {train_metrics['mask_accuracy']:.4f}, "
                   f"Mask IoU: {train_metrics['mask_iou']:.4f}")

        # Evaluate
        if eval_loader is not None and epoch % args.eval_every == 0:
            eval_metrics = evaluate(
                model, eval_loader, args.device, logger, use_external_esm=args.use_esm_cache
            )

            # Log eval metrics to TensorBoard
            writer.add_scalar("Eval/Loss", eval_metrics["loss"], epoch)
            writer.add_scalar("Eval/DiceLoss", eval_metrics["dice_loss"], epoch)
            writer.add_scalar("Eval/CELoss", eval_metrics["ce_loss"], epoch)
            writer.add_scalar("Eval/MaskAcc", eval_metrics["mask_accuracy"], epoch)
            writer.add_scalar("Eval/MaskIoU", eval_metrics["mask_iou"], epoch)

            logger.info(f"Epoch {epoch} Evaluation - Loss: {eval_metrics['loss']:.4f}, "
                       f"Dice: {eval_metrics['dice_loss']:.4f}, CE: {eval_metrics['ce_loss']:.4f}, "
                       f"Mask Acc: {eval_metrics['mask_accuracy']:.4f}, "
                       f"Mask IoU: {eval_metrics['mask_iou']:.4f}")

            # Save best model
            if eval_metrics['mask_iou'] > best_mask_iou:
                best_mask_iou = eval_metrics['mask_iou']
                best_model_path = os.path.join(args.output_dir, "best_model.pt")
                model.save_model(best_model_path)
                logger.info(f"New best model saved with mask IoU: {best_mask_iou:.4f}")
                writer.add_scalar("Eval/BestMaskIoU", best_mask_iou, epoch)

        # Save checkpoint
        if epoch % args.save_every == 0:
            checkpoint_path = os.path.join(args.output_dir, f"checkpoint_epoch_{epoch}.pt")
            model.save_model(checkpoint_path)
            logger.info(f"Checkpoint saved: {checkpoint_path}")

    # Save final model
    final_model_path = os.path.join(args.output_dir, "final_model.pt")
    model.save_model(final_model_path)
    logger.info(f"Final model saved: {final_model_path}")

    # Close TensorBoard writer
    writer.close()
    logger.info(f"TensorBoard logs saved to: {tensorboard_log_dir}")
    logger.info("Training completed!")


if __name__ == "__main__":
    main()