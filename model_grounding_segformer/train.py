"""
Training script for ProteinSAM model.
Single GPU training for protein functional region grounding.
"""

import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
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

from protein_sam import ProteinSAM
from dataset import get_datasets_and_collator


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
    epoch: int
) -> Dict[str, float]:
    """Train for one epoch."""
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
        outputs = model(
            protein_input_ids=batch["protein_input_ids"],
            protein_attention_mask=batch["protein_attention_mask"],
            text_input_ids=batch.get("text_input_ids"),
            text_attention_mask=batch.get("text_attention_mask"),
            categories=batch["categories"],
            point_positions=point_positions,
            residue_labels=batch["residue_labels"]
        )
        
        loss = outputs["loss"]
        dice_loss = outputs["dice_loss"]
        ce_loss = outputs["ce_loss"]
                
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        adjusted_mask = batch["protein_attention_mask"][:, 1:-1]

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
        total_acc += mask_acc  # Reuse variable names for compatibility
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
        "mask_accuracy": total_acc / num_batches,
        "mask_iou": total_iou / num_batches
    }


def evaluate(
    model: ProteinSAM,
    eval_loader: DataLoader,
    device: str,
    logger: logging.Logger
) -> Dict[str, float]:
    """Evaluate the model."""
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
            outputs = model(
                protein_input_ids=batch["protein_input_ids"],
                protein_attention_mask=batch["protein_attention_mask"],
                text_input_ids=batch.get("text_input_ids"),
                text_attention_mask=batch.get("text_attention_mask"),
                categories=batch["categories"],
                point_positions=point_positions,
                residue_labels=batch["residue_labels"]
            )
            
            loss = outputs["loss"]
            dice_loss = outputs["dice_loss"]
            ce_loss = outputs["ce_loss"]
            
            adjusted_mask = batch["protein_attention_mask"][:, 1:-1]
            
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
        "mask_accuracy": total_acc / num_batches,
        "mask_iou": total_iou / num_batches
    }


def main():
    parser = argparse.ArgumentParser(description="Train ProteinSAM model")
    
    # Model arguments
    # parser.add_argument("--esm_model_path", type=str, 
    #                    default="/home/lfj/projects_dir/pretrained_model/esm2_t30_150M_UR50D",
    #                    help="Path to ESM model")
    # parser.add_argument("--esm_model_path", type=str, 
    #                    default="/home/lfj/projects_dir/pretrained_model/models--facebook--esm2_t33_650M_UR50D",
    #                    help="Path to ESM model")
    parser.add_argument("--esm_model_path", type=str, 
                       default="/home/lfj/projects_dir/pretrained_model/esm2_t36_3B_UR50D",
                       help="Path to ESM model")
    parser.add_argument("--llama_model_path", type=str,
                       default="/home/lfj/projects_dir/pretrained_model/Llama-3.1-8B-Instruct",
                       help="Path to Llama model")
    parser.add_argument("--output_llama_layer", type=int, default=16,
                       help="Which Llama layer to use for text encoding")
    
    # Data arguments
    parser.add_argument("--data_root", type=str, default="../data",
                       help="Root directory for datasets")
    parser.add_argument("--data_name", type=str, default="VenusX_Dom||VenusX_Act||VenusX_BindI||VenusX_Motif||VenusX_Evo",
                       help="Dataset name(s). Single: 'VenusX_Dom' or Multiple: 'VenusX_Dom||VenusX_Act||VenusX_BindI'")
    parser.add_argument("--max_sequence_length", type=int, default=1021,
                       help="Maximum protein sequence length")
    parser.add_argument("--max_text_length", type=int, default=128,
                       help="Maximum text length")
    parser.add_argument("--category_embeddings_path", type=str, 
                       default="./category_embeddings.pt",
                       help="Path to pre-computed category embeddings")
    parser.add_argument("--use_category_cache", action="store_true", default=True,
                       help="Use pre-computed category embeddings cache")
    
    # Training arguments
    parser.add_argument("--batch_size", type=int, default=8,
                       help="Training batch size")
    parser.add_argument("--eval_batch_size", type=int, default=16,
                       help="Evaluation batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-5,
                       help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=20,
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
    # parser.add_argument("--decoder_intermediate_size", type=int, default=512,
    #                    help="Decoder intermediate size")  # 150M
    # parser.add_argument("--decoder_intermediate_size", type=int, default=1280,
    #                    help="Decoder intermediate size")  # 650M
    parser.add_argument("--decoder_intermediate_size", type=int, default=2560,
                       help="Decoder intermediate size")  # 3B
    parser.add_argument("--dropout_rate", type=float, default=0.1,
                       help="Dropout rate")
    
    # Dataset-specific arguments
    parser.add_argument("--null_position_prob", type=float, default=0.3,
                       help="Probability to set position prompt to null")
    parser.add_argument("--random_position_prob", type=float, default=0.0,
                       help="Probability to set random position")
    parser.add_argument("--position_noise_std", type=float, default=20,
                       help="Standard deviation for position noise")
    
    # Other arguments
    parser.add_argument("--output_dir", type=str, default="./checkpoints_grounding_3B",
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
    
    # Load datasets and collator
    logger.info("Loading datasets...")
    datasets, collator = get_datasets_and_collator(
        root_dir=args.data_root,
        data_name=args.data_name,
        esm_model_path=args.esm_model_path,
        llama_model_path=args.llama_model_path if not args.use_category_cache else None,
        max_sequence_length=args.max_sequence_length,
        max_text_length=args.max_text_length,
        use_category_cache=args.use_category_cache,
        null_position_prob=args.null_position_prob,
        random_position_prob=args.random_position_prob,
        position_noise_std=args.position_noise_std
    )
    
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
    if "valid" in datasets:
        eval_loader = DataLoader(
            datasets["valid"],
            batch_size=args.eval_batch_size,
            shuffle=False,
            collate_fn=collator,
            num_workers=args.num_workers,
            pin_memory=True
        )
    
    # Initialize model
    logger.info("Initializing model...")
    
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
        "category_embeddings_path": args.category_embeddings_path if args.use_category_cache else None
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
        category_embeddings_path=args.category_embeddings_path if args.use_category_cache else None
    )
    
    model = model.to(args.device)
    
    # Log model info
    param_info = model.get_trainable_parameters()
    logger.info(f"Model parameters: {param_info}")
    
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
    
    for epoch in range(1, args.num_epochs + 1):
        logger.info(f"Starting epoch {epoch}/{args.num_epochs}")
        
        # Train
        train_metrics = train_epoch(model, train_loader, optimizer, args.device, logger, epoch)
        scheduler.step()
        
        # Log training metrics
        logger.info(f"Epoch {epoch} Training - Loss: {train_metrics['loss']:.4f}, "
                   f"Mask Acc: {train_metrics['mask_accuracy']:.4f}, "
                   f"Mask IoU: {train_metrics['mask_iou']:.4f}")
        
        # Evaluate
        if eval_loader is not None and epoch % args.eval_every == 0:
            eval_metrics = evaluate(model, eval_loader, args.device, logger)
            
            logger.info(f"Epoch {epoch} Evaluation - Loss: {eval_metrics['loss']:.4f}, "
                       f"Mask Acc: {eval_metrics['mask_accuracy']:.4f}, "
                       f"Mask IoU: {eval_metrics['mask_iou']:.4f}")
            
            # Save best model
            if eval_metrics['mask_iou'] > best_mask_iou:
                best_mask_iou = eval_metrics['mask_iou']
                best_model_path = os.path.join(args.output_dir, "best_model.pt")
                model.save_model(best_model_path)
                logger.info(f"New best model saved with mask IoU: {best_mask_iou:.4f}")
        
        # Save checkpoint
        if epoch % args.save_every == 0:
            checkpoint_path = os.path.join(args.output_dir, f"checkpoint_epoch_{epoch}.pt")
            model.save_model(checkpoint_path)
            logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    # Save final model
    final_model_path = os.path.join(args.output_dir, "final_model.pt")
    model.save_model(final_model_path)
    logger.info(f"Final model saved: {final_model_path}")
    
    logger.info("Training completed!")


if __name__ == "__main__":
    main()