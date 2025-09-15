"""
ProteinSAM: Segment Anything Model for protein functional region grounding.
Integrates protein encoder, prompt encoder, and position decoder.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, Any, List
import os

from protein_encoder import ProteinEncoder
from prompt_encoder import PromptEncoder
from position_decoder import PositionDecoder


class ProteinSAM(nn.Module):
    """
    ProteinSAM model for protein functional region grounding.
    
    Architecture:
    - Protein Encoder: ESM2 (frozen)
    - Prompt Encoder: Llama text encoder (frozen) + learnable projection + position embedding
    - Position Decoder: Lightweight transformer decoder for position prediction
    """
    
    def __init__(
        self,
        esm_model_path: str,
        llama_model_path: str,
        output_llama_layer: int = 16,
        decoder_num_heads: int = 8,
        decoder_num_layers: int = 2,
        decoder_intermediate_size: int = 512,
        max_sequence_length: int = 2048,
        dropout_rate: float = 0.1,
        device: str = "cuda",
        use_category_cache: bool = True,
        category_embeddings_path: Optional[str] = None
    ):
        super().__init__()
        
        self.device = device
        self.max_sequence_length = max_sequence_length
        self.use_category_cache = use_category_cache
        
        # Initialize protein encoder
        self.protein_encoder = ProteinEncoder(
            esm_model_path=esm_model_path,
            device=device
        )
        
        # Initialize prompt encoder
        self.prompt_encoder = PromptEncoder(
            llama_model_path=llama_model_path,
            protein_hidden_size=self.protein_encoder.hidden_size,
            output_llama_layer=output_llama_layer,
            max_sequence_length=max_sequence_length,
            dropout_rate=dropout_rate,
            use_cache=use_category_cache
        )
        
        # Load category embeddings if provided
        if use_category_cache and category_embeddings_path is not None:
            self.load_category_embeddings(category_embeddings_path)
        
        # Initialize position decoder
        self.position_decoder = PositionDecoder(
            hidden_size=self.protein_encoder.hidden_size,
            num_attention_heads=decoder_num_heads,
            num_layers=decoder_num_layers,
            intermediate_size=decoder_intermediate_size,
            dropout_rate=dropout_rate
        )
        
        # Loss functions
        self.ce_loss_fn = nn.CrossEntropyLoss(ignore_index=-1)
        self.smooth = 1e-6  # For IoU loss smoothing
    
    def load_category_embeddings(self, embeddings_path: str):
        """Load pre-computed category embeddings."""
        checkpoint = torch.load(embeddings_path, map_location=self.device)
        category_embeddings = checkpoint["category_embeddings"]
        
        # Move embeddings to device
        for cat in category_embeddings:
            category_embeddings[cat] = category_embeddings[cat].to(self.device)
        
        self.prompt_encoder.set_category_embeddings_cache(category_embeddings)
        print(f"Loaded category embeddings from {embeddings_path}")
    
    def forward(
        self,
        protein_input_ids: torch.Tensor,      # (batch_size, seq_len)
        protein_attention_mask: torch.Tensor, # (batch_size, seq_len)
        text_input_ids: Optional[torch.Tensor] = None,         # (batch_size, text_len) - fallback
        text_attention_mask: Optional[torch.Tensor] = None,    # (batch_size, text_len) - fallback
        categories: Optional[list] = None,                     # List of category names - preferred
        point_positions: Optional[torch.Tensor] = None,       # (batch_size,)
        start_labels: Optional[torch.Tensor] = None,          # (batch_size,)
        end_labels: Optional[torch.Tensor] = None             # (batch_size,)
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass of ProteinSAM model.
        
        Args:
            protein_input_ids: Tokenized protein sequences
            protein_attention_mask: Attention mask for proteins
            text_input_ids: Tokenized functional region names
            text_attention_mask: Attention mask for text
            point_positions: Position prompts (optional)
            start_labels: Ground truth start positions for training
            end_labels: Ground truth end positions for training
            
        Returns:
            Dictionary containing logits, predictions, and loss (if labels provided)
        """
        batch_size, seq_len = protein_input_ids.shape
        
        # Encode protein sequences
        protein_embeddings = self.protein_encoder(
            protein_input_ids=protein_input_ids,
            attention_mask=protein_attention_mask
        )  # (batch_size, seq_len, hidden_size)
        
        # Get text token (without position encoding)
        text_token = self.prompt_encoder(
            text_input_ids=text_input_ids,
            text_attention_mask=text_attention_mask,
            categories=categories
        )  # (batch_size, 1, hidden_size)
        
        # Combine protein embeddings and text token
        combined_embeddings = torch.cat([text_token, protein_embeddings], dim=1)  # (batch_size, seq_len+1, hidden_size)
        
        # Add positional encoding to combined embeddings
        combined_embeddings_with_pos = self.prompt_encoder.get_positional_encoding(
            embeddings=combined_embeddings,
            point_positions=point_positions,
            prompt_token_idx=0  # Text token is at index 0
        )  # (batch_size, seq_len+1, hidden_size)
        
        # Split back to text token and protein embeddings
        text_token_with_pos = combined_embeddings_with_pos[:, 0:1, :]  # (batch_size, 1, hidden_size)
        protein_embeddings_with_pos = combined_embeddings_with_pos[:, 1:, :]  # (batch_size, seq_len, hidden_size)
        
        # Decode binary mask using protein embeddings with position encoding and text token
        mask_logits = self.position_decoder(
            protein_embeddings=protein_embeddings_with_pos,
            prompt_embeddings=text_token_with_pos,
            protein_attention_mask=protein_attention_mask
        )  # (batch_size, seq_len, 2)
        
        # Get mask predictions
        mask_predictions = torch.argmax(mask_logits, dim=-1)  # (batch_size, seq_len)
        mask_probs = F.softmax(mask_logits, dim=-1)  # (batch_size, seq_len, 2)
        
        outputs = {
            "mask_logits": mask_logits,
            "mask_predictions": mask_predictions,
            "mask_probs": mask_probs
        }
        
        # Compute loss if labels are provided
        if start_labels is not None and end_labels is not None:
            # Create binary mask from start/end labels
            mask_labels = self._create_mask_labels(start_labels, end_labels, seq_len, batch_size)
            
            # Compute IoU loss and CrossEntropy loss
            iou_loss = self._compute_iou_loss(mask_logits, mask_labels, protein_attention_mask)
            ce_loss = self._compute_ce_loss(mask_logits, mask_labels, protein_attention_mask)
            
            # Combine losses
            total_loss = iou_loss + ce_loss
            
            outputs.update({
                "loss": total_loss,
                "iou_loss": iou_loss,
                "ce_loss": ce_loss,
                "mask_labels": mask_labels
            })
        
        return outputs
    
    def _create_mask_labels(
        self, 
        start_labels: torch.Tensor, 
        end_labels: torch.Tensor, 
        seq_len: int, 
        batch_size: int
    ) -> torch.Tensor:
        """Create binary mask labels from start/end positions."""
        mask_labels = torch.zeros(batch_size, seq_len, dtype=torch.long, device=start_labels.device)
        
        for b in range(batch_size):
            start = start_labels[b].item()
            end = end_labels[b].item()
            if start >= 0 and end >= 0 and start <= end:
                mask_labels[b, start:end+1] = 1
        
        return mask_labels
    
    def _compute_iou_loss(
        self, 
        mask_logits: torch.Tensor, 
        mask_labels: torch.Tensor, 
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Compute IoU loss for binary segmentation."""
        # Get foreground probabilities
        mask_probs = F.softmax(mask_logits, dim=-1)  # (batch_size, seq_len, 2)
        pred_fg = mask_probs[:, :, 1]  # (batch_size, seq_len)
        true_fg = mask_labels.float()  # (batch_size, seq_len)
        
        # Apply attention mask
        if attention_mask is not None:
            pred_fg = pred_fg * attention_mask.float()
            true_fg = true_fg * attention_mask.float()
        
        # Compute IoU
        intersection = (pred_fg * true_fg).sum(dim=1)  # (batch_size,)
        union = pred_fg.sum(dim=1) + true_fg.sum(dim=1) - intersection  # (batch_size,)
        
        # IoU with smoothing
        iou = (intersection + self.smooth) / (union + self.smooth)  # (batch_size,)
        
        # IoU loss (1 - mean IoU)
        iou_loss = 1.0 - iou.mean()
        
        return iou_loss
    
    def _compute_ce_loss(
        self, 
        mask_logits: torch.Tensor, 
        mask_labels: torch.Tensor, 
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Compute CrossEntropy loss for binary classification."""
        batch_size, seq_len, _ = mask_logits.shape
        
        # Reshape for loss computation
        mask_logits_flat = mask_logits.view(-1, 2)  # (batch_size * seq_len, 2)
        mask_labels_flat = mask_labels.view(-1)     # (batch_size * seq_len,)
        
        # Apply attention mask
        if attention_mask is not None:
            attention_mask_flat = attention_mask.view(-1)  # (batch_size * seq_len,)
            valid_indices = attention_mask_flat == 1
            mask_logits_flat = mask_logits_flat[valid_indices]
            mask_labels_flat = mask_labels_flat[valid_indices]
        
        ce_loss = F.cross_entropy(mask_logits_flat, mask_labels_flat)
        
        return ce_loss
    
    def predict(
        self,
        protein_input_ids: torch.Tensor,
        protein_attention_mask: torch.Tensor,
        text_input_ids: torch.Tensor,
        text_attention_mask: torch.Tensor,
        point_positions: Optional[torch.Tensor] = None,
        return_probabilities: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Inference method for mask prediction.
        
        Args:
            protein_input_ids: Tokenized protein sequences
            protein_attention_mask: Attention mask for proteins
            text_input_ids: Tokenized functional region names
            text_attention_mask: Attention mask for text
            point_positions: Position prompts (optional)
            return_probabilities: Whether to return probabilities
            
        Returns:
            Predicted masks and optionally probabilities
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(
                protein_input_ids=protein_input_ids,
                protein_attention_mask=protein_attention_mask,
                text_input_ids=text_input_ids,
                text_attention_mask=text_attention_mask,
                point_positions=point_positions
            )
            
            predictions = {
                "mask_predictions": outputs["mask_predictions"]
            }
            
            if return_probabilities:
                predictions.update({
                    "mask_probabilities": outputs["mask_probs"]
                })
        
        return predictions
    
    def batch_predict(
        self,
        protein_sequences: List[str],
        categories: List[str],
        point_positions: Optional[List[int]] = None,
        category_embeddings: Optional[Dict[str, torch.Tensor]] = None
    ) -> List[Dict[str, Any]]:
        """
        Batch prediction for multiple protein sequences and categories.
        
        Args:
            protein_sequences: List of protein sequences
            categories: List of functional region categories
            point_positions: List of position prompts (optional)
            category_embeddings: Pre-computed category embeddings (optional)
            
        Returns:
            List of prediction results
        """
        results = []
        
        # Tokenize protein sequences
        protein_tokenized = self.protein_encoder.tokenizer(
            protein_sequences,
            padding=True,
            truncation=True,
            max_length=self.max_sequence_length,
            return_tensors="pt"
        )
        protein_input_ids = protein_tokenized['input_ids'].to(self.device)
        protein_attention_mask = protein_tokenized['attention_mask'].to(self.device)
        
        # Tokenize categories
        text_tokenized = self.prompt_encoder.tokenizer(
            categories,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt"
        )
        text_input_ids = text_tokenized['input_ids'].to(self.device)
        text_attention_mask = text_tokenized['attention_mask'].to(self.device)
        
        # Convert point positions to tensor if provided
        point_tensor = None
        if point_positions is not None:
            point_tensor = torch.tensor(point_positions, device=self.device)
        
        # Predict
        predictions = self.predict(
            protein_input_ids=protein_input_ids,
            protein_attention_mask=protein_attention_mask,
            text_input_ids=text_input_ids,
            text_attention_mask=text_attention_mask,
            point_positions=point_tensor,
            return_probabilities=True
        )
        
        # Convert to list of dictionaries
        for i in range(len(protein_sequences)):
            result = {
                "protein_sequence": protein_sequences[i],
                "category": categories[i],
                "start_position": predictions["start_predictions"][i].item(),
                "end_position": predictions["end_predictions"][i].item(),
                "start_probability": predictions["start_probabilities"][i].cpu().numpy(),
                "end_probability": predictions["end_probabilities"][i].cpu().numpy()
            }
            if point_positions is not None:
                result["point_position"] = point_positions[i]
            
            results.append(result)
        
        return results
    
    def save_model(self, save_path: str):
        """Save model state dict."""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Only save trainable parameters
        trainable_state_dict = {}
        for name, param in self.named_parameters():
            if param.requires_grad:
                trainable_state_dict[name] = param
        
        torch.save({
            'model_state_dict': trainable_state_dict,
            'model_config': {
                'decoder_num_heads': self.position_decoder.cross_attention_layers[0].num_heads,
                'decoder_num_layers': len(self.position_decoder.cross_attention_layers),
                'decoder_intermediate_size': self.position_decoder.feed_forward_layers[0][0].out_features,
                'max_sequence_length': self.max_sequence_length,
                'protein_hidden_size': self.protein_encoder.hidden_size,
                'output_llama_layer': self.prompt_encoder.output_llama_layer
            }
        }, save_path)
        
        print(f"Model saved to {save_path}")
    
    def load_model(self, load_path: str):
        """Load model state dict."""
        checkpoint = torch.load(load_path, map_location=self.device)
        model_state_dict = checkpoint['model_state_dict']
        
        # Load only trainable parameters
        current_state_dict = self.state_dict()
        for name, param in model_state_dict.items():
            if name in current_state_dict:
                current_state_dict[name].copy_(param)
        
        print(f"Model loaded from {load_path}")
    
    def get_trainable_parameters(self):
        """Get count of trainable parameters."""
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.parameters())
        
        return {
            "trainable_parameters": trainable_params,
            "total_parameters": total_params,
            "trainable_percentage": trainable_params / total_params * 100
        }