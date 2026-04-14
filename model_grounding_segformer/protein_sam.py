"""
ProteinSAM: Segment Anything Model for protein functional region grounding.
Integrates protein encoder, prompt encoder, and position decoder.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, Any, List
import os

try:
    from .protein_encoder import ProteinEncoder
    from .prompt_encoder import PromptEncoder
    from .position_decoder import PositionDecoder
except ImportError:
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
        esm_model_path: Optional[str] = None,
        llama_model_path: Optional[str] = None,
        output_llama_layer: int = 16,
        decoder_num_heads: int = 8,
        decoder_num_layers: int = 2,
        decoder_intermediate_size: int = 512,
        decoder_num_self_attention_heads: int = None,  # New parameter for self-attention heads
        max_sequence_length: int = 2048,
        dropout_rate: float = 0.1,
        device: str = "cuda",
        use_category_cache: bool = True,
        category_embeddings_path: Optional[str] = None,
        use_external_embeddings: bool = False,  # New parameter for direct embedding input
        use_external_esm: bool = False,
        use_sigmoid_head: bool = False  # If True, use N*1 sigmoid output instead of N*2 softmax
    ):
        super().__init__()

        self.device = device
        self.max_sequence_length = max_sequence_length
        self.use_category_cache = use_category_cache
        self.use_external_embeddings = use_external_embeddings
        self.use_sigmoid_head = use_sigmoid_head
        
        # Initialize protein encoder
        if not use_external_esm:
            self.protein_encoder = ProteinEncoder(
                esm_model_path=esm_model_path,
                device=device
            )
            self.protein_encoder_hidden_size = self.protein_encoder.hidden_size
        else:
            self.protein_encoder = None  # Will use external ESM embeddings directly
            from transformers import AutoConfig
            esm_config = AutoConfig.from_pretrained(esm_model_path)
            self.protein_encoder_hidden_size = esm_config.hidden_size
        
        # Initialize prompt encoder only if not using external embeddings
        if not use_external_embeddings:
            self.prompt_encoder = PromptEncoder(
                llama_model_path=llama_model_path,
                protein_hidden_size=self.protein_encoder_hidden_size,
                output_llama_layer=output_llama_layer,
                max_sequence_length=max_sequence_length,
                dropout_rate=dropout_rate,
                use_cache=use_category_cache
            )
            
            # Load category embeddings if provided
            if use_category_cache and category_embeddings_path is not None:
                self.load_category_embeddings(category_embeddings_path)
        else:
            # For external embeddings, still need prompt encoder components for weight loading
            self.prompt_encoder = None
            
            # Initialize prompt encoder components that exist in pretrained weights
            llama_hidden_size = 4096  # Default LLaMA hidden size
            self.text_projection = nn.Sequential(
                nn.Linear(llama_hidden_size, self.protein_encoder_hidden_size),
                nn.LayerNorm(self.protein_encoder_hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            )
            self.layer_norm = nn.LayerNorm(self.protein_encoder_hidden_size)
            self.dropout = nn.Dropout(dropout_rate)
            
            # Create sinusoidal positional encoding with buffer for BOS/EOS tokens
            import math
            
            # actual_max_length = max_sequence_length + 2 + 1  # +2 for BOS/EOS tokens from ESM, +1 for text token
            actual_max_length = max_sequence_length + 1  # +1 for text token

            pe = torch.zeros(actual_max_length, self.protein_encoder_hidden_size)
            position = torch.arange(0, actual_max_length).unsqueeze(1).float()
            div_term = torch.exp(torch.arange(0, self.protein_encoder_hidden_size, 2).float() *
                               -(math.log(10000.0) / self.protein_encoder_hidden_size))
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            self.register_buffer('position_encoding', pe)
        
        # Initialize position decoder
        self.position_decoder = PositionDecoder(
            hidden_size=self.protein_encoder_hidden_size,
            num_attention_heads=decoder_num_heads,
            num_layers=decoder_num_layers,
            intermediate_size=decoder_intermediate_size,
            dropout_rate=dropout_rate,
            num_self_attention_heads=decoder_num_self_attention_heads,
            use_sigmoid_head=use_sigmoid_head
        )
        
        # Loss functions
        self.ce_loss_fn = nn.CrossEntropyLoss(ignore_index=-1)
        self.smooth = 1e-3  # For Dice loss smoothing
    
    def load_category_embeddings(self, embeddings_path: str):
        """Load pre-computed category embeddings."""
        checkpoint = torch.load(embeddings_path, map_location=self.device)
        category_embeddings = checkpoint["category_embeddings"]
        
        for cat in category_embeddings:
            category_embeddings[cat] = category_embeddings[cat].to(self.device)
        
        self.prompt_encoder.set_category_embeddings_cache(category_embeddings)
        print(f"Loaded category embeddings from {embeddings_path}")
    
    def forward(
        self,
        protein_input_ids: Optional[torch.Tensor] = None,      # (batch_size, seq_len) - Optional when using external_esm_embeddings
        protein_attention_mask: Optional[torch.Tensor] = None, # (batch_size, seq_len) - Optional when using external_esm_embeddings
        text_input_ids: Optional[torch.Tensor] = None,         # (batch_size, text_len) - fallback
        text_attention_mask: Optional[torch.Tensor] = None,    # (batch_size, text_len) - fallback
        categories: Optional[list] = None,                     # List of category names - preferred
        point_positions: Optional[torch.Tensor] = None,       # (batch_size,)
        residue_labels: Optional[torch.Tensor] = None,        # (batch_size, seq_len) - unified residue-level labels
        external_prompt_embeddings: Optional[torch.Tensor] = None,  # (batch_size, 1, hidden_size) - Direct embedding input
        external_esm_embeddings: Optional[torch.Tensor] = None   # (batch_size, seq_len, hidden_size) - Direct ESM embeddings
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass of ProteinSAM model.

        Args:
            protein_input_ids: Tokenized protein sequences
            protein_attention_mask: Attention mask for proteins (optional when using external_esm_embeddings)
            text_input_ids: Tokenized functional region names
            text_attention_mask: Attention mask for text
            categories: List of category names for prompt encoding
            point_positions: Position prompts (optional)
            residue_labels: Ground truth residue-level labels for training
            external_prompt_embeddings: Pre-computed prompt embeddings to skip prompt encoding
            external_esm_embeddings: Pre-computed ESM embeddings to skip protein encoding
        Returns:
            Dictionary containing logits, predictions, and loss (if labels provided)
        """
        batch_size, seq_len = protein_attention_mask.shape  # 这个seq其实没啥用

        # Encode protein sequences - use external embeddings if provided
        if external_esm_embeddings is not None:
            seq_len = seq_len - 2  # Adjusted sequence length after removing BOS/EOS
            # Note: external embeddings should already have BOS/EOS removed
            protein_embeddings = external_esm_embeddings  # (batch_size, seq_len, hidden_size)
            
            seq_len = protein_embeddings.shape[1]
            if protein_attention_mask is None:
                protein_attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long, device=protein_embeddings.device)
            else:
                protein_attention_mask = protein_attention_mask[:, 1:-1]
        else:
            protein_embeddings = self.protein_encoder(
                protein_input_ids=protein_input_ids,
                attention_mask=protein_attention_mask
            )  # (batch_size, seq_len, hidden_size)

            protein_embeddings = protein_embeddings[:, 1:-1, :]  # Remove BOS/EOS tokens
            protein_attention_mask = protein_attention_mask[:, 1:-1]  # Adjust attention mask
            seq_len = seq_len - 2  # Adjusted sequence length after removing BOS/EOS
        
        # Get text token - either from external embeddings or internal encoder
        if self.use_external_embeddings and external_prompt_embeddings is not None:
            text_token_raw = external_prompt_embeddings  # (batch_size, 1, llama_hidden_size)
            # Reshape for projection if needed
            batch_size, seq_len_text, hidden_size = text_token_raw.shape
            text_token_flat = text_token_raw.view(batch_size * seq_len_text, hidden_size)

            # Apply text projection 
            text_token_projected = self.text_projection(text_token_flat)

            # Apply layer norm and dropout like in original prompt encoder
            text_token_projected = self.layer_norm(text_token_projected)
            text_token_projected = self.dropout(text_token_projected)

            # Reshape back
            text_token = text_token_projected.view(batch_size, seq_len_text, -1)
        else:
            # Use internal prompt encoder (original behavior)
            if self.prompt_encoder is None:
                raise ValueError("prompt_encoder is None but external_prompt_embeddings not provided")
            text_token = self.prompt_encoder(
                text_input_ids=text_input_ids,
                text_attention_mask=text_attention_mask,
                categories=categories
            )  # (batch_size, 1, hidden_size)
        
        # Combine protein embeddings and text token
        combined_embeddings = torch.cat([text_token, protein_embeddings], dim=1)  # (batch_size, seq_len+1, hidden_size)
        
        # Add positional encoding to combined embeddings
        if self.use_external_embeddings:
            # For external embeddings, use simple positional encoding
            seq_len_with_prompt = combined_embeddings.shape[1]
            # Use pre-computed position encoding (sequences are guaranteed to be within max length)
            pos_encodings = self.position_encoding[:seq_len_with_prompt].unsqueeze(0).expand(
                combined_embeddings.shape[0], -1, -1
            )
            # 保留第一个位置不变（text token位置），不对其添加位置编码
            pos_encodings[:, 0, :] = 0.0
            combined_embeddings_with_pos = combined_embeddings + pos_encodings
        else:
            # Use original positional encoding logic with point prompts
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
        )  # (batch_size, seq_len, 2) or (batch_size, seq_len, 1) if use_sigmoid_head
        
        # Get mask predictions
        if self.use_sigmoid_head:
            # N*1 logit: sigmoid to get probability, threshold at 0.5 for binary prediction
            mask_probs_fg = torch.sigmoid(mask_logits.squeeze(-1))  # (batch_size, seq_len)
            mask_predictions = (mask_probs_fg >= 0.5).long()
            mask_probs = mask_probs_fg  # expose raw sigmoid probs
        else:
            # N*2 logit: argmax for prediction, softmax for probs
            mask_predictions = torch.argmax(mask_logits, dim=-1)  # (batch_size, seq_len)
            mask_probs = F.softmax(mask_logits, dim=-1)  # (batch_size, seq_len, 2)
        
        # 将预测的01暂时转化为start和end，只有在训练好sam后inference时有用，这里实际上应该用更聪明一点的算法
        start_predictions, end_predictions = self._mask_to_positions(mask_predictions)
        
        outputs = {
            "mask_logits": mask_logits,
            "mask_predictions": mask_predictions,
            "mask_probs": mask_probs,
            "start_predictions": start_predictions,
            "end_predictions": end_predictions
        }
        
        # Compute loss if labels are provided
        if residue_labels is not None:
            # Unified loss computation using residue-level labels
            dice_loss = self._compute_dice_loss(mask_logits, residue_labels, protein_attention_mask)
            ce_loss = self._compute_ce_loss(mask_logits, residue_labels, protein_attention_mask)
            
            # Combine losses
            total_loss = 0.5 * dice_loss + 0.5 * ce_loss  # Balanced weighting for stability
            # total_loss = 0.5 * dice_loss
            # total_loss = 1.0 * dice_loss + 0.5 * ce_loss 
            
            outputs.update({
                "loss": total_loss,
                "dice_loss": dice_loss,
                "ce_loss": ce_loss,
                "residue_labels": residue_labels  # Add for debugging
            })
        
        return outputs
    
    def _morphological_1d(self, mask: torch.Tensor, kernel_size: int, op: str) -> torch.Tensor:
        """Apply 1D morphological opening or closing to a binary mask."""
        # mask: (seq_len,) float tensor with values 0/1
        m = mask.float().unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len)
        pad = kernel_size // 2
        if op == 'open':
            # erosion then dilation
            eroded = (torch.nn.functional.max_pool1d(-m, kernel_size, stride=1, padding=pad) * -1)
            result = torch.nn.functional.max_pool1d(eroded, kernel_size, stride=1, padding=pad)
        else:  # close
            # dilation then erosion
            dilated = torch.nn.functional.max_pool1d(m, kernel_size, stride=1, padding=pad)
            result = (torch.nn.functional.max_pool1d(-dilated, kernel_size, stride=1, padding=pad) * -1)
        return (result.squeeze(0).squeeze(0) >= 0.5).long()

    def _mask_to_positions(self, mask_predictions: torch.Tensor, morph_kernel_size: int = 3) -> Tuple[torch.Tensor, torch.Tensor]:

        batch_size, seq_len = mask_predictions.shape
        start_positions = torch.zeros(batch_size, dtype=torch.long, device=mask_predictions.device)
        end_positions = torch.zeros(batch_size, dtype=torch.long, device=mask_predictions.device)

        for b in range(batch_size):
            mask = mask_predictions[b]
            # Apply morphological opening (removes small noise) then closing (fills small gaps)
            mask = self._morphological_1d(mask, morph_kernel_size, 'open')
            mask = self._morphological_1d(mask, morph_kernel_size, 'close')
            # Find first and last positive positions
            positive_indices = torch.where(mask == 1)[0]
            if len(positive_indices) > 0:
                start_positions[b] = positive_indices[0]
                end_positions[b] = positive_indices[-1]
            else:
                # No positive predictions - default to position 0
                start_positions[b] = 0
                end_positions[b] = 0

        return start_positions, end_positions
    
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
    
    def _compute_dice_loss(
        self,
        mask_logits: torch.Tensor,
        mask_labels: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Compute Dice loss for binary segmentation."""
        if self.use_sigmoid_head:
            # N*1: sigmoid directly on the single logit
            pred_probs = torch.sigmoid(mask_logits.squeeze(-1))  # (batch_size, seq_len)
        else:
            # N*2: sigmoid on foreground logit (index 1)
            pred_probs = torch.sigmoid(mask_logits[:, :, 1])  # (batch_size, seq_len)

        true_mask = mask_labels.float()  # (batch_size, seq_len)

        # Apply attention mask
        if attention_mask is not None:
            pred_probs = pred_probs * attention_mask.float()
            true_mask = true_mask * attention_mask.float()

        # Compute Dice coefficient
        intersection = (pred_probs * true_mask).sum(dim=1)  # (batch_size,)
        union = pred_probs.sum(dim=1) + true_mask.sum(dim=1)  # (batch_size,)

        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice.mean()
    
    def _compute_ce_loss(
        self,
        mask_logits: torch.Tensor,
        mask_labels: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Compute loss for binary classification."""
        if self.use_sigmoid_head:
            # N*1: binary cross entropy with logits
            logits_flat = mask_logits.squeeze(-1).reshape(-1)  # (batch_size * seq_len,)
            labels_flat = mask_labels.float().reshape(-1)       # (batch_size * seq_len,)

            if attention_mask is not None:
                valid = attention_mask.reshape(-1) == 1
                logits_flat = logits_flat[valid]
                labels_flat = labels_flat[valid]

            # return F.binary_cross_entropy_with_logits(logits_flat, labels_flat)
            return F.binary_cross_entropy_with_logits(logits_flat, labels_flat, pos_weight=torch.tensor(3.0).to(logits_flat.device))  # Adjust pos_weight for class imbalance
        else:
            # N*2: cross entropy
            batch_size, seq_len, _ = mask_logits.shape
            mask_logits_flat = mask_logits.view(-1, 2)
            mask_labels_flat = mask_labels.view(-1)

            if attention_mask is not None:
                valid = attention_mask.reshape(-1) == 1
                mask_logits_flat = mask_logits_flat[valid]
                mask_labels_flat = mask_labels_flat[valid]

            return F.cross_entropy(mask_logits_flat, mask_labels_flat)
    
    def predict(
        self,
        protein_input_ids: Optional[torch.Tensor] = None,
        protein_attention_mask: Optional[torch.Tensor] = None,
        text_input_ids: Optional[torch.Tensor] = None,
        text_attention_mask: Optional[torch.Tensor] = None,
        point_positions: Optional[torch.Tensor] = None,
        return_probabilities: bool = False,
        external_prompt_embeddings: Optional[torch.Tensor] = None,
        external_esm_embeddings: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Inference method for mask prediction.

        Args:
            protein_input_ids: Tokenized protein sequences
            protein_attention_mask: Attention mask for proteins (optional when using external_esm_embeddings)
            text_input_ids: Tokenized functional region names
            text_attention_mask: Attention mask for text
            point_positions: Position prompts (optional)
            return_probabilities: Whether to return probabilities
            external_prompt_embeddings: External prompt embeddings (optional)
            external_esm_embeddings: External ESM embeddings (optional)

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
                point_positions=point_positions,
                external_prompt_embeddings=external_prompt_embeddings,
                external_esm_embeddings=external_esm_embeddings
            )
            
            predictions = {
                "mask_predictions": outputs["mask_predictions"],
                "start_predictions": outputs["start_predictions"],
                "end_predictions": outputs["end_predictions"]
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
        
        # Tokenize categories (only if not using external embeddings)
        text_input_ids = None
        text_attention_mask = None
        if not self.use_external_embeddings:
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
                'decoder_num_self_attention_heads': self.position_decoder.self_attention_layers[0].num_heads,
                'decoder_num_layers': len(self.position_decoder.cross_attention_layers),
                'decoder_intermediate_size': self.position_decoder.feed_forward_layers[0][0].out_features,
                'max_sequence_length': self.max_sequence_length,
                'output_llama_layer': self.output_llama_layer if self.use_external_embeddings else self.prompt_encoder.output_llama_layer
            }
        }, save_path)
    
        print(f"Model saved to {save_path}")
    
    def load_model(self, load_path: str):
        """Load model state dict."""
        checkpoint = torch.load(load_path, map_location=self.device)
        loaded_state_dict = checkpoint['model_state_dict']

        new_state_dict = {}
        current_model_keys = set(self.state_dict().keys())
        
        for name, param in loaded_state_dict.items():
            if self.use_external_embeddings and name.startswith('prompt_encoder.'):
                name = name.replace('prompt_encoder.', '')
            if name in current_model_keys:
                new_state_dict[name] = param
    
        self.load_state_dict(new_state_dict, strict=False)
        
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