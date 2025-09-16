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
        llama_model_path: Optional[str] = None,
        output_llama_layer: int = 16,
        decoder_num_heads: int = 8,
        decoder_num_layers: int = 2,
        decoder_intermediate_size: int = 512,
        max_sequence_length: int = 2048,
        dropout_rate: float = 0.1,
        device: str = "cuda",
        use_category_cache: bool = True,
        category_embeddings_path: Optional[str] = None,
        use_external_embeddings: bool = False  # New parameter for direct embedding input
    ):
        super().__init__()
        
        self.device = device
        self.max_sequence_length = max_sequence_length
        self.use_category_cache = use_category_cache
        self.use_external_embeddings = use_external_embeddings
        
        # Initialize protein encoder
        self.protein_encoder = ProteinEncoder(
            esm_model_path=esm_model_path,
            device=device
        )
        
        # Initialize prompt encoder only if not using external embeddings
        if not use_external_embeddings:
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
        else:
            # For external embeddings, still need prompt encoder components for weight loading
            self.prompt_encoder = None
            self.protein_hidden_size = self.protein_encoder.hidden_size
            self.output_llama_layer = output_llama_layer
            
            # Initialize prompt encoder components that exist in pretrained weights
            llama_hidden_size = 4096  # Default LLaMA hidden size
            self.text_projection = nn.Sequential(
                nn.Linear(llama_hidden_size, self.protein_hidden_size),
                nn.LayerNorm(self.protein_hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            )
            self.layer_norm = nn.LayerNorm(self.protein_hidden_size)
            self.dropout = nn.Dropout(dropout_rate)
            
            # Create sinusoidal positional encoding
            import math
            pe = torch.zeros(max_sequence_length, self.protein_hidden_size)
            position = torch.arange(0, max_sequence_length).unsqueeze(1).float()
            div_term = torch.exp(torch.arange(0, self.protein_hidden_size, 2).float() *
                               -(math.log(10000.0) / self.protein_hidden_size))
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            self.register_buffer('position_encoding', pe)
        
        # Initialize position decoder
        self.position_decoder = PositionDecoder(
            hidden_size=self.protein_encoder.hidden_size,
            num_attention_heads=decoder_num_heads,
            num_layers=decoder_num_layers,
            intermediate_size=decoder_intermediate_size,
            dropout_rate=dropout_rate
        )
        
        # Loss function
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-1)
    
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
        end_labels: Optional[torch.Tensor] = None,            # (batch_size,)
        external_prompt_embeddings: Optional[torch.Tensor] = None  # (batch_size, 1, hidden_size) - Direct embedding input
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
        
        # Get text token - either from external embeddings or internal encoder
        if self.use_external_embeddings and external_prompt_embeddings is not None:
            # Use external prompt embeddings, apply text_projection for dimension matching
            text_token_raw = external_prompt_embeddings  # (batch_size, 1, llama_hidden_size)
            # Reshape for projection if needed
            batch_size, seq_len, hidden_size = text_token_raw.shape
            text_token_flat = text_token_raw.view(batch_size * seq_len, hidden_size)
            # Apply text projection 
            text_token_projected = self.text_projection(text_token_flat)
            # Apply layer norm and dropout like in original prompt encoder
            text_token_projected = self.layer_norm(text_token_projected)
            text_token_projected = self.dropout(text_token_projected)
            # Reshape back
            text_token = text_token_projected.view(batch_size, seq_len, -1)
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
            # Handle case where sequence is longer than position_encoding buffer
            if seq_len_with_prompt > self.position_encoding.shape[0]:
                # Extend position encoding dynamically
                import math
                device = self.position_encoding.device
                dtype = self.position_encoding.dtype
                pe = torch.zeros(seq_len_with_prompt, self.protein_hidden_size, device=device, dtype=dtype)
                position = torch.arange(0, seq_len_with_prompt, device=device).unsqueeze(1).float()
                div_term = torch.exp(torch.arange(0, self.protein_hidden_size, 2, device=device).float() *
                                   -(math.log(10000.0) / self.protein_hidden_size))
                pe[:, 0::2] = torch.sin(position * div_term)
                pe[:, 1::2] = torch.cos(position * div_term)
                pos_encodings = pe.unsqueeze(0).expand(combined_embeddings.shape[0], -1, -1)
            else:
                pos_encodings = self.position_encoding[:seq_len_with_prompt].unsqueeze(0).expand(
                    combined_embeddings.shape[0], -1, -1
                )
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
        
        # Decode positions using protein embeddings with position encoding and text token
        position_logits = self.position_decoder(
            protein_embeddings=protein_embeddings_with_pos,
            prompt_embeddings=text_token_with_pos,
            protein_attention_mask=protein_attention_mask
        )  # (batch_size, seq_len, 2)
        
        start_logits = position_logits[:, :, 0]  # (batch_size, seq_len)
        end_logits = position_logits[:, :, 1]    # (batch_size, seq_len)
        
        # Predict positions
        start_predictions = torch.argmax(start_logits, dim=1)  # (batch_size,)
        end_predictions = torch.argmax(end_logits, dim=1)      # (batch_size,)
        
        outputs = {
            "start_logits": start_logits,
            "end_logits": end_logits,
            "start_predictions": start_predictions,
            "end_predictions": end_predictions,
            "position_logits": position_logits
        }
        
        # Compute loss if labels are provided
        if start_labels is not None and end_labels is not None:
            start_loss = self.loss_fn(start_logits, start_labels)
            end_loss = self.loss_fn(end_logits, end_labels)
            total_loss = start_loss + end_loss
            
            outputs.update({
                "loss": total_loss,
                "start_loss": start_loss,
                "end_loss": end_loss
            })
        
        return outputs
    
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
        Inference method for position prediction.
        
        Args:
            protein_input_ids: Tokenized protein sequences
            protein_attention_mask: Attention mask for proteins
            text_input_ids: Tokenized functional region names
            text_attention_mask: Attention mask for text
            point_positions: Position prompts (optional)
            return_probabilities: Whether to return probabilities
            
        Returns:
            Predicted positions and optionally probabilities
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
                "start_predictions": outputs["start_predictions"],
                "end_predictions": outputs["end_predictions"]
            }
            
            if return_probabilities:
                start_probs = F.softmax(outputs["start_logits"], dim=1)
                end_probs = F.softmax(outputs["end_logits"], dim=1)
                predictions.update({
                    "start_probabilities": start_probs,
                    "end_probabilities": end_probs
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
        
        # Load parameters, handling prompt_encoder weights for external embeddings mode
        current_state_dict = self.state_dict()
        loaded_keys = []
        
        for name, param in model_state_dict.items():
            # Handle prompt_encoder weights when using external embeddings
            if self.use_external_embeddings and name.startswith('prompt_encoder.'):
                # Map prompt_encoder weights to direct components
                component_name = name.replace('prompt_encoder.', '')
                if component_name in current_state_dict:
                    current_state_dict[component_name].copy_(param)
                    loaded_keys.append(component_name)
            elif name in current_state_dict:
                current_state_dict[name].copy_(param)
                loaded_keys.append(name)
        
        print(f"Model loaded from {load_path}, loaded {len(loaded_keys)} parameters")
    
    def get_trainable_parameters(self):
        """Get count of trainable parameters."""
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.parameters())
        
        return {
            "trainable_parameters": trainable_params,
            "total_parameters": total_params,
            "trainable_percentage": trainable_params / total_params * 100
        }