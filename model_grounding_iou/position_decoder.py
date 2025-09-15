"""
Position decoder component for ProteinSAM model.
Lightweight decoder that fuses protein and prompt encodings to predict start/end positions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import math


class CrossAttention(nn.Module):
    """
    Cross-attention mechanism for fusing protein and prompt encodings.
    """
    
    def __init__(self, hidden_size: int, num_heads: int = 8, dropout_rate: float = 0.1):
        super().__init__()
        assert hidden_size % num_heads == 0
        
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = math.sqrt(self.head_dim)
        
        self.query_proj = nn.Linear(hidden_size, hidden_size)
        self.key_proj = nn.Linear(hidden_size, hidden_size)
        self.value_proj = nn.Linear(hidden_size, hidden_size)
        self.output_proj = nn.Linear(hidden_size, hidden_size)
        
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(
        self,
        query: torch.Tensor,    # (batch_size, query_len, hidden_size)
        key: torch.Tensor,      # (batch_size, key_len, hidden_size)
        value: torch.Tensor,    # (batch_size, value_len, hidden_size)
        attention_mask: Optional[torch.Tensor] = None  # (batch_size, key_len)
    ) -> torch.Tensor:          # (batch_size, query_len, hidden_size)
        
        batch_size, query_len, _ = query.shape
        key_len = key.shape[1]
        
        # Project to Q, K, V
        Q = self.query_proj(query)  # (batch_size, query_len, hidden_size)
        K = self.key_proj(key)      # (batch_size, key_len, hidden_size)
        V = self.value_proj(value)  # (batch_size, value_len, hidden_size)
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size, query_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, key_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, key_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        # Apply attention mask if provided
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(1).unsqueeze(1)  # (batch_size, 1, 1, key_len)
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Apply softmax
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        attended = torch.matmul(attention_weights, V)  # (batch_size, num_heads, query_len, head_dim)
        
        # Reshape and project output
        attended = attended.transpose(1, 2).contiguous().view(batch_size, query_len, self.hidden_size)
        output = self.output_proj(attended)
        
        return output


class PositionDecoder(nn.Module):
    """
    Lightweight position decoder for predicting start and end positions of functional regions.
    """
    
    def __init__(
        self, 
        hidden_size: int,
        num_attention_heads: int = 8,
        num_layers: int = 2,
        intermediate_size: int = 512,
        dropout_rate: float = 0.1
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Cross-attention layers for fusing protein and prompt encodings
        self.cross_attention_layers = nn.ModuleList([
            CrossAttention(hidden_size, num_attention_heads, dropout_rate)
            for _ in range(num_layers)
        ])
        
        # Feed-forward networks
        self.feed_forward_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, intermediate_size),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(intermediate_size, hidden_size),
                nn.Dropout(dropout_rate)
            )
            for _ in range(num_layers)
        ])
        
        # Layer norms
        self.layer_norms_1 = nn.ModuleList([
            nn.LayerNorm(hidden_size) for _ in range(num_layers)
        ])
        self.layer_norms_2 = nn.ModuleList([
            nn.LayerNorm(hidden_size) for _ in range(num_layers)
        ])
        
        # Binary mask prediction head
        self.mask_head = nn.Sequential(
            nn.Linear(hidden_size, intermediate_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(intermediate_size, 2)  # 2 for background/foreground probabilities
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(
        self,
        protein_embeddings: torch.Tensor,    # (batch_size, seq_len, hidden_size)
        prompt_embeddings: torch.Tensor,     # (batch_size, 1, hidden_size)
        protein_attention_mask: torch.Tensor # (batch_size, seq_len)
    ) -> torch.Tensor:  # (batch_size, seq_len, 2)
        """
        Forward pass of position decoder.
        
        Args:
            protein_embeddings: Encoded protein sequences
            prompt_embeddings: Encoded prompts (text + position)
            protein_attention_mask: Attention mask for protein sequences
            
        Returns:
            Binary mask logits for each sequence position (background/foreground probabilities)
        """
        batch_size, seq_len, _ = protein_embeddings.shape
        
        # Use protein embeddings as initial hidden states
        hidden_states = protein_embeddings
        
        # Apply transformer layers
        for i in range(self.num_layers):
            # Cross-attention: protein sequence attends to prompt
            residual = hidden_states
            attended = self.cross_attention_layers[i](
                query=hidden_states,
                key=prompt_embeddings,
                value=prompt_embeddings
            )
            hidden_states = self.layer_norms_1[i](residual + attended)
            
            # Feed-forward
            residual = hidden_states
            ff_output = self.feed_forward_layers[i](hidden_states)
            hidden_states = self.layer_norms_2[i](residual + ff_output)
        
        # Predict binary mask
        mask_logits = self.mask_head(hidden_states)  # (batch_size, seq_len, 2)
        
        # Apply attention mask to logits
        if protein_attention_mask is not None:
            mask = protein_attention_mask.unsqueeze(-1)  # (batch_size, seq_len, 1)
            mask_logits = mask_logits.masked_fill(mask == 0, -1e9)
        
        return mask_logits
    
    def predict_mask(
        self,
        protein_embeddings: torch.Tensor,
        prompt_embeddings: torch.Tensor,
        protein_attention_mask: torch.Tensor,
        return_probabilities: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Predict binary mask for functional regions.
        
        Args:
            protein_embeddings: Encoded protein sequences
            prompt_embeddings: Encoded prompts
            protein_attention_mask: Attention mask for protein sequences
            return_probabilities: Whether to return softmax probabilities
            
        Returns:
            Predicted binary mask and optionally probabilities
        """
        mask_logits = self.forward(
            protein_embeddings, 
            prompt_embeddings, 
            protein_attention_mask
        )  # (batch_size, seq_len, 2)
        
        # Apply softmax across class dimension (background/foreground)
        mask_probs = F.softmax(mask_logits, dim=-1)  # (batch_size, seq_len, 2)
        
        # Get binary mask (argmax of background vs foreground)
        binary_mask = torch.argmax(mask_logits, dim=-1)  # (batch_size, seq_len)
        
        probabilities = None
        if return_probabilities:
            probabilities = mask_probs
        
        return binary_mask, probabilities