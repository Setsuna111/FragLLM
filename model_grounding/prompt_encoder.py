"""
Prompt encoder component for ProteinSAM model.
Encodes text prompts and position prompts for protein functional region grounding.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import LlamaModel, LlamaTokenizer
from typing import Optional, Dict, Any
import math


class PromptEncoder(nn.Module):
    """
    Prompt encoder that combines text prompts and position prompts.
    Uses pre-computed category embeddings to avoid loading Llama during training.
    """
    
    def __init__(
        self,
        llama_model_path: str,
        protein_hidden_size: int,
        output_llama_layer: int = 16,
        max_sequence_length: int = 2048,
        dropout_rate: float = 0.1,
        use_cache: bool = True
    ):
        super().__init__()
        
        self.llama_model_path = llama_model_path
        self.output_llama_layer = output_llama_layer
        self.use_cache = use_cache
        self.protein_hidden_size = protein_hidden_size
        
        # Only load Llama model if not using cache or for initialization
        self.llama_model = None
        self.tokenizer = None
        self.llama_hidden_size = 4096  # Default Llama hidden size
        
        if not use_cache:
            self._load_llama_model()
        
        # Category embeddings cache - will be populated during preprocessing
        self.category_embeddings_cache = nn.Parameter(
            torch.empty(0, self.llama_hidden_size), requires_grad=False
        )
        self.category_to_idx = {}  # Map category names to indices
        
        # Learnable projection from Llama hidden size to protein hidden size
        self.text_projection = nn.Sequential(
            nn.Linear(self.llama_hidden_size, protein_hidden_size),
            nn.LayerNorm(protein_hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # Sinusoidal position encoding for point prompts and sequence positions
        self.max_sequence_length = max_sequence_length
        self.register_buffer('position_encoding', self._create_sinusoidal_encoding(max_sequence_length, protein_hidden_size))
        
        # Layer norm and dropout
        self.layer_norm = nn.LayerNorm(protein_hidden_size)
        self.dropout = nn.Dropout(dropout_rate)
    
    def _load_llama_model(self):
        """Load Llama model and tokenizer (only when needed)."""
        from transformers import LlamaModel, LlamaTokenizer
        
        self.llama_model = LlamaModel.from_pretrained(self.llama_model_path)
        self.tokenizer = LlamaTokenizer.from_pretrained(self.llama_model_path)
        
        # Set pad token if not exists
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = '<|reserved_special_token_0|>'
            
        # Freeze Llama parameters
        for param in self.llama_model.parameters():
            param.requires_grad = False
            
        self.llama_hidden_size = self.llama_model.config.hidden_size
    
    def _create_sinusoidal_encoding(self, max_len: int, d_model: int) -> torch.Tensor:
        """
        Create sinusoidal positional encoding matrix.
        
        Args:
            max_len: Maximum sequence length
            d_model: Model dimension
            
        Returns:
            Positional encoding matrix (max_len, d_model)
        """
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                           -(math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        return pe
        
    def set_category_embeddings_cache(self, category_embeddings: Dict[str, torch.Tensor]):
        """
        Set pre-computed category embeddings cache.
        
        Args:
            category_embeddings: Dictionary mapping category names to embeddings
        """
        categories = list(category_embeddings.keys())
        embeddings = torch.stack([category_embeddings[cat] for cat in categories])
        
        # Update cache and mapping
        self.category_embeddings_cache = nn.Parameter(embeddings, requires_grad=False)
        self.category_to_idx = {cat: i for i, cat in enumerate(categories)}
        
        print(f"Loaded {len(categories)} category embeddings into cache")
    
    def encode_text_from_cache(self, categories: list) -> torch.Tensor:
        """
        Get text embeddings from pre-computed cache.
        
        Args:
            categories: List of category names
            
        Returns:
            Text embeddings (batch_size, llama_hidden_size)
        """
        indices = []
        for cat in categories:
            if cat not in self.category_to_idx:
                raise ValueError(f"Category '{cat}' not found in cache. Available: {list(self.category_to_idx.keys())}")
            indices.append(self.category_to_idx[cat])
        
        indices = torch.tensor(indices, device=self.category_embeddings_cache.device)
        return self.category_embeddings_cache[indices]
    
    def encode_text(
        self, 
        text_input_ids: torch.Tensor,     # (batch_size, text_len)
        text_attention_mask: torch.Tensor # (batch_size, text_len)
    ) -> torch.Tensor:                    # (batch_size, protein_hidden_size)
        """
        Encode text using Llama model (fallback when cache not available).
        """
        if self.llama_model is None:
            self._load_llama_model()
            
        with torch.no_grad():
            outputs = self.llama_model(
                input_ids=text_input_ids,
                attention_mask=text_attention_mask,
                output_hidden_states=True,
                return_dict=True
            )
            
            # Get embeddings from specified layer and mean pool
            hidden_states = outputs.hidden_states[self.output_llama_layer]
            masked_embeddings = hidden_states * text_attention_mask.unsqueeze(-1)
            sum_embeddings = masked_embeddings.sum(dim=1)
            valid_tokens = text_attention_mask.sum(dim=1, keepdim=True)
            text_embeddings = sum_embeddings / valid_tokens
            
        return self.text_projection(text_embeddings)
    
    def get_text_token(
        self,
        text_input_ids: Optional[torch.Tensor] = None,       # (batch_size, text_len) - for fallback
        text_attention_mask: Optional[torch.Tensor] = None,  # (batch_size, text_len) - for fallback
        categories: Optional[list] = None                    # List of category names - for cache
    ) -> torch.Tensor:                                       # (batch_size, protein_hidden_size)
        """
        Get text token embeddings (without position encoding).
        
        Args:
            text_input_ids: Tokenized functional region names (fallback)
            text_attention_mask: Attention mask for text (fallback)  
            categories: List of category names (preferred, uses cache)
            
        Returns:
            Text token embeddings projected to protein hidden size
        """
        if categories is not None and len(self.category_to_idx) > 0:
            raw_embeddings = self.encode_text_from_cache(categories)
            return self.text_projection(raw_embeddings)
        elif text_input_ids is not None and text_attention_mask is not None:
            return self.encode_text(text_input_ids, text_attention_mask)
        else:
            raise ValueError("Must provide either 'categories' or 'text_input_ids/text_attention_mask'")
    
    def get_positional_encoding(
        self, 
        embeddings: torch.Tensor,                          # (batch_size, seq_len, hidden_size)
        point_positions: Optional[torch.Tensor] = None,    # (batch_size,) - position for prompt token
        prompt_token_idx: int = 0                          # Index of prompt token in sequence
    ) -> torch.Tensor:                                     # (batch_size, seq_len, hidden_size)
        """
        Add positional encoding to embeddings, with special handling for prompt token.
        
        Args:
            embeddings: Input embeddings (protein + prompt tokens)
            point_positions: Position prompts for the text token
            prompt_token_idx: Index of the prompt token in the sequence (usually 0)
            
        Returns:
            Embeddings with positional encoding
        """
        batch_size, seq_len, hidden_size = embeddings.shape
        device = embeddings.device
        
        # Create position indices for the sequence
        position_ids = torch.arange(seq_len, device=device, dtype=torch.long).unsqueeze(0).expand(batch_size, -1).clone()
        
        # If point_positions provided, replace prompt token position
        if point_positions is not None:
            # Clamp positions to valid range and ensure long dtype
            point_positions = torch.clamp(point_positions, 0, self.max_sequence_length - 1).long()
            position_ids[:, prompt_token_idx] = point_positions
        else:
            # Use default position (0) for prompt token
            position_ids[:, prompt_token_idx] = 0
        
        # Get position embeddings from sinusoidal encoding
        # Ensure position_ids are within valid range and convert to long
        position_ids = torch.clamp(position_ids, 0, self.position_encoding.size(0) - 1).long()
        pos_embeddings = self.position_encoding[position_ids]  # (batch_size, seq_len, hidden_size)
        
        # Add position embeddings to input embeddings
        return embeddings + pos_embeddings
    
    def forward(
        self,
        text_input_ids: Optional[torch.Tensor] = None,       # (batch_size, text_len) - for fallback
        text_attention_mask: Optional[torch.Tensor] = None,  # (batch_size, text_len) - for fallback
        categories: Optional[list] = None                    # List of category names - for cache
    ) -> torch.Tensor:                                       # (batch_size, 1, protein_hidden_size)
        """
        Forward pass of prompt encoder - returns single text token.
        
        Args:
            text_input_ids: Tokenized functional region names (fallback)
            text_attention_mask: Attention mask for text (fallback)  
            categories: List of category names (preferred, uses cache)
            
        Returns:
            Single text token embedding (without position encoding - to be added later)
        """
        # Get text token embedding
        text_token = self.get_text_token(
            text_input_ids=text_input_ids,
            text_attention_mask=text_attention_mask,
            categories=categories
        )  # (batch_size, protein_hidden_size)
        
        # Apply layer norm and dropout
        text_token = self.layer_norm(text_token)
        text_token = self.dropout(text_token)
        
        # Add sequence dimension for consistency
        text_token = text_token.unsqueeze(1)  # (batch_size, 1, protein_hidden_size)
        
        return text_token
    