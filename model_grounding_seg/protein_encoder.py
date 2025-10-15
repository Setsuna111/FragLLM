"""
Protein encoder component for ProteinSAM model.
Uses ESM2 for protein sequence encoding.
"""

import torch
import torch.nn as nn
from transformers import EsmModel, EsmTokenizer
from typing import Optional, Tuple


class ProteinEncoder(nn.Module):
    """
    Protein encoder using ESM2.
    ESM2 parameters are frozen during training.
    """
    
    def __init__(
        self, 
        esm_model_path: str,
        device: str = "cuda"
    ):
        super().__init__()
        
        # Load ESM2 model and tokenizer
        self.esm_model = EsmModel.from_pretrained(esm_model_path)
        self.tokenizer = EsmTokenizer.from_pretrained(esm_model_path)
        
        # Freeze ESM2 parameters
        for param in self.esm_model.parameters():
            param.requires_grad = False
            
        self.hidden_size = self.esm_model.config.hidden_size
        self.device = device
        
    def forward(
        self, 
        protein_input_ids: torch.Tensor,  # (batch_size, seq_len)
        attention_mask: torch.Tensor      # (batch_size, seq_len)
    ) -> torch.Tensor:                    # (batch_size, seq_len, hidden_size)
        """
        Encode protein sequences using ESM2.
        
        Args:
            protein_input_ids: Tokenized protein sequences
            attention_mask: Attention mask for protein sequences
            
        Returns:
            Protein embeddings from ESM2 last hidden state
        """
        with torch.no_grad():  # ESM2 parameters are frozen
            outputs = self.esm_model(
                input_ids=protein_input_ids,
                attention_mask=attention_mask,
                return_dict=True
            )
            
        return outputs.last_hidden_state  # (batch_size, seq_len, hidden_size)