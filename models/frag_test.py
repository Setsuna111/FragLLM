from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint
from transformers import Cache, PreTrainedModel
from transformers.generation.utils import GenerateOutput
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.models.esm.modeling_esm import EsmModel
from transformers.models.llama import LlamaForCausalLM
class FeedForwardNetwork(nn.Module):
    """General FFN module."""

    def __init__(self, emb_dim: int, dropout: float, ff_expansion: float = 2.0) -> None:
        super().__init__()
        self.ffn = nn.Sequential(
            nn.LayerNorm(emb_dim),
            nn.Linear(emb_dim, int(ff_expansion * emb_dim)),
            nn.GELU(),
            nn.Linear(int(ff_expansion * emb_dim), emb_dim),
            nn.Dropout(dropout),
        )

    def forward(self, hidden_states: Tensor) -> Tensor:
        out: Tensor = self.ffn(hidden_states)
        return out

class AttentionLayer(nn.Module):
    """Simple self attenition layer."""

    def __init__(self, emb_dim: int, num_heads: int, dropout: float, ff_expansion: int = 2) -> None:
        """Init."""
        super().__init__()
        self.attn = nn.MultiheadAttention(emb_dim, num_heads, dropout=dropout, batch_first=True)
        self.ffn = FeedForwardNetwork(emb_dim, dropout, ff_expansion=ff_expansion)
        self.output_layer_norm = nn.LayerNorm(emb_dim)

    def forward(self, hidden_states: Tensor) -> Tensor:
        """self-attend."""
        residuals = hidden_states
        hidden_states, _ = self.attn(hidden_states, hidden_states, hidden_states)
        hidden_states = self.ffn(residuals + hidden_states) + residuals
        out: Tensor = self.output_layer_norm(hidden_states)
        return out
    
class PerceiverLayer(nn.Module):
    """Simple Perceiver layer."""
    def __init__(self, emb_dim: int, num_heads: int, dropout: float) -> None:
        """Init."""
        super().__init__()
        self.attn = nn.MultiheadAttention(emb_dim, num_heads, dropout=dropout, batch_first=True)
        self.ffn = FeedForwardNetwork(emb_dim, dropout, ff_expansion=0.5)
        self.output_layer_norm = nn.LayerNorm(emb_dim)

    def forward(self, latents: Tensor, hidden_states: Tensor) -> Tensor:
        """Cross-attend hidden_states and latents and self-attend latents."""
        residuals = latents
        hidden_latents = torch.cat((hidden_states, latents), dim=-2)
        latents, _ = self.attn(latents, hidden_latents, hidden_latents)
        latents = self.ffn(residuals + latents) + residuals
        out: Tensor = self.output_layer_norm(latents)
        return out
class Perceiver(nn.Module):
    """Perceiver module that handles dim mismatch."""

    def __init__(
        self, input_dim: int, latent_size: int, output_dim: int, num_heads: int, num_layers: int, dropout: float
    ) -> None:
        """
        """
        super().__init__()
        self.latents = nn.Parameter(torch.randn(latent_size, input_dim))
        self.latent_layer_norm = nn.LayerNorm(input_dim)
        self.perceiver = PerceiverLayer(input_dim, num_heads, dropout)
        self.self_attention_layers = nn.ModuleList(
            [AttentionLayer(input_dim, num_heads, dropout, ff_expansion=1) for _ in range(num_layers - 1)]
        )
        self.output_proj = nn.Linear(input_dim, output_dim, bias=False)
        self.out_layer_norm = nn.LayerNorm(output_dim)

    def forward(self, hidden_states: Tensor) -> Tensor:
        latents = self.latents
        latents = self.latent_layer_norm(latents)
        latents = self.perceiver(latents, hidden_states)
        for layer in self.self_attention_layers:
            latents = layer(latents)
        out = self.output_proj(latents)
        out: Tensor = self.out_layer_norm(out)
        return out
# Q-former for fragment
class FragmentLayer(nn.Module):
    def __init__(
        self,
        protein_emb_dim: int,
        text_emb_dim: int,
        perceiver_latent_size: int, #  length of the fragment hidden states
        num_perceiver_heads: int,
        num_perceiver_layers: int,
        enable_gradient_checkpointing: bool,
        dropout: float,
    ) -> None:
        super(FragmentLayer, self).__init__()
        self.enable_gradient_checkpointing = enable_gradient_checkpointing
        self.protein_layer_norm = nn.LayerNorm(protein_emb_dim)
        self.perceiver = Perceiver(
            protein_emb_dim, perceiver_latent_size, text_emb_dim, num_perceiver_heads, num_perceiver_layers, dropout
        )

    def forward(
        self,
        position_refs: List[List[int]],
        encoder_hidden_states: Tensor,
        encoder_attention_mask: Tensor,
        **kwargs: Any,
    ) -> Union[Tuple[Tensor], Optional[Tuple[Tensor, Tuple[Tensor, ...]]]]:
        assert encoder_hidden_states is not None
        encoder_hidden_states = self.protein_layer_norm(encoder_hidden_states)
        frag_latents = [None] * len(position_refs)
        for i, (protein_emb, position_ref, encoder_mask) in enumerate(zip(encoder_hidden_states, position_refs, encoder_attention_mask)):
            if position_ref is not None:
                frag_hidden_states = protein_emb[encoder_mask][position_ref[0]:position_ref[1]]
                if self.training and self.enable_gradient_checkpointing:
                    frag_latents[i] = checkpoint(self.perceiver, frag_hidden_states)
                else:
                    frag_latents[i] = self.perceiver(frag_hidden_states)
            else:
                frag_latents[i] = None
        return frag_latents
if __name__ == "__main__":
    Fragperceiver = FragmentLayer(
    protein_emb_dim=1280,
    text_emb_dim=768,
    perceiver_latent_size=1,
    num_perceiver_heads=16,
    num_perceiver_layers=12,
    enable_gradient_checkpointing=True,
        dropout=0.05,
    )
    position_refs = [[5, 20], None]
    encoder_hidden_states = torch.randn(2, 1021, 1280)
    encoder_attention_mask = torch.ones((2, 1021),dtype=torch.long)
    frag_latents = Fragperceiver(position_refs, encoder_hidden_states, encoder_attention_mask)
    import pdb; pdb.set_trace()