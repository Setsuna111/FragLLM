from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
from torch import Tensor, nn
from transformers.models.esm.modeling_esm import EsmModel
from transformers import AutoModelForCausalLM
import sys
import os
import json
from pathlib import Path
from model_grounding_segformer.protein_sam import ProteinSAM
try:
    from .hierarchical_fragment_adapter import HierarchicalFragmentAdapter
except ImportError:
    from models.hierarchical_fragment_adapter import HierarchicalFragmentAdapter


def load_protein_sam_params_with_overrides(checkpoint_path: str, esm_model_path: str) -> Dict[str, Any]:
    """
    Load ProteinSAM parameters from training and override special parameters for current use.
    """
    params_file = os.path.join(os.path.dirname(checkpoint_path), "protein_sam_init_params.json")
    assert os.path.exists(params_file), f"ProteinSAM checkpoint not found at {checkpoint_path}"
    with open(params_file, 'r') as f:
        params = json.load(f)
    
    params['use_category_cache'] = False  # Not using category cache
    params['use_external_embeddings'] = True  # Use external embeddings from LLM
    params['use_external_esm'] = True  # Use external ESM embeddings
    params['llama_model_path'] = None  # No LLaMA needed
    # params['esm_model_path'] = None  # No ESM model needed, 但还是得传进去获取维度信息
    params['esm_model_path'] = esm_model_path
    params['category_embeddings_path'] = None  # No category embeddings
    params['device'] = 'cpu'  # Will be moved to correct device later
    
    print(f"Loading ProteinSAM parameters from {params_file}")
    return params


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

class MultiScalePerceiverLayer(nn.Module):
    """Multi-scale Perceiver layer that handles global and fragment features."""
    def __init__(self, emb_dim: int, num_heads: int, dropout: float,
                 protein_emb_dim: Optional[int] = None, text_emb_dim: Optional[int] = None) -> None:
        super().__init__()
        self.emb_dim = emb_dim
        self.protein_emb_dim = protein_emb_dim or emb_dim
        self.text_emb_dim = text_emb_dim or emb_dim

        # Two independent paths with different dimensions
        # Global path (text_emb_dim)
        self.global_attn = nn.MultiheadAttention(self.text_emb_dim, num_heads, dropout=dropout, batch_first=True)
        self.global_ffn = FeedForwardNetwork(self.text_emb_dim, dropout, ff_expansion=0.5)
        self.global_layer_norm = nn.LayerNorm(self.text_emb_dim)
        self.global_linear = nn.Linear(self.text_emb_dim, emb_dim)

        # Fragment path (protein_emb_dim)
        self.fragment_attn = nn.MultiheadAttention(self.protein_emb_dim, num_heads, dropout=dropout, batch_first=True)
        self.fragment_ffn = FeedForwardNetwork(self.protein_emb_dim, dropout, ff_expansion=0.5)
        self.fragment_layer_norm = nn.LayerNorm(self.protein_emb_dim)
        self.fragment_linear = nn.Linear(self.protein_emb_dim, emb_dim)

        # Final fusion
        self.global_gate = nn.Linear(emb_dim, 1)
        self.fragment_gate = nn.Linear(emb_dim, 1)
        self.output_layer_norm = nn.LayerNorm(emb_dim)

    def forward(self, latents_global: Tensor, latents_fragment: Tensor,
                global_features: Tensor, fragment_features: Tensor) -> Tensor:
        """Multi-scale cross-attention with independent processing paths.

        Args:
            latents_global: [latent_size, text_emb_dim] - learnable latents for global path
            latents_fragment: [latent_size, protein_emb_dim] - learnable latents for fragment path
            global_features: [1, text_emb_dim] - global feature from adapter output
            fragment_features: [fragment_len, protein_emb_dim] - residue-level features from ESM
        """
        # Global path: cross attention + FFN + residual (text_emb_dim)
        global_residual = latents_global
        global_attended, _ = self.global_attn(latents_global, global_features, global_features)
        global_attended = self.global_ffn(global_residual + global_attended) + global_residual
        global_attended = self.global_layer_norm(global_attended)

        # Fragment path: cross attention + FFN + residual (protein_emb_dim)
        fragment_residual = latents_fragment
        fragment_attended, _ = self.fragment_attn(latents_fragment, fragment_features, fragment_features)
        fragment_attended = self.fragment_ffn(fragment_residual + fragment_attended) + fragment_residual
        fragment_attended = self.fragment_layer_norm(fragment_attended)

        # Linear projection to unified dimension
        global_projected = self.global_linear(global_attended)    # [latent_size, emb_dim]
        fragment_projected = self.fragment_linear(fragment_attended)  # [latent_size, emb_dim]

        # Adaptive gating for weighted combination
        global_weight = torch.sigmoid(self.global_gate(global_projected))
        fragment_weight = torch.sigmoid(self.fragment_gate(fragment_projected))

        # Normalize weights
        total_weight = global_weight + fragment_weight
        global_weight = global_weight / (total_weight + 1e-8)
        fragment_weight = fragment_weight / (total_weight + 1e-8)

        # Final weighted combination
        latents = global_weight * global_projected + fragment_weight * fragment_projected
        out: Tensor = self.output_layer_norm(latents)
        return out

class MultiScalePerceiver(nn.Module):
    """Multi-scale Perceiver that integrates global protein and fragment features."""

    def __init__(
        self, input_dim: int, latent_size: int, output_dim: int, num_heads: int, num_layers: int, dropout: float,
        protein_emb_dim: Optional[int] = None, text_emb_dim: Optional[int] = None
    ) -> None:
        super().__init__()
        self.protein_emb_dim = protein_emb_dim or input_dim
        self.text_emb_dim = text_emb_dim or input_dim

        # Two sets of learnable latents with different dimensions
        self.latents_global = nn.Parameter(torch.randn(latent_size, self.text_emb_dim))
        self.latents_fragment = nn.Parameter(torch.randn(latent_size, self.protein_emb_dim))
        self.global_latent_norm = nn.LayerNorm(self.text_emb_dim)
        self.fragment_latent_norm = nn.LayerNorm(self.protein_emb_dim)

        # First layer uses multi-scale perceiver
        self.multi_scale_perceiver = MultiScalePerceiverLayer(
            input_dim, num_heads, dropout, self.protein_emb_dim, self.text_emb_dim
        )

        # Subsequent layers use standard self-attention
        self.self_attention_layers = nn.ModuleList(
            [AttentionLayer(input_dim, num_heads, dropout, ff_expansion=1) for _ in range(num_layers - 1)]
        )

        self.output_proj = nn.Linear(input_dim, output_dim, bias=False)
        self.out_layer_norm = nn.LayerNorm(output_dim)

    def forward(self, fragment_features: Tensor, global_feature: Tensor) -> Tensor:
        """
        Args:
            fragment_features: [fragment_len, protein_emb_dim] - residue-level features from ESM
            global_feature: [1, text_emb_dim] - global feature from adapter output
        Returns:
            latents: [latent_size, output_dim] - fixed-length fragment representation
        """
        # Initialize two sets of latents with different dimensions
        latents_global = self.latents_global
        latents_fragment = self.latents_fragment
        latents_global = self.global_latent_norm(latents_global)
        latents_fragment = self.fragment_latent_norm(latents_fragment)

        # Multi-scale cross-attention with dual paths
        latents = self.multi_scale_perceiver(
            latents_global, latents_fragment, global_feature, fragment_features
        )

        # Self-attention refinement layers
        for layer in self.self_attention_layers:
            latents = layer(latents)

        # Project to output dimension
        out = self.output_proj(latents)
        out: Tensor = self.out_layer_norm(out)
        return out

# Q-former for fragment with dual-mode support
class FragmentAdapter(nn.Module):
    def __init__(
        self,
        protein_emb_dim: int,
        text_emb_dim: int,
        perceiver_latent_size: int, #  length of the fragment hidden states
        num_perceiver_heads: int,
        num_perceiver_layers: int,
        dropout: float,
        frag_adapter_type: str = "qformer",  # "qformer" or "multilevel"
        fragment_block_size: int = 4,
        global_block_size: int = 32,
        global_topk: int = 8,
        max_sub_tokens: int = 8,
    ) -> None:
        super(FragmentAdapter, self).__init__()
        self.protein_layer_norm = nn.LayerNorm(protein_emb_dim)
        self.frag_adapter_type = frag_adapter_type

        if frag_adapter_type == "qformer":
            # Original simple Perceiver
            self.perceiver_layer = Perceiver(
                protein_emb_dim, perceiver_latent_size, text_emb_dim, num_perceiver_heads, num_perceiver_layers, dropout
            )
        elif frag_adapter_type == "multilevel":
            # Multi-scale Perceiver with global and fragment paths
            self.perceiver_layer = MultiScalePerceiver(
                text_emb_dim, perceiver_latent_size, text_emb_dim, num_perceiver_heads, num_perceiver_layers, dropout,
                protein_emb_dim=protein_emb_dim, text_emb_dim=text_emb_dim
            )
        elif frag_adapter_type == "hierarchical":
            self.perceiver_layer = HierarchicalFragmentAdapter(
                protein_emb_dim=protein_emb_dim,
                text_emb_dim=text_emb_dim,
                latent_size=perceiver_latent_size,
                num_heads=num_perceiver_heads,
                num_layers=num_perceiver_layers,
                dropout=dropout,
                fragment_block_size=fragment_block_size,
                global_block_size=global_block_size,
                global_topk=global_topk,
                max_sub_tokens=max_sub_tokens,
            )
        else:
            raise ValueError(f"Unknown frag_adapter_type: {frag_adapter_type}. Must be 'qformer', 'multilevel', or 'hierarchical'")

    def forward(
        self,
        position_refs: List[List[int]],
        encoder_hidden_states: Tensor,
        encoder_attention_mask: Tensor,
        adapter_output: Optional[Tensor] = None,
        **kwargs: Any,
    ) -> Union[Tuple[Tensor], Optional[Tuple[Tensor, Tuple[Tensor, ...]]]]:

        assert encoder_hidden_states is not None
        batch_size = len(position_refs)

        # 1. Normalize the protein embeddings first. This is a static operation.
        encoder_hidden_states = self.protein_layer_norm(encoder_hidden_states)

        # 2. Prepare a list of inputs for the perceiver.
        # For items without a real position_ref, we create a standard dummy input.
        all_frag_latents = []

        if self.frag_adapter_type == "qformer":
            # Original qformer mode: only uses ESM encoder hidden states
            for i in range(batch_size):
                protein_emb = encoder_hidden_states[i]
                encoder_mask = encoder_attention_mask[i]
                position_ref = position_refs[i]

                if position_ref is not None:
                    # This is a REAL input
                    frag_hidden_states = protein_emb[encoder_mask][position_ref[0]:position_ref[1]]
                    all_frag_latents.append(self.perceiver_layer(frag_hidden_states))
                else:
                    # This is a DUMMY input to make the batch complete
                    # Using a slice of length 1 is a safe default
                    dummy_hidden_states = protein_emb[encoder_mask][0:1]
                    all_frag_latents.append(self.perceiver_layer(dummy_hidden_states))

        elif self.frag_adapter_type == "multilevel":
            # Multi-level mode: uses both ESM hidden states and adapter output
            assert adapter_output is not None, "adapter_output is required for multilevel mode"

            for i in range(batch_size):
                protein_emb = encoder_hidden_states[i]  # For fragment features (residue-level)
                encoder_mask = encoder_attention_mask[i]
                position_ref = position_refs[i]

                if position_ref is not None:
                    # Extract fragment features from ESM output (residue-level)
                    frag_features = protein_emb[encoder_mask][position_ref[0]:position_ref[1]]  # [fragment_len, protein_emb_dim]

                    # Global feature from adapter output (semantically aligned with text)
                    global_feature = adapter_output[i][encoder_mask].mean(dim=0, keepdim=True)  # [1, text_emb_dim]

                    # Multi-scale processing: fragment from ESM, global from adapter
                    fragment_latents = self.perceiver_layer(frag_features, global_feature)
                    all_frag_latents.append(fragment_latents)
                else:
                    # Dummy processing for batch completeness
                    dummy_frag_features = protein_emb[encoder_mask][0:1]  # [1, protein_emb_dim]
                    dummy_global_feature = adapter_output[i][encoder_mask].mean(dim=0, keepdim=True)  # [1, text_emb_dim]
                    dummy_latents = self.perceiver_layer(dummy_frag_features, dummy_global_feature)
                    all_frag_latents.append(dummy_latents)

        elif self.frag_adapter_type == "hierarchical":
            # Hierarchical mode uses only original ESM protein features.
            for i in range(batch_size):
                protein_emb = encoder_hidden_states[i]
                encoder_mask = encoder_attention_mask[i].bool()
                if not encoder_mask.any():
                    encoder_mask = torch.ones(
                        protein_emb.size(0), dtype=torch.bool, device=protein_emb.device
                    )
                protein_features = protein_emb[encoder_mask]
                position_ref = position_refs[i]

                if position_ref is not None:
                    all_frag_latents.append(self.perceiver_layer(protein_features, tuple(position_ref)))
                else:
                    dummy_position_ref = (0, protein_features.size(0))
                    all_frag_latents.append(self.perceiver_layer(protein_features, dummy_position_ref))

        # 3. Reconstruct the final output list.
        # This final loop is fine because it doesn't call any nn.Modules.
        # It just selects the results based on the original condition.
        final_frag_latents = [None] * batch_size
        for i in range(batch_size):
            if position_refs[i] is not None:
                final_frag_latents[i] = all_frag_latents[i]
            # If position_refs[i] was None, the list entry correctly remains None.

        return final_frag_latents

# ProteinSAM replaces FragmentPositionDecoder
# The original FragmentPositionDecoder has been replaced by ProteinSAM

class ModalityAdapter(nn.Module):
    """2-layer adapter to match the hidden size of different modalities."""
    def __init__(self, protein_emb_dim: int, 
                 intermediate_dim: int, 
                 text_emb_dim: int, 
                 dropout_rate: float):
        super().__init__()
        self.protein_emb_dim = protein_emb_dim
        self.intermediate_dim = intermediate_dim
        self.text_emb_dim = text_emb_dim
        self.dropout_rate = dropout_rate
        self.fc1 = nn.Linear(protein_emb_dim, intermediate_dim)
        self.fc2 = nn.Linear(intermediate_dim, text_emb_dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(p=dropout_rate)
        self.ln1 = nn.LayerNorm(normalized_shape=intermediate_dim)  # DEPRECATED
        self.ln2 = nn.LayerNorm(normalized_shape=text_emb_dim)  # DEPRECATED processing

    def forward(self, hidden_states: torch.FloatTensor) -> torch.FloatTensor:
        # input: (bsz, seq_len, input_dim)
        hidden_states = self.activation(self.fc1(hidden_states))
        hidden_states = self.dropout(hidden_states)
        # interm: (bsz, seq_len, interm_dim)
        hidden_states = self.activation(self.fc2(hidden_states))
        hidden_states = self.dropout(hidden_states)
        hidden_states = torch.nn.functional.normalize(hidden_states, p=2, dim=-1)
        return hidden_states  # (bsz, seq_len, output_dim)
    

class ProteinMetaModel:
    def __init__(self, config):
        super(ProteinMetaModel, self).__init__(config)
        if hasattr(config, "esm_path"):
            self.esm_encoder = EsmModel.from_pretrained(config.esm_path, add_pooling_layer=False)
            self.adapter = ModalityAdapter(config.protein_emb_dim, config.intermediate_dim, config.hidden_size, config.dropout_rate)
            frag_adapter_type = getattr(config, "frag_adapter_type", "qformer")  # Default to qformer for backward compatibility
            self.fragment_adapter = FragmentAdapter(
                config.protein_emb_dim,
                config.hidden_size,
                config.perceiver_latent_size,
                config.num_perceiver_heads,
                config.num_perceiver_layers,
                config.dropout_rate,
                frag_adapter_type=frag_adapter_type,
                fragment_block_size=getattr(config, "fragment_block_size", 4),
                global_block_size=getattr(config, "global_block_size", 32),
                global_topk=getattr(config, "global_topk", 8),
                max_sub_tokens=getattr(config, "max_sub_tokens", 8),
            )
            self.protein_sam = ProteinSAM(**load_protein_sam_params_with_overrides(config.protein_sam_checkpoint_path, config.esm_path))
    
    def get_esm_encoder(self):
        esm_encoder = getattr(self, "esm_encoder", None)
        if type(esm_encoder) is list:
            return esm_encoder[0]
        return esm_encoder
    
    def initialize_modules(self, model_args, fsdp=None):
        self.config.esm_path = model_args.esm_path
        self.config.intermediate_dim = model_args.intermediate_dim
        self.config.dropout_rate = model_args.dropout_rate
        self.config.perceiver_latent_size = model_args.perceiver_latent_size
        self.config.num_perceiver_heads = model_args.num_perceiver_heads
        self.config.num_perceiver_layers = model_args.num_perceiver_layers
        self.config.frag_adapter_type = getattr(model_args, "frag_adapter_type", "qformer")  # Default to qformer
        self.config.fragment_block_size = getattr(model_args, "fragment_block_size", 4)
        self.config.global_block_size = getattr(model_args, "global_block_size", 32)
        self.config.global_topk = getattr(model_args, "global_topk", 8)
        self.config.max_sub_tokens = getattr(model_args, "max_sub_tokens", 8)

        if self.get_esm_encoder() is None:
            esm_encoder = EsmModel.from_pretrained(model_args.esm_path, add_pooling_layer=False)
            if fsdp is not None and len(fsdp) > 0:
                self.esm_encoder = [esm_encoder]
            else:
                self.esm_encoder = esm_encoder
            self.esm_encoder.requires_grad_(False)
        else:
            if fsdp is not None and len(fsdp) > 0:
                esm_encoder = self.esm_encoder[0]
            else:
                esm_encoder = self.esm_encoder
            esm_encoder.requires_grad_(False)
        self.config.protein_emb_dim = esm_encoder.config.hidden_size
        if getattr(self, "adapter", None) is None:
            self.adapter = ModalityAdapter(self.config.protein_emb_dim,self.config.intermediate_dim, self.config.hidden_size, self.config.dropout_rate)
        if getattr(self, "fragment_adapter", None) is None:
            self.fragment_adapter = FragmentAdapter(
                self.config.protein_emb_dim,
                self.config.hidden_size,
                self.config.perceiver_latent_size,
                self.config.num_perceiver_heads,
                self.config.num_perceiver_layers,
                self.config.dropout_rate,
                frag_adapter_type=self.config.frag_adapter_type,
                fragment_block_size=self.config.fragment_block_size,
                global_block_size=self.config.global_block_size,
                global_topk=self.config.global_topk,
                max_sub_tokens=self.config.max_sub_tokens,
            )
        if getattr(self, "protein_sam", None) is None:
            self.protein_sam = ProteinSAM(**load_protein_sam_params_with_overrides(model_args.protein_sam_checkpoint_path, model_args.esm_path))
            self.protein_sam.load_model(model_args.protein_sam_checkpoint_path)

        if model_args.load_adapter_checkpoint_dir is not None:
            adapter_weights = torch.load(model_args.load_adapter_checkpoint_dir, map_location="cpu")
            def get_w(weights, keyword):
                return {k.split(keyword + '.')[1]: v for k, v in weights.items() if '.' + keyword + '.' in k}
            self.adapter.load_state_dict(get_w(adapter_weights, "adapter"))
            print("Loaded adapter weights from {}".format(model_args.load_adapter_checkpoint_dir))
            print("Loaded adapter weights keys: {}".format(get_w(adapter_weights, "adapter").keys()))
        if model_args.load_fragment_checkpoint_dir is not None:
            fragment_weights = torch.load(model_args.load_fragment_checkpoint_dir, map_location="cpu")
            def get_w(weights, keyword):
                return {k.split(keyword + '.')[1]: v for k, v in weights.items() if '.' + keyword + '.' in k}
            self.fragment_adapter.load_state_dict(get_w(fragment_weights, "fragment_adapter"))
            print("Loaded fragment adapter weights from {}".format(model_args.load_fragment_checkpoint_dir))
            print("Loaded fragment adapter weights keys: {}".format(get_w(fragment_weights, "fragment_adapter").keys()))

class ProteinMetaForCausalLM(ABC):
    @abstractmethod
    def get_model(self):
        pass

    def get_esm_encoder(self):
        return self.get_model().get_esm_encoder()
    
    def prepare_inputs_labels_for_protein(
            self, input_ids, position_ids, attention_mask, past_key_values, labels,
            protein_input_ids, protein_attention_mask, protein_position_ids, protein_head_mask, protein_inputs_embeds, position_refs,output_attentions,output_hidden_states,return_dict
    ):
        if input_ids is not None:
            # import pdb; pdb.set_trace()
            # print("input_ids:", input_ids)
            # print("self.get_model().get_input_embeddings():", self.get_model().get_input_embeddings())
            # inputs_embeds_old = self.get_model().get_input_embeddings()(input_ids)
            # inputs_embeds = inputs_embeds_old.clone()
            inputs_embeds = self.get_model().get_input_embeddings()(input_ids)
            # inputs_embeds = inputs_embeds_old.clone()
        if protein_input_ids is not None:
            esm_encoder = self.get_esm_encoder()
            encoder_output = esm_encoder(
                input_ids=protein_input_ids,
                attention_mask=protein_attention_mask,
                position_ids=protein_position_ids,
                head_mask=protein_head_mask,
                inputs_embeds=protein_inputs_embeds,
                use_cache=False, # because config.esm_config.is_decoder=False
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict
            )
            encoder_hidden_states = encoder_output[0]
            encoder_attention_mask = protein_attention_mask
            # adapter forward
            adapter_output = self.get_model().adapter(encoder_hidden_states)
            if input_ids is None:
                return None, position_ids, None, None, None, labels, encoder_output, adapter_output, encoder_attention_mask, encoder_hidden_states
            # preparation
            batch_size, seq_len = input_ids.size()
            _, encoder_seq_len, _ = adapter_output.size()
            if attention_mask is None: 
                attention_mask = torch.ones(
                    (batch_size, seq_len), 
                    dtype=torch.long, 
                    device=input_ids.device
                )
            if encoder_attention_mask is None: 
                encoder_attention_mask = torch.ones(
                    (batch_size, encoder_seq_len), 
                    dtype=torch.long, 
                    device=adapter_output.device
                )
        # inputs_embeds = self.get_model().get_input_embeddings()(input_ids)
        # if protein_input_ids is not None:
            # replacement
            placeholder_mask = input_ids == self.config.sequence_placeholder_id
            encoder_mask = encoder_attention_mask.bool()
            # B, L, D
            # print("-------------:", inputs_embeds.requires_grad)
            # print("*************:", adapter_output.requires_grad)
            # inputs_embeds[placeholder_mask] = adapter_output[encoder_mask]

            inputs_embeds[placeholder_mask] = adapter_output[encoder_mask]  # debug only for #
            # inputs_embeds[placeholder_mask] = adapter_output[encoder_mask].to(torch.bfloat16) # debug only

            # mask3d = placeholder_mask.unsqueeze(-1).expand_as(inputs_embeds)  # [B, T, D]
            # src = encoder_hidden_states[encoder_mask].reshape(-1)             # [N*D]
            # inputs_embeds = inputs_embeds.masked_scatter(mask3d, src) 
            # list, B, 1, D
            # replace placeholder with fragment embeds
            # if torch.isnan(inputs_embeds).any():
            #     import pdb; pdb.set_trace()
            if not all(x is None for x in position_refs):
                fragment_embeds = self.get_model().fragment_adapter(
                    position_refs=position_refs,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    adapter_output=adapter_output,  # Pass adapter_output for multilevel mode
                )
                fragment_mask = input_ids == self.config.fragment_placeholder_id
                inputs_embeds[fragment_mask] = torch.cat([fragment_embed for fragment_embed in fragment_embeds if fragment_embed is not None], dim=-2)
                # inputs_embeds[fragment_mask] = torch.cat([fragment_embed for fragment_embed in fragment_embeds if fragment_embed is not None], dim=-2).to(torch.bfloat16)  # debug only
            else:
                dummy_protein_hidden_states= torch.zeros(encoder_hidden_states.size(0), encoder_hidden_states.size(1), encoder_hidden_states.size(2), device=encoder_hidden_states.device, dtype=encoder_hidden_states.dtype)
                dummy_position_refs = [[0, encoder_hidden_states.size(1)] for _ in range(batch_size)]
                dummy_fragment_embeds = self.get_model().fragment_adapter(
                    position_refs=dummy_position_refs,
                    encoder_hidden_states=dummy_protein_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    adapter_output=adapter_output,  # Pass adapter_output for multilevel mode
                )
                inputs_embeds = inputs_embeds +(0.0 * torch.cat(dummy_fragment_embeds, dim=-2)).sum()
        else:
            return None, position_ids, attention_mask, past_key_values, inputs_embeds, labels, None, None, None, None

        # inputs_embeds_list = []
        # for i in range(len(position_refs)):
        #     if position_refs[i] is not None:
        #         frag_encoder_hidden_states = encoder_hidden_states[i][encoder_attention_mask[i]][position_refs[i][0]:position_refs[i][1]]
        #         frag_latents = self.get_model().fragment_adapter(frag_encoder_hidden_states)
        #         frag_mask_i = input_ids[i] == self.config.fragment_placeholder_id
        #         inputs_embeds[i][frag_mask_i] = frag_latents
        #         inputs_embeds_list.append(inputs_embeds[i])
        #     else:
        #         frag_encoder_hidden_states = encoder_hidden_states[i][encoder_attention_mask[i]][0:encoder_attention_mask[i].sum()]
        #         frag_latents = self.get_model().fragment_adapter(frag_encoder_hidden_states)
        #         inputs_embeds[i] = inputs_embeds[i] + (0.0 * frag_latents).sum()
        #         inputs_embeds_list.append(inputs_embeds[i])
        # inputs_embeds = torch.stack(inputs_embeds_list, dim=0)

        return None, position_ids, attention_mask, past_key_values, inputs_embeds, labels, encoder_output, adapter_output, encoder_attention_mask, encoder_hidden_states
