from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
from torch import Tensor, nn
from transformers.models.esm.modeling_esm import EsmModel

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
class FragmentAdapter(nn.Module):
    def __init__(
        self,
        protein_emb_dim: int,
        text_emb_dim: int,
        perceiver_latent_size: int, #  length of the fragment hidden states
        num_perceiver_heads: int,
        num_perceiver_layers: int,
        dropout: float,
    ) -> None:
        super(FragmentAdapter, self).__init__()
        self.protein_layer_norm = nn.LayerNorm(protein_emb_dim)
        self.perceiver_layer = Perceiver(
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
        batch_size = len(position_refs)
        
        # 1. Normalize the protein embeddings first. This is a static operation.
        encoder_hidden_states = self.protein_layer_norm(encoder_hidden_states)
        
        # 2. Prepare a list of inputs for the perceiver.
        # For items without a real position_ref, we create a standard dummy input.
        all_frag_latents = []
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

        
        # 3. Reconstruct the final output list.
        # This final loop is fine because it doesn't call any nn.Modules.
        # It just selects the results based on the original condition.
        final_frag_latents = [None] * batch_size
        for i in range(batch_size):
            if position_refs[i] is not None:
                final_frag_latents[i] = all_frag_latents[i]
            # If position_refs[i] was None, the list entry correctly remains None.
                
        return final_frag_latents

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
            self.fragment_adapter = FragmentAdapter(config.protein_emb_dim, config.hidden_size, config.perceiver_latent_size, config.num_perceiver_heads, config.num_perceiver_layers, config.dropout_rate)

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
            self.fragment_adapter = FragmentAdapter(self.config.protein_emb_dim, self.config.hidden_size, self.config.perceiver_latent_size,self.config.num_perceiver_heads,self.config.num_perceiver_layers,self.config.dropout_rate)

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
            inputs_embeds_old = self.get_model().get_input_embeddings()(input_ids)
            inputs_embeds = inputs_embeds_old.clone()
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
                return None, position_ids, None, None, None, labels, encoder_output, adapter_output, encoder_attention_mask
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
            inputs_embeds[placeholder_mask] = adapter_output[encoder_mask]
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
                )
                fragment_mask = input_ids == self.config.fragment_placeholder_id
                inputs_embeds[fragment_mask] = torch.cat([fragment_embed for fragment_embed in fragment_embeds if fragment_embed is not None], dim=-2)
            else:
                dummy_protein_hidden_states= torch.zeros(encoder_hidden_states.size(0), encoder_hidden_states.size(1), encoder_hidden_states.size(2), device=encoder_hidden_states.device, dtype=encoder_hidden_states.dtype)
                dummy_position_refs = [[0, encoder_hidden_states.size(1)] for _ in range(batch_size)]
                dummy_fragment_embeds = self.get_model().fragment_adapter(
                    position_refs=dummy_position_refs,
                    encoder_hidden_states=dummy_protein_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                )
                inputs_embeds = inputs_embeds +(0.0 * torch.cat(dummy_fragment_embeds, dim=-2)).sum()
        else:
            return None, position_ids, attention_mask, past_key_values, inputs_embeds, labels, None, None, None

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

    
        return None, position_ids, attention_mask, past_key_values, inputs_embeds, labels, encoder_output, adapter_output, encoder_attention_mask