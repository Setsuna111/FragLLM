"""
Configuration class for the assembled Esm2LlamaInstructForCausalLM model. 

Esm2LlamaInstructForCausalLM = EsmModel + ModalityAdapter + LlamaForCausalLM

For training/evaluation under teacher-forcing scenario, the model `forward` 
function shall take following arguments: 
    * input_ids: (bsz, prompt_len+description_len)  # whole chat template
    * attention_mask: (bsz, prompt_len+description_len)  # left & right padding
    * position_ids: (bsz, prompt_len+description_len)  # optional
    * past_key_values: None
    * labels: (bsz, prompt_len+description_len)  # -100 for padding & prompt
    * protein_input_ids: (bsz, prot_seq_len)  # either ids or embeds
    * protein_attention_mask: (bsz, prot_seq_len)  # right padding
    * protein_position_ids: (bsz, prot_seq_len)  # optional
    * protein_head_mask: (num_heads,) or (num_layers, num_heads)  # optional
    * protein_inputs_embeds: (bsz, prot_seq_len, hidden_size)  # optional
    * use_cache: False
    * return_decoder_inputs: False

For inference, the model `generate` function shall take following arguments: 
    * inputs: (bsz, prompt_len)  # prompt part of chat template
    * attention_mask: (bsz, prompt_len)  # left padding
    * protein_input_ids: (bsz, prot_seq_len)  # either ids or embeds
    * protein_attention_mask: (bsz, prot_seq_len)  # right padding
    * protein_inputs_embeds: (bsz, prot_seq_len, hidden_size)  # optional
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint
from transformers import Cache, PreTrainedModel
from transformers.generation.utils import GenerateOutput
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.models.esm.modeling_esm import EsmModel
from transformers.models.llama import LlamaForCausalLM

from .configuration_esm2llama_instruct_frag import (
    ModalityAdapterConfig, 
    Esm2LlamaInstructConfig
)

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
        enable_gradient_checkpointing: bool,
        dropout: float,
    ) -> None:
        super(FragmentAdapter, self).__init__()
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
        frag_latents = [None] * len(position_refs)
        assert encoder_hidden_states is not None
        encoder_hidden_states = self.protein_layer_norm(encoder_hidden_states)
        for i, (protein_emb, position_ref, encoder_mask) in enumerate(zip(encoder_hidden_states, position_refs, encoder_attention_mask)):
            if position_ref is not None:
                frag_hidden_states = protein_emb[encoder_mask][position_ref[0]:position_ref[1]]
                frag_latents[i] = self.perceiver(frag_hidden_states)
            else:
                frag_latents[i] = None
        return frag_latents


class ModalityAdapter(PreTrainedModel):
    """2-layer adapter to match the hidden size of different modalities."""
    config_class = ModalityAdapterConfig  # configuration class for this model

    def __init__(self, config: ModalityAdapterConfig):
        super().__init__(config)
        self.config = config
        self.fc1 = torch.nn.Linear(config.input_dim, config.intermediate_dim)
        self.fc2 = torch.nn.Linear(config.intermediate_dim, config.output_dim)
        self.activation = torch.nn.GELU()
        self.dropout = torch.nn.Dropout(p=config.dropout_rate)
        self.ln1 = torch.nn.LayerNorm(normalized_shape=config.intermediate_dim)  # DEPRECATED
        self.ln2 = torch.nn.LayerNorm(normalized_shape=config.output_dim)  # DEPRECATED
        self.post_init()  # initialize weights and apply final processing

    def forward(self, hidden_states: torch.FloatTensor) -> torch.FloatTensor:
        # input: (bsz, seq_len, input_dim)
        hidden_states = self.activation(self.fc1(hidden_states))
        hidden_states = self.dropout(hidden_states)
        # interm: (bsz, seq_len, interm_dim)
        hidden_states = self.activation(self.fc2(hidden_states))
        hidden_states = self.dropout(hidden_states)
        hidden_states = torch.nn.functional.normalize(hidden_states, p=2, dim=-1)
        return hidden_states  # (bsz, seq_len, output_dim)


class Esm2LlamaInstructForCausalLM(PreTrainedModel):
    """
    Esm2LlamaInstructForCausalLM model for protein function prediction.
    Similar to `EncoderDecoderModel` but with more complicated architecture.
    Initialize with either a configuration OR all three components.
    `kwargs` can override standalone attributes in `Esm2LlamaInstructConfig`.
    """
    config_class = Esm2LlamaInstructConfig  # configuration class for this model

    def __init__(
            self, 
            config: Optional[Esm2LlamaInstructConfig] = None, 
            esm_encoder: Optional[EsmModel] = None, 
            adapter: Optional[ModalityAdapter] = None,
            llama_decoder: Optional[LlamaForCausalLM] = None, 
            **kwargs
        ):
        if config is not None:  # components ignored if config is provided
            super().__init__(config)
            self.esm_encoder = EsmModel(
                config.esm_config, 
                add_pooling_layer=False
            )
            self.adapter = ModalityAdapter(config.adapter_config)
            self.llama_decoder = LlamaForCausalLM(config.llama_config)
        else: 
            config = Esm2LlamaInstructConfig(
                esm_config=esm_encoder.config,
                adapter_config=adapter.config,
                llama_config=llama_decoder.config, 
                **kwargs  # override standalone attributes
            ) 
            super().__init__(config)
            self.esm_encoder = esm_encoder
            self.adapter = adapter
            self.llama_decoder = llama_decoder
        # self.config = config
        self.fragment_adapter = FragmentAdapter(
            protein_emb_dim=config.adapter_config.input_dim,
            text_emb_dim=config.adapter_config.output_dim,
            perceiver_latent_size=config.perceiver_latent_size,
            num_perceiver_heads=config.num_perceiver_heads,
            num_perceiver_layers=config.num_perceiver_layers,
            enable_gradient_checkpointing=hasattr(self.llama_decoder, "gradient_checkpointing_enable"),
            dropout=config.adapter_config.dropout_rate,
        )
            
    def prepare_decoder_inputs(
            self, 
            input_ids: torch.LongTensor,
            encoder_hidden_states: torch.FloatTensor,
            protein_hidden_states: torch.FloatTensor,
            attention_mask: Optional[torch.LongTensor] = None,
            encoder_attention_mask: Optional[torch.LongTensor] = None, 
            position_refs: Optional[List[List[int]]] = None,
    ): 
        """
        Embed and replace placeholder in `input_ids` by encoder hidden states.
        `input_ids` must be passed to locate placeholder for replacement.
        """
        # preparation
        batch_size, seq_len = input_ids.size()
        _, encoder_seq_len, _ = encoder_hidden_states.size()
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
                device=encoder_hidden_states.device
            )
        inputs_embeds = self.llama_decoder.get_input_embeddings()(input_ids)
        print("-------------:", inputs_embeds.requires_grad)
        # replacement
        
        placeholder_mask = input_ids == self.config.sequence_placeholder_id
        encoder_mask = encoder_attention_mask.bool()
        # B, L, D
        inputs_embeds[placeholder_mask] = encoder_hidden_states[encoder_mask]
        print("***********:",  encoder_hidden_states.requires_grad)

        # list, B, 1, D
        # replace placeholder with fragment embeds
        if not all(x is None for x in position_refs):
            fragment_embeds = self.fragment_adapter(
            position_refs=position_refs,
            encoder_hidden_states=protein_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            )
            fragment_mask = input_ids == self.config.fragment_placeholder_id
            inputs_embeds[fragment_mask] = torch.cat([fragment_embed for fragment_embed in fragment_embeds if fragment_embed is not None], dim=-2)
        else:
            dummy_protein_hidden_states= torch.zeros(protein_hidden_states.size(0), protein_hidden_states.size(1), protein_hidden_states.size(2), device=protein_hidden_states.device, dtype=protein_hidden_states.dtype)
            dummy_position_refs = [[0, protein_hidden_states.size(1)] for _ in range(batch_size)]
            dummy_fragment_embeds = self.fragment_adapter(
                position_refs=dummy_position_refs,
                encoder_hidden_states=dummy_protein_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
            )
            inputs_embeds += (0.0 * torch.cat(dummy_fragment_embeds, dim=-2)).sum()

        
        return inputs_embeds, attention_mask

    def forward(
            self, 
            # chat template text inputs
            input_ids: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.LongTensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[Cache] = None,
            labels: Optional[torch.LongTensor] = None,
            # protein amino-acid sequence inputs
            protein_input_ids: Optional[torch.LongTensor] = None,
            protein_attention_mask: Optional[torch.LongTensor] = None,
            protein_position_ids: Optional[torch.LongTensor] = None, 
            protein_head_mask: Optional[torch.LongTensor] = None,
            protein_inputs_embeds: Optional[torch.FloatTensor] = None,
            # fragment inputs
            position_refs: Optional[List[List[int]]] = None,
            # behavior control arguments
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            return_encoder_outputs: bool = False,
            return_adapter_outputs: bool = False, 
            return_decoder_inputs: bool = False,
            cache_position: Optional[torch.LongTensor] = None,
            **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]: 
        """
        Compute encoder and adapter outputs, then pass to decoder.
        `input_ids` is expected to be [prompt + description] in teacher-forcing 
        scenario and [prompt] only in first iteration of inference (with 
        return_decoder_inputs=True). 
        Attention: possible concatenation of the mask and labels should be 
        handled before calling this method.
        `inputs_embeds` not allowed due to placeholder replacement scheme. 
        """
        # esm_encoder forward
        encoder_output = self.esm_encoder(
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
        if return_encoder_outputs:
            return encoder_output
        # adapter forward
        adapter_output = self.adapter(encoder_hidden_states)
        if return_adapter_outputs:
            return adapter_output, encoder_attention_mask
        # decoder input preparation
        inputs_embeds, attention_mask = self.prepare_decoder_inputs(
            input_ids=input_ids, 
            encoder_hidden_states=adapter_output, 
            protein_hidden_states=encoder_hidden_states,
            attention_mask=attention_mask, 
            encoder_attention_mask=encoder_attention_mask, 
            position_refs=position_refs,
        )
        if return_decoder_inputs:
            return inputs_embeds, attention_mask
        # llama_decoder forward
        return self.llama_decoder.forward(
            input_ids=None,
            attention_mask=attention_mask, 
            position_ids=position_ids, 
            past_key_values=past_key_values, 
            inputs_embeds=inputs_embeds, 
            labels=labels, 
            use_cache=use_cache, 
            output_attentions=output_attentions, 
            return_dict=return_dict, 
            cache_position=cache_position
        )

    def generate(
        self,
        inputs: torch.LongTensor,  # alias of `input_ids`
        attention_mask: Optional[torch.LongTensor] = None,
        protein_input_ids: Optional[torch.LongTensor] = None,
        protein_attention_mask: Optional[torch.LongTensor] = None,
        protein_inputs_embeds: Optional[torch.FloatTensor] = None,
        # fragment inputs
        position_refs: Optional[List[List[int]]] = None,
        **kwargs
    ) -> Union[GenerateOutput, torch.LongTensor]:
        """
        Do inference based on given input prompt. 
        `inputs` is expected to be [prompt] only. 
        Output will not keep the input prompt due to input in form of embeds.
        Generation behavior can be controlled by `args` and `kwargs`, read 
        `GenerationMixin.generate` for more info. 
        """
        # get decoder inputs
        prompt_inputs_embeds, prompt_attention_mask = self(
            input_ids=inputs, 
            attention_mask=attention_mask,
            protein_input_ids=protein_input_ids,
            protein_attention_mask=protein_attention_mask,
            protein_inputs_embeds=protein_inputs_embeds,
            # fragment inputs
            position_refs=position_refs,
            use_cache=False, 
            output_attentions=False,
            output_hidden_states=False,
            return_dict=False,
            return_decoder_inputs=True
        )
        # do generate on llama_decoder
        return self.llama_decoder.generate(
            inputs_embeds=prompt_inputs_embeds, 
            attention_mask=prompt_attention_mask, 
            **kwargs
        )

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        """
        Enable gradient checkpointing for all submodules that support it.
        Attention! Model need to be in train mode before calling this method.
        """
        if hasattr(self.esm_encoder, "gradient_checkpointing_enable"):
                self.esm_encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)
        if hasattr(self.llama_decoder, "gradient_checkpointing_enable"):
            self.llama_decoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)
        # simple adapter no need to implement gradient checkpointing

    def gradient_checkpointing_disable(self):
        if hasattr(self.esm_encoder, "gradient_checkpointing_disable"):
            self.esm_encoder.       gradient_checkpointing_disable()
        if hasattr(self.llama_decoder, "gradient_checkpointing_disable"):
            self.llama_decoder.gradient_checkpointing_disable()
        # simple adapter no need to implement gradient checkpointing
    
if __name__ == "__main__":
    # Fragperceiver = FragmentAdapter(
    # protein_emb_dim=1280,
    # text_emb_dim=768,
    # perceiver_latent_size=1,
    # num_perceiver_heads=16,
    # num_perceiver_layers=12,
    # enable_gradient_checkpointing=True,
    # dropout=0.05,
    # )
    # position_refs = [[5, 20], [10, 30]]
    # encoder_hidden_states = torch.randn(2, 1021, 1280)
    # encoder_attention_mask = torch.ones((2, 1021),dtype=torch.long)
    # frag_latents = Fragperceiver(position_refs, encoder_hidden_states, encoder_attention_mask)
    

    # 初始化Esm2LlamaInstructForCausalLM
    esm_encoder = EsmModel.from_pretrained(
        "/home/djy/projects/Data/HF_models/esm2_t36_3B_UR50D", 
        add_pooling_layer=False,
        torch_dtype=torch.float16, 
        device_map="cpu"
    )
    llama_decoder = LlamaForCausalLM.from_pretrained(
        "/home/djy/projects/Data/HF_models/RedHatAI-Llama-3.1-8B-Instruct", 
        torch_dtype=torch.float16, 
        device_map="cpu"
    )
    adapter_config = ModalityAdapterConfig(
        input_dim=esm_encoder.config.hidden_size,
        intermediate_dim=2048,
        output_dim=llama_decoder.config.hidden_size,
    )
    model_config = Esm2LlamaInstructConfig(
        esm_config=esm_encoder.config,
        adapter_config=adapter_config,
        llama_config=llama_decoder.config,
    )
    import transformers
    _orig_repr = transformers.PretrainedConfig.__repr__

    def patched_repr(self):
        print(f"Entering repr for {type(self)} at dict keys: {list(self.__dict__.keys())}")
        return _orig_repr(self)

    transformers.PretrainedConfig.__repr__ = patched_repr
    print(model_config)
    import pdb; pdb.set_trace()
    adapter = ModalityAdapter(adapter_config)
    adapter.to(torch.float16)
    # import pdb; pdb.set_trace()
    model = Esm2LlamaInstructForCausalLM(
        esm_encoder=esm_encoder,
        adapter=adapter,
        llama_decoder=llama_decoder,
    )
    import pdb; pdb.set_trace()