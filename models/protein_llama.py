from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch import Tensor, nn
from transformers.models.llama import LlamaModel, LlamaForCausalLM, LlamaConfig
from transformers.generation.utils import GenerateOutput
from transformers.modeling_outputs import CausalLMOutputWithPast
from .protein_arch import ProteinMetaForCausalLM, ProteinMetaModel
from transformers import AutoConfig, AutoModelForCausalLM, \
                         Cache
class ProteinLlamaConfig(LlamaConfig):
    model_type = "protein_llama"

class ProteinLlamaModel(ProteinMetaModel, LlamaModel):
    config_class = ProteinLlamaConfig

    def __init__(self, config: LlamaConfig):
        super(ProteinLlamaModel, self).__init__(config)

class ProteinLlamaForCausalLM(LlamaForCausalLM, ProteinMetaForCausalLM):
    config_class = ProteinLlamaConfig
    def __init__(self, config):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = ProteinLlamaModel(config)
        self.pretraining_tp = config.pretraining_tp
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()


    def get_model(self):
        return self.model
    
    def forward(
            self, 
            # chat template text inputs
            input_ids: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.LongTensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[Cache] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
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
        if inputs_embeds is None:
            input_ids, position_ids, attention_mask, past_key_values, inputs_embeds, labels, encoder_output, adapter_output, encoder_attention_mask = self.prepare_inputs_labels_for_protein(
                input_ids, position_ids, attention_mask, past_key_values, labels,
                protein_input_ids, protein_attention_mask, protein_position_ids, protein_head_mask, protein_inputs_embeds, position_refs,output_attentions,output_hidden_states,return_dict
            )
            
        if return_encoder_outputs:
            return encoder_output
        
        if return_decoder_inputs:
            return inputs_embeds, attention_mask
        
        if return_adapter_outputs:
            return adapter_output, encoder_attention_mask

        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.LongTensor,  # alias of `input_ids`
        attention_mask: Optional[torch.LongTensor] = None,
        protein_input_ids: Optional[torch.LongTensor] = None,
        protein_attention_mask: Optional[torch.LongTensor] = None,
        protein_inputs_embeds: Optional[torch.FloatTensor] = None,
        # fragment inputs
        position_refs: Optional[List[List[int]]] = None,
        **kwargs
    ) -> Union[GenerateOutput, torch.LongTensor]:
        
        input_ids, position_ids, attention_mask, past_key_values, inputs_embeds, labels, encoder_output, adapter_output, encoder_attention_mask = self.prepare_inputs_labels_for_protein(
                input_ids, position_ids, attention_mask, past_key_values, labels,
                protein_input_ids, protein_attention_mask, None, None, protein_inputs_embeds, position_refs,None,None,None
            )
        return super().generate(
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs
        )
    
AutoConfig.register("protein_llama", ProteinLlamaConfig)
AutoModelForCausalLM.register(ProteinLlamaConfig, ProteinLlamaForCausalLM)


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    from dataset.dataloader_frag import make_multitask_dataset
    from scripts.train_llama import FragDataArguments, FragModelArguments
    model_args = FragModelArguments()
    from transformers import AutoTokenizer
    data_args = FragDataArguments()
    model_args.esm_path = "/home/djy/projects/Data/HF_models/esm2_t36_3B_UR50D"
    model_args.llama_path = "/home/djy/projects/Data/HF_models/RedHatAI-Llama-3.1-8B-Instruct"
    data_args.sequence_tokenizer = AutoTokenizer.from_pretrained(model_args.esm_path)
    data_args.llm_tokenizer = AutoTokenizer.from_pretrained(model_args.llama_path,pad_token='<|reserved_special_token_0|>')
    data_module = make_multitask_dataset(data_args)
    # import pdb; pdb.set_trace()
    train_dataloader = DataLoader(
        data_module["train_dataset"],
        batch_size=1,
        shuffle=True,
        num_workers=0,
        collate_fn=data_module["data_collator"],
        pin_memory=True, 
        drop_last=True
    )
    model = ProteinLlamaForCausalLM.from_pretrained(
        "/home/djy/projects/Data/HF_models/RedHatAI-Llama-3.1-8B-Instruct",
        torch_dtype=torch.float16,
    )
    if model_args.esm_path is not None:
        model.get_model().initialize_modules(model_args=model_args, fsdp=None)
        model.config.sequence_placeholder_id = model_args.sequence_placeholder_id
        model.config.fragment_placeholder_id = model_args.fragment_placeholder_id
        model.config.pos_start_placeholder_id = model_args.pos_start_placeholder_id
        model.config.pos_end_placeholder_id = model_args.pos_end_placeholder_id
    model.to("cuda:0")
    for batch in train_dataloader:

        model(
            input_ids=batch["input_ids"].to("cuda:0"),
            attention_mask=batch["attention_mask"].to("cuda:0"),
            protein_input_ids=batch["protein_input_ids"].to("cuda:0"),
            protein_attention_mask=batch["protein_attention_mask"].to("cuda:0"),
            position_refs=batch["position_refs"],
        )

        import pdb; pdb.set_trace()