from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch import Tensor, nn
from transformers.models.llama import LlamaModel, LlamaForCausalLM, LlamaConfig
from transformers.generation.utils import GenerateOutput
from transformers.modeling_outputs import CausalLMOutputWithPast
from .protein_arch_addtoken_lfj import ProteinMetaForCausalLM, ProteinMetaModel
from transformers import AutoConfig, AutoModelForCausalLM, \
                         Cache
class ProteinLlamaConfig(LlamaConfig):
    model_type = "protein_llama"

class ProteinLlamaModel(ProteinMetaModel, LlamaModel):
    config_class = ProteinLlamaConfig

    def __init__(self, config: LlamaConfig):
        super(ProteinLlamaModel, self).__init__(config)

def count_nested_elements_recursive(data):
    """
    使用递归方法计算多层嵌套列表中最里层元素的总数。
    """
    count = 0
    # 遍历列表中的每一个元素
    for element in data:
        # 如果元素是列表，则递归调用函数并将结果累加
        if isinstance(element, list):
            count += count_nested_elements_recursive(element)
        # 如果元素不是列表，说明它是一个最里层的元素
        else:
            count += 1
    return count


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

    def _create_postoken_mask(self, input_ids):
        # Only use position token as input to ProteinSAM  
        mask = input_ids == self.config.position_placeholder_id
        return mask

    def _create_seq_mask(self, input_ids):
        mask = input_ids == self.config.sequence_placeholder_id
        return mask
    
    def model_forward(
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
            position_grds: Optional[List[List[int]]] = None,
            # behavior control arguments
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            return_encoder_outputs: bool = False,
            return_adapter_outputs: bool = False, 
            return_decoder_inputs: bool = False,
            cache_position: Optional[torch.LongTensor] = None,
            inference: bool = False,
            **kwargs,
    ):
        input_ids_old = input_ids.clone()
        if inputs_embeds is None:
            input_ids, position_ids, attention_mask, past_key_values, inputs_embeds, labels, encoder_output, adapter_output, encoder_attention_mask, protein_encoder_hidden_states = self.prepare_inputs_labels_for_protein(
                input_ids, position_ids, attention_mask, past_key_values, labels,
                protein_input_ids, protein_attention_mask, protein_position_ids, protein_head_mask, protein_inputs_embeds, position_refs, output_attentions,output_hidden_states,return_dict
            )

        output = super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=True,
                return_dict=True
            )
        hidden_states = output['hidden_states'][-1]

        # shift for get the hidden states of tokens to predict
        input_ids_old = input_ids_old[:, 1:]
        hidden_states = hidden_states[:, :-1, :]

        position_masks = self._create_postoken_mask(input_ids_old)

        if position_masks.any():
            position_grds_pred = []
            # Use ProteinSAM for position prediction. Handle multiple position tokens per sample
            for i, position_mask in enumerate(position_masks):
                if position_mask.any():
                    assert position_grds[i] is not None
                    # Get all position tokens for this sample
                    postoken_hidden_states_all = hidden_states[i][position_mask]  # (num_tokens, hidden_size)
                    num_tokens = postoken_hidden_states_all.shape[0]
                    
                    # Process each position token separately
                    sample_outputs = []
                    for token_idx in range(num_tokens):
                        # Get this specific token's hidden state
                        postoken_hidden_states = postoken_hidden_states_all[token_idx:token_idx+1]  # (1, hidden_size)
                        
                        # Extract corresponding ground truth labels for training
                        start_labels = None
                        end_labels = None
                        if (position_grds[i] is not None and len(position_grds[i]) > 0 and 
                            len(position_grds[i][0]) > token_idx):
                            # Extract position pair for this specific token
                            position_pair = position_grds[i][0][token_idx]  # [group][position_idx][start,end]
                            start_labels = torch.tensor([position_pair[0]], device=protein_input_ids.device)
                            end_labels = torch.tensor([position_pair[1] - 1], device=protein_input_ids.device)
                        # for segmentation loss calculation
                        start_end_labels = torch.zeros_like(protein_input_ids[i:i+1], dtype=torch.int64, device=protein_input_ids.device)
                        start_end_labels = start_end_labels[:, 1:-1]
                        if start_labels is not None and end_labels is not None:
                            start_end_labels[0, start_labels.item():end_labels.item() + 1] = 1

                        # Use special token embedding directly as external prompt
                        special_token_embedding = postoken_hidden_states.unsqueeze(1)  # (1, 1, hidden_size)
                        esm_embeddings = protein_encoder_hidden_states[i:i+1, 1: -1]  # Exclude CLS and SEP embeddings
                        
                        # Call ProteinSAM with external embedding
                        sam_outputs = self.get_model().protein_sam(
                            protein_attention_mask=protein_attention_mask[i:i+1],
                            external_prompt_embeddings=special_token_embedding,  # Use special token embedding
                            external_esm_embeddings=esm_embeddings,
                            residue_labels=start_end_labels
                        )
                    
                        sample_outputs.append(sam_outputs)
                    
                    # Store all outputs for this sample
                    position_grds_pred.append(sample_outputs)
                else:
                    assert position_grds[i] is None
                    position_grds_pred.append(None)
        else:
            # 当没有grounding任务时,创建dummy输入保持梯度流
            # breakpoint()
            dummy_external_prompt = hidden_states[0:1, 0:1, :].contiguous()
            dummy_esm_embeddings = protein_encoder_hidden_states[0:1, 1:-1, :].contiguous()  # 完整序列，去掉CLS和SEP
            dummy_labels = torch.zeros(1, protein_input_ids.size(1)-2,
                            dtype=torch.long, device=protein_input_ids.device)

            dummy_sam_outputs = self.get_model().protein_sam(
                protein_attention_mask=protein_attention_mask[0:1],
                external_prompt_embeddings=dummy_external_prompt,
                external_esm_embeddings=dummy_esm_embeddings,
                residue_labels=dummy_labels
            )

            # 将dummy loss乘以0加到output.loss上，保持梯度流但不影响训练
            if 'loss' in dummy_sam_outputs:
                output.loss = output.loss + 0.0 * dummy_sam_outputs['loss']

            # 设置为None，不传递给_calculate_loss进行实际的loss计算
            position_grds_pred = None
            
        if inference:
            return position_grds_pred, position_grds
        
        return self._calculate_loss(position_grds_pred, position_grds, output)
            
    def _calculate_loss(self, position_grds_pred, position_grds, output):
        # ProteinSAM now handles position prediction internally
        # position_grds_pred now contains ProteinSAM outputs with integrated loss
        ce_loss = output.loss * getattr(self.config, "ce_loss_weight", 1.0)
        
        if position_grds_pred is None:
            return {"loss": ce_loss, "ce_loss": ce_loss, "position_loss": torch.tensor(0.0, device=ce_loss.device, dtype=ce_loss.dtype)}
        
        # Extract position loss from ProteinSAM outputs if available
        position_loss = torch.tensor(0.0, device=ce_loss.device, dtype=ce_loss.dtype)
        if hasattr(position_grds_pred, 'get') and "loss" in position_grds_pred:
            position_loss = position_grds_pred["loss"]
        elif isinstance(position_grds_pred, list) and len(position_grds_pred) > 0:
            # For batch processing, aggregate losses from ProteinSAM outputs
            # position_grds_pred is now [sample][token][outputs]
            batch_losses = []
            for sample_preds in position_grds_pred:
                if sample_preds is not None:
                    if isinstance(sample_preds, list):
                        # Multiple tokens per sample
                        for token_pred in sample_preds:
                            if token_pred is not None and hasattr(token_pred, 'get') and "loss" in token_pred:
                                batch_losses.append(token_pred["loss"])
                    else:
                        # Single prediction (backward compatibility)
                        if hasattr(sample_preds, 'get') and "loss" in sample_preds:
                            batch_losses.append(sample_preds["loss"])
            if batch_losses:
                position_loss = torch.stack(batch_losses).mean()
        
        total_loss = ce_loss + position_loss * getattr(self.config, "position_loss_weight", 0.1)
        return {"loss": total_loss, "ce_loss": ce_loss, "position_loss": position_loss}
    
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
            position_grds: Optional[List[List[int]]] = None,
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
        if past_key_values is not None:
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
        else:
            return self.model_forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                labels=labels,
                # protein amino-acid sequence inputs
                protein_input_ids=protein_input_ids,
                protein_attention_mask=protein_attention_mask,
                protein_position_ids=protein_position_ids, 
                protein_head_mask=protein_head_mask,
                protein_inputs_embeds=protein_inputs_embeds,
                # fragment inputs
                position_refs=position_refs,
                position_grds=position_grds,
                # behavior control arguments
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                return_encoder_outputs=return_encoder_outputs,
                return_adapter_outputs=return_adapter_outputs, 
                return_decoder_inputs=return_decoder_inputs,
                cache_position=cache_position,
                **kwargs,
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
        position_grds: Optional[List[List[int]]] = None,
        grounding_inference: bool = False,
        **kwargs
    ) -> Union[GenerateOutput, torch.LongTensor]:
        input_ids, position_ids, attention_mask, past_key_values, inputs_embeds, labels, encoder_output, adapter_output, encoder_attention_mask = self.prepare_inputs_labels_for_protein(
                input_ids, None, attention_mask, None, None,
                protein_input_ids, protein_attention_mask, None, None, protein_inputs_embeds, position_refs,None,None,None
            )
        generate_output = super().generate(
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            return_dict_in_generate=True,
            output_hidden_states=True,
            return_dict=True,
            **kwargs
        )

        generate_output_ids = generate_output["sequences"]

        # 推理时，generate_output["hidden_states"]是一个tuple, 第一个元素包含input_ids的hidden states,其余元素为(1， L， D)
        output_hidden_states = []
        for hidden_state in generate_output["hidden_states"][1:]:
            output_hidden_states.append(hidden_state[-1]) # (B, 1, D)
        output_hidden_states = torch.cat(output_hidden_states, dim=1) # (B, L_gen, D)
        
        if not grounding_inference:
            return generate_output_ids
        else:
            position_masks = self._create_postoken_mask(generate_output_ids[:, 1:])  # shift for prediction
            if position_masks.any():
                position_grds_pred = []
                for i, position_mask in enumerate(position_masks):
                    if position_mask.any():
                        # Use ProteinSAM for inference position prediction
                        generated_hidden_states = output_hidden_states[i]
                        
                        if position_mask.any():
                            # Get all position tokens for this sample
                            postoken_hidden_states_all = generated_hidden_states[position_mask]  # (num_tokens, hidden_size)
                            num_tokens = postoken_hidden_states_all.shape[0]
                            
                            # Process each position token separately
                            sample_predictions = []
                            for token_idx in range(num_tokens):
                                # Get this specific token's hidden state
                                postoken_hidden_states = postoken_hidden_states_all[token_idx:token_idx+1]  # (1, hidden_size)
                                
                                # Use special token embedding directly as external prompt
                                special_token_embedding = postoken_hidden_states.unsqueeze(1)  # (1, 1, hidden_size)
                                
                                # Call ProteinSAM for grounding inference
                                sam_outputs = self.get_model().protein_sam(
                                    protein_input_ids=protein_input_ids[i:i+1],
                                    protein_attention_mask=protein_attention_mask[i:i+1],
                                    external_prompt_embeddings=special_token_embedding  # Use special token embedding
                                )
                                
                                # Store predictions for this token
                                sample_predictions.append({
                                    "start_predictions": sam_outputs["start_predictions"],
                                    "end_predictions": sam_outputs["end_predictions"] + 1  # Convert back to inclusive end
                                })
                            
                            # Store all predictions for this sample
                            position_grds_pred.append(sample_predictions)
            else:
                position_grds_pred = []
            
            # Check if we have any valid predictions before processing
            if len(position_grds_pred) == 0:
                position_grds_batch = None
            else:
                # Process both start and end predictions (already argmax'ed by ProteinSAM)
                # Flatten the nested structure: position_grds_pred is now [sample][token][predictions]
                start_positions = []
                end_positions = []
                for sample_preds in position_grds_pred:
                    for token_pred in sample_preds:
                        start_positions.append(token_pred["start_predictions"])
                        end_positions.append(token_pred["end_predictions"])
                
                # Concatenate all position predictions
                start_positions_cat = torch.cat(start_positions, dim=0)  # [all_positions]
                end_positions_cat = torch.cat(end_positions, dim=0)      # [all_positions]
                
                # Divide by batch based on position masks
                position_masks_nums = position_masks.sum(dim=-1)
                position_masks_offset = torch.cumsum(position_masks_nums, dim=-1)
                position_masks_offset = torch.cat([torch.zeros(1).long().to(position_masks.device), position_masks_offset], dim=0)
                
                position_grds_batch = []
                for i in range(len(position_masks_offset)-1):
                    start_index = position_masks_offset[i]
                    end_index = position_masks_offset[i+1]
                    batch_start_positions = start_positions_cat[start_index:end_index]
                    batch_end_positions = end_positions_cat[start_index:end_index]
                    position_grds_batch.append({
                        "start_positions": batch_start_positions,
                        "end_positions": batch_end_positions
                    })
        # Return generated text IDs along with position predictions
        return generate_output_ids, position_grds_batch
    
AutoConfig.register("protein_llama", ProteinLlamaConfig)
AutoModelForCausalLM.register(ProteinLlamaConfig, ProteinLlamaForCausalLM)
