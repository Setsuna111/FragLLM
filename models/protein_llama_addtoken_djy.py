from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch import Tensor, nn
from transformers.models.llama import LlamaModel, LlamaForCausalLM, LlamaConfig
from transformers.generation.utils import GenerateOutput
from transformers.modeling_outputs import CausalLMOutputWithPast
from .protein_arch_addtoken_djy import ProteinMetaForCausalLM, ProteinMetaModel
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


class ProteinLlamaForCausalLM_Simple(LlamaForCausalLM, ProteinMetaForCausalLM):
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
        mask = (input_ids == self.config.pos_start_placeholder_id) | (input_ids == self.config.pos_end_placeholder_id)
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

        if return_encoder_outputs:
            return encoder_output
        
        if return_decoder_inputs:
            return inputs_embeds, attention_mask
        
        if return_adapter_outputs:
            return adapter_output, encoder_attention_mask

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
            for i, position_mask in enumerate(position_masks):
                if position_mask.any():
                    assert position_grds[i] is not None
                    assert count_nested_elements_recursive(position_grds[i]) == position_mask.sum(), "count_nested_elements_recursive(position_grds[i]) != position_mask.sum()"
                    postoken_hidden_states = hidden_states[i][position_mask] 
                    # TODO: 用序列做0，1分类时，要注意对ESM_hidden_states进行掐头去尾
                    protein_hidden_states = adapter_output[i][encoder_attention_mask[i].bool()][1:-1]
                    position_grds_pred.append(self.get_model().protein_sam(postoken_hidden_states.unsqueeze(1).contiguous(), protein_hidden_states.unsqueeze(0).expand(postoken_hidden_states.shape[0], -1, -1).contiguous()).squeeze(1))
                else:
                    assert position_grds[i] is None
                    position_grds_pred.append(None)
        else:
            dummy_hidden_states = hidden_states[0][0:1]
            position_grds_pred = self.get_model().protein_sam(dummy_hidden_states.unsqueeze(1).contiguous(), dummy_hidden_states.unsqueeze(0).expand(dummy_hidden_states.shape[0], -1, -1).contiguous()).squeeze(1)
            position_grds_pred = None
        if inference:
            return position_grds_pred, position_grds
        return self._calculate_loss(position_grds_pred, position_grds, output)
            
    def _calculate_loss(self, position_grds_pred, position_grds, output):
        # position_grds_pred中每个元素都是一个(num_positions, 1021)的logit, 与position_grds中的每个元素进行交叉熵损失
        # 自回归损失
        ce_loss = output.loss * getattr(self.config, "ce_loss_weight", 1.0)
        if position_grds_pred is None:
            return {"loss": ce_loss, "ce_loss": ce_loss, "position_loss": torch.tensor(0.0, device=ce_loss.device, dtype=ce_loss.dtype)}
        # 位置损失
        num_positions = 0
        position_labels = []
        position_grds_filtered = [item for item in position_grds if item is not None]
        position_grds_pred_filtered = [item for item in position_grds_pred if item is not None]
        assert len(position_grds_filtered) == len(position_grds_pred_filtered), "position_grds_filtered != position_grds_pred_filtered"
        if len(position_grds_filtered) == 0:
            return {"loss": ce_loss, "ce_loss": ce_loss, "position_loss": torch.tensor(0.0, device=ce_loss.device, dtype=ce_loss.dtype)}
        for batch_position_grd in position_grds_filtered: # batch层
            if batch_position_grd is not None:
                for position_grd in batch_position_grd: # group层
                    for position in position_grd: # position层
                        position_labels.append(position[0])
                        position_labels.append(position[1]-1) # TODO:position_labels.append(position[1]-1),用序列作为二分类时，要注意end_index-1
                        num_positions += 2
            else:
                continue 
        # TODO:  用序列做0，1分类时，position_grds_pred_filtered中每个元素长度不一致
        position_loss = torch.tensor(0.0, device=ce_loss.device, dtype=ce_loss.dtype)
        position_nums = [len(item) for item in position_grds_pred_filtered]
        position_nums_cumsum = torch.cumsum(torch.tensor(position_nums), dim=0)
        assert position_nums_cumsum[-1] == num_positions, "position_nums_cumsum[-1] != num_positions"
        position_offset = torch.cat([torch.zeros(1).long(), position_nums_cumsum], dim=0)
        # import pdb; pdb.set_trace()
        for i, position_grd_pred in enumerate(position_grds_pred_filtered):
            position_loss += nn.CrossEntropyLoss()(position_grd_pred, torch.tensor(position_labels[position_offset[i]:position_offset[i+1]], dtype=torch.long, device=position_grd_pred.device))
        # import pdb; pdb.set_trace()
        position_loss = position_loss / len(position_grds_pred_filtered)
        loss = ce_loss + position_loss * getattr(self.config, "position_loss_weight", 0.1)
        return {"loss": loss, "ce_loss": ce_loss, "position_loss": position_loss}
    
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
                input_ids, position_ids, attention_mask, past_key_values, inputs_embeds, labels, encoder_output, adapter_output, encoder_attention_mask, encoder_hidden_states = self.prepare_inputs_labels_for_protein(
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
        input_ids, position_ids, attention_mask, past_key_values, inputs_embeds, labels, encoder_output, adapter_output, encoder_attention_mask, encoder_hidden_states = self.prepare_inputs_labels_for_protein(
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

        # NOTE: using the proteins hidden states output by the adapter
        # 推理时，generate_output["hidden_states"]是一个tuple, 第一个元素包含input_ids的hidden states,其余元素为(1， L， D)
        output_hidden_states = [generate_output["hidden_states"][0][-1][:,-1:,:]] # (B, 1, D) 
        for hidden_state in generate_output["hidden_states"][1:]:
            output_hidden_states.append(hidden_state[-1]) # (B, 1, D)
        output_hidden_states = torch.cat(output_hidden_states, dim=1) # (B, L_gen, D)

        if not grounding_inference:
            return generate_output_ids
        else:
            # NOTE: using the proteins hidden states output by the adapter
            position_masks = self._create_postoken_mask(generate_output_ids)

            if position_masks.any():
                position_grds_pred = []
                for i, position_mask in enumerate(position_masks):
                    if position_mask.any():
                        # NOTE: using the proteins hidden states output by the adapter
                        postoken_hidden_states = output_hidden_states[i][position_mask] # (num_positions, 4096)
                        protein_hidden_states = adapter_output[i][encoder_attention_mask[i].bool()][1:-1] #(num_proteins, 4096)
                        # protein_hidden_states = encoder_output[0][i][encoder_attention_mask[i].bool()][1:-1] #(num_proteins, 4096)
                        position_grds_pred.append(self.get_model().protein_sam(postoken_hidden_states.unsqueeze(1).contiguous(), protein_hidden_states.unsqueeze(0).expand(postoken_hidden_states.shape[0], -1, -1).contiguous()).squeeze(1))
                    else:
                        position_grds_pred.append(None)
            else:
                position_grds_pred = None
            if position_grds_pred is None:
                position_grds_batch = None
            else:
                # TODO： 需要按batch划分预测结果
                """0, 1""" 
                # position_grds_pred list 每个元素为当前sample的(num_positions, seq_L_sample)的logit
                # import pdb; pdb.set_trace()
                position_masks_nums = position_masks.sum(dim=-1)
                position_grds_batch = []
                for i in range(len(position_masks_nums)):
                    if position_masks_nums[i] == 0:
                        position_grds_batch.append([])
                    else:
                        assert len(position_grds_pred[i]) == position_masks_nums[i], "len(position_grds_pred[i]) != position_masks_nums[i]"
                        # 奇数位置坐标+1，偶数位置坐标不变
                        position_grds_pred_i = position_grds_pred[i].argmax(dim=-1)
                        position_grds_pred_i[1::2] += 1
                        position_grds_batch.append(position_grds_pred_i.cpu().numpy())

        # 将预测值恢复成坐标位置

        return generate_output_ids, position_grds_batch
    
AutoConfig.register("protein_llama", ProteinLlamaConfig)
AutoModelForCausalLM.register(ProteinLlamaConfig, ProteinLlamaForCausalLM_Simple)
