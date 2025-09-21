"""
Light-weight dataset and data collater class for protein function prediction 
instruction tuning. To be used with Esm2LlamaInstructForCausalLM. 

Such flexible implementation is designed to fetch raw text data from a CSV file 
and perform tokenization and padding on-the-fly. This is useful when the default 
user message and chat template is not suitable for the task at hand.

Can only be used if the model is not requiring graph-related data.

Every batch from DataLoader will contain following attributes:
    * Training mode (train-eval with teacher-forcing): 
        - graph related features:
            None
        - amino-acid sequence: 
            - protein_input_ids (bsz, max_seq_len+2)  # bos and eos tokens
            - protein_attention_mask (bsz, max_seq_len+2)  # right padding
        - concatenated chat:
            - input_ids (bsz, max_prompt_len+max_text_len+1)
            - attention_mask (bsz, max_prompt_len+max_text_len+1)
            - labels (bsz, max_prompt_len+max_text_len+1)
        - standalone description for contrastive learning: 
            - description_input_ids (bsz, max_text_len+1)  # eos token only
            - description_attention_mask (bsz, max_text_len+1)  # right padding
            
        ids       = [left-pad + bos  + prompt & description + eot  + right-pad]
        mask      = [0s       + 1    + 1s     & 1s          + 1    + 0s       ]
        labels    = [-100s    + -100 + -100s  & description + eot  + -100s    ]
        desc_ids  =                         [ & description + eot  + right-pad]
        desc_mask =                         [ & 1s          + 1    + 0s       ] 
        
    * Inference mode (iterative generation):
        - graph related features: 
            None
        - amino-acid sequence: 
            - protein_input_ids (bsz, max_seq_len+2)  # bos and eos tokens
            - protein_attention_mask (bsz, max_seq_len+2)  # right padding
        - prompt chat: 
            - input_ids (bsz, max_prompt_len)
            - attention_mask (bsz, max_prompt_len)
            - description_input_ids (bsz, max_text_len+1)  # for evaluation

        ids      = [left-pad + bos + prompt & ]
        mask     = [0s       + 1   + 1s     & ]
        desc_ids =                        [ & description + eot + right-pad]

Example of usage: 
>>> from torch.utils.data import DataLoader
>>> from transformers import AutoTokenizer
>>> from dataset import Prot2TextLightDataset, Prot2TextLightCollater
>>> esm_tokenizer = AutoTokenizer.from_pretrained("/data/esm2_t33_650M_UR50D")
>>> llama_tokenizer = AutoTokenizer.from_pretrained(
        "/data/Meta-Llama-3.1-8B-Instruct-hf", 
        pad_token='<|reserved_special_token_0|>'
    )
>>> train_dataset = Prot2TextLightDataset("./data/train.csv")
>>> train_collater = Prot2TextLightCollater(
        sequence_tokenizer=esm_tokenizer,
        description_tokenizer=llama_tokenizer,
        mode="train"
    )
>>> train_dataloader = DataLoader(
        train_dataset,
        batch_size=4,
        shuffle=True,
        num_workers=4,
        collate_fn=train_collater, 
        pin_memory=True, 
        drop_last=True
    )
"""

import random
from typing import Dict, List, Literal, Optional

import pandas as pd
import torch
import torch.utils.data
from transformers import PreTrainedTokenizer
import os
import json
from .dataloader_refferring import FragRefDataset
from .templates import *

# grounding single
class FragGroundingSingle(FragRefDataset):
    def __init__(
            self, 
            root_dir: str, 
            data_name:str, 
            split: str,
            task_type: str,
            max_sequence_length: Optional[int] = 1021,
            **kwargs,
            ):
        super().__init__(
            root_dir=root_dir, 
            data_name=data_name, 
            split=split, 
            task_type=task_type, 
            max_sequence_length=max_sequence_length, 
            **kwargs,
            )
        self.data_infos = self._filter_grounding(self.data_infos)
        self.use_detailed_template = True

    # 过滤掉所有片段最大位置和最小位置之差大于max_sequence_length的data
    def _filter_grounding(self, data_infos):
        filtered_data_infos = []
        dataset_idx = 0
        for info in data_infos:
            frags = info["frags"]
            start_pos_list = [frag["start_position"] for frag in frags]
            end_pos_list = [frag["end_position"] for frag in frags]
            if max(end_pos_list) - min(start_pos_list) + 1 <= self.max_sequence_length:
                info["dataset_idx"] = dataset_idx
                dataset_idx += 1
                filtered_data_infos.append(info)
        print('\033[92m' + "-----{}-{}-{}: Filtered {} data ----".format(self.data_name, self.task_type, self.split, len(data_infos) - len(filtered_data_infos)) + '\033[0m')
        return filtered_data_infos
    # 以Motif数据的Referring_Class为例
    def _load_annotations(self, ann_file):
        data_infos = []
        for item in ann_file:
            for motif in item["fragments"]:  # 一个item中可能包含多个motif类别
                data_item = {}
                # full sequence info
                data_item["uid"] = item["uid"]
                data_item["sequence"] = item["sequence"]
                # motif info
                data_item["interpro_id"] = motif["interpro_id"]
                data_item["category"] = motif["category"]
                data_item["shortname"] = motif["shortname"]
                data_item["description"] = motif["description"]
                # fragment info
                data_item["frags"] = motif["frags"]
                data_infos.append(data_item)
        return data_infos
    
    # def create_conversations(self, sequence, answer, position_grd):
    #     question_template = random.choice(self.question_template)
    #     answer_template = random.choice(self.answer_template) if self.answer_template is not None else None
    #     conversation = [
    #         {"role": "system", "content": self.system_message},
    #         {"role": "user", "content": question_template.format(full_sequence=self.sequence_placeholder * (len(sequence)+2), N=len(sequence), class_name=answer)
    #          }
    #     ]
    #     position = ""
    #     # for i, (start, end) in enumerate(position_grd[0]):
    #     #     # position +=  f"{self.pos_start_placeholder}({start},{end}){self.pos_end_placeholder}"
    #     #     position +=  f"({start},{end})"
    #     #     if i < len(position_grd[0]) - 2:
    #     #         position += ","
    #     #     elif i == len(position_grd[0]) - 2:
    #     #         position += " and "
    #     for i, (start, end) in enumerate(position_grd[0]):
    #         position +=  f"({self.pos_start_placeholder}, {self.pos_end_placeholder})"
    #         if i < len(position_grd[0]) - 2:
    #             position += ","
    #         elif i == len(position_grd[0]) - 2:
    #             position += " and "
    #     position = f"{self.phrase_start_placeholder}{answer}:{position}{self.phrase_end_placeholder}"
    #     answer = answer_template.format(class_name=answer, position=position)
    #     return conversation, answer

    def create_conversations(self, sequence, answer, position_grd):
        # Choose question template based on detailed template setting
        if self.use_detailed_template:
            question_template = Frag_Ground_Single_Detailed[0]  # Only one template as requested
            answer_template = Grounding_Answer_Single_Detailed[0]  # Only one template as requested
        else:
            question_template = random.choice(self.question_template)
            answer_template = random.choice(self.answer_template) if self.answer_template is not None else None
            
        conversation = [
            {"role": "system", "content": self.system_message},
            {"role": "user", "content": question_template.format(full_sequence=self.sequence_placeholder * (len(sequence)+2), N=len(sequence), class_name=answer)
             }
        ]
        
        if self.use_detailed_template:
            # New detailed template format
            region_count = len(position_grd[0])
            detailed_positions = []
            
            for i, (start, end) in enumerate(position_grd[0]):
                region_num = i + 1
                position_placeholder =  f"({self.pos_start_placeholder}, {self.pos_end_placeholder})"
                detailed_position = f"region {region_num} lies within {position_placeholder}"
                detailed_positions.append(detailed_position)
            
            # Join with appropriate connectors
            if len(detailed_positions) == 1:
                detailed_positions_str = detailed_positions[0]
            elif len(detailed_positions) == 2:
                detailed_positions_str = f"{detailed_positions[0]} and {detailed_positions[1]}"
            else:
                detailed_positions_str = ", ".join(detailed_positions[:-1]) + f", and {detailed_positions[-1]}"
            
            answer = answer_template.format(
                class_name=answer, 
                region_count=region_count,
                detailed_positions=f"{self.phrase_start_placeholder}{answer}:{detailed_positions_str}{self.phrase_end_placeholder}"
            )
        else:
            # Original template format
            position = ""
            for i, (start, end) in enumerate(position_grd[0]):
                # Only use start token for ProteinSAM (as requested)
                position +=  f"({self.pos_start_placeholder}, {self.pos_end_placeholder})"
                if i < len(position_grd[0]) - 2:
                    position += ","
                elif i == len(position_grd[0]) - 2:
                    position += " and "
            position = f"{self.phrase_start_placeholder}{answer}:{position}{self.phrase_end_placeholder}"
            answer = answer_template.format(class_name=answer, position=position)
            
        return conversation, answer

    def sort_position(self, position_grd):
        position_grd.sort(key=lambda x: x[0])
        return position_grd
    
    def process_data(self, data_item):
        sequence = data_item["sequence"]
        answer = data_item["category"]
        frags = data_item["frags"]
        start_pos_list = [frag["start_position"] for frag in frags]
        end_pos_list = [frag["end_position"] for frag in frags]
        if len(sequence) > self.max_sequence_length and not self.filter_sequence:
           # 截断窗口的起点不能晚于 fragment 的起点，否则会切掉 fragment 的开头
            max_start = min(start_pos_list)
            # 截断窗口的起点不能早于某个位置，否则窗口的结尾会切掉 fragment 的结尾
            min_start = max(0, max(end_pos_list) - self.max_sequence_length + 1)
            # 有效范围 [min_start, max_start] 内随机选择一个起点并截断
            assert min_start <= max_start, f"min_start: {min_start}, max_start: {max_start}, max_len: {self.max_sequence_length}, seq_len: {len(sequence)}"                
            start = random.randint(min_start, max_start)
            sequence = sequence[start:start + self.max_sequence_length]
            start_new_list = [start_pos - start for start_pos in start_pos_list]
            end_new_list = [end_pos - start + 1 for end_pos in end_pos_list]
            position_grd = [[[start_new, end_new] for start_new, end_new in zip(start_new_list, end_new_list)]]
        else:
            start = 0
            position_grd = [[[start_pos, end_pos+1] for start_pos, end_pos in zip(start_pos_list, end_pos_list)]]
        # 将片段按初始位置排序
        position_grd[0] = self.sort_position(position_grd[0])
        conversation, answer = self.create_conversations(sequence, answer, position_grd)
        position_ref = None
        return {
                "sequence": sequence,
                "conversation": conversation,
                "answer": answer,
                "position_ref": position_ref,
                "position_grd": position_grd,
                "start": start,
                "dataset_idx": data_item["dataset_idx"]
            }
    
    def __getitem__(self, idx: int) -> Dict[str, str]:
        data_item = self.data_infos[idx]
        return self.process_data(data_item)

class DomainGroundingSingle(FragGroundingSingle):
    def __init__(
            self, 
            root_dir: str, 
            split: str, 
            max_sequence_length: Optional[int] = 1021,
            **kwargs,
            ):
        data_name = "VenusX_Dom"
        task_type = "grounding_single"
        question_template = Frag_Ground_Single
        answer_template = Grounding_Answer_Single
        super().__init__(
            root_dir=root_dir, 
            data_name=data_name, 
            split=split, 
            task_type=task_type, 
            max_sequence_length=max_sequence_length,
            question_template=question_template,
            answer_template=answer_template,
            **kwargs,
            )
class ActGroundingSingle(FragGroundingSingle):
    def __init__(
            self, 
            root_dir: str, 
            split: str, 
            max_sequence_length: Optional[int] = 1021,
            **kwargs,
            ):
        data_name = "VenusX_Act"
        task_type = "grounding_single"
        question_template = Frag_Ground_Single
        answer_template = Grounding_Answer_Single
        super().__init__(
            root_dir=root_dir, 
            data_name=data_name, 
            split=split, 
            task_type=task_type, 
            max_sequence_length=max_sequence_length,
            question_template=question_template,
            answer_template=answer_template,
            **kwargs,
            )
class BindIGroundingSingle(FragGroundingSingle):
    def __init__(
            self, 
            root_dir: str, 
            split: str, 
            max_sequence_length: Optional[int] = 1021,
            **kwargs,
            ):
        data_name = "VenusX_BindI"
        task_type = "grounding_single"
        question_template = Frag_Ground_Single
        answer_template = Grounding_Answer_Single
        super().__init__(
            root_dir=root_dir, 
            data_name=data_name, 
            split=split, 
            task_type=task_type, 
            max_sequence_length=max_sequence_length,
            question_template=question_template,
            answer_template=answer_template,
            **kwargs,
            )
class MotifGroundingSingle(FragGroundingSingle):
    def __init__(
            self, 
            root_dir: str, 
            split: str, 
            max_sequence_length: Optional[int] = 1021,
            **kwargs,
            ):
        data_name = "VenusX_Motif"
        task_type = "grounding_single"
        question_template = Frag_Ground_Single
        answer_template = Grounding_Answer_Single
        super().__init__(
            root_dir=root_dir, 
            data_name=data_name, 
            split=split, 
            task_type=task_type, 
            max_sequence_length=max_sequence_length,
            question_template=question_template,
            answer_template=answer_template,
            **kwargs,
            )
        
class EvoGroundingSingle(FragGroundingSingle):
    def __init__(
            self, 
            root_dir: str, 
            split: str, 
            max_sequence_length: Optional[int] = 1021,
            **kwargs,
            ):
        data_name = "VenusX_Evo"
        task_type = "grounding_single"
        question_template = Frag_Ground_Single
        answer_template = Grounding_Answer_Single
        super().__init__(
            root_dir=root_dir, 
            data_name=data_name, 
            split=split, 
            task_type=task_type, 
            max_sequence_length=max_sequence_length,
            question_template=question_template,
            answer_template=answer_template,
            **kwargs,
            )

# grounding group
class FragGroundingGroup(FragRefDataset):
    def __init__(
            self, 
            root_dir: str, 
            data_name:str, 
            split: str,
            task_type: str,
            max_sequence_length: Optional[int] = 1021,
            **kwargs,
            ):
        super().__init__(
            root_dir=root_dir, 
            data_name=data_name, 
            split=split, 
            task_type=task_type, 
            max_sequence_length=max_sequence_length, 
            **kwargs,
            )
        self.data_infos = self._filter_grounding(self.data_infos)

    # 过滤掉所有片段最大位置和最小位置之差大于max_sequence_length的data
    def _filter_grounding(self, data_infos):
        filtered_data_infos = []
        dataset_idx = 0
        for info in data_infos:
            frags = info["fragments"]
            start_pos_list = []
            end_pos_list = []
            for frag in frags:
                start_pos_list.extend([frag_item["start_position"] for frag_item in frag["frags"]])
                end_pos_list.extend([frag_item["end_position"] for frag_item in frag["frags"]])
            if max(end_pos_list) - min(start_pos_list) + 1 <= self.max_sequence_length:
                info["dataset_idx"] = dataset_idx
                dataset_idx += 1
                filtered_data_infos.append(info)
        print('\033[92m' + "-----{}-{}-{}: Filtered {} data ----".format(self.data_name, self.task_type, self.split, len(data_infos) - len(filtered_data_infos)) + '\033[0m')
        return filtered_data_infos

    # 以Motif数据的Referring_Class为例
    def _load_annotations(self, ann_file):
        data_infos = []
        for item in ann_file:
            if item["completeness"] == "yes":
                data_item = {}
                data_item["uid"] = item["uid"]
                data_item["sequence"] = item["sequence"]
                data_item["fragments"] = item["fragments"]
                data_infos.append(data_item)
        return data_infos
    
    # def create_conversations(self, sequence, answer, position_grd):
    #     question_template = random.choice(self.question_template)
    #     answer_template = random.choice(self.answer_template) if self.answer_template is not None else None
    #     conversation = [
    #         {"role": "system", "content": self.system_message},
    #         {"role": "user", "content": question_template.format(full_sequence=self.sequence_placeholder * (len(sequence)+2), N=len(sequence), task_name=self.task_name_map[self.data_name])
    #          }
    #     ]
    #     answer_i = ""
    #     for j in range(len(answer)):
    #         position = ""
    #         for i, (start, end) in enumerate(position_grd[j]):
    #             # position +=  f"{self.pos_start_placeholder}({start},{end}){self.pos_end_placeholder}"
    #             position +=  f"({start},{end})"
    #             if i < len(position_grd[j]) - 2:
    #                 position += ", "
    #             elif i == len(position_grd[j]) - 2:
    #                 position += " and "
    #         answer_i += f"{answer[j]} at {position}"
    #         if j < len(answer) - 1:
    #             answer_i += "; "
    #         # elif j == len(answer) - 2:
    #         #     answer_i += " and "      
    #     answer = answer_template.format(task_name=self.task_name_map[self.data_name], contents=answer_i)
    #     return conversation, answer

    def create_conversations(self, sequence, answer, position_grd):
        question_template = random.choice(self.question_template)
        answer_template = random.choice(self.answer_template) if self.answer_template is not None else None
        conversation = [
            {"role": "system", "content": self.system_message},
            {"role": "user", "content": question_template.format(full_sequence=self.sequence_placeholder * (len(sequence)+2), N=len(sequence), task_name=self.task_name_map[self.data_name])
             }
        ]
        answer_i = ""
        for j in range(len(answer)):
            position = ""
            for i, (start, end) in enumerate(position_grd[j]):
                position +=  f"({self.pos_start_placeholder}, {self.pos_end_placeholder})"
                if i < len(position_grd[j]) - 2:
                    position += ", "
                elif i == len(position_grd[j]) - 2:
                    position += " and "
            answer_i += f"{self.phrase_start_placeholder}{answer[j]}:{position}{self.phrase_end_placeholder}"
            if j < len(answer) - 2:
                answer_i += "; "
            elif j == len(answer) - 2:
                answer_i += " and "      
        answer = answer_template.format(task_name=self.task_name_map[self.data_name], contents=answer_i)
        return conversation, answer

    def sort_position(self, position_grd):
        position_grd.sort(key=lambda x: x[0])
        return position_grd
    
    def process_data(self, data_item):
        sequence = data_item["sequence"]
        frags = data_item["fragments"]
        start_pos_list = []
        end_pos_list = []
        for frag in frags:
            start_pos_list.extend([frag_item["start_position"] for frag_item in frag["frags"]])
            end_pos_list.extend([frag_item["end_position"] for frag_item in frag["frags"]])
        # 确保截断sequence时，保证fragment的完整性
        if len(sequence) > self.max_sequence_length and not self.filter_sequence:
            # 截断窗口的起点不能晚于 fragment 的起点，否则会切掉 fragment 的开头
            max_start = min(start_pos_list)
            # 截断窗口的起点不能早于某个位置，否则窗口的结尾会切掉 fragment 的结尾
            min_start = max(0, max(end_pos_list) - self.max_sequence_length + 1)
            # 有效范围 [min_start, max_start] 内随机选择一个起点并截断
            assert min_start <= max_start, f"min_start: {min_start}, max_start: {max_start}, max_len: {self.max_sequence_length}, seq_len: {len(sequence)}"                
            start = random.randint(min_start, max_start)
            sequence = sequence[start:start + self.max_sequence_length]
        else:
            start = 0
        answer_list = []
        position_grd = []
        for frag in frags:
            answer_list.append(frag["category"])
            # position_grd.append([[frag_item["start_position"]-start, frag_item["end_position"]-start+1] for frag_item in frag["frags"]])
            position_temp = [[frag_item["start_position"]-start, frag_item["end_position"]-start+1] for frag_item in frag["frags"]]
            position_grd.append(self.sort_position(position_temp))
        conversation, answer = self.create_conversations(sequence, answer_list, position_grd)
        position_ref = None
        return {
                "sequence": sequence,
                "conversation": conversation,
                "answer": answer,
                "position_ref": position_ref,
                "position_grd": position_grd,
                "start": start,
                "dataset_idx": data_item["dataset_idx"]
            }
    
    def __getitem__(self, idx: int) -> Dict[str, str]:
        data_item = self.data_infos[idx]
        return self.process_data(data_item)


class DomainGroundingGroup(FragGroundingGroup):
    def __init__(
            self, 
            root_dir: str, 
            split: str, 
            max_sequence_length: Optional[int] = 1021,
            **kwargs,
            ):
        data_name = "VenusX_Dom"
        task_type = "grounding_group"
        question_template = Frag_Ground_Group
        answer_template = Grounding_Answer_Group
        super().__init__(
            root_dir=root_dir, 
            data_name=data_name, 
            split=split, 
            task_type=task_type, 
            max_sequence_length=max_sequence_length,
            question_template=question_template,
            answer_template=answer_template,
            **kwargs,
            )

class ActGroundingGroup(FragGroundingGroup):
    def __init__(
            self, 
            root_dir: str, 
            split: str, 
            max_sequence_length: Optional[int] = 1021,
            **kwargs,
            ):
        data_name = "VenusX_Act"
        task_type = "grounding_group"
        question_template = Frag_Ground_Group
        answer_template = Grounding_Answer_Group
        super().__init__(
            root_dir=root_dir, 
            data_name=data_name, 
            split=split, 
            task_type=task_type, 
            max_sequence_length=max_sequence_length,
            question_template=question_template,
            answer_template=answer_template,
            **kwargs,
            )

class BindIGroundingGroup(FragGroundingGroup):
    def __init__(
            self, 
            root_dir: str, 
            split: str, 
            max_sequence_length: Optional[int] = 1021,
            **kwargs,
            ):
        data_name = "VenusX_BindI"
        task_type = "grounding_group"
        question_template = Frag_Ground_Group
        answer_template = Grounding_Answer_Group
        super().__init__(
            root_dir=root_dir, 
            data_name=data_name, 
            split=split, 
            task_type=task_type, 
            max_sequence_length=max_sequence_length,
            question_template=question_template,
            answer_template=answer_template,
            **kwargs,
            )
class MotifGroundingGroup(FragGroundingGroup):
    def __init__(
            self, 
            root_dir: str, 
            split: str, 
            max_sequence_length: Optional[int] = 1021,
            **kwargs,
            ):
        data_name = "VenusX_Motif"
        task_type = "grounding_group"
        question_template = Frag_Ground_Group
        answer_template = Grounding_Answer_Group
        super().__init__(
            root_dir=root_dir, 
            data_name=data_name, 
            split=split, 
            task_type=task_type, 
            max_sequence_length=max_sequence_length,
            question_template=question_template,
            answer_template=answer_template,
            **kwargs,
            )
class EvoGroundingGroup(FragGroundingGroup):
    def __init__(
            self, 
            root_dir: str, 
            split: str, 
            max_sequence_length: Optional[int] = 1021,
            **kwargs,
            ):
        data_name = "VenusX_Evo"
        task_type = "grounding_group"
        question_template = Frag_Ground_Group
        answer_template = Grounding_Answer_Group
        super().__init__(
            root_dir=root_dir, 
            data_name=data_name, 
            split=split, 
            task_type=task_type, 
            max_sequence_length=max_sequence_length,
            question_template=question_template,
            answer_template=answer_template,
            **kwargs,
            )

if __name__ == "__main__":
    import numpy as np
    from transformers import AutoTokenizer
    from .dataloader_frag import FragDataCollator
    root_dir = "./data"
    def count_nested_elements_recursive(data):
        """
        使用递归方法提取多层嵌套列表中的每个元素。
        """
        data_list = []
        # 遍历列表中的每一个元素
        for element in data:
            # 如果元素是列表，则递归调用函数并将结果累加
            if isinstance(element, list):
                data_list.extend(count_nested_elements_recursive(element))
            # 如果元素不是列表，说明它是一个最里层的元素
            else:
                data_list.append(element)
        return data_list
    # data_name = "VenusX_Motif"
    # split = "test"
    # task_type = "referring_class"
    sequence_tokenizer = AutoTokenizer.from_pretrained(
        "/home/djy/projects/Data/HF_models/esm2_t36_3B_UR50D/"
        )
    llm_tokenizer = AutoTokenizer.from_pretrained(
        "/home/djy/projects/Data/HF_models/RedHatAI-Llama-3.1-8B-Instruct/",
        pad_token='<|reserved_special_token_0|>'
        )
    # dataset = FragDataset(
    #     root_dir=root_dir, 
    #     data_name=data_name, 
    #     split=split, 
    #     task_type=task_type, 
    #     question_template=Frag_Class,
    #     answer_template=Class_Answer,
    #     )

    # data_name = "VenusX_Dom"
    # split = "test"
    # task_type = "referring_desc"
    # dataset = FragDataset(
    #     root_dir=root_dir, 
    #     data_name=data_name, 
    #     split=split, 
    #     task_type=task_type, 
    #     question_template=Frag_Dom_Des,
    #     answer_template=None,
    #     )


    # split = "test"
    # dataset = DomainGroundingSingle(
    #     root_dir=root_dir, 
    #     split=split, 
    #     question_template=Frag_Ground_Single,
    #     answer_template=Grounding_Answer_Single,
    #     )
    split = "test"
    dataset = DomainGroundingGroup(
        root_dir=root_dir, 
        split=split, 
        )
    print(len(dataset))
    print(dataset[0])
    train_collater = FragDataCollator(
        sequence_tokenizer=sequence_tokenizer,
        llm_tokenizer=llm_tokenizer,
        mode="train", 
    )
    from torch.utils.data import DataLoader
    train_dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        num_workers=0,
        collate_fn=train_collater, 
        pin_memory=True, 
        drop_last=True
    )
    for batch in train_dataloader:
        # print(np.array(count_nested_elements_recursive(batch["position_grds"])).max())
        # if np.array(count_nested_elements_recursive(batch["position_grds"])).max() >= 1021:
        #     print(count_nested_elements_recursive(batch["position_grds"]))
        if np.array(count_nested_elements_recursive(batch["position_grds"])).min() == 0:
            print(count_nested_elements_recursive(batch["position_grds"]))
        # import pdb; pdb.set_trace()
        # break