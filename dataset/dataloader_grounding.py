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
    
    def create_conversations(self, sequence, answer, position_grd):
        question_template = random.choice(self.question_template)
        answer_template = random.choice(self.answer_template) if self.answer_template is not None else None
        conversation = [
            {"role": "system", "content": self.system_message},
            {"role": "user", "content": question_template.format(full_sequence=self.sequence_placeholder * (len(sequence)+2), N=len(sequence), class_name=answer)
             }
        ]
        position = ""
        for i, (start, end) in enumerate(position_grd[0]):
            position +=  f"{self.pos_start_placeholder}({start},{end}){self.pos_end_placeholder}"
            if i < len(position_grd[0]) - 2:
                position += ","
            elif i == len(position_grd[0]) - 2:
                position += " and "
        answer = answer_template.format(class_name=answer, position=position)
        return conversation, answer
    
    def process_data(self, data_item):
        sequence = data_item["sequence"]
        answer = data_item["category"]
        frags = data_item["frags"]
        start_pos_list = [frag["start_position"] for frag in frags]
        end_pos_list = [frag["end_position"] for frag in frags]
        if len(sequence) > self.max_sequence_length and not self.filter_sequence:
            start = random.randint(0, len(sequence) - self.max_sequence_length)
            sequence = sequence[start:start + self.max_sequence_length]
            start_new_list = [start_pos - start for start_pos in start_pos_list]
            end_new_list = [end_pos - start + 1 for end_pos in end_pos_list]
            position_grd = [[[start_new, end_new] for start_new, end_new in zip(start_new_list, end_new_list)]]
        else:
            start = 0
            position_grd = [[[start_pos, end_pos+1] for start_pos, end_pos in zip(start_pos_list, end_pos_list)]]
        conversation, answer = self.create_conversations(sequence, answer, position_grd)
        position_ref = None
        return {
                "sequence": sequence,
                "conversation": conversation,
                "answer": answer,
                "position_ref": position_ref,
                "position_grd": position_grd,
                "start": start,
            }
    
    def __getitem__(self, idx: int) -> Dict[str, str]:
        data_item = self.data_infos[idx]
        sequence = data_item["sequence"]
        frags = data_item["frags"]
        start_pos_list = [frag["start_position"] for frag in frags]
        end_pos_list = [frag["end_position"] for frag in frags]
        # 确保截断sequence时，保证fragment的完整性
        if len(sequence) > self.max_sequence_length and not self.filter_sequence:
            start = random.randint(0, len(sequence) - self.max_sequence_length)
            sequence = sequence[start:start + self.max_sequence_length]
            start_new_list = [start_pos - start for start_pos in start_pos_list]
            end_new_list = [end_pos - start + 1 for end_pos in end_pos_list]
        else:
            start_new_list = start_pos_list
            end_new_list = [end_pos + 1 for end_pos in end_pos_list]
        while any(start_new < 0 or end_new > len(sequence) for start_new, end_new in zip(start_new_list, end_new_list)):
            idx = random.randint(0, len(self.data_infos) - 1)
            data_item = self.data_infos[idx]
            sequence = data_item["sequence"]
            frags = data_item["frags"]
            start_pos_list = [frag["start_position"] for frag in frags]
            end_pos_list = [frag["end_position"] for frag in frags]
            # 确保截断sequence时，保证fragment的完整性
            if len(sequence) > self.max_sequence_length and not self.filter_sequence:
                start = random.randint(0, len(sequence) - self.max_sequence_length)
                sequence = sequence[start:start + self.max_sequence_length]
                start_new_list = [start_pos - start for start_pos in start_pos_list]
                end_new_list = [end_pos - start + 1 for end_pos in end_pos_list]
            else:
                start_new_list = start_pos_list
                end_new_list = [end_pos + 1 for end_pos in end_pos_list]
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
                position +=  f"{self.pos_start_placeholder}({start},{end}){self.pos_end_placeholder}"
                if i < len(position_grd[j]) - 2:
                    position += ", "
                elif i == len(position_grd[j]) - 2:
                    position += " and "
            answer_i += f"{answer[j]} at {position}"
            if j < len(answer) - 2:
                answer_i += "; "
            elif j == len(answer) - 2:
                answer_i += " and "      
        answer = answer_template.format(task_name=self.task_name_map[self.data_name], contents=answer_i)
        return conversation, answer
    
    def process_data(self, data_item):
        sequence = data_item["sequence"]
        frags = data_item["fragments"]
        
        if len(sequence) > self.max_sequence_length and not self.filter_sequence:
            start = random.randint(0, len(sequence) - self.max_sequence_length)
            sequence = sequence[start:start + self.max_sequence_length]
        else:
            start = 0
        answer_list = []
        position_grd = []
        for frag in frags:
            answer_list.append(frag["category"])
            position_grd.append([[frag_item["start_position"]-start, frag_item["end_position"]-start+1] for frag_item in frag["frags"]])
        conversation, answer = self.create_conversations(sequence, answer_list, position_grd)
        position_ref = None
        return {
                "sequence": sequence,
                "conversation": conversation,
                "answer": answer,
                "position_ref": position_ref,
                "position_grd": position_grd,
                "start": start,
            }
    
    def __getitem__(self, idx: int) -> Dict[str, str]:
        data_item = self.data_infos[idx]
        sequence = data_item["sequence"]
        frags = data_item["fragments"]
        start_pos_list = []
        end_pos_list = []
        for frag in frags:
            start_pos_list.extend([frag_item["start_position"] for frag_item in frag["frags"]])
            end_pos_list.extend([frag_item["end_position"] for frag_item in frag["frags"]])
        # 确保截断sequence时，保证fragment的完整性
        if len(sequence) > self.max_sequence_length and not self.filter_sequence:
            start = random.randint(0, len(sequence) - self.max_sequence_length)
            sequence = sequence[start:start + self.max_sequence_length]
            start_new_list = [start_pos - start for start_pos in start_pos_list]
            end_new_list = [end_pos - start + 1 for end_pos in end_pos_list]
        else:
            start_new_list = start_pos_list
            end_new_list = [end_pos + 1 for end_pos in end_pos_list]
        while any(start_new < 0 or end_new > len(sequence) for start_new, end_new in zip(start_new_list, end_new_list)):
            idx = random.randint(0, len(self.data_infos) - 1)
            data_item = self.data_infos[idx]
            sequence = data_item["sequence"]
            frags = data_item["fragments"]
            start_pos_list = []
            end_pos_list = []
            for frag in frags:
                start_pos_list.extend([frag_item["start_position"] for frag_item in frag["frags"]])
                end_pos_list.extend([frag_item["end_position"] for frag_item in frag["frags"]])
            # 确保截断sequence时，保证fragment的完整性
            if len(sequence) > self.max_sequence_length and not self.filter_sequence:
                start = random.randint(0, len(sequence) - self.max_sequence_length)
                sequence = sequence[start:start + self.max_sequence_length]
                start_new_list = [start_pos - start for start_pos in start_pos_list]
                end_new_list = [end_pos - start + 1 for end_pos in end_pos_list]
            else:
                start_new_list = start_pos_list
                end_new_list = [end_pos + 1 for end_pos in end_pos_list]
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
    from transformers import AutoTokenizer
    from .dataloader_frag import FragDataCollator
    root_dir = "./data"
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
        batch_size=4,
        shuffle=True,
        num_workers=0,
        collate_fn=train_collater, 
        pin_memory=True, 
        drop_last=True
    )
    for batch in train_dataloader:
        print(batch)
        # break