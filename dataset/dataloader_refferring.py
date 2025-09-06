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
from .templates import *
# from .dataloader_frag import FragDataCollator
class FragRefDataset(torch.utils.data.Dataset): 
    def __init__(
            self, 
            root_dir: str, # data directory
            data_name: str, # dataset name (e.g. "Pro2Text", "Venux_Dom", "Venux_Act", "VenusX_BindI", "VenusX_Motif", "VenusX_Evo")
            split: str, # "train", "valid", "test"
            task_type: str, # referring_desc, referering_class, grounding_single, grounding_group, function
            max_sequence_length: Optional[int] = 1021, 
            system_message: str = (
            "You are a scientific assistant specializing in protein sequence analysis. Based on protein sequence embeddings and other related information, please answer the relevant questions using professional language. "
            ),
            question_template: List[str] = None,
            answer_template: List[str] = None,
            filter_sequence: bool = False,
            sequence_placeholder: str = '<|reserved_special_token_1|>',
            fragment_placeholder: str = '<|reserved_special_token_2|>',
            pos_start_placeholder: str = '<|reserved_special_token_3|>',
            pos_end_placeholder: str = '<|reserved_special_token_4|>',
            **kwargs,
            ):
        super().__init__()
        self.root_dir = root_dir
        self.data_name = data_name
        self.split = split
        self.task_type = task_type
        self.max_sequence_length = max_sequence_length
        self.system_message = system_message
        self.filter_sequence = filter_sequence
        self.sequence_placeholder = sequence_placeholder
        self.fragment_placeholder = fragment_placeholder
        self.pos_start_placeholder = pos_start_placeholder
        self.pos_end_placeholder = pos_end_placeholder
        if self.data_name == "Function":
            self.ann_file = pd.read_csv(os.path.join(self.root_dir, f"{self.data_name}/{self.split}.csv"))
        else:
            self.ann_file = json.load(open(os.path.join(self.root_dir, f"{self.data_name}/{self.split}.json")))
        self.data_infos = self._load_annotations(self.ann_file)
        if self.filter_sequence:
            self.data_infos = self._filter_sequence(self.data_infos)
        self.question_template = question_template
        self.answer_template = answer_template
        self.task_name_map = {
            "VenusX_Dom": "domain",
            "VenusX_Motif": "motif domain",
            "VenusX_BindI": "binding site",
            "VenusX_Evo": "conserved site",
            "VenusX_Act": "active site",
        }

    # 以Motif数据的Referring_Class为例
    def _load_annotations(self, ann_file):
        data_infos = []
        dataset_idx = 0
        for item in ann_file:
            for motif in item["fragments"]:  # 一个item中可能包含多个motif类别
                for frag in motif["frags"]:  # 一个motif类别中可能包含多个fragment
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
                    data_item["start_pos"] = frag["start_position"]
                    data_item["end_pos"] = frag["end_position"]
                    data_item["length"] = frag["length"]
                    data_item["frag_sequence"] = frag["sequence"]

                    # debug info
                    data_item["dataset_idx"] = dataset_idx
                    dataset_idx += 1 

                    data_infos.append(data_item)
        return data_infos
    
    def _filter_sequence(self, data_infos):
        filtered_data_infos = [info for info in data_infos if len(info["sequence"]) <= self.max_sequence_length]
        print('\033[92m' + "-----{}-{}-{}: Filtered {} data ----".format(self.data_name, self.task_type, self.split, len(data_infos) - len(filtered_data_infos)) + '\033[0m')
        return filtered_data_infos
    
    def create_conversations(self, sequence, answer):
        question_template = random.choice(self.question_template)
        answer_template = random.choice(self.answer_template) if self.answer_template is not None else None
        conversation = [
            {"role": "system", "content": self.system_message},
            {"role": "user", "content": question_template.format(full_sequence=self.sequence_placeholder * (len(sequence)+2), task_name=self.task_name_map[self.data_name], fragment=self.fragment_placeholder) if self.answer_template is not None else question_template.format(full_sequence=self.sequence_placeholder * (len(sequence)+2), fragment=self.fragment_placeholder)
             }
        ]
        answer = answer_template.format(class_name=answer) if self.answer_template is not None else answer
        return conversation, answer
    
    def __len__(self) -> int:
        return len(self.data_infos)
    
    def __getitem__(self, idx: int) -> Dict[str, str]:
        data_item = self.data_infos[idx]

        sequence = data_item["sequence"]
        start_pos = data_item["start_pos"]
        end_pos = data_item["end_pos"]

        # 长度处理 0904修改
        if len(sequence) > self.max_sequence_length and not self.filter_sequence:
            fragment_len = end_pos - start_pos + 1

            if fragment_len > self.max_sequence_length:
                # 当片段比窗口还长，截断片段自身
                start = start_pos
                sequence = sequence[start_pos : start_pos + self.max_sequence_length]
                start_new = 0
                end_new = self.max_sequence_length
            else:
                # 片段比窗口短，随机截断序列，但保证功能片段完整

                # 截断窗口的起点不能晚于 fragment 的起点，否则会切掉 fragment 的开头
                max_start = start_pos

                # 截断窗口的起点不能早于某个位置，否则窗口的结尾会切掉 fragment 的结尾
                min_start = end_pos - self.max_sequence_length + 1
                min_start = max(0, min_start)

                # 有效范围 [min_start, max_start] 内随机选择一个起点并截断
                assert min_start <= max_start, f"min_start: {min_start}, max_start: {max_start}, frag_len: {fragment_len}, max_len: {self.max_sequence_length}, seq_len: {len(sequence)}"                
                start = random.randint(min_start, max_start)

                sequence = sequence[start : start + self.max_sequence_length]
                start_new = start_pos - start
                end_new = end_pos - start + 1

        else:
            start = 0
            start_new = start_pos
            end_new = end_pos + 1
        
        answer = data_item["category"] if self.task_type == "referring_class" else data_item["description"]
        position_ref = [start_new, end_new]

        conversation, answer = self.create_conversations(sequence, answer)
        position_grd = None
        return {
                "sequence": sequence,
                "conversation": conversation,
                "answer": answer,
                "position_ref": position_ref,
                "position_grd": position_grd,
                "start": start,
                "dataset_idx": data_item["dataset_idx"]
        }

# referring description

class DomainRefDesc(FragRefDataset):
    def __init__(
            self, 
            root_dir: str,  
            split: str, 
            max_sequence_length: Optional[int] = 1021,
            **kwargs,
            ):
        data_name = "VenusX_Dom"
        task_type = "referring_desc"
        question_template = Frag_Dom_Des
        answer_template = None
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

class ActRefDesc(FragRefDataset):
    def __init__(
            self, 
            root_dir: str,  
            split: str, 
            max_sequence_length: Optional[int] = 1021,
            **kwargs,
            ):
        data_name = "VenusX_Act"
        task_type = "referring_desc"
        question_template = Frag_Act_Des
        answer_template = None
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

class BindIRefDesc(FragRefDataset):
    def __init__(
            self, 
            root_dir: str,  
            split: str, 
            max_sequence_length: Optional[int] = 1021,
            **kwargs,
            ):
        data_name = "VenusX_BindI"
        task_type = "referring_desc"
        question_template = Frag_BindI_Des
        answer_template = None
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
         
class EvoRefDesc(FragRefDataset):
    def __init__(
            self, 
            root_dir: str,  
            split: str, 
            max_sequence_length: Optional[int] = 1021,
            **kwargs,
            ):
        data_name = "VenusX_Evo"
        task_type = "referring_desc"
        question_template = Frag_Evo_Des
        answer_template = None
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

class MotifRefDesc(FragRefDataset):
    def __init__(
            self, 
            root_dir: str,  
            split: str, 
            max_sequence_length: Optional[int] = 1021,
            **kwargs,
            ):
        data_name = "VenusX_Motif"
        task_type = "referring_desc"
        question_template = Frag_Motif_Des
        answer_template = None
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

# referring classification
class DomainRefClass(FragRefDataset):
    def __init__(
            self, 
            root_dir: str,  
            split: str, 
            max_sequence_length: Optional[int] = 1021,
            **kwargs,
            ):
        data_name = "VenusX_Dom"
        task_type = "referring_class"
        question_template = Frag_Class
        answer_template = Class_Answer
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

class ActRefClass(FragRefDataset):
    def __init__(
            self, 
            root_dir: str,  
            split: str, 
            max_sequence_length: Optional[int] = 1021,
            **kwargs,
            ):
        data_name = "VenusX_Act"
        task_type = "referring_class"
        question_template = Frag_Class
        answer_template = Class_Answer
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
        
class BindIRefClass(FragRefDataset):
    def __init__(
            self, 
            root_dir: str,  
            split: str, 
            max_sequence_length: Optional[int] = 1021,
            **kwargs,
            ):
        data_name = "VenusX_BindI"
        task_type = "referring_class"
        question_template = Frag_Class
        answer_template = Class_Answer
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
        
class EvoRefClass(FragRefDataset):
    def __init__(
            self, 
            root_dir: str,  
            split: str, 
            max_sequence_length: Optional[int] = 1021,
            **kwargs,
            ):
        data_name = "VenusX_Evo"
        task_type = "referring_class"
        question_template = Frag_Class
        answer_template = Class_Answer
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

class MotifRefClass(FragRefDataset):
    def __init__(
            self, 
            root_dir: str,  
            split: str, 
            max_sequence_length: Optional[int] = 1021,
            **kwargs,
            ):
        data_name = "VenusX_Motif"
        task_type = "referring_class"
        question_template = Frag_Class
        answer_template = Class_Answer
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

    data_name = "VenusX_Dom"
    split = "test"
    task_type = "referring_desc"
    dataset = FragRefDataset(
        root_dir=root_dir, 
        data_name=data_name, 
        split=split, 
        task_type=task_type, 
        question_template=Frag_Dom_Des,
        answer_template=None,
        )


    # split = "test"
    # dataset = DomainGroundingSingle(
    #     root_dir=root_dir, 
    #     split=split, 
    #     question_template=Frag_Ground_Single,
    #     answer_template=Grounding_Answer_Single,
    #     )
    # split = "test"
    # dataset = DomainGroundingGroup(
    #     root_dir=root_dir, 
    #     split=split, 
    #     question_template=Frag_Ground_Group,
    #     answer_template=Grounding_Answer_Group,
    #     )
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