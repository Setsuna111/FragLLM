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


class FunctionDataset(torch.utils.data.Dataset): 
    def __init__(
            self, 
            root_dir: str, # data directory
            data_name: str, 
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
        self.ann_file = pd.read_csv(os.path.join(self.root_dir, f"{self.data_name}/{self.split}.csv"))
        self.data_infos = self._load_annotations(self.ann_file)
        if self.filter_sequence:
            self.data_infos = self._filter_sequence(self.data_infos)
        self.question_template = question_template
        self.answer_template = answer_template

    # 以Motif数据的Referring_Class为例
    def _load_annotations(self, ann_file):
        data_infos = []
        dataset_idx = 0
        for i in range(len(ann_file)):
            data_item = {}
            data_item["sequence"] = ann_file.iloc[i]["sequence"]
            data_item["fullname"] = ann_file.iloc[i]["Full Name"]
            data_item["taxon"] = ann_file.iloc[i]["taxon"]
            data_item["description"] = ann_file.iloc[i]["function"]

            # debug info
            data_item["dataset_idx"] = dataset_idx
            dataset_idx += 1 
            
            data_infos.append(data_item)
        return data_infos
    
    def _filter_sequence(self, data_infos):
        filtered_data_infos = [info for info in data_infos if len(info["sequence"]) <= self.max_sequence_length]
        print('\033[92m' + "-----{}-{}-{}: Filtered {} data ----".format(self.data_name, self.task_type, self.split, len(data_infos) - len(filtered_data_infos)) + '\033[0m')
        return filtered_data_infos
    
    def create_conversations(self, sequence, answer, fullname, taxon):
        question_template = random.choice(self.question_template)
        answer_template = random.choice(self.answer_template) if self.answer_template is not None else None
        conversation = [
            {"role": "system", "content": self.system_message},
            {"role": "user", "content": question_template.format(fullname=fullname, taxon=taxon, full_sequence=self.sequence_placeholder * (len(sequence)+2))
             }
        ]
        answer = answer_template.format(class_name=answer) if self.answer_template is not None else answer
        return conversation, answer
    
    def process_data(self, data_item):
        sequence = data_item["sequence"]
        answer = data_item["description"]
        fullname = data_item["fullname"]
        taxon = data_item["taxon"]
        if len(sequence) > self.max_sequence_length and not self.filter_sequence:
            start = random.randint(0, len(sequence) - self.max_sequence_length)
            sequence = sequence[start:start + self.max_sequence_length]
        else:
            start = 0
        conversation, answer = self.create_conversations(sequence, answer, fullname, taxon)
        position_grd = None
        # start = 0
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

        
    def __len__(self) -> int:
        return len(self.data_infos)
    
    def __getitem__(self, idx: int) -> Dict[str, str]:
        data_item = self.data_infos[idx]
        return self.process_data(data_item)
    


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
    import time
    split = "train"
    start_time = time.time()
    dataset = FunctionDataset(
        root_dir=root_dir,
        data_name="Pro2Text",
        split=split, 
        task_type="function",
        question_template=ProteinFunction,
        answer_template=None,
        )
    print(len(dataset))
    print(dataset[0])
    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds")
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
        break