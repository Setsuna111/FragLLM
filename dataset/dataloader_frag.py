"""
DataLoader class for protein function prediction instruction tuning. To be used 
with Prot2TextInstructDataset for Esm2LlamaInstructForCausalLM. 

Every batch from DataLoader will contain following attributes:
    * Training mode (train-eval with teacher-forcing): 
        - graph related features: 
            - x: (sum_num_nodes, num_node_features)
            - edge_index: (2, sum_num_edges)
            - edge_type: (sum_num_edges,)
            - batch: (sum_num_nodes,)
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
            - x: (sum_num_nodes, num_node_features)
            - edge_index: (2, sum_num_edges)
            - edge_type: (sum_num_edges,)
            - batch: (sum_num_nodes,)
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
>>> from transformers import AutoTokenizer
>>> from dataset import Prot2TextInstructDataset, Prot2TextInstructDataLoader
>>> esm_tokenizer = AutoTokenizer.from_pretrained("/data/esm2_t33_650M_UR50D")
>>> llama_tokenizer = AutoTokenizer.from_pretrained(
        "/data/Meta-Llama-3.1-8B-Instruct-hf", 
        pad_token='<|reserved_special_token_0|>'
    )
>>> train_dataset = Prot2TextInstructDataset(
        root_dir="/data/Prot2Text-Llama3-Data/train", 
        csv_path="./data/train.csv", 
        sequence_tokenizer=esm_tokenizer, 
        description_tokenizer=llama_tokenizer, 
        skip_download=True,  # assume data is already downloaded
        skip_reload=True,  # assume data is already preprocessed
    )
>>> train_dataloader = Prot2TextInstructDataLoader(
        dataset=train_dataset, 
        mode="train", 
        batch_size=2, 
        shuffle=True, 
    )
"""


from typing import Dict, List, Literal, Optional, Union

import torch
import torch.utils.data
import torch_geometric.loader.dataloader
from transformers import PreTrainedTokenizer
from .dataloader_refferring import *
from .dataloader_grounding import *
from .dataloader_function import *
from transformers import AutoTokenizer
from torch.utils.data import ConcatDataset
import copy
import numpy as np
from dataclasses import dataclass, field
from .templates import *

class FragDataCollator:
    def __init__(self, 
            sequence_tokenizer: PreTrainedTokenizer,
            llm_tokenizer: PreTrainedTokenizer,
            mode: Literal["train", "inference"] = "train",
            max_sequence_length: Optional[int] = 1021, 
            ):
        self.sequence_tokenizer = sequence_tokenizer
        self.llm_tokenizer = llm_tokenizer
        self.mode = mode
        self.max_sequence_length = max_sequence_length

    def __call__(self, batch: List[Dict[str, str]]) -> Dict[str, torch.Tensor]:
        sequences = [item["sequence"] for item in batch]
        conversations = [item["conversation"] for item in batch]
        answers = [item["answer"] for item in batch]
        position_refs = [item["position_ref"] for item in batch]
        position_grds = [item["position_grd"] for item in batch]
        starts = [item["start"] for item in batch]
        # truncate and tokenize sequences
        self.sequence_tokenizer.padding_side = "right"
        tokenized_sequences = self.sequence_tokenizer(
            sequences, 
            truncation=True, 
            padding="longest", 
            max_length=self.max_sequence_length + 2,  # including bos and eos tokens of esm tokenizer
            return_tensors="pt"
        )
        sequence_input_ids = tokenized_sequences["input_ids"]
        sequence_attention_mask = tokenized_sequences["attention_mask"]

        self.llm_tokenizer.padding_side = "left"
        tokenized_prompts = self.llm_tokenizer.apply_chat_template(
            conversations, 
            add_generation_prompt=True, 
            tokenize=True, 
            padding="longest", 
            return_tensors="pt", 
            return_dict=True
        )
        prompt_input_ids = tokenized_prompts["input_ids"]
        prompt_attention_mask = tokenized_prompts["attention_mask"]

        # tokenize descriptions
        self.llm_tokenizer.padding_side = "right"
        tokenized_answers = self.llm_tokenizer(
            [answer + self.llm_tokenizer.eos_token for answer in answers], 
            add_special_tokens=False,  # do not add bos token to the beginning
            truncation=True, 
            padding="longest", 
            return_tensors="pt"
        )
        answer_input_ids = tokenized_answers["input_ids"]
        answer_attention_mask = tokenized_answers["attention_mask"]

        # truncate descriptions
        # if (self.max_description_length is not None) and (answer_input_ids.size(1) > self.max_description_length):
        #     answer_input_ids = answer_input_ids[:, :self.max_description_length]
        #     answer_attention_mask = answer_attention_mask[:, :self.max_description_length]

        # prepare labels
        labels = answer_input_ids.clone()
        labels[answer_attention_mask == 0] = -100
        # assemble
        if self.mode == "train": 
            return {
                "protein_input_ids": sequence_input_ids, 
                "protein_attention_mask": sequence_attention_mask, 
                "input_ids": torch.cat([
                    prompt_input_ids, 
                    answer_input_ids, 
                ], dim=1), 
                "attention_mask": torch.cat([
                    prompt_attention_mask, 
                    answer_attention_mask, 
                ], dim=1),
                "labels": torch.cat([
                    torch.full_like(
                        prompt_input_ids, 
                        fill_value=-100, 
                    ), 
                    labels,
                ], dim=1), 
                "answer_input_ids": answer_input_ids,
                "answer_attention_mask": answer_attention_mask,
                "position_refs": position_refs,
                "position_grds": position_grds,
                "starts": starts,
            }

        elif self.mode == "inference":
            return {
                "protein_input_ids": sequence_input_ids, 
                "protein_attention_mask": sequence_attention_mask, 
                "input_ids": prompt_input_ids, 
                "attention_mask": prompt_attention_mask, 
                "answer_input_ids": answer_input_ids, 
                "answer_attention_mask": answer_attention_mask,
                "position_refs": position_refs,
                "position_grds": position_grds,
                "starts": starts,
            }

        else: 
            raise ValueError(f"Invalid mode: {self.mode}")
        

# Hybrid Dataset

class HybridDatasetBase(torch.utils.data.Dataset):
    def __init__(self, 
            root_dir: str, 
            data_list: List[str], 
            sample_rate: str,
            split: str, 
            max_sequence_length: Optional[int] = 1021, 
            epoch_samples: Optional[int] = None,
            **kwargs,
            ):
        self.root_dir = root_dir
        self.dataset_config = {
            # protein function
            "ProFunction": FunctionDataset,
            # protein referring calss
            "ActRefClass": ActRefClass,
            "BindIRefClass": BindIRefClass,
            "DomRefClass": DomainRefClass,
            "EvoRefClass": EvoRefClass,
            "MotifRefClass": MotifRefClass,
            # protein referring description
            "ActRefDesc": ActRefDesc,
            "BindIRefDesc": BindIRefDesc,
            "DomRefDesc": DomainRefDesc,
            "EvoRefDesc": EvoRefDesc,
            "MotifRefDesc": MotifRefDesc,
            # protein grounding single
            "ActGroundSingle": ActGroundingSingle,
            "BindIGroundSingle": BindIGroundingSingle,
            "DomGroundSingle": DomainGroundingSingle,
            "EvoGroundSingle": EvoGroundingSingle,
            "MotifGroundSingle": MotifGroundingSingle,
            # protein grounding group
            "ActGroundGroup": ActGroundingGroup,
            "BindIGroundGroup": BindIGroundingGroup,
            "DomGroundGroup": DomainGroundingGroup,
            "EvoGroundGroup": EvoGroundingGroup,
            "MotifGroundGroup": MotifGroundingGroup,
        }
        self.max_sequence_length = max_sequence_length
        self.data_list = data_list
        self.kwargs = kwargs
        self.dataset_list = []
        self.split = split
        self.sample_rate = np.array([float(x) for x in sample_rate.split(",")]) if sample_rate is not None else np.array([1] * len(self.dataset_list))

        self.sample_rate = self.sample_rate.astype(np.float64)
        self.sample_rate /= self.sample_rate.sum()
        self.all_datasets = self.create_datasets()
        self.epoch_samples = epoch_samples if epoch_samples is not None else sum(len(item) for item in self.all_datasets)
    
    def create_datasets(self):
        for ds in self.data_list:
            dataset_class = self.dataset_config[ds]
            if ds == "ProFunction":
                dataset = FunctionDataset(
                    root_dir=self.root_dir, 
                    split=self.split, 
                    max_sequence_length=self.max_sequence_length, 
                    data_name="Pro2Text", 
                    task_type="function",
                    question_template=ProteinFunction, 
                    answer_template=None,
                    **self.kwargs)
            else:
                dataset = dataset_class(
                    root_dir=self.root_dir, 
                    split=self.split, 
                    max_sequence_length=self.max_sequence_length, 
                    **self.kwargs)
            self.dataset_list.append(dataset)
        return self.dataset_list
    
    def __len__(self):
        return self.epoch_samples
    
    def __getitem__(self, idx):
        dataset_idx = np.random.choice(len(self.dataset_list), p=self.sample_rate)
        selected_dataset = self.all_datasets[dataset_idx]
        index = np.random.choice(len(selected_dataset))
        data = selected_dataset[index]
        return data
    

class HybridReferringDataset(HybridDatasetBase):
    def __init__(self, 
            root_dir: str, 
            data_reffring: str, 
            sample_rate_referring: str,
            split: str, 
            max_sequence_length: Optional[int] = 1021, 
            epoch_samples_referring: Optional[int] = None,
            **kwargs,
            ):
        data_list = data_reffring.split("||")
        sample_rate = sample_rate_referring
        super().__init__(
            root_dir=root_dir, 
            data_list=data_list, 
            sample_rate=sample_rate, 
            split=split, 
            max_sequence_length=max_sequence_length, 
            epoch_samples=epoch_samples_referring, 
            **kwargs,
            )
class HybridGroundingDataset(HybridDatasetBase):
    def __init__(self, 
            root_dir: str, 
            data_grounding: str, 
            sample_rate_grounding: str,
            split: str, 
            max_sequence_length: Optional[int] = 1021, 
            epoch_samples_grounding: Optional[int] = None,
            **kwargs,
            ):
        data_list = data_grounding.split("||")
        sample_rate = sample_rate_grounding
        super().__init__(
            root_dir=root_dir, 
            data_list=data_list, 
            sample_rate=sample_rate, 
            split=split, 
            max_sequence_length=max_sequence_length, 
            epoch_samples=epoch_samples_grounding, 
            **kwargs,
            )
class HybridFunctionDataset(HybridDatasetBase):
    def __init__(self, 
            root_dir: str, 
            data_function: str, 
            sample_rate_function: str,
            split: str, 
            max_sequence_length: Optional[int] = 1021, 
            epoch_samples_function: Optional[int] = None,
            **kwargs,
            ):
        data_list = data_function.split("||")
        sample_rate = sample_rate_function
        super().__init__(
            root_dir=root_dir, 
            data_list=data_list, 
            sample_rate=sample_rate, 
            split=split, 
            max_sequence_length=max_sequence_length, 
            epoch_samples=epoch_samples_function, 
            **kwargs,
            )



class ConcatDataset(ConcatDataset):
    def __init__(self, datasets):
        super().__init__(datasets)

    def collater(self, samples):

        all_keys = set()
        for s in samples:
            all_keys.update(s)
        shared_keys = all_keys
        for s in samples:
            shared_keys = shared_keys & set(s.keys())
        samples_shared_keys = []
        for s in samples:
            samples_shared_keys.append({k: s[k] for k in s.keys() if k in shared_keys})

        return self.datasets[0].collater(samples_shared_keys)



class HybridTrainDataset(HybridDatasetBase):
    def __init__(self, 
            root_dir: str, 
            data_train: str, 
            sample_rate_train: str, 
            split: str, 
            max_sequence_length: Optional[int] = 1021, 
            epoch_samples: Optional[int] = None,
            **kwargs,
            ):
        data_list = data_train.split("||")
        sample_rate = sample_rate_train
        super().__init__(
            root_dir=root_dir, 
            data_list=data_list, 
            sample_rate=sample_rate, 
            split=split, 
            max_sequence_length=max_sequence_length, 
            epoch_samples=epoch_samples, 
            **kwargs,
            )

class HybridValidDataset(HybridDatasetBase):
    def __init__(self, 
            root_dir: str, 
            data_valid: str, 
            sample_rate_valid: str, 
            split: str, 
            max_sequence_length: Optional[int] = 1021, 
            epoch_samples: Optional[int] = None,
            **kwargs,
            ):
        data_list = data_valid.split("||")
        sample_rate = sample_rate_valid
        super().__init__(
            root_dir=root_dir, 
            data_list=data_list, 
            sample_rate=sample_rate, 
            split=split, 
            max_sequence_length=max_sequence_length, 
            epoch_samples=epoch_samples, 
            **kwargs,
            )



def build_frag_dataset(
        dataset_config: Union[str, List[str]],
        data_args: Dict=None,
        data_split: str="train",
        **kwargs,
        ):
    if isinstance(dataset_config, list):
        datasets = []
        for cfg in dataset_config:
            temp_dataset = build_frag_dataset(cfg, data_args=data_args, data_split=data_split, **kwargs)
            datasets.append(temp_dataset)
        for dataset in datasets:
            print(type(dataset), f'len = {len(dataset)}')
        return ConcatDataset(datasets)
    dataset_type = dataset_config
    params_dict = copy.deepcopy(data_args.__dict__)
    params_dict["split"] = data_split
    if dataset_type == "ProFunction":  # protein function
        dataset = FunctionDataset(**params_dict, data_name="Pro2Text", task_type="function",question_template=ProteinFunction, answer_template=None)
    elif dataset_type == "ActRefClass":  # protein referring class
        dataset = ActRefClass(**params_dict)
    elif dataset_type == "BindIRefClass": 
        dataset = BindIRefClass(**params_dict)
    elif dataset_type == "DomRefClass":
        dataset = DomainRefClass(**params_dict)
    elif dataset_type == "EvoRefClass":
        dataset = EvoRefClass(**params_dict)
    elif dataset_type == "MotifRefClass":
        dataset = MotifRefClass(**params_dict)
    elif dataset_type == "ActRefDesc":  # protein referring description
        dataset = ActRefDesc(**params_dict)
    elif dataset_type == "BindIRefDesc":
        dataset = BindIRefDesc(**params_dict)
    elif dataset_type == "DomRefDesc":
        dataset = DomainRefDesc(**params_dict)
    elif dataset_type == "EvoRefDesc":
        dataset = EvoRefDesc(**params_dict)
    elif dataset_type == "MotifRefDesc":
        dataset = MotifRefDesc(**params_dict)
    elif dataset_type == "ActGroundSingle":  # protein grounding single
        dataset = ActGroundingSingle(**params_dict)
    elif dataset_type == "BindIGroundSingle":
        dataset = BindIGroundingSingle(**params_dict)
    elif dataset_type == "DomGroundSingle":
        dataset = DomainGroundingSingle(**params_dict)
    elif dataset_type == "EvoGroundSingle":
        dataset = EvoGroundingSingle(**params_dict)
    elif dataset_type == "MotifGroundSingle":
        dataset = MotifGroundingSingle(**params_dict)
    elif dataset_type == "ActGroundGroup":  # protein grounding group
        dataset = ActGroundingGroup(**params_dict)
    elif dataset_type == "BindIGroundGroup":
        dataset = BindIGroundingGroup(**params_dict)
    elif dataset_type == "DomGroundGroup":
        dataset = DomainGroundingGroup(**params_dict)
    elif dataset_type == "EvoGroundGroup":
        dataset = EvoGroundingGroup(**params_dict)
    elif dataset_type == "MotifGroundGroup":
        dataset = MotifGroundingGroup(**params_dict)
    else:
        raise NotImplementedError(f"Invalid dataset type: {dataset_type}")
    return dataset


# def make_multitask_dataset(data_args):
#     dataset_configs_train = data_args.dataset_train_config.split("||")
#     dataset_config_train = dataset_configs_train[0] if len(dataset_configs_train) == 1 else dataset_configs_train
#     train_dataset = build_frag_dataset(dataset_config_train, data_args=data_args, data_split="train")
#     data_collator = FragDataCollator(
#         sequence_tokenizer=data_args.sequence_tokenizer,
#         llm_tokenizer=data_args.llm_tokenizer,
#         mode="train",
#         max_sequence_length=data_args.max_sequence_length,
#     )
#     dataset_configs_eval = data_args.dataset_valid_config.split("||") if data_args.dataset_valid_config is not None else None
#     eval_dataset = build_frag_dataset(dataset_configs_eval, data_args=data_args, data_split="valid") if dataset_configs_eval is not None else None

#     return dict(train_dataset=train_dataset,
#                 eval_dataset=eval_dataset,
#                 data_collator=data_collator)


def make_multitask_dataset(data_args):
    train_dataset = HybridTrainDataset(
        root_dir=data_args.root_dir,
        data_train=data_args.dataset_train_config,
        sample_rate_train=data_args.sample_rate_train,
        split="train",
        max_sequence_length=data_args.max_sequence_length,
    )
    data_collator = FragDataCollator(
        sequence_tokenizer=data_args.sequence_tokenizer,
        llm_tokenizer=data_args.llm_tokenizer,
        mode="train",
        max_sequence_length=data_args.max_sequence_length,
    )
    eval_dataset = HybridValidDataset(
        root_dir=data_args.root_dir,
        data_valid=data_args.dataset_valid_config,
        sample_rate_valid=data_args.sample_rate_valid,
        split="valid",
        max_sequence_length=data_args.max_sequence_length,
    ) if data_args.dataset_valid_config is not None else None

    return dict(train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                data_collator=data_collator)

@dataclass
class FragDataArguments:
    """Data arguments for fragment training."""
    root_dir: Optional[str] = field(default="./data", metadata={"help": "Root directory for datasets"})
    dataset_train_config: Optional[str] = field(default="ProFunction||ActRefClass||BindIRefClass||DomRefClass||EvoRefClass||MotifRefClass", metadata={"help": "Dataset config for training"})
    sample_rate_train: Optional[str] = field(default="1,1,1,1,1,1", metadata={"help": "Sample rate for training"})
    dataset_valid_config: Optional[str] = field(default="ProFunction", metadata={"help": "Dataset config for evaluation"})
    sample_rate_valid: Optional[str] = field(default="1", metadata={"help": "Sample rate for evaluation"})
    sequence_tokenizer_path: Optional[str] = field(default="/home/djy/projects/Data/HF_models/esm2_t36_3B_UR50D", metadata={"help": "Sequence tokenizer"})
    llm_tokenizer_path: Optional[str] = field(default="/home/djy/projects/Data/HF_models/RedHatAI-Llama-3.1-8B-Instruct", metadata={"help": "LLM tokenizer"})
    max_sequence_length: Optional[int] = field(default=1021, metadata={"help": "Maximum sequence length"})
    filter_sequence: Optional[bool] = field(default=False, metadata={"help": "Whether to filter sequence"})
    sequence_placeholder: Optional[str] = field(default="<|reserved_special_token_1|>", metadata={"help": "Sequence placeholder"})
    fragment_placeholder: Optional[str] = field(default="<|reserved_special_token_2|>", metadata={"help": "Fragment placeholder"})
    pos_start_placeholder: Optional[str] = field(default="<|reserved_special_token_3|>", metadata={"help": "Position start placeholder"})
    pos_end_placeholder: Optional[str] = field(default="<|reserved_special_token_4|>", metadata={"help": "Position end placeholder"})
    system_message: Optional[str] = field(default="You are a scientific assistant specializing in protein sequence analysis. Based on protein sequence embeddings and other related information, please answer the relevant questions using professional language. ", metadata={"help": "System message"})
    def __repr__(self):
        # 获取 dataclass 默认字段
        fields = {field.name: getattr(self, field.name) for field in self.__dataclass_fields__.values()}
        # 获取动态添加的字段
        dynamic_fields = {k: v for k, v in self.__dict__.items() if k not in fields}
        # 合并
        all_fields = {**fields, **dynamic_fields}
        return f"DataArguments({all_fields})"

if __name__ == "__main__":
    from torch.utils.data import DataLoader
    data_args = FragDataArguments()
    data_args.sequence_tokenizer = AutoTokenizer.from_pretrained(data_args.sequence_tokenizer_path)
    data_args.llm_tokenizer = AutoTokenizer.from_pretrained(data_args.llm_tokenizer_path,pad_token='<|reserved_special_token_0|>')
    data_module = make_multitask_dataset(data_args)
    # import pdb; pdb.set_trace()
    train_dataloader = DataLoader(
        data_module["train_dataset"],
        batch_size=4,
        shuffle=True,
        num_workers=0,
        collate_fn=data_module["data_collator"],
        pin_memory=True, 
        drop_last=True
    )
    for batch in train_dataloader:
        print(batch)
        break
        if (batch["input_ids"]==128003).sum() != (batch["protein_attention_mask"]==1).sum():
            import pdb; pdb.set_trace()
        # break
