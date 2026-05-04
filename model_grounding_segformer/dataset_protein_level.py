"""
Protein-level batching support for ProteinSAM evaluation.

Provides:
  - ProteinGroupedSampler: yields indices grouped by uid so each batch
    contains all fragments of one protein.
  - ProteinSAMCollatorWithESMCacheAndUID: adds 'uids' to the batch dict.
  - get_datasets_and_collator_with_esm_cache_protein_level: factory function.
"""

import os
from collections import defaultdict
from typing import Dict, Iterator, List, Optional, Tuple, Any

import torch
from torch.utils.data import Sampler
from transformers import EsmTokenizer, LlamaTokenizer

from dataset import (
    ProteinSAMDatasetWithESMCache,
    ProteinSAMCollatorWithESMCache,
    get_datasets_and_collator_with_esm_cache,
)


class ProteinGroupedSampler(Sampler):
    """
    Yields index batches where each batch contains all fragments of one protein.

    Usage with DataLoader:
        sampler = ProteinGroupedSampler(dataset)
        loader = DataLoader(dataset, batch_sampler=sampler, collate_fn=collator)

    The order of proteins is shuffled when shuffle=True; within each protein
    the fragment order follows the dataset index order.
    """

    def __init__(self, dataset: ProteinSAMDatasetWithESMCache, shuffle: bool = False):
        self.shuffle = shuffle

        # Group dataset indices by uid
        uid_to_indices: Dict[str, List[int]] = defaultdict(list)
        for idx, item in enumerate(dataset.data_infos):
            uid_to_indices[item["uid"]].append(idx)

        self.groups: List[List[int]] = list(uid_to_indices.values())

    def __iter__(self) -> Iterator[List[int]]:
        groups = self.groups
        if self.shuffle:
            import random
            groups = groups[:]
            random.shuffle(groups)
        for group in groups:
            yield group

    def __len__(self) -> int:
        return len(self.groups)


class ProteinSAMCollatorWithESMCacheAndUID(ProteinSAMCollatorWithESMCache):
    """
    Extends ProteinSAMCollatorWithESMCache to include 'uids' in the batch dict.
    Everything else is identical to the parent collator.
    """

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        batch_dict = super().__call__(batch)
        batch_dict["uids"] = [item["uid"] for item in batch]
        return batch_dict


def get_datasets_and_collator_with_esm_cache_protein_level(
    root_dir: str,
    data_name: str,
    esm_model_path: str,
    esm_embeddings_dir: str,
    llama_model_path: Optional[str] = None,
    max_sequence_length: int = 1021,
    max_text_length: int = 128,
    use_category_cache: bool = True,
    **dataset_kwargs
) -> Tuple[Dict[str, ProteinSAMDatasetWithESMCache], ProteinSAMCollatorWithESMCacheAndUID]:
    """
    Same as get_datasets_and_collator_with_esm_cache but returns a UID-aware
    collator. Pair with ProteinGroupedSampler for protein-level batching.
    """
    if not os.path.isdir(esm_embeddings_dir):
        raise ValueError(f"ESM embeddings directory not found: {esm_embeddings_dir}")

    datasets, _ = get_datasets_and_collator_with_esm_cache(
        root_dir=root_dir,
        data_name=data_name,
        esm_model_path=esm_model_path,
        esm_embeddings_dir=esm_embeddings_dir,
        llama_model_path=llama_model_path,
        max_sequence_length=max_sequence_length,
        max_text_length=max_text_length,
        use_category_cache=use_category_cache,
        **dataset_kwargs
    )

    esm_tokenizer = EsmTokenizer.from_pretrained(esm_model_path)

    llama_tokenizer = None
    if not use_category_cache and llama_model_path is not None:
        llama_tokenizer = LlamaTokenizer.from_pretrained(llama_model_path)
        if llama_tokenizer.pad_token is None:
            llama_tokenizer.pad_token = '<|reserved_special_token_0|>'

    collator = ProteinSAMCollatorWithESMCacheAndUID(
        esm_tokenizer=esm_tokenizer,
        llama_tokenizer=llama_tokenizer,
        max_protein_length=max_sequence_length,
        max_text_length=max_text_length,
        use_category_cache=use_category_cache,
        esm_hidden_size=0,
    )

    return datasets, collator
