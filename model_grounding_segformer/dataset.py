"""
Dataset and data collator for ProteinSAM model training.
Supports protein functional region grounding with single region queries.
"""

import json
import random
import torch
import torch.utils.data as data
from typing import Dict, List, Optional, Tuple, Any
from transformers import EsmTokenizer, LlamaTokenizer
import numpy as np
import os


class ProteinSAMDataset(data.Dataset):
    """
    Dataset for ProteinSAM training.
    Supports both single dataset and multi-dataset (concatenated) loading.
    Multi-dataset format: "VenusX_Dom||VenusX_Act||VenusX_BindI"
    """
    
    def __init__(
        self,
        root_dir: str,
        data_name: str,  # e.g., "VenusX_Dom" or "VenusX_Dom||VenusX_Act||VenusX_BindI"
        split: str,      # "train", "valid", "test"
        max_sequence_length: int = 1021,
        null_position_prob: float = 0.3,  # Probability to set position prompt to null
        random_position_prob: float = 0.2,  # Probability to set random position
        position_noise_std: float = 10.0,    # Standard deviation for position noise
        filter_long_sequences: bool = True,
        **kwargs
    ):
        self.root_dir = root_dir
        self.data_name = data_name
        self.split = split
        self.max_sequence_length = max_sequence_length
        self.null_position_prob = null_position_prob
        self.random_position_prob = random_position_prob
        self.position_noise_std = position_noise_std
        self.filter_long_sequences = filter_long_sequences
        
        # Parse multiple datasets if separated by ||
        self.data_names = self._parse_data_names(data_name)
        
        # Load data from all datasets
        self.data_infos = self._load_multi_dataset_data()
        
        print(f"Loaded {len(self.data_infos)} samples from {len(self.data_names)} dataset(s): {self.data_names} / {split}")
    
    def _parse_data_names(self, data_name: str) -> List[str]:
        """Parse data_name string to extract multiple dataset names."""
        return [name.strip() for name in data_name.split("||")] if "||" in data_name else [data_name]
    
    def _load_multi_dataset_data(self) -> List[Dict[str, Any]]:
        """Load and process data from multiple datasets."""
        all_processed_data = []
        
        for single_data_name in self.data_names:
            single_data = self._load_single_dataset_data(single_data_name)
            all_processed_data.extend(single_data)
            print(f"  - Loaded {len(single_data)} samples from {single_data_name}/{self.split}")
        
        return all_processed_data
    
    def _load_single_dataset_data(self, data_name: str) -> List[Dict[str, Any]]:
        """Load and process data from a single dataset."""
        data_path = os.path.join(self.root_dir, data_name, f"{self.split}.json")
        
        if not os.path.exists(data_path):
            print(f"Warning: Data file not found: {data_path}, skipping...")
            return []
        
        with open(data_path, 'r') as f:
            raw_data = json.load(f)
        
        processed_data = []
        
        for item in raw_data:
            sequence = item["sequence"]
            
            # Filter long sequences if requested
            if self.filter_long_sequences and len(sequence) > self.max_sequence_length:
                continue
            
            # Collect fragments grouped by category for this protein
            fragments_by_category = {}
            for fragment in item["fragments"]:
                category = fragment["category"]
                if category not in fragments_by_category:
                    fragments_by_category[category] = []
                
                for frag in fragment["frags"]:
                    fragment_info = {
                        "category": fragment["category"],
                        "start_position": frag["start_position"],
                        "end_position": frag["end_position"],
                        "fragment_sequence": frag["sequence"],
                        "description": fragment.get("description", ""),
                        "interpro_id": fragment.get("interpro_id", ""),
                        "shortname": fragment.get("shortname", "")
                    }
                    fragments_by_category[category].append(fragment_info)
            
            # Process each fragment in the item (original logic for single fragment samples)
            for fragment in item["fragments"]:
                for frag in fragment["frags"]:
                    data_sample = {
                        "uid": item["uid"],
                        "sequence": sequence,
                        "category": fragment["category"],
                        "start_position": frag["start_position"],
                        "end_position": frag["end_position"],
                        "fragment_sequence": frag["sequence"],
                        "description": fragment.get("description", ""),
                        "interpro_id": fragment.get("interpro_id", ""),
                        "shortname": fragment.get("shortname", ""),
                        "dataset_source": data_name,  # Add source dataset info
                        "same_category_fragments": fragments_by_category.get(fragment["category"], [])  # Add same-category fragments for multi-region task
                    }
                    processed_data.append(data_sample)
        
        return processed_data
    
    def _get_point_position(
        self, 
        start_pos: int, 
        end_pos: int, 
        sequence_length: int,
        training: bool = True
    ) -> Optional[int]:
        """
        Get point position for position prompt.
        
        Args:
            start_pos: Fragment start position
            end_pos: Fragment end position
            sequence_length: Total sequence length
            training: Whether in training mode
            
        Returns:
            Point position or None (for null prompt)
        """
        if training:
            # With some probability, return null position (for multi-region task)
            if random.random() < self.null_position_prob:
                return None
            
            # With some probability, return random position
            # if random.random() < self.random_position_prob:
            #     return random.randint(0, sequence_length - 1)
        else:
            # if random.random() < self.null_position_prob:
                # return None  # Multi-region task
            pass

        # Calculate center position with noise
        center_pos = (start_pos + end_pos) // 2
        
        if training and self.position_noise_std > 0:
            # Add Gaussian noise to center position
            noise = np.random.normal(0, self.position_noise_std)
            center_pos = int(center_pos + noise)
        
            # Clamp to valid range
            center_pos = max(0, min(center_pos, sequence_length - 1))
        
        return center_pos
    
    def _process_sequence(
        self, 
        sequence: str, 
        start_pos: int, 
        end_pos: int
    ) -> Tuple[str, int, int]:
        """
        Process sequence by truncating if necessary while preserving fragment.
        Uses randomized truncation for better data augmentation.
        
        Args:
            sequence: Full protein sequence
            start_pos: Fragment start position
            end_pos: Fragment end position
            
        Returns:
            Processed sequence, adjusted start position, adjusted end position
        """
        if len(sequence) <= self.max_sequence_length:
            return sequence, start_pos, end_pos
        
        # Need to truncate sequence while preserving the fragment
        fragment_length = end_pos - start_pos + 1
        
        if fragment_length > self.max_sequence_length:
            # Fragment itself is too long, take a portion of it
            truncated_sequence = sequence[start_pos:start_pos + self.max_sequence_length]
            return truncated_sequence, 0, self.max_sequence_length - 1
        
        # Randomized truncation strategy that preserves the fragment
        # Calculate valid range for truncation start position
        
        # Truncation window start cannot be later than fragment start, 
        # otherwise it would cut off the beginning of the fragment
        max_start = start_pos
        
        # Truncation window start cannot be too early, 
        # otherwise the window end would cut off the end of the fragment
        min_start = max(0, end_pos - self.max_sequence_length + 1)
        
        assert min_start <= max_start, "Invalid truncation range"
        
        # Randomly choose truncation start within valid range
        truncate_start = random.randint(min_start, max_start)
        
        # Ensure truncation doesn't exceed sequence bounds
        truncate_start = max(0, min(truncate_start, len(sequence) - self.max_sequence_length))
        truncate_end = truncate_start + self.max_sequence_length
        
        # Truncate sequence
        truncated_sequence = sequence[truncate_start:truncate_end]
        
        # Adjust positions relative to new sequence start
        new_start_pos = start_pos - truncate_start
        new_end_pos = end_pos - truncate_start
        
        # Ensure positions are valid (should always be true with correct logic)
        new_start_pos = max(0, new_start_pos)
        new_end_pos = min(len(truncated_sequence) - 1, new_end_pos)
        
        return truncated_sequence, new_start_pos, new_end_pos
    
    def __len__(self) -> int:
        return len(self.data_infos)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a single data sample.
        
        Returns:
            Dictionary containing processed data sample
        """
        data_item = self.data_infos[idx]
        
        sequence = data_item["sequence"]
        start_pos = data_item["start_position"]
        end_pos = data_item["end_position"]
        category = data_item["category"]
        
        # Process sequence (truncate if necessary)
        processed_sequence, adj_start_pos, adj_end_pos = self._process_sequence(
            sequence, start_pos, end_pos
        )
        
        # Get point position for position prompt
        point_position = self._get_point_position(
            adj_start_pos, 
            adj_end_pos, 
            len(processed_sequence),
            training=self.split == "train"
        )
        
        # If no point position, this becomes a multi-region detection task
        is_multi_region = point_position is None
        
        if is_multi_region:
            # For multi-region task, collect all fragments of the same category within the processed sequence
            same_category_fragments = data_item["same_category_fragments"]
            valid_fragments = []
            
            # Calculate truncation offset
            original_length = len(data_item["sequence"])
            processed_length = len(processed_sequence)
            
            if original_length > processed_length:
                # Find truncation offset by comparing original fragment position
                truncate_offset = start_pos - adj_start_pos
            else:
                truncate_offset = 0
            
            # Filter and adjust fragment positions based on processed sequence
            for frag in same_category_fragments:
                frag_start = frag["start_position"] - truncate_offset
                frag_end = frag["end_position"] - truncate_offset
                
                # Only include fragments that are fully within the processed sequence
                if frag_start >= 0 and frag_end < processed_length:
                    valid_fragments.append({
                        "category": frag["category"],
                        "start_position": frag_start,
                        "end_position": frag_end,
                        "description": frag["description"]
                    })
            
            return {
                "uid": data_item["uid"],
                "sequence": processed_sequence,
                "category": category,  # Keep original category for text prompt
                "point_position": point_position,  # None
                "is_multi_region": True,
                "same_category_fragments": valid_fragments,
                "description": data_item["description"],  # Use current functional region's description
                "original_start": start_pos,
                "original_end": end_pos
            }
        else:
            # Single region task (original logic)
            return {
                "uid": data_item["uid"],
                "sequence": processed_sequence,
                "category": category,
                "start_position": adj_start_pos,
                "end_position": adj_end_pos,
                "point_position": point_position,
                "is_multi_region": False,
                "description": data_item["description"],
                "original_start": start_pos,
                "original_end": end_pos
            }


class ProteinSAMCollator:
    """
    Data collator for ProteinSAM dataset.
    Handles tokenization and batch preparation.
    """
    
    def __init__(
        self,
        esm_tokenizer: EsmTokenizer,
        llama_tokenizer: Optional[LlamaTokenizer] = None,
        max_protein_length: int = 1024,
        max_text_length: int = 128,
        use_category_cache: bool = True
    ):
        self.esm_tokenizer = esm_tokenizer
        self.llama_tokenizer = llama_tokenizer
        self.max_protein_length = max_protein_length
        self.max_text_length = max_text_length
        self.use_category_cache = use_category_cache
        
        # Set pad token for Llama if not exists and if tokenizer is provided
        if self.llama_tokenizer is not None and self.llama_tokenizer.pad_token is None:
            self.llama_tokenizer.pad_token = '<|reserved_special_token_0|>'
    
    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Collate a batch of samples.
        
        Args:
            batch: List of data samples
            
        Returns:
            Batched tensors ready for model input
        """
        # Extract data
        sequences = [item["sequence"] for item in batch]
        categories = [item["category"] for item in batch]
        point_positions = [item["point_position"] for item in batch]
        is_multi_region = [item.get("is_multi_region", False) for item in batch]
        
        # Tokenize protein sequences
        protein_tokenized = self.esm_tokenizer(
            sequences,
            padding=True,
            truncation=True,
            max_length=self.max_protein_length+2,  # +2 for BOS/EOS
            return_tensors="pt"
        )
        
        # Handle text tokenization (only if not using cache)
        text_input_ids = None
        text_attention_mask = None
        
        if not self.use_category_cache and self.llama_tokenizer is not None:
            text_tokenized = self.llama_tokenizer(
                categories,
                padding=True,
                truncation=True,
                max_length=self.max_text_length,
                return_tensors="pt"
            )
            text_input_ids = text_tokenized["input_ids"]
            text_attention_mask = text_tokenized["attention_mask"]
        
        # Unified label construction - all samples get residue-level labels
        batch_size = len(batch)
        max_seq_len = protein_tokenized["input_ids"].shape[1] - 2  # Remove BOS/EOS tokens
        
        # Create unified residue-level labels (0: background, 1: functional region)
        residue_labels = torch.zeros(batch_size, max_seq_len, dtype=torch.long)
        
        for i, item in enumerate(batch):
            if item.get("is_multi_region", False):
                # Multi-region task: mark all fragments of the same category
                same_category_fragments = item.get("same_category_fragments", [])
                for frag in same_category_fragments:
                    start_pos = frag["start_position"]
                    end_pos = frag["end_position"]
                    # Ensure positions are within bounds
                    start_pos = max(0, min(start_pos, max_seq_len - 1))
                    end_pos = max(0, min(end_pos, max_seq_len - 1))
                    residue_labels[i, start_pos:end_pos+1] = 1
            else:
                # Single region task: mark single functional region
                start_pos = item["start_position"]
                end_pos = item["end_position"]
                # Ensure positions are within bounds
                start_pos = max(0, min(start_pos, max_seq_len - 1))
                end_pos = max(0, min(end_pos, max_seq_len - 1))
                residue_labels[i, start_pos:end_pos+1] = 1
        
        # Handle point positions (some might be None)
        point_tensor = torch.zeros(len(batch), dtype=torch.long)
        point_mask = torch.zeros(len(batch), dtype=torch.bool)
        
        for i, point_pos in enumerate(point_positions):
            if point_pos is not None:
                # Adjust point position for BOS token and ensure non-negative
                point_tensor[i] = max(0, point_pos)
                point_mask[i] = True
        
        batch_dict = {
            "protein_input_ids": protein_tokenized["input_ids"],
            "protein_attention_mask": protein_tokenized["attention_mask"],
            "point_positions": point_tensor,
            "point_mask": point_mask,
            "residue_labels": residue_labels,  # Unified residue-level labels
            "is_multi_region": is_multi_region,
            "categories": categories,
            "sequences": sequences
        }
        
        # Add text tokens only if available
        if text_input_ids is not None and text_attention_mask is not None:
            batch_dict.update({
                "text_input_ids": text_input_ids,
                "text_attention_mask": text_attention_mask
            })
        
        return batch_dict


def get_datasets_and_collator(
    root_dir: str,
    data_name: str,
    esm_model_path: str,
    llama_model_path: Optional[str] = None,
    max_sequence_length: int = 1021,
    max_text_length: int = 128,
    use_category_cache: bool = True,
    **dataset_kwargs
) -> Tuple[Dict[str, ProteinSAMDataset], ProteinSAMCollator]:
    """
    Create datasets and collator for ProteinSAM training.

    Args:
        root_dir: Root directory containing data
        data_name: Dataset name (e.g., "VenusX_Dom")
        esm_model_path: Path to ESM tokenizer
        llama_model_path: Path to Llama tokenizer (optional if using cache)
        max_sequence_length: Maximum protein sequence length
        max_text_length: Maximum text length
        use_category_cache: Whether to use category embeddings cache
        **dataset_kwargs: Additional arguments for dataset

    Returns:
        Dictionary of datasets and data collator
    """
    # Initialize tokenizers
    esm_tokenizer = EsmTokenizer.from_pretrained(esm_model_path)

    llama_tokenizer = None
    if not use_category_cache and llama_model_path is not None:
        llama_tokenizer = LlamaTokenizer.from_pretrained(llama_model_path)
        if llama_tokenizer.pad_token is None:
            llama_tokenizer.pad_token = '<|reserved_special_token_0|>'

    # Create datasets
    datasets = {}
    for split in ["train", "valid", "test"]:
        datasets[split] = ProteinSAMDataset(
            root_dir=root_dir,
            data_name=data_name,
            split=split,
            max_sequence_length=max_sequence_length,
            **dataset_kwargs
        )

    # Create collator
    collator = ProteinSAMCollator(
        esm_tokenizer=esm_tokenizer,
        llama_tokenizer=llama_tokenizer,
        max_protein_length=max_sequence_length,
        max_text_length=max_text_length,
        use_category_cache=use_category_cache
    )

    return datasets, collator


class ProteinSAMDatasetWithESMCache(ProteinSAMDataset):
    """
    Dataset for ProteinSAM training with pre-computed ESM embeddings (lazy loading).
    Extends ProteinSAMDataset to support loading ESM embeddings on-demand from a directory
    of per-uid .pt files, avoiding loading all embeddings into memory at once.

    Note: Pre-computed ESM embeddings should NOT include BOS/EOS tokens.
    The embeddings are stored as (seq_len, hidden_size) tensors where seq_len
    equals the actual protein sequence length.
    """

    def __init__(
        self,
        root_dir: str,
        data_name: str,
        split: str,
        esm_embeddings_dir: str,
        max_sequence_length: int = 1021,
        null_position_prob: float = 0.3,
        random_position_prob: float = 0.2,
        position_noise_std: float = 10.0,
        filter_long_sequences: bool = True,
        **kwargs
    ):
        self.esm_embeddings_dir = esm_embeddings_dir
        super().__init__(
            root_dir=root_dir,
            data_name=data_name,
            split=split,
            max_sequence_length=max_sequence_length,
            null_position_prob=null_position_prob,
            random_position_prob=random_position_prob,
            position_noise_std=position_noise_std,
            filter_long_sequences=filter_long_sequences,
            **kwargs
        )

        # Verify all UIDs have embedding files
        missing_uids = set()
        for item in self.data_infos:
            uid = item["uid"]
            if not os.path.exists(os.path.join(esm_embeddings_dir, f"{uid}.pt")):
                missing_uids.add(uid)

        if missing_uids:
            print(f"Warning: {len(missing_uids)} UIDs missing from ESM cache dir in {split} split")
            # Filter out samples without embedding files
            self.data_infos = [
                item for item in self.data_infos
                if os.path.exists(os.path.join(esm_embeddings_dir, f"{item['uid']}.pt"))
            ]
            print(f"Filtered to {len(self.data_infos)} samples with ESM embeddings")

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a single data sample with pre-computed ESM embedding (loaded on demand).

        Returns:
            Dictionary containing processed data sample with ESM embedding
        """
        # Get base sample from parent class
        sample = super().__getitem__(idx)

        # Get original data item for uid
        data_item = self.data_infos[idx]
        uid = data_item["uid"]

        # Lazy load: read only this uid's embedding file
        # The embedding is (full_seq_len, hidden_size) without BOS/EOS
        emb_path = os.path.join(self.esm_embeddings_dir, f"{uid}.pt")
        full_embedding = torch.load(emb_path, map_location="cpu", weights_only=True)  # (full_seq_len, hidden_size)

        # Handle sequence truncation - need to slice embedding accordingly
        original_sequence = data_item["sequence"]
        processed_sequence = sample["sequence"]

        if len(processed_sequence) < len(original_sequence):
            # Sequence was truncated, find where processed_sequence starts in original
            truncate_offset = original_sequence.find(processed_sequence)
            if truncate_offset == -1:
                # Fallback: use start position based calculation
                truncate_offset = data_item["start_position"] - sample.get("start_position", 0)
                truncate_offset = max(0, truncate_offset)

            embedding = full_embedding[truncate_offset:truncate_offset + len(processed_sequence)]
        else:
            embedding = full_embedding[:len(processed_sequence)]

        sample["esm_embedding"] = embedding  # (seq_len, hidden_size)
        sample["uid"] = uid

        return sample


class ProteinSAMCollatorWithESMCache:
    """
    Data collator for ProteinSAM dataset with pre-computed ESM embeddings.
    Handles batch preparation with ESM embeddings instead of tokenization.

    Note: ESM embeddings in the cache do NOT include BOS/EOS tokens.
    The collator pads embeddings and creates appropriate attention masks.
    """

    def __init__(
        self,
        esm_tokenizer: EsmTokenizer,  # Still needed for attention mask computation
        llama_tokenizer: Optional[LlamaTokenizer] = None,
        max_protein_length: int = 1024,
        max_text_length: int = 128,
        use_category_cache: bool = True,
        esm_hidden_size: int = 2560  # ESM 3B hidden size; inferred from batch if 0
    ):
        self.esm_tokenizer = esm_tokenizer
        self.llama_tokenizer = llama_tokenizer
        self.max_protein_length = max_protein_length
        self.max_text_length = max_text_length
        self.use_category_cache = use_category_cache
        self.esm_hidden_size = esm_hidden_size

        # Set pad token for Llama if not exists and if tokenizer is provided
        if self.llama_tokenizer is not None and self.llama_tokenizer.pad_token is None:
            self.llama_tokenizer.pad_token = '<|reserved_special_token_0|>'

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Collate a batch of samples with pre-computed ESM embeddings.

        Args:
            batch: List of data samples with esm_embedding field

        Returns:
            Batched tensors ready for model input
        """
        # Extract data
        sequences = [item["sequence"] for item in batch]
        categories = [item["category"] for item in batch]
        point_positions = [item["point_position"] for item in batch]
        is_multi_region = [item.get("is_multi_region", False) for item in batch]
        esm_embeddings = [item["esm_embedding"] for item in batch]

        batch_size = len(batch)

        # Find max sequence length in this batch (without BOS/EOS)
        seq_lengths = [emb.shape[0] for emb in esm_embeddings]
        max_seq_len = max(seq_lengths)
        hidden_size = esm_embeddings[0].shape[1] if self.esm_hidden_size == 0 else self.esm_hidden_size

        # Pad ESM embeddings to max length
        # Shape: (batch_size, max_seq_len, hidden_size)
        padded_embeddings = torch.zeros(batch_size, max_seq_len, hidden_size)

        # Create attention mask WITH BOS/EOS positions to match forward() expectation
        # Forward function will slice it with [:, 1:-1] to get actual sequence mask
        # So we create mask of shape (batch_size, max_seq_len + 2) with 1s at BOS/EOS positions
        attention_mask = torch.zeros(batch_size, max_seq_len + 2, dtype=torch.long)

        for i, emb in enumerate(esm_embeddings):
            seq_len = emb.shape[0]
            padded_embeddings[i, :seq_len, :] = emb
            # Set BOS position (index 0) = 1
            attention_mask[i, 0] = 1
            # Set actual sequence positions (index 1 to seq_len) = 1
            attention_mask[i, 1:seq_len+1] = 1
            # Set EOS position (index seq_len + 1) = 1
            attention_mask[i, seq_len + 1] = 1
            # 注意这里即使序列emb已经去掉了EOS和BOS，attention_mask还要再虚构上，这样主模型1：-1才对。。

        # Handle text tokenization (only if not using cache)
        text_input_ids = None
        text_attention_mask = None

        if not self.use_category_cache and self.llama_tokenizer is not None:
            text_tokenized = self.llama_tokenizer(
                categories,
                padding=True,
                truncation=True,
                max_length=self.max_text_length,
                return_tensors="pt"
            )
            text_input_ids = text_tokenized["input_ids"]
            text_attention_mask = text_tokenized["attention_mask"]

        # Create unified residue-level labels (0: background, 1: functional region)
        residue_labels = torch.zeros(batch_size, max_seq_len, dtype=torch.long)

        for i, item in enumerate(batch):
            if item.get("is_multi_region", False):
                # Multi-region task: mark all fragments of the same category
                same_category_fragments = item.get("same_category_fragments", [])
                for frag in same_category_fragments:
                    start_pos = frag["start_position"]
                    end_pos = frag["end_position"]
                    # Ensure positions are within bounds
                    start_pos = max(0, min(start_pos, max_seq_len - 1))
                    end_pos = max(0, min(end_pos, max_seq_len - 1))
                    residue_labels[i, start_pos:end_pos+1] = 1
            else:
                # Single region task: mark single functional region
                start_pos = item["start_position"]
                end_pos = item["end_position"]
                # Ensure positions are within bounds
                start_pos = max(0, min(start_pos, max_seq_len - 1))
                end_pos = max(0, min(end_pos, max_seq_len - 1))
                residue_labels[i, start_pos:end_pos+1] = 1

        # Handle point positions (some might be None)
        point_tensor = torch.zeros(batch_size, dtype=torch.long)
        point_mask = torch.zeros(batch_size, dtype=torch.bool)

        for i, point_pos in enumerate(point_positions):
            if point_pos is not None:
                point_tensor[i] = max(0, point_pos)
                point_mask[i] = True

        batch_dict = {
            "external_esm_embeddings": padded_embeddings,  # (batch_size, seq_len, hidden_size) - actual protein embeddings without BOS/EOS
            "protein_attention_mask": attention_mask,  # (batch_size, seq_len+2) - WITH BOS/EOS positions for forward() compatibility
            "point_positions": point_tensor,
            "point_mask": point_mask,
            "residue_labels": residue_labels,  # (batch_size, seq_len) - actual sequence length without BOS/EOS
            "is_multi_region": is_multi_region,
            "categories": categories,
            "sequences": sequences
        }

        # Add text tokens only if available
        if text_input_ids is not None and text_attention_mask is not None:
            batch_dict.update({
                "text_input_ids": text_input_ids,
                "text_attention_mask": text_attention_mask
            })

        return batch_dict


def get_datasets_and_collator_with_esm_cache(
    root_dir: str,
    data_name: str,
    esm_model_path: str,
    esm_embeddings_dir: str,
    llama_model_path: Optional[str] = None,
    max_sequence_length: int = 1021,
    max_text_length: int = 128,
    use_category_cache: bool = True,
    **dataset_kwargs
) -> Tuple[Dict[str, "ProteinSAMDatasetWithESMCache"], "ProteinSAMCollatorWithESMCache"]:
    """
    Create datasets and collator for ProteinSAM training with pre-computed ESM embeddings.
    Embeddings are loaded lazily per-uid from individual .pt files in esm_embeddings_dir.

    Args:
        root_dir: Root directory containing data
        data_name: Dataset name (e.g., "VenusX_Dom")
        esm_model_path: Path to ESM tokenizer
        esm_embeddings_dir: Directory containing per-uid .pt embedding files
        llama_model_path: Path to Llama tokenizer (optional if using cache)
        max_sequence_length: Maximum protein sequence length
        max_text_length: Maximum text length
        use_category_cache: Whether to use category embeddings cache
        **dataset_kwargs: Additional arguments for dataset

    Returns:
        Dictionary of datasets and data collator
    """
    print(f"Using ESM embeddings directory: {esm_embeddings_dir}")
    if not os.path.isdir(esm_embeddings_dir):
        raise ValueError(f"ESM embeddings directory not found: {esm_embeddings_dir}")

    # Initialize tokenizers
    esm_tokenizer = EsmTokenizer.from_pretrained(esm_model_path)

    llama_tokenizer = None
    if not use_category_cache and llama_model_path is not None:
        llama_tokenizer = LlamaTokenizer.from_pretrained(llama_model_path)
        if llama_tokenizer.pad_token is None:
            llama_tokenizer.pad_token = '<|reserved_special_token_0|>'

    # Create datasets
    datasets = {}
    # for split in ["train", "valid", "test"]:
    for split in ["train", "test"]:
        datasets[split] = ProteinSAMDatasetWithESMCache(
            root_dir=root_dir,
            data_name=data_name,
            split=split,
            esm_embeddings_dir=esm_embeddings_dir,
            max_sequence_length=max_sequence_length,
            **dataset_kwargs
        )

    # Create collator (hidden size inferred dynamically from embeddings)
    collator = ProteinSAMCollatorWithESMCache(
        esm_tokenizer=esm_tokenizer,
        llama_tokenizer=llama_tokenizer,
        max_protein_length=max_sequence_length,
        max_text_length=max_text_length,
        use_category_cache=use_category_cache,
        esm_hidden_size=0  # 0 = infer from batch
    )

    return datasets, collator