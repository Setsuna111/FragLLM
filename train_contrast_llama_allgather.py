"""
Stage 1 - contrastive learning training script for ESM-LLAMA modality alignment 
on Esm2LlamaInstructForCausalLM model with cross-GPU negative sampling.

Enhanced version with torch.distributed.all_gather for global negative sampling.

Without LoRA. 

DistributedDataParallel training script implemented from scratch. 

The script currently supports gradient accumulation, AutoMixedPrecision, 
inter-epoch evaluation, and cross-GPU negative sampling for improved contrastive learning.

The script currently does not support save/load pretrained, gradient checkpointing 
or generation under FSDP. 

reference for AMP: https://pytorch.org/tutorials/recipes/recipes/amp_recipe.html 

* The script is designed for multi-GPU parallelism on single node.
* On the cluster, print(...) will go to stdout and tqdm(...) will go to stderr.
* Enhanced with all_gather for global negative sampling across GPUs.
"""

# tensorboard --logdir ./tensorboard_logs --port=6301

import argparse
from datetime import datetime
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'  # restrict GPU visibility
from typing import Any, Dict, Literal, Union
import pdb
import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel
from torch.nn.parallel import DistributedDataParallel
from torch.optim import Adam, AdamW, Optimizer
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
from torch.utils.data.distributed import DistributedSampler
from torch.cuda.amp import autocast
from tqdm import tqdm
import transformers
from transformers import AutoTokenizer, PreTrainedModel
from transformers import EsmModel, LlamaModel, LlamaForCausalLM
from dataset.dataloader_function import FunctionDataset
from dataset.dataloader_frag import FragDataCollator
from dataset.templates import *
from models.protein_llama import *
import scripts.utils_argparse as utils_argparse
from dataclasses import dataclass, field
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

@dataclass
class FragModelArguments:

    """Model arguments for fragment training."""
    # esm_path: Optional[str] = field(default="/home/djy/projects/Data/HF_models/esm2_t36_3B_UR50D", metadata={"help": "Path to ESM model"})
    # llama_path: Optional[str] = field(default="/home/djy/projects/Data/HF_models/RedHatAI-Llama-3.1-8B-Instruct", metadata={"help": "Path to LLaMA model"})
    esm_path: Optional[str] = field(default="/home/lfj/projects_dir/pretrained_model/esm2_t36_3B_UR50D", metadata={"help": "Path to ESM model"})
    llama_path: Optional[str] = field(default="/home/lfj/projects_dir/pretrained_model/Llama-3.1-8B-Instruct", metadata={"help": "Path to LLaMA model"})
    
    load_adapter_checkpoint_dir: Optional[str] = field(default=None, metadata={"help": "Path to load adapter checkpoint"})
    load_fragment_checkpoint_dir: Optional[str] = field(default=None, metadata={"help": "Path to load fragment checkpoint"})
    
    # Model architecture arguments
    fix_modality_adapter: Optional[bool] = field(default=False, metadata={"help": "Whether to fix modality adapter"})
    
    # Fragment adapter arguments
    perceiver_latent_size: Optional[int] = field(default=1, metadata={"help": "Perceiver latent size"})
    num_perceiver_heads: Optional[int] = field(default=8, metadata={"help": "Number of perceiver heads"})
    num_perceiver_layers: Optional[int] = field(default=2, metadata={"help": "Number of perceiver layers"})
    intermediate_dim: Optional[int] = field(default=2048, metadata={"help": "Intermediate dimension"})
    dropout_rate: Optional[float] = field(default=0.3, metadata={"help": "Dropout rate"})
    freeze_backbone: Optional[bool] = field(default=False, metadata={"help": "Whether to freeze backbone"})
    # placeholder ids
    sequence_placeholder_id: int = 128003
    fragment_placeholder_id: int = 128005
    pos_start_placeholder_id: int = 128011
    pos_end_placeholder_id: int = 128012
    
argParser = argparse.ArgumentParser()

# argParser.add_argument("--esm_path", type=str, default="/home/djy/projects/Data/HF_models/esm2_t36_3B_UR50D")
# argParser.add_argument("--llama_path", type=str, default="/home/djy/projects/Data/HF_models/RedHatAI-Llama-3.1-8B-Instruct")
argParser.add_argument("--esm_path", type=str, default="/home/lfj/projects_dir/pretrained_model/esm2_t36_3B_UR50D")
argParser.add_argument("--llama_path", type=str, default="/home/lfj/projects_dir/pretrained_model/Llama-3.1-8B-Instruct")
    
argParser.add_argument("--root_dataset_dir", type=str, default="./data")
argParser.add_argument("--load_model_checkpoint_path", type=str, default="")
argParser.add_argument("--load_optimizer_scheduler_checkpoint_path", type=str, default="")

argParser.add_argument("--torch_dtype", type=utils_argparse.str2dtype, default="bfloat16")
argParser.add_argument("--batch_size_per_device", type=int, default=32)
argParser.add_argument("--num_epochs", type=int, default=36)
argParser.add_argument("--save_every_epochs", type=int, default=1)
argParser.add_argument("--gradient_accumulation_steps", type=int, default=8)
argParser.add_argument("--learning_rate", type=float, default=2e-4)
argParser.add_argument("--gradient_clipping", type=float, default=1.0)
argParser.add_argument("--lr_scheduler_type", type=str, default="cosine", choices=["step", "cosine"], help="Type of learning rate scheduler")
argParser.add_argument("--scheduler_gamma", type=float, default=0.95, help="Gamma for StepLR scheduler")
argParser.add_argument("--cosine_eta_min", type=float, default=1e-6, help="Minimum learning rate for CosineAnnealingLR")
argParser.add_argument("--random_seed", type=int, default=42)
argParser.add_argument("--contrastive_num_segments", type=int, default=4)
argParser.add_argument("--minibatch_size", type=int, default=4, help="Size of mini-batch for memory-efficient encoding")
argParser.add_argument("--save_checkpoint_dir", type=str, default="./checkpoints_contrast")
argParser.add_argument("--tensorboard_log_dir", type=str, default="./tensorboard_logs", help="Directory for TensorBoard logs")
argParser.add_argument("--logging_steps", type=int, default=10, help="Log every N training steps to TensorBoard")


class GlobalBatchInfoNCELoss(torch.nn.Module):
    r"""
    Global batch version of InfoNCE loss for contrastive learning with cross-GPU negative sampling.
    Positive and negative pairs are picked across all GPUs using all_gather.

    $L_{\text{InfoNCE}} = -\log \frac{\exp(s(x, x^+)/\tau)}{\sum_{x'} \exp(s(x, x')/\tau)}$

    temperature: float, temperature parameter for the softmax function. The smaller
    the temperature, the sharper the softmax distribution.
    """
    def __init__(self, temperature: float = 0.05):
        super().__init__()
        self.temperature = temperature

    def forward(
        self, 
        local_protein_embeddings: torch.Tensor,  # (local_bsz, hidden_dim)
        local_description_embeddings: torch.Tensor,  # (local_bsz, hidden_dim) 
        rank: int,
        world_size: int
    ) -> torch.Tensor:
        """
        Forward pass with cross-GPU negative sampling.
        
        Args:
            local_protein_embeddings: Local protein embeddings on current GPU
            local_description_embeddings: Local description embeddings on current GPU
            rank: Current GPU rank
            world_size: Total number of GPUs
            
        Returns:
            InfoNCE loss computed with global negative samples
        """
        local_bsz = local_protein_embeddings.size(0)
        
        # Gather embeddings from all GPUs
        # all_protein_embeddings will be (world_size * local_bsz, hidden_dim)
        all_protein_embeddings = [torch.zeros_like(local_protein_embeddings) for _ in range(world_size)]
        all_description_embeddings = [torch.zeros_like(local_description_embeddings) for _ in range(world_size)]
        
        dist.all_gather(all_protein_embeddings, local_protein_embeddings)
        dist.all_gather(all_description_embeddings, local_description_embeddings)
        
        # Concatenate all embeddings
        global_protein_embeddings = torch.cat(all_protein_embeddings, dim=0)  # (global_bsz, hidden_dim)
        global_description_embeddings = torch.cat(all_description_embeddings, dim=0)  # (global_bsz, hidden_dim)
        
        # Compute similarity matrix between local proteins and global descriptions
        similarity_matrix = torch.mm(local_protein_embeddings, global_description_embeddings.t())  # (local_bsz, global_bsz)
        logits = similarity_matrix / self.temperature
        
        # Compute positive pair indices (accounting for rank offset)
        positive_indices = torch.arange(
            rank * local_bsz, 
            (rank + 1) * local_bsz, 
            device=local_protein_embeddings.device
        )
        
        # Extract positive logits
        numerator = torch.exp(logits[torch.arange(local_bsz), positive_indices])  # (local_bsz,)
        
        # Compute denominator (sum over all global negatives + positive)
        denominator = torch.sum(torch.exp(logits), dim=1)  # (local_bsz,)
        
        # InfoNCE loss
        loss = -torch.log(numerator / denominator).mean()
        
        return loss


class SegmentedGlobalBatchInfoNCELoss(torch.nn.Module):
    """
    Segmented version of GlobalBatchInfoNCELoss for memory-efficient contrastive learning
    with cross-GPU negative sampling.
    """
    def __init__(self, temperature: float = 0.05):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        segment_protein_embeddings: torch.Tensor,  # (segment_size, hidden_dim)
        local_description_embeddings: torch.Tensor,  # (local_bsz, hidden_dim)
        segment_labels: torch.Tensor,  # (segment_size,) - global indices
        rank: int,
        world_size: int
    ) -> torch.Tensor:
        """
        Forward pass with segmented processing and cross-GPU negative sampling.
        
        Args:
            segment_protein_embeddings: Protein embeddings for current segment
            local_description_embeddings: All local description embeddings on current GPU
            segment_labels: Global indices for positive pairs in this segment
            rank: Current GPU rank, only used for label calculation
            world_size: Total number of GPUs
            
        Returns:
            InfoNCE loss for the current segment with global negative samples
        """
        # Gather description embeddings from all GPUs
        all_description_embeddings = [torch.zeros_like(local_description_embeddings) for _ in range(world_size)]
        dist.all_gather(all_description_embeddings, local_description_embeddings)
        
        # Concatenate to get global description embeddings
        global_description_embeddings = torch.cat(all_description_embeddings, dim=0)  # (global_bsz, hidden_dim)

        # print(f"rank:{rank} - global_description_embeddings: {global_description_embeddings.shape}")  # debug

        # print(f"rank:{rank} - global_description_embeddings 0: {global_description_embeddings[0, 0]}")  # debug
        
        # Compute similarity matrix
        similarity_matrix = torch.mm(segment_protein_embeddings, global_description_embeddings.t())  # (segment_size, global_bsz)
        logits = similarity_matrix / self.temperature

        # print(f"rank:{rank} - similarity_matrix: {similarity_matrix.shape}")  # debug
        
        # Extract positive logits using global labels
        segment_size = segment_protein_embeddings.size(0)
        numerator = torch.exp(logits[torch.arange(segment_size), segment_labels])  # (segment_size,)
        
        # Compute denominator (sum over all global negatives + positive)
        denominator = torch.sum(torch.exp(logits), dim=1)  # (segment_size,)
        
        # InfoNCE loss
        loss = -torch.log(numerator / denominator).mean()
        
        return loss


def load_model(args: Dict[str, Any], model_args: FragModelArguments) -> PreTrainedModel:
    """
    Standard API for different models. Used in both `train` and `generate`.
    Load base model of the given name, and load weights from the checkpoint path 
    if provided.

    Returned model should be on CPU and under default data type.
    A general checkpoint shall contain the model state dict, optimizer state dict,
    and scheduler state dict.
    """
    torch_dtype = args["torch_dtype"]
    model = ProteinLlamaForCausalLM.from_pretrained(
        args["llama_path"],
        torch_dtype=torch_dtype,
        device_map="cpu"
    )
    if args["load_model_checkpoint_path"]:
        print(f"Loading {args['load_model_checkpoint_path']}")
        model_state_dict = torch.load(
            args["load_model_checkpoint_path"], 
            weights_only=True, 
            map_location="cpu"  # load to CPU first
            # will be loaded to where the weights were saved from if not specified
        )
        model.load_state_dict(model_state_dict)


    if model_args.esm_path is not None:
        model.get_model().initialize_modules(model_args=model_args, fsdp=None)
        model.config.sequence_placeholder_id = model_args.sequence_placeholder_id
        model.config.fragment_placeholder_id = model_args.fragment_placeholder_id
        model.config.pos_start_placeholder_id = model_args.pos_start_placeholder_id
        model.config.pos_end_placeholder_id = model_args.pos_end_placeholder_id

        esm_encoder = model.get_model().get_esm_encoder()
        esm_encoder.to(dtype=torch_dtype, device="cpu")
    # WARNING: esm and llama weights are fixed
    # import pdb; pdb.set_trace()
    print(model)
    model.get_model().esm_encoder.requires_grad_(False)
    model.model.requires_grad_(False)
    for p in model.get_model().adapter.parameters():
        p.requires_grad_(True)
    model.to(torch_dtype)
    return model


def readout_embeddings(
        embeddings: torch.Tensor,  # (bsz, seq_len, hidden_dim)
        attention_mask: torch.Tensor,  # (bsz, text_len)
        readout_fn: Literal["last", "mean", "std", "mix"], 
) -> torch.Tensor:
    """
    Perform a readout operation on the output sequence embeddings of the forward 
    pass, given the attention mask. 
    """
    # embeddings = embeddings.to(torch.float32)
    if readout_fn == "last":
        # inputs must be right padded
        # for left padding simply take the last token and do not use this function
        last_token_indices = attention_mask.sum(dim=1) - 1  # (bsz,)
        batch_indices = torch.arange(
            attention_mask.size(0), 
            device=attention_mask.device
        )  # (bsz,)
        return embeddings[batch_indices, last_token_indices, :]  # (bsz, hidden_dim)

    elif readout_fn == "mean":
        masked_embeddings = embeddings * attention_mask.unsqueeze(-1)
        sum_embeddings = masked_embeddings.sum(dim=1)  # (bsz, hidden_dim)
        count_attn_mask = attention_mask.sum(dim=1, keepdim=True)  # (bsz, 1)
        return sum_embeddings / count_attn_mask  # (bsz, hidden_dim)
    
    elif readout_fn == "std":
        mean_embeddings = readout_embeddings(
            embeddings=embeddings, 
            attention_mask=attention_mask, 
            readout_fn="mean"
        )  # (bsz, hidden_dim)
        diff_embeddings = embeddings - mean_embeddings.unsqueeze(1)
            # (bsz, text_len, hidden_dim)
        diff_embeddings_2 = diff_embeddings.pow(2) 
        masked_diff_embeddings_2 = diff_embeddings_2 * attention_mask.unsqueeze(-1)
        sum_diff_embeddings_2 = masked_diff_embeddings_2.sum(dim=1)  # (bsz, hidden_dim)
        count_attn_mask = attention_mask.sum(dim=1, keepdim=True)  # (bsz, 1)
        return (sum_diff_embeddings_2 / count_attn_mask).sqrt()  # (bsz, hidden_dim)

    elif readout_fn == "mix": 
        mean_embeddings = readout_embeddings(
            embeddings=embeddings, 
            attention_mask=attention_mask, 
            readout_fn="mean"
        )
        std_embeddings = readout_embeddings(
            embeddings=embeddings, 
            attention_mask=attention_mask, 
            readout_fn="std"
        )
        return torch.cat([mean_embeddings, std_embeddings], dim=1)  # (bsz, 2 * hidden_dim)


def get_sequence_embeddings(
        model: ProteinLlamaForCausalLM, 
        sequence_input_ids: torch.Tensor, 
        sequence_attention_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Take mean pooling and the std pooling of the adapter outputs for each valid 
    token in the sequence as the sequence embedding for contrastive learning. 

    sequence_input_ids: (bsz, max_seq_len)
    sequence_attention_mask: (bsz, max_seq_len)
    return: (bsz, decoder_hidden_size)
    """
    with torch.no_grad():  # WARNING: esm encoder fixed
        encoder_output = model.forward(
            protein_input_ids=sequence_input_ids,
            protein_attention_mask=sequence_attention_mask,
            return_encoder_outputs=True,
        )
    # print("esm_encoder_output", encoder_output[0])
    adapter_output = model.get_model().adapter(encoder_output[0])
    # print("adapter_output", adapter_output)
    protein_attention_mask = sequence_attention_mask
    # adapter_output: (bsz, max_seq_len, decoder_hidden_size)

    return readout_embeddings(
        embeddings=adapter_output, 
        attention_mask=protein_attention_mask, 
        readout_fn="mean"
    )  # (bsz, decoder_hidden_size)


def get_sequence_embeddings_minibatch(
        model: ProteinLlamaForCausalLM, 
        sequence_input_ids: torch.Tensor, 
        sequence_attention_mask: torch.Tensor,
        minibatch_size: int = 4
) -> torch.Tensor:
    """
    Mini-batch version of get_sequence_embeddings to reduce memory usage.
    Processes the input in smaller mini-batches and concatenates results.
    
    Args:
        model: The protein language model
        sequence_input_ids: (bsz, max_seq_len)
        sequence_attention_mask: (bsz, max_seq_len)
        minibatch_size: Size of each mini-batch
    
    Returns:
        (bsz, decoder_hidden_size) - concatenated embeddings
    """
    batch_size = sequence_input_ids.size(0)
    device = sequence_input_ids.device
    
    # Get the hidden dimension from the model config
    hidden_dim = model.get_model().adapter.text_emb_dim
    
    # Initialize output tensor
    all_embeddings = torch.zeros(batch_size, hidden_dim, device=device, dtype=model.dtype)
    
    # Process in mini-batches
    for i in range(0, batch_size, minibatch_size):
        end_idx = min(i + minibatch_size, batch_size)
        
        minibatch_input_ids = sequence_input_ids[i:end_idx]
        minibatch_attention_mask = sequence_attention_mask[i:end_idx]
        
        # Get embeddings for this mini-batch
        minibatch_embeddings = get_sequence_embeddings(
            model, 
            minibatch_input_ids, 
            minibatch_attention_mask
        )
        
        # Store in output tensor
        all_embeddings[i:end_idx] = minibatch_embeddings
        
        # Clear cache to free memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    return all_embeddings


def get_description_embeddings(
        model: ProteinLlamaForCausalLM,
        description_input_ids: torch.Tensor,
        description_attention_mask: torch.Tensor,
        output_llama_layer: int = 16,
) -> torch.Tensor:
    """Take output corresponding to eot_token in the description. """
    hidden_states = model(
        input_ids=description_input_ids,
        attention_mask=description_attention_mask,
        use_cache=False, 
        output_attentions=False, 
        output_hidden_states=True, 
        return_dict=False,
    )[1]  # (bsz, max_desc_len, hidden_dim)

    return readout_embeddings(
        embeddings=hidden_states[output_llama_layer],
        attention_mask=description_attention_mask,
        readout_fn="mean"
    )  # (bsz, decoder_hidden_size)


def get_description_embeddings_minibatch(
        model: ProteinLlamaForCausalLM,
        description_input_ids: torch.Tensor,
        description_attention_mask: torch.Tensor,
        minibatch_size: int = 4,
        output_llama_layer: int = 16,
) -> torch.Tensor:
    """
    Mini-batch version of get_description_embeddings to reduce memory usage.
    Processes the input in smaller mini-batches and concatenates results.
    
    Args:
        model: The protein language model
        description_input_ids: (bsz, max_desc_len)
        description_attention_mask: (bsz, max_desc_len)
        minibatch_size: Size of each mini-batch
        output_llama_layer: Which LLAMA layer to use for embeddings
    
    Returns:
        (bsz, decoder_hidden_size) - concatenated embeddings
    """
    batch_size = description_input_ids.size(0)
    device = description_input_ids.device
    
    # Get the hidden dimension from model config
    hidden_dim = model.config.hidden_size
    
    # Initialize output tensor
    all_embeddings = torch.zeros(batch_size, hidden_dim, device=device, dtype=model.dtype)
    
    # Process in mini-batches
    for i in range(0, batch_size, minibatch_size):
        end_idx = min(i + minibatch_size, batch_size)
        
        minibatch_input_ids = description_input_ids[i:end_idx]
        minibatch_attention_mask = description_attention_mask[i:end_idx]
        
        # Get embeddings for this mini-batch
        minibatch_embeddings = get_description_embeddings(
            model, 
            minibatch_input_ids, 
            minibatch_attention_mask, 
            output_llama_layer
        )
        
        # Store in output tensor
        all_embeddings[i:end_idx] = minibatch_embeddings
        
        # Clear cache to free memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    return all_embeddings


def teacher_forcing_forward_pass_global(
        rank: int,
        world_size: int,
        model: Union[DistributedDataParallel, FullyShardedDataParallel],
        data_batch: Dict[str, Any],
        contrastive_num_segments: int,
        minibatch_size: int = 4, 
) -> torch.Tensor:  # loss
    """
    Enhanced forward pass with cross-GPU negative sampling for contrastive learning.
    Now includes mini-batch processing to reduce memory usage for ESM and LoRA encoding.
    
    Standard API for different models. Used in both `train_epoch` and `eval_epoch`.
    Prepare inputs from dataloader, migrate variable to the same device as the model, 
    and execute the forward pass with teacher forcing.

    Returned loss is not scaled with gradient accumulation steps.

    Due to the memory limit on GPUs, the similarity matrix will be computed in 
    segments, and the loss will be averaged over the segments. Now with global
    negative sampling across all GPUs using all_gather.
    
    Args:
        minibatch_size: Size of mini-batch for memory-efficient encoding
    """
    protein_input_ids = data_batch["protein_input_ids"].to(rank)
    protein_attention_mask = data_batch["protein_attention_mask"].to(rank)
    description_input_ids = data_batch["answer_input_ids"].to(rank)
    description_attention_mask = data_batch["answer_attention_mask"].to(rank)

    base_model = model
    if isinstance(model, DistributedDataParallel):
        base_model = model.module

    local_batch_size = protein_input_ids.size(0)
    segment_size = local_batch_size // contrastive_num_segments
    if segment_size * contrastive_num_segments != local_batch_size:
        print(
            "WARNING: Given batch size is not divisible by the number of segments "
            "for contrastive learning."
        )
    
    acc_loss = torch.zeros([]).to(rank)
    loss_fn = SegmentedGlobalBatchInfoNCELoss()

    # Get local description embeddings using mini-batch processing (fixed weights, no gradients)
    with torch.no_grad():  # WARNING: llama decoder fixed
        local_description_embeddings = get_description_embeddings_minibatch(
            base_model,  
            description_input_ids, 
            description_attention_mask,
            minibatch_size=minibatch_size
        )
        local_description_embeddings = torch.nn.functional.normalize(
            local_description_embeddings, p=2, dim=-1, eps=1e-8
        )
    
    # print(f"rank:{rank} - local_description_embeddings: {local_description_embeddings.shape}")  # debug

    # Process in segments for memory efficiency
    for segment_id in range(contrastive_num_segments):
        segment_protein_input_ids = protein_input_ids[
            segment_id * segment_size:(segment_id + 1) * segment_size
        ]
        segment_protein_attention_mask = protein_attention_mask[
            segment_id * segment_size:(segment_id + 1) * segment_size
        ]

        # Get segment protein embeddings using mini-batch processing (trainable adapter)
        segment_protein_embeddings = get_sequence_embeddings_minibatch(
            base_model, 
            segment_protein_input_ids, 
            segment_protein_attention_mask,
            minibatch_size=minibatch_size
        )

        # print(f"rank:{rank} - segment:{segment_id+1}/{contrastive_num_segments} - segment_protein_embeddings: {segment_protein_embeddings.shape}")  # debug
        
        # # Normalize protein embeddings
        # segment_protein_embeddings = torch.nn.functional.normalize(
        #     segment_protein_embeddings, p=2, dim=-1, eps=1e-8
        # )
        
        # Compute global labels for this segment
        segment_labels = torch.arange(
            rank * local_batch_size + segment_id * segment_size, 
            rank * local_batch_size + (segment_id + 1) * segment_size, 
            device=rank
        )

        # print(f"rank:{rank} - segment:{segment_id+1}/{contrastive_num_segments} - segment_labels: {segment_labels}")  # debug
        
        # Compute loss with global negative sampling
        segment_loss = loss_fn(
            segment_protein_embeddings=segment_protein_embeddings,
            local_description_embeddings=local_description_embeddings,
            segment_labels=segment_labels,
            rank=rank,
            world_size=world_size
        )
        
        acc_loss += segment_loss

    return acc_loss / contrastive_num_segments


def setup(rank: int, world_size: int):
    """
    Initialize processes for distributed training before first epoch. 
    Fetch from job script or launcher to set the IP address and the port of the 
    master node. 
    """
    os.environ['MASTER_ADDR'] = os.getenv('MASTER_ADDR', 'localhost')
    os.environ['MASTER_PORT'] = os.getenv('MASTER_PORT', '9908')
    # initialize the process group
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)


def cleanup():
    """End processes for distributed training after last epoch"""
    dist.destroy_process_group()


def train_epoch(
        rank: int,
        world_size: int,
        current_epoch: int,
        model: Union[DistributedDataParallel, FullyShardedDataParallel],
        dataloader: FragDataCollator,
        optimizer: Optimizer,
        args: Dict[str, Any],
        writer: SummaryWriter = None,
        global_step: int = 0
):
    """Iterate over all batches for one epoch in training with teacher forcing and global negative sampling"""
    model.train()
    ddp_loss = torch.zeros(2).to(rank)  
        # [0] for acc. loss and [1] for num. of seen batches
    ddp_gradnorm = torch.zeros(2).to(rank)  
        # [0] for acc. gradnorm and [1] for num. of passed steps
    optimizer.zero_grad()  # erase accumulated gradients from last epoch

    t = tqdm(iter(dataloader))
    for batch_idx, data_batch in enumerate(t):
        # with autocast, logits will be in AUTOCAST_DTYPE 
        # but loss will be re-casted to torch.float32
        # and model weights will stay in torch.float32
        loss = teacher_forcing_forward_pass_global(
            rank=rank,
            world_size=world_size, 
            model=model, 
            data_batch=data_batch, 
            contrastive_num_segments=args["contrastive_num_segments"],
            minibatch_size=args["minibatch_size"]
        )

        # rescale loss for consistency with different gradient accumulation steps
        loss = loss / args["gradient_accumulation_steps"]

        # summary current batch
        t.set_postfix({
            "mode": "train",
            "epoch": f"{current_epoch}/{args['num_epochs']}",
            "batch_loss": loss.item() * args["gradient_accumulation_steps"],
            "device": f"rank:{rank}"
        })
        ddp_loss[0] += loss.item() * args["gradient_accumulation_steps"]
        ddp_loss[1] += 1  # the loss is the weighted mean of the output of every batch

        # scale the loss up by a large factor to prevent them from becoming too small
        # then accumulate the scaled grads
        loss.backward()
        # print("adapter.fc1.weight", model.module.get_model().adapter.fc1.weight)  
        # backward out of autocast, but still uses same dtype as for forward

        # update weights by loss if accumulation step is reached
        if (batch_idx + 1) % args["gradient_accumulation_steps"] == 0: 
            gradnorm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=(
                    float("inf") 
                    if args["gradient_clipping"] is None 
                    else args["gradient_clipping"]
                )
            )
            ddp_gradnorm[0] += gradnorm
            ddp_gradnorm[1] += 1

            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            
            # Log to TensorBoard (only on rank 0 to avoid duplicate logs)
            if rank == 0 and writer is not None and global_step % args["logging_steps"] == 0:
                writer.add_scalar("train/batch_loss", loss.item() * args["gradient_accumulation_steps"], global_step)
                writer.add_scalar("train/learning_rate", optimizer.param_groups[0]['lr'], global_step)
                writer.add_scalar("train/gradient_norm", gradnorm.item(), global_step)
            
            global_step += 1

    # summary current epoch
    dist.all_reduce(ddp_loss, op=dist.ReduceOp.SUM)
    dist.all_reduce(ddp_gradnorm, op=dist.ReduceOp.SUM)
    if rank == 0:
        epoch_train_loss = ddp_loss[0] / ddp_loss[1]
        epoch_lr = optimizer.param_groups[0]['lr']
        epoch_gradnorm = ddp_gradnorm[0] / ddp_gradnorm[1]
        
        print(
            f"[epoch={current_epoch}/{args['num_epochs']}, "
            f"train_loss={epoch_train_loss}, "
            f"epoch_lr={epoch_lr}, "
            f"epoch_gradnorm={epoch_gradnorm}]"
        )
        
        # Log epoch metrics to TensorBoard
        if writer is not None:
            writer.add_scalar("epoch/train_loss", epoch_train_loss, current_epoch)
            writer.add_scalar("epoch/learning_rate", epoch_lr, current_epoch)
            writer.add_scalar("epoch/gradient_norm", epoch_gradnorm, current_epoch)
        
        # NaN detection
        if ddp_loss[0] != ddp_loss[0]:
            raise ValueError(
                "NaN detected in the training loss of the epoch, training interrupted."
            )
    
    return global_step


def eval_epoch(
        rank: int,
        world_size: int,
        current_epoch: int, 
        model: Union[DistributedDataParallel, FullyShardedDataParallel],
        dataloader: FragDataCollator,
        args: Dict[str, Any],
        writer: SummaryWriter = None
):
    """Iterate over all batches in evaluation with teacher forcing and global negative sampling"""
    model.eval()
    ddp_loss = torch.zeros(2).to(rank)  
        # [0] for acc. loss and [1] for num. of seen batches

    t = tqdm(iter(dataloader))
    for data_batch in t:
        with torch.no_grad():
            loss = teacher_forcing_forward_pass_global(
                rank=rank,
                world_size=world_size,
                model=model,
                data_batch=data_batch,
                contrastive_num_segments=args["contrastive_num_segments"],
                minibatch_size=args["minibatch_size"]
            )

            t.set_postfix({
                "mode": "eval",
                "epoch": f"{current_epoch}/{args['num_epochs']}",
                "batch_loss": loss.item(),
                "device": f"rank:{rank}"
            })
            ddp_loss[0] += loss.item()
            ddp_loss[1] += 1

    dist.all_reduce(ddp_loss, op=dist.ReduceOp.SUM)
    if rank == 0:
        epoch_eval_loss = ddp_loss[0] / ddp_loss[1]
        print(
            f"[epoch={current_epoch}/{args['num_epochs']}, "
            f"eval_loss={epoch_eval_loss}]"
        )
        
        # Log evaluation loss to TensorBoard
        if writer is not None:
            writer.add_scalar("epoch/eval_loss", epoch_eval_loss, current_epoch)


def train_on_device(
        rank: int,
        world_size: int,
        args: Dict[str, Any],
        model_args: FragModelArguments
):
    """
    Training and evaluation process for each device, including epochs of training 
    with teacher forcing and cross-GPU negative sampling. 
    """
    setup(rank, world_size)

    # prepare datasets and dataloaders
    esm_tokenizer = AutoTokenizer.from_pretrained(args["esm_path"])
    llama_tokenizer = AutoTokenizer.from_pretrained(
        args["llama_path"], 
        pad_token='<|reserved_special_token_0|>'
    )

    train_dataset = FunctionDataset(
        root_dir=args["root_dataset_dir"],
        data_name="Pro2Text",
        split="train",
        task_type="function",
        question_template=ProteinFunction,
        answer_template=None,
    )

    train_sampler = DistributedSampler(
        train_dataset, 
        rank=rank, 
        num_replicas=world_size, 
        shuffle=True
        )
    data_collator = FragDataCollator(
        sequence_tokenizer=esm_tokenizer,
        llm_tokenizer=llama_tokenizer,
        mode="train",
        max_sequence_length=1021,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args["batch_size_per_device"],
        sampler=train_sampler,
        num_workers=4,  # parallel CPU cores used for data loading
        pin_memory=True,  # enable page-locked memory allocation for faster data transfer to GPUs
        collate_fn=data_collator,
        shuffle=False,  # avoid shuffling twice with DistributedSampler
        drop_last=True,  # avoid incomplete batch at the end
    )
    print(f"Train dataset loaded on rank:{rank}")

    eval_dataset = FunctionDataset(
        root_dir=args["root_dataset_dir"],
        data_name="Pro2Text",
        split="valid",
        task_type="function",
        question_template=ProteinFunction,
        answer_template=None,
    )
    eval_sampler = DistributedSampler(
        eval_dataset, 
        rank=rank, 
        num_replicas=world_size, 
        shuffle=False
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=args["batch_size_per_device"],
        sampler=eval_sampler,
        collate_fn=data_collator,
        num_workers=4,
        pin_memory=True,
        shuffle=False,
        drop_last=True,
    )
    print(f"Eval dataset loaded on rank:{rank}")

    torch.cuda.set_device(rank)

    model = load_model(args=args, model_args=model_args)
    model = model.to(rank)

    model = DistributedDataParallel(
        model, 
        find_unused_parameters=True  # suppress error for unused parameters in wrapped model
    )
    print(f"DDP model loaded on rank:{rank}")

    # initialization of the optimizer after wrapping the model
    optimizer = Adam(model.parameters(), lr=args["learning_rate"])
    
    # Initialize scheduler based on type
    if args["lr_scheduler_type"] == "cosine":
        scheduler = CosineAnnealingLR(
            optimizer, 
            T_max=args["num_epochs"], 
            eta_min=args["cosine_eta_min"]
        )
        print(f"Using CosineAnnealingLR scheduler with T_max={args['num_epochs']}, eta_min={args['cosine_eta_min']}")
    elif args["lr_scheduler_type"] == "step":
        scheduler = StepLR(optimizer, step_size=1, gamma=args["scheduler_gamma"])
        print(f"Using StepLR scheduler with step_size=1, gamma={args['scheduler_gamma']}")
    else:
        raise ValueError(f"Unsupported scheduler type: {args['lr_scheduler_type']}")
    if args["load_optimizer_scheduler_checkpoint_path"]:
        print(f"Loading {args['load_optimizer_scheduler_checkpoint_path']}")
        checkpoint_state_dicts = torch.load(
            args["load_optimizer_scheduler_checkpoint_path"], 
            weights_only=True
        )
        optimizer_state_dict = checkpoint_state_dicts["optimizer_state_dict"]
        scheduler_state_dict = checkpoint_state_dicts["scheduler_state_dict"]
        optimizer.load_state_dict(optimizer_state_dict)
        scheduler.load_state_dict(scheduler_state_dict)

    # Initialize TensorBoard writer (only on rank 0)
    writer = None
    if rank == 0:
        timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
        log_dir = os.path.join(args["tensorboard_log_dir"], f"contrast_training_{timestamp}")
        os.makedirs(log_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=log_dir)
        print(f"TensorBoard logging to: {log_dir}")

    global_step = 0
    # core loop of epochs
    for epoch_idx in range(1, args["num_epochs"] + 1):
        # shuffle data differently at each epoch across all processes
        train_sampler.set_epoch(epoch=epoch_idx)

        global_step = train_epoch(
            rank=rank,
            world_size=world_size,
            current_epoch=epoch_idx,
            model=model,    
            dataloader=train_loader,
            optimizer=optimizer,
            args=args,
            writer=writer,
            global_step=global_step
        )
        scheduler.step()
        dist.barrier()  # use a barrier to make sure training is done on all ranks
        
        eval_epoch(
            rank=rank,
            world_size=world_size,
            model=model,
            current_epoch=epoch_idx,
            dataloader=eval_loader,
            args=args,
            writer=writer
        )
        dist.barrier()

        if (
            epoch_idx == 1 
            or epoch_idx == args["num_epochs"] 
            or epoch_idx % args["save_every_epochs"] == 0
        ):
            model_state_dict = model.module.state_dict()
            if rank == 0:
                model_checkpoint_path = os.path.join(
                    args["save_checkpoint_dir"], 
                    f"model_checkpoint_{epoch_idx}.pt"
                )
                torch.save(model_state_dict, model_checkpoint_path)
                print(f"Saving {model_checkpoint_path}")

                optimizer_scheduler_checkpoint_path = os.path.join(
                    args["save_checkpoint_dir"], 
                    f"optimizer_scheduler_checkpoint_{epoch_idx}.pt"
                )
                torch.save(
                    {
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                    }, 
                    optimizer_scheduler_checkpoint_path
                )
                print(f"Saving {optimizer_scheduler_checkpoint_path}")

            dist.barrier()

    # Close TensorBoard writer
    if writer is not None:
        writer.close()
        print("TensorBoard writer closed.")

    cleanup()


def train_distributed(
        args: Dict[str, Any],  # replace **kwargs for compatibility with spawn
        model_args: FragModelArguments
):
    """
    Core training process across multiple devices with epochs of training and 
    inter-epoch evaluation with cross-GPU negative sampling.
    """
    torch.multiprocessing.spawn(
        train_on_device, 
        args=(args["world_size"], args, model_args),
        nprocs=args["world_size"],
        join=True
    )


if __name__ == '__main__':
    # suppress messages from AutoTokenizer parallelism and Graphein respectively
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    os.environ["LOGURU_LEVEL"] = "INFO"

    parsed_args = argParser.parse_args()
    model_args = transformers.HfArgumentParser(FragModelArguments).parse_args_into_dataclasses()[0]
    # os.environ['CUDA_VISIBLE_DEVICES'] = '1'  # restrict GPU visibility
    parsed_args.world_size = torch.cuda.device_count()  # use up all visible GPUs

    torch.manual_seed(parsed_args.random_seed)
    torch.cuda.manual_seed(parsed_args.random_seed)
    
    # initialize checkpoint directory
    timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
    parsed_args.save_checkpoint_dir = os.path.join(
        parsed_args.save_checkpoint_dir, 
        f"checkpoints_allgather_{timestamp}"
    )
    if not os.path.exists(parsed_args.save_checkpoint_dir):
        os.mkdir(parsed_args.save_checkpoint_dir)
    
    # initialize tensorboard log directory
    if not os.path.exists(parsed_args.tensorboard_log_dir):
        os.makedirs(parsed_args.tensorboard_log_dir, exist_ok=True)
    
    print("####################")
    print("Enhanced Training Script with Cross-GPU Negative Sampling and TensorBoard Support")
    for key, value in parsed_args.__dict__.items(): 
        print(f"{key}: {value}")
    print("####################")

    train_distributed(parsed_args.__dict__, model_args)