"""
FullyShardedDataParallel / DistributedDataParallel training script implemented 
from scratch. 

The script currently supports gradient accumulation, AutoMixedPrecision, 
and inter-epoch evaluation. 

The script currently does not support save/load pretrained or gradient checkpointing.

reference for FSDP: https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html
reference for AMP: https://pytorch.org/tutorials/recipes/recipes/amp_recipe.html 

* The script is designed for multi-GPU parallelism on single node.
* On the cluster, print(...) will go to stdout and tqdm(...) will go to stderr.
"""

import argparse
from datetime import datetime
import functools
import os
from typing import Any, Dict, Optional, Tuple, Union

import torch
from torch.amp import GradScaler
from torch.cuda.amp import autocast
import torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel, 
    FullStateDictConfig, 
    StateDictType
)
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
from torch.nn.parallel import DistributedDataParallel
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import StepLR
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from transformers import AutoTokenizer, PreTrainedModel

from dataset import Prot2TextInstructDataset, Prot2TextDerivedDataLoader
from models import Esm2LlamaForCausalLM
import scripts.utils_argparse as utils_argparse


argParser = argparse.ArgumentParser()

argParser.add_argument("--esm_path", type=str)
argParser.add_argument("--llama_path", type=str)
argParser.add_argument("--root_dataset_dir", type=str)
argParser.add_argument("--root_csv_dir", type=str)
argParser.add_argument("--save_checkpoint_dir", type=str)
argParser.add_argument("--load_general_checkpoint_path", type=str, default="")

argParser.add_argument("--wrap_model", type=utils_argparse.str2bool)
argParser.add_argument("--autocast_dtype", type=utils_argparse.str2dtype)
argParser.add_argument("--batch_size_per_device", type=int)
argParser.add_argument("--num_epochs", type=int)
argParser.add_argument("--save_every_epochs", type=int)
argParser.add_argument("--gradient_accumulation_steps", type=int)
argParser.add_argument("--learning_rate", type=float)
argParser.add_argument("--gradient_clipping", type=float, default=None)
argParser.add_argument("--scheduler_gamma", type=float)
argParser.add_argument("--random_seed", type=int)
argParser.add_argument("--train_split", type=str)
argParser.add_argument("--eval_split", type=str)
argParser.add_argument("--debug_trim_train_split", type=int, default=None)
argParser.add_argument("--debug_trim_eval_split", type=int, default=None)
argParser.add_argument("--max_sequence_length", type=int, default=None)
argParser.add_argument("--max_description_length", type=int, default=None)


def load_model(
        esm_path: Union[str, os.PathLike],
        llama_path: Union[str, os.PathLike],
        load_general_checkpoint_path: Optional[Union[str, os.PathLike]] = None
) -> PreTrainedModel:
    """
    Standard API for different models. Used in both `train` and `generate`.
    Load base model of the given name, and load weights from the checkpoint path if provided.
    Returned model is on CPU by default.
    `load_general_checkpoint_path` will be ignored if `load_checkpoint_path` is provided.
    """
    model = Esm2LlamaForCausalLM.from_pretrained(
        pretrained_esm_model_name_or_path=esm_path,
        pretrained_llama_model_name_or_path=llama_path,
        esm_kwargs={"decoder_hidden_size": 2048}
    )

    # load checkpoint if any
    if load_general_checkpoint_path:
        print(f"Loading {load_general_checkpoint_path}")
        checkpoint_state_dicts = torch.load(load_general_checkpoint_path, weights_only=True)
        model_state_dict = checkpoint_state_dicts["model_state_dict"]
        model.load_state_dict(model_state_dict)

    return model


def teacher_forcing_forward_pass(
        rank: int,
        model: torch.nn.Module,
        data_batch: Dict[str, Any],
) -> Tuple[torch.Tensor]:
    """
    Standard API for different models. Used in both `train_epoch` and `eval_epoch`.
    1) Prepare inputs for the forward pass with teacher forcing using data_batch from dataloader.
    2) Execute the forward pass and return the direct output.
    Returned loss is not scaled with gradient accumulation steps.
    Returned logits are un-normalized predictions representing the scores for each token in the vocabulary.
    """
    return model(
        input_ids=data_batch["input_ids"].to(rank),
        attention_mask=data_batch["attention_mask"].to(rank),
        labels=data_batch["labels"].to(rank),
        protein_input_ids=data_batch["protein_input_ids"].to(rank),
        protein_attention_mask=data_batch["protein_attention_mask"].to(rank),
        use_cache=False,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=False,  # force return tuple (loss, logits)
        return_encoder_output=False,
    )
        

def setup(rank: int, world_size: int):
    """
    Initialize processes for distributed training before first epoch. 
    Fetch from job script or launcher to set the IP address and the port of the master node. 
    """
    os.environ['MASTER_ADDR'] = os.getenv('MASTER_ADDR', 'localhost')
    os.environ['MASTER_PORT'] = os.getenv('MASTER_PORT', '9901')
    # initialize the process group
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)


def cleanup():
    """End processes for distributed training after last epoch"""
    dist.destroy_process_group()


def train_epoch(
        rank: int,
        current_epoch: int,
        model: Union[DistributedDataParallel, FullyShardedDataParallel],
        dataloader: Prot2TextDerivedDataLoader,
        optimizer: Optimizer,
        scaler: GradScaler, 
        args: Dict[str, Any]
):
    """Iterate over all batches for one epoch in training with teacher forcing"""
    model.train()
    ddp_loss = torch.zeros(2).to(rank)  # [0] for acc. loss and [1] for num. of seen batches
    ddp_gradnorm = torch.zeros(2).to(rank)  # [0] for acc. gradnorm and [1] for num. of passed steps
    optimizer.zero_grad()  # erase accumulated gradients from last epoch

    t = tqdm(iter(dataloader))
    for batch_idx, data_batch in enumerate(t):
        # with autocast, logits will be in AUTOCAST_DTYPE and loss will be re-casted to torch.float32
        with autocast(dtype=args["autocast_dtype"]):
            output = teacher_forcing_forward_pass(
                rank=rank, 
                model=model, 
                data_batch=data_batch, 
            )

        # rescale loss for consistency with different gradient accumulation steps
        loss = output[0] / args["gradient_accumulation_steps"]

        # summary current batch
        t.set_postfix({
            "mode": "train",
            "epoch": f"{current_epoch}/{args['num_epochs']}",
            "batch_loss": loss.item() * args["gradient_accumulation_steps"],
            "device": f"rank:{rank}"
        })
        ddp_loss[0] += loss.item() * args["gradient_accumulation_steps"]
        ddp_loss[1] += 1  # the loss is the weighted mean of the output of every batch

        # scale the loss up by a large factor to prevent them from becoming too small, then accumulate the scaled grads
        scaler.scale(loss).backward()  # backward out of autocast, but still uses same dtype as for forward

        # update weights by loss if accumulation step is reached
        if (batch_idx + 1) % args["gradient_accumulation_steps"] == 0:  # Perform optimizer step after accumulation
            scaler.unscale_(optimizer)  # unscale gradients for gradient examination and clipping
            gradnorm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=float("inf") if args["gradient_clipping"] is None else args["gradient_clipping"]
            )
            ddp_gradnorm[0] += gradnorm
            ddp_gradnorm[1] += 1

            scaler.step(optimizer)  # first unscale the gradients, then do step only if no INF or NaN is in grad
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

    # summary current epoch
    dist.all_reduce(ddp_loss, op=dist.ReduceOp.SUM)
    if rank == 0:
        print(
            f"[epoch={current_epoch}/{args['num_epochs']}, "
            f"train_loss={ddp_loss[0] / ddp_loss[1]}, "
            f"epoch_lr={optimizer.param_groups[0]['lr']}, "
            f"epoch_gradnorm={ddp_gradnorm[0] / ddp_gradnorm[1]}]"
        )
        # NaN detection
        if ddp_loss[0] != ddp_loss[0]:
            raise ValueError("NaN detected in the training loss of the epoch, training interrupted.")


def eval_epoch(
        rank: int,
        current_epoch: int, 
        model: Union[DistributedDataParallel, FullyShardedDataParallel],
        dataloader: Prot2TextDerivedDataLoader,
        args: Dict[str, Any]
):
    """Iterate over all batches in evaluation with teacher forcing"""
    model.eval()
    ddp_loss = torch.zeros(2).to(rank)  # [0] for acc. loss and [1] for num. of seen batches

    t = tqdm(iter(dataloader))
    for data_batch in t:
        with torch.no_grad():
            output = teacher_forcing_forward_pass(
                rank=rank,
                model=model,
                data_batch=data_batch,
            )

            loss = output[0]
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
        print(f"[epoch={current_epoch}/{args['num_epochs']}, eval_loss={ddp_loss[0] / ddp_loss[1]}]")


def train_on_device(
        rank: int,
        world_size: int,
        args: Dict[str, Any]
):
    """Training and evaluation process for each device, including epochs of training with teacher forcing"""
    setup(rank, world_size)

    # prepare datasets and dataloaders
    esm_tokenizer = AutoTokenizer.from_pretrained(args["esm_path"])
    llama_tokenizer = AutoTokenizer.from_pretrained(args["llama_path"], pad_token='<|reserved_special_token_0|>')

    train_dataset = Prot2TextInstructDataset(
        root_dir=os.path.join(args["root_dataset_dir"], f"{args['train_split']}"),
        csv_path=os.path.join(args["root_csv_dir"], f"{args['train_split']}.csv"),
        sequence_tokenizer=esm_tokenizer,
        description_tokenizer=llama_tokenizer,
        skip_reload=True,
        skip_download=True,
        ignore_graph_features=False,
        max_sequence_length=args["max_sequence_length"],
        max_description_length=args["max_description_length"],
    )
    if args["debug_trim_train_split"]:
        train_dataset.usable_file_names = train_dataset.usable_file_names[:args["debug_trim_train_split"]]
    train_sampler = DistributedSampler(train_dataset, rank=rank, num_replicas=world_size, shuffle=True)
    train_loader = Prot2TextDerivedDataLoader(
        train_dataset,
        batch_size=args["batch_size_per_device"],
        sampler=train_sampler,
        num_workers=4,  # parallel CPU cores used for data loading
        pin_memory=True,  # enable page-locked memory allocation for faster data transfer to GPUs
        shuffle=False,
        drop_last=True,  # avoid incomplete batch at the end
    )

    eval_dataset = Prot2TextInstructDataset(
        root_dir=os.path.join(args["root_dataset_dir"], f"{args['eval_split']}"),
        csv_path=os.path.join(args["root_csv_dir"], f"{args['eval_split']}.csv"),
        sequence_tokenizer=esm_tokenizer,
        description_tokenizer=llama_tokenizer,
        skip_reload=True,
        skip_download=True,
        ignore_graph_features=False,
        max_sequence_length=args["max_sequence_length"],
        max_description_length=args["max_description_length"],
    )
    if args["debug_trim_eval_split"]:
        eval_dataset.usable_file_names = eval_dataset.usable_file_names[:args["debug_trim_eval_split"]]
    eval_sampler = DistributedSampler(eval_dataset, rank=rank, num_replicas=world_size, shuffle=False)
    eval_loader = Prot2TextDerivedDataLoader(
        eval_dataset,
        batch_size=args["batch_size_per_device"],
        sampler=eval_sampler,
        num_workers=4,
        pin_memory=True,
        shuffle=False,
        drop_last=True,
    )
    
    torch.cuda.set_device(rank)

    model = load_model(
        esm_path=args["esm_path"],
        llama_path=args["llama_path"],
        load_general_checkpoint_path=args["load_general_checkpoint_path"], 
    )
    model = model.to(rank)

    if args["wrap_model"]: 
        # shard all layers with size of parameters greater than min_num_params
        my_auto_wrap_policy = functools.partial(size_based_auto_wrap_policy, min_num_params=10000)
        model = FullyShardedDataParallel(model, auto_wrap_policy=my_auto_wrap_policy)
        print(f"FSDP model loaded on rank:{rank}")
    else: 
        model = DistributedDataParallel(model, find_unused_parameters=True)
        print(f"DDP model loaded on rank:{rank}")

    # initialization of the optimizer after wrapping the model
    optimizer = AdamW(model.parameters(), lr=args["learning_rate"])
    scheduler = StepLR(optimizer, step_size=1, gamma=args["scheduler_gamma"])
    if args["load_general_checkpoint_path"]:
        checkpoint_state_dicts = torch.load(args["load_general_checkpoint_path"], weights_only=True)
        optimizer_state_dict = checkpoint_state_dicts["optimizer_state_dict"]
        scheduler_state_dict = checkpoint_state_dicts["scheduler_state_dict"]
        optimizer.load_state_dict(optimizer_state_dict)
        scheduler.load_state_dict(scheduler_state_dict)

    # initialization of scaler for mixed precision and control of gradient accumulation
    grad_scaler = GradScaler("cuda")

    # core loop of epochs
    for epoch_idx in range(1, args["num_epochs"] + 1):
        # shuffle data differently at each epoch across all processes
        train_sampler.set_epoch(epoch=epoch_idx)

        train_epoch(
            rank=rank,
            current_epoch=epoch_idx,
            model=model,    
            dataloader=train_loader,
            optimizer=optimizer,
            scaler=grad_scaler,
            args=args
        )
        scheduler.step()
        dist.barrier()  # use a barrier to make sure training is done on all ranks
        
        eval_epoch(
            rank=rank,
            model=model,
            current_epoch=epoch_idx,
            dataloader=eval_loader,
            args=args
        )
        dist.barrier()

        # save model checkpoint with CPU offload to avoid CUDA OOM, save/load_pretrained not available in FSDP
        if epoch_idx == 1 or epoch_idx == args["num_epochs"] or epoch_idx % args["save_every_epochs"] == 0:
            if args["wrap_model"]:  # FSDP
                save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
                with FullyShardedDataParallel.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_policy):
                    model_state_dict = model.state_dict()
            else:  # DDP
                model_state_dict = model.module.state_dict()
            if rank == 0:
                checkpoint_path = os.path.join(args["save_checkpoint_dir"], f"general_checkpoint_{epoch_idx}.pt")
                torch.save(
                    {
                        "model_state_dict": model_state_dict,
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                    },
                    checkpoint_path
                )
                print(f"Saving {checkpoint_path}")
            dist.barrier()

    cleanup()


def train_distributed(args: Dict[str, Any]):
    """
    Core training process across multiple devices with epochs of training and inter-epoch evaluation.
    Use args: Dict[str, Any] instead of **kwargs for compatibility with torch.multiprocessing.spawn.
    """
    torch.multiprocessing.spawn(
        train_on_device, 
        args=(args["world_size"], args),
        nprocs=args["world_size"],
        join=True
    )


if __name__ == '__main__':
    # suppress messages from AutoTokenizer parallelism and Graphein respectively
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    os.environ["LOGURU_LEVEL"] = "INFO"

    parsed_args = argParser.parse_args()
    # os.environ['CUDA_VISIBLE_DEVICES'] = '1'  # restrict PyTorch to see only the specified GPUs
    parsed_args.world_size = torch.cuda.device_count()  # use up all available devices across nodes

    torch.manual_seed(parsed_args.random_seed)
    torch.cuda.manual_seed(parsed_args.random_seed)
    
    # initialize checkpoint directory
    timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
    parsed_args.save_checkpoint_dir = os.path.join(parsed_args.save_checkpoint_dir, f"checkpoints_{timestamp}")
    if not os.path.exists(parsed_args.save_checkpoint_dir):
        os.mkdir(parsed_args.save_checkpoint_dir)

    print("####################")
    for key, value in parsed_args.__dict__.items(): 
        print(f"{key}: {value}")
    print("####################")

    train_distributed(parsed_args.__dict__)
