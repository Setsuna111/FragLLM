#!/bin/bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# The raw checkpoint was saved by four DeepSpeed ranks; use the same topology.
export MASTER_PORT="${MASTER_PORT:-29501}"
export WANDB_DISABLED=true
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export DS_SKIP_CUDA_CHECK=1

PYTHON_BIN="/home/dataset-local/anaconda3/envs/fragllm/bin/python"
RESUME_CHECKPOINT_PATH="${1:-/home/dataset-local/projects_dir/FragLLM/checkpoints/0529_all/checkpoint-315000/}"
TASK4_TRAIN_CSV="/home/dataset-local/projects_dir/FragLLM/data_70/Pro2Text/task4_cot_train.csv"
OUTPUT_DIR="/home/dataset-local/projects_dir/FragLLM/checkpoints/task4_cot_0529_all"

# The Python entry point loads raw DeepSpeed module state through a local
# checkpoint-0 proxy, while resetting Task4 optimizer, scheduler, RNG, and step.
deepspeed --include "localhost:0,1,2,3" --master_port "$MASTER_PORT" scripts/train_llama_addtoken_task4_cot.py \
    --resume_checkpoint_path "$RESUME_CHECKPOINT_PATH" \
    --task4_cot_train_csv "$TASK4_TRAIN_CSV" \
    --esm_path "/home/dataset-local/projects_dir/pretrained_model/esm2_t36_3B_UR50D" \
    --protein_sam_checkpoint_path './model_grounding_segformer/checkpoints_grounding_3B_cluster_70_point_only/checkpoint_epoch_40.pt' \
    --use_detailed_template True \
    --frag_adapter_type "hierarchical" \
    --pos_decoder_type "ProteinSAM" \
    --dropout_rate 0.3 \
    --intermediate_dim 2048 \
    --perceiver_latent_size 4 \
    --num_perceiver_heads 8 \
    --num_perceiver_layers 2 \
    --ce_loss_weight 1.0 \
    --position_loss_weight 5.0 \
    --root_dir "./data_70" \
    --dataset_train_config "Task4CoT" \
    --max_sequence_length 1021 \
    --filter_sequence False \
    --output_dir "$OUTPUT_DIR" \
    --num_train_epochs 50 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 20 \
    --learning_rate 5e-5 \
    --lr_scheduler_type "cosine" \
    --weight_decay 0. \
    --warmup_ratio 0.06 \
    --adam_epsilon 1e-6 \
    --adam_beta1 0.9 \
    --adam_beta2 0.999 \
    --logging_steps 1 \
    --dataloader_num_workers 8 \
    --remove_unused_columns False \
    --bf16 True \
    --tf32 True \
    --gradient_checkpointing True \
    --deepspeed ./scripts/zero2_offload.json \
    --tune_fragment_adapter False \
    --tune_adapter False \
    --freeze_adapter False \
    --freeze_fragment_adapter False \
    --report_to tensorboard \
    --lora_enable True \
    --lora_r 32 \
    --lora_alpha 64 \
    --lora_dropout 0.1 \
    --lora_bias "none" \
    --lora_target_modules "self_attn.q_proj,self_attn.k_proj,self_attn.v_proj,self_attn.o_proj,mlp.gate_proj,mlp.up_proj,mlp.down_proj"
