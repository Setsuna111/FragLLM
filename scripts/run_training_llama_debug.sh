#!/bin/bash
# Example script to run fragment training with transformers.Trainer
# Modify the paths and parameters according to your setup
export MASTER_PORT=$(shuf -i 2000-65000 -n 1)
export WANDB_DISABLED=true
export PYTHONPATH=$(pwd):$PYTHONPATH
export TOKENIZERS_PARALLELISM=false
deepspeed  --include "localhost:4" --master_port $MASTER_PORT  scripts/train_llama.py \
    --esm_path "/home/djy/projects/Data/HF_models/esm2_t36_3B_UR50D" \
    --llama_path "/home/djy/projects/Data/HF_models/RedHatAI-Llama-3.1-8B-Instruct" \
    --dropout_rate 0.3 \
    --intermediate_dim 2048 \
    --perceiver_latent_size 1 \
    --num_perceiver_heads 8 \
    --num_perceiver_layers 2 \
    --root_dir "./data" \
    --dataset_train_config "ProFunction||ActRefClass||BindIRefClass" \
    --sample_rate_train "1,1,1" \
    --max_sequence_length 1021 \
    --filter_sequence False \
    --output_dir "./checkpoints/fragment_training_test" \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --save_strategy "steps" \
    --save_steps 200 \
    --save_total_limit 3 \
    --learning_rate 2e-5 \
    --lr_scheduler_type "cosine" \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --logging_steps 1 \
    --dataloader_num_workers 0 \
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
    --lora_target_modules "self_attn.q_proj,self_attn.k_proj,self_attn.v_proj,self_attn.o_proj,mlp.gate_proj,mlp.up_proj,mlp.down_proj" \

