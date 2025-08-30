#!/bin/bash
# Example script to run fragment training with transformers.Trainer
# Modify the paths and parameters according to your setup
export MASTER_PORT=$(shuf -i 2000-65000 -n 1)
export WANDB_DISABLED=true
export PYTHONPATH=$(pwd):$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0
deepspeed  --master_port $MASTER_PORT  scripts/train_trainer.py \
    --esm_path "/home/djy/projects/Data/HF_models/esm2_t36_3B_UR50D" \
    --llama_path "/home/djy/projects/Data/HF_models/RedHatAI-Llama-3.1-8B-Instruct" \
    --root_dir "./data" \
    --dataset_train_config "ProFunction||ActRefClass||BindIRefClass||DomRefClass||EvoRefClass||MotifRefClass" \
    --dataset_valid_config "ProFunction" \
    --output_dir "./checkpoints/fragment_training_test" \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy "steps" \
    --save_strategy "steps" \
    --save_total_limit 3 \
    --learning_rate 2e-5 \
    --weight_decay 0.01 \
    --warmup_ratio 0.1 \
    --logging_steps 100 \
    --dataloader_num_workers 0 \
    --remove_unused_columns False \
    --lora_enable True \
    --lora_r 32 \
    --lora_alpha 64 \
    --lora_dropout 0.1 \
    --lora_bias "none" \
    --lora_target_modules "self_attn.q_proj,self_attn.k_proj,self_attn.v_proj,self_attn.o_proj,mlp.gate_proj,mlp.up_proj,mlp.down_proj" \
    --max_sequence_length 1021 \
    --bf16 True \
    --tf32 True \
    --gradient_checkpointing True \
    --deepspeed ./scripts/zero3.json

