#!/bin/bash
# Example script to run fragment training with transformers.Trainer
# Modify the paths and parameters according to your setup
export MASTER_PORT=$(shuf -i 2000-65000 -n 1)
export WANDB_DISABLED=true
export PYTHONPATH=$(pwd):$PYTHONPATH
export TOKENIZERS_PARALLELISM=false
# export CUDA_LAUNCH_BLOCKING=1
# export TORCH_USE_CUDA_DSA=1
# --include localhost:0
export DS_SKIP_CUDA_CHECK=1
deepspeed --include "localhost:0,1,2" --master_port $MASTER_PORT  scripts/train_llama_addtoken.py \
    --esm_path "/home/lfj/projects_dir/pretrained_model/esm2_t36_3B_UR50D" \
    --llama_path "/home/lfj/projects_dir/pretrained_model/Llama-3.1-8B-Instruct" \
    --load_adapter_checkpoint_dir "/home/lfj/projects_dir/FragLLM/checkpoints_contrast/checkpoints_allgather_250827_143913/model_checkpoint_12.pt" \
    --use_detailed_template True \
    --grounding_model_type "cls" \
    --protein_sam_checkpoint_path "/home/lfj/projects_dir/FragLLM/model_grounding_cls/checkpoints_grounding/best_model.pt" \
    --dropout_rate 0.3 \
    --intermediate_dim 2048 \
    --perceiver_latent_size 1 \
    --num_perceiver_heads 8 \
    --num_perceiver_layers 2 \
    --ce_loss_weight 1.0 \
    --position_loss_weight 0.5 \
    --root_dir "./data" \
    --dataset_train_config "ActGroundSingle||BindIGroundSingle||MotifGroundSingle||DomGroundSingle||EvoGroundSingle||ActGroundGroup||BindIGroundGroup||DomGroundGroup||EvoGroundGroup||MotifGroundGroup" \
    --max_sequence_length 1021 \
    --filter_sequence False \
    --output_dir "/home/lfj/projects_dir/FragLLM/checkpoints/grounding_lora_base" \
    --num_train_epochs 50 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --save_strategy "steps" \
    --save_steps 5000 \
    --save_total_limit 3 \
    --learning_rate 5e-5 \
    --lr_scheduler_type "cosine" \
    --weight_decay 0. \
    --warmup_ratio 0.06 \
    --adam_epsilon 1e-6 \
    --adam_beta1 0.9 \
    --adam_beta2 0.999 \
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

    # -dataset_valid_config "ActRefClass||BindIRefClass||DomRefClass||EvoRefClass||MotifRefClass" \
    # --sample_rate_valid "1,1,1,1,1" \
    # --evaluation_strategy "steps" \

