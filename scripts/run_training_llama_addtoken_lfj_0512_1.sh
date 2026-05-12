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
# "ActGroundSingle||BindIGroundSingle||MotifGroundSingle||DomGroundSingle||EvoGroundSingle||ActGroundGroup||BindIGroundGroup||DomGroundGroup||EvoGroundGroup||MotifGroundGroup"
export DS_SKIP_CUDA_CHECK=1
deepspeed --include "localhost:2,3" --master_port $MASTER_PORT  scripts/train_llama_addtoken.py \
    --esm_path "/home/dataset-local/projects_dir/pretrained_model/esm2_t36_3B_UR50D" \
    --llama_path "/home/dataset-local/projects_dir/pretrained_model/Llama-3.1-8B-Instruct" \
    --load_pro2text_checkpoint_dir "/home/dataset-local/projects_dir/pretrained_model/Prot2Text-V2-11B-Instruct-hf" \
    --protein_sam_checkpoint_path './model_grounding_segformer/checkpoints_grounding_3B_cluster_70_point_only/checkpoint_epoch_40.pt' \
    --use_detailed_template True \
    --frag_adapter_type "qformer" \
    --pos_decoder_type "ProteinSAM" \
    --dropout_rate 0.3 \
    --intermediate_dim 2048 \
    --perceiver_latent_size 1 \
    --num_perceiver_heads 8 \
    --num_perceiver_layers 2 \
    --ce_loss_weight 1.0 \
    --position_loss_weight 5.0 \
    --root_dir "./data_70" \
    --dataset_train_config "ActRefClass||BindIRefClass||EvoRefClass||MotifRefClass" \
    --max_sequence_length 1021 \
    --filter_sequence False \
    --output_dir "/home/dataset-local/projects_dir/FragLLM/checkpoints/0512_ref_small_4" \
    --num_train_epochs 50 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 16 \
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
    --lora_target_modules "self_attn.q_proj,self_attn.k_proj,self_attn.v_proj,self_attn.o_proj,mlp.gate_proj,mlp.up_proj,mlp.down_proj" \

    # -dataset_valid_config "ActRefClass||BindIRefClass||DomRefClass||EvoRefClass||MotifRefClass" \
    # --sample_rate_valid "1,1,1,1,1" \
    # --evaluation_strategy "steps" \

