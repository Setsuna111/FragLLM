#!/bin/bash
export MASTER_PORT=$(shuf -i 2000-65000 -n 1)
# export WANDB_DISABLED=true
# export WANDB_PROJECT="FragLLM_2601_alter"
export PYTHONPATH=$(pwd):$PYTHONPATH
export TOKENIZERS_PARALLELISM=false
DEVICES="localhost:0,1,2,3" 
RUN_NAME="alter_test_sh"
deepspeed --include $DEVICES --master_port $MASTER_PORT  scripts/train_llama_addtoken_alter.py \
    --esm_path "/home/dataset-local/projects/Data/HF_models/esm2_t36_3B_UR50D" \
    --llama_path "/home/dataset-local/projects/Data/HF_models/Meta-Llama-3.1-8B-Instruct" \
    --load_pro2text_checkpoint_dir '/home/dataset-local/projects/Data/HF_models/Prot2Text-V2-11B-Instruct-hf' \
    --protein_sam_checkpoint_path './model_grounding_segformer/checkpoints_grounding_3B/checkpoint_epoch_0.pt' \
    --use_detailed_template \
    --frag_adapter_type "multilevel" \
    --pos_decoder_type "Simple" \
    --dropout_rate 0.3 \
    --intermediate_dim 2048 \
    --perceiver_latent_size 1 \
    --num_perceiver_heads 8 \
    --num_perceiver_layers 2 \
    --ce_loss_weight 1.0 \
    --position_loss_weight 0.5 \
    --root_dir "./data" \
    --dataset_grounding_config "ActGroundSingle||BindIGroundSingle||MotifGroundSingle||DomGroundSingle||EvoGroundSingle||ActGroundGroup||BindIGroundGroup||DomGroundGroup||EvoGroundGroup||MotifGroundGroup" \
    --dataset_referring_config "ActRefClass||BindIRefClass||DomRefClass||EvoRefClass||MotifRefClass||ActRefDesc||BindIRefDesc||DomRefDesc||EvoRefDesc||MotifRefDesc" \
    --dataset_func_config "ProFunction" \
    --dataset_valid_config "ActGroundSingle" \
    --max_sequence_length 1021 \
    --output_dir "./checkpoints/$RUN_NAME" \
    --run_name $RUN_NAME \
    --num_train_epochs 2 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --learning_rate 5e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.06 \
    --adam_epsilon 1e-6 \
    --adam_beta1 0.9 \
    --adam_beta2 0.999 \
    --logging_steps 1 \
    --dataloader_num_workers 0 \
    --deepspeed ./scripts/zero2.json \
    --report_to wandb \
    --lora_enable \
    --lora_r 32 \
    --lora_alpha 64 \
    --lora_dropout 0.1 \
    --lora_bias "none" \
    --lora_target_modules "self_attn.q_proj,self_attn.k_proj,self_attn.v_proj,self_attn.o_proj,mlp.gate_proj,mlp.up_proj,mlp.down_proj" \
    --grounding_validation \
    --precision bf16 \
    --steps_per_epoch 10 \
    --start_epoch 0 \

    # -dataset_valid_config "ActRefClass||BindIRefClass||DomRefClass||EvoRefClass||MotifRefClass" \
    # --sample_rate_valid "1,1,1,1,1" \
    # --evaluation_strategy "steps" \

