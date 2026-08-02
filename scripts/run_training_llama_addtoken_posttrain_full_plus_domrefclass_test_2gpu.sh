#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPT_DIR="$PROJECT_ROOT/scripts"

export CUDA_VISIBLE_DEVICES="0,1"
export CUDA_HOME="/home/dataset-local/anaconda3/envs/fragllm"
export PATH="/home/dataset-local/anaconda3/envs/fragllm/bin:${PATH:-}"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export WANDB_DISABLED=true
export DS_SKIP_CUDA_CHECK=1
export MASTER_PORT="${MASTER_PORT:-29528}"

PYTHON_BIN="/home/dataset-local/anaconda3/envs/fragllm/bin/python"
DEEPSPEED_BIN="/home/dataset-local/anaconda3/envs/fragllm/bin/deepspeed"
DATA_ROOT="$PROJECT_ROOT/data_70"
RESUME_CHECKPOINT_PATH="${1:-$PROJECT_ROOT/checkpoints/0529_all/checkpoint-315000}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/checkpoints/posttrain_full_plus_domrefclass_test_2gpu}"

# One approximate epoch of (full train + DomRefClass test) at effective batch 64.
# Override ADDITIONAL_TRAINING_STEPS when a different continuation length is needed.
ADDITIONAL_TRAINING_STEPS="${ADDITIONAL_TRAINING_STEPS:-21480}"
NUM_NEW_CHECKPOINTS="${NUM_NEW_CHECKPOINTS:-30}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-$((NUM_NEW_CHECKPOINTS + 1))}"

if (( ADDITIONAL_TRAINING_STEPS <= 0 )); then
    echo "ADDITIONAL_TRAINING_STEPS must be positive" >&2
    exit 1
fi
if (( ADDITIONAL_TRAINING_STEPS % NUM_NEW_CHECKPOINTS != 0 )); then
    echo "ADDITIONAL_TRAINING_STEPS must be divisible by $NUM_NEW_CHECKPOINTS" >&2
    exit 1
fi
SAVE_STEPS=$((ADDITIONAL_TRAINING_STEPS / NUM_NEW_CHECKPOINTS))

export POSTTRAIN_DATA_ROOT="$DATA_ROOT"
export POSTTRAIN_DATA_MANIFEST="$OUTPUT_DIR/data_manifest.json"

mkdir -p "$OUTPUT_DIR"
exec > >(tee -a "$OUTPUT_DIR/training.log") 2>&1

if [[ ! -d "$RESUME_CHECKPOINT_PATH" ]]; then
    echo "Missing resume checkpoint: $RESUME_CHECKPOINT_PATH" >&2
    exit 1
fi
for required in trainer_state.json latest; do
    if [[ ! -f "$RESUME_CHECKPOINT_PATH/$required" ]]; then
        echo "Resume checkpoint is missing $required: $RESUME_CHECKPOINT_PATH" >&2
        exit 1
    fi
done

"$PYTHON_BIN" -m py_compile \
    "$SCRIPT_DIR/zero2_resume_compat_4to2.py" \
    "$SCRIPT_DIR/train_llama_addtoken_posttrain_full_plus_domrefclass_test_2gpu.py"

echo "[posttrain-2gpu] physical GPUs=0,1"
echo "[posttrain-2gpu] source=$RESUME_CHECKPOINT_PATH"
echo "[posttrain-2gpu] data=full data_70 train + data_70 DomRefClass test"
echo "[posttrain-2gpu] per_device_train_batch_size=8"
echo "[posttrain-2gpu] gradient_accumulation_steps=4"
echo "[posttrain-2gpu] effective global batch=2*8*4=64"
echo "[posttrain-2gpu] additional_steps=$ADDITIONAL_TRAINING_STEPS"
echo "[posttrain-2gpu] save_steps=$SAVE_STEPS, new_checkpoints=$NUM_NEW_CHECKPOINTS"
echo "[posttrain-2gpu] save_only_model=False; optimizer state and non-LoRA state are retained"

"$DEEPSPEED_BIN" --include "localhost:0,1" --master_port "$MASTER_PORT" \
    "$SCRIPT_DIR/train_llama_addtoken_posttrain_full_plus_domrefclass_test_2gpu.py" \
    --resume-checkpoint-path "$RESUME_CHECKPOINT_PATH" \
    --additional-training-steps "$ADDITIONAL_TRAINING_STEPS" \
    --esm_path "/home/dataset-local/projects_dir/pretrained_model/esm2_t36_3B_UR50D" \
    --llama_path "/home/dataset-local/projects_dir/pretrained_model/Llama-3.1-8B-Instruct" \
    --load_pro2text_checkpoint_dir "/home/dataset-local/projects_dir/pretrained_model/Prot2Text-V2-11B-Instruct-hf" \
    --protein_sam_checkpoint_path "$PROJECT_ROOT/model_grounding_segformer/checkpoints_grounding_3B_cluster_70_point_only/checkpoint_epoch_40.pt" \
    --use_detailed_template True \
    --frag_adapter_type hierarchical \
    --pos_decoder_type ProteinSAM \
    --dropout_rate 0.3 \
    --intermediate_dim 2048 \
    --perceiver_latent_size 4 \
    --num_perceiver_heads 8 \
    --num_perceiver_layers 2 \
    --ce_loss_weight 1.0 \
    --position_loss_weight 5.0 \
    --root_dir "$DATA_ROOT" \
    --dataset_train_config "ProFunction||ActRefClass||BindIRefClass||DomRefClass||EvoRefClass||MotifRefClass||ActRefDesc||BindIRefDesc||DomRefDesc||EvoRefDesc||MotifRefDesc||ActGroundSingle||BindIGroundSingle||MotifGroundSingle||DomGroundSingle||EvoGroundSingle||ActGroundGroup||BindIGroundGroup||DomGroundGroup||EvoGroundGroup||MotifGroundGroup" \
    --max_sequence_length 1021 \
    --filter_sequence False \
    --output_dir "$OUTPUT_DIR" \
    --num_train_epochs 50 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --save_strategy steps \
    --save_steps "$SAVE_STEPS" \
    --save_total_limit "$SAVE_TOTAL_LIMIT" \
    --save_only_model False \
    --learning_rate 5e-5 \
    --lr_scheduler_type cosine \
    --weight_decay 0. \
    --warmup_ratio 0.06 \
    --adam_epsilon 1e-6 \
    --adam_beta1 0.9 \
    --adam_beta2 0.999 \
    --logging_steps 10 \
    --dataloader_num_workers 8 \
    --dataloader_persistent_workers True \
    --ignore_data_skip True \
    --remove_unused_columns False \
    --bf16 True \
    --tf32 True \
    --gradient_checkpointing True \
    --deepspeed "$SCRIPT_DIR/zero2_offload_4to2_elastic.json" \
    --tune_fragment_adapter False \
    --tune_adapter False \
    --freeze_adapter False \
    --freeze_fragment_adapter False \
    --report_to tensorboard \
    --lora_enable True \
    --lora_r 32 \
    --lora_alpha 64 \
    --lora_dropout 0.1 \
    --lora_bias none \
    --lora_target_modules "self_attn.q_proj,self_attn.k_proj,self_attn.v_proj,self_attn.o_proj,mlp.gate_proj,mlp.up_proj,mlp.down_proj"
