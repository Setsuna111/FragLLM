# Fragment Training with Transformers Trainer

This document explains how to use `train_trainer.py` to train the `Esm2LlamaInstructForCausalLM` model using the HuggingFace Transformers Trainer framework.

## Features

- ✅ **Transformers Trainer Integration**: Uses the official HuggingFace Trainer for robust training
- ✅ **LoRA Fine-tuning**: Supports efficient parameter-efficient fine-tuning with LoRA
- ✅ **Fragment-Aware Training**: Supports fragment position references and grounding
- ✅ **Checkpoint Management**: Automatic saving and loading of model checkpoints
- ✅ **Custom Data Collator**: Uses `FragDataCollator` for fragment-specific data processing
- ✅ **Multi-Component Model**: Handles ESM encoder, modality adapter, and LLaMA decoder
- ✅ **Flexible Configuration**: Supports various training configurations through command-line arguments

## Model Architecture

The training script works with `Esm2LlamaInstructForCausalLM` which consists of:

1. **ESM Encoder**: Protein sequence encoder
2. **Modality Adapter**: Bridges protein and text representations
3. **Fragment Adapter**: Handles fragment-specific processing with Perceiver architecture
4. **LLaMA Decoder**: Text generation decoder

## Dataset Support

The script uses `FragRefDataset` which supports various fragment-related tasks:

- **referring_desc**: Fragment description generation
- **referring_class**: Fragment classification
- **grounding_single**: Single fragment grounding
- **grounding_group**: Multiple fragment grounding

## Usage

### Basic Usage

```bash
python train_trainer.py \
    --esm_path "/path/to/esm2_model" \
    --llama_path "/path/to/llama_model" \
    --root_dir "/path/to/fragment_data" \
    --data_name "Venux_Dom" \
    --task_type "referring_desc" \
    --output_dir "./checkpoints" \
    --num_train_epochs 3 \
    --per_device_train_batch_size 4 \
    --learning_rate 2e-5
```

### Advanced Configuration

```bash
python train_trainer.py \
    --esm_path "/path/to/esm2_t33_650M_UR50D" \
    --llama_path "/path/to/Meta-Llama-3.1-8B-Instruct-hf" \
    --root_dir "/path/to/fragment_data" \
    --data_name "Venux_Dom" \
    --task_type "referring_desc" \
    --output_dir "./checkpoints/fragment_training" \
    --overwrite_output_dir \
    --num_train_epochs 5 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "epoch" \
    --save_strategy "epoch" \
    --save_total_limit 3 \
    --learning_rate 1e-4 \
    --weight_decay 0.01 \
    --warmup_ratio 0.1 \
    --logging_steps 50 \
    --fp16 \
    --dataloader_num_workers 4 \
    --remove_unused_columns False \
    --lora_rank 64 \
    --lora_alpha 128 \
    --lora_dropout 0.1 \
    --intermediate_dim 4096 \
    --perceiver_latent_size 1 \
    --num_perceiver_heads 16 \
    --num_perceiver_layers 12 \
    --max_sequence_length 1021 \
    --torch_dtype "float16"
```

## Key Arguments

### Model Arguments
- `--esm_path`: Path to ESM model
- `--llama_path`: Path to LLaMA model
- `--load_model_checkpoint_path`: Path to load pretrained model weights
- `--load_adapter_checkpoint_dir`: Path to load LoRA adapter weights

### LoRA Arguments
- `--lora_rank`: LoRA rank (default: 32)
- `--lora_alpha`: LoRA alpha (default: 64)
- `--lora_dropout`: LoRA dropout (default: 0.1)
- `--target_modules`: Target modules for LoRA

### Data Arguments
- `--root_dir`: Root directory containing datasets
- `--data_name`: Dataset name (e.g., "Venux_Dom", "Venux_Act")
- `--task_type`: Task type ("referring_desc", "referring_class", etc.)
- `--max_sequence_length`: Maximum protein sequence length

### Training Arguments
All standard HuggingFace training arguments are supported:
- `--output_dir`: Output directory for checkpoints
- `--num_train_epochs`: Number of training epochs
- `--per_device_train_batch_size`: Batch size per device
- `--learning_rate`: Learning rate
- `--evaluation_strategy`: Evaluation strategy
- `--save_strategy`: Checkpoint saving strategy

## Checkpointing

The script automatically saves:
- LoRA adapter weights
- Modality adapter weights  
- Fragment adapter weights
- Optimizer and scheduler states

To resume training from a checkpoint:
```bash
python train_trainer.py \
    [your arguments] \
    --resume_from_checkpoint "/path/to/checkpoint-1000"
```

## Data Format

The script expects data in the format used by `FragRefDataset`:

### For JSON datasets:
```json
{
  "sequence": "MKTV...",
  "conversation": [
    {"role": "user", "content": "Describe the fragment..."},
    {"role": "assistant", "content": "This fragment..."}
  ],
  "answer": "This fragment represents...",
  "position_ref": [10, 50],
  "position_grd": [[10, 20], [30, 40]],
  "start": 0
}
```

### For CSV datasets:
Columns should include: sequence, conversation, answer, position_ref, position_grd, start

## Memory Requirements

- **Minimum GPU Memory**: 16GB (with small batch sizes and fp16)
- **Recommended GPU Memory**: 24GB or higher
- **Batch Size Guidelines**:
  - 16GB GPU: batch_size=1-2, gradient_accumulation_steps=8-16
  - 24GB GPU: batch_size=2-4, gradient_accumulation_steps=4-8
  - 48GB GPU: batch_size=4-8, gradient_accumulation_steps=2-4

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**:
   - Reduce `per_device_train_batch_size`
   - Increase `gradient_accumulation_steps`
   - Use `fp16` or `bf16`

2. **Data Loading Issues**:
   - Check `root_dir` and `data_name` paths
   - Verify data format matches expected structure
   - Ensure `FragRefDataset` can load your data

3. **Model Loading Issues**:
   - Verify `esm_path` and `llama_path` are correct
   - Check model compatibility with expected architectures

### Performance Tips

1. **Use Mixed Precision**: Add `--fp16` or `--bf16`
2. **Optimize Data Loading**: Set `--dataloader_num_workers 4`
3. **Use Gradient Checkpointing**: Add `--gradient_checkpointing`
4. **Tune Batch Size**: Balance memory usage and training speed

## Example Results

After successful training, you should see outputs like:
```
Model:
PeftModel(
  (base_model): Esm2LlamaInstructForCausalLM(...)
)
Training dataset size: 10000
Evaluation dataset size: 1000
Starting training from scratch...
...
Training completed!
```

## Comparison with train_instruct.py

| Feature | train_trainer.py | train_instruct.py |
|---------|------------------|-------------------|
| Framework | HuggingFace Trainer | Custom DDP |
| Ease of Use | High | Medium |
| Customization | Medium | High |
| Built-in Features | Many (logging, checkpointing, etc.) | Manual implementation |
| Multi-GPU | Automatic | Manual DDP setup |
| Debugging | Easier | More complex |

Choose `train_trainer.py` for easier development and standard workflows.
Choose `train_instruct.py` for maximum control and custom training loops.
