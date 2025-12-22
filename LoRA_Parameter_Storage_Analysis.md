# LoRA 外设置参数存储机制详细分析

## 一、整体架构概览

### 模型结构层次
```
ProteinLlamaForCausalLM
├── LlamaForCausalLM (基础LLaMA模型)
│   └── lm_head (线性层)
│
└── ProteinMetaModel (蛋白质模型元信息)
    ├── esm_encoder (冻结的ESM2编码器)
    ├── adapter (模态适配器 - ModalityAdapter)
    │   ├── fc1: Linear(protein_emb_dim -> intermediate_dim)
    │   ├── fc2: Linear(intermediate_dim -> text_emb_dim)
    │   ├── activation: GELU
    │   └── dropout: Dropout
    │
    └── fragment_adapter (片段适配器 - FragmentAdapter)
        └── perceiver_layer: MultiScalePerceiver
            ├── latents_global: Parameter
            ├── latents_fragment: Parameter
            ├── global_latent_norm: LayerNorm
            ├── fragment_latent_norm: LayerNorm
            ├── multi_scale_perceiver: MultiScalePerceiverLayer
            │   ├── global_attn: MultiheadAttention
            │   ├── global_ffn: FeedForwardNetwork
            │   ├── fragment_attn: MultiheadAttention
            │   ├── fragment_ffn: FeedForwardNetwork
            │   └── 自适应门控机制
            └── self_attention_layers: ModuleList of AttentionLayer
```

---

## 二、可更新参数分类

### 2.1 LoRA 相关参数

**应用位置**: LLaMA 模型的特定层

**参数名称模式**:
- `lora_A`: LoRA矩阵A（从原始维度投影到秩r）
- `lora_B`: LoRA矩阵B（从秩r投影回原始维度）
- 仅在 `lora_bias="all"` 时保存bias

**目标模块** (`lora_target_modules`):
```
self_attn.q_proj      # Query投影
self_attn.k_proj      # Key投影
self_attn.v_proj      # Value投影
self_attn.o_proj      # Output投影
mlp.gate_proj         # 门控投影
mlp.up_proj           # 上投影
mlp.down_proj         # 下投影
```

**LoRA 配置**:
- `lora_r`: 秩 (例如: 32)
- `lora_alpha`: 缩放因子 (例如: 64)
- `lora_dropout`: Dropout率 (例如: 0.05)
- `lora_bias`: "none" | "all" | "lora_only"

**参数数量示例** (以 r=32, alpha=64为例):
```
对于维度为4096的层:
- lora_A: [4096, 32] (131,072参数)
- lora_B: [32, 4096]   (131,072参数)
- 总计每层: ~262K参数
- 7层目标模块: ~1.8M LoRA参数
```

### 2.2 非LoRA 可学习参数

当启用LoRA训练时，LLaMA主体冻结，但以下参数仍可学习:

**1. 适配器层 (Adapter)**
- `model.adapter.fc1.weight`: [intermediate_dim, protein_emb_dim]
- `model.adapter.fc1.bias`: [intermediate_dim]
- `model.adapter.fc2.weight`: [text_emb_dim, intermediate_dim]
- `model.adapter.fc2.bias`: [text_emb_dim]
- `model.adapter.ln1.weight` & `model.adapter.ln1.bias` (已弃用)
- `model.adapter.ln2.weight` & `model.adapter.ln2.bias` (已弃用)

**参数数量** (以 protein_emb_dim=1280, intermediate_dim=2048, text_emb_dim=4096为例):
```
fc1: 1280 * 2048 + 2048 = 2,624,512参数
fc2: 2048 * 4096 + 4096 = 8,392,704参数
总计adapter: ~11M参数
```

**2. 片段适配器 (Fragment Adapter)**
- `model.fragment_adapter.protein_layer_norm.weight` & `.bias`
- `model.fragment_adapter.perceiver_layer.latents_global`: Parameter [latent_size, text_emb_dim]
- `model.fragment_adapter.perceiver_layer.latents_fragment`: Parameter [latent_size, protein_emb_dim]
- `model.fragment_adapter.perceiver_layer.global_latent_norm.*`
- `model.fragment_adapter.perceiver_layer.fragment_latent_norm.*`
- `model.fragment_adapter.perceiver_layer.multi_scale_perceiver.*` (所有权重)
- `model.fragment_adapter.perceiver_layer.self_attention_layers[*].*` (所有权重)
- `model.fragment_adapter.perceiver_layer.output_proj.*`
- `model.fragment_adapter.perceiver_layer.out_layer_norm.*`

**参数数量** (以 latent_size=1, num_heads=8, num_layers=2为例):
```
latents和层norm: 相对较小
MultiScalePerceiver中的注意力层: 较大
总计: ~500K-1M参数 (取决于具体维度)
```

---

## 三、参数保存机制

### 3.1 训练过程中的检查点保存

**文件位置**: `train_llama_lfj.py:455-494`

```python
def _save_checkpoint(self, model, trial, metrics=None):
    if getattr(self.args, 'tune_frag_adapter', False):
        # 场景1: 仅训练片段适配器
        keys_to_match = ['fragment_adapter']
        weight_to_save = get_frag_adapter_state_maybe_zero_3(
            self.model.named_parameters(), keys_to_match
        )
        # 保存到: {output_dir}/checkpoint-{step}/fragment_adapter.bin
        torch.save(weight_to_save,
                   os.path.join(output_dir, 'fragment_adapter.bin'))
    else:
        # 场景2: 标准保存 + LoRA额外保存
        super(FragTrainer, self)._save_checkpoint(model, trial, metrics)

        if getattr(self.args, 'lora_enable', False):
            # 保存非LoRA可学习权重
            non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(
                self.model.named_parameters(),
                require_grad_only=True
            )
            # 保存到: {output_dir}/checkpoint-{step}/non_lora_trainables.bin
            torch.save(non_lora_state_dict,
                       os.path.join(output_dir, 'non_lora_trainables.bin'))
```

**保存内容分布**:

| 场景 | LoRA参数 | 非LoRA参数 | 位置 |
|------|---------|----------|------|
| tune_frag_adapter=True | ❌ | fragment_adapter | fragment_adapter.bin |
| lora_enable=True | ✅ (HF保存) | ✅ | checkpoint/adapter_model.bin + non_lora_trainables.bin |
| 标准训练 | ❌ | ✅ (全部) | checkpoint/pytorch_model.bin |

### 3.2 权重过滤函数

**函数1: `get_frag_adapter_state_maybe_zero_3()` (第197-200行)**
```python
def get_frag_adapter_state_maybe_zero_3(named_params, keys_to_match):
    # 选择名称中包含匹配字符串的参数
    to_return = {k: t for k, t in named_params
                 if any(key_match in k for key_match in keys_to_match)}
    # 处理DeepSpeed ZeRO-3分区参数
    to_return = {k: maybe_zero_3(v, ignore_status=True, name=k).cpu()
                 for k, v in to_return.items()}
    return to_return
```

**函数2: `get_peft_state_maybe_zero_3()` (第203-225行)**
```python
def get_peft_state_maybe_zero_3(named_params, bias):
    # 根据bias配置过滤LoRA参数
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params
                     if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        # 复杂逻辑: 只保存LoRA相关的bias
        ...

    # 处理DeepSpeed ZeRO-3
    to_return = {k: maybe_zero_3(v, ignore_status=True)
                 for k, v in to_return.items()}
    return to_return
```

**函数3: `get_peft_state_non_lora_maybe_zero_3()` (第227-232行)**
```python
def get_peft_state_non_lora_maybe_zero_3(named_params, require_grad_only=True):
    # 排除所有"lora_"开头的参数
    to_return = {k: t for k, t in named_params if "lora_" not in k}

    if require_grad_only:
        # 只保留需要梯度的参数
        to_return = {k: t for k, t in to_return.items()
                     if t.requires_grad}

    # 处理DeepSpeed ZeRO-3
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu()
                 for k, v in to_return.items()}
    return to_return
```

### 3.3 最终模型保存

**文件位置**: `train_llama_lfj.py:705-723`

```python
if training_args.lora_enable:
    # LoRA训练的最终保存
    state_dict = get_peft_state_maybe_zero_3(
        model.named_parameters(),
        training_args.lora_bias
    )
    # 仅保存LoRA权重 (HuggingFace PEFT标准格式)
    model.save_pretrained(output_dir, state_dict=state_dict)

    # 额外保存非LoRA权重
    non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(
        model.named_parameters()
    )
    torch.save(non_lora_state_dict,
               os.path.join(output_dir, 'non_lora_trainables.bin'))
else:
    # 标准训练的最终保存
    safe_save_model_for_hf_trainer(trainer=trainer,
                                   output_dir=output_dir)
```

---

## 四、训练期间参数梯度配置

### 4.1 参数冻结策略

**代码位置**: `train_llama_lfj.py:645-658`

```python
# 1. 情况1: tune_adapter=True (仅训练适配器)
if training_args.tune_adapter:
    model.requires_grad_(False)  # 冻结全部
    for p in model.get_model().adapter.parameters():
        p.requires_grad_(True)   # 解冻适配器

# 2. 情况2: freeze_adapter=True (冻结适配器)
if training_args.freeze_adapter:
    for p in model.get_model().adapter.parameters():
        p.requires_grad_(False)

# 3. 情况3: tune_fragment_adapter=True (仅训练片段适配器)
if training_args.tune_fragment_adapter:
    model.requires_grad_(False)  # 冻结全部
    for p in model.get_model().fragment_adapter.parameters():
        p.requires_grad_(True)   # 解冻片段适配器

# 4. 情况4: freeze_fragment_adapter=True (冻结片段适配器)
if training_args.freeze_fragment_adapter:
    for p in model.get_model().fragment_adapter.parameters():
        p.requires_grad_(False)
```

### 4.2 优化器参数分组

**代码位置**: `train_llama_lfj.py:417-453`

```python
def create_optimizer(self):
    # 参数分类
    protein_params = []          # 蛋白质编码器相关
    llm_params = []              # LLaMA主体
    protein_params_wo_decay = [] # 蛋白质(无weight decay)
    llm_params_wo_decay = []     # LLaMA(无weight decay)

    decay_params = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
    decay_params = [n for n in decay_params if "bias" not in n]

    for k, v in opt_model.named_parameters():
        if v.requires_grad:
            # 归类: esm_encoder, adapter, fragment_adapter
            if any(component in k for component in
                   ["esm_encoder", "adapter", "fragment_adapter"]):
                if k in decay_params:
                    protein_params.append(v)
                else:
                    protein_params_wo_decay.append(v)
            else:
                if k in decay_params:
                    llm_params.append(v)
                else:
                    llm_params_wo_decay.append(v)

    # 所有组使用相同的学习率
    optimizer_grouped_parameters = [
        {
            "params": protein_params + llm_params,
            "lr": self.args.learning_rate,
            "weight_decay": self.args.weight_decay
        },
        {
            "params": protein_params_wo_decay + llm_params_wo_decay,
            "lr": self.args.learning_rate,
            "weight_decay": 0.0
        },
    ]
```

---

## 五、恢复训练中的权重加载

### 5.1 LoRA 权重恢复机制

**代码位置**: `train_llama_lfj.py:664-688`

```python
checkpoints = list(pathlib.Path(training_args.output_dir).glob("checkpoint-*"))
resume_from_checkpoint = len(checkpoints) > 0

if resume_from_checkpoint:
    latest_checkpoint = max(checkpoints, key=lambda x: int(x.name.split('-')[1]))

    if training_args.lora_enable:
        non_lora_path = os.path.join(latest_checkpoint, 'non_lora_trainables.bin')

        if os.path.exists(non_lora_path):
            # 加载保存的非LoRA权重
            non_lora_state_dict = torch.load(non_lora_path, map_location='cpu')
            load_result = model.load_state_dict(non_lora_state_dict, strict=False)

            if load_result.unexpected_keys:
                print(f"Warning: Unexpected keys: {load_result.unexpected_keys}")
            if load_result.missing_keys:
                print(f"Warning: Missing keys: {load_result.missing_keys}")
        else:
            print("Warning: non_lora_trainables.bin not found")
            print("Continuing with LoRA weights only...")

    # HuggingFace Trainer 自动处理 LoRA 权重的加载
    trainer.train(resume_from_checkpoint=True)
```

**恢复流程**:
1. HF Trainer 自动加载 `adapter_model.bin` (LoRA权重)
2. 脚本手动加载 `non_lora_trainables.bin` (适配器等)
3. 两者合并后继续训练

---

## 六、参数存储文件结构

### 6.1 LoRA 训练的检查点结构

```
output_dir/
├── checkpoint-1000/
│   ├── adapter_model.bin          # LoRA权重 (HF PEFT格式)
│   │   ├── base_model.model.model.layers[0].self_attn.q_proj.lora_A
│   │   ├── base_model.model.model.layers[0].self_attn.q_proj.lora_B
│   │   ├── base_model.model.model.layers[0].self_attn.k_proj.lora_A
│   │   ├── base_model.model.model.layers[0].self_attn.k_proj.lora_B
│   │   ├── ... (其他目标模块)
│   │   └── scaling参数
│   ├── non_lora_trainables.bin    # 非LoRA权重
│   │   ├── model.adapter.fc1.weight
│   │   ├── model.adapter.fc1.bias
│   │   ├── model.adapter.fc2.weight
│   │   ├── model.adapter.fc2.bias
│   │   ├── model.fragment_adapter.protein_layer_norm.weight
│   │   ├── model.fragment_adapter.protein_layer_norm.bias
│   │   ├── model.fragment_adapter.perceiver_layer.latents_global
│   │   ├── model.fragment_adapter.perceiver_layer.latents_fragment
│   │   └── ... (所有片段适配器参数)
│   ├── config.json                # 模型配置
│   └── trainer_state.json         # 训练状态
├── checkpoint-2000/
│   └── ...
└── final_model/
    ├── adapter_model.bin
    ├── non_lora_trainables.bin
    └── config.json
```

### 6.2 标准训练的检查点结构

```
output_dir/
├── checkpoint-1000/
│   ├── pytorch_model.bin          # 完整模型
│   ├── config.json
│   └── trainer_state.json
└── checkpoint-2000/
    └── ...
```

---

## 七、配置参数总结表

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `lora_enable` | bool | False | 启用LoRA微调 |
| `lora_r` | int | 8 | LoRA秩 |
| `lora_alpha` | int | 16 | LoRA缩放因子 |
| `lora_dropout` | float | 0.05 | LoRA dropout率 |
| `lora_bias` | str | "none" | bias处理方式 |
| `lora_target_modules` | str | 注意力和MLP | LoRA应用的模块 |
| `tune_adapter` | bool | False | 仅训练适配器 |
| `freeze_adapter` | bool | False | 冻结适配器 |
| `tune_fragment_adapter` | bool | False | 仅训练片段适配器 |
| `freeze_fragment_adapter` | bool | False | 冻结片段适配器 |
| `freeze_backbone` | bool | False | 冻结ESM2编码器 |

---

## 八、参数更新流程总结

```
训练开始
    ↓
初始化模型 (ProteinLlamaForCausalLM)
    ↓
[若lora_enable=True]
    ├→ 应用LoRA包装 (get_peft_model)
    ├→ 参数分类:
    │   ├→ LoRA参数 (lora_A, lora_B) ← 可梯度更新
    │   ├→ 适配器参数 (fc1, fc2) ← 可梯度更新
    │   ├→ 片段适配器参数 ← 可梯度更新
    │   └→ LLaMA主体参数 ← 冻结 (无梯度)
    │
[若tune_adapter=True]
    ├→ 模型全部冻结
    └→ 仅解冻adapter参数
    │
[若tune_fragment_adapter=True]
    ├→ 模型全部冻结
    └→ 仅解冻fragment_adapter参数
    │
创建优化器
    ├→ 参数分组 (蛋白质vs LLaMA, 衰减vs不衰减)
    └→ 所有组使用相同学习率
    │
每个检查点保存
    ├→ [lora_enable且非frag_adapter模式]:
    │   ├→ LoRA权重 → adapter_model.bin (HF保存)
    │   └→ 非LoRA权重 → non_lora_trainables.bin (手动保存)
    ├→ [frag_adapter模式]:
    │   └→ 片段适配器 → fragment_adapter.bin
    └→ [标准训练]:
        └→ 完整模型 → pytorch_model.bin
    │
恢复训练 (若存在检查点)
    ├→ HF自动加载LoRA权重
    └→ 手动加载非LoRA权重
    │
最终保存
    ├→ [lora_enable]:
    │   ├→ adapter_model.bin
    │   └→ non_lora_trainables.bin
    └→ [标准]:
        └→ pytorch_model.bin
```

---

## 九、关键代码引用

### DeepSpeed ZeRO-3 参数处理
- **第177-188行**: `maybe_zero_3()` - 从ZeRO-3分区中聚合参数到CPU
- **第197-200行**: `get_frag_adapter_state_maybe_zero_3()` - 片段适配器状态提取
- **第203-225行**: `get_peft_state_maybe_zero_3()` - LoRA状态提取
- **第227-232行**: `get_peft_state_non_lora_maybe_zero_3()` - 非LoRA状态提取

### 参数初始化与冻结
- **第631-658行**: ESM编码器初始化、适配器加载、参数冻结配置
- **第417-453行**: 自定义优化器创建，参数分组策略

### 检查点管理
- **第455-494行**: `_save_checkpoint()` - 训练中的检查点保存
- **第664-688行**: 恢复训练时的权重加载逻辑

---

## 十、实际应用场景

### 场景1: LoRA微调 + 适配器学习
```bash
python train_llama_lfj.py \
  --lora_enable True \
  --lora_r 32 \
  --lora_alpha 64 \
  --learning_rate 2e-4 \
  # → 保存: adapter_model.bin + non_lora_trainables.bin
```

**更新参数**:
- ✅ LLaMA注意力/MLP的LoRA权重
- ✅ 模态适配器 (fc1, fc2)
- ✅ 片段适配器 (Perceiver层)
- ❌ ESM2编码器 (冻结)
- ❌ LLaMA主体 (冻结)

### 场景2: 仅训练片段适配器
```bash
python train_llama_lfj.py \
  --tune_fragment_adapter True \
  # → 保存: fragment_adapter.bin
```

**更新参数**:
- ✅ 片段适配器所有参数
- ❌ 其他所有参数 (冻结)

### 场景3: 标准全参数微调
```bash
python train_llama_lfj.py \
  --lora_enable False \
  # → 保存: pytorch_model.bin (完整模型)
```

**更新参数**:
- ✅ 除ESM2外的所有参数 (若freeze_backbone=False)
- ✅ LLaMA主体、适配器、片段适配器

