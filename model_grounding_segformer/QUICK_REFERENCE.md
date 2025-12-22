# Collator 函数 - 快速参考

## 核心概念

### ProteinSAMDataset 返回的数据
```python
{
    "uid": str,
    "sequence": str,
    "category": str,
    "start_position": int,      # 功能区域的起始位置
    "end_position": int,        # 功能区域的结束位置
    "point_position": int | None,  # 位置提示（None表示多区域）
    "is_multi_region": bool,
    "same_category_fragments": [...],  # 多区域时的所有片段
    ...
}
```

### ProteinSAMCollator 返回的数据

#### 核心返回格式
```python
{
    "protein_input_ids": Tensor(B, seq_len+2),         # ESM输出的令牌ID
    "protein_attention_mask": Tensor(B, seq_len+2),    # 注意力掩码
    
    # ★ 关键：训练标签 ★
    "residue_labels": Tensor(B, seq_len),              # 二分类标签 (0/1)
    
    # 位置提示信息
    "point_positions": Tensor(B),                      # 位置索引
    "point_mask": Tensor(B, dtype=bool),               # 有效位置标记
    
    # 元数据
    "is_multi_region": List[bool](B),
    "categories": List[str](B),
    "sequences": List[str](B),
    
    # 可选（仅use_category_cache=False时）
    "text_input_ids": Tensor(B, text_len),
    "text_attention_mask": Tensor(B, text_len)
}
```

## 关键字段详解

### residue_labels（★最重要★）
- **形状**: (batch_size, max_seq_len)  其中 max_seq_len = 1024
- **值**: 0 (背景) 或 1 (功能区域)
- **用途**: 训练时用于计算损失函数
- **处理逻辑**:
  - 单区域任务: residue_labels[i, start:end+1] = 1
  - 多区域任务: 对每个片段标记为1

### point_positions 和 point_mask
- **point_positions**: 长度为(batch_size,)的张量，存储位置提示
- **point_mask**: 长度为(batch_size,)的bool张量
  - `True`: 该样本有有效的位置提示（单区域任务）
  - `False`: 该样本无有效位置提示（多区域任务）

### protein_input_ids 和 protein_attention_mask
- **长度**: seq_len + 2（包含BOS和EOS令牌）
- **注意**: collator移除BOS/EOS后计算residue_labels，所以residue_labels长度 = seq_len

## 在模型中的使用

### train.py 中的使用
```python
# 从collator获取批次
batch = next(iter(train_loader))

# 处理位置提示
point_mask = batch["point_mask"]
point_positions = batch["point_positions"] if torch.any(point_mask) else None

# 模型前向传递
outputs = model(
    protein_input_ids=batch["protein_input_ids"],
    protein_attention_mask=batch["protein_attention_mask"],
    categories=batch["categories"],
    point_positions=point_positions,
    residue_labels=batch["residue_labels"]  # ← 关键标签
)

# 获取损失
loss = outputs["loss"]
```

### protein_sam.py 中的处理
```python
# 在forward方法中
if residue_labels is not None:
    # 计算Dice Loss
    dice_loss = self._compute_dice_loss(mask_logits, residue_labels, attention_mask)
    
    # 计算Cross-Entropy Loss
    ce_loss = self._compute_ce_loss(mask_logits, residue_labels, attention_mask)
    
    # 总损失
    total_loss = 1.0 * dice_loss + 0.5 * ce_loss
```

## 初始化示例

### 最简单的方式
```python
from model_grounding_segformer.dataset import get_datasets_and_collator

datasets, collator = get_datasets_and_collator(
    root_dir="./data",
    data_name="VenusX_Dom||VenusX_Act",
    esm_model_path="/path/to/esm2_model",
    llama_model_path="/path/to/llama_model",
    max_sequence_length=1021,
    use_category_cache=True
)

train_dataset = datasets["train"]
valid_dataset = datasets["valid"]

from torch.utils.data import DataLoader
train_loader = DataLoader(
    train_dataset,
    batch_size=8,
    shuffle=True,
    collate_fn=collator,
    num_workers=4,
    pin_memory=True
)
```

## 关键对比

### 与主项目(dataloader_grounding.py)的区别

| 特性 | segformer | 主项目 |
|------|----------|--------|
| 返回标签格式 | residue_labels (二分类) | position_grd (位置列表) |
| 用途 | 分割任务 | 文本生成/指代任务 |
| 单/多区域 | 都支持 | 仅指代 |
| 标签值 | 0/1 | 列表嵌套 |

## 常见问题

### Q: residue_labels的长度为什么是1024？
A: protein_input_ids长度为1026（包括BOS和EOS），collator移除这2个令牌后得到1024。

### Q: point_mask都是False说明什么？
A: 整个批次都是多区域任务。此时point_positions实际不被使用。

### Q: 如何区分单/多区域任务？
A: 查看`is_multi_region`列表：
   - `True`: 多区域任务，标记同类别的多个片段
   - `False`: 单区域任务，有point_position提示

### Q: 如何自定义数据集配置？
A: 传递额外参数给`get_datasets_and_collator()`：
   ```python
   datasets, collator = get_datasets_and_collator(
       ...,
       null_position_prob=0.5,      # 多区域任务的概率
       position_noise_std=10,        # 位置噪声标准差
       filter_long_sequences=True
   )
   ```

## 数据流总结

```
JSON数据
  ↓
Dataset.__getitem__()  → 单个样本dict
  ↓
DataLoader batch      → 样本列表
  ↓
Collator.__call__()   → 张量批次 ★关键步骤★
  ↓
Model.forward()       → 损失和预测
```

## 文件位置

- 实现: `/home/lfj/projects_dir/FragLLM/model_grounding_segformer/dataset.py`
- 训练脚本: `/home/lfj/projects_dir/FragLLM/model_grounding_segformer/train.py`
- 模型: `/home/lfj/projects_dir/FragLLM/model_grounding_segformer/protein_sam.py`

## 详细文档

- 完整分析: `COLLATOR_ANALYSIS.md`
- 代码示例: `COLLATOR_CODE_EXAMPLES.md`
- 本快速参考: `QUICK_REFERENCE.md`
