# Model GroundingSegformer - Dataset 和 Collator 函数详细分析

## 项目结构

`model_grounding_segformer` 是一个用于蛋白质功能区域接地(grounding)的分割模型项目。

### 核心文件位置
- `/home/lfj/projects_dir/FragLLM/model_grounding_segformer/dataset.py` - 数据集和 collator 实现
- `/home/lfj/projects_dir/FragLLM/model_grounding_segformer/train.py` - 训练脚本
- `/home/lfj/projects_dir/FragLLM/model_grounding_segformer/protein_sam.py` - 模型主体

---

## 1. ProteinSAMDataset 类

### 功能：
加载和处理蛋白质序列及其功能区域的数据集。支持单个或多个数据集的并联加载。

### 初始化参数：

```python
def __init__(
    self,
    root_dir: str,                      # 数据根目录
    data_name: str,                     # 数据集名称，支持多个用"||"分隔
    split: str,                         # 数据集划分: "train", "valid", "test"
    max_sequence_length: int = 1021,    # 最大蛋白质序列长度
    null_position_prob: float = 0.3,    # 位置提示为空的概率
    random_position_prob: float = 0.2,  # 随机位置的概率
    position_noise_std: float = 10.0,   # 位置噪声的标准差
    filter_long_sequences: bool = True  # 是否过滤超长序列
)
```

### 数据流处理：

1. **数据加载**（`_load_single_dataset_data`）
   - 从 JSON 文件读取原始数据
   - 格式：`{root_dir}/{data_name}/{split}.json`
   - 对每个蛋白质的每个功能片段生成一个数据样本

2. **序列截断**（`_process_sequence`）
   - 如果序列长度超过 `max_sequence_length`，进行随机截断
   - 保证截断后功能片段仍完整
   - 返回：(处理后的序列, 调整的起始位置, 调整的结束位置)

3. **位置处理**（`_get_point_position`）
   - 生成位置提示(point prompt)用于模型输入
   - 训练时：
     - 30% 概率返回 None（多区域检测任务）
     - 其他情况返回片段中心加高斯噪声

### __getitem__ 返回格式：

#### 情况 1：单区域任务（is_multi_region=False）
```python
{
    "uid": str,                    # 样本唯一标识
    "sequence": str,               # 处理后的蛋白质序列
    "category": str,               # 功能区域类别
    "start_position": int,         # 起始位置
    "end_position": int,           # 结束位置
    "point_position": int,         # 位置提示（可能为None）
    "is_multi_region": False,
    "description": str,            # 功能区域描述
    "original_start": int,         # 原始起始位置
    "original_end": int            # 原始结束位置
}
```

#### 情况 2：多区域任务（is_multi_region=True）
```python
{
    "uid": str,
    "sequence": str,
    "category": str,
    "point_position": None,        # 多区域任务中为None
    "is_multi_region": True,
    "same_category_fragments": [   # 所有相同类别的片段
        {
            "category": str,
            "start_position": int,
            "end_position": int,
            "description": str
        },
        ...
    ],
    "description": str,
    "original_start": int,
    "original_end": int
}
```

---

## 2. ProteinSAMCollator 类

### 功能：
将数据集批次转换为模型可以处理的张量形式。处理蛋白质序列、文本提示和标签的对齐。

### 初始化参数：

```python
def __init__(
    self,
    esm_tokenizer: EsmTokenizer,        # ESM2 分词器用于蛋白质序列
    llama_tokenizer: Optional[LlamaTokenizer] = None,  # LLaMA 分词器用于文本
    max_protein_length: int = 1024,     # 最大蛋白质序列长度
    max_text_length: int = 128,         # 最大文本长度
    use_category_cache: bool = True     # 是否使用预计算的类别嵌入缓存
)
```

### __call__ 方法 - 数据整合流程

#### 输入：
```python
batch: List[Dict[str, Any]]  # 来自Dataset的样本列表
```

#### 处理步骤：

1. **提取基本信息**
   ```python
   sequences = [item["sequence"] for item in batch]
   categories = [item["category"] for item in batch]
   point_positions = [item["point_position"] for item in batch]
   is_multi_region = [item.get("is_multi_region", False) for item in batch]
   ```

2. **蛋白质序列分词**
   ```python
   protein_tokenized = self.esm_tokenizer(
       sequences,
       padding=True,
       truncation=True,
       max_length=self.max_protein_length+2,  # +2 for BOS/EOS
       return_tensors="pt"
   )
   # 返回：
   # - input_ids: (batch_size, seq_len+2)
   # - attention_mask: (batch_size, seq_len+2)
   ```

3. **文本分词**（仅当 use_category_cache=False 时）
   ```python
   text_tokenized = self.llama_tokenizer(
       categories,
       padding=True,
       truncation=True,
       max_length=self.max_text_length,
       return_tensors="pt"
   )
   # 返回：
   # - text_input_ids: (batch_size, text_len)
   # - text_attention_mask: (batch_size, text_len)
   ```

4. **构建残基级标签**（关键！）
   
   创建 `residue_labels` 张量用于训练：
   
   ```python
   batch_size = len(batch)
   max_seq_len = protein_tokenized["input_ids"].shape[1] - 2  # 移除BOS/EOS
   
   residue_labels = torch.zeros(batch_size, max_seq_len, dtype=torch.long)
   
   for i, item in enumerate(batch):
       if item.get("is_multi_region", False):
           # 多区域任务：标记所有相同类别的片段
           same_category_fragments = item.get("same_category_fragments", [])
           for frag in same_category_fragments:
               start_pos = frag["start_position"]
               end_pos = frag["end_position"]
               start_pos = max(0, min(start_pos, max_seq_len - 1))
               end_pos = max(0, min(end_pos, max_seq_len - 1))
               residue_labels[i, start_pos:end_pos+1] = 1  # 标记为1
       else:
           # 单区域任务：标记单个功能区域
           start_pos = item["start_position"]
           end_pos = item["end_position"]
           start_pos = max(0, min(start_pos, max_seq_len - 1))
           end_pos = max(0, min(end_pos, max_seq_len - 1))
           residue_labels[i, start_pos:end_pos+1] = 1
   ```
   
   **标签格式**：
   - `0` = 背景(非功能区域)
   - `1` = 功能区域(foreground)
   - 形状：`(batch_size, max_seq_len)` 其中 max_seq_len 不包括 BOS/EOS 令牌

5. **处理位置提示**
   
   ```python
   point_tensor = torch.zeros(len(batch), dtype=torch.long)
   point_mask = torch.zeros(len(batch), dtype=torch.bool)
   
   for i, point_pos in enumerate(point_positions):
       if point_pos is not None:
           point_tensor[i] = max(0, point_pos)
           point_mask[i] = True  # 有效的位置提示
       # else: 多区域任务中 point_mask[i] 为 False
   ```

#### 返回字典：

```python
{
    # 蛋白质序列相关
    "protein_input_ids": torch.Tensor,           # (batch_size, seq_len+2)
    "protein_attention_mask": torch.Tensor,      # (batch_size, seq_len+2)
    
    # 位置提示相关
    "point_positions": torch.Tensor,             # (batch_size,) 位置索引
    "point_mask": torch.Tensor,                  # (batch_size,) bool类型，标记有效位置
    
    # 训练标签
    "residue_labels": torch.Tensor,              # (batch_size, max_seq_len) 残基级标签
                                                  # 值：0(背景) 或 1(功能区域)
    
    # 元数据
    "is_multi_region": List[bool],               # 每个样本是否为多区域任务
    "categories": List[str],                     # 类别名称列表
    "sequences": List[str],                      # 原始序列列表
    
    # 可选：文本分词结果（仅当 use_category_cache=False 时）
    "text_input_ids": torch.Tensor,              # (batch_size, text_len)
    "text_attention_mask": torch.Tensor          # (batch_size, text_len)
}
```

---

## 3. 关键对比：残基标签 vs 位置标签

### residue_labels（残基级标签）- 新方法

| 特性 | 描述 |
|------|------|
| **形状** | (batch_size, max_seq_len) |
| **值** | 0 = 背景，1 = 功能区域 |
| **用途** | 二分类分割任务 |
| **处理方式** | 对于每个片段 [start, end]，将 residue_labels[i, start:end+1] 设为 1 |
| **多区域支持** | 可以标记同一样本中的多个片段 |

### start_labels / end_labels（原始方法）

主项目中的 `dataloader_grounding.py` 返回 `position_grd`，格式为：
```python
position_grd = [[[start1, end1+1], [start2, end2+1], ...]]
# 这是一个嵌套列表，可以包含多个片段的位置
```

这种格式用于指代式和文本生成任务，不是用于分割模型。

---

## 4. 模型中的数据使用

### protein_sam.py 中的 forward 方法

```python
def forward(
    self,
    protein_input_ids: torch.Tensor,           # (batch_size, seq_len)
    protein_attention_mask: torch.Tensor,      # (batch_size, seq_len)
    text_input_ids: Optional[torch.Tensor] = None,
    text_attention_mask: Optional[torch.Tensor] = None,
    categories: Optional[list] = None,
    point_positions: Optional[torch.Tensor] = None,  # (batch_size,)
    residue_labels: Optional[torch.Tensor] = None   # (batch_size, seq_len) 标签
) -> Dict[str, torch.Tensor]:
```

**标签使用**：
```python
if residue_labels is not None:
    # 使用两种损失函数
    dice_loss = self._compute_dice_loss(mask_logits, residue_labels, 
                                       protein_attention_mask)
    ce_loss = self._compute_ce_loss(mask_logits, residue_labels, 
                                   protein_attention_mask)
    
    # 组合损失
    total_loss = 1.0 * dice_loss + 0.5 * ce_loss
```

**损失计算**：
- Dice Loss：用于处理不平衡的前景/背景
- Cross-Entropy Loss：用于逐像素分类

---

## 5. 训练流程中的数据流

```
原始JSON数据
    ↓
ProteinSAMDataset.__getitem__()
    ↓
Dataset样本 (dict with sequence, positions, category, etc.)
    ↓
DataLoader batch (List[dict])
    ↓
ProteinSAMCollator.__call__()
    ↓
整合的批次字典
    {
        "protein_input_ids": ...,
        "protein_attention_mask": ...,
        "point_positions": ...,
        "point_mask": ...,
        "residue_labels": ...,  ← 用于训练的标签
        ...
    }
    ↓
Model.forward()
    ↓
Loss 和 预测
```

---

## 6. 重要细节

### 位置坐标转换

1. **Dataset 中的位置** = 相对于处理后序列
2. **Collator 中的 residue_labels** = 基于相对位置
3. **序列长度处理** = 移除 BOS/EOS 令牌
   - ESM 输出：[BOS, token1, token2, ..., EOS]
   - Collator 移除：[token1, token2, ...]
   - residue_labels 长度 = seq_len - 2

### 多区域任务

当 `point_position = None` 时：
- `is_multi_region = True`
- `point_mask[i] = False`
- `residue_labels[i]` 包含同类别的所有片段标记
- 模型进行多目标检测而非单目标+位置提示

### 无效位置处理

```python
# 在Dataset中
point_mask[i] = False  if point_position is None
point_mask[i] = True   if point_position is not None

# 在train.py中
point_positions = batch["point_positions"] if torch.any(point_mask) else None
# 只有当至少有一个有效位置时才传递
```

---

## 7. 数据集配置示例

从 `train.py` 的默认参数：

```python
--data_name "VenusX_Dom||VenusX_Act||VenusX_BindI||VenusX_Motif||VenusX_Evo"
--max_sequence_length 1021
--null_position_prob 0.3
--random_position_prob 0.0
--position_noise_std 20
```

支持的数据集：
- VenusX_Dom = Domain fragments
- VenusX_Act = Activity fragments
- VenusX_BindI = Binding interface fragments
- VenusX_Motif = Motif fragments
- VenusX_Evo = Evolution fragments

---

## 总结

**Collator 不返回 start_labels 和 end_labels**，而是返回统一的 **residue_labels**：
- 这是一个二分类标签张量 (batch_size, seq_len)
- 值为 0 或 1
- 支持单区域和多区域检测任务
- 与蛋白质分割任务相比，这是一个更灵活的设计

