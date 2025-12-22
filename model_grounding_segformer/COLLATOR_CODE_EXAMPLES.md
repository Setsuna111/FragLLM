# Collator 函数实现细节 - 代码示例

## 完整的 Collator 实现 (from dataset.py)

### 类定义和初始化

```python
class ProteinSAMCollator:
    """
    Data collator for ProteinSAM dataset.
    Handles tokenization and batch preparation.
    """
    
    def __init__(
        self,
        esm_tokenizer: EsmTokenizer,
        llama_tokenizer: Optional[LlamaTokenizer] = None,
        max_protein_length: int = 1024,
        max_text_length: int = 128,
        use_category_cache: bool = True
    ):
        self.esm_tokenizer = esm_tokenizer
        self.llama_tokenizer = llama_tokenizer
        self.max_protein_length = max_protein_length
        self.max_text_length = max_text_length
        self.use_category_cache = use_category_cache
        
        # Set pad token for Llama if not exists and if tokenizer is provided
        if self.llama_tokenizer is not None and self.llama_tokenizer.pad_token is None:
            self.llama_tokenizer.pad_token = '<|reserved_special_token_0|>'
```

### __call__ 方法 - 核心逻辑

```python
def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """
    Collate a batch of samples.
    
    Args:
        batch: List of data samples
        
    Returns:
        Batched tensors ready for model input
    """
    # ============ 第1步：提取基本信息 ============
    sequences = [item["sequence"] for item in batch]
    categories = [item["category"] for item in batch]
    point_positions = [item["point_position"] for item in batch]
    is_multi_region = [item.get("is_multi_region", False) for item in batch]
    
    # ============ 第2步：蛋白质序列分词 ============
    protein_tokenized = self.esm_tokenizer(
        sequences,
        padding=True,
        truncation=True,
        max_length=self.max_protein_length+2,  # +2 for BOS/EOS
        return_tensors="pt"
    )
    
    # ============ 第3步：文本分词（可选） ============
    text_input_ids = None
    text_attention_mask = None
    
    if not self.use_category_cache and self.llama_tokenizer is not None:
        text_tokenized = self.llama_tokenizer(
            categories,
            padding=True,
            truncation=True,
            max_length=self.max_text_length,
            return_tensors="pt"
        )
        text_input_ids = text_tokenized["input_ids"]
        text_attention_mask = text_tokenized["attention_mask"]
    
    # ============ 第4步：构建残基级标签（关键步骤）============
    batch_size = len(batch)
    max_seq_len = protein_tokenized["input_ids"].shape[1] - 2  # Remove BOS/EOS tokens
    
    # 初始化标签张量：全部为0（背景）
    residue_labels = torch.zeros(batch_size, max_seq_len, dtype=torch.long)
    
    # 逐样本处理
    for i, item in enumerate(batch):
        if item.get("is_multi_region", False):
            # 多区域任务：标记所有相同类别的片段
            same_category_fragments = item.get("same_category_fragments", [])
            for frag in same_category_fragments:
                start_pos = frag["start_position"]
                end_pos = frag["end_position"]
                # 边界检查和约束
                start_pos = max(0, min(start_pos, max_seq_len - 1))
                end_pos = max(0, min(end_pos, max_seq_len - 1))
                # 标记为1（功能区域）
                residue_labels[i, start_pos:end_pos+1] = 1
        else:
            # 单区域任务：标记单个功能区域
            start_pos = item["start_position"]
            end_pos = item["end_position"]
            # 边界检查和约束
            start_pos = max(0, min(start_pos, max_seq_len - 1))
            end_pos = max(0, min(end_pos, max_seq_len - 1))
            # 标记为1（功能区域）
            residue_labels[i, start_pos:end_pos+1] = 1
    
    # ============ 第5步：处理位置提示 ============
    point_tensor = torch.zeros(len(batch), dtype=torch.long)
    point_mask = torch.zeros(len(batch), dtype=torch.bool)
    
    for i, point_pos in enumerate(point_positions):
        if point_pos is not None:
            # 调整位置并设置掩码
            point_tensor[i] = max(0, point_pos)
            point_mask[i] = True
    
    # ============ 第6步：组织返回字典 ============
    batch_dict = {
        "protein_input_ids": protein_tokenized["input_ids"],
        "protein_attention_mask": protein_tokenized["attention_mask"],
        "point_positions": point_tensor,
        "point_mask": point_mask,
        "residue_labels": residue_labels,  # 统一的残基级标签
        "is_multi_region": is_multi_region,
        "categories": categories,
        "sequences": sequences
    }
    
    # 添加文本令牌（如果可用）
    if text_input_ids is not None and text_attention_mask is not None:
        batch_dict.update({
            "text_input_ids": text_input_ids,
            "text_attention_mask": text_attention_mask
        })
    
    return batch_dict
```

---

## 实际数据流示例

### 单个样本示例

#### 输入（来自 Dataset.__getitem__）

```python
# 单区域任务示例
sample_single = {
    "uid": "PROTEIN_001",
    "sequence": "MKLVVT...",      # 长度 = 500
    "category": "Domain",
    "start_position": 100,
    "end_position": 150,
    "point_position": 125,          # 片段中心 + 噪声
    "is_multi_region": False,
    "description": "Catalytic domain",
    "original_start": 100,
    "original_end": 150
}

# 多区域任务示例
sample_multi = {
    "uid": "PROTEIN_002",
    "sequence": "MKLVVT...",      # 长度 = 600
    "category": "Motif",
    "point_position": None,         # 多区域时为None
    "is_multi_region": True,
    "same_category_fragments": [
        {
            "category": "Motif",
            "start_position": 50,
            "end_position": 80,
            "description": "Motif 1"
        },
        {
            "category": "Motif",
            "start_position": 200,
            "end_position": 230,
            "description": "Motif 2"
        }
    ],
    "description": "Zinc finger motif",
    "original_start": 50,
    "original_end": 80
}
```

#### 批次处理（batch_size = 2）

```python
batch = [sample_single, sample_multi]

# ════════════════════════════════════════════════════════════
# Collator 处理流程
# ════════════════════════════════════════════════════════════

# 第1步：提取信息
sequences = [
    "MKLVVT...",           # 长度 500
    "MKLVVT..."            # 长度 600
]
categories = ["Domain", "Motif"]
point_positions = [125, None]
is_multi_region = [False, True]

# 第2步：ESM 分词
protein_tokenized = {
    "input_ids": torch.Tensor([
        [0, ..., 500个token..., 2],       # [BOS, ..., EOS]
        [0, ..., 600个token..., 2]        # [BOS, ..., EOS]
    ]),  # 形状: (2, 1026) 其中 1026 = max_len=1024 + 2
    "attention_mask": torch.Tensor([
        [1, 1, ..., 1],                   # 502个1
        [1, 1, ..., 1]                    # 602个1
    ])  # 形状: (2, 1026)
}

# 第3步：计算最大序列长度（移除BOS/EOS）
max_seq_len = 1026 - 2 = 1024

# 第4步：构建残基标签
residue_labels = torch.zeros(2, 1024, dtype=torch.long)

# 处理样本1（单区域）
start_pos = 100
end_pos = 150
residue_labels[0, 100:151] = 1  # 标记51个残基

# 处理样本2（多区域）
# 片段1: [50, 80]
residue_labels[1, 50:81] = 1    # 标记31个残基
# 片段2: [200, 230]
residue_labels[1, 200:231] = 1  # 标记31个残基

# 结果形状: (2, 1024)
# residue_labels[0] 中有 [100:151] = 1，其余 = 0
# residue_labels[1] 中有 [50:81] = 1 和 [200:231] = 1，其余 = 0

# 第5步：处理位置提示
point_tensor = torch.tensor([125, 0], dtype=torch.long)
point_mask = torch.tensor([True, False], dtype=torch.bool)
# point_tensor[0] = 125（有效）
# point_tensor[1] = 0（无效）
# point_mask[1] = False（表示多区域任务）
```

#### 返回的批次字典

```python
batch_dict = {
    # 蛋白质信息
    "protein_input_ids": torch.Tensor,
        # 形状: (2, 1026)
        # 值: [[0, token, token, ..., 2], [0, token, token, ..., 2]]
    
    "protein_attention_mask": torch.Tensor,
        # 形状: (2, 1026)
        # 值: [[1, 1, ..., 1], [1, 1, ..., 1]]
    
    # 位置提示
    "point_positions": torch.tensor([125, 0], dtype=torch.long),
        # 形状: (2,)
        # 样本1的位置提示 = 125
        # 样本2的位置提示 = 0（实际未使用，因为point_mask[1]=False）
    
    "point_mask": torch.tensor([True, False], dtype=torch.bool),
        # 形状: (2,)
        # 样本1有有效位置提示
        # 样本2无有效位置提示（多区域任务）
    
    # 训练标签 ← 关键！
    "residue_labels": torch.Tensor,
        # 形状: (2, 1024)
        # [0, :] = [0,0,...,0,1,1,1,...,1(151个),0,...,0]  # 样本1
        # [1, :] = [0,...,1,1,...,1(31个),...,1,1,...,1(31个),...,0]  # 样本2
    
    # 元数据
    "is_multi_region": [False, True],
    "categories": ["Domain", "Motif"],
    "sequences": ["MKLVVT...", "MKLVVT..."],
    
    # 文本信息（仅当 use_category_cache=False）
    # "text_input_ids": torch.Tensor,        # (2, 128)
    # "text_attention_mask": torch.Tensor    # (2, 128)
}
```

---

## 在训练中的使用

### train.py 中的前向传递

```python
# 在训练循环中
for batch_idx, batch in enumerate(progress_bar):
    # 将张量移到设备
    for key in batch:
        if isinstance(batch[key], torch.Tensor):
            batch[key] = batch[key].to(device)
    
    # 处理位置提示（仅当有效时传递）
    point_mask = batch["point_mask"]
    point_positions = batch["point_positions"] if torch.any(point_mask) else None
    
    # 前向传递
    outputs = model(
        protein_input_ids=batch["protein_input_ids"],
        protein_attention_mask=batch["protein_attention_mask"],
        text_input_ids=batch.get("text_input_ids"),
        text_attention_mask=batch.get("text_attention_mask"),
        categories=batch["categories"],
        point_positions=point_positions,
        residue_labels=batch["residue_labels"]  # ← 关键标签
    )
    
    # 获取损失
    loss = outputs["loss"]
    dice_loss = outputs["dice_loss"]
    ce_loss = outputs["ce_loss"]
```

### 模型中的标签处理

```python
# 在 protein_sam.py 的 forward 方法中

if residue_labels is not None:
    # residue_labels 形状: (batch_size, seq_len)
    # 值: 0 = 背景，1 = 功能区域
    
    # 计算 Dice Loss
    dice_loss = self._compute_dice_loss(
        mask_logits,           # (batch_size, seq_len, 2)
        residue_labels,        # (batch_size, seq_len)
        protein_attention_mask # (batch_size, seq_len)
    )
    
    # 计算 Cross-Entropy Loss
    ce_loss = self._compute_ce_loss(
        mask_logits,           # (batch_size, seq_len, 2)
        residue_labels,        # (batch_size, seq_len)
        protein_attention_mask # (batch_size, seq_len)
    )
    
    # 组合损失
    total_loss = 1.0 * dice_loss + 0.5 * ce_loss
    
    outputs.update({
        "loss": total_loss,
        "dice_loss": dice_loss,
        "ce_loss": ce_loss,
        "residue_labels": residue_labels
    })
```

---

## 完整的使用示例

```python
from torch.utils.data import DataLoader
from transformers import EsmTokenizer
from model_grounding_segformer.dataset import (
    ProteinSAMDataset, 
    ProteinSAMCollator,
    get_datasets_and_collator
)

# ════════════════════════════════════════════════════════════
# 方法1：手动初始化（更灵活）
# ════════════════════════════════════════════════════════════

# 初始化数据集
dataset = ProteinSAMDataset(
    root_dir="./data",
    data_name="VenusX_Dom||VenusX_Act",
    split="train",
    max_sequence_length=1021,
    null_position_prob=0.3,
    position_noise_std=20
)

# 初始化 collator
esm_tokenizer = EsmTokenizer.from_pretrained(
    "/path/to/esm2_t36_3B_UR50D"
)

collator = ProteinSAMCollator(
    esm_tokenizer=esm_tokenizer,
    max_protein_length=1024,
    use_category_cache=True
)

# 创建数据加载器
dataloader = DataLoader(
    dataset,
    batch_size=8,
    shuffle=True,
    collate_fn=collator,
    num_workers=4,
    pin_memory=True
)

# 迭代批次
for batch in dataloader:
    print(f"Batch keys: {batch.keys()}")
    print(f"Protein input IDs shape: {batch['protein_input_ids'].shape}")
    print(f"Residue labels shape: {batch['residue_labels'].shape}")
    print(f"Point positions shape: {batch['point_positions'].shape}")
    print(f"Is multi-region: {batch['is_multi_region']}")
    break  # 仅显示第一个批次

# ════════════════════════════════════════════════════════════
# 方法2：使用辅助函数（推荐）
# ════════════════════════════════════════════════════════════

datasets, collator = get_datasets_and_collator(
    root_dir="./data",
    data_name="VenusX_Dom||VenusX_Act||VenusX_BindI",
    esm_model_path="/path/to/esm2_t36_3B_UR50D",
    llama_model_path="/path/to/Llama-3.1-8B-Instruct",
    max_sequence_length=1021,
    max_text_length=128,
    use_category_cache=True,
    null_position_prob=0.3,
    position_noise_std=20
)

# 获取不同的数据集
train_dataset = datasets["train"]
valid_dataset = datasets["valid"]

# 创建数据加载器
train_loader = DataLoader(
    train_dataset,
    batch_size=8,
    shuffle=True,
    collate_fn=collator,
    num_workers=4,
    pin_memory=True
)

valid_loader = DataLoader(
    valid_dataset,
    batch_size=16,
    shuffle=False,
    collate_fn=collator,
    num_workers=4,
    pin_memory=True
)

# 迭代训练批次
for epoch in range(num_epochs):
    for batch in train_loader:
        # 使用 batch["residue_labels"] 进行训练
        model_output = model(
            protein_input_ids=batch["protein_input_ids"],
            protein_attention_mask=batch["protein_attention_mask"],
            residue_labels=batch["residue_labels"],
            # ...其他参数
        )
```

---

## 关键点总结

### Collator 返回的关键字段

| 字段 | 类型 | 形状 | 说明 |
|------|------|------|------|
| `protein_input_ids` | Tensor | (B, L+2) | ESM 分词后的蛋白质ID |
| `protein_attention_mask` | Tensor | (B, L+2) | 蛋白质的注意力掩码 |
| `residue_labels` | Tensor | (B, L) | 残基级标签（0/1） |
| `point_positions` | Tensor | (B,) | 位置提示索引 |
| `point_mask` | Tensor | (B,) | bool，标记有效位置 |
| `is_multi_region` | List[bool] | (B,) | 每个样本是否多区域 |
| `categories` | List[str] | (B,) | 功能区域类别 |
| `sequences` | List[str] | (B,) | 原始序列 |

### 最重要的变化

**不使用 start_labels 和 end_labels，而使用 residue_labels：**

- 旧方法：分别预测起始和结束位置
- 新方法：进行像素级的二分类分割
- 优势：支持多区域检测，更灵活

