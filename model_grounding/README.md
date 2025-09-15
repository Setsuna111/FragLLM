# ProteinSAM: Segment Anything Model for Protein Functional Region Grounding

ProteinSAM是一个类似SAM(Segment Anything Model)的蛋白质功能片段定位预训练模型，能够根据功能区名称和可选的位置提示，准确预测蛋白质序列中功能区域的起始和终止位置。

## 模型架构

### 整体架构
- **Protein Encoder**: 基于ESM2的蛋白质序列编码器（冻结参数）
- **Prompt Encoder**: 文本+位置提示编码器（Llama文本编码器冻结，投影层可训练）
- **Position Decoder**: 轻量化位置解码器（完全可训练）

### 关键特性
1. **多模态输入**: 支持文本提示（功能区名称）和位置提示（point prompt）
2. **参数效率**: 只有投影层和解码器参数可训练，大大减少训练成本
3. **特殊位置编码**: 文本token的位置编码会被point prompt位置替换，增强空间感知能力
4. **灵活的位置提示**: 训练时随机置空或随机化位置提示，提高鲁棒性
5. **轻量化设计**: 解码器采用简洁的cross-attention架构

## 文件结构

```
model_grounding/
├── __init__.py                 # 模块初始化
├── protein_encoder.py          # 蛋白质编码器（ESM2）
├── prompt_encoder.py           # 提示编码器（文本+位置）
├── position_decoder.py         # 位置解码器
├── protein_sam.py              # 完整的ProteinSAM模型
├── dataset.py                  # 数据集和数据加载器
├── train.py                    # 训练脚本
├── inference_example.py        # 推理示例
└── README.md                   # 本文件
```

## 安装依赖

```bash
pip install torch transformers numpy tqdm
```

## 使用方法

### 1. 预处理类别编码（推荐）

为了避免训练过程中反复调用Llama模型，建议先预计算所有类别的文本编码：

```bash
cd /home/lfj/projects_dir/FragLLM/model_grounding

# 预处理所有数据集的类别编码
python preprocess_categories.py \
    --data_root ./data \
    --data_name "VenusX_Dom||VenusX_Act||VenusX_BindI||VenusX_Motif||VenusX_Evo" \
    --llama_model_path /home/lfj/projects_dir/pretrained_model/Llama-3.1-8B-Instruct \
    --output_path ./category_embeddings.pt \
    --batch_size 8
```

### 2. 训练模型

```bash
# 单数据集训练
python train.py \
    --data_name VenusX_Dom \
    --batch_size 8 \
    --learning_rate 1e-4 \
    --num_epochs 10 \
    --output_dir ./checkpoints_grounding/domain_model \
    --use_category_cache \
    --category_embeddings_path ./category_embeddings.pt

# 全部五个数据集联合训练
python train.py \
    --data_name "VenusX_Dom||VenusX_Act||VenusX_BindI||VenusX_Motif||VenusX_Evo" \
    --batch_size 8 \
    --learning_rate 1e-4 \
    --num_epochs 10 \
    --output_dir ./checkpoints_grounding/all_datasets_model \
    --use_category_cache \
    --category_embeddings_path ./category_embeddings.pt

```

### 3. 模型推理

```bash
# 使用训练好的模型进行推理
python inference_example.py \
    --checkpoint_path ./checkpoints_grounding/domain_model/best_model.pt \
    --output_file predictions.json
```

### 4. 代码使用示例

```python
from model_grounding import ProteinSAM
import torch

# 加载预训练模型（使用类别缓存）
model = ProteinSAM(
    esm_model_path="/home/lfj/projects_dir/pretrained_model/esm2_t30_150M_UR50D",
    llama_model_path="/home/lfj/projects_dir/pretrained_model/Llama-3.1-8B-Instruct",
    use_category_cache=True,
    category_embeddings_path="./category_embeddings.pt"
)

# 加载训练权重
model.load_model("./checkpoints_grounding/best_model.pt")
model.eval()

# 单个预测
protein_sequence = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGT..."
category = "EF-hand domain"
point_position = 50  # 可选的位置提示

results = model.batch_predict(
    protein_sequences=[protein_sequence],
    categories=[category], 
    point_positions=[point_position]
)

print(f"Predicted region: {results[0]['start_position']}-{results[0]['end_position']}")
```

## 训练数据格式

模型使用VenusX数据集，支持以下数据类型：
- `VenusX_Dom`: 蛋白质域片段
- `VenusX_Act`: 活性位点片段
- `VenusX_BindI`: 结合位点片段  
- `VenusX_Motif`: 基序片段
- `VenusX_Evo`: 进化相关片段

数据格式：
```json
[
  {
    "uid": "protein_id",
    "sequence": "MKTAYIAK...",
    "fragments": [
      {
        "category": "FAD dependent oxidoreductase",
        "description": "...",
        "frags": [
          {
            "start_position": 10,
            "end_position": 150,
            "sequence": "TAYIAK..."
          }
        ]
      }
    ]
  }
]
```

## 训练参数说明

### 核心参数
- `--batch_size`: 批次大小（默认8）
- `--learning_rate`: 学习率（默认1e-4）
- `--num_epochs`: 训练轮数（默认10）
- `--max_sequence_length`: 最大蛋白质序列长度（默认1021）

### 数据增强参数
- `--null_position_prob`: 位置提示置空概率（默认0.3）
- `--random_position_prob`: 随机位置提示概率（默认0.2）
- `--position_noise_std`: 位置噪声标准差（默认10.0）

### 模型架构参数
- `--decoder_num_heads`: 解码器注意力头数（默认8）
- `--decoder_num_layers`: 解码器层数（默认2）
- `--decoder_intermediate_size`: 解码器中间层大小（默认512）

## 性能指标

模型训练过程中会计算以下指标：
- **Start Accuracy**: 起始位置精确匹配准确率
- **End Accuracy**: 终止位置精确匹配准确率  
- **IoU Accuracy**: IoU > 0.5的区域重叠准确率