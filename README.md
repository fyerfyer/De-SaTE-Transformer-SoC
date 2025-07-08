# De-SaTE-Transformer-SoC

```bibtex
@inproceedings{desate,
  author={Shinde, Gaurav and Mohapatra, Rohan and Krishan, Pooja and Sengupta, Saptarshi},
  booktitle={2023 IEEE International Conference on Big Data (BigData)}, 
  title={De-SaTE: Denoising Self-attention Transformer Encoders for Li-ion Battery Health Prognostics}, 
  year={2023},
  volume={},
  number={},
  pages={2221-2228},
  doi={10.1109/BigData59044.2023.10386134}}
```

本项目实现了一个专为电池健康监测设计的深度学习模型，采用去噪自注意力Transformer编码器架构。该系统能够通过分析电池的历史放电数据，准确预测其健康状态。

## 项目结构

```
De-SaTE-Transformer-Battery/
├── scripts/                        # 主要执行脚本
│   ├── preprocess.py              # 数据预处理脚本
│   ├── train.py                   # 模型训练脚本
│   ├── evaluate_with_e_value.py   # E值评估脚本
│   └── visualize_results.py       # 结果可视化脚本
├── src/                           # 源代码模块
│   ├── model/                     # 模型实现
│   │   ├── desate_transformer.py  # De-SaTE Transformer主模型
│   │   └── transformer.py         # 基础Transformer模型
│   ├── data_processing/           # 数据处理模块
│   │   ├── base_processor.py      # 基础数据处理器
│   │   └── official_processor.py  # 官方数据处理器
│   ├── utils/                     # 工具函数
│   │   ├── data_loader.py         # 数据加载器
│   │   ├── metrics.py             # 评估指标
│   │   └── evaluation.py          # 模型评估
│   └── unified_processor.py       # 统一数据处理器
├── results/                       # 训练结果和可视化
├── requirements.txt               # Python依赖包
└── README.md                      # 项目说明文档
```

## De-SaTE-Transformer 架构详解

### 1. 整体架构概述

De-SaTE-Transformer是一个多分支的Transformer架构，专门设计用于处理含噪的时间序列数据。其核心思想是：

```
输入数据 → 多分支去噪 → 特征提取 → 位置编码 → 自注意力机制 → 预测输出
```

### 2. 去噪自编码器模块

#### 2.1 数学原理

去噪自编码器的目标函数为：

$L = \|x - f(g(x + \text{noise}))\|^2 + \lambda \|g(x)\|^2$

其中：

- $x$：原始输入数据  
- $g(\cdot)$：编码器函数  
- $f(\cdot)$：解码器函数  
- $\text{noise}$：添加的噪声  
- $\lambda$：正则化参数  

#### 2.2 噪声类型

系统支持四种噪声类型：

**高斯噪声**：
```python
corrupted_x = x + noise_level * torch.randn_like(x)
```

**泊松噪声**：
```python
rate = torch.abs(x) / (noise_level + 1e-6)
poisson_noise = torch.poisson(rate) * (noise_level + 1e-6)
corrupted_x = x + poisson_noise
```

**斑点噪声**：
```python
corrupted_x = x * (1 + noise_level * torch.randn_like(x))
```

**均匀噪声**：
```python
corrupted_x = x + noise_level * (torch.rand_like(x) - 0.5)
```

### 3. 小波去噪模块

#### 3.1 小波变换理论

小波变换能够在时频域同时提供信号的局部化信息：

$W(a, b) = \frac{1}{\sqrt{a}} \int x(t)\, \psi^*\left(\frac{t-b}{a}\right) dt$

其中：

- $a$：尺度参数  
- $b$：平移参数  
- $\psi(t)$：小波基函数  

#### 3.2 阈值处理方法

**软阈值**：
```python
if |x| > threshold:
    x_thresh = sign(x) * (|x| - threshold)
else:
    x_thresh = 0
```

**硬阈值**：
```python
if |x| > threshold:
    x_thresh = x
else:
    x_thresh = 0
```

**Garrote阈值**：
```python
if |x| > threshold:
    x_thresh = x - sign(x) * threshold²/|x|
else:
    x_thresh = 0
```

### 4. 多头自注意力机制

#### 4.1 注意力计算

自注意力机制的核心公式为：

$\mathrm{Attention}(Q, K, V) = \mathrm{softmax}\left(\frac{QK^\mathrm{T}}{\sqrt{d_k}}\right)V$

其中：

- $Q$：查询矩阵 (Query)  
- $K$：键矩阵 (Key)  
- $V$：值矩阵 (Value)  
- $d_k$：键向量

#### 4.2 多头注意力

多头注意力允许模型同时关注不同位置的信息：

$\mathrm{MultiHead}(Q, K, V) = \mathrm{Concat}(\text{head}_1, \ldots, \text{head}_h) W^O$

其中：

$\text{head}_i = \mathrm{Attention}(Q W_i^Q, K W_i^K, V W_i^V)$


### 5. 位置编码

#### 5.1 正弦位置编码

为了让模型理解序列中的位置信息，使用正弦位置编码：

$\mathrm{PE}(pos, 2i) = \sin\left(\frac{pos}{10000^{\frac{2i}{d_\mathrm{model}}}}\right)$  
$\mathrm{PE}(pos, 2i+1) = \cos\left(\frac{pos}{10000^{\frac{2i}{d_\mathrm{model}}}}\right)$

其中：

- $pos$：位置  
- $i$：维度  
- $d_\mathrm{model}$：模型维度

### 6. 损失函数设计

#### 6.1 总损失函数

总损失包含预测损失和重构损失：

$L_\mathrm{total} = L_\mathrm{prediction} + \alpha \cdot L_\mathrm{reconstruction}$

其中：

- $L_\mathrm{prediction} = \mathrm{MSE}(y_\mathrm{pred},\ y_\mathrm{true})$：预测损失  
- $L_\mathrm{reconstruction} = \mathrm{MSE}(x_\mathrm{decoded},\ x_\mathrm{original})$：重构损失  
- $\alpha$：重构损

#### 6.2 多分支损失

对于多分支模型，损失函数为：

$L_\mathrm{total} = \frac{1}{N} \sum_{i=1}^{N} \left( L_{\mathrm{prediction}_i} + \alpha \cdot L_{\mathrm{reconstruction}_i} \right)$

其中 $N$ 为分支数量。

## 配置环境

### 环境要求

- Python 3.8或更高版本
- PyTorch 1.8或更高版本
- CUDA兼容的GPU（推荐用于训练）
- 8GB以上内存

### 安装步骤

1. 克隆项目仓库：
```bash
git clone <repository-url>
cd De-SaTE-Transformer-Battery
```

2. 创建虚拟环境：
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

3. 安装依赖包：
```bash
pip install -r requirements.txt
```

4. 验证安装：
```bash
python scripts/train.py --help
```

## 使用指南

### 1. 数据预处理

数据预处理是模型训练的关键步骤，脚本支持多种数据集和配置选项。

#### 1.1 基本用法

```bash
# 处理特定数据集
python scripts/preprocess.py --dataset official --raw_data_dir /path/to/raw/data

# 处理所有可用数据集
python scripts/preprocess.py --all --raw_data_dir /path/to/raw/data
```

#### 1.2 完整参数说明

```bash
python scripts/preprocess.py \
    --dataset official \              # 数据集类型：official, nasa, calce, oxford
    --raw_data_dir /path/to/raw \     # 原始数据目录
    --output_dir /path/to/processed \ # 输出目录
    --samples 1000 \                  # 样本数量限制（可选）
    --normalize \                     # 归一化容量值
    --smooth \                        # 平滑容量曲线
    --min_cycles 50 \                 # 最少循环次数
    --verbose \                       # 详细输出
    --overwrite                       # 覆盖现有文件
```

#### 1.3 数据集结构要求

预处理脚本期望以下目录结构：

```
raw_data_dir/
├── SOH-dataset/          # 官方数据集
│   ├── JBGSRS250006644/  # 电池ID目录
│   │   ├── data_*.csv    # 数据文件
│   │   └── ...
│   └── ...
├── NASA/                 # NASA数据集
│   ├── Battery_01.mat
│   └── ...
└── CALCE/                # CALCE数据集
    ├── CS2_35/
    └── ...
```

### 2. 模型训练

训练脚本提供了丰富的配置选项，支持多种模型架构和训练策略。

#### 2.1 基本训练命令

```bash
# 最简单的训练命令
python scripts/train.py \
    --data_path /path/to/processed_data.npy \
    --test_battery JBGSRS250006644
```

#### 2.2 完整训练参数

```bash
python scripts/train.py \
    --data_path /path/to/processed_data.npy \     # 处理后的数据路径
    --test_battery JBGSRS250006644 \              # 测试电池名称
    --epochs 100 \                                # 训练轮数
    --batch_size 128 \                            # 批量大小
    --lr 0.0001 \                                 # 学习率
    --feature_size 128 \                          # 特征维度
    --num_layers 3 \                              # Transformer层数
    --nhead 8 \                                   # 注意力头数
    --dropout 0.1 \                               # Dropout概率
    --window_size 16 \                            # 序列窗口大小
    --output_dir results/ \                       # 结果输出目录
    --save_model \                                # 保存模型
    --seed 42                                     # 随机种子
```

#### 2.3 训练参数详解

**数据参数**：
- `--data_path`: 处理后的数据文件路径（.npy格式）
- `--test_battery`: 用于测试的电池名称（其他电池用于训练）
- `--samples`: 使用的样本数量（None表示全部）

**模型参数**：
- `--feature_size`: Transformer的特征维度（建议128-512）
- `--nhead`: 多头注意力的头数（必须整除feature_size）
- `--num_layers`: Transformer编码器层数（建议1-6）
- `--dropout`: Dropout概率（防止过拟合）
- `--window_size`: 输入序列的窗口大小（建议8-64）

**训练参数**：
- `--epochs`: 训练轮数（建议50-200）
- `--batch_size`: 批量大小（根据GPU内存调整）
- `--lr`: 学习率（Adam优化器，建议0.0001-0.001）
- `--seed`: 随机种子（确保结果可复现）

### 3. 模型评估

#### 3.1 E值评估

E值是项目要求的关键评估指标：

```bash
python scripts/evaluate_with_e_value.py \
    --predictions_file results/predictions_BATTERY.npy \
    --ground_truth_file results/ground_truth_BATTERY.npy \
    --output_dir results/
```

#### 3.2 E值计算公式

```
E = mean(|measured_capacity - true_capacity| / true_capacity) × 100%
```

**评分标准**：
- E ≤ 10%: 得分 = (10 - E) × 1.5（最高15分）
- E > 10%: 得分 = 0分

#### 3.3 其他评估指标

**均方根误差(RMSE)**：
```
RMSE = sqrt(mean((y_pred - y_true)²))
```

**平均绝对误差(MAE)**：
```
MAE = mean(|y_pred - y_true|)
```

**相对误差(RE)**：
```
RE = |true_RUL - pred_RUL| / true_RUL
```

### 4. 结果可视化

#### 4.1 基本可视化

```bash
python scripts/visualize_results.py \
    --results_dir results/ \
    --test_battery JBGSRS250006644 \
    --predictions_file results/predictions_JBGSRS250006644.npy \
    --ground_truth_file results/ground_truth_JBGSRS250006644.npy
```

#### 4.2 可视化内容

生成的可视化包括：

1. **训练损失曲线**：显示模型收敛情况
2. **预测vs实际散点图**：评估预测准确性
3. **时间序列对比图**：显示容量衰减趋势
4. **残差分析图**：评估预测误差分布
5. **E值统计图**：展示E值分布和得分

### 5. 批量训练和超参数优化

#### 5.1 批量训练不同电池

```bash
#!/bin/bash
# 批量训练脚本示例
batteries=("JBGSRS250006644" "JBGSRS250008783" "JBGSRS250006645")

for battery in "${batteries[@]}"; do
    echo "Training on battery: $battery"
    python scripts/train.py \
        --data_path data/processed_data.npy \
        --test_battery $battery \
        --epochs 100 \
        --save_model \
        --output_dir results/$battery
done
```

#### 5.2 超参数网格搜索

```bash
#!/bin/bash
# 超参数搜索脚本
learning_rates=(0.001 0.0001 0.00001)
window_sizes=(8 16 32)
layer_nums=(1 2 3)

for lr in "${learning_rates[@]}"; do
    for ws in "${window_sizes[@]}"; do
        for nl in "${layer_nums[@]}"; do
            echo "Training with lr=$lr, window_size=$ws, num_layers=$nl"
            python scripts/train.py \
                --data_path data/processed_data.npy \
                --test_battery JBGSRS250006644 \
                --lr $lr \
                --window_size $ws \
                --num_layers $nl \
                --epochs 50 \
                --output_dir results/grid_search/lr_${lr}_ws_${ws}_nl_${nl}
        done
    done
done
```

## 高级功能

### 1. De-SaTE多分支架构

#### 1.1 分支配置

默认使用四个分支，每个分支对应不同的去噪方法：

```python
branch_configs = [
    {'denoiser_type': 'autoencoder', 'noise_type': 'gaussian', 'noise_level': 0.01},
    {'denoiser_type': 'autoencoder', 'noise_type': 'poisson', 'noise_level': 0.01},
    {'denoiser_type': 'autoencoder', 'noise_type': 'speckle', 'noise_level': 0.01},
    {'denoiser_type': 'wavelet', 'noise_type': None, 'noise_level': 0},
]
```

#### 1.2 自定义分支

```python
# 自定义分支配置示例
custom_configs = [
    {'denoiser_type': 'autoencoder', 'noise_type': 'gaussian', 'noise_level': 0.05},
    {'denoiser_type': 'wavelet', 'noise_type': None, 'noise_level': 0},
]

model = DeSaTETransformer(
    input_size=input_size,
    branch_configs=custom_configs
)
```

### 2. 模型集成

#### 2.1 多模型投票

```python
# 训练多个模型
models = []
for seed in [42, 123, 456]:
    model = train_model(seed=seed)
    models.append(model)

# 集成预测
ensemble_predictions = []
for model in models:
    pred = model.predict(test_data)
    ensemble_predictions.append(pred)

final_prediction = np.mean(ensemble_predictions, axis=0)
```

#### 2.2 加权集成

```python
# 根据验证集性能确定权重
weights = [0.4, 0.35, 0.25]  # 基于验证集RMSE确定
weighted_prediction = np.average(ensemble_predictions, weights=weights, axis=0)
```

### 3. 迁移学习

#### 3.1 预训练模型

```bash
# 在大数据集上预训练
python scripts/train.py \
    --data_path large_dataset.npy \
    --test_battery test1 \
    --epochs 200 \
    --save_model \
    --output_dir pretrained_models/
```

#### 3.2 微调

```bash
# 在特定数据集上微调
python scripts/train.py \
    --data_path target_dataset.npy \
    --test_battery test2 \
    --load_model pretrained_models/best_model.pth \
    --lr 0.00001 \
    --epochs 50 \
    --output_dir fine_tuned_models/
```

> TODO: 实际运行模型并没有微调，只是进行了默认的train，之后会进行微调相关测试。