<p align="center">
    <img src="./assets/logo.svg" alt="Logo" />
    <p align="center">
        <a href="https://github.com/PKU-DAIR">
            <img alt="Static Badge" src="https://img.shields.io/badge/%C2%A9-PKU--DAIR-%230e529d?labelColor=%23003985">
        </a>
        <a href="https://github.com/PKU-DAIR/SAS-Bench">
            <img alt="Static Badge" src="https://img.shields.io/badge/SAS--Bench-black?logo=github">
        </a>
        <a href="https://github.com/PKU-DAIR/SAS-Bench">
            <img alt="GitHub Repo stars" src="https://img.shields.io/github/stars/PKU-DAIR/SAS-Bench?logo=github&style=flat">
        </a>
    </p>
</p>

## SAS-Bench: A Comprehensive Benchmark for Short Answer Scoring with LLMs

[数据集](https://huggingface.co/datasets/aleversn/SAS-Bench) | [论文](https://arxiv.org/pdf/2505.07247) | [代码](https://github.com/PKU-DAIR/SAS-Bench)

## 🔍 项目概述

SAS-Bench是首个专门针对大语言模型(LLM)的简答题评分(SAS)基准测试。基于中国高考真实试题构建，本基准测试具有以下特点：

- **1030道试题**覆盖9大学科领域
- **4109份专家标注的学生答案**
- **分步评分**与**分布错因分析**
- **多维度评估体系**（整体评分、分步评分、错因诊断一致性）

## 🚀 核心特色

### 突破传统SAS系统局限
SAS-Bench解决了传统简答题评分系统的关键缺陷：

| 维度           | 传统SAS系统   | SAS-Bench优势      |
| -------------- | ------------- | ------------------ |
| **评分粒度**   | 单一总分      | 分步分解评分       |
| **可解释性**   | 黑箱机制      | 完备的错因类型体系 |
| **答案多样性** | 单一学科/题型 | 跨学科非模板化评估 |

### 数据集特性

<p align="center">
    <img src="./assets/annotation.png" alt="SAS人工标注系统" width="50%" />
</p>

数据集包含三类题目及丰富标注：

1. **选择题**（自由填写形式）
2. **填空题**
3. **简答题**（含步骤分解）

每份答案包含：
- ✅ 人工标注整体得分
- 🔍 步骤划分与分项评分
- ❌ 步骤错因归类

## 🌟 评估框架

### CCS评估（协同一致性评分）

**目的**  
衡量模型预测与人工评分在整体得分和步骤得分上的协同一致性，确保模型理解详细推理过程。

**公式**  
调整权重矩阵结合整体与步骤差异：
```math
W_{i,j} = \alpha \cdot \frac{(r_i - r_j)^2}{(N_r - 1)^2} + \frac{1 - \alpha}{m} \sum_{k=1}^{m} \frac{(s_{i,k} - s_{j,k})^2}{(N_{s_k} - 1)^2}
```
其中：  
- $r_i, r_j$：模型/人工整体评分  
- $s_{i,k}, s_{j,k}$：第$k$步得分  
- $\alpha=0.5$：平衡权重  
- $N_r, N_{s_k}$：可能得分等级  

最终CCS计算：
```math
\text{CCS} := 1 - \frac{\sum_{i,j} O_{i,j} \cdot W_{i,j}}{\sum_{i,j} E_{i,j} \cdot W_{i,j}}
```

### ECS评估（错因一致性评分）

**目的**  
量化模型识别错因类型的能力，按答案质量分层评估。

**公式**  
1. 使用分位数阈值$\tau_1, \tau_2$将样本分为3组（低/中/高）：
```math
\phi(x) = \mathbb{I}(x \geq \tau_1) + \mathbb{I}(x \geq \tau_2)
```
2. 计算每组的错因频率矩阵$\mathbf{M}^p_k, \mathbf{M}^g_k$
3. 计算组内Spearman相关性：
```math
\rho_k = \text{SpearmanR}(\mathbf{M}^p_k, \mathbf{M}^g_k)
```
最终ECS：
```math
\text{ECS} := \frac{1}{m} \sum_{k=0}^{2} \rho_k
```

**关键特性**  
- 采用**3级性能分层**（m=3）确保稳健评估  
- 关联**错因类型分布**（而非简单计数）  
- 标准化评分支持跨数据集比较

## ⚙️ 安装指南

### 核心依赖
```bash
pip install protobuf transformers>=4.44.1 cpm_kernels torch>=2.0 gradio mdtex2html sentencepiece accelerate json_repair openai
```

或：
```bash
pip install -r requirements.txt
```

### vLLM环境配置（推荐）
```bash
conda create -n vllm python=3.12 -y
conda activate vllm
pip install vllm  # 需CUDA 12.0+
```

其他配置请参考官方[vLLM安装指南](https://docs.vllm.ai/en/latest/getting_started/installation/gpu.html)。

## 📊 基准测试流程

![工作流程](./assets/workflow.png)

### 目录结构

```
|- discuss/        - 分析脚本
|- docs/           - 文档资源
|- main/           - 模型训练/推理代码
|- prompts/        - 预定义提示模板
|- sas_pipelines/  - 主要评估代码
|- utils/          - 工具函数
```

### 实施选项

#### 0. 数据预处理（标注阶段）
- 原始标注数据位于`backend_data`
- 运行`preprocess.py`进行数据整合
- 修改`DATANAME`变量指定源文件（不含扩展名）

> 此流程处理来自我们标注系统（系统即将开源）的原始数据

#### 1. 数据获取
数据集发布于[HuggingFace数据集](https://huggingface.co/datasets/aleversn/SAS-Bench)。下载文件存放于`datasets/`：
- 文件命名格式为`{q_id}_{course}_{question_type}.jsonl`
- 错因分类体系在`error_type.jsonl`中：
  ```json
  {"q_id": 2, "course": "", "question_type": "", "guideline": "", "errors": [{"name": "", "description": ""}...]}
  ```
- `ID_Dict.json`包含学科-ID映射

#### 2. LLM预测
支持Jupyter或命令行执行：

**选项A：Jupyter Notebook**
- 在`1_predict_scores.py`中设置`cmd_args = False`
- 配置：
  - `save_type_name`：模型标识符/输出前缀
  - `model_from_pretrained`：模型路径
  - `file_name`：数据集标识（如`7_Math_ShortAns`）

**选项B：命令行**
设置`cmd_args = True`

*使用vLLM（推荐）*：
```bash
cd sas_pipelines/
python 1_predict_scores.py --file_name=6_Chinese_ShortAns --save_type_name=<模型ID> --model_from_pretrained=<路径> --batch_size=1000 --vllm=1
```

*启用Tensor并行*：
```bash
python 1_predict_scores.py --n_gpu=0,1 --file_name=6_Chinese_ShortAns --save_type_name=<模型ID> --model_from_pretrained=<路径> --batch_size=1000 --vllm=1 --tensor_parallel_size=2
```

*HuggingFace预测器*：
```bash
python 1_predict_scores.py --file_name=6_Chinese_ShortAns --save_type_name=<模型ID> --model_from_pretrained=<路径> --batch_size=5
```

*OpenAI API预测*：
1. 在`sas_pipeline/`创建`api_key.txt`，格式：
   ```text
   OpenAI <API密钥>
   Deepseek <API密钥>
   ```
2. 执行：
   ```bash
   python 1_predict_scores.py --file_name=6_Chinese_ShortAns --llm_name=deepseek-chat --save_type_name=Deepseek_V3
   ```

**附加参数**：
- 使用小样本示例：`--few_shot_num >0`
- 禁用评分指南：`--use_guideline=0`
- 跳过深度思考：`--skip_thinking=1`
- `llm_name`默认为`save_type_name`（GLM3/OpenAI模型除外）

#### 3. 预测处理
**选项A：Jupyter**
- 在`2_process_prediction.py`设置`cmd_args = False`
- 配置`file_name`（使用`all`进行批量处理）

**选项B：命令行**
```bash
python 2_process_prediction.py --file_name=all
```

#### 4. CCS计算
**选项A：Jupyter**
- 在`3_compute_ccs.py`配置`file_name`和`save_type_name`

**选项B：命令行**
```bash
python 3_compute_ccs.py --save_type_name=<模型前缀>
```

#### 5. ECS计算
**选项A：Jupyter**
- 在`4_compute_ecs.py`调整参数

**选项B：命令行**
```bash
python 4_compute_ecs.py --save_type_name=<模型前缀>
```

## 📈 Model Performance Insights

在16个LLMs上进行了实验:

- QWK

![Workflow](./assets/qwk.png)

- CCS

| Models                 | Phy. (S.) | Phy. (M.) | His. (S.) | Geo. (S.) | Bio. (G.) | Chi. (G.) | Chi. (S.) | Math (S.) | Math (G.) | Pol. (S.) | Eng. (G.) | Che. (G.) | Avg.      |
| ---------------------- | --------- | --------- | --------- | --------- | --------- | --------- | --------- | --------- | --------- | --------- | --------- | --------- | --------- |
| Deepseek-R1            | 38.43     | **95.01** | **80.98** | 67.92     | **79.12** | 95.09     | 69.07     | 57.85     | **83.56** | 71.92     | 73.19     | 72.92     | 73.76     |
| QwQ-32B                | 48.53     | 87.23     | 75.43     | **77.06** | 72.52     | **96.00** | 31.77     | 48.66     | 45.51     | 74.48     | 54.79     | 62.17     | 64.51     |
| TinyR1-32B-Preview     | 38.17     | 84.88     | 75.83     | 71.52     | 73.45     | 92.57     | 52.61     | 48.28     | 74.77     | 70.70     | 57.92     | 41.37     | 65.17     |
| Qwen3-32B              | 47.29     | 85.51     | 64.96     | 80.43     | 63.15     | 92.21     | 50.43     | 51.26     | 80.77     | 73.30     | 59.33     | 57.82     | 67.20     |
| Qwen3-8B               | 54.33     | 76.17     | 45.54     | 68.89     | 43.22     | 86.01     | 42.02     | 46.33     | 73.33     | 64.25     | 50.55     | 50.52     | 58.43     |
| MiMo-7B-RL             | 52.77     | 41.01     | 61.33     | 67.10     | 35.93     | 54.72     | 43.09     | 38.09     | 55.79     | 36.78     | 34.69     | 31.05     | 46.03     |
| Deepseek-Prover-V2-7B  | 22.59     | 10.75     | 2.92      | 30.71     | 50.63     | 55.48     | 12.95     | 0.87      | 2.29      | 10.44     | 30.19     | 28.76     | 21.55     |
| DeepSeek-R1-Distill-7B | 33.71     | 29.24     | 50.92     | 32.35     | 52.18     | 52.44     | 44.29     | 29.52     | 39.55     | 53.77     | 32.98     | 34.27     | 40.44     |
| Deepseek-V3            | 53.89     | 85.72     | 69.85     | 76.23     | 76.51     | 93.42     | **69.49** | **58.81** | 80.18     | **76.75** | **73.82** | **74.64** | **74.11** |
| GPT 4o-mini-20240718   | **58.90** | 81.19     | 54.85     | 76.59     | 65.39     | 87.65     | 55.25     | 43.56     | 37.38     | 63.44     | 22.60     | 55.98     | 58.56     |
| Llama3.3-70B-Instruct  | 45.34     | 70.03     | 72.02     | 72.51     | 67.94     | 85.30     | 35.83     | 58.60     | 74.97     | 63.68     | 67.60     | 38.94     | 62.73     |
| Mixtral 8×7B-Instruct  | 30.78     | 42.27     | 33.43     | 4.99      | 44.45     | 29.85     | 24.00     | 26.73     | 70.04     | 43.92     | 33.40     | 42.05     | 35.49     |
| Qwen2.5-32B-Instruct   | 40.53     | 77.02     | 62.34     | 74.50     | 72.07     | 94.85     | 66.37     | 50.08     | 32.59     | 64.09     | 53.35     | 62.87     | 62.56     |
| Qwen2.5-14B-Instruct   | 53.76     | 66.12     | 60.96     | 74.30     | 67.50     | 92.81     | 63.08     | 43.28     | 75.62     | 62.03     | 56.34     | 57.53     | 64.44     |
| GLM4-9B-Chat           | 45.62     | 52.33     | 36.81     | 69.41     | 39.19     | 63.92     | 42.94     | 35.50     | 56.95     | 54.83     | 33.92     | 30.79     | 46.85     |
| Llama3-8B-Instruct     | 41.09     | 35.10     | 37.52     | 31.29     | 32.19     | 38.13     | 32.89     | 23.55     | 62.43     | 37.78     | 31.68     | 29.27     | 36.08     |

- ECS

| Models                 | Phy. (S.) | Phy. (M.) | His. (S.) | Geo. (S.) | Bio. (G.) | Chi. (G.) | Chi. (S.) | Math (S.) | Math (G.) | Pol. (S.) | Eng. (G.) | Che. (G.) | Avg.      |
| ---------------------- | --------- | --------- | --------- | --------- | --------- | --------- | --------- | --------- | --------- | --------- | --------- | --------- | --------- |
| Deepseek-R1            | 23.25     | 30.59     | 57.53     | 56.08     | 69.20     | 86.04     | 72.68     | **94.29** | 15.20     | 65.56     | _18.65_   | _81.76_   | **55.90** |
| QwQ-32B                | 4.74      | **63.92** | 67.06     | _70.04_   | 53.68     | 51.08     | 69.20     | 79.05     | 16.82     | 48.81     | -22.53    | 48.94     | 45.90     |
| TinyR1-32B-Preview     | 3.10      | **63.92** | 65.71     | **77.02** | 56.61     | 64.42     | 74.83     | 82.86     | 23.33     | 40.17     | -31.52    | 17.35     | 44.82     |
| Qwen3-32B              | -4.17     | 24.18     | _69.52_   | 54.29     | 53.67     | 52.70     | 47.31     | 82.21     | 18.33     | 62.14     | -26.99    | 36.27     | 39.12     |
| Qwen3-8B               | 23.39     | **63.92** | 14.29     | -4.96     | 52.21     | 47.75     | 34.01     | 39.20     | -8.14     | 57.19     | -27.13    | 59.28     | 29.25     |
| MiMo-7B-RL             | **51.05** | 24.18     | 14.29     | 38.85     | 58.35     | _92.17_   | 63.07     | 13.39     | 35.12     | -27.10    | -4.41     | 1.04      | 30.00     |
| Deepseek-Prover-V2-7B  | -24.10    | -5.20     | 42.86     | -6.23     | 29.54     | -80.81    | 23.25     | 46.67     | -1.51     | -58.64    | -45.23    | -21.91    | -8.44     |
| DeepSeek-R1-Distill-7B | -45.19    | 24.18     | 0.95      | -38.66    | 23.55     | -20.36    | 3.87      | -23.81    | -13.57    | -18.81    | -19.59    | -44.58    | -14.34    |
| Deepseek-V3            | 7.79      | 46.58     | 58.10     | 32.62     | _72.38_   | **96.58** | 57.43     | _92.38_   | _33.33_   | 40.26     | **24.77** | **85.83** | _54.00_   |
| GPT 4o-mini-20240718   | 17.91     | 24.18     | 62.14     | 36.68     | 55.20     | 79.01     | **78.00** | 67.62     | **46.90** | **92.31** | 10.04     | 36.39     | 50.53     |
| Llama3.3-70B-Instruct  | 22.56     | _57.35_   | 54.29     | 42.11     | 45.09     | 52.70     | 46.25     | 54.29     | 30.00     | 58.81     | -12.53    | -15.83    | 36.26     |
| Mixtral 8×7B-Instruct  | 11.99     | 17.34     | **80.38** | 35.84     | 32.74     | 42.77     | 75.82     | 56.19     | 30.00     | 6.84      | -31.16    | -7.18     | 29.30     |
| Qwen2.5-32B-Instruct   | 11.95     | 17.41     | 53.33     | 59.34     | 62.96     | 46.90     | 75.08     | 62.86     | 30.00     | 46.67     | -4.50     | 27.08     | 40.76     |
| Qwen2.5-14B-Instruct   | 21.50     | 24.18     | 47.92     | 37.43     | **73.36** | 64.97     | 74.32     | 64.94     | 18.21     | 61.97     | -20.00    | 47.39     | 43.02     |
| GLM4-9B-Chat           | 35.00     | 24.18     | 32.49     | 34.73     | 62.12     | 20.36     | _77.34_   | 63.81     | **46.90** | _82.40_   | -25.35    | 7.18      | 38.43     |
| Llama3-8B-Instruct     | _48.25_   | 27.46     | 17.23     | 31.58     | 61.37     | -14.05    | 41.23     | 57.77     | 21.55     | -69.07    | -26.50    | -27.19    | 14.14     |

## 📅 待办事项

- [ ] 提供英文本地化版本数据集
- [ ] 开源标注系统（前端 & 后端）

## 📜 许可声明

SAS-Bench采用`Apache License 2.0`协议发布。本数据集仅限研究用途使用。

## 📚 引用方式

```bibtex
@article{lai2025sasbenchfinegrainedbenchmarkevaluating,
      title={SAS-Bench: A Fine-Grained Benchmark for Evaluating Short Answer Scoring with Large Language Models}, 
      author={Peichao Lai and Kexuan Zhang and Yi Lin and Linyihan Zhang and Feiyang Ye and Jinhao Yan and Yanwei Xu and Conghui He and Yilei Wang and Wentao Zhang and Bin Cui},
      year={2025},
      journal={arXiv preprint arXiv:2505.07247},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2505.07247}, 
}
```
