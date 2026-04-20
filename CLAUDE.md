# CLAUDE.md

本文件为 Claude Code (claude.ai/code) 在本代码库中工作时提供指导。

## 项目概述

这是一个用于视觉语言模型（VLM）的多轮强化学习训练项目，结合了：
- **verl**：字节跳动 Seed 团队开发的灵活 RL 训练库
- **agent_tools**：图像恢复工具包（ESRGAN、Retinexformer、IDT 等）
- **checkpoints**：预训练模型权重

主要用例是训练 VLM 通过多轮对话中的工具调用执行迭代图像恢复。

## 仓库结构

```
RL4VLM/
├── verl/                          # 主 verl RL 训练库
│   ├── verl/
│   │   ├── trainer/               # PPO/GRPO 训练器和入口
│   │   │   ├── main_ppo.py        # 主入口：python -m verl.trainer.main_ppo
│   │   │   └── ppo/               # 核心训练逻辑 (ray_trainer.py, core_algos.py)
│   │   ├── workers/               # FSDP、vLLM、SGLang rollout workers
│   │   │   ├── fsdp_workers.py    # FSDP 训练后端
│   │   │   └── engine_workers.py  # vLLM/SGLang 推理后端
│   │   ├── interactions/          # 用于奖励计算的交互代理
│   │   │   ├── base.py            # BaseInteraction 抽象基类
│   │   │   └── image_restoration_interaction.py  # 图像恢复奖励逻辑
│   │   ├── tools/                 # 多轮 rollout 的工具实现
│   │   │   ├── base_tool.py       # BaseTool 抽象基类
│   │   │   └── restoration_tool.py # 图像恢复工具 (ESRGAN, Retinexformer 等)
│   │   └── protocols.py           # DataProto 消息传递
│   └── examples/
│       └── sglang_multiturn/     # 多轮 RL 训练示例
│           └── config/            # 训练配置 (restoration_multiturn_grpo.yaml)
├── agent_tools/                   # 图像恢复工具包
│   ├── ESRGAN/                   # 超分辨率模型
│   ├── Retinexformer/            # 低光照增强
│   ├── IDT/                      # 去雨/去水滴
│   ├── RIDCP/                    # 去雾
│   ├── SCUNet/                   # 降噪
│   ├── HVICIDNet/                # 低光照校正
│   └── LightenDiffusion/         # 低光照增强（扩散模型）
└── checkpoints/                   # 预训练模型权重
    └── q_align/                  # IQA（图像质量评估）模型
```

## 运行训练

### 多轮图像恢复训练

```bash
cd /home/LXJ/Python_Projects/verl/verl
python3 -m verl.trainer.main_ppo \
  --config-path="$(pwd)/examples/sglang_multiturn/config" \
  --config-name='restoration_multiturn_grpo'
```

配置文件位于 `verl/examples/sglang_multiturn/config/`：
- `restoration_multiturn_grpo.yaml` - 主训练配置
- `tool_config/restoration_tool_config.yaml` - 工具定义（恢复动作）
- `interaction_config/restoration_interaction_config.yaml` - 奖励计算和交互参数

### 数据格式

训练数据以 parquet 文件形式存储，具有多轮对话结构。数据路径见 `restoration_multiturn_grpo.yaml`：
- `train_files`：训练 parquet 文件列表
- `val_files`：验证 parquet 文件列表

## 核心架构概念

### 多轮 Rollout 流程

1. **SGLang Rollout Worker** 通过 `verl/workers/rollout/` 生成响应
2. **工具调用** 从响应中解析（格式：`<answer>tool_name</answer>`）
3. **RestorationTool** 使用 `agent_tools/` 中的模型执行图像恢复
4. **ImageRestorationInteraction** 使用 IQA 指标计算奖励（QAlign、MANIQA、MUSIQ、CLIPIQA、NIQE）
5. **终止决策** 基于最大迭代次数或 stop 动作

### 交互系统

交互定义了如何计算奖励以及何时终止 rollout 回合：
- `BaseInteraction`（`verl/interactions/base.py`）- 抽象基类
- `ImageRestorationInteraction` 使用基于 IQA 的奖励和特定降解类型权重计算
- 方法：`start_interaction`、`generate_response`、`calculate_score`、`finalize_interaction`

### 工具系统

工具在 YAML 配置中定义，在 Python 中实现：
- `BaseTool`（`verl/tools/base_tool.py`）- 抽象基类
- `RestorationTool` 包装 `agent_tools/restoration_toolkit.py`
- 支持的动作：real_esrgan、scunet、retinexformer_fivek、hvicidnet、lightdiff、turbo_rain、s2former、idt、ridcp、kanet、turbo_snow

### 训练 Workers

- **FSDP Workers**（`verl/workers/fsdp_workers.py`）：用于训练的模型并行
- **Engine Workers**（`verl/workers/engine_workers.py`）：vLLM/SGLang 推理
- **基于 Ray 的协调** 通过 `verl/single_controller/ray/`

## 代码检查和测试

```bash
# 使用 ruff 检查（配置在 verl/pyproject.toml）
cd verl && ruff check .

# 运行测试（CPU 测试有 _on_cpu 后缀）
cd verl && python3 -m pytest tests/special_sanity/ -v
```

## 依赖

- Python >= 3.10
- PyTorch（带 CUDA）
- Ray（用于分布式训练）
- SGLang 或 vLLM（用于推理）
- Hydra（用于配置管理）
- transformers、peft（用于模型处理）

## 重要提示

- **agent_tools 路径**：`agent_tools/` 位于项目根目录，由恢复工具添加到 `sys.path`
- **IQA 评分**：由 `ImageRestorationInteraction` 使用 `agent_tools/iqa_reward.py` 处理，不由 RestorationTool 处理
- **GPU 内存管理**：恢复模型使用专用 GPU（如 `cuda:3`）以避免与 SGLang rollout 冲突
- **动态模型加载**：RestorationTool 使用 `preload=False、auto_unload=True` 管理 GPU 内存
