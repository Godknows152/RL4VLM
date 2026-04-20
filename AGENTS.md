# 项目指南

## 概述

面向视觉语言模型（VLM）的多轮强化学习训练项目，通过工具调用执行迭代式图像恢复。基于 [verl](verl/)（字节跳动 RL 框架）构建，搭配自定义图像恢复工具包 [agent_tools/](agent_tools/)。

详细的代码库结构和架构说明见 [CLAUDE.md](CLAUDE.md)。

## 构建与测试

```bash
# 安装（在 verl/ 目录下）
cd verl && pip install -e .[test,sglang]

# 代码检查（ruff，行长限制 120）
cd verl && pre-commit run --all-files

# 运行测试
cd verl && python3 -m pytest tests/special_sanity/ -v
```

## 启动训练

```bash
cd /home/LXJ/Python_Projects/RL4VLM/verl/verl
python3 -m verl.trainer.main_ppo \
  --config-path="$(pwd)/examples/sglang_multiturn/config" \
  --config-name='restoration_multiturn_grpo'
```

## 架构

### 多轮 Rollout 流程

1. **SGLang Rollout** 生成模型响应（GPU 0-2）
2. **工具调用** 从 `<answer>tool_name</answer>` 格式中解析
3. **RestorationTool** 在专用 GPU（`cuda:3`）上执行图像恢复
4. **ImageRestorationInteraction** 计算基于 IQA 的奖励
5. 循环直到触发 `stop` 动作或达到 `max_iterations`（默认 5 轮）

### 关键抽象

| 抽象层 | 基类 | 实现 | 配置 |
|---|---|---|---|
| Interaction | [verl/verl/interactions/base.py](verl/verl/interactions/base.py) | [image_restoration_interaction.py](verl/verl/interactions/image_restoration_interaction.py) | [restoration_interaction_config.yaml](verl/examples/sglang_multiturn/config/interaction_config/restoration_interaction_config.yaml) |
| Tool | [verl/verl/tools/base_tool.py](verl/verl/tools/base_tool.py) | [restoration_tool.py](verl/verl/tools/restoration_tool.py) | [restoration_tool_config.yaml](verl/examples/sglang_multiturn/config/tool_config/restoration_tool_config.yaml) |
| Toolkit | — | [agent_tools/restoration_toolkit.py](agent_tools/restoration_toolkit.py) | — |
| IQA 评分 | — | [agent_tools/iqa_reward.py](agent_tools/iqa_reward.py) | — |

### 扩展：添加新的降质类型

1. 将模型添加到 `agent_tools/`
2. 更新 `RestorationToolkit.all_model_paths`，添加加载/推理函数
3. 在 `restoration_tool.py` 的 `ALLOWED_ACTIONS` 中添加新动作
4. 在 `image_restoration_interaction.py` 的 `SCORE_WEIGHT_MAP` 中添加 IQA 权重
5. 在工具配置 YAML 的 OpenAI schema 中添加新动作

## 代码规范

- **全面异步**：所有 interaction 和 tool 方法必须是 `async`
- **懒加载模型**：使用 `preload=False, auto_unload=True` 管理 GPU 内存
- **GPU 隔离**：恢复模型使用 `cuda:3`，SGLang rollout 使用 GPU 0-2
- **日志**：`logger = logging.getLogger(__name__)`，通过 `VERL_LOGGING_LEVEL` 环境变量控制
- **配置**：训练配置使用 Hydra，工具/交互配置使用 YAML
- **导入**：`from verl.interactions.base import BaseInteraction`；agent_tools 在运行时注入 `sys.path`

## 关键陷阱

- **GPU 内存冲突**：禁止在 SGLang rollout 使用的 GPU 上加载恢复/IQA 模型
- **agent_tools 路径**：`agent_tools/` 位于项目根目录，由 restoration_tool.py 注入 `sys.path`
- **IQA 评分**：由 `ImageRestorationInteraction` 负责，**不是** `RestorationTool`
- **NumPy 版本**：必须 <2.0.0 以兼容 PyTorch
- **数据格式**：训练数据为 parquet 文件，采用多轮对话结构

## 文档

- [verl/CONTRIBUTING.md](verl/CONTRIBUTING.md) — 开发环境搭建、CI、测试
- [verl/README.md](verl/README.md) — verl 框架概述
- [verl/docs/](verl/docs/) — 完整文档（使用 `make html` 构建）
