---
applyTo: "verl/verl/**/*.py"
description: "编辑 verl 核心 Python 模块时使用：interactions、tools、workers、trainers、protocol。涵盖 async 模式、GPU 内存规范和扩展接口。"
---

# verl 核心模块规范

## 异步模式

所有 interaction 和 tool 方法必须是 `async`：

```python
async def start_interaction(self, instance_id=None, **kwargs) -> str:
async def generate_response(self, instance_id, messages, **kwargs) -> tuple[bool, str, float, dict]:
async def execute(self, instance_id, action, **kwargs) -> tuple[ToolResponse, float, dict]:
```

## 设备管理

- 以配置参数形式接收 `device`（字符串或 `torch.device`）
- 恢复/IQA 模型使用 `cuda:3`，避免与 SGLang（GPU 0-2）冲突
- 使用 `preload=False, auto_unload=True` 进行动态 GPU 内存管理

## 关键基类

- `BaseInteraction`（`verl/interactions/base.py`）— 实现 `start_interaction`、`generate_response`、`calculate_score`、`finalize_interaction`
- `BaseTool`（`verl/tools/base_tool.py`）— 实现 `create`、`execute`、`calc_reward`、`get_openai_tool_schema`

## 导入规范

```python
from verl.interactions.base import BaseInteraction
from verl.tools.base_tool import BaseTool
from verl.protocol import DataProto
```

## 日志

```python
import logging
logger = logging.getLogger(__name__)
```

通过 `VERL_LOGGING_LEVEL` 环境变量控制日志级别。

## 错误处理

- 可选操作（如 IQA 计算失败）应记录警告而非抛出异常
- 使用 `contextmanager` 在模型加载时屏蔽 stdout 输出
