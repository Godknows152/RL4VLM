---
applyTo: "agent_tools/**/*.py"
description: "编辑 agent_tools 图像恢复代码时使用：RestorationToolkit、IQA 评分、各模型封装（ESRGAN、Retinexformer、SCUNet 等）。"
---

# agent_tools 规范

## 架构

- `restoration_toolkit.py` — 统一封装所有恢复模型的 `RestorationToolkit` 类
- `iqa_reward.py` — 多指标图像质量评估的 `IQAScore` 类
- 各子目录（ESRGAN/、Retinexformer/ 等）各自包含独立模型

## 模型加载模式

模型使用懒加载 + 自动卸载：

```python
toolkit = RestorationToolkit(device="cuda:3", preload=False, auto_unload=True)
```

- `preload=False`：首次调用时加载，非初始化时加载
- `auto_unload=True`：每次推理后释放 GPU 内存
- `all_model_paths`：动作名称到检查点路径的映射字典

## 路径管理

- `agent_tools/` 位于项目根目录，**不在** `verl/` 内部
- 由 `restoration_tool.py` 在运行时注入 `sys.path`
- 子模块内部使用 `Path(__file__).resolve().parent` 解析相对路径

## IQA 指标

支持的指标：QAlign、MANIQA、MUSIQ、CLIPIQA、NIQE。各降质类型的权重在 `image_restoration_interaction.py` 的 `SCORE_WEIGHT_MAP` 中定义。

## 添加新模型

1. 在 `agent_tools/` 下创建子目录并放入模型代码
2. 将检查点添加到 `checkpoints/agent_tools/`
3. 在 `RestorationToolkit` 中添加加载/推理函数
4. 在 `all_model_paths` 字典中注册新模型
