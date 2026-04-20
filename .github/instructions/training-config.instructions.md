---
applyTo: "verl/examples/sglang_multiturn/config/**/*.yaml"
description: "编辑多轮 RL 训练流程的 Hydra 训练配置、工具配置或交互配置时使用。"
---

# 训练配置规范

## 配置层级

```
restoration_multiturn_grpo.yaml    # 主 Hydra 配置（数据、算法、模型、训练器）
├── tool_config/
│   └── restoration_tool_config.yaml   # 工具类、设备、OpenAI 函数 schema
└── interaction_config/
    └── restoration_interaction_config.yaml  # 奖励计算、IQA 参数、最大迭代次数
```

## 关键参数

- `algorithm.adv_estimator: grpo` — 使用 GRPO 优势估计
- `actor_rollout_ref.rollout.name: sglang` — SGLang 推理后端
- `max_iterations: 5` — 每个 episode 最大工具调用轮次
- `device: cuda:3` — 恢复/IQA 专用 GPU（与 SGLang 的 GPU 0-2 隔离）
- `preload: false, auto_unload: true` — 动态 GPU 内存管理

## 工具配置

`tools` 部分使用 OpenAI 函数调用 schema 格式。添加新动作时需同步更新：
1. schema 中 `action` 参数的 `enum` 列表
2. 列出可用工具的 `description` 字段

## 交互配置

奖励公式：`alpha * (当前分数 - 上一轮分数) + (1-alpha) * (当前分数 - identity 分数)`
- `alpha=0.9` 侧重边际提升
- `reward_scale` 控制整体奖励幅度
