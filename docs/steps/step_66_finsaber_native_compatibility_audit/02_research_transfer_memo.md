# 研究观点与可迁移点备忘录

## 研究判断整理

### 从当前审计得到的判断
- 当前问题更像“方法与契约没有完全对齐”，不是“DRL 天生不适合投资”。
- native backend 下最核心的研究问题不是再加多少技术指标，而是：
  - raw state contract 是否被 LLM 正确理解；
  - revise path 和 intrinsic path 是否都能独立产生可观测增益；
  - prompt 中的 regime/context/portfolio memory 是否真正进入候选机制，而不是停留在口头提示。
- 当前窗口已经证明：
  - `ppo` 还能出现相对明确的 intrinsic 独立增益；
  - `td3` 和 `sac` 在当前 best candidate 上更像“state gain 被误记成 LESR full-path gain”。

### 与截图命题的对应关系

| 截图命题 | 当前判断 | 备注 |
| --- | --- | --- |
| `DRL 本身就有波动和不稳定` | 部分成立 | 需要跨窗口、多 seed、stress slice 进一步验证，单窗口不足以下结论 |
| `问题可能在方法而非 DRL 是否适合投资` | 目前最可信 | native contract 与 LESR 反馈链残留旧假设，足以解释当前很多失效现象 |
| `先 baseline 再优化` | 成立 | 应先把 contract、prompt、selection 归因问题理清，再继续扩 reward/state 结构 |
| `滚动窗口与样本选择也要解释` | 成立 | walk-forward、scenario family、per-algo branch 已经明显影响候选排名和结论 |

## 六个方向的研究观点

### 1. State Representation
- 当前最值得迁移的方向不是继续堆 price-only technical indicators，而是把 `portfolio memory + regime + dispersion + running risk state + global context` 明确升为一等状态。
- 对你这个项目最直接的含义：
  - native raw prefix 不动；
  - revised dims 重点追加 portfolio-level summary，而不是假装 raw state 里有 OHLCV。
- 可迁移来源：
  - Jiang et al. 2017 强调 `Portfolio-Vector Memory (PVM)`，适合映射到你们当前的 holdings/cash/exposure/memory 线索。
  - 2025 的 dynamic embedding 论文强调 market information embedding，不是单纯手工指标堆叠。
  - FinRL 官方 portfolio allocation 教程把 `covariance + technical indicators` 当作基础 state，而不是只看价格。

### 2. Reward Shaping
- 当前问题不适合继续把所有目标塞进单一 intrinsic 标量里。
- 更合理的迁移路线：
  - 多 reward heads
  - 或多专家候选，在 selection/eval 端再融合
- 当前 repo 最接近的切入点：
  - `candidate_scoring`
  - `intrinsic_postprocess`
  - final selection 逻辑

### 3. Stability / Robustness
- 连续动作环境里，真正该优先追的不是“更大的分数”，而是：
  - action smoothness
  - turnover stability
  - confidence-gated penalty
  - bounded intrinsic
- 当前 repo 已经有这条线索：
  - `policy_behavior_summary.json`
  - `td3_action_saturation.json`
  - `action_bound_penalty`
- 下一步不是重新发明指标，而是把这些稳定性约束前移到 selection gate 或 prompt contract。

### 4. Regime / Context
- 当前 prompt 已经写了 `scenario_family`、`regime_vol_ratio_20_60`、`trend_strength_20` 等上下文，但这些更多还是“提示词上下文”，还不是强制的状态机制。
- 更可迁移的做法：
  - 把 regime/context 做成 revised feature groups 的硬要求；
  - 或者按 regime / algorithm 做 expert routing，而不是所有算法共用同一个刷新与 promotion 逻辑。

### 5. Raw-vs-Revised Consistency
- 这是当前项目最特殊、也最关键的一条。
- 对 native backend，真正的研究问题不是“revise_state 能不能带来增益”，而是：
  - `intrinsic_reward(s)` 在 raw native state 上是否已经非平凡；
  - revised dims 是 refinement，还是把所有 alpha 都搬到了额外维度里。
- 当前 run 表明这条约束还没有被稳定满足，尤其在 `td3` 上最明显。

### 6. Training Protocol
- 当前最优先的迁移不是换更大模型或重做 DRL，而是低侵入 protocol：
  - contract hardening
  - candidate prefilter
  - selection gate
  - algorithm-specific refresh
- 原因很直接：
  - 这些改动最便宜；
  - 最容易归因；
  - 最适合先回答“方法问题还是算法问题”。

## Transfer Matrix

| 来源 | 可迁移机制 | 本项目接入点 | 预期收益 | 最小实验 | 主要风险 |
| --- | --- | --- | --- | --- | --- |
| Jiang et al., 2017, DRL for Portfolio Management, <https://arxiv.org/abs/1706.10059> | `Portfolio-Vector Memory (PVM)` | `revise_state` 的 feature-group contract；`portfolio_memory` 候选模板 | 强化 holdings/cash/exposure 的状态连续性 | 只增加 native-safe portfolio memory summary dims，不改 intrinsic | 若直接拼太多 memory dims，可能让 revise-only 更强、进一步压制 intrinsic |
| RL Portfolio Allocation with Dynamic Embedding of Market Information, 2025, <https://arxiv.org/abs/2501.17992> | global market embedding / context token | `prompt contract`、`revise_state` appended context dims、cross-window distillation | 让 regime/context 从提示词变成状态机制 | 仅加 1 组 regime embedding dims，观察 `G2/G3` 是否更一致 | embedding 过强时可能引入 window-specific overfit |
| Risk-Adjusted DRL: Multi-reward Approach, 2025, <https://link.springer.com/article/10.1007/s44196-025-00875-8> | multi-reward experts / action fusion | `candidate_scoring`、final selection、official candidate bundle summary | 降低单 reward 目标偏置，提高稳健性 | 保持训练不变，只在 selection 端做多目标重排 | 若 selection 逻辑过复杂，短期内难归因 |
| Risk-Sensitive DRL for Portfolio Optimization, 2025, <https://econpapers.repec.org/article/gamjjrfmx/v_3a18_3ay_3a2025_3ai_3a7_3ap_3a347-_3ad_3a1684765.htm> | risk-sensitive objective / downside-aware shaping | `intrinsic_postprocess`、risk proxy feature groups、selection metrics | 让 reward 目标更贴近下行保护 | 只加 downside-aware selection gate，不改 env reward | 可能让收益型窗口表现变差，需要分 regime 看 |
| CAPS, 2021, <https://arxiv.org/abs/2012.06644> | action smoothness regularization | `action_bound_penalty`、selection behavior score、prompt contract | 降低抖动、降低 near-bound oscillation | 只在 selection 端增加 smoothness hard gate | 如果 gate 太强，可能把有效激进候选也删掉 |
| Recurrent RL with expected MDD, 2017, <https://www.sciencedirect.com/science/article/abs/pii/S0957417417304402> | downside-risk-aware objective | candidate scoring / confirmatory evaluation slices | 把收益目标改成回撤感知目标 | 仅新增 MDD-aware ranking 列，不改训练 | 与当前 Sharpe 主目标可能冲突，需要并行报表 |
| SB3 RL Tips, <https://stable-baselines3.readthedocs.io/en/v2.7.0/guide/rl_tips.html> | normalization, multi-run evaluation discipline | `state_norm_effective`、per-seed eval、test env 独立评估 | 让“方法问题”和“噪声问题”分开 | 只加强报告，不改训练 | 会增加报表体量，但不会直接提分 |
| TD3 docs, <https://stable-baselines3.readthedocs.io/en/v2.7.0/modules/td3.html> | smoothing bias / continuous control discipline | TD3 branch prompt、behavior gate、delta-weight design | 更贴近连续动作环境的真实约束 | 保持算法不变，只在 prompt 中明确“避免纯 revise-only gain” | prompt-only 改动效果可能有限 |

## 对当前 repo 的研究建议

### 先研究什么
- 先研究 `raw intrinsic 是否独立有效`，不要先研究“更复杂的 revise_state”。
- 先研究 `selection gate 是否能筛掉 revise-only 假阳性`，不要先研究“更复杂的 LLM 采样策略”。
- 先研究 `algorithm-specific refresh / promotion`，不要先强求所有算法共享同一套节奏。

### 暂时不要优先什么
- 不要先重做 DRL backbone。
- 不要先大规模扩展论文列表。
- 不要在同一轮里同时改 prompt、state、reward、selection 四个维度。

## 最小可执行研究单元

| 主题 | 最小实验 | 预期回答的问题 |
| --- | --- | --- |
| prompt contract hardening | 只修 native CoT 的 `source_dim` / contract replay | LLM 是否因为旧 schema feedback 被带偏 |
| raw intrinsic gate | 只新增 `raw intrinsic nontriviality` 硬门槛 | 能否筛掉 `G2 ~= G0` 的候选 |
| smoothness gate | 只提高对 near-bound / turnover 抖动的惩罚 | TD3/SAC 是否更少出现行为复制 |
| regime features | 只增加一组 native-safe regime summary dims | 提升是否来自 context，而不是 revise 维度数增多 |
| multi-objective selection | 只在最终排序里加入 downside-aware 指标 | 是否能减少单窗口偶然最优 |
