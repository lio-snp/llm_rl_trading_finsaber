# FINSABER Native DRL x LESR 兼容性审计报告

## Executive Summary
- `llm_rl_trading_finsaber` 当前并不是单一后端，而是三套语义并存：
  - `current`: 通用 `StateSchema + TradingEnv + TradingGymEnv`
  - `finsaber_native`: 保留 FINSABER native raw state layout，但加上 LESR 钩子
  - `finsaber_compat`: 兼容层，当前仍是 phase-1 baseline-only
- native prompt 已经部分适配：最终落盘的 `prompt.txt` 已明确写入 native contract note，要求禁止 generic OHLCV 假设，并说明 raw-state fallback 约束。
- 但 LESR 迁移仍是“前半段 native 化，后半段仍残留旧 schema 模型”：
  - prompt base 已切换成 native `state_desc + state_contract_note`
  - 迭代 CoT 的 `source_dim` 和若干维度统计仍使用 `schema.dim()`
  - system prompt 仍是后端无感知的通用模板，native 约束主要依赖配置里的 `system_prompt_extra`
- 当前 run 已出现明显的“路径失衡”证据：
  - `td3` 最佳候选的 `intrinsic_probe_delta_sharpe = 0.0`
  - `td3` 的 `G2_intrinsic_only` 行为统计几乎复制 `G0_baseline`
  - `td3` 的 `G3_revise_intrinsic` 行为统计几乎复制 `G1_revise_only`
  - 这说明至少在该窗口里，best candidate 的有效提升几乎全部来自 revise/state path，而 intrinsic path 基本没有独立贡献

## 三个核心问题的答案

### 1. native DRL 的真实 `state/action/reward` 契约是什么
- raw state 不是旧 generic LESR 的 `global + OHLCV + holding + indicators`。
- `finsaber_native` raw state 是：
  - `cash`
  - `close-price block`
  - `holdings block`
  - `indicator-major blocks`
- native action 不是旧 `current` 后端的离散组合动作，也不是 generic schema 驱动的 per-asset OHLCV state 上动作；它继承 FINSABER 风格的连续交易语义，再由 native env 处理成交和 portfolio value 更新。
- native reward 继承 FINSABER 风格的环境收益语义，再在 LESR/Gym 包装层叠加 intrinsic 相关逻辑。

### 2. 旧 `llm_rl_trading` 的 LESR 假设还残留在哪里
- system prompt 仍保持后端无感知的通用叙述。
- build_cot_prompt 的 `source_dim` 仍沿用 `schema.dim()`。
- revised-dim 相关诊断仍以 generic schema 为基线。
- search context 虽然加入了 `drl_backend=finsaber_native`，但仍保留大量通用 LESR framing，没有把 native contract 的字段边界反复固化到后续反馈环。

### 3. 这些残留是否已经在当前 run 中表现为问题
- 是，且已经体现在：
  - LLM validation failure 中大量出现 native sample 上的 `revise_state_exception`、raw/revised 分支错位、越界索引和非有限值；
  - `td3` 最佳候选的 intrinsic 独立贡献接近零；
  - `td3` 的行为统计呈现 `G2 ~= G0`、`G3 ~= G1`；
  - run 结果说明“候选可运行”并不等于“intrinsic 路径真的在 native backend 上生效”。

## State Contract Diff

| 视角 | 维度公式 | 字段顺序 | observation 语义 | action 语义 | reward 语义 | 对 LESR 的直接含义 |
| --- | --- | --- | --- | --- | --- | --- |
| 旧 `llm_rl_trading` generic LESR schema | `len(global_features) + len(assets) * (6 + len(indicators))` | `global + [open, high, low, close, volume, holding] + indicators` | 默认 policy 看到的就是这套 schema 或其 revise 版本 | 离散/连续取决于算法，但 state 解释以 generic schema 为准 | generic env PnL 语义 | prompt/候选函数天然会假设 OHLCV 在 state 里显式存在 |
| 当前 `llm_rl_trading_finsaber` `current` backend | 同上 | 同上 | `policy_state_fn` 或 `revise_state` 还能进一步改写 obs | A2C/PPO 可离散，SAC/TD3 为连续股数尺度动作 | env reward + `intrinsic_w * intrinsic` - penalty | generic LESR 仍适用，但已不是原 FINSABER 语义 |
| 当前 `llm_rl_trading_finsaber` `finsaber_native` backend | `1 + 2 * stock_dim + len(indicators) * stock_dim` | `cash + close block + holdings block + indicator-major blocks` | raw native state 可再被 `policy_state_fn` / `use_revised` 改写 | native continuous trading semantics | native env reward，再叠加 LESR 相关 intrinsic/trace | prompt 必须把 close/holding/indicator block 当作 authoritative prefix，不可按 OHLCV 理解 |
| 原始 `FINSABER-main` native env | 与当前 native contract 一致 | 与当前 native contract 一致 | obs 基本就是原生 state | FINSABER 原始 continuous action handling | FINSABER 原始 reward/update 逻辑 | 当前 native raw contract 的源头 |

## LESR Mismatch List

| 严重度 | 类别 | 不适配点 | 影响 | 主要证据 |
| --- | --- | --- | --- | --- |
| 严重 | prompt 假设 | native 路径下迭代 CoT 的 `source_dim` 仍来自 `schema.dim()`，不是 native raw dim | 后续 Lipschitz 反馈、额外维度边界、代码分析对象全部可能错位 | `src/pipeline/demo.py` 中 native 分支生成 `raw_state_dim = native_contract.state_dim`，但 `build_cot_prompt(..., schema.dim(), ...)` 仍传 generic dim |
| 严重 | prompt 假设 | native 指标来源实际走 `execution.finsaber_native.tech_indicator_list` / FinRL 默认指标，generic `schema` 仍绑定 `cfg.indicators` | 只要后续逻辑仍引用 schema，就会和真实 native state 不一致 | `src/pipeline/demo.py` 的 native cfg 解析与 `StateSchema(...)` 构造并行存在 |
| 高 | 候选代码验证 | revised dim delta 等诊断统计继续以 generic schema 做基线 | 搜索日志和候选解释会被污染，难以判断 candidate 到底新增了哪些 native-safe 维度 | `src/pipeline/demo.py` / `src/pipeline/branch_iteration_worker.py` 仍保留 schema-driven revised-dim 统计 |
| 高 | prompt 假设 | system prompt 没有代码级 native 绑定，native 对齐主要靠配置 `system_prompt_extra` | 换配置或新 run 时容易回退到旧通用 framing | `src/lesr/prompt_templates.py` + `runs/.../config.yaml` |
| 中 | 算法分支偏置 | branch prompt 已提示 backend-specific state semantics，但 search context 仍以通用 LESR 机制为主 | LLM 容易回到旧 representation 心智模型，特别是迭代反馈阶段 | `src/pipeline/demo.py` 的 `_build_common_search_context()` 与 `_algo_branch_instruction()` |
| 中 | 候选代码验证 | native-safe fallback candidate 库缺位 | LLM 采样失败时，native 路径只能退到 identity/fallback，搜索多样性下降 | `src/lesr/revision_candidates.py` 在 native 场景下基本不构成等价候选库 |
| 中 | intrinsic 生效性 | 当前 scorer 虽强调 raw-policy probe，但最佳 native candidate 仍可能让 intrinsic 形同虚设 | 容易把“revise_state 生效”误判成“LESR 全路径生效” | `metrics.json`、`policy_behavior_summary.json`、`td3_action_saturation.json` |
| 低 | 迁移完整度 | `finsaber_compat` 仍是 baseline-only，未进入 LESR 迁移面 | 兼容层不在本轮主审计面，但说明迁移尚未全覆盖 | `src/pipeline/demo.py` baseline-only 路径 |

## Evidence Table

| 命题 | 证据类型 | 结论 | 证据位置 |
| --- | --- | --- | --- |
| native raw state 已与 generic schema 分叉 | 代码证据 | 成立 | `src/finsaber_native/state_contract.py`, `src/env/state_schema.py` |
| prompt 最终落盘文本已禁止 generic OHLCV 假设 | 运行证据 | 成立 | `runs/20260330_170843_316_7504_demo/wf_window_00/prompt.txt` |
| native prompt 适配已进入 initial prompt | 代码证据 | 成立 | `src/pipeline/demo.py` 中 `state_desc = native_contract.describe_compact()` 和 `state_contract_note = native_contract.prompt_note()` |
| system prompt 仍是后端无感知的旧模板 | 代码证据 | 基本成立 | `src/lesr/prompt_templates.py` 的 `build_system_prompt()` 仍是通用 LESR 叙述 |
| CoT 反馈链路仍残留旧 schema.dim 假设 | 代码证据 | 成立 | `src/pipeline/demo.py` 中 `build_cot_prompt(..., schema.dim(), ...)` |
| validation failure 已出现 native 特有错位症状 | 运行证据 | 成立 | `runs/.../llm_errors.json` 中 `revise_state_exception_native_sample_0` 6 次，`intrinsic_exception_native_raw_sample_0` 2 次，`intrinsic_exception_native_revised_sample_0` 2 次 |
| `td3` 最佳候选的 intrinsic path 基本无独立贡献 | 运行证据 | 成立 | `metrics.json` 中 best TD3 candidate: `state_probe_delta_sharpe = 0.1023`, `intrinsic_probe_delta_sharpe = 0.0`, `intrinsic_signal_nontrivial_raw = False` |
| `td3` 已出现 `G2 ~= G0`、`G3 ~= G1` 的行为复制 | 运行证据 | 成立 | `td3_action_saturation.json`: `G0` 与 `G2` 的 `near_actor_ratio_mean = 0.0351`; `G1` 与 `G3` 的 `near_actor_ratio_mean = 0.1030` |
| `DRL 本身不稳定` | 运行证据 | 部分支持 | 当前窗口内算法间差异和 seed 行为差异较大，但单窗口不足以证明“DRL 普遍不稳定” |
| `问题可能在方法而非 DRL 是否适合投资` | 证据判断 | 支持度较高 | `ppo` 和部分 `sac` 候选仍能得到非零 intrinsic/state 增益，说明不是 DRL 全面失效，而是 native LESR 适配质量不均匀 |
| `先 baseline 再优化` | 证据判断 | 支持 | 当前 run 的关键问题不是“完全跑不起来”，而是“路径归因不清、intrinsic 未独立生效”，先 baseline/contract 清理更合理 |
| `滚动窗口与样本选择也要解释` | 证据判断 | 支持 | run 使用 walk-forward + scenario family + per-algo branches，结果已明显受窗口与 family 影响，不能只把问题归到稀疏择时 |

## 当前 Run 的算法级证据摘要

| 算法 | 最佳候选 | design_mode | `performance_delta_sharpe` | `state_probe_delta_sharpe` | `intrinsic_probe_delta_sharpe` | raw intrinsic 非平凡 | 结论 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `a2c` | `a2c_llm_it5_mean_revert_k2` | `state_first` | `1.7261` | `-0.3249` | `-0.0149` | 是 | 提升主要来自 revise/state path，intrinsic 没有独立帮助 |
| `ppo` | `ppo_llm_it0_mean_revert_k2` | `state_first` | `0.0435` | `0.0455` | `0.4829` | 是 | 当前窗口里 intrinsic 独立贡献最明确 |
| `sac` | `sac_llm_it4_risk_shield_k0` | `intrinsic_first` | `0.3729` | `0.3463` | `-0.0020` | 是 | 名义上是 intrinsic-first，但独立 intrinsic 增益几乎为零 |
| `td3` | `td3_llm_it1_trend_follow_k1` | `balanced` | `0.1024` | `0.1023` | `0.0000` | 否 | 典型的 `G1/G3` 有效、`G2` 失效，说明 balanced 名义下仍是 revise 驱动 |

## 审计结论

### 结论 1
native prompt 不是完全没适配，而是“落盘 prompt 已适配，迭代反馈链还没适配干净”。

### 结论 2
当前最大的结构性问题不是候选代码完全无效，而是 native backend 下：
- state contract 已切换；
- obs contract 已可重写；
- intrinsic input 可 raw/revised 分叉；
- 但 LESR 的部分解释和反馈环仍在按旧 schema 思考。

### 结论 3
在当前窗口里，最值得优先修的不是“继续发明更复杂的 intrinsic 公式”，而是：
- 把 prompt/CoT/诊断链条中的 generic schema 假设清掉；
- 明确区分 state path 和 intrinsic path 的独立贡献；
- 防止 `balanced` / `intrinsic_first` 候选最终仍退化为 revise-only 的假阳性。
