# 优先级改造 Backlog

## Baseline 先跑

### 1. 修正 native CoT 的 `source_dim`
- 目标：让后续 LLM 反馈链使用真实 native raw dim，而不是 generic `schema.dim()`
- 改动点：
  - `src/pipeline/demo.py`
  - `src/pipeline/branch_iteration_worker.py`
  - 与 `build_cot_prompt()` 相关的 source-dim 传参
- 成功信号：
  - CoT 文本中的 `source state dim` 与 native contract 一致
  - revised-dim 边界不再混入 generic schema 假设
- 风险：低，属于纯契约修正

### 2. 把 native contract 绑定进 system prompt
- 目标：把 native 约束从配置补丁升级为代码级默认行为
- 改动点：
  - `src/lesr/prompt_templates.py`
  - `src/pipeline/demo.py`
- 成功信号：
  - 不依赖 `system_prompt_extra` 也不会回退到 generic OHLCV 叙述
- 风险：低，主要是 prompt 行为变化

### 3. 为 native run 输出一页 `path attribution` 摘要
- 目标：每个算法都显式报告 `state_probe`、`intrinsic_probe`、`joint`
- 改动点：
  - `metrics.json` 汇总逻辑
  - `run_summary.md`
- 成功信号：
  - 一眼看出 `G1-only`、`G2-only`、`G3-joint` 的真实贡献
- 风险：低，只是报表增强

## 轻改可试

### 4. 新增 `raw intrinsic nontriviality` 硬门槛
- 目标：直接过滤 `intrinsic_reward(s)` 在 raw native state 上近乎零贡献的候选
- 改动点：
  - candidate prefilter
  - final selection gate
- 成功信号：
  - `td3`/`sac` 中 `intrinsic_probe_delta_sharpe ~= 0` 的最佳候选显著减少
- 风险：中，可能错杀一些 joint-only 候选

### 5. 增加 native-safe fallback candidate 库
- 目标：减少 LLM 采样失败时 native 路径退化为 identity 的概率
- 改动点：
  - `src/lesr/revision_candidates.py`
  - native validator 兼容清单
- 成功信号：
  - `llm_errors.json` 中 native validation failure 不再主导候选供给
- 风险：中，需要严格控制 prefix-preserving 约束

### 6. 把 smoothness / turnover 约束前移到 selection gate
- 目标：避免“分数变了但动作没变”或“动作更糟但勉强提分”的候选晋级
- 改动点：
  - `candidate_scoring`
  - `policy_behavior_summary` 相关阈值
- 成功信号：
  - `td3_action_saturation.json` 中 `G2 ~= G0`、`G3 ~= G1` 型复制减少
- 风险：中，阈值过严可能抑制正常探索

### 7. 按算法分离 refresh / promotion 节奏
- 目标：不要强迫 `a2c/ppo/sac/td3` 共用一个 LESR 更新节奏
- 改动点：
  - branch schedule
  - per-algo promotion policy
- 成功信号：
  - `ppo` 与 `td3` 不再被同一套更新频率绑死
- 风险：中，实验矩阵会变大

## 中期重构

### 8. 把 `portfolio_memory + regime + dispersion + running_risk_state` 升级为一等 revised contract
- 目标：从“提示词建议”升级为“候选设计与评估的一等接口”
- 改动点：
  - prompt contract
  - fallback candidate families
  - feature-group audit
- 成功信号：
  - 候选差异更多来自机制差异，而不是杂乱的手工索引技巧
- 风险：中到高，需要统一 prompt、validator、selection

### 9. 引入多目标 selection，而不是单目标排序
- 目标：把收益、下行风险、稳定性拆开排序，再做融合
- 改动点：
  - final selection
  - official candidate bundle summary
  - walk-forward report
- 成功信号：
  - 单窗口偶然最优对最终 winner 的影响下降
- 风险：高，排序逻辑会更复杂

### 10. 把动作设计从“直接 target”推向“delta-weight / budget / horizon”
- 目标：减少连续动作环境中的无效抖动和边界压力
- 改动点：
  - env action semantics
  - penalty design
  - policy behavior diagnostics
- 成功信号：
  - action entropy、turnover、near-bound 三者更一致
- 风险：高，这已经触及行为语义，不适合与其他大改并行

### 11. 准备 offline warm-start / historical winner distillation
- 目标：减少纯在线 LESR 搜索的高方差
- 改动点：
  - candidate library
  - historical winner replay
  - distillation artifacts
- 成功信号：
  - 新窗口初始候选质量更高、无效采样更少
- 风险：高，需要单独设计数据边界和防泄漏规则

## 推荐执行顺序
1. `修 source_dim`
2. `system prompt 原生绑定`
3. `path attribution 报表`
4. `raw intrinsic nontriviality gate`
5. `smoothness / turnover gate`
6. `per-algo refresh`
7. `native-safe fallback candidates`
8. `一等 revised contract`
9. `多目标 selection`
10. `动作语义重构`
11. `offline warm-start`
