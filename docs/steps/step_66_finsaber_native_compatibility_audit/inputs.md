# Inputs

## User Intent
- 对 `llm_rl_trading_finsaber` 做一次 FINSABER native DRL x LESR 的“契约级”兼容审计。
- 把代码、当前 run、以及用户提供的聊天截图整理成研究判断和下一轮可执行 backlog。
- 本轮默认不改代码行为，只沉淀结论。

## Primary Code Scope
- `llm_rl_trading_finsaber/src/finsaber_native/`
- `llm_rl_trading_finsaber/src/env/`
- `llm_rl_trading_finsaber/src/lesr/`
- `llm_rl_trading_finsaber/src/pipeline/demo.py`
- `llm_rl_trading_finsaber/src/pipeline/branch_iteration_worker.py`
- `复现/FINSABER-main/rl_traders/finrl/`
- `llm_rl_trading/src/lesr/`
- `llm_rl_trading/src/pipeline/demo.py`

## Primary Run Scope
- `llm_rl_trading_finsaber/runs/20260330_170843_316_7504_demo/wf_window_00/prompt.txt`
- `llm_rl_trading_finsaber/runs/20260330_170843_316_7504_demo/wf_window_00/system_prompt.txt`
- `llm_rl_trading_finsaber/runs/20260330_170843_316_7504_demo/wf_window_00/llm_errors.json`
- `llm_rl_trading_finsaber/runs/20260330_170843_316_7504_demo/wf_window_00/metrics.json`
- `llm_rl_trading_finsaber/runs/20260330_170843_316_7504_demo/wf_window_00/policy_behavior_summary.json`
- `llm_rl_trading_finsaber/runs/20260330_170843_316_7504_demo/wf_window_00/td3_action_saturation.json`
- `llm_rl_trading_finsaber/runs/20260330_170843_316_7504_demo/wf_window_00/run_summary.md`

## Screenshot-Derived Research Claims
Based on the user-provided chat screenshots in this thread, the current working hypotheses are:

1. `DRL 本身就有波动和不稳定`，效果不好不一定异常。
2. `效果差更可能是方法问题，不是 DRL 必然不适合投资`。
3. `先把 baseline 跑稳，再讨论怎么围绕 DRL 择时继续优化`。
4. `稀疏/择时不是唯一关键`，滚动窗口、样本选择、刷新频率也可能解释效果差。

These claims are treated as hypotheses to test against repo evidence, not as settled facts.
