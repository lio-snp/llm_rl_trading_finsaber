from __future__ import annotations

from types import SimpleNamespace


def load_functions_from_code(code: str):
    scope = {}
    exec(code, scope)
    revise_state = scope.get("revise_state")
    intrinsic_reward = scope.get("intrinsic_reward")
    if revise_state is None or intrinsic_reward is None:
        raise ValueError("Code must define revise_state and intrinsic_reward")
    return revise_state, intrinsic_reward
