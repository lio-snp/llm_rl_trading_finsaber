from __future__ import annotations

import json
import os
from typing import List, Dict
from urllib import request


def _coerce_bool(value, default: bool = False) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        low = value.strip().lower()
        if low in {"1", "true", "yes", "y", "on"}:
            return True
        if low in {"0", "false", "no", "n", "off"}:
            return False
    return bool(default)


class DeepSeekClient:
    def __init__(self, api_key: str, base_url: str, timeout_s: int = 60, use_env_proxy: bool = False):
        if not api_key:
            raise ValueError("Missing DEEPSEEK_API_KEY")
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout_s = int(max(1, timeout_s))
        self.use_env_proxy = bool(use_env_proxy)
        self._proxy_handler = request.ProxyHandler() if self.use_env_proxy else request.ProxyHandler({})

    def _build_opener(self):
        # Avoid reusing long-lived HTTPS connections through the proxy; stale keep-alive
        # sockets can hang the whole LESR branch when one request stops returning.
        return request.build_opener(self._proxy_handler)

    def chat(self, model: str, messages: List[Dict], temperature: float, max_tokens: int) -> str:
        url = self.base_url
        if not url.endswith("/v1"):
            url = url + "/v1"
        url = url + "/chat/completions"

        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        data = json.dumps(payload).encode("utf-8")
        req = request.Request(
            url,
            data=data,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
                "User-Agent": "llm-rl-trading/lesr",
                "Connection": "close",
            },
            method="POST",
        )
        with self._build_opener().open(req, timeout=self.timeout_s) as resp:
            content = resp.read().decode("utf-8")
        obj = json.loads(content)
        return obj["choices"][0]["message"]["content"]


def from_env(base_url: str, timeout_s: int = 60, use_env_proxy: bool | None = None) -> DeepSeekClient:
    api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    if use_env_proxy is None:
        use_env_proxy = _coerce_bool(os.environ.get("DEEPSEEK_USE_ENV_PROXY"), default=False)
    return DeepSeekClient(
        api_key=api_key,
        base_url=base_url,
        timeout_s=timeout_s,
        use_env_proxy=bool(use_env_proxy),
    )
