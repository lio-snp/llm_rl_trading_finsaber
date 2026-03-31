from __future__ import annotations

import re
from typing import List, Dict, Tuple

from src.llm.deepseek_client import DeepSeekClient


CODE_BLOCK_RE = re.compile(r"```(?:python)?(.*?)```", re.DOTALL | re.IGNORECASE)


def extract_code(text: str) -> str:
    match = CODE_BLOCK_RE.search(text)
    if match:
        return match.group(1).strip()
    # fallback: strip any fenced markers if present
    cleaned = text.replace("```python", "").replace("```", "")
    return cleaned.strip()


def _trim_to_function_block(text: str) -> str:
    cleaned = text.replace("\r\n", "\n").replace("`", "")
    starts = [
        cleaned.find("import numpy as np"),
        cleaned.find("def revise_state"),
        cleaned.find("def intrinsic_reward"),
    ]
    starts = [idx for idx in starts if idx >= 0]
    start = min(starts) if starts else 0
    return cleaned[start:].strip() + "\n"


def extract_lesr_code(text: str) -> str:
    code = extract_code(text)
    if ("def revise_state" in code) and ("def intrinsic_reward" in code):
        return _trim_to_function_block(code)
    # Fallback to full response body if fenced extraction was incomplete/truncated.
    return _trim_to_function_block(text)


def is_valid_code(code: str) -> bool:
    return ("def revise_state" in code) and ("def intrinsic_reward" in code)


def sample_candidates(
    client: DeepSeekClient,
    model: str,
    system_prompt: str,
    user_prompt: str,
    k: int,
    temperature: float,
    max_tokens: int,
    max_retries: int,
) -> Tuple[List[str], List[Dict]]:
    codes: List[str] = []
    raw_responses: List[Dict] = []

    for i in range(k):
        retries = 0
        while retries <= max_retries:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
            content = client.chat(model=model, messages=messages, temperature=temperature, max_tokens=max_tokens)
            raw_responses.append({"index": i, "content": content})
            code = extract_code(content)
            if is_valid_code(code):
                codes.append(code)
                break
            retries += 1
        if retries > max_retries:
            # Keep going; caller may fallback to static candidates
            raw_responses.append({"index": i, "error": "invalid_code_after_retries"})

    return codes, raw_responses


def sample_candidates_from_dialogs(
    client: DeepSeekClient,
    model: str,
    dialogs: List[Dict],
    k: int,
    temperature: float,
    max_tokens: int,
    max_retries: int,
) -> Tuple[List[str], List[Dict]]:
    codes: List[str] = []
    raw_responses: List[Dict] = []

    for i in range(k):
        retries = 0
        while retries <= max_retries:
            content = client.chat(model=model, messages=dialogs, temperature=temperature, max_tokens=max_tokens)
            raw_responses.append({"index": i, "content": content})
            code = extract_lesr_code(content)
            if is_valid_code(code):
                codes.append(code)
                break
            retries += 1
        if retries > max_retries:
            raw_responses.append({"index": i, "error": "invalid_code_after_retries"})

    return codes, raw_responses
