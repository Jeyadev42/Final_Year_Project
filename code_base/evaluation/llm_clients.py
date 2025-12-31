from __future__ import annotations

from functools import partial
from typing import Callable, Dict

import requests


def run_local_llm(prompt: str, model: str = "gemma3:4b", timeout: int = 120) -> str:
    """
    Run a local Ollama LLM (gemma3:4b).
    """
    try:
        resp = requests.post(
            "http://localhost:11434/api/chat",
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
            },
            timeout=timeout,
        )
        response_json = resp.json()
        return response_json["message"]["content"].strip()
    except Exception as exc:
        return f"(Local LLM error: {exc})"


def _run_openai_compatible_llm(
    prompt: str,
    *,
    api_key: str,
    base_url: str,
    model: str,
    timeout: int = 60,
) -> str:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
    }
    try:
        resp = requests.post(
            f"{base_url.rstrip('/')}/chat/completions",
            json=payload,
            headers=headers,
            timeout=timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"].strip()
    except Exception as exc:
        return f"(Eval LLM error: {exc})"


def build_eval_llm(cfg: Dict[str, str]) -> Callable[[str], str]:
    """
    Return an evaluation LLM function based on config.

    Supported providers:
      - local (default)
      - openai (GPT-4o-mini)
      - deepseek (DeepSeek R1)
    """
    provider = cfg.get("EVAL_LLM_PROVIDER", "local").strip().lower()

    if provider in {"openai", "gpt-4o-mini", "gpt4o-mini"}:
        api_key = cfg.get("OPENAI_API_KEY") or cfg.get("EVAL_LLM_API_KEY")
        if not api_key:
            return run_local_llm
        base_url = cfg.get("EVAL_LLM_BASE_URL", "https://api.openai.com/v1")
        model = cfg.get("EVAL_LLM_MODEL", "gpt-4o-mini")
        return partial(
            _run_openai_compatible_llm,
            api_key=api_key,
            base_url=base_url,
            model=model,
        )

    if provider in {"deepseek", "deepseek-r1", "deepseek_r1"}:
        api_key = cfg.get("DEEPSEEK_API_KEY") or cfg.get("EVAL_LLM_API_KEY")
        if not api_key:
            return run_local_llm
        base_url = cfg.get("EVAL_LLM_BASE_URL", "https://api.deepseek.com/v1")
        model = cfg.get("EVAL_LLM_MODEL", "deepseek-r1")
        return partial(
            _run_openai_compatible_llm,
            api_key=api_key,
            base_url=base_url,
            model=model,
        )

    return run_local_llm
