"""Backbone selection shared by the fitting and evaluation scripts.

Pick a backbone with PMA_PROVIDER (default: the paper's GPT-4o via OpenRouter);
override individual models with PMA_GEN_MODEL / PMA_EXTRACT_MODEL.
"""

import os

from openai import OpenAI

PROVIDERS = {
    # Backbone used by the paper (exact snapshots).
    "openrouter": dict(
        base_url="https://openrouter.ai/api/v1", key_env="OPENROUTER_API_KEY",
        gen="openai/gpt-4o-2024-05-13", extract="openai/gpt-4o-mini-2024-07-18",
        max_tokens=350),
    # gemini-3.5-flash is a thinking model: hidden reasoning tokens are charged
    # against max_tokens while usage.completion_tokens reports only visible
    # output (median 91, max 212 here). Sizing the cap from that number
    # truncates before the bracketed answer -- measured 100% truncation at 350,
    # 96% at 500, 0/240 at 2000.
    "gemini": dict(
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        key_env="GEMINI_API_KEY",
        gen="gemini-3.5-flash", extract="gemini-3.5-flash", max_tokens=2000),
}

PROVIDER = os.getenv("PMA_PROVIDER", "openrouter")
_cfg = PROVIDERS[PROVIDER]

client = OpenAI(base_url=_cfg["base_url"], api_key=os.environ[_cfg["key_env"]])
GEN_MODEL = os.getenv("PMA_GEN_MODEL", _cfg["gen"])
EXTRACT_MODEL = os.getenv("PMA_EXTRACT_MODEL", _cfg["extract"])
GEN_MAX_TOKENS = _cfg["max_tokens"]
