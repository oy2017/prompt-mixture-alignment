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
    # Pro thinks longer than flash (probed: visible output <= 264 tokens, 0/10
    # truncations at 8000; 4000 leaves ample headroom). Extraction stays on
    # flash — it is a trivial integer-extraction task and pro is ~4x slower.
    "gemini-pro": dict(
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        key_env="GEMINI_API_KEY",
        gen="gemini-pro-latest", extract="gemini-3.5-flash", max_tokens=4000),
}

PROVIDER = os.getenv("PMA_PROVIDER", "openrouter")
_cfg = PROVIDERS[PROVIDER]

client = OpenAI(base_url=_cfg["base_url"], api_key=os.environ[_cfg["key_env"]])
GEN_MODEL = os.getenv("PMA_GEN_MODEL", _cfg["gen"])
EXTRACT_MODEL = os.getenv("PMA_EXTRACT_MODEL", _cfg["extract"])
GEN_MAX_TOKENS = _cfg["max_tokens"]


def api_call(model, messages, max_tokens=None, retries=6,
             quota_wait_s=600, quota_patience_s=8 * 3600):
    """One chat completion with two-tier retry: exponential backoff on
    transient errors, and long sleeps on daily-quota exhaustion (Gemini free
    tiers reset at midnight PT) so unattended sweeps survive rather than crash.
    """
    import time as _t
    quota_spent = 0
    attempt = 0
    while True:
        try:
            completion = client.chat.completions.create(
                model=model, messages=messages, n=1,
                max_tokens=max_tokens or GEN_MAX_TOKENS)
            if completion.choices[0].message.content:
                return completion
            raise RuntimeError("empty completion")
        except Exception as e:
            msg = str(e)
            if "RESOURCE_EXHAUSTED" in msg or "exceeded your current quota" in msg:
                if quota_spent >= quota_patience_s:
                    raise
                _t.sleep(quota_wait_s)
                quota_spent += quota_wait_s
                continue
            attempt += 1
            if attempt >= retries:
                raise
            _t.sleep(2 ** attempt)
