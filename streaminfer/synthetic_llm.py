"""Deterministic LLM-style model profile for local benchmark sweeps."""

from __future__ import annotations

import asyncio


def estimate_tokens(text: str) -> int:
    """Estimate token count with a transparent whitespace tokenizer."""
    return len(text.split())


class SyntheticLLMModel:
    """Async model that simulates prompt/completion token work without external APIs."""

    def __init__(
        self,
        *,
        base_latency_ms: float = 8,
        per_token_latency_ms: float = 0.25,
        output_tokens: int = 32,
    ) -> None:
        self.base_latency_ms = base_latency_ms
        self.per_token_latency_ms = per_token_latency_ms
        self.output_tokens = output_tokens

    async def predict(self, inputs: list[dict]) -> list[dict]:
        prompts = [_extract_prompt(item) for item in inputs]
        prompt_token_counts = [estimate_tokens(prompt) for prompt in prompts]
        max_tokens = max(prompt_token_counts, default=0) + self.output_tokens
        delay_ms = self.base_latency_ms + (max_tokens * self.per_token_latency_ms)
        if delay_ms > 0:
            await asyncio.sleep(delay_ms / 1000)

        return [
            {
                "result": f"synthetic completion for: {prompt}",
                "model": "synthetic-llm",
                "prompt_tokens": prompt_tokens,
                "completion_tokens": self.output_tokens,
                "total_tokens": prompt_tokens + self.output_tokens,
                "batch_size": len(inputs),
            }
            for prompt, prompt_tokens in zip(prompts, prompt_token_counts)
        ]


def _extract_prompt(item: dict) -> str:
    return str(item.get("prompt") or item.get("text") or "")
