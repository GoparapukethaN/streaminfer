import pytest

from streaminfer.synthetic_llm import SyntheticLLMModel, estimate_tokens


def test_estimate_tokens_uses_plain_whitespace_tokenization() -> None:
    assert estimate_tokens("batching improves throughput") == 3
    assert estimate_tokens("") == 0


@pytest.mark.asyncio
async def test_synthetic_llm_reports_token_and_batch_metadata() -> None:
    model = SyntheticLLMModel(base_latency_ms=0, per_token_latency_ms=0, output_tokens=12)

    results = await model.predict([
        {"text": "short prompt"},
        {"prompt": "longer prompt with more tokens"},
    ])

    assert results == [
        {
            "result": "synthetic completion for: short prompt",
            "model": "synthetic-llm",
            "prompt_tokens": 2,
            "completion_tokens": 12,
            "total_tokens": 14,
            "batch_size": 2,
        },
        {
            "result": "synthetic completion for: longer prompt with more tokens",
            "model": "synthetic-llm",
            "prompt_tokens": 5,
            "completion_tokens": 12,
            "total_tokens": 17,
            "batch_size": 2,
        },
    ]
