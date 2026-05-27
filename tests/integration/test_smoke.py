"""Integration smoke test.

Slow — requires downloading a small model. Run with:
    pytest tests/integration -v -m "not slow"   # skips this
    pytest tests/integration -v                  # runs everything
"""

from __future__ import annotations

import os

import pytest

requires_network = pytest.mark.skipif(
    os.environ.get("HF_HUB_OFFLINE") == "1",
    reason="HF_HUB_OFFLINE set; skipping tests that download models",
)


@pytest.mark.slow
@pytest.mark.integration
@requires_network
def test_can_load_tiny_model_and_generate():
    """Load a tiny LM and generate one token. End-to-end smoke."""
    from advsafe.models.loader import generate, load_model
    from advsafe.types import GenerationConfig, ModelConfig

    cfg = ModelConfig(
        name="tiny-gpt2",
        hf_id="sshleifer/tiny-gpt2",
        family="llama",  # tokenizer.apply_chat_template will be missing; the loader handles
    )

    handle = load_model(cfg)
    response = generate(
        handle,
        prompt="Hello",
        gen_config=GenerationConfig(max_new_tokens=4, do_sample=False),
    )
    assert response.n_output_tokens > 0
    assert response.elapsed_seconds >= 0
