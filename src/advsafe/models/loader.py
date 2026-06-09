"""Model loading + generation.

Wraps HuggingFace's `AutoModelForCausalLM` / `AutoTokenizer` with:
  - Device detection (CUDA / MPS / CPU)
  - Optional 4-bit quantization for QLoRA (CUDA only)
  - Chat-template handling that works across the four model families in our panel
  - A clean `generate()` helper that returns a typed `GeneratedResponse`
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

from advsafe.types import GeneratedResponse, GenerationConfig, ModelConfig, ModelHandle
from advsafe.utils.device import get_device, get_dtype
from advsafe.utils.logging import get_logger

if TYPE_CHECKING:
    import torch

logger = get_logger(__name__)


def load_model(
    config: ModelConfig,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
    quantize_4bit: bool | None = None,
) -> ModelHandle:
    """Load a model + tokenizer from HuggingFace.

    Args:
        config: ModelConfig (typically from YAML).
        device: Override device (else auto-detect).
        dtype: Override dtype (else picked based on device).
        quantize_4bit: Override quantization (else uses config.use_quantization).
            4-bit requires bitsandbytes (CUDA-only).

    Returns:
        Loaded ModelHandle.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = device or get_device()
    dtype = dtype or get_dtype(device)
    if quantize_4bit is None:
        quantize_4bit = config.use_quantization

    logger.info(
        "Loading model",
        extra={
            "model": config.name,
            "hf_id": config.hf_id,
            "revision": config.revision,
            "device": str(device),
            "dtype": str(dtype),
            "quantize_4bit": quantize_4bit,
        },
    )

    # Build kwargs for from_pretrained
    model_kwargs: dict[str, Any] = {
        "torch_dtype": dtype,
        "trust_remote_code": config.trust_remote_code,
    }
    if config.revision:
        model_kwargs["revision"] = config.revision

    if quantize_4bit:
        if device.type != "cuda":
            raise RuntimeError(
                "4-bit quantization requires CUDA (bitsandbytes). " f"Current device: {device}"
            )
        try:
            from transformers import BitsAndBytesConfig

            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=dtype,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
        except ImportError as e:
            raise ImportError(
                "bitsandbytes not installed. Install with `pip install -e '.[cuda]'`."
            ) from e
        # Quantized models manage their own device placement
        model_kwargs["device_map"] = "auto"
    else:
        # Non-quantized: explicit device_map or fall back to manual .to()
        if device.type == "cuda":
            model_kwargs["device_map"] = "auto"

    if config.use_flash_attn and device.type == "cuda":
        model_kwargs["attn_implementation"] = "flash_attention_2"

    # Tokenizer
    tokenizer_kwargs = {
        "trust_remote_code": config.trust_remote_code,
    }
    if config.revision:
        tokenizer_kwargs["revision"] = config.revision

    tokenizer = AutoTokenizer.from_pretrained(config.hf_id, **tokenizer_kwargs)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Model
    model = AutoModelForCausalLM.from_pretrained(config.hf_id, **model_kwargs)

    # For non-quantized + non-CUDA paths, manually move to device.
    if not quantize_4bit and device.type != "cuda":
        model = model.to(device)

    model.eval()

    # Record the actual loaded revision (commit SHA) into the manifest.
    revision_sha = getattr(model.config, "_commit_hash", config.revision or "unknown")

    return ModelHandle(
        model=model,
        tokenizer=tokenizer,
        config=config,
        device=device,
        dtype=dtype,
        revision_sha=str(revision_sha),
    )


def _format_chat(
    handle: ModelHandle,
    user_message: str,
    system_prompt: str | None = None,
) -> str:
    """Apply the model's chat template to format a single turn.

    Handles family-specific quirks:
      - Llama 3.x: ChatML-like, special <|begin_of_text|> handled by tokenizer
      - Qwen 2.5/3: similar ChatML
      - Gemma 2/3: no system role; we prepend system content to the user message
      - DeepSeek-R1-Distill: ChatML with optional <think> tags
    """
    messages: list[dict[str, str]] = []

    family = handle.config.family.lower()
    if family == "gemma":
        # Gemma doesn't support a system role; merge into user turn.
        user_content = user_message
        if system_prompt:
            user_content = f"{system_prompt}\n\n{user_message}"
        messages.append({"role": "user", "content": user_content})
    else:
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_message})

    return handle.tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def generate(
    handle: ModelHandle,
    prompt: str,
    gen_config: GenerationConfig | None = None,
    system_prompt: str | None = None,
    prompt_id: str = "",
) -> GeneratedResponse:
    """Run one generation through the model.

    Args:
        handle: Loaded model handle.
        prompt: The user prompt.
        gen_config: Generation parameters; defaults to greedy decoding.
        system_prompt: Optional system message.
        prompt_id: Stable identifier (passed through to the response).

    Returns:
        GeneratedResponse with timing and token counts.
    """
    import torch
    from transformers import GenerationConfig as HFGenerationConfig

    gen_config = gen_config or GenerationConfig()

    formatted = _format_chat(handle, prompt, system_prompt=system_prompt)

    inputs = handle.tokenizer(
        formatted,
        return_tensors="pt",
        add_special_tokens=False,  # template already includes them
    ).to(handle.device)
    n_input = inputs["input_ids"].shape[1]

    hf_gen = HFGenerationConfig(
        max_new_tokens=gen_config.max_new_tokens,
        temperature=gen_config.temperature if gen_config.do_sample else 1.0,
        top_p=gen_config.top_p,
        top_k=gen_config.top_k,
        do_sample=gen_config.do_sample,
        repetition_penalty=gen_config.repetition_penalty,
        pad_token_id=handle.tokenizer.pad_token_id,
        eos_token_id=handle.tokenizer.eos_token_id,
    )

    torch.manual_seed(gen_config.seed)
    t0 = time.perf_counter()
    with torch.inference_mode():
        output = handle.model.generate(**inputs, generation_config=hf_gen)
    elapsed = time.perf_counter() - t0

    # Slice off the input portion
    new_tokens = output[0, n_input:]
    response_text = handle.tokenizer.decode(new_tokens, skip_special_tokens=True)
    n_output = new_tokens.shape[0]

    # Finish reason heuristic
    finish_reason = "length" if n_output >= gen_config.max_new_tokens else "stop"

    return GeneratedResponse(
        prompt_id=prompt_id,
        prompt=prompt,
        response=response_text.strip(),
        elapsed_seconds=elapsed,
        n_input_tokens=int(n_input),
        n_output_tokens=int(n_output),
        finish_reason=finish_reason,
        metadata={
            "model_name": handle.config.name,
            "revision_sha": handle.revision_sha,
            "system_prompt_present": system_prompt is not None,
        },
    )


def generate_batch(
    handle: ModelHandle,
    prompts: list[tuple[str, str]],
    gen_config: GenerationConfig | None = None,
    system_prompt: str | None = None,
) -> list[GeneratedResponse]:
    """Sequential batch generation.

    Args:
        prompts: List of (prompt_id, prompt_text) tuples.

    Note: This is intentionally sequential for simplicity and determinism.
    For real throughput in the cloud sweep, use vLLM via `runners.sweep`.
    """
    results = []
    for pid, p in prompts:
        results.append(
            generate(handle, p, gen_config=gen_config, system_prompt=system_prompt, prompt_id=pid)
        )
    return results
