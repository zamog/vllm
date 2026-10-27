# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PrefixLMScheduler through LLM."""

import pytest

from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt
from vllm.platforms import current_platform

MODEL = "Qwen/Qwen3.5-0.8B"


@pytest.mark.skipif(
    not (current_platform.is_cuda() or current_platform.is_cpu()),
    reason="PrefixLMScheduler is validated on CUDA and CPU only",
)
def test_a_prompt_no_step_can_prefill_fails(monkeypatch: pytest.MonkeyPatch):
    """A PrefixLM prompt is prefilled whole, so one longer than a step fails
    with an error instead of holding the queue."""
    if current_platform.is_cuda():
        # Its Mamba prefill checkpoints need Model Runner V2.
        monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    llm = LLM(
        model=MODEL,
        max_model_len=2048,
        max_num_batched_tokens=256,
        enable_prefix_caching=True,
        scheduler_cls="vllm.v1.core.sched.prefix_lm_scheduler.PrefixLMScheduler",
        enforce_eager=True,
    )
    too_long = TokensPrompt(prompt_token_ids=list(range(1000, 1300)))
    [output] = llm.generate(
        too_long,
        SamplingParams(max_tokens=4, extra_args={"prefix_lm_len": 100}),
        use_tqdm=False,
    )
    assert output.finished
    assert output.outputs[0].finish_reason == "error"

    [output] = llm.generate(
        TokensPrompt(prompt_token_ids=list(range(1000, 1100))),
        SamplingParams(max_tokens=4),
        use_tqdm=False,
    )
    assert len(output.outputs[0].token_ids) == 4
