# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The Decision 3.0 config hook that caches prefix_bidir state blocks."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from vllm.model_executor.layers.mamba.gdn import qwen_gdn_linear_attn
from vllm.model_executor.models import config as model_configs
from vllm.model_executor.models.config import D3ForSequenceClassificationConfig
from vllm.platforms import current_platform
from vllm.v1.attention.backends.registry import AttentionBackendEnum

pytestmark = pytest.mark.cpu_test

BIDIRECTIONAL = "vllm.v1.core.sched.prefix_lm_scheduler.BidirectionalPrefixLMScheduler"


def vllm_config(
    attention_mode="prefix_bidir",
    caching=True,
    max_num_batched_tokens=8192,
    user_budget=False,
    ssm_dtype="auto",
    **overrides,
):
    config = SimpleNamespace(
        model_config=SimpleNamespace(
            architecture="D3ForSequenceClassification",
            max_model_len=65536,
            hf_config=SimpleNamespace(
                decision_config={"attention_mode": attention_mode}
            ),
            hf_text_config=SimpleNamespace(),
        ),
        cache_config=SimpleNamespace(
            enable_prefix_caching=caching,
            mamba_ssm_cache_dtype=ssm_dtype,
            kv_offloading_size=None,
            prefix_match_unit=None,
        ),
        scheduler_config=SimpleNamespace(
            enable_chunked_prefill=False,
            long_prefill_token_threshold=4096,
            scheduler_cls=None,
            max_num_batched_tokens=max_num_batched_tokens,
            max_num_scheduled_tokens=None,
            max_num_encoder_input_tokens=max_num_batched_tokens,
            encoder_cache_size=max_num_batched_tokens,
            user_specified_max_num_batched_tokens=user_budget,
        ),
        parallel_config=SimpleNamespace(
            prefill_context_parallel_size=1,
            decode_context_parallel_size=1,
            use_ubatching=False,
        ),
        attention_config=SimpleNamespace(
            backend=AttentionBackendEnum.FLASH_ATTN, flash_attn_version=4
        ),
        kv_transfer_config=None,
        speculative_config=None,
    )
    for path, value in overrides.items():
        target, name = config, path.split(".")
        for part in name[:-1]:
            target = getattr(target, part)
        setattr(target, name[-1], value)
    return config


@pytest.fixture(autouse=True)
def hopper_with_flashinfer_gdn(monkeypatch):
    monkeypatch.setattr(
        qwen_gdn_linear_attn,
        "_resolve_gdn_prefill_backend",
        lambda _: ("auto", "flashinfer"),
    )
    platform = type(current_platform)
    monkeypatch.setattr(platform, "is_cuda", classmethod(lambda _: True))
    monkeypatch.setattr(platform, "is_cuda_alike", classmethod(lambda _: True))
    monkeypatch.setattr(
        platform,
        "is_device_capability_family",
        classmethod(lambda _, capability, device_id=0: capability == 90),
    )


def test_caches_state_blocks_when_asked():
    config = vllm_config()
    D3ForSequenceClassificationConfig.verify_and_update_config(config)
    scheduler = config.scheduler_config
    assert config.cache_config.enable_prefix_caching
    assert scheduler.scheduler_cls == BIDIRECTIONAL
    assert scheduler.enable_chunked_prefill
    assert scheduler.long_prefill_token_threshold == 0
    # A whole prompt and its media fit one step.
    assert scheduler.max_num_batched_tokens == 65536
    assert scheduler.max_num_encoder_input_tokens == 65536
    assert scheduler.encoder_cache_size == 65536


def test_sizes_the_encoder_budget_to_one_prompt():
    """A larger step budget does not grow the encoder budget, whose startup
    profiling would overflow the vision tower's int32 indexing."""
    config = vllm_config(max_num_batched_tokens=393216, user_budget=True)
    D3ForSequenceClassificationConfig.verify_and_update_config(config)
    scheduler = config.scheduler_config
    assert scheduler.max_num_batched_tokens == 393216
    assert scheduler.max_num_encoder_input_tokens == 65536
    assert scheduler.encoder_cache_size == 65536
    assert config.cache_config.mamba_ssm_cache_dtype == "float32"


def test_overrides_the_checkpoints_bfloat16_state_default():
    """Only a dtype the user asked for is refused; the checkpoint's default
    is replaced."""
    config = vllm_config(**{"model_config.hf_text_config.mamba_ssm_dtype": "bfloat16"})
    D3ForSequenceClassificationConfig.verify_and_update_config(config)
    assert config.cache_config.mamba_ssm_cache_dtype == "float32"


def test_leaves_the_config_alone_without_prefix_caching():
    config = vllm_config(caching=False)
    D3ForSequenceClassificationConfig.verify_and_update_config(config)
    assert config.scheduler_config.scheduler_cls is None
    assert not config.scheduler_config.enable_chunked_prefill
    assert config.cache_config.mamba_ssm_cache_dtype == "auto"


def test_noncausal_checkpoints_cache_nothing():
    config = vllm_config(attention_mode="noncausal_full_attention")
    D3ForSequenceClassificationConfig.verify_and_update_config(config)
    assert not config.cache_config.enable_prefix_caching
    assert config.scheduler_config.scheduler_cls is None


def test_causal_checkpoints_are_left_to_the_defaults():
    config = vllm_config(attention_mode="causal")
    D3ForSequenceClassificationConfig.verify_and_update_config(config)
    assert config.cache_config.enable_prefix_caching
    assert config.scheduler_config.scheduler_cls is None


@pytest.mark.parametrize(
    "gdn,cached", [("flashinfer", True), ("triton", True), ("cutedsl", False)]
)
def test_needs_a_gdn_backend_that_saves_mid_prefill_states(monkeypatch, gdn, cached):
    monkeypatch.setattr(
        qwen_gdn_linear_attn, "_resolve_gdn_prefill_backend", lambda _: ("auto", gdn)
    )
    config = vllm_config()
    D3ForSequenceClassificationConfig.verify_and_update_config(config)
    assert config.cache_config.enable_prefix_caching == cached
    assert (config.scheduler_config.scheduler_cls == BIDIRECTIONAL) == cached


@pytest.mark.parametrize(
    "override,feature",
    [
        ({"speculative_config": object()}, "speculative decoding"),
        ({"kv_transfer_config": object()}, "KV transfer connectors"),
        ({"cache_config.kv_offloading_size": 4}, "--kv-offloading-size"),
        ({"cache_config.prefix_match_unit": 256}, "--prefix-match-unit"),
        ({"parallel_config.use_ubatching": True}, "micro-batching"),
        (
            {"parallel_config.decode_context_parallel_size": 2},
            "context parallelism",
        ),
        ({"scheduler_config.scheduler_cls": "my.Scheduler"}, "--scheduler-cls"),
    ],
)
def test_refuses_what_it_cannot_cache_with(override, feature):
    with pytest.raises(ValueError, match=feature):
        D3ForSequenceClassificationConfig.verify_and_update_config(
            vllm_config(**override)
        )


def test_refuses_a_bfloat16_state_cache():
    with pytest.raises(ValueError, match="mamba-ssm-cache-dtype float32"):
        D3ForSequenceClassificationConfig.verify_and_update_config(
            vllm_config(ssm_dtype="bfloat16")
        )


def test_refuses_a_user_budget_below_a_prompt():
    with pytest.raises(ValueError, match="--max-num-batched-tokens"):
        D3ForSequenceClassificationConfig.verify_and_update_config(
            vllm_config(max_num_batched_tokens=32768, user_budget=True)
        )


@pytest.mark.parametrize(
    "backend,version,warns",
    [
        (AttentionBackendEnum.FLASH_ATTN, 4, False),
        (AttentionBackendEnum.FLASH_ATTN, 3, True),
        (None, None, True),
        (AttentionBackendEnum.TRITON_ATTN, None, True),
    ],
)
def test_warns_when_hopper_runs_prefix_bidir_on_triton(
    monkeypatch, backend, version, warns
):
    warning = Mock()
    monkeypatch.setattr(model_configs.logger, "warning", warning)
    config = vllm_config(
        caching=False,
        **{
            "attention_config.backend": backend,
            "attention_config.flash_attn_version": version,
        },
    )
    D3ForSequenceClassificationConfig.verify_and_update_config(config)
    assert warning.called == warns
