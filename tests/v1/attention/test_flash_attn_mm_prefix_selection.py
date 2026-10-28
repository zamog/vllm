# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FlashAttention 2/3 take PrefixLM ranges unless the model has a sliding
window, which only FlashAttention 4 (or Triton) can combine with them."""

from types import SimpleNamespace

import pytest
import torch

from vllm.platforms.interface import DeviceCapability
from vllm.v1.attention.backends import flash_attn

pytestmark = pytest.mark.cpu_test


@pytest.mark.parametrize(
    "version,sliding_window,supported",
    [
        (2, None, True),
        (3, None, True),
        (2, 1024, False),
        (4, 1024, True),
    ],
)
def test_mm_prefix_on_flash_attention(monkeypatch, version, sliding_window, supported):
    monkeypatch.setattr(flash_attn, "get_flash_attn_version", lambda **_: version)
    model_config = SimpleNamespace(get_sliding_window=lambda: sliding_window)
    monkeypatch.setattr(
        flash_attn,
        "get_current_vllm_config_or_none",
        lambda: SimpleNamespace(model_config=model_config),
    )
    reason = flash_attn.FlashAttentionBackend.supports_combination(
        head_size=256,
        dtype=torch.bfloat16,
        kv_cache_dtype="auto",
        block_size=784,
        use_mla=False,
        has_sink=False,
        use_sparse=False,
        use_mm_prefix=True,
        device_capability=DeviceCapability(8, 0),
    )
    assert (reason is None) == supported, reason


@pytest.mark.parametrize(
    "fa4,sliding_window,selectable",
    [(False, None, True), (False, 1024, False), (True, 1024, True)],
)
def test_validate_configuration_takes_mm_prefix(
    monkeypatch, fa4, sliding_window, selectable
):
    """The selector validates mm_prefix through supports_mm_prefix() before
    supports_combination(); FlashAttention 2/3 must pass both on Ampere."""
    monkeypatch.setattr(flash_attn, "is_flash_attn_varlen_func_available", lambda: True)
    monkeypatch.setattr(type(flash_attn.current_platform), "is_cuda", lambda *_: True)
    monkeypatch.setattr(flash_attn, "is_fa_version_supported", lambda v: fa4)
    monkeypatch.setattr(
        flash_attn, "get_flash_attn_version", lambda **_: 4 if fa4 else 2
    )
    monkeypatch.setattr(
        flash_attn, "_mm_prefix_as_segments_supported", lambda: sliding_window is None
    )
    reasons = flash_attn.FlashAttentionBackend.validate_configuration(
        head_size=256,
        dtype=torch.bfloat16,
        kv_cache_dtype="auto",
        block_size=784,
        use_mla=False,
        has_sink=False,
        use_sparse=False,
        use_mm_prefix=True,
        use_per_head_quant_scales=False,
        device_capability=DeviceCapability(8, 0),
        attn_type="decoder",
    )
    mm_reasons = [r for r in reasons if "multimodal" in r or "mm_prefix" in r]
    assert (not mm_reasons) == selectable, reasons
