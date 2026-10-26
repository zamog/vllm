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
