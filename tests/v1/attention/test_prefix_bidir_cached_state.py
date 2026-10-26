# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A Decision 3.0 prefix_bidir question read over its cached state block.

The state block [0, s) reads itself in both directions; the question block
[s, n) reads the state and itself in both directions. With prefix caching the
state's KV up to its last Mamba block boundary c comes from the cache, and the
step computes [c, n): state rows whose range starts in the cached context, and
question rows. The output must match a dense reference of the whole prompt.
"""

from types import MethodType

import pytest
import torch

from tests.v1.attention.test_attention_backends import (
    create_and_prepopulate_kv_cache,
    run_attention_backend,
)
from tests.v1.attention.test_mm_prefix import _dense_reference, _enable_mm_prefix
from tests.v1.attention.utils import (
    BatchSpec,
    create_common_attn_metadata,
    create_vllm_config,
)
from vllm.platforms import current_platform
from vllm.v1.attention.backends.registry import AttentionBackendEnum
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheLayout

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda(), reason="CUDA attention kernels"
)

DEVICE = torch.device("cuda:0")
DTYPE = torch.bfloat16
# d3-turbo's full attention layers.
NUM_HEADS = 24
NUM_KV_HEADS = 4
HEAD_SIZE = 256
# Config only; no weights are loaded. Made a PrefixLM model in _backend_config.
MODEL = "google/gemma-4-E2B-it"


def _requests(block: int) -> list[tuple[int, int, int]]:
    """(state tokens, question tokens, cached tokens) per request."""
    return [
        # A long state: c = its last boundary, mid-state.
        (3 * block + 300, 250, 3 * block),
        # A state ending one token past a boundary.
        (block + 1, 40, block),
        # A state within its first block: nothing cached.
        (block - 100, 60, 0),
    ]


def _backend_config(backend: AttentionBackendEnum, block: int, version: int | None):
    # The FlashAttention builder fills the ranges only for PrefixLM models,
    # as the Decision 3.0 prefix_bidir model is.
    vllm_config = _enable_mm_prefix(
        create_vllm_config(
            model_name=MODEL,
            max_model_len=8 * block,
            block_size=block,
            num_gpu_blocks=64,
        )
    )
    assert vllm_config.model_config.is_mm_prefix_lm
    mc = vllm_config.model_config
    mc.get_head_size = MethodType(lambda self: HEAD_SIZE, mc)
    mc.get_num_attention_heads = MethodType(lambda self, _=None: NUM_HEADS, mc)
    mc.get_num_kv_heads = MethodType(lambda self, _=None: NUM_KV_HEADS, mc)
    if backend == AttentionBackendEnum.FLASH_ATTN:
        vllm_config.attention_config.flash_attn_version = version
    return vllm_config


def run_prefix_bidir(
    backend: AttentionBackendEnum,
    block: int,
    requests: list[tuple[int, int, int]],
    version: int | None = 4,
) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
    """Per request: the backend's output rows, the dense reference and a
    causal-only reference, for ``requests`` of (state tokens, question
    tokens, cached tokens)."""
    torch.manual_seed(0)
    vllm_config = _backend_config(backend, block, version)
    seq_lens = [s + q for s, q, _ in requests]
    query_lens = [s + q - c for s, q, c in requests]
    ranges = {i: [(0, s - 1), (s, s + q - 1)] for i, (s, q, _) in enumerate(requests)}

    qs, keys, values, k_fulls, v_fulls, k_ctxs, v_ctxs = [], [], [], [], [], [], []
    for q_len, s_len in zip(query_lens, seq_lens):
        ctx = s_len - q_len
        qs.append(torch.randn(q_len, NUM_HEADS, HEAD_SIZE, dtype=DTYPE, device=DEVICE))
        k_full = torch.randn(s_len, NUM_KV_HEADS, HEAD_SIZE, dtype=DTYPE, device=DEVICE)
        v_full = torch.randn(s_len, NUM_KV_HEADS, HEAD_SIZE, dtype=DTYPE, device=DEVICE)
        k_fulls.append(k_full)
        v_fulls.append(v_full)
        k_ctxs.append(k_full[:ctx])
        v_ctxs.append(v_full[:ctx])
        keys.append(k_full[ctx:])
        values.append(v_full[ctx:])

    batch = BatchSpec(seq_lens=seq_lens, query_lens=query_lens, name="prefix_bidir")
    common = create_common_attn_metadata(batch, block, DEVICE)
    common.mm_req_doc_ranges = ranges
    kv_cache_spec = FullAttentionSpec(
        block_size=block,
        num_kv_heads=NUM_KV_HEADS,
        head_size=HEAD_SIZE,
        dtype=DTYPE,
    )
    layouts = backend.get_class().supported_kv_cache_layouts()
    kv_cache = create_and_prepopulate_kv_cache(
        k_contexts=k_ctxs,
        v_contexts=v_ctxs,
        block_size=block,
        num_kv_heads=NUM_KV_HEADS,
        head_size=HEAD_SIZE,
        dtype=DTYPE,
        device=DEVICE,
        num_blocks=64,
        common_attn_metadata=common,
        layout=layouts[0] if layouts else KVCacheLayout.LBNHC,
        randomize_blocks=True,
    )
    query = torch.cat(qs)
    actual = run_attention_backend(
        backend,
        kv_cache_spec,
        ["model.layers.0.self_attn.attn"],
        vllm_config,
        DEVICE,
        common,
        query,
        torch.cat(keys),
        torch.cat(values),
        kv_cache,
    )
    scale = HEAD_SIZE**-0.5
    ref_args = (query, k_fulls, v_fulls, query_lens, seq_lens)
    expected = _dense_reference(*ref_args, ranges, None, scale)
    causal = _dense_reference(*ref_args, {}, None, scale)
    return (
        list(actual.float().split(query_lens)),
        list(expected.split(query_lens)),
        list(causal.split(query_lens)),
    )


@pytest.mark.parametrize(
    "backend,version,block",
    [
        # H200: FlashAttention 4 (mask hook) with FlashInfer GDN's 832-token
        # blocks, and FlashAttention 3 (virtual sequences).
        pytest.param(AttentionBackendEnum.FLASH_ATTN, 4, 832, id="fa4-832"),
        pytest.param(AttentionBackendEnum.FLASH_ATTN, 3, 832, id="fa3-832"),
        # A100: FlashAttention 2 (virtual sequences) and Triton with Triton/FLA
        # GDN's 784-token blocks.
        pytest.param(AttentionBackendEnum.FLASH_ATTN, 2, 784, id="fa2-784"),
        pytest.param(AttentionBackendEnum.TRITON_ATTN, None, 784, id="triton-784"),
    ],
)
def test_question_over_cached_state_matches_whole_prompt(backend, version, block):
    if backend == AttentionBackendEnum.FLASH_ATTN:
        from vllm.v1.attention.backends.fa_utils import is_fa_version_supported

        hopper = current_platform.is_device_capability_family(90)
        if not is_fa_version_supported(version) or (version in (3, 4) and not hopper):
            pytest.skip(f"FlashAttention {version} on this GPU")
    actual, expected, causal = run_prefix_bidir(
        backend, block, _requests(block), version
    )
    for i, (got, want, causal_only) in enumerate(zip(actual, expected, causal)):
        torch.testing.assert_close(
            got, want, atol=2e-2, rtol=2e-2, msg=lambda m, i=i: f"request {i}: {m}"
        )
        # The ranges change the result: a causal read would differ.
        assert not torch.allclose(want, causal_only, atol=2e-2, rtol=2e-2)
