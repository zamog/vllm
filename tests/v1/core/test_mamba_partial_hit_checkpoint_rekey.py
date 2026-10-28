# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Mamba "align" partial hit followed by a checkpointed chunk spanning columns.

Invariant: a hash-cached Mamba slot holds the state its key claims.

With a hash unit finer than the Mamba block and in-step prefill checkpoints,
request A (2500 tokens) prefills in one chunk and keys its checkpoint slot as
state@2496 (mid-block). Request B, A's prompt plus 2500 tokens, hits that
checkpoint, gets a private CoW copy of it at column 1, and is scheduled as one
chunk [2496, 5000) because its own internal checkpoint (column 2) is valid, so
`_mamba_block_aligned_split` suppresses the next-block-boundary stop. The
worker writes only the chunk-end column (3) and the checkpoint column (2); the
CoW copy at column 1 keeps state@2496. `allocate_slots` then caches 5000 tokens
unrounded and `cache_full_blocks` keys every non-null block positionally, so
the CoW copy is published as state@3200. A later request C sharing that 3200
token prefix restores state@2496 under that key.

Worker model (CPU test, no kernel): the initial state is read from column
`(start - 1) // B` (gpu/model_states/mamba_hybrid.py `add_request`), the
running state is written to column `(end - 1) // B` (backends/utils.py
`(seq_lens - 1) // block_size`), the in-step checkpoint lands in the column
`compute_mamba_prefill_checkpoints` returns, a CoW copy makes the destination
hold the source's state, and no other column is written during prefill.
"""

from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.layers.mamba.checkpoint import (
    compute_mamba_prefill_checkpoints,
)
from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    MambaSpec,
    get_mamba_prefill_checkpoint_position,
)
from vllm.v1.request import Request

pytestmark = pytest.mark.cpu_test

ATTN_BLOCK_SIZE = 16  # hash unit U
MAMBA_BLOCK_SIZE = 1600  # Mamba block B
NUM_SPEC = 3
MAMBA_GROUP_ID = 1

# A's checkpoint is floor((2500 - 1) / 16) * 16 = 2496, inside column 1.
PROMPT_A = 2500
# B must end past column 3 so its checkpoint column (2) exceeds B's
# initial-state column (1); see `is_mamba_prefill_checkpoint_valid`.
PROMPT_B = 5000
# C shares B's first 3200 tokens: the column-1 boundary B's chunk crossed.
PROMPT_C = 3300


def _make_hybrid_kv_cache_manager() -> KVCacheManager:
    config = KVCacheConfig(
        num_blocks=10000,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["full_layer"],
                FullAttentionSpec(
                    block_size=ATTN_BLOCK_SIZE,
                    num_kv_heads=1,
                    head_size=1,
                    dtype=torch.float32,
                ),
            ),
            KVCacheGroupSpec(
                ["mamba_layer"],
                MambaSpec(
                    block_size=MAMBA_BLOCK_SIZE,
                    shapes=((1, 1),),
                    dtypes=(torch.float32,),
                    mamba_cache_mode="align",
                    num_speculative_blocks=NUM_SPEC,
                    num_prefill_checkpoint_blocks=1,
                    prefill_checkpoint_alignment=ATTN_BLOCK_SIZE,
                ),
            ),
        ],
    )
    return KVCacheManager(
        config,
        max_model_len=262144,
        scheduler_block_size=MAMBA_BLOCK_SIZE,
        hash_block_size=ATTN_BLOCK_SIZE,
        enable_caching=True,
        use_eagle=False,
    )


def _make_request(request_id: str, prompt_token_ids: list[int]) -> Request:
    init_none_hash(sha256)
    sampling_params = SamplingParams(max_tokens=16)
    return Request(
        request_id=request_id,
        prompt_token_ids=prompt_token_ids,
        sampling_params=sampling_params,
        pooling_params=None,
        mm_features=None,
        block_hasher=get_request_block_hasher(ATTN_BLOCK_SIZE, sha256),
    )


def _split(
    request: Request, num_new_tokens: int, num_new_local_computed_tokens: int = 0
) -> int:
    """Call the real `Scheduler._mamba_block_aligned_split` on a stub self."""
    stub = SimpleNamespace(
        block_size=MAMBA_BLOCK_SIZE,
        cache_config=SimpleNamespace(block_size=MAMBA_BLOCK_SIZE),
        use_eagle_block_drop=False,
        num_hash_lookahead_tokens=0,
        max_num_scheduled_tokens=16384,
        scheduler_config=SimpleNamespace(long_prefill_token_threshold=0),
        mamba_partial_cache_hit=True,
        mamba_shared_prefix_checkpoint=False,
        hash_block_size=ATTN_BLOCK_SIZE,
        mamba_has_prefill_checkpoint_blocks=True,
        mamba_prefill_checkpoint_alignment=ATTN_BLOCK_SIZE,
    )
    return Scheduler._mamba_block_aligned_split(
        stub,
        request,
        num_new_tokens,
        num_new_local_computed_tokens=num_new_local_computed_tokens,
    )


def _apply_worker_step(
    manager: KVCacheManager,
    request: Request,
    start: int,
    end: int,
    state_at: dict[int, int],
) -> None:
    """Mirror what the forward pass writes for the chunk `[start, end)`.

    `state_at` maps physical mamba block id -> token offset of the state it
    holds (see the module docstring for the modelled kernel behaviour).
    """
    mamba_manager = manager.coordinator.single_type_managers[MAMBA_GROUP_ID]
    blocks = mamba_manager.req_to_blocks[request.request_id]
    # The queued copy fills the CoW block before the forward reads it.
    for source, cow in mamba_manager.take_pending_cow_copies():
        state_at[cow.block_id] = state_at[source.block_id]
    if start > 0:
        initial = blocks[(start - 1) // MAMBA_BLOCK_SIZE]
        assert state_at.get(initial.block_id) == start, (
            f"initial-state column holds state@{state_at.get(initial.block_id)}, "
            f"the chunk resumes at {start}"
        )
    (offset,), (col,) = compute_mamba_prefill_checkpoints(
        [end],
        [end - start],
        hash_block_size=ATTN_BLOCK_SIZE,
        mamba_block_size=MAMBA_BLOCK_SIZE,
        checkpoint_alignment=ATTN_BLOCK_SIZE,
        drop_eagle_block=False,
        cache_hit_alignment_tokens=ATTN_BLOCK_SIZE,
        max_cache_hit_lengths=[request.max_cache_hit_length],
    )
    if col >= 0:
        assert not blocks[col].is_null
        state_at[blocks[col].block_id] = start + offset
    state_at[blocks[(end - 1) // MAMBA_BLOCK_SIZE].block_id] = end


def _assert_cached_mamba_slots_hold_claimed_state(
    manager: KVCacheManager, request: Request, state_at: dict[int, int]
) -> int:
    """Assert every hash-cached mamba slot holds the state its hash claims."""
    mamba_manager = manager.coordinator.single_type_managers[MAMBA_GROUP_ID]
    checked = 0
    for pos, block in enumerate(mamba_manager.req_to_blocks[request.request_id]):
        if block.is_null or block.block_hash is None:
            continue
        claimed = block.block_hash_num_tokens
        assert state_at.get(block.block_id) == claimed, (
            f"mamba slot {pos} (block {block.block_id}) is hashed as "
            f"state@{claimed} but holds state@{state_at.get(block.block_id)}"
        )
        checked += 1
    return checked


def _prefill_in_one_chunk(
    manager: KVCacheManager,
    request: Request,
    state_at: dict[int, int],
) -> int:
    """Schedule `request`'s whole prompt in one step. Returns its hit length."""
    manager.new_step_starts()
    blocks, num_hit, _ = manager.get_computed_blocks(request)
    num_new = _split(
        request,
        request.num_tokens - num_hit,
        num_new_local_computed_tokens=num_hit,
    )
    assert num_new == request.num_tokens - num_hit, (
        f"{request.request_id}: split [{num_hit}, {num_hit + num_new}) is not "
        "the whole remaining prompt"
    )
    assert (
        manager.allocate_slots(
            request,
            num_new,
            num_new_computed_tokens=num_hit,
            new_computed_blocks=blocks,
            num_lookahead_tokens=NUM_SPEC,
        )
        is not None
    )
    request.num_computed_tokens = num_hit + num_new
    _apply_worker_step(manager, request, num_hit, request.num_tokens, state_at)
    return num_hit


def test_partial_hit_then_checkpointed_chunk_does_not_rekey_cow_block() -> None:
    """B's CoW copy of A's mid-block checkpoint must not be keyed as a boundary.

    B resumes mid-block, so it carries `skip_writing_prefix_cache` (as
    PrefixLMScheduler sets it) and caches nothing. Without it, B's column-1
    CoW block is keyed as state@3200 while it holds state@2496.
    """
    manager = _make_hybrid_kv_cache_manager()
    state_at: dict[int, int] = {}
    shared = [1] * PROMPT_A

    req_a = _make_request("A", shared)
    assert _prefill_in_one_chunk(manager, req_a, state_at) == 0
    assert _assert_cached_mamba_slots_hold_claimed_state(manager, req_a, state_at)
    manager.free(req_a)

    req_b = _make_request("B", shared + [2] * (PROMPT_B - PROMPT_A))
    req_b.skip_writing_prefix_cache = True
    hit_b = _prefill_in_one_chunk(manager, req_b, state_at)
    # Preconditions of the hazard: a mid-block hit on A's checkpoint, then a
    # single chunk crossing two column boundaries (3200, 4800).
    assert hit_b == get_mamba_prefill_checkpoint_position(
        PROMPT_A, ATTN_BLOCK_SIZE, drop_eagle_block=False
    )
    assert hit_b % MAMBA_BLOCK_SIZE != 0
    assert (PROMPT_B - 1) // MAMBA_BLOCK_SIZE - (hit_b - 1) // MAMBA_BLOCK_SIZE >= 2

    _assert_cached_mamba_slots_hold_claimed_state(manager, req_b, state_at)
    # A decode step's caching would key the same block: the flag outlives
    # the prompt.
    req_b.append_output_token_ids(7)
    req_b.num_computed_tokens = PROMPT_B + 1
    manager.cache_blocks(req_b, PROMPT_B + 1)
    _assert_cached_mamba_slots_hold_claimed_state(manager, req_b, state_at)

    # The user-visible consequence: C's Mamba hit block must hold the state
    # the hit length promises.
    manager.new_step_starts()
    req_c = _make_request("C", shared + [2] * (PROMPT_C - PROMPT_A))
    blocks_c, hit_c, _ = manager.get_computed_blocks(req_c)
    mamba_hit = [b for b in blocks_c.blocks[MAMBA_GROUP_ID] if not b.is_null]
    assert len(mamba_hit) == 1
    assert state_at.get(mamba_hit[0].block_id) == hit_c, (
        f"C hits {hit_c} tokens but its mamba block {mamba_hit[0].block_id} "
        f"holds state@{state_at.get(mamba_hit[0].block_id)}"
    )


@pytest.mark.parametrize(
    "cap,position", [(None, 960), (500, 448), (999, 960), (960, 960)]
)
def test_checkpoint_position_stops_at_the_hit_cap(cap, position):
    """A capped checkpoint moves to the cap's last boundary; the block column
    stays the chunk's second to last, which the manager re-keys."""
    assert (
        get_mamba_prefill_checkpoint_position(1000, 64, False, max_cache_hit_length=cap)
        == position
    )
    offsets, cols = compute_mamba_prefill_checkpoints(
        [1000],
        [1000],
        hash_block_size=64,
        mamba_block_size=832,
        checkpoint_alignment=64,
        drop_eagle_block=False,
        cache_hit_alignment_tokens=64,
        max_cache_hit_lengths=[cap],
    )
    assert (offsets, cols) == ([position], [0])


def test_capped_checkpoint_is_keyed_where_it_is_saved() -> None:
    """A prompt whose hits stop at 2500 (a PrefixLM prefix of 4000 tokens)
    saves its checkpoint at 2496, not 3984, and the manager keys it there:
    a request sharing the prefix resumes from the state it holds."""
    manager = _make_hybrid_kv_cache_manager()
    state_at: dict[int, int] = {}
    prefix = [1] * PROMPT_A

    req_a = _make_request("A", prefix + [2] * (4000 - PROMPT_A))
    req_a.max_cache_hit_length = PROMPT_A
    assert _prefill_in_one_chunk(manager, req_a, state_at) == 0
    assert _assert_cached_mamba_slots_hold_claimed_state(manager, req_a, state_at)
    manager.free(req_a)

    manager.new_step_starts()
    req_b = _make_request("B", prefix + [3] * 700)
    blocks_b, hit_b, _ = manager.get_computed_blocks(req_b, PROMPT_A)
    assert hit_b == get_mamba_prefill_checkpoint_position(
        4000, ATTN_BLOCK_SIZE, False, max_cache_hit_length=PROMPT_A
    )
    (mamba_hit,) = [b for b in blocks_b.blocks[MAMBA_GROUP_ID] if not b.is_null]
    assert state_at.get(mamba_hit.block_id) == hit_b
