# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from vllm.config.compilation import CUDAGraphMode
from vllm.model_executor.models.d3 import STATE_TOKENS, D3PrefixBidirModelState
from vllm.pooling_params import PoolingParams
from vllm.v1.worker.gpu.model_states import mamba_hybrid
from vllm.v1.worker.gpu.model_states.mamba_hybrid import MambaHybridModelState


def _new_request(
    state_tokens: object, pooling: bool = True, num_computed_tokens: int = 0
) -> SimpleNamespace:
    extra = {} if state_tokens is None else {STATE_TOKENS: state_tokens}
    return SimpleNamespace(
        req_id="r",
        prompt_token_ids=list(range(10)),
        pooling_params=PoolingParams(extra_kwargs=extra) if pooling else None,
        num_computed_tokens=num_computed_tokens,
        max_cache_hit_length=6,
    )


def test_keeps_the_hybrid_model_state() -> None:
    # Mamba "align" prefix caching lives in MambaHybridModelState.
    assert issubclass(D3PrefixBidirModelState, MambaHybridModelState)


@pytest.mark.parametrize(
    ("state_tokens", "pooling", "blocks"),
    [
        (6, True, [(0, 5), (6, 9)]),
        (None, True, [(0, 9)]),  # no state block size: one block
        (None, False, [(0, 9)]),  # no pooling params
        (10, True, [(0, 9)]),  # a state-only prompt
        (0, True, [(0, 9)]),
        (True, True, [(0, 9)]),  # a bool is not a token count
    ],
)
def test_state_and_question_blocks(
    monkeypatch: pytest.MonkeyPatch,
    state_tokens: object,
    pooling: bool,
    blocks: list[tuple[int, int]],
) -> None:
    monkeypatch.setattr(MambaHybridModelState, "add_request", lambda *_: None)
    monkeypatch.setattr(MambaHybridModelState, "remove_request", lambda *_: None)
    state = object.__new__(D3PrefixBidirModelState)
    state.blocks = {}
    state.add_request(0, _new_request(state_tokens, pooling))  # type: ignore[arg-type]
    batch = SimpleNamespace(req_ids=["other", "r"])
    assert state.get_mm_prefix_ranges(batch) == {1: blocks}  # type: ignore[arg-type]
    state.remove_request("r")
    assert state.get_mm_prefix_ranges(batch) is None  # type: ignore[arg-type]


def test_add_request_seeds_the_aligned_mamba_state() -> None:
    """The hybrid add_request runs under the blocks: with prefix caching the
    running Mamba state column starts at the resumed position."""
    state = object.__new__(D3PrefixBidirModelState)
    state.blocks = {}
    state.cache_config = SimpleNamespace(block_size=16, mamba_block_size=832)
    state._align_mode = True
    state.rope_state = None
    state.prompt_embeds_state = None
    state.num_accepted_tokens_gpu = torch.zeros(2, dtype=torch.int32)
    state._mamba_state_idx_gpu = torch.zeros(2, dtype=torch.int32)
    state._max_cache_hit_length_np = np.full(2, -1, dtype=np.int32)

    # A question resuming over a cached 1,664-token state.
    state.add_request(1, _new_request(6, num_computed_tokens=1_664))  # type: ignore[arg-type]

    assert state._mamba_state_idx_gpu[1] == 1
    # Its prefill checkpoint is saved at its hit cap, where the scheduler keys it.
    assert state._max_cache_hit_length_np[1] == 6
    assert state.num_accepted_tokens_gpu[1] == 1
    assert state.blocks["r"] == [(0, 5), (6, 9)]


def test_prepare_attn_passes_the_blocks(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without the blocks in the attention metadata the state reads causally."""
    state = object.__new__(D3PrefixBidirModelState)
    state.blocks = {"r": [(0, 5), (6, 9)]}
    state.vllm_config = SimpleNamespace(num_speculative_tokens=0)
    state.model_config = SimpleNamespace(max_model_len=8192)
    state._align_mode = False
    state.recoverssm = None

    input_batch = SimpleNamespace(
        req_ids=["r"],
        num_reqs=1,
        num_tokens=10,
        num_reqs_after_padding=1,
        num_tokens_after_padding=10,
        query_start_loc_np=torch.tensor([0, 10], dtype=torch.int32).numpy(),
        query_start_loc=torch.tensor([0, 10], dtype=torch.int32),
        num_scheduled_tokens=torch.tensor([10], dtype=torch.int32),
        max_query_len=None,
        seq_lens_cpu_upper_bound=torch.tensor([10], dtype=torch.int32),
        seq_lens=torch.tensor([10], dtype=torch.int32),
        is_prefilling_np=torch.tensor([True]).numpy(),
        prefill_runs_as_decode_np=None,
        dcp_local_seq_lens=None,
        positions=torch.arange(10),
        prompt_lens=torch.tensor([10], dtype=torch.int32),
    )
    build_attn_metadata = Mock(return_value={})
    monkeypatch.setattr(mamba_hybrid, "build_attn_metadata", build_attn_metadata)

    state.prepare_attn(
        input_batch=input_batch,
        cudagraph_mode=CUDAGraphMode.NONE,
        block_tables=(),
        slot_mappings=torch.empty(0, dtype=torch.int64),
        attn_groups=[],
        kv_cache_config=Mock(),
    )

    ranges = build_attn_metadata.call_args.kwargs["mm_req_doc_ranges"]
    assert ranges == {0: [(0, 5), (6, 9)]}
