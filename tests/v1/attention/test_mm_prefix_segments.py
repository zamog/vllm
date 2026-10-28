# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PrefixLM ranges as non-causal and causal virtual sequences."""

import random

import numpy as np
import pytest
import torch

from vllm.v1.attention.backends.utils import split_mm_prefix_segments

pytestmark = pytest.mark.cpu_test


def split(ranges, query_lens, seq_lens):
    query_start_loc = torch.tensor([0, *np.cumsum(query_lens)], dtype=torch.int32)
    return split_mm_prefix_segments(
        ranges, query_start_loc, torch.tensor(seq_lens, dtype=torch.int32)
    )


def segments(group):
    """(request, first packed row, rows, keys) of each segment."""
    if group is None:
        return []
    rows = (
        group.rows if group.rows is not None else np.arange(int(group.cu_seqlens_q[-1]))
    )
    return [
        (int(r), int(rows[a]), int(b - a), int(k))
        for r, a, b, k in zip(
            group.req_idx,
            group.cu_seqlens_q[:-1],
            group.cu_seqlens_q[1:],
            group.seqused_k,
        )
    ]


def test_state_and_question_without_cache():
    """A Decision 3.0 prompt: state [0, 600), question [600, 650)."""
    non_causal, causal = split({0: [(0, 599), (600, 649)]}, [650], [650])
    assert segments(non_causal) == [(0, 0, 600, 600), (0, 600, 50, 650)]
    assert non_causal.rows is None  # every row, in order: no gather
    assert causal is None
    assert (non_causal.max_seqlen_q, non_causal.max_seqlen_k) == (600, 650)


def test_question_over_a_cached_state():
    """The state is cached up to 512: the step reads the state's tail rows and
    the question; the tail range starts in the cached context."""
    non_causal, causal = split({0: [(0, 599), (600, 649)]}, [138], [650])
    assert segments(non_causal) == [(0, 0, 88, 600), (0, 88, 50, 650)]
    assert causal is None


def test_question_only():
    non_causal, causal = split({0: [(0, 599), (600, 649)]}, [50], [650])
    assert segments(non_causal) == [(0, 0, 50, 650)]
    assert causal is None


def test_image_inside_a_causal_prompt():
    non_causal, causal = split({0: [(10, 19)]}, [30], [30])
    assert segments(non_causal) == [(0, 10, 10, 20)]
    assert segments(causal) == [(0, 0, 10, 10), (0, 20, 10, 30)]
    np.testing.assert_array_equal(non_causal.rows, np.arange(10, 20))


def test_mixed_batch_and_decode_rows():
    """A causal prompt, a decode row past its image, and a PrefixLM prompt."""
    non_causal, causal = split(
        {1: [(2, 5)], 2: [(0, 7), (8, 9)]}, [6, 1, 10], [6, 40, 10]
    )
    assert segments(causal) == [(0, 0, 6, 6), (1, 6, 1, 40)]
    assert segments(non_causal) == [(2, 7, 8, 8), (2, 15, 2, 10)]
    # Causal segments ending at their request's end take the device's exact
    # sequence length (a decode row's CPU length is an upper bound).
    assert causal.ends_at_seq_len.tolist() == [True, True]
    assert not non_causal.ends_at_seq_len.any()


def test_causal_rows_before_a_range_keep_their_key_count():
    _, causal = split({0: [(10, 19)]}, [30], [30])
    assert causal.ends_at_seq_len.tolist() == [False, True]


def test_range_past_the_computed_tokens():
    """Keys past the scheduled tokens are not in the cache yet."""
    non_causal, causal = split({0: [(0, 99)]}, [40], [40])
    assert segments(non_causal) == [(0, 0, 40, 40)]
    assert causal is None


@pytest.mark.parametrize("ranges", [None, {}, {0: []}, {0: [(5, 5)]}, {0: [(50, 60)]}])
def test_no_range_covers_a_scheduled_row(ranges):
    assert split(ranges, [20], [20]) is None


def dense_keys(ranges, query_len, seq_len):
    """Keys each scheduled row may read under (causal OR same range)."""
    context = seq_len - query_len
    out = []
    for q in range(context, seq_len):
        keys = set(range(q + 1))
        for start, end in ranges:
            if start < end and start <= q <= end:
                keys |= set(range(start, min(end + 1, seq_len)))
        out.append(keys)
    return out


@pytest.mark.parametrize("seed", range(20))
def test_segments_read_the_dense_mask(seed):
    rng = random.Random(seed)
    num_reqs = rng.randint(1, 4)
    query_lens, seq_lens, ranges = [], [], {}
    for r in range(num_reqs):
        seq_len = rng.randint(1, 60)
        query_lens.append(rng.randint(1, seq_len))
        seq_lens.append(seq_len)
        bounds = sorted(rng.sample(range(70), rng.choice([0, 2, 4, 6])))
        ranges[r] = list(zip(bounds[::2], bounds[1::2]))
    result = split(ranges, query_lens, seq_lens)
    query_start = np.cumsum([0, *query_lens])
    want = {}
    for r in range(num_reqs):
        for i, keys in enumerate(dense_keys(ranges[r], query_lens[r], seq_lens[r])):
            want[int(query_start[r] + i)] = keys
    if result is None:
        # No range covers a row: every row reads causally.
        for r in range(num_reqs):
            context = seq_lens[r] - query_lens[r]
            for i in range(query_lens[r]):
                assert want[int(query_start[r] + i)] == set(range(context + i + 1))
        return
    got = {}
    for group, is_causal in zip(result, (False, True)):
        for req, first_row, rows, keys in segments(group):
            for i in range(rows):
                row = first_row + i
                visible = keys - rows + i + 1 if is_causal else keys
                assert row not in got
                got[row] = set(range(visible))
    assert got == want
