# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A deferred offload store whose source block the next step reuses is flushed.

The worker submits a step's stores at the start of the next step, and the
Model Runner V2 writes that step's new blocks (zeroing, CoW, the Mamba align
pre-copy) in the same step. The scheduler marks every pending store whose
source block it reuses as ``jobs_to_flush``; the runner must submit and wait
for those before it writes (``KVConnector.handle_preemptions``, called first
in ``execute_model``), or the CPU copy holds the new contents (zeros, or
another request's Mamba state read back as K/V: NaN).
Hybrid model (full attention + Mamba "align"), native OffloadingConnector with
the test MockOffloadingSpec, 8 usable blocks.
"""

import pytest
import torch

from tests.v1.kv_connector.unit.offloading_connector.utils import (
    generate_store_output,
)
from vllm.config import (
    CacheConfig,
    KVTransferConfig,
    ModelConfig,
    ParallelConfig,
    SchedulerConfig,
    VllmConfig,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.common import (
    OffloadingConnectorMetadata,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading_connector import (
    OffloadingConnector,
)
from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.core.single_type_kv_cache_manager import register_all_kvcache_specs
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    MambaSpec,
)
from vllm.v1.kv_offload.base import GPULoadStoreSpec
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request
from vllm.v1.structured_output import StructuredOutputManager

pytestmark = pytest.mark.cpu_test

BLOCK = 16
EOS = 50256


@pytest.fixture(autouse=True)
def model_runner_v2(monkeypatch):
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")


def make_scheduler(num_usable_blocks: int) -> Scheduler:
    model_config = ModelConfig(
        model="facebook/opt-125m",
        trust_remote_code=True,
        dtype="float16",
        seed=42,
        skip_tokenizer_init=True,
        max_model_len=2048,
    )
    scheduler_config = SchedulerConfig(
        max_num_seqs=16,
        max_num_batched_tokens=8192,
        max_model_len=2048,
        enable_chunked_prefill=True,
        async_scheduling=False,
        watermark=0.0,
        is_encoder_decoder=model_config.is_encoder_decoder,
    )
    cache_config = CacheConfig(
        block_size=BLOCK,
        gpu_memory_utilization=0.9,
        cache_dtype="auto",
        enable_prefix_caching=True,
        mamba_cache_mode="align",
    )
    kv_transfer_config = KVTransferConfig(
        kv_connector="OffloadingConnector",
        kv_role="kv_both",
        kv_connector_extra_config={
            "spec_name": "MockOffloadingSpec",
            "spec_module_path": "tests.v1.kv_connector.unit.offloading_connector.utils",  # noqa: E501
        },
    )
    vllm_config = VllmConfig(
        scheduler_config=scheduler_config,
        model_config=model_config,
        cache_config=cache_config,
        parallel_config=ParallelConfig(),
        kv_transfer_config=kv_transfer_config,
    )
    assert vllm_config.use_v2_model_runner  # from VLLM_USE_V2_MODEL_RUNNER=1
    mamba = MambaSpec(
        block_size=BLOCK,
        shapes=((1, 1),),
        dtypes=(torch.float32,),
        mamba_cache_mode="align",
        num_prefill_checkpoint_blocks=1,
        prefill_checkpoint_alignment=BLOCK,
    )
    attn = FullAttentionSpec(
        block_size=BLOCK, num_kv_heads=1, head_size=1, dtype=torch.float32
    )
    num_blocks = num_usable_blocks + 1  # block 0 is the null block
    kv_cache_config = KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(["attn"], attn),
            KVCacheGroupSpec(["mamba"], mamba),
        ],
    )
    cache_config.num_gpu_blocks = num_blocks
    register_all_kvcache_specs(vllm_config)
    scheduler = Scheduler(
        vllm_config=vllm_config,
        kv_cache_config=kv_cache_config,
        block_size=BLOCK,
        hash_block_size=None,
        log_stats=True,
        structured_output_manager=StructuredOutputManager(vllm_config),
    )
    scheduler.use_v2_model_runner = True
    assert scheduler.needs_kv_cache_zeroing  # hybrid -> attention blocks zeroed
    connector = scheduler.connector
    assert isinstance(connector, OffloadingConnector)
    manager = connector.scheduler.manager
    manager.prepare_store.side_effect = lambda keys, ctx: generate_store_output(keys)
    return scheduler


def make_request(req_id: str, prompt: list[int]) -> Request:
    init_none_hash(sha256)
    params = SamplingParams(max_tokens=1)
    params.update_from_generation_config({}, EOS)
    return Request(
        request_id=req_id,
        prompt_token_ids=prompt,
        sampling_params=params,
        pooling_params=None,
        block_hasher=get_request_block_hasher(BLOCK, sha256),
    )


def step(scheduler: Scheduler, *requests: Request):
    """One engine step; never reports any transfer completion, so every store
    job the step creates stays in flight (as it would until the worker's
    deferred submission completes)."""
    for request in requests:
        scheduler.add_request(request)
    output = scheduler.schedule()
    ids = list(output.num_scheduled_tokens)
    scheduler.update_from_output(
        output,
        ModelRunnerOutput(ids, {rid: i for i, rid in enumerate(ids)}, [[0]] * len(ids)),
    )
    meta = output.kv_connector_metadata
    assert isinstance(meta, OffloadingConnectorMetadata)
    return output, meta


def store_sources(
    meta: OffloadingConnectorMetadata,
) -> dict[int, tuple[list[int], list[int]]]:
    """job_id -> (source block ids, group_sizes) of this step's store jobs."""
    out = {}
    for job_id, job in meta.store_jobs.items():
        assert isinstance(job.src_spec, GPULoadStoreSpec)
        out[job_id] = (job.src_spec.block_ids.tolist(), list(job.src_spec.group_sizes))
    return out


def new_req(output, req_id):
    [req] = [r for r in output.scheduled_new_reqs if r.req_id == req_id]
    return req


# Two "warm-ups": 40-token prompts, prefilled whole. Each caches attention
# blocks 0,1 (tokens [0,32)) and a Mamba checkpoint at 32 (block idx 1); the
# connector stores both attention chunks and the Mamba boundary state, and the
# worker submits those stores at the start of the NEXT step.
P2 = list(range(1000, 1040))
P1 = list(range(2000, 2040))


def test_store_whose_block_is_zeroed_this_step_is_flushed():
    """(A) The flushed store's source block is a fresh attention block of a new
    request -> in new_block_ids_to_zero of the same step."""
    scheduler = make_scheduler(num_usable_blocks=8)
    _, meta_w2 = step(scheduler, make_request("w2", P2))
    w2_jobs = store_sources(meta_w2)
    assert len(w2_jobs) == 2  # attention chunks [0,1] + Mamba boundary state
    _, meta_w1 = step(scheduler, make_request("w1", P1))
    w1_jobs = store_sources(meta_w1)
    assert len(w1_jobs) == 2

    # A fresh 48-token prompt re-allocates the freed blocks of w2 (LRU head).
    out, meta = step(scheduler, make_request("q3", list(range(3000, 3048))))
    assert meta.jobs_to_flush, "a pending store whose source was re-allocated"
    flushed_src = {
        bid for jid in meta.jobs_to_flush for bid in (w2_jobs | w1_jobs)[jid][0]
    }
    zeroed = set(out.new_block_ids_to_zero or ())
    q3 = new_req(out, "q3")
    assert flushed_src & zeroed, (
        f"flushed store sources {flushed_src} not zeroed this step {zeroed}; "
        f"q3 blocks {q3.block_ids}"
    )


def test_store_whose_block_is_a_mamba_precopy_target_is_flushed():
    """(B) The flushed store's source (w1's attention chunk block) becomes the
    Mamba running column of a question resumed from w2's GPU state -> the V2
    align pre-copy writes w2's SSM state into it before the store copies it."""
    scheduler = make_scheduler(num_usable_blocks=8)
    _, meta_w2 = step(scheduler, make_request("w2", P2))
    w2_jobs = store_sources(meta_w2)
    _, meta_w1 = step(scheduler, make_request("w1", P1))
    w1_jobs = store_sources(meta_w1)

    # Question on w2's state: 40-token prefix + 10 new tokens, hit at 32.
    out, meta = step(scheduler, make_request("q2", P2 + list(range(4000, 4010))))
    q2 = new_req(out, "q2")
    assert q2.num_computed_tokens == 32, q2.num_computed_tokens
    attn_blocks, mamba_blocks = q2.block_ids
    # Mamba align table: [null, w2 state@32, (checkpoint), running col]
    assert mamba_blocks[1] != 0
    running_col = (q2.num_computed_tokens + 18 - 1) // BLOCK
    src_col = (q2.num_computed_tokens - 1) // BLOCK
    assert running_col == 3 and src_col == 1  # pre-copy src != dst -> copies
    running_block = mamba_blocks[running_col]
    assert running_block != 0

    assert meta.jobs_to_flush
    flushed = {jid: (w2_jobs | w1_jobs)[jid] for jid in meta.jobs_to_flush}
    # The running column is a block of w1's ATTENTION store (group 0 source).
    hits = {
        jid: src
        for jid, (src, group_sizes) in flushed.items()
        if running_block in src and group_sizes[0] > 0
    }
    assert hits, (
        f"q2 running block {running_block} is not a flushed attention store "
        f"source; flushed={flushed}, q2.block_ids={q2.block_ids}"
    )
