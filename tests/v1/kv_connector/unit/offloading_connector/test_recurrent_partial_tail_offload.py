# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU offloading of a state kept only at a mid-block Mamba checkpoint.

A PrefixLM prefix prefilled in one chunk saves its Mamba state at its hit cap's
last prefix-match-unit boundary, inside a block, stored as a partial tail and
under no complete-chunk key. After the GPU copy is evicted, a request sharing
the prefix loads it from the CPU tier at that position. Hybrid model (full
attention + Mamba "align", block 64, prefix match unit 16),
BidirectionalPrefixLMScheduler, native OffloadingConnector with a fake tier.
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
    OffloadingWorkerMetadata,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading_connector import (
    OffloadingConnector,
)
from vllm.pooling_params import PoolingParams
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
from vllm.v1.core.sched.prefix_lm_scheduler import (
    PREFIX_LM_LEN,
    BidirectionalPrefixLMScheduler,
)
from vllm.v1.core.single_type_kv_cache_manager import register_all_kvcache_specs
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    MambaSpec,
)
from vllm.v1.kv_offload.base import LookupResult
from vllm.v1.outputs import KVConnectorOutput, ModelRunnerOutput
from vllm.v1.request import Request, RequestStatus
from vllm.v1.structured_output import StructuredOutputManager

pytestmark = pytest.mark.cpu_test

BLOCK = 64
HASH = 16


@pytest.fixture(autouse=True)
def model_runner_v2(monkeypatch):
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")


class Harness:
    def __init__(self, num_usable_blocks: int = 64):
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
            prefix_match_unit=HASH,
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
        assert vllm_config.use_v2_model_runner
        mamba = MambaSpec(
            block_size=BLOCK,
            shapes=((1, 1),),
            dtypes=(torch.float32,),
            mamba_cache_mode="align",
            num_prefill_checkpoint_blocks=1,
            prefill_checkpoint_alignment=HASH,
        )
        attn = FullAttentionSpec(
            block_size=BLOCK, num_kv_heads=1, head_size=1, dtype=torch.float32
        )
        num_blocks = num_usable_blocks + 1
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
        self.scheduler = BidirectionalPrefixLMScheduler(
            vllm_config=vllm_config,
            kv_cache_config=kv_cache_config,
            block_size=BLOCK,
            hash_block_size=HASH,
            log_stats=True,
            structured_output_manager=StructuredOutputManager(vllm_config),
        )
        connector = self.scheduler.connector
        assert isinstance(connector, OffloadingConnector)
        self.connector = connector
        self.cs = connector.scheduler
        self.manager = self.cs.manager
        # Stateful fake tier: every offered key is stored and readable.
        self.stored: dict = {}

        def prepare_store(keys, ctx):
            keys = list(keys)
            new = [k for k in keys if k not in self.stored]
            for k in new:
                self.stored[k] = ctx.req_id
            return generate_store_output(new)

        self.manager.prepare_store.side_effect = prepare_store
        self.manager.lookup.side_effect = lambda key, ctx: (
            LookupResult.HIT if key in self.stored else LookupResult.MISS
        )
        self.pending_jobs: set[int] = set()
        init_none_hash(sha256)
        self.hasher = get_request_block_hasher(HASH, sha256)

    def request(self, req_id: str, prompt: list[int], prefix_len: int | None):
        extra = None if prefix_len is None else {PREFIX_LM_LEN: prefix_len}
        return Request(
            request_id=req_id,
            prompt_token_ids=prompt,
            sampling_params=None,
            pooling_params=PoolingParams(task="classify", extra_kwargs=extra),
            block_hasher=self.hasher,
        )

    def step(self, *requests: Request, finished_recving=None):
        for r in requests:
            self.scheduler.add_request(r)
        out = self.scheduler.schedule()
        meta = out.kv_connector_metadata
        assert isinstance(meta, OffloadingConnectorMetadata)
        ids = list(out.num_scheduled_tokens)
        # Report every job of earlier steps complete (stores + loads).
        worker_meta = OffloadingWorkerMetadata(
            completed_jobs={jid: 1 for jid in self.pending_jobs}
        )
        self.pending_jobs = set(meta.store_jobs) | set(meta.load_jobs)
        mro = ModelRunnerOutput(
            ids,
            {rid: i for i, rid in enumerate(ids)},
            [[] for _ in ids],
            pooler_output=[torch.zeros(1) for _ in ids],
            kv_connector_output=KVConnectorOutput(
                finished_recving=finished_recving,
                kv_connector_worker_meta=worker_meta,
            ),
        )
        self.scheduler.update_from_output(out, mro)
        return out, meta


def test_a_question_loads_its_prefix_state_from_cpu_at_the_hit_cap():
    h = Harness()
    s = 150  # its hits stop at 149, so the checkpoint is at 144, inside block 2
    prefix = list(range(1000, 1000 + s))
    warm_up = h.request("w", prefix, prefix_len=None)  # the whole prompt
    out, _ = h.step(warm_up)
    assert out.num_scheduled_tokens == {"w": s}
    assert warm_up.is_finished()
    for _ in range(3):  # the stores complete
        h.step()
    assert h.scheduler.kv_cache_manager.reset_prefix_cache()  # GPU copy gone

    prompt = prefix + list(range(5000, 5020))
    question = h.request("q", prompt, prefix_len=s)
    out, meta = h.step(question)
    assert not out.num_scheduled_tokens
    assert question.status == RequestStatus.WAITING_FOR_REMOTE_KVS
    assert question.num_computed_tokens == 144
    [load] = meta.load_jobs.values()
    # Attention: chunks [0, 128) and the tail block; Mamba: the tail state.
    assert load.dst_spec.group_sizes == [3, 1]
    assert len(load.src_spec.offload_keys) == 4

    h.step(finished_recving={"q"})
    out, _ = h.step()
    assert out.num_scheduled_tokens == {"q": len(prompt) - 144}
    [scheduled] = [r for r in out.scheduled_new_reqs if r.req_id == "q"]
    assert scheduled.block_ids[1][(144 - 1) // BLOCK] != 0  # the loaded state
    assert question.is_finished()
    assert question.skip_writing_prefix_cache  # resumed inside a block
