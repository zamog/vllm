# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PrefixLMScheduler on a full attention + Mamba ("align" cache mode) model."""

from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from vllm.config import ModelConfig, SchedulerConfig, VllmConfig
from vllm.multimodal.inputs import MultiModalFeatureSpec, PlaceholderRange
from vllm.platforms import current_platform
from vllm.pooling_params import PoolingParams
from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.core.encoder_cache_manager import EncoderCacheManager
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
from vllm.v1.core.sched import prefix_lm_scheduler
from vllm.v1.core.sched.prefix_lm_scheduler import (
    PREFIX_LM_LEN,
    BidirectionalPrefixLMScheduler,
    PrefixLMAsyncScheduler,
    PrefixLMScheduler,
    prefix_lm_salt,
)
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.kv_cache_interface import MambaSpec
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request, RequestStatus

from .utils import EOS_TOKEN_ID, create_scheduler

pytestmark = pytest.mark.cpu_test

#: A Mamba backend that saves prefill checkpoints, as FlashInfer GDN does.
MAMBA = MambaSpec(
    block_size=16,
    shapes=((1, 1),),
    dtypes=(torch.float32,),
    mamba_cache_mode="align",
    num_prefill_checkpoint_blocks=1,
    prefill_checkpoint_alignment=16,
)


def make_scheduler(budget: int = 256, cls: type[Scheduler] = PrefixLMScheduler):
    return create_scheduler(
        max_num_batched_tokens=budget,
        max_model_len=2048,
        enable_prefix_caching=True,
        skip_tokenizer_init=True,
        mamba_spec=MAMBA,
        scheduler_cls=cls,
    )


def image(identifier: str, offset: int, length: int) -> MultiModalFeatureSpec:
    return MultiModalFeatureSpec(
        data=None,
        modality="image",
        identifier=identifier,
        mm_position=PlaceholderRange(offset=offset, length=length),
    )


def make_request(
    req_id,
    prompt,
    prefix_len,
    salt="own",
    resumable=False,
    mm_features=None,
    pooling=False,
):
    """A PrefixLM request with the strategy's salt, or causal if prefix_len is None."""
    if salt == "own":
        salt = (
            None
            if prefix_len is None
            else prefix_lm_salt(prompt[:prefix_len], None, mm_features or ())
        )
    extra_args = None if prefix_len is None else {PREFIX_LM_LEN: prefix_len}
    sampling_params, pooling_params = None, None
    if pooling:
        pooling_params = PoolingParams(task="classify", extra_kwargs=extra_args)
    else:
        sampling_params = SamplingParams(max_tokens=1, extra_args=extra_args)
        sampling_params.update_from_generation_config({}, EOS_TOKEN_ID)
    init_none_hash(sha256)
    return Request(
        request_id=req_id,
        prompt_token_ids=prompt,
        sampling_params=sampling_params,
        pooling_params=pooling_params,
        cache_salt=salt,
        resumable=resumable,
        mm_features=mm_features,
        block_hasher=get_request_block_hasher(16, sha256),
    )


def step(scheduler: Scheduler, *requests: Request) -> dict[str, tuple[int, int]]:
    """Add the requests and run one step. Returns the cached and scheduled
    tokens of each new request."""
    for request in requests:
        scheduler.add_request(request)
    output = scheduler.schedule()
    ids = list(output.num_scheduled_tokens)
    scheduler.update_from_output(
        output,
        ModelRunnerOutput(ids, {i: n for n, i in enumerate(ids)}, [[0]] * len(ids)),
    )
    return {
        r.req_id: (r.num_computed_tokens, output.num_scheduled_tokens[r.req_id])
        for r in output.scheduled_new_reqs
    }


@pytest.mark.parametrize("cls", [PrefixLMScheduler, PrefixLMAsyncScheduler])
def test_hits_stop_at_the_prefix_and_prompts_prefill_whole(cls):
    """Hits stop at a prefix of the same tokens and length, and the rest of the
    prompt is prefilled in one step, also where Mamba alignment would split it."""
    scheduler = make_scheduler(cls=cls)
    prefix = list(range(100, 140))
    # The questions share their first 24 tokens: blocks [0, 64) match.
    q1, q2, q3 = (prefix + list(range(200, 224)) + [i] * 6 for i in (1, 2, 3))

    assert step(scheduler, make_request("q1", q1, 40)) == {"q1": (0, 70)}
    # q1 left Mamba state only at 64, past the prefix. q2's attention hit at 32
    # is a junction where alignment would stop, and the step would wait forever.
    assert step(scheduler, make_request("q2", q2, 40)) == {"q2": (0, 70)}
    # The prefix alone leaves its state at 32.
    assert step(scheduler, make_request("prefix", prefix, 40)) == {"prefix": (0, 40)}
    assert step(
        scheduler, make_request("q3", q3, 40), make_request("q1 again", q1, 40)
    ) == {"q3": (32, 38), "q1 again": (32, 38)}
    # Another prefix length, or none, shares no block with these prompts.
    assert step(
        scheduler, make_request("p48", q1, 48), make_request("causal", q1, None)
    ) == {"p48": (0, 70), "causal": (0, 70)}


def test_pooling_requests_send_their_prefix_in_extra_kwargs():
    """A pooling request's prefix length comes from its extra_kwargs: its
    hits stop at the prefix like a sampling request's."""
    scheduler = make_scheduler()
    prefix = list(range(100, 140))
    q1, q2 = (prefix + [i] * 30 for i in (1, 2))

    def run(request: Request) -> tuple[int, int]:
        scheduler.add_request(request)
        output = scheduler.schedule()
        scheduled = output.num_scheduled_tokens[request.request_id]
        scheduler.finish_requests(request.request_id, RequestStatus.FINISHED_STOPPED)
        return request.num_computed_tokens - scheduled, scheduled

    assert run(make_request("prefix", prefix, 40, pooling=True)) == (0, 40)
    assert run(make_request("q1", q1, 40, pooling=True)) == (32, 38)
    # The same prompt as a sampling request has the same salt and hits too.
    assert run(make_request("q2", q2, 40)) == (32, 38)
    assert make_request("q", q1, 40, pooling=True).cache_salt == prefix_lm_salt(
        prefix, None
    )
    # The prefix length can be dropped from pooling params too.
    across = make_request(
        "across", q1, 40, salt=None, mm_features=[image("a", 30, 20)], pooling=True
    )
    scheduler.add_request(across)
    assert PREFIX_LM_LEN not in across.pooling_params.extra_kwargs


def test_a_bidirectional_prompt_without_prefix_length_is_one_prefix():
    """For a model that reads a prompt without a prefix length in both
    directions, the prompt is one prefix: it hits only the same prompt."""
    scheduler = make_scheduler(cls=BidirectionalPrefixLMScheduler)
    first = list(range(100, 170))
    other = first[:40] + [7] * 30
    assert step(scheduler, make_request("first", first, None)) == {"first": (0, 70)}
    # The same first 40 tokens, but every block depends on the whole prompt.
    assert step(scheduler, make_request("other", other, None)) == {"other": (0, 70)}
    # The same prompt again hits up to its Mamba checkpoint at 64.
    assert step(scheduler, make_request("again", first, None)) == {"again": (64, 6)}
    request = make_request("salted", first, None)
    scheduler.add_request(request)
    assert request.cache_salt == prefix_lm_salt(first, None)
    # PrefixLMScheduler reads the same prompt causally, with no salt.
    causal = make_request("causal", first, None)
    make_scheduler().add_request(causal)
    assert causal.cache_salt is None


def test_a_split_prefix_lm_prompt_fails_loudly(monkeypatch):
    """If a change lets the stock Mamba split cut a PrefixLM prompt, the step
    fails instead of reading the prompt's ranges in parts."""
    monkeypatch.setattr(
        PrefixLMScheduler,
        "_mamba_block_aligned_split",
        Scheduler._mamba_block_aligned_split,
    )
    scheduler = make_scheduler(64)
    with pytest.raises(AssertionError, match="scheduled in part: 64 of 70"):
        step(scheduler, make_request("q", list(range(70)), 40))
    # A causal prompt may be split.
    assert step(make_scheduler(64), make_request("c", list(range(70)), None)) == {
        "c": (0, 64)
    }


def test_an_async_kv_load_step_is_not_a_split(monkeypatch):
    """A request loading its KV asynchronously is listed with 0 tokens while
    its computed tokens count the external hit."""
    monkeypatch.setattr(Scheduler, "_update_after_schedule", lambda *_: None)
    scheduler = make_scheduler()
    request = make_request("q", list(range(70)), 40)
    scheduler.add_request(request)
    request.num_computed_tokens = 32
    output = SimpleNamespace(num_scheduled_tokens={"q": 0})
    scheduler._update_after_schedule(output)
    with pytest.raises(AssertionError, match="scheduled in part"):
        scheduler._update_after_schedule(SimpleNamespace(num_scheduled_tokens={"q": 6}))


@pytest.mark.parametrize(
    "cls,prefix_len,cap",
    [
        (PrefixLMScheduler, 40, 40),
        (PrefixLMScheduler, None, None),
        (BidirectionalPrefixLMScheduler, None, 70),
    ],
)
def test_prefix_requests_cap_every_cache_hit(cls, prefix_len, cap):
    """KV connectors read the request's cap: a loaded hit must stop at the
    prefix like a local one."""
    request = make_request("q", list(range(70)), prefix_len)
    make_scheduler(cls=cls).add_request(request)
    assert request.max_cache_hit_length == cap


def test_prompt_one_token_over_budget_waits():
    # a leaves 32 tokens of budget for b's 33, and 32 is block aligned.
    scheduler = make_scheduler(102)
    a, b = list(range(70)), list(range(100, 133))
    assert step(scheduler, make_request("a", a, 40), make_request("b", b, 20)) == {
        "a": (0, 70)
    }
    assert step(scheduler) == {"b": (0, 33)}


def test_waiting_prompt_holds_the_queue():
    """A PrefixLM prompt that does not fit is not skipped by the causal c behind it."""
    scheduler = make_scheduler(100)
    a, b, c = list(range(64)), list(range(100, 164)), list(range(200, 216))
    assert step(
        scheduler,
        make_request("a", a, None),
        make_request("b", b, 32),
        make_request("c", c, None),
    ) == {"a": (0, 64)}
    assert step(scheduler) == {"b": (0, 64), "c": (0, 16)}


@pytest.mark.parametrize(
    "prefix_len,salt,hit,expected",
    [
        (40, None, 32, "{own}"),
        (40, "{own}", 32, "{own}"),
        (40, "{own}:user", 0, "{own}:user"),
        (40, "user", 0, "{own}:user"),
        (40, "{p1}", 0, "{own}:{p1}"),
        (None, "{own}", 0, "causal:{own}"),
    ],
)
def test_requests_get_the_salt_of_their_prefix(prefix_len, salt, hit, expected):
    """A request lacking its prefix's salt is moved into it, a forged one misses,
    and a causal request leaves the PrefixLM namespace."""
    scheduler = make_scheduler()
    p1 = list(range(100, 140))
    p2 = p1[:35] + [7] * 5
    step(scheduler, make_request("p1", p1, 40))
    # Same length and first 32 tokens, but another prefix: no hit.
    assert step(scheduler, make_request("p2", p2, 40)) == {"p2": (0, 40)}
    salts = {"own": prefix_lm_salt(p2, None), "p1": prefix_lm_salt(p1, None)}
    salt = salt and salt.format(**salts)
    request = make_request("q", p2 + list(range(200, 230)), prefix_len, salt=salt)
    assert step(scheduler, request) == {"q": (hit, 70 - hit)}
    assert request.cache_salt == expected.format(**salts)


def test_streamed_requests_are_read_causally():
    request = make_request("streamed", list(range(150)), 40, resumable=True)
    assert step(make_scheduler(100), request) == {"streamed": (0, 96)}


def test_media_inside_the_prefix_are_part_of_its_salt():
    prompt = list(range(100))
    salts = {
        name: make_request(name, prompt, 64, salt=None, mm_features=media)
        for name, media in {
            "a": [image("a", 10, 20)],
            "a_again": [image("a", 10, 20)],
            "b": [image("b", 10, 20)],
            "a_after": [image("a", 10, 20), image("c", 70, 20)],
        }.items()
    }
    scheduler = make_scheduler()
    for request in salts.values():
        scheduler.add_request(request)
    salt = {name: request.cache_salt for name, request in salts.items()}
    assert salt["a"] == salt["a_again"] == salt["a_after"]  # only media in the prefix
    assert salt["a"] != salt["b"]
    assert salt["a"] == prefix_lm_salt(prompt[:64], None, [image("a", 10, 20)])
    assert salt["a"] != prefix_lm_salt(prompt[:64], None)


def test_media_across_the_prefix_end_reads_causally():
    request = make_request(
        "across", list(range(100)), 64, salt=None, mm_features=[image("a", 50, 20)]
    )
    make_scheduler().add_request(request)
    assert PREFIX_LM_LEN not in request.sampling_params.extra_args


def test_encoder_limits_never_split_a_prefix_lm_prompt(monkeypatch):
    # An encoder budget that only fits the tokens before the image.
    monkeypatch.setattr(
        Scheduler,
        "_try_schedule_encoder_inputs",
        lambda self, request, computed, new, budget, shift=0: ([], 10, budget, [], []),
    )
    scheduler = make_scheduler()
    media = [image("a", 10, 20)]
    prefix_lm = make_request("p", list(range(100)), 64, mm_features=media)
    causal = make_request("c", list(range(100)), None, mm_features=media)
    for request, expected in ((prefix_lm, 0), (causal, 10)):
        scheduled = scheduler._try_schedule_encoder_inputs(request, 0, 100, 500)
        assert scheduled == ([], expected, 500, [], [])


def test_waiting_for_the_encoder_holds_no_cached_inputs():
    """A PrefixLM prompt that waits for encoder budget must not keep cached
    inputs from being evicted, or it and others can wait forever."""
    scheduler = make_scheduler(budget=512)
    scheduler.encoder_cache_manager = EncoderCacheManager(cache_size=160)
    scheduler.max_num_encoder_input_tokens = 160

    def run(*requests):
        for request in requests:
            scheduler.add_request(request)
        output = scheduler.schedule()
        ids = list(output.num_scheduled_tokens)
        prefilled = [
            [0]
            if scheduler.requests[i].num_computed_tokens
            >= len(scheduler.requests[i].prompt_token_ids)
            else []
            for i in ids
        ]
        scheduler.update_from_output(
            output,
            ModelRunnerOutput(ids, {i: n for n, i in enumerate(ids)}, prefilled),
        )
        return output.num_scheduled_tokens

    # Images X and A end up cached and freeable.
    run(make_request("x", list(range(40)), None, mm_features=[image("X", 0, 32)]))
    run(make_request("a", list(range(50, 90)), None, mm_features=[image("A", 0, 32)]))
    # A running prompt takes most of the next step's encoder budget.
    run(
        make_request(
            "g", list(range(100, 700)), None, mm_features=[image("G", 520, 64)]
        )
    )
    r0 = make_request(
        "r0",
        list(range(1000, 1200)),
        150,
        mm_features=[image("X", 0, 32), image("Y", 40, 100)],
    )
    r1 = make_request(
        "r1",
        list(range(2000, 2200)),
        150,
        mm_features=[image("A", 0, 32), image("B", 40, 100)],
    )
    scheduled = set()
    for i in range(5):
        scheduled |= set(run(*([r0, r1] if i == 0 else [])))
        if i == 0:
            # Both wait for the encoder, holding no references to X or A.
            assert not scheduled & {"r0", "r1"}
            assert not scheduler.encoder_cache_manager.cached["X"]
            assert not scheduler.encoder_cache_manager.cached["A"]
    assert {"r0", "r1"} <= scheduled


def test_needs_mamba_align_mode():
    with pytest.raises(ValueError, match="align"):
        create_scheduler(skip_tokenizer_init=True, scheduler_cls=PrefixLMScheduler)


@pytest.mark.parametrize("checkpoint_blocks", [0, 1])
def test_warns_when_prefixes_cannot_be_cached(monkeypatch, checkpoint_blocks):
    warning = Mock()
    monkeypatch.setattr(prefix_lm_scheduler.logger, "warning", warning)
    create_scheduler(
        enable_prefix_caching=True,
        skip_tokenizer_init=True,
        mamba_spec=replace(MAMBA, num_prefill_checkpoint_blocks=checkpoint_blocks),
        scheduler_cls=PrefixLMScheduler,
    )
    assert warning.called == (checkpoint_blocks == 0)


@pytest.mark.parametrize("name", ["PrefixLM", "BidirectionalPrefixLM"])
@pytest.mark.parametrize("kind", ["", "Async"])
def test_async_scheduling_takes_the_async_twin(monkeypatch, name, kind):
    # Keep the platform from deciding async scheduling.
    monkeypatch.setattr(
        type(current_platform), "check_and_update_config", classmethod(lambda *_: None)
    )
    cls = f"vllm.v1.core.sched.prefix_lm_scheduler.{name}{{}}Scheduler"
    config = VllmConfig(
        model_config=ModelConfig(model="facebook/opt-125m", skip_tokenizer_init=True),
        scheduler_config=SchedulerConfig.default_factory(
            scheduler_cls=cls.format(""), async_scheduling=kind == "Async"
        ),
    )
    assert config.scheduler_config.scheduler_cls == cls.format(kind)
