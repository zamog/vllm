# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Prefix caching for PrefixLM requests, which send their prefix length in
`SamplingParams.extra_args["prefix_lm_len"]` or
`PoolingParams.extra_kwargs["prefix_lm_len"]`: salts each prefix's blocks by
its digest, as the KV of every prefix token depends on the whole prefix, caps
hits at the prefix and prefills the rest of the prompt in one step."""

import copy
import hashlib
from collections.abc import Sequence

import numpy as np

from vllm.logger import init_logger
from vllm.multimodal.inputs import MultiModalFeatureSpec
from vllm.pooling_params import PoolingParams
from vllm.sampling_params import SamplingParams
from vllm.v1.core.kv_cache_manager import KVCacheBlocks
from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.request import Request

logger = init_logger(__name__)

PREFIX_LM_LEN = "prefix_lm_len"
"""`SamplingParams.extra_args` or `PoolingParams.extra_kwargs` key of the length
of a prompt's PrefixLM prefix."""

PREFIX_LM_SALT = "prefix-lm:"
"""Start of the `cache_salt` of every PrefixLM request, and of no other."""


def _extra(params: SamplingParams | PoolingParams | None) -> dict | None:
    if isinstance(params, SamplingParams):
        return params.extra_args
    if isinstance(params, PoolingParams):
        return params.extra_kwargs
    return None


def prefix_lm_len(
    params: SamplingParams | PoolingParams | None,
    prompt_token_ids: Sequence[int] | None,
) -> int | None:
    """The valid PrefixLM prefix length a request sends, else None."""
    extra = _extra(params)
    prefix_len = extra.get(PREFIX_LM_LEN) if extra else None
    if (
        type(prefix_len) is int
        and prompt_token_ids is not None
        and 0 < prefix_len <= len(prompt_token_ids)
    ):
        return prefix_len
    return None


def prefix_lm_salt(
    prefix_token_ids: Sequence[int],
    cache_salt: str | None,
    mm_features: Sequence[MultiModalFeatureSpec] = (),
) -> str:
    """PrefixLM `cache_salt`: namespace, prefix digest, then the caller's salt.

    The digest covers the media inside the prefix too: their placeholder
    tokens are the same for every item, and every prefix block attends to
    every item."""
    digest = hashlib.sha256(np.asarray(prefix_token_ids, dtype="<i8").tobytes())
    for feature in mm_features:
        if feature.mm_position.offset < len(prefix_token_ids):
            digest.update(
                f"\0{feature.identifier}@{feature.mm_position.offset}".encode()
            )
    salt = PREFIX_LM_SALT + digest.hexdigest()
    return f"{salt}:{cache_salt}" if cache_salt else salt


def _straddles(features: Sequence[MultiModalFeatureSpec], position: int) -> bool:
    """Whether a media item starts before `position` and ends after it."""
    return any(
        f.mm_position.offset < position < f.mm_position.offset + f.mm_position.length
        for f in features
    )


def _request_params(request: Request) -> SamplingParams | PoolingParams | None:
    return request.sampling_params or request.pooling_params


def _drop_prefix_lm_len(request: Request) -> None:
    """Remove the prefix length from a copy of the request's params."""
    params = copy.copy(_request_params(request))
    if isinstance(params, SamplingParams):
        params.extra_args = dict(params.extra_args or {})
        del params.extra_args[PREFIX_LM_LEN]
        request.sampling_params = params
    else:
        assert isinstance(params, PoolingParams)
        params.extra_kwargs = dict(params.extra_kwargs or {})
        del params.extra_kwargs[PREFIX_LM_LEN]
        request.pooling_params = params


class PrefixLMScheduler(Scheduler):
    #: Whether a prompt without a prefix length is a prefix as a whole, for
    #: models that read such a prompt in both directions. Otherwise it is
    #: read causally.
    whole_prompt_prefix_by_default = False

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # Scheduler calls _mamba_block_aligned_split only in this mode.
        if not self.need_mamba_block_aligned_split:
            raise ValueError(
                f"{type(self).__name__} needs a hybrid model in Mamba 'align' "
                "cache mode"
            )
        if not self.mamba_has_prefill_checkpoint_blocks:
            # A prefix is prefilled with the rest of its prompt in one chunk,
            # so its Mamba state is cached inside the chunk only if the backend
            # saves it there; otherwise only at a chunk end on a Mamba block
            # boundary.
            logger.warning(
                "The Mamba backend saves no prefill checkpoints, so %s caches "
                "only PrefixLM prefixes that are whole prompts ending on a "
                "Mamba block boundary",
                type(self).__name__,
            )

    def _prefix_lm_len(self, request: Request) -> int | None:
        """The PrefixLM prefix length of a request, else None (causal)."""
        prompt_token_ids = request.prompt_token_ids
        prefix_len = prefix_lm_len(_request_params(request), prompt_token_ids)
        if (
            prefix_len is None
            and self.whole_prompt_prefix_by_default
            and prompt_token_ids
            and not request.resumable
        ):
            return len(prompt_token_ids)
        return prefix_len

    def add_request(self, request: Request) -> None:
        prefix_len = self._prefix_lm_len(request)
        if prefix_len is not None and (
            request.resumable or _straddles(request.mm_features, prefix_len)
        ):
            # Later inputs can cut a streamed prompt short; a media item can
            # be read in only one of the prefix and the rest.
            logger.warning_once(
                "Requests with streamed inputs or a media item across the "
                "prefix end ignore %s: they are read causally, or by "
                "BidirectionalPrefixLMScheduler as one whole-prompt prefix",
                PREFIX_LM_LEN,
            )
            _drop_prefix_lm_len(request)
        # A streaming update re-adds an existing request.
        if request.request_id not in self.requests:
            self._salt_by_kind(request)
            prefix_len = self._prefix_lm_len(request)
            if prefix_len is not None:
                # KV past the prefix depends on the rest of the prompt, and the
                # last token is always computed for its output. KV connectors
                # (CPU offloading) read the same cap.
                request.max_cache_hit_length = min(
                    prefix_len, len(request.prompt_token_ids or ()) - 1
                )
        super().add_request(request)

    def _salt_by_kind(self, request: Request) -> None:
        """Salt PrefixLM requests by prefix; keep causal requests out."""
        prefix_len = self._prefix_lm_len(request)
        salt = request.cache_salt
        if prefix_len is None:
            if not (salt and salt.startswith(PREFIX_LM_SALT)):
                return
            salt = f"causal:{salt}"
        else:
            assert request.prompt_token_ids is not None
            own = prefix_lm_salt(
                request.prompt_token_ids[:prefix_len], None, request.mm_features
            )
            if salt == own or (salt and salt.startswith(f"{own}:")):
                return
            salt = f"{own}:{salt}" if salt else own
        request.cache_salt = salt
        request.block_hashes = []
        request.update_block_hashes()

    def _get_local_prefix_cache_hit(
        self, request: Request, max_cache_hit_length: int | None = None
    ) -> tuple[KVCacheBlocks, int, int, bool]:
        cap = request.max_cache_hit_length  # set in add_request
        if cap is not None:
            max_cache_hit_length = (
                cap if max_cache_hit_length is None else min(cap, max_cache_hit_length)
            )
        return super()._get_local_prefix_cache_hit(request, max_cache_hit_length)

    def _try_schedule_encoder_inputs(
        self,
        request: Request,
        num_computed_tokens: int,
        num_new_tokens: int,
        encoder_compute_budget: int,
        shift_computed_tokens: int = 0,
    ) -> tuple[list[int], int, int, list[int], list[int]]:
        encoder_cache = self.encoder_cache_manager
        held = set(encoder_cache.get_cached_input_ids(request))
        scheduled = super()._try_schedule_encoder_inputs(
            request,
            num_computed_tokens,
            num_new_tokens,
            encoder_compute_budget,
            shift_computed_tokens,
        )
        if (
            scheduled[1] < num_new_tokens
            and num_computed_tokens < request.num_prompt_tokens
            and self._prefix_lm_len(request) is not None
        ):
            # An encoder budget or cache limit would split the prompt; wait
            # for a step that can encode all of it. Drop the references this
            # call took on cached inputs: a waiting request holding them keeps
            # them from being evicted, which can starve it and others.
            for input_id in encoder_cache.get_cached_input_ids(request) - held:
                encoder_cache.free_encoder_input(request, input_id)
            return [], 0, encoder_compute_budget, [], []
        return scheduled

    def _mamba_block_aligned_split(
        self,
        request: Request,
        num_new_tokens: int,
        num_new_local_computed_tokens: int = 0,
        num_external_computed_tokens: int = 0,
    ) -> int:
        start = (
            request.num_computed_tokens
            + num_new_local_computed_tokens
            + num_external_computed_tokens
        )
        prefix_lm = (
            start < request.num_prompt_tokens
            and self._prefix_lm_len(request) is not None
        )
        if prefix_lm:
            # A prompt resumed inside a Mamba block holds the state of its
            # hit in that block; caching its blocks would key it as the
            # state at the block's end once the step crosses it. Kept past
            # the prompt: a later cache_blocks would key the same block.
            # Hits a KV connector loads skip this split; they end on Mamba
            # block boundaries (the D3 config refuses finer offloaded hits).
            request.skip_writing_prefix_cache = (
                start % self.cache_config.block_size != 0
            )
        if prefix_lm and start + num_new_tokens == request.num_tokens:
            # A whole prompt needs no Mamba split, which would stop at a
            # shared-prefix junction or at the block its hit ends in.
            return num_new_tokens
        num_new_tokens = super()._mamba_block_aligned_split(
            request,
            num_new_tokens,
            num_new_local_computed_tokens,
            num_external_computed_tokens,
        )
        if prefix_lm and start + num_new_tokens < request.num_prompt_tokens:
            # Zero keeps the request waiting at the head of the queue.
            return 0
        return num_new_tokens

    def _update_after_schedule(self, scheduler_output: SchedulerOutput) -> None:
        super()._update_after_schedule(scheduler_output)
        # A prompt split across steps reads its bidirectional ranges in parts,
        # which changes its outputs silently.
        for req_id, num_scheduled in scheduler_output.num_scheduled_tokens.items():
            if not num_scheduled:
                # An async KV load runs no forward pass this step.
                continue
            request = self.requests[req_id]
            assert (
                request.num_computed_tokens >= request.num_prompt_tokens
                or self._prefix_lm_len(request) is None
            ), (
                f"PrefixLM request {req_id} was scheduled in part: "
                f"{request.num_computed_tokens} of {request.num_prompt_tokens} "
                "prompt tokens"
            )


class PrefixLMAsyncScheduler(PrefixLMScheduler, AsyncScheduler): ...


class BidirectionalPrefixLMScheduler(PrefixLMScheduler):
    """PrefixLMScheduler for models that read a prompt without a prefix length
    in both directions: such a prompt is one prefix, cached only for the same
    prompt."""

    whole_prompt_prefix_by_default = True


class BidirectionalPrefixLMAsyncScheduler(
    BidirectionalPrefixLMScheduler, AsyncScheduler
): ...
