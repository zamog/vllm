# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Prefix caching for PrefixLM requests, which send their prefix length in
`extra_args["prefix_lm_len"]`: salts each prefix's blocks by its digest, as the KV
of every prefix token depends on the whole prefix, caps hits at the prefix and
prefills the rest of the prompt in one step."""

import copy
import hashlib
from collections.abc import Sequence

import numpy as np

from vllm.logger import init_logger
from vllm.multimodal.inputs import MultiModalFeatureSpec
from vllm.sampling_params import SamplingParams
from vllm.v1.core.kv_cache_manager import KVCacheBlocks
from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.request import Request

logger = init_logger(__name__)

PREFIX_LM_LEN = "prefix_lm_len"
"""`SamplingParams.extra_args` key of the length of a prompt's PrefixLM prefix."""

PREFIX_LM_SALT = "prefix-lm:"
"""Start of the `cache_salt` of every PrefixLM request, and of no other."""


def prefix_lm_len(
    params: SamplingParams | None, prompt_token_ids: Sequence[int] | None
) -> int | None:
    """The valid PrefixLM prefix length a request sends, else None."""
    extra_args = params.extra_args if params is not None else None
    prefix_len = extra_args.get(PREFIX_LM_LEN) if extra_args else None
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


def _request_prefix_lm_len(request: Request) -> int | None:
    return prefix_lm_len(request.sampling_params, request.prompt_token_ids)


class PrefixLMScheduler(Scheduler):
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

    def add_request(self, request: Request) -> None:
        params = request.sampling_params
        prefix_len = _request_prefix_lm_len(request)
        if prefix_len is not None and (
            request.resumable or _straddles(request.mm_features, prefix_len)
        ):
            # Later inputs can cut a streamed prompt short; a media item can
            # be read in only one of the prefix and the rest.
            logger.warning_once(
                "Requests with streamed inputs or a media item across the "
                "prefix end ignore %s",
                PREFIX_LM_LEN,
            )
            assert params is not None
            request.sampling_params = params = copy.copy(params)
            params.extra_args = dict(params.extra_args or {})
            del params.extra_args[PREFIX_LM_LEN]
        # A streaming update re-adds an existing request.
        if request.request_id not in self.requests:
            self._salt_by_kind(request)
        super().add_request(request)

    def _salt_by_kind(self, request: Request) -> None:
        """Salt PrefixLM requests by prefix; keep causal requests out."""
        prefix_len = _request_prefix_lm_len(request)
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
        prefix_len = _request_prefix_lm_len(request)
        if prefix_len is not None:
            # KV past the prefix depends on the rest of the prompt. The last
            # token is always computed to get its logits.
            prefix_len = min(prefix_len, request.num_tokens - 1)
            if max_cache_hit_length is not None:
                prefix_len = min(prefix_len, max_cache_hit_length)
            max_cache_hit_length = prefix_len
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
            and _request_prefix_lm_len(request) is not None
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
            and _request_prefix_lm_len(request) is not None
        )
        if (
            prefix_lm
            and start % self.cache_config.block_size == 0
            and start + num_new_tokens == request.num_tokens
        ):
            # A whole prompt from a block boundary needs no Mamba split, which
            # would stop at a shared-prefix junction.
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


class PrefixLMAsyncScheduler(PrefixLMScheduler, AsyncScheduler): ...
