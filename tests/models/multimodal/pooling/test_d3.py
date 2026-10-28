# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Decision 3.0 against the checkpoint's own runtime (`trust_remote_code`).

The runtime reads a question's options from the 255 code logits of its readout
at the last prompt token. vLLM's classify head must give the same code logits
for the same prompt tokens, and for an image prompt that vLLM processes itself.
d3-edge has an irregular layer layout; d3-nano ships its backbone as a single
weight file without an index, next to the readout file.
"""

import json
import shutil
from pathlib import Path

import pytest
import torch
from transformers import AutoModel

from vllm import LLM, PoolingParams
from vllm.assets.image import ImageAsset
from vllm.distributed import cleanup_dist_env_and_memory
from vllm.platforms import current_platform
from vllm.transformers_utils.repo_utils import hf_api
from vllm.v1.metrics.reader import Counter

MODELS = ["vllm-sr/d3-edge", "vllm-sr/d3-nano"]
# d3-turbo's runtime, the reference for its prefix_bidir attention mode.
TURBO, TURBO_REVISION = "vllm-sr/d3-turbo", "689b3af198baaf6f11110be38a9daa7188dfd8ad"
TURBO_RUNTIME = [
    "modeling_d3.py",
    "d3_runtime.py",
    "d3_format.py",
    "d3_compose.py",
    "d3_fast.py",
    "d3_kernels.py",
]
STATE = (
    "The order arrived damaged yesterday. The customer has a receipt and asks "
    "for a replacement today."
)
QUESTIONS = {
    "route": {
        "type": "choice",
        "instructions": "Which team should handle this request?",
        "criteria": {
            "returns": "Refunds, replacements and damaged deliveries",
            "billing": "Payments, invoices and charges",
            "technical": "Product setup and faults",
        },
    },
    "receipt": {"type": "noul", "instructions": "Does the customer have a receipt?"},
    "urgency": {
        "type": "score",
        "instructions": "How urgent is this request?",
        "criteria": ["Routine", "Soon", "Today"],
    },
    "department": {
        "type": "choice",
        "criteria": {f"department {i}": None for i in range(40)},
    },
}
IMAGE_QUESTIONS = {"photo": {"type": "noul", "instructions": "Is this a photograph?"}}


def option_probs(logits: torch.Tensor, count: int) -> torch.Tensor:
    return torch.softmax(logits[:count].float(), dim=-1)


def prefix_bidir_edge(tmp_path: Path) -> str:
    """d3-edge's weights under d3-turbo's prefix_bidir attention mode and
    runtime: d3-turbo itself is too large for CI."""
    edge = Path(hf_api().snapshot_download("vllm-sr/d3-edge"))
    for path in edge.iterdir():
        if path.name not in [*TURBO_RUNTIME, "decision_config.json"]:
            (tmp_path / path.name).symlink_to(path)
    for name in TURBO_RUNTIME:
        runtime_file = hf_api().hf_hub_download(TURBO, name, revision=TURBO_REVISION)
        shutil.copy(runtime_file, tmp_path)
    decision = json.loads((edge / "decision_config.json").read_text())
    (tmp_path / "decision_config.json").write_text(
        json.dumps({**decision, "attention_mode": "prefix_bidir"})
    )
    return str(tmp_path)


@pytest.mark.parametrize("model", MODELS)
def test_d3_matches_the_checkpoint_runtime(vllm_runner, model: str) -> None:
    reference = AutoModel.from_pretrained(model, trust_remote_code=True)
    runtime = reference.runtime
    text = runtime.prepare(STATE, QUESTIONS)
    keys = text.runnable
    counts = [len(text.questions[key].keys) for key in keys]
    sequences = [text.sequences[key] for key in keys]
    ref_logits = list(runtime.logits(sequences, counts).cpu())

    image = ImageAsset("stop_sign").pil_image
    visual = runtime.prepare("Describe the attached image.", IMAGE_QUESTIONS, [image])
    (image_key,) = visual.runnable
    image_prompt = visual.texts[image_key]
    ref_logits += list(
        runtime.image_logits(
            [image_prompt], visual.images, [2], visual.lengths[image_key]
        ).cpu()
    )
    counts.append(2)
    del reference, runtime
    cleanup_dist_env_and_memory()

    with vllm_runner(model, runner="pooling", max_model_len=4096) as vllm_model:
        outputs = vllm_model.llm.classify(
            [{"prompt_token_ids": ids} for ids in sequences]
            + [{"prompt": image_prompt, "multi_modal_data": {"image": image}}],
            use_tqdm=False,
        )
    # The head returns raw code logits by default.
    vllm_logits = [torch.tensor(output.outputs.probs) for output in outputs]
    assert all(len(logits) == 255 for logits in vllm_logits)

    for ref, ours, count in zip(ref_logits, vllm_logits, counts):
        ref_probs, our_probs = option_probs(ref, count), option_probs(ours, count)
        torch.testing.assert_close(our_probs, ref_probs, atol=0.02, rtol=0)


def test_d3_prefix_bidir_matches_the_checkpoint_runtime(
    vllm_runner, tmp_path: Path
) -> None:
    """vllm-sr/d3-turbo declares prefix_bidir attention but is too large for
    CI, so d3-edge's weights run under that mode on both sides, with
    d3-turbo's runtime as the reference. Each prompt's state block size goes to
    vLLM in its pooling params, as /v1/systemone sends it."""
    model = prefix_bidir_edge(tmp_path)

    reference = AutoModel.from_pretrained(model, trust_remote_code=True, verify="none")
    runtime = reference.runtime
    text = runtime.prepare(STATE, QUESTIONS)
    keys = text.runnable
    counts = [len(text.questions[key].keys) for key in keys]
    sequences = [text.sequences[key] for key in keys]
    states = [text.states[key] for key in keys]
    ref_logits = list(runtime.logits(sequences, counts, states).cpu())

    image = ImageAsset("stop_sign").pil_image
    visual = runtime.prepare("Describe the attached image.", IMAGE_QUESTIONS, [image])
    (image_key,) = visual.runnable
    image_prompt, length = visual.texts[image_key], visual.lengths[image_key]
    states.append(visual.states[image_key])
    ref_logits += list(
        runtime.image_logits(
            [image_prompt], visual.images, [2], length, [length], states[-1:]
        ).cpu()
    )
    counts.append(2)
    del reference, runtime
    cleanup_dist_env_and_memory()

    with vllm_runner(model, runner="pooling", max_model_len=4096) as vllm_model:
        outputs = vllm_model.llm.classify(
            [{"prompt_token_ids": ids} for ids in sequences]
            + [{"prompt": image_prompt, "multi_modal_data": {"image": image}}],
            pooling_params=[
                PoolingParams(extra_kwargs={"prefix_lm_len": state}) for state in states
            ],
            use_tqdm=False,
        )
    vllm_logits = [torch.tensor(output.outputs.probs) for output in outputs]

    for ref, ours, count in zip(ref_logits, vllm_logits, counts):
        ref_probs, our_probs = option_probs(ref, count), option_probs(ours, count)
        torch.testing.assert_close(our_probs, ref_probs, atol=0.02, rtol=0)


# The Gated DeltaNet prefill must save a state inside a step (FlashInfer, or
# Triton with a prefix match unit); on other platforms the config hook turns
# prefix caching off.
state_caching = pytest.mark.skipif(
    not current_platform.is_cuda(), reason="D3 state caching needs CUDA"
)


def read(llm: LLM, prompts: list[list[int]], state_tokens: int) -> list[torch.Tensor]:
    outputs = llm.classify(
        [{"prompt_token_ids": ids} for ids in prompts],
        pooling_params=[
            PoolingParams(extra_kwargs={"prefix_lm_len": state_tokens}) for _ in prompts
        ],
        use_tqdm=False,
    )
    return [option_probs(torch.tensor(output.outputs.probs), 3) for output in outputs]


def counter(llm: LLM, name: str) -> float:
    return sum(
        m.value for m in llm.get_metrics() if isinstance(m, Counter) and m.name == name
    )


@state_caching
@pytest.mark.parametrize("state_blocks", [2, 2.5])
def test_d3_prefix_bidir_states_reload_from_cpu(
    vllm_runner, tmp_path: Path, monkeypatch, state_blocks: float
) -> None:
    """States evicted from the GPU are read back from CPU memory, all at once,
    with the answers of a run without caching; a state that ends on a Mamba
    block boundary (2 blocks) as well as one that ends inside a block."""
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    model = prefix_bidir_edge(tmp_path)

    def prompts(size: int, question: int) -> list[list[int]]:
        return [
            list(range(1000 + 4000 * i, 1000 + 4000 * i + size))
            + list(range(question + 50 * i, question + 50 * i + 20))
            for i in range(3)
        ]

    with vllm_runner(
        model,
        runner="pooling",
        max_model_len=8192,
        enable_prefix_caching=True,
        kv_offloading_size=2,
        disable_log_stats=False,
    ) as runner:
        llm = runner.llm
        cache_config = llm.llm_engine.vllm_config.cache_config
        size = int(state_blocks * cache_config.block_size)
        for prompt in prompts(size, 200):
            read(llm, [prompt], size)
        for n in range(3):  # later steps submit and finish the stores
            llm.classify([{"prompt_token_ids": [10 + n] * 8}], use_tqdm=False)
        assert llm.reset_prefix_cache()
        loaded = counter(llm, "vllm:external_prefix_cache_hits")
        reloaded = read(llm, prompts(size, 600), size)
        loaded = counter(llm, "vllm:external_prefix_cache_hits") - loaded
        unit = cache_config.prefix_match_unit
    assert loaded >= 3 * (size - unit)

    with vllm_runner(model, runner="pooling", max_model_len=8192) as runner:
        expected = read(runner.llm, prompts(size, 600), size)
    for ours, ref in zip(reloaded, expected):
        torch.testing.assert_close(ours, ref, atol=0.02, rtol=0)


@state_caching
def test_d3_prefix_bidir_caching_settings_at_engine_start(
    vllm_runner, tmp_path: Path, monkeypatch, caplog_vllm
) -> None:
    """The config hook refuses a setting it can't honour, keeps the user's
    scheduler, and logs the settings in effect."""
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    model = prefix_bidir_edge(tmp_path)
    with pytest.raises(ValueError, match="--long-prefill-token-threshold"):
        LLM(
            model=model,
            runner="pooling",
            max_model_len=4096,
            enable_prefix_caching=True,
            long_prefill_token_threshold=512,
        )
    asynchronous = (
        "vllm.v1.core.sched.prefix_lm_scheduler.BidirectionalPrefixLMAsyncScheduler"
    )
    with vllm_runner(
        model,
        runner="pooling",
        max_model_len=4096,
        enable_prefix_caching=True,
        scheduler_cls=asynchronous,
    ) as runner:
        scheduler_config = runner.llm.llm_engine.vllm_config.scheduler_config
        assert scheduler_config.scheduler_cls == asynchronous
        assert scheduler_config.enable_chunked_prefill
    assert f"scheduler {asynchronous}, chunked prefill on" in caplog_vllm.text
