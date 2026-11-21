# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections import OrderedDict
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from cache_dit import ForwardPattern
from torch import nn

from vllm_omni.diffusion.cache.cachedit import CacheDiTAdapterConfig, CacheDiTBackend
from vllm_omni.diffusion.models.hidream_o1_image.hidream_o1_image_transformer import (
    HiDreamO1TextModel,
)
from vllm_omni.diffusion.models.hidream_o1_image.pipeline_hidream_o1_image import (
    HiDreamO1ImagePipeline,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def _empty_text_model() -> HiDreamO1TextModel:
    model = HiDreamO1TextModel.__new__(HiDreamO1TextModel)
    nn.Module.__init__(model)
    model.layers = nn.ModuleList([nn.Identity()])
    return model


def test_cache_dit_targets_the_decoder_module_that_owns_layers():
    text_model = _empty_text_model()
    pipeline = SimpleNamespace(
        _dit_modules=HiDreamO1ImagePipeline._dit_modules,
        model=SimpleNamespace(
            model=SimpleNamespace(language_model=text_model),
        ),
    )

    with (
        patch("vllm_omni.diffusion.cache.cachedit.backend.BlockAdapter") as block_adapter_cls,
        patch("vllm_omni.diffusion.cache.cachedit.backend.cache_dit") as cache_dit,
    ):
        backend = CacheDiTBackend()
        backend.enable(pipeline)
        backend.refresh(pipeline, num_inference_steps=50)

    adapter_config = HiDreamO1TextModel._cache_dit_adapter_config
    assert isinstance(adapter_config, CacheDiTAdapterConfig)
    assert adapter_config.block_forward_patterns == {"layers": ForwardPattern.Pattern_3}
    assert adapter_config.has_separate_cfg is True

    block_adapter_cls.assert_called_once_with(
        transformer=text_model,
        blocks=[text_model.layers],
        forward_pattern=[ForwardPattern.Pattern_3],
        has_separate_cfg=True,
        check_forward_pattern=True,
    )
    cache_dit.enable_cache.assert_called_once()
    enable_call = cache_dit.enable_cache.call_args
    assert enable_call.args == (block_adapter_cls.return_value,)
    assert enable_call.kwargs["cache_config"].Fn_compute_blocks == 1
    assert enable_call.kwargs["calibrator_config"] is None
    cache_dit.refresh_context.assert_called_once_with(
        text_model,
        num_inference_steps=50,
        verbose=True,
    )


class _MockImageModel:
    def __init__(self) -> None:
        self.call_count = 0
        self.last_kwargs = None

    def __call__(self, **kwargs):
        self.call_count += 1
        self.last_kwargs = kwargs
        batch_size = kwargs["input_ids"].shape[0]
        seq_len = kwargs["position_ids"].shape[-1]
        patch_dim = kwargs["vinputs"].shape[-1]
        return SimpleNamespace(
            x_pred=torch.zeros(
                batch_size,
                seq_len,
                patch_dim,
                dtype=torch.float32,
            )
        )


def _cfg_sample(*, seq_len: int = 4, patch_dim: int = 3):
    return {
        "input_ids": torch.zeros((1, seq_len - 2), dtype=torch.long),
        "position_ids": torch.zeros((3, 1, seq_len), dtype=torch.long),
        "attention_mask": torch.zeros((1, 1, seq_len, seq_len), dtype=torch.float32),
        "full_attn_spans": [[(seq_len - 2, seq_len)]],
        "vinput_mask": torch.tensor([[False, False, True, True]]),
    }


def test_hidream_cfg_samples_can_batch_only_with_matching_shapes():
    cond_sample = _cfg_sample()
    uncond_sample = _cfg_sample()
    assert HiDreamO1ImagePipeline._cfg_samples_can_batch(cond_sample, uncond_sample)

    mismatch = _cfg_sample(seq_len=5)
    mismatch["input_ids"] = torch.zeros((1, 3), dtype=torch.long)
    mismatch["position_ids"] = torch.zeros((3, 1, 5), dtype=torch.long)
    mismatch["attention_mask"] = torch.zeros((1, 1, 5, 5), dtype=torch.float32)
    mismatch["vinput_mask"] = torch.tensor([[False, False, True, True, True]])
    assert not HiDreamO1ImagePipeline._cfg_samples_can_batch(cond_sample, mismatch)


def test_hidream_cfg_batch_runs_one_forward_and_splits_outputs():
    pipeline = HiDreamO1ImagePipeline.__new__(HiDreamO1ImagePipeline)
    pipeline.device = torch.device("cpu")
    pipeline.dtype = torch.float32
    pipeline.model = _MockImageModel()

    cond_sample = _cfg_sample()
    uncond_sample = _cfg_sample()
    z_in = torch.zeros((1, 2, 3), dtype=torch.float32)
    timestep = torch.tensor([0.25], dtype=torch.float32)

    cond_output, uncond_output = pipeline._forward_cfg_batch(
        cond_sample,
        uncond_sample,
        z_in,
        timestep,
    )

    assert pipeline.model.call_count == 1
    assert pipeline.model.last_kwargs["input_ids"].shape == (2, 2)
    assert pipeline.model.last_kwargs["position_ids"].shape == (3, 2, 4)
    assert pipeline.model.last_kwargs["vinputs"].shape == (2, 2, 3)
    assert pipeline.model.last_kwargs["timestep"].shape == (2,)
    assert pipeline.model.last_kwargs["attention_mask"].shape == (2, 1, 4, 4)
    assert pipeline.model.last_kwargs["full_attn_spans"] == [
        [(2, 4)],
        [(2, 4)],
    ]
    assert cond_output.shape == (1, 2, 3)
    assert uncond_output.shape == (1, 2, 3)


def test_hidream_text_sample_cache_reuses_matching_static_sample():
    pipeline = HiDreamO1ImagePipeline.__new__(HiDreamO1ImagePipeline)
    pipeline.od_config = SimpleNamespace(revision="test-revision")
    pipeline.model_dir = "test/model"
    pipeline.dtype = torch.float32
    pipeline.device = torch.device("cpu")
    pipeline.model = SimpleNamespace(
        config=SimpleNamespace(),
        supports_piecewise_attention=True,
    )
    pipeline.tokenizer = object()
    pipeline.processor = object()
    pipeline._text_sample_cache = OrderedDict()
    pipeline._text_sample_cache_hits = 0
    pipeline._text_sample_cache_misses = 0

    build_calls = 0

    def mock_build_t2i_text_sample(**_kwargs):
        nonlocal build_calls
        build_calls += 1
        return {
            "input_ids": torch.zeros((1, 2), dtype=torch.long),
            "position_ids": torch.zeros((3, 1, 4), dtype=torch.long),
            "attention_mask": None,
            "full_attn_spans": [[(2, 4)]],
            "vinput_mask": torch.tensor([[False, False, True, True]]),
            "token_types": torch.tensor([[0, 0, 1, 1]]),
        }

    with (
        patch(
            "vllm_omni.diffusion.models.hidream_o1_image.pipeline_hidream_o1_image.build_t2i_text_sample",
            side_effect=mock_build_t2i_text_sample,
        ),
        patch(
            "vllm_omni.diffusion.models.hidream_o1_image.pipeline_hidream_o1_image.build_hidream_o1_full_attn_spans",
            return_value=[[(2, 4)]],
        ),
    ):
        first = pipeline._build_and_validate_sample(
            "same prompt",
            64,
            64,
            4,
        )
        second = pipeline._build_and_validate_sample(
            "same prompt",
            64,
            64,
            4,
        )

    assert first is second
    assert build_calls == 1
    assert pipeline.get_text_sample_cache_stats() == {
        "hits": 1,
        "misses": 1,
        "entries": 1,
        "capacity": 128,
    }
