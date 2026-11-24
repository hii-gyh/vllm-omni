# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from torch import nn

from vllm_omni.diffusion.models.hidream_o1_image import pipeline_hidream_o1_image as pipeline_module
from vllm_omni.diffusion.models.hidream_o1_image.hidream_o1_image_transformer import (
    HiDreamO1ImageTransformer,
    HiDreamO1PrefixKVCache,
    HiDreamO1TextModel,
    merge_prefix_kv,
    validate_prefix_kv_cache,
)
from vllm_omni.diffusion.models.hidream_o1_image.pipeline_hidream_o1_image import (
    HiDreamO1ImagePipeline,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]

PATCH_DIM = 3


def _pipeline(model) -> HiDreamO1ImagePipeline:
    pipeline = HiDreamO1ImagePipeline.__new__(HiDreamO1ImagePipeline)
    pipeline.device = torch.device("cpu")
    pipeline.dtype = torch.float32
    pipeline.model = model
    return pipeline


def _text_sample(*, text_tokens: int = 3, image_tokens: int = 2) -> dict:
    # [text][tms][image]: the causal prefix is [0, text_tokens), tms sits at
    # text_tokens and starts the full span.
    seq_len = text_tokens + 1 + image_tokens
    vinput_mask = [False] * (text_tokens + 1) + [True] * image_tokens
    return {
        "input_ids": torch.zeros((1, text_tokens + 1), dtype=torch.long),
        "position_ids": torch.zeros((3, 1, seq_len), dtype=torch.long),
        "attention_mask": None,
        "full_attn_spans": [[(text_tokens, seq_len)]],
        "vinput_mask": torch.tensor([vinput_mask]),
    }


def test_hidream_prefix_cache_gate_defaults_off_after_h800_measurement():
    # Interleaved forward-level A/B on H800: reuse/baseline 1.02-1.03 @32 text
    # tokens (slower), 0.94-0.95 @298 tokens. Ships off; opt in by prompt length.
    assert pipeline_module.PREFIX_KV_CACHE_MIN_TEXT_TOKENS == 0


def test_hidream_prefix_cache_is_built_only_for_prompts_above_the_gate():
    pipeline = _pipeline(SimpleNamespace(model=SimpleNamespace(language_model=SimpleNamespace(layers=[0, 0, 0]))))

    def sample(text_tokens: int) -> dict:
        return {"input_ids": torch.zeros((1, text_tokens + 1), dtype=torch.long)}

    assert pipeline._prefix_cache_for(sample(4096)) is None

    with patch.object(pipeline_module, "PREFIX_KV_CACHE_MIN_TEXT_TOKENS", 100):
        assert pipeline._prefix_cache_for(sample(99)) is None
        cache = pipeline._prefix_cache_for(sample(100))
        assert isinstance(cache, HiDreamO1PrefixKVCache)
        assert cache.prefix_len == 100
        assert cache.num_layers == 3
        assert cache.keys == [None, None, None]
        assert cache.values == [None, None, None]
        assert cache.filled is False
        # Batched rows (trailing-padded CFG) are not served.
        assert pipeline._prefix_cache_for({"input_ids": torch.zeros((2, 200), dtype=torch.long)}) is None


class _PrefixAwareImageModel:
    """Rows are numbered in the sample's coordinates, like the real model."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        cache = kwargs.get("prefix_cache")
        rows = kwargs["position_ids"].shape[-1]
        start = 0
        if cache is not None and cache.filled:
            start = cache.prefix_len
            rows -= start
        values = torch.arange(start * PATCH_DIM, (start + rows) * PATCH_DIM, dtype=torch.float32)
        if cache is not None:
            cache.filled = True
        return SimpleNamespace(x_pred=values.reshape(1, rows, PATCH_DIM))


def test_hidream_forward_once_fills_then_reuses_prefix_cache():
    model = _PrefixAwareImageModel()
    pipeline = _pipeline(model)
    sample = _text_sample()
    z_in = torch.zeros((1, 2, PATCH_DIM), dtype=torch.float32)
    timestep = torch.tensor([0.25], dtype=torch.float32)

    cache = HiDreamO1PrefixKVCache(prefix_len=3, num_layers=1)
    fill_output = pipeline._forward_once(sample, z_in, timestep, cache)
    assert cache.filled is True
    assert model.calls[-1]["prefix_cache"] is cache

    reuse_output = pipeline._forward_once(sample, z_in, timestep, cache)
    assert reuse_output.shape == (1, 2, PATCH_DIM)

    baseline = pipeline._forward_once(sample, z_in, timestep)
    assert "prefix_cache" not in model.calls[-1]
    assert torch.equal(fill_output, baseline)
    assert torch.equal(reuse_output, baseline)


class _StubLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.calls: list[tuple] = []

    def forward(
        self,
        hidden_states,
        position_embeddings,
        attention_mask,
        full_attn_spans,
        prefix_cache=None,
        layer_idx=0,
    ):
        self.calls.append(
            (hidden_states.shape[1], position_embeddings, attention_mask, full_attn_spans, prefix_cache, layer_idx)
        )
        return hidden_states


def _stub_text_model(num_layers: int = 2) -> HiDreamO1TextModel:
    model = HiDreamO1TextModel.__new__(HiDreamO1TextModel)
    nn.Module.__init__(model)
    model.layers = nn.ModuleList([_StubLayer() for _ in range(num_layers)])
    # cos/sin shaped like the embeddings: (B, S, D).
    model.rotary_emb = lambda embeds, _position_ids: (embeds, embeds)
    model.norm = nn.Identity()
    return model


def test_hidream_text_model_runs_only_suffix_rows_on_reuse():
    model = _stub_text_model()
    embeds = torch.arange(24, dtype=torch.float32).reshape(1, 6, 4)
    position_ids = torch.zeros((3, 1, 6), dtype=torch.long)
    attention_mask = torch.zeros((1, 1, 6, 6))
    spans = [[(3, 6)]]
    cache = HiDreamO1PrefixKVCache(prefix_len=3, num_layers=2)

    fill_output = model(embeds, position_ids, attention_mask, spans, cache)
    assert fill_output.shape == (1, 6, 4)
    assert cache.filled is True
    for idx, layer in enumerate(model.layers):
        length, pe, mask, layer_spans, layer_cache, layer_idx = layer.calls[-1]
        assert length == 6
        assert pe[0].shape == (1, 6, 4)
        assert mask is attention_mask
        assert layer_spans == spans
        assert layer_cache is cache
        assert layer_idx == idx

    reuse_output = model(embeds, position_ids, attention_mask, spans, cache)
    assert reuse_output.shape == (1, 3, 4)
    assert torch.equal(reuse_output, fill_output[:, 3:])
    for idx, layer in enumerate(model.layers):
        length, pe, mask, layer_spans, layer_cache, layer_idx = layer.calls[-1]
        assert length == 3
        assert pe[0].shape == (1, 3, 4)
        assert pe[1].shape == (1, 3, 4)
        assert mask.shape == (1, 1, 3, 6)
        # Attention rebuilds the span as [(prefix_len, S); the caller passes none.
        assert layer_spans is None
        assert layer_cache is cache
        assert layer_idx == idx

    plain_output = model(embeds, position_ids, attention_mask, spans)
    assert torch.equal(plain_output, fill_output)
    for layer in model.layers:
        assert layer.calls[-1][2] is attention_mask
        assert layer.calls[-1][3] == spans
        assert layer.calls[-1][4] is None


def test_hidream_prefix_kv_merge_matches_concat():
    prefix = torch.randn(1, 3, 2, 4)
    current = torch.randn(1, 5, 2, 4)
    merged = merge_prefix_kv(prefix, current)
    assert merged.shape == (1, 8, 2, 4)
    assert torch.equal(merged, torch.cat([prefix, current], dim=1))


def test_hidream_prefix_cache_validation_rejects_unsupported_layouts():
    cache = HiDreamO1PrefixKVCache(prefix_len=3, num_layers=1)
    validate_prefix_kv_cache(cache, 1, 6, [[(3, 6)]])

    with pytest.raises(ValueError, match="batch_size == 1"):
        validate_prefix_kv_cache(cache, 2, 6, [[(3, 6)], [(3, 6)]])
    with pytest.raises(ValueError, match="prefix_len=0"):
        validate_prefix_kv_cache(HiDreamO1PrefixKVCache(prefix_len=0, num_layers=1), 1, 6, [[(0, 6)]])
    with pytest.raises(ValueError, match="does not start at the full span"):
        validate_prefix_kv_cache(cache, 1, 6, [[(2, 6)]])
    with pytest.raises(ValueError, match="exactly one full_attn_spans"):
        validate_prefix_kv_cache(cache, 1, 6, None)


def test_hidream_transformer_forwards_prefix_cache_instead_of_swallowing_it():
    recorded: dict = {}

    def recording_model(**kwargs):
        recorded.update(kwargs)
        return SimpleNamespace(x_pred=None)

    transformer = HiDreamO1ImageTransformer.__new__(HiDreamO1ImageTransformer)
    nn.Module.__init__(transformer)
    transformer.model = recording_model
    cache = HiDreamO1PrefixKVCache(prefix_len=1, num_layers=1)

    transformer(
        input_ids=torch.zeros((1, 2), dtype=torch.long),
        position_ids=torch.zeros((3, 1, 3), dtype=torch.long),
        vinputs=torch.zeros((1, 1, PATCH_DIM)),
        timestep=torch.zeros(1),
        attention_mask=None,
        full_attn_spans=[[(1, 3)]],
        prefix_cache=cache,
    )
    assert recorded["prefix_cache"] is cache
