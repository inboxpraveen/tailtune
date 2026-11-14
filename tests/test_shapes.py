from __future__ import annotations

import pytest

from conftest import load_config
from tailtune._core.errors import (
    EXIT_REFUSED,
    RemoteCodeRequiredError,
    TargetTypeError,
    UnsupportedModelError,
)
from tailtune.hardware.shapes import shapes_from_config


def test_qwen3_8b_matches_the_worked_example() -> None:
    # 07 section 9.2: P = 8.19B, P_t = 6.95B across 252 target matrices, 1.24B embeddings.
    s = shapes_from_config(load_config("Qwen_Qwen3-8B"))
    assert len(s.targets) == 252
    assert s.p_total / 1e9 == pytest.approx(8.19, abs=0.005)
    assert s.p_targets / 1e9 == pytest.approx(6.95, abs=0.005)
    assert s.p_embed / 1e9 == pytest.approx(1.24, abs=0.01)
    assert not s.tied
    assert len(s.block) == 7


@pytest.mark.parametrize(
    ("name", "tied"),
    [
        ("Qwen_Qwen3-0.6B", True),
        ("Qwen_Qwen3-1.7B", True),
        ("Qwen_Qwen3-4B", True),
        ("Qwen_Qwen3-8B", False),
        ("Qwen_Qwen3-14B", False),
    ],
)
def test_tied_embeddings_follow_the_config(name: str, tied: bool) -> None:
    s = shapes_from_config(load_config(name))
    assert s.tied is tied
    assert len(s.embed) == (1 if tied else 2)


def test_tied_defaults_to_true_when_the_key_is_absent() -> None:
    cfg = load_config("Qwen_Qwen3-8B")
    del cfg["tie_word_embeddings"]
    assert shapes_from_config(cfg).tied


def test_query_width_can_differ_from_hidden_size() -> None:
    # Qwen3-0.6B: 16 heads x 128 = 2048 query width against hidden size 1024.
    s = shapes_from_config(load_config("Qwen_Qwen3-0.6B"))
    q, k, v, o = s.block[:4]
    assert q == (2048, 1024)
    assert k == v == (1024, 1024)
    assert o == (1024, 2048)


def test_head_dim_falls_back_to_hidden_over_heads() -> None:
    cfg = load_config("Qwen_Qwen3-8B")
    del cfg["head_dim"]
    s = shapes_from_config(cfg)
    assert s.block[0] == (4096, 4096)


def test_moe_experts_count_as_matrices() -> None:
    s = shapes_from_config(load_config("Qwen_Qwen3-235B-A22B"))
    cfg = load_config("Qwen_Qwen3-235B-A22B")
    per_layer = 4 + 3 * cfg["num_experts"] + 1  # attention, experts, router
    assert len(s.targets) == cfg["num_hidden_layers"] * per_layer
    assert s.p_total / 1e9 == pytest.approx(235.09, abs=0.01)


def test_moe_dense_layers_use_the_dense_ffn() -> None:
    cfg = load_config("Qwen_Qwen3-235B-A22B")
    cfg["mlp_only_layers"] = [0]
    s = shapes_from_config(cfg)
    dense = (cfg["intermediate_size"], cfg["hidden_size"])
    assert dense in s.targets[:7]


def test_quantized_checkpoints_are_refused() -> None:
    with pytest.raises(TargetTypeError) as info:
        shapes_from_config(load_config("openai_gpt-oss-120b"), "openai/gpt-oss-120b")
    assert info.value.code == "TT-T003"
    assert info.value.exit_code == EXIT_REFUSED
    assert "mxfp4" in info.value.message


def test_multimodal_configs_are_refused() -> None:
    cfg = {"model_type": "gemma3", "text_config": {"model_type": "gemma3_text"}}
    with pytest.raises(UnsupportedModelError):
        shapes_from_config(cfg)


def test_unknown_architectures_are_refused() -> None:
    cfg = load_config("Qwen_Qwen3-8B") | {"model_type": "granitemoehybrid"}
    with pytest.raises(UnsupportedModelError) as info:
        shapes_from_config(cfg)
    assert info.value.code == "TT-T004"


def test_remote_code_is_named_as_such() -> None:
    cfg = load_config("Qwen_Qwen3-8B") | {"model_type": "custom", "auto_map": {"AutoModel": "x.y"}}
    with pytest.raises(RemoteCodeRequiredError) as info:
        shapes_from_config(cfg)
    assert info.value.code == "TT-M003"


def test_missing_fields_are_refused_clearly() -> None:
    cfg = load_config("Qwen_Qwen3-8B")
    del cfg["hidden_size"]
    with pytest.raises(UnsupportedModelError, match="hidden_size"):
        shapes_from_config(cfg)
