"""Config arithmetic against real safetensors headers on the Hub. Run with `-m network`."""

from __future__ import annotations

import json

import pytest

from tailtune.hardware.shapes import shapes_from_config

pytestmark = pytest.mark.network

# Repos whose header count should equal config arithmetic to within norms and biases.
REPOS = [
    "Qwen/Qwen3-8B",
    "Qwen/Qwen2.5-0.5B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
    "mistralai/Mistral-Nemo-Instruct-2407",
    "Qwen/Qwen3-30B-A3B",
]


def _header_and_config(repo: str) -> tuple[int, dict[str, object], set[str]]:
    from huggingface_hub import get_safetensors_metadata, hf_hub_download

    meta = get_safetensors_metadata(repo)
    total = sum(meta.parameter_count.values())
    names = {n for f in meta.files_metadata.values() for n in f.tensors}
    with open(hf_hub_download(repo, "config.json"), encoding="utf-8") as fh:
        return total, json.load(fh), names


@pytest.mark.parametrize("repo", REPOS)
def test_parameter_count_matches_the_header(repo: str) -> None:
    total, cfg, _ = _header_and_config(repo)
    ours = shapes_from_config(cfg, repo).p_total
    assert abs(ours - total) / total < 0.001


def test_tied_checkpoint_that_stores_the_head_twice() -> None:
    # Qwen3-0.6B ties embeddings in memory but its file also stores lm_head.weight.
    # Counting header tensors naively overstates P by the vocabulary matrix.
    total, cfg, names = _header_and_config("Qwen/Qwen3-0.6B")
    s = shapes_from_config(cfg)
    assert s.tied
    assert "lm_head.weight" in names
    vocab_matrix = s.vocab * s.hidden
    assert abs((total - vocab_matrix) - s.p_total) / s.p_total < 0.001
