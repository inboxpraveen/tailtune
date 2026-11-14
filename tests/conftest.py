from __future__ import annotations

import json
import socket
from pathlib import Path
from typing import Any

import pytest

FIXTURES = Path(__file__).parent / "fixtures" / "configs"


def config_path(name: str) -> Path:
    return FIXTURES / f"{name}.json"


def load_config(name: str) -> dict[str, Any]:
    return json.loads(config_path(name).read_text(encoding="utf-8"))


@pytest.fixture
def no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail any attempt to open a socket."""

    def refuse(*args: object, **kwargs: object) -> None:
        raise AssertionError("network access attempted")

    monkeypatch.setattr(socket, "socket", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    monkeypatch.setattr(socket, "getaddrinfo", refuse)


@pytest.fixture
def no_gpu(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make the probe see a machine without GPUs or torch, whatever the host has."""
    from tailtune.hardware import probe

    monkeypatch.setattr(probe, "query_nvidia_smi", lambda: None)
    monkeypatch.setattr(
        probe,
        "_probe_torch",
        lambda: {
            "torch": None,
            "torch_error": None,
            "cuda": None,
            "hip": None,
            "accelerator": "cpu",
            "gpus": [],
            "real_bf16": None,
        },
    )
