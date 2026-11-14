"""Hub failures map to the right codes. The Hub client is mocked; no network."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import huggingface_hub
import pytest
from huggingface_hub import errors as hub_errors

from conftest import config_path
from tailtune._core import hub
from tailtune._core.errors import (
    ConfigError,
    GatedModelError,
    HubUnavailableError,
    MissingTokenError,
    UnsafeWeightsError,
)
from tailtune.cli import main


def _make(cls: type[Exception], message: str = "mocked") -> Exception:
    exc = cls.__new__(cls)
    Exception.__init__(exc, message)
    return exc


@dataclass
class _Sibling:
    rfilename: str


@dataclass
class _Info:
    sha: str = "0123456789abcdef0123456789abcdef01234567"
    siblings: list[_Sibling] = field(
        default_factory=lambda: [_Sibling("config.json"), _Sibling("model.safetensors")]
    )


class _Api:
    def __init__(self, info: _Info | None = None, exc: Exception | None = None) -> None:
        self.info, self.exc = info, exc

    def model_info(self, repo_id: str, revision: str | None = None) -> _Info:
        if self.exc is not None:
            raise self.exc
        return self.info or _Info()


def _patch(
    monkeypatch: pytest.MonkeyPatch,
    *,
    info: _Info | None = None,
    info_exc: Exception | None = None,
    download_exc: Exception | None = None,
    token: str | None = None,
    cached: Path | None = None,
) -> None:
    monkeypatch.setattr(huggingface_hub, "HfApi", lambda: _Api(info, info_exc))

    def download(repo_id: str, filename: str, **kwargs: object) -> str:
        if kwargs.get("local_files_only"):
            if cached is None:
                raise _make(hub_errors.LocalEntryNotFoundError)
            return str(cached)
        if download_exc is not None:
            raise download_exc
        return str(config_path("Qwen_Qwen3-0.6B"))

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    monkeypatch.setattr(huggingface_hub, "get_token", lambda: token)


def test_hub_config_is_read_at_the_resolved_commit(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch(monkeypatch)
    src = hub.load_config("Qwen/Qwen3-0.6B")
    assert src.kind == "hub"
    assert src.revision == _Info().sha
    assert src.config["model_type"] == "qwen3"
    assert src.label == "Qwen/Qwen3-0.6B @ 0123456789ab"


def test_gated_without_a_token_is_missing_token(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch(monkeypatch, download_exc=_make(hub_errors.GatedRepoError))
    with pytest.raises(MissingTokenError) as info:
        hub.load_config("meta-llama/Llama-3.1-8B-Instruct")
    assert info.value.code == "TT-M002"
    assert info.value.exit_code == 4


def test_gated_with_a_token_is_not_granted(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch(monkeypatch, info_exc=_make(hub_errors.GatedRepoError), token="hf_x")
    with pytest.raises(GatedModelError) as info:
        hub.load_config("meta-llama/Llama-3.1-8B-Instruct")
    assert info.value.code == "TT-M001"


def test_unknown_repo_is_a_usage_error(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch(monkeypatch, info_exc=_make(hub_errors.RepositoryNotFoundError))
    with pytest.raises(ConfigError) as info:
        hub.load_config("nobody/nothing")
    assert info.value.exit_code == 5
    assert "HF_TOKEN" in info.value.message


def test_unknown_revision_is_a_usage_error(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch(monkeypatch, info_exc=_make(hub_errors.RevisionNotFoundError))
    with pytest.raises(ConfigError, match="--revision"):
        hub.load_config("Qwen/Qwen3-0.6B", revision="nope")


def test_pickle_only_repos_are_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    info = _Info(siblings=[_Sibling("config.json"), _Sibling("pytorch_model.bin")])
    _patch(monkeypatch, info=info)
    with pytest.raises(UnsafeWeightsError) as exc:
        hub.load_config("someone/old-model")
    assert exc.value.code == "TT-M004"


def test_offline_falls_back_to_the_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch(monkeypatch, info_exc=ConnectionError("offline"), cached=config_path("Qwen_Qwen3-8B"))
    src = hub.load_config("Qwen/Qwen3-8B")
    assert src.kind == "hub-cache"
    assert "local cache" in src.label


def test_offline_without_a_cache_exits_1(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch(monkeypatch, info_exc=ConnectionError("offline"))
    with pytest.raises(HubUnavailableError) as info:
        hub.load_config("Qwen/Qwen3-8B")
    assert info.value.exit_code == 1


def test_cli_maps_gated_to_exit_4(monkeypatch: pytest.MonkeyPatch, no_gpu: None) -> None:
    _patch(monkeypatch, download_exc=_make(hub_errors.GatedRepoError))
    assert main(["estimate", "meta-llama/Llama-3.1-8B-Instruct"]) == 4


def test_local_dir_with_only_pickle_weights_is_refused(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text(
        config_path("Qwen_Qwen3-0.6B").read_text(encoding="utf-8"), encoding="utf-8"
    )
    (tmp_path / "pytorch_model.bin").write_bytes(b"")
    with pytest.raises(UnsafeWeightsError):
        hub.load_config(str(tmp_path))
    (tmp_path / "model.safetensors").write_bytes(b"")
    assert hub.load_config(str(tmp_path)).kind == "local"


def test_local_dir_without_config_is_a_usage_error(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match=r"no config\.json"):
        hub.load_config(str(tmp_path))


def test_invalid_json_is_a_usage_error(tmp_path: Path) -> None:
    bad = tmp_path / "config.json"
    bad.write_text("{not json", encoding="utf-8")
    with pytest.raises(ConfigError):
        hub.load_config(str(bad))
