"""Model config access: a local path or a Hugging Face Hub id.

Downloads `config.json` only, never weights. Maps Hub failures to tailtune's
error codes: gated or tokenless access, pickle-only weights and remote code are
refusals (exit 4); an unknown id or revision is a usage error (exit 5).
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from tailtune._core.errors import (
    ConfigError,
    GatedModelError,
    HubUnavailableError,
    MissingTokenError,
    UnsafeWeightsError,
    UnsupportedModelError,
)

PICKLE_SUFFIXES = (".bin", ".pt", ".pth", ".ckpt", ".pkl")
SAFE_SUFFIX = ".safetensors"


@dataclass(frozen=True)
class ModelSource:
    """Where a model's config came from."""

    ref: str
    kind: Literal["hub", "local", "hub-cache"]
    config: dict[str, Any]
    revision: str | None = None
    weight_files: tuple[str, ...] = field(default_factory=tuple)

    @property
    def label(self) -> str:
        if self.kind == "local":
            return self.ref
        rev = self.revision[:12] if self.revision else "unresolved revision"
        suffix = " (from local cache, Hub unreachable)" if self.kind == "hub-cache" else ""
        return f"{self.ref} @ {rev}{suffix}"


def check_weight_formats(ref: str, files: list[str]) -> None:
    """Refuse a model whose only weights are pickle files (D-08, TT-M004)."""
    names = [f.lower() for f in files]
    has_safe = any(n.endswith(SAFE_SUFFIX) for n in names)
    has_pickle = any(n.endswith(PICKLE_SUFFIXES) for n in names)
    if has_pickle and not has_safe:
        raise UnsafeWeightsError(
            f"{ref} has no safetensors weights. tailtune does not load pickle files.",
            cause="Only .bin, .pt or similar pickle checkpoints were found (D-08).",
            fix="Use a safetensors revision, or convert the weights yourself in an "
            "isolated environment.",
        )


def _read_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ConfigError(
            f'Invalid value for MODEL: "{path}". The config file could not be read as JSON.',
            cause=str(exc),
        ) from exc
    if not isinstance(data, dict):
        raise ConfigError(f'Invalid value for MODEL: "{path}". config.json must hold an object.')
    return data


def _load_local(ref: str, path: Path) -> ModelSource:
    if path.is_file():
        return ModelSource(ref=ref, kind="local", config=_read_json(path))
    cfg = path / "config.json"
    if not cfg.is_file():
        raise ConfigError(
            f'Invalid value for MODEL: "{ref}". The directory has no config.json.',
            fix="Pass a model directory saved by Transformers, a config.json file, or a Hub id.",
        )
    files = [p.name for p in path.iterdir() if p.is_file()]
    check_weight_formats(ref, files)
    return ModelSource(
        ref=ref, kind="local", config=_read_json(cfg), weight_files=tuple(sorted(files))
    )


def _has_token() -> bool:
    from huggingface_hub import get_token

    return get_token() is not None


def _gated(ref: str) -> Exception:
    if _has_token():
        return GatedModelError(
            f"{ref} is gated, and this account has not been granted access.",
            fix="Accept the licence on the model's Hugging Face page with the same account, "
            "then rerun.",
        )
    return MissingTokenError(
        f"{ref} requires a Hugging Face token, and none was found.",
        fix="Set HF_TOKEN, or log in with the Hugging Face CLI. In notebooks, add HF_TOKEN "
        "to Colab or Kaggle secrets.",
    )


def _load_hub(ref: str, revision: str | None) -> ModelSource:
    from huggingface_hub import HfApi, hf_hub_download
    from huggingface_hub.errors import (
        EntryNotFoundError,
        GatedRepoError,
        HFValidationError,
        LocalEntryNotFoundError,
        RepositoryNotFoundError,
        RevisionNotFoundError,
    )

    try:
        info = HfApi().model_info(ref, revision=revision)
    except HFValidationError as exc:
        raise ConfigError(
            f'Invalid value for MODEL: "{ref}". It is neither a local path nor a valid Hub id.',
            cause=str(exc).splitlines()[0],
        ) from exc
    except GatedRepoError as exc:
        raise _gated(ref) from exc
    except RevisionNotFoundError as exc:
        raise ConfigError(
            f'Invalid value for --revision: "{revision}". {ref} has no such branch, tag or commit.'
        ) from exc
    except RepositoryNotFoundError as exc:
        hint = "" if _has_token() else " If it is private, set HF_TOKEN."
        raise ConfigError(
            f'Invalid value for MODEL: "{ref}". No such model on the Hugging Face Hub.{hint}'
        ) from exc
    except Exception as exc:  # network failure: fall back to the local cache
        try:
            path = hf_hub_download(ref, "config.json", revision=revision, local_files_only=True)
        except (LocalEntryNotFoundError, OSError, ValueError):
            raise HubUnavailableError(
                f"Could not reach the Hugging Face Hub to read {ref}, and its config is not "
                "cached.",
                cause=f"{type(exc).__name__}: {exc}",
                fix="Check the network or proxy settings, or pass a local model directory.",
            ) from exc
        return ModelSource(ref=ref, kind="hub-cache", config=_read_json(Path(str(path))))

    files = [s.rfilename for s in (info.siblings or [])]
    check_weight_formats(ref, files)
    try:
        path = hf_hub_download(ref, "config.json", revision=info.sha)
    except GatedRepoError as exc:
        raise _gated(ref) from exc
    except EntryNotFoundError as exc:
        raise UnsupportedModelError(
            f"{ref} has no config.json, so tailtune cannot read its shapes.",
            fix="Use a model repository saved by Transformers.",
        ) from exc
    except Exception as exc:
        raise HubUnavailableError(
            f"Could not download config.json for {ref}.",
            cause=f"{type(exc).__name__}: {exc}",
            fix="Check the network or proxy settings, then rerun.",
        ) from exc
    return ModelSource(
        ref=ref,
        kind="hub",
        config=_read_json(Path(str(path))),
        revision=info.sha,
        weight_files=tuple(sorted(files)),
    )


def load_config(model: str, revision: str | None = None) -> ModelSource:
    """Read a model's config from a local path or the Hub, without weights."""
    path = Path(model).expanduser()
    if path.exists():
        return _load_local(model, path)
    with warnings.catch_warnings():
        # The Hub client warns on Windows without symlink support; caching still works.
        warnings.filterwarnings("ignore", message=r".*cache-system uses symlinks.*")
        return _load_hub(model, revision)
