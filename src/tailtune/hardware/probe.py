"""Machine facts for `doctor` and `estimate`. No network access.

torch is optional. When it is missing, NVIDIA GPUs are still found through
`nvidia-smi`, so `estimate` can price this machine without torch installed.
"""

from __future__ import annotations

import importlib.util
import os
import platform
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import psutil

GIB = 2**30
Accelerator = Literal["cuda", "rocm", "mps", "xpu", "cpu"]
NVIDIA_SMI_TIMEOUT_S = 8


@dataclass(frozen=True)
class GpuInfo:
    index: int
    name: str
    total_gib: float
    free_gib: float | None
    capability: tuple[int, int] | None
    vendor: Literal["nvidia", "amd", "apple", "intel"]


@dataclass(frozen=True)
class HardwareInfo:
    system: str
    machine: str
    python: str
    wsl2: bool
    hosted: str | None
    accelerator: Accelerator
    gpus: tuple[GpuInfo, ...]
    host_ram_gib: float
    host_ram_available_gib: float
    torch: str | None
    torch_error: str | None
    cuda: str | None
    hip: str | None
    driver: str | None
    real_bf16: bool | None
    gpu_source: Literal["torch", "nvidia-smi", "none"]
    notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def vram_gib(self) -> float | None:
        """Memory of the first device, which is what single-GPU training uses."""
        return self.gpus[0].total_gib if self.gpus else None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["vram_gib"] = self.vram_gib
        return data


@dataclass(frozen=True)
class SmiGpu:
    index: int
    name: str
    total_gib: float
    free_gib: float | None
    capability: tuple[int, int] | None
    driver: str | None


def is_wsl2() -> bool:
    try:
        release = Path("/proc/sys/kernel/osrelease").read_text(encoding="utf-8").lower()
    except OSError:
        return False
    return "microsoft" in release and "wsl2" in release


def hosted_service() -> str | None:
    """Colab or Kaggle, from environment markers. The markers are unconfirmed [U]."""
    if "COLAB_RELEASE_TAG" in os.environ or "COLAB_GPU" in os.environ:
        return "colab"
    if "KAGGLE_KERNEL_RUN_TYPE" in os.environ or "KAGGLE_URL_BASE" in os.environ:
        return "kaggle"
    return None


def _float(text: str) -> float | None:
    try:
        return float(text)
    except ValueError:
        return None


def _capability(text: str) -> tuple[int, int] | None:
    major, _, minor = text.strip().partition(".")
    if major.isdigit() and minor.isdigit():
        return int(major), int(minor)
    return None


def query_nvidia_smi() -> list[SmiGpu] | None:
    """GPUs reported by nvidia-smi, or None when it is absent or fails."""
    exe = shutil.which("nvidia-smi")
    if exe is None:
        return None
    fields = "index,name,memory.total,memory.free,compute_cap,driver_version"
    for query in (fields, fields.replace(",compute_cap", "")):
        try:
            # Fixed argument list from a resolved executable path; no shell, no user input.
            out = subprocess.run(  # noqa: S603
                [exe, f"--query-gpu={query}", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                timeout=NVIDIA_SMI_TIMEOUT_S,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        if out.returncode != 0:
            continue
        gpus: list[SmiGpu] = []
        has_cap = "compute_cap" in query
        for line in out.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < (6 if has_cap else 5):
                continue
            total = _float(parts[2])
            if total is None:
                continue
            free = _float(parts[3])
            gpus.append(
                SmiGpu(
                    index=int(parts[0]) if parts[0].isdigit() else len(gpus),
                    name=parts[1],
                    total_gib=total / 1024,  # MiB
                    free_gib=None if free is None else free / 1024,
                    capability=_capability(parts[4]) if has_cap else None,
                    driver=parts[-1] or None,
                )
            )
        return gpus
    return None


def _probe_torch() -> dict[str, Any]:
    """Facts that need torch. Imported here only, never at module level."""
    result: dict[str, Any] = {
        "torch": None,
        "torch_error": None,
        "cuda": None,
        "hip": None,
        "accelerator": "cpu",
        "gpus": [],
        "real_bf16": None,
    }
    if importlib.util.find_spec("torch") is None:
        return result
    try:
        import torch
    except Exception as exc:  # a broken install is a result, not a crash
        result["torch_error"] = f"{type(exc).__name__}: {exc}"
        return result
    result["torch"] = str(torch.__version__)
    result["cuda"] = torch.version.cuda
    result["hip"] = getattr(torch.version, "hip", None)
    try:
        if torch.cuda.is_available():
            result["accelerator"] = "rocm" if result["hip"] else "cuda"
            vendor = "amd" if result["hip"] else "nvidia"
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                try:
                    free_b, total_b = torch.cuda.mem_get_info(i)
                    free: float | None = free_b / GIB
                    total = total_b / GIB
                except Exception:
                    free, total = None, props.total_memory / GIB
                result["gpus"].append(
                    GpuInfo(
                        index=i,
                        name=str(props.name),
                        total_gib=total,
                        free_gib=free,
                        capability=tuple(torch.cuda.get_device_capability(i)),  # type: ignore[arg-type]
                        vendor=vendor,
                    )
                )
            try:
                result["real_bf16"] = bool(torch.cuda.is_bf16_supported(including_emulation=False))
            except TypeError:  # older torch without the keyword
                major = result["gpus"][0].capability[0] if result["gpus"] else 0
                result["real_bf16"] = major >= 8
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            result["accelerator"] = "mps"
            vm = psutil.virtual_memory()
            result["gpus"].append(
                GpuInfo(0, "Apple GPU (unified memory)", vm.total / GIB, None, None, "apple")
            )
        elif hasattr(torch, "xpu") and torch.xpu.is_available():
            result["accelerator"] = "xpu"
            for i in range(torch.xpu.device_count()):
                props = torch.xpu.get_device_properties(i)
                result["gpus"].append(
                    GpuInfo(i, str(props.name), props.total_memory / GIB, None, None, "intel")
                )
    except Exception as exc:
        result["torch_error"] = f"device query failed: {type(exc).__name__}: {exc}"
    return result


def probe(*, use_torch: bool = True) -> HardwareInfo:
    """Collect machine facts. Never raises for a missing or broken torch."""
    notes: list[str] = []
    tor = (
        _probe_torch()
        if use_torch
        else {
            "torch": None,
            "torch_error": None,
            "cuda": None,
            "hip": None,
            "accelerator": "cpu",
            "gpus": [],
            "real_bf16": None,
        }
    )
    smi = query_nvidia_smi()
    driver = smi[0].driver if smi else None
    gpus: list[GpuInfo] = list(tor["gpus"])
    accelerator: Accelerator = tor["accelerator"]
    source: Literal["torch", "nvidia-smi", "none"] = "torch" if gpus else "none"
    if not gpus and smi:
        gpus = [
            GpuInfo(g.index, g.name, g.total_gib, g.free_gib, g.capability, "nvidia") for g in smi
        ]
        accelerator = "cuda"
        source = "nvidia-smi"
        if tor["torch"] is not None:
            notes.append(
                "An NVIDIA GPU is present, but the installed torch cannot use it "
                "(a CPU-only build?)."
            )
        else:
            notes.append("GPU found with nvidia-smi; torch is not installed.")
    vm = psutil.virtual_memory()
    return HardwareInfo(
        system=platform.system(),
        machine=platform.machine().lower(),
        python=platform.python_version(),
        wsl2=is_wsl2(),
        hosted=hosted_service(),
        accelerator=accelerator,
        gpus=tuple(gpus),
        host_ram_gib=vm.total / GIB,
        host_ram_available_gib=vm.available / GIB,
        torch=tor["torch"],
        torch_error=tor["torch_error"],
        cuda=tor["cuda"],
        hip=tor["hip"],
        driver=driver,
        real_bf16=tor["real_bf16"],
        gpu_source=source,
        notes=tuple(notes),
    )


def python_tuple() -> tuple[int, int]:
    return sys.version_info[:2]
