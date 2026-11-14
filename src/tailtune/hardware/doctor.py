"""`tailtune doctor`: checks D01 to D18 from docs/plan/07-hardware.md section 14.

Rules: nothing heavy is imported until a check needs it, a missing torch is a
result rather than a crash, and no check touches the network.
"""

from __future__ import annotations

import os
import platform
import shutil
import struct
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

from packaging.version import InvalidVersion, Version

from tailtune._core.versions import describe_range, installed_version, standing
from tailtune.hardware.probe import HardwareInfo, probe

Status = Literal["PASS", "WARN", "FAIL", "SKIP"]
_RANK = {"SKIP": 0, "PASS": 1, "WARN": 2, "FAIL": 3}

TESTED_PYTHON = ((3, 11), (3, 13))
DRIVER_FLOOR = 580
DISK_FLOOR_GIB = 60.0  # [J] model cache plus bases cache for one 8B model
SVD_SLOW_HOURS = 8.0  # [J] above this, CPU SVD for an 8B model is called slow
# Host RAM needed by sst/host-bases relative to its GPU need, from the Qwen3-8B worked
# example (47.2 GiB host for 137.7 GiB GPU, 07 section 9.2). Computed [C], not measured.
HOST_TO_GPU_RATIO = 47.2 / 137.7

T4_PROMISE = (
    "lora on a 4-bit base up to ~8B, lora up to ~4B, osf up to ~1B, "
    "sst up to ~0.7B (~1.1B with 8-bit Adam, a variant)"
)
TIER_PROMISES = {
    "Tier 1": "every v0.1 method (lora, lora on a 4-bit base, osf, sst, full, freeze) within "
    "the limits estimate prints",
    "Tier 1 at release gate": "every v0.1 method within the limits estimate prints, tested "
    "once per release",
    "Tier 1 reduced": T4_PROMISE,
    "Tier 1 numerics": "tests only. CPU is a test platform for tailtune, not a training target",
    "Tier 2": "installs and is expected to train. Known limits are documented; fixes are best "
    "effort",
    "Tier 3": "community support. doctor recognises the platform; there is no CI for it",
    "Unsupported": "nothing. This platform is not supported and gets no fixes",
    "Unknown": "unknown until the GPU can be identified",
}


@dataclass(frozen=True)
class Check:
    id: str
    name: str
    status: Status
    value: str
    note: str = ""


@dataclass(frozen=True)
class Tier:
    label: str
    reason: str

    @property
    def promise(self) -> str:
        return TIER_PROMISES.get(self.label, TIER_PROMISES["Unknown"])


@dataclass
class DoctorReport:
    version: str
    checks: list[Check]
    tier: Tier
    hardware: HardwareInfo
    elapsed_s: float = 0.0
    notes: list[str] = field(default_factory=list)

    @property
    def failed(self) -> list[Check]:
        return [c for c in self.checks if c.status == "FAIL"]

    def to_dict(self) -> dict[str, Any]:
        return {
            "tailtune": self.version,
            "tier": self.tier.label,
            "tier_reason": self.tier.reason,
            "promised_from_v0_1": self.tier.promise,
            "checks": [asdict(c) for c in self.checks],
            "failed": [c.id for c in self.failed],
            "hardware": self.hardware.to_dict(),
            "elapsed_s": round(self.elapsed_s, 2),
            "notes": self.notes,
        }


# ---- tier ---------------------------------------------------------------------------------


def _mac_major() -> int | None:
    release = platform.mac_ver()[0]
    head = release.split(".")[0] if release else ""
    return int(head) if head.isdigit() else None


def _torch_at_least(hw: HardwareInfo, floor: str) -> bool:
    if hw.torch is None:
        return False
    try:
        return Version(Version(hw.torch).base_version) >= Version(floor)
    except InvalidVersion:
        return False


def classify(hw: HardwareInfo) -> Tier:
    """The D-17 support tier for this machine (07 sections 1 and 2)."""
    if sys.version_info[:2] < TESTED_PYTHON[0]:
        return Tier("Unsupported", "Python 3.10 or earlier")
    if hw.system == "Darwin" and hw.machine in {"x86_64", "amd64"}:
        return Tier("Unsupported", "Intel Mac: PyTorch publishes macOS wheels for arm64 only")
    if hw.accelerator == "cpu":
        return Tier("Tier 1 numerics", "no supported GPU found")
    if hw.accelerator == "mps":
        mac = _mac_major()
        if mac is not None and mac >= 15 and _torch_at_least(hw, "2.15"):
            return Tier("Tier 2", "Apple MPS on macOS 15+ with torch 2.15+")
        return Tier("Unsupported", "MPS needs macOS 15+ and torch 2.15+")
    if hw.accelerator == "xpu":
        return Tier("Tier 3", "Intel XPU")
    gpu = hw.gpus[0]
    if hw.accelerator == "rocm":
        if "MI" in gpu.name.upper().replace("AMD", ""):
            return Tier("Tier 2", f"AMD Instinct ({gpu.name})")
        return Tier("Tier 3", f"AMD Radeon ({gpu.name})")
    cc = gpu.capability
    if cc is None:
        return Tier("Unknown", f"{gpu.name}: compute capability not readable")
    if cc < (7, 5):
        return Tier(
            "Unsupported",
            f"compute capability {cc[0]}.{cc[1]}: Pascal, Volta and "
            "Maxwell are not supported by CUDA 13",
        )
    if hw.system == "Windows":
        return Tier("Tier 2", "NVIDIA on native Windows: single GPU only")
    if hw.wsl2:
        return Tier("Tier 2", "NVIDIA under WSL2: single GPU, host RAM capped by WSL")
    if hw.machine not in {"x86_64", "amd64"}:
        return Tier("Tier 2", f"NVIDIA on Linux {hw.machine}")
    if cc == (7, 5):
        return Tier("Tier 1 reduced", f"{gpu.name}, compute capability 7.5 (Turing)")
    if (8, 0) <= cc <= (9, 0):
        return Tier("Tier 1", f"{gpu.name}, compute capability {cc[0]}.{cc[1]}")
    if cc in {(10, 0), (12, 0)}:
        return Tier("Tier 1 at release gate", f"{gpu.name}, compute capability {cc[0]}.{cc[1]}")
    return Tier("Tier 3", f"{gpu.name}, compute capability {cc[0]}.{cc[1]} is not yet classified")


# ---- individual checks -------------------------------------------------------------------


def check_python() -> Check:
    v = sys.version_info[:2]
    text = f"{v[0]}.{v[1]}"
    if v < TESTED_PYTHON[0]:
        return Check("D01", "Python", "FAIL", text, "3.11 or later is required")
    if v > TESTED_PYTHON[1]:
        return Check("D01", "Python", "WARN", text, "newer than the tested 3.11 to 3.13")
    return Check("D01", "Python", "PASS", text)


def check_os(hw: HardwareInfo) -> Check:
    value = f"{hw.system} {hw.machine}" + (", WSL2" if hw.wsl2 else "")
    if hw.system == "Darwin" and hw.machine in {"x86_64", "amd64"}:
        return Check("D02", "Platform", "FAIL", value, "Intel Mac: no current torch wheel")
    if hw.system == "Linux" and hw.machine in {"x86_64", "amd64"} and not hw.wsl2:
        return Check("D02", "Platform", "PASS", value)
    if hw.wsl2:
        return Check("D02", "Platform", "WARN", value, "Tier 2: single GPU, WSL caps host RAM")
    if hw.system == "Windows":
        return Check("D02", "Platform", "WARN", value, "Tier 2: single GPU only (no NCCL)")
    return Check("D02", "Platform", "WARN", value, "Tier 2 platform")


def check_hosted(hw: HardwareInfo) -> Check:
    if hw.hosted == "colab":
        return Check("D03", "Hosted", "WARN", "Google Colab detected", "no guaranteed GPU quota")
    if hw.hosted == "kaggle":
        return Check(
            "D03",
            "Hosted",
            "WARN",
            "Kaggle detected",
            "weekly GPU quota and session time limits apply",
        )
    return Check("D03", "Hosted", "SKIP", "not a hosted notebook")


def check_torch(hw: HardwareInfo) -> Check:
    if hw.torch_error and hw.torch is None:
        return Check("D04", "torch", "FAIL", "installed but fails to import", hw.torch_error)
    if hw.torch is None:
        return Check(
            "D04",
            "torch",
            "FAIL",
            "not installed",
            "0.0.1 does not need it; training from v0.1 does",
        )
    build = []
    if hw.cuda:
        build.append(f"CUDA {hw.cuda}")
    if hw.hip:
        build.append(f"ROCm {hw.hip}")
    value = hw.torch + (f" ({', '.join(build)})" if build else " (CPU build)")
    st = standing("torch", hw.torch)
    cpu_build_next_to_gpu = (
        not build and hw.gpu_source == "nvidia-smi" and st.standing in {"tested", "newer"}
    )
    if cpu_build_next_to_gpu:
        # TT-H004: plain pip resolved a CPU wheel on a machine with an NVIDIA GPU.
        return Check(
            "D04",
            "torch",
            "WARN",
            value,
            "CPU-only build next to an NVIDIA GPU: "
            "uv pip install --reinstall torch --torch-backend=auto",
        )
    if st.standing == "tested":
        return Check("D04", "torch", "PASS", value, f"tested: {describe_range('torch')}")
    if st.standing == "newer":
        return Check("D04", "torch", "WARN", value, "newer than tested (2.14, 2.15)")
    return Check("D04", "torch", "FAIL", value, "2.14 or later is required")


def check_accelerator(hw: HardwareInfo, tier: Tier) -> Check:
    if not hw.gpus:
        return Check("D05", "Accelerator", "WARN", "none (CPU only)", tier.label)
    g = hw.gpus[0]
    value = g.name
    if g.capability is not None and g.vendor == "nvidia":
        value += f", compute capability {g.capability[0]}.{g.capability[1]}"
    if len(hw.gpus) > 1:
        value += f" (+{len(hw.gpus) - 1} more)"
    status: Status
    if tier.label == "Unsupported":
        status = "FAIL"
    elif tier.label in {"Tier 1", "Tier 1 at release gate"}:
        status = "PASS"
    else:
        status = "WARN"
    note = tier.label
    if hw.gpu_source == "nvidia-smi":
        note += "; found with nvidia-smi"
    return Check("D05", "Accelerator", status, value, note)


def _major(version: str | None) -> int | None:
    if not version:
        return None
    head = version.split(".")[0]
    return int(head) if head.isdigit() else None


def check_driver(hw: HardwareInfo) -> Check:
    nvidia = any(g.vendor == "nvidia" for g in hw.gpus)
    if not nvidia:
        return Check("D06", "Driver", "SKIP", "no NVIDIA GPU")
    major = _major(hw.driver)
    if major is None:
        return Check(
            "D06",
            "Driver",
            "WARN",
            "not readable",
            "nvidia-smi gave no driver version (common under WSL2)",
        )
    if major >= DRIVER_FLOOR:
        return Check("D06", "Driver", "PASS", hw.driver or "")
    cuda_major = _major(hw.cuda)
    if cuda_major is not None and cuda_major >= 13:
        return Check(
            "D06",
            "Driver",
            "FAIL",
            hw.driver or "",
            "CUDA 13 torch builds need driver 580 or later",
        )
    return Check("D06", "Driver", "WARN", hw.driver or "", "580 or later is recommended")


def check_vram(hw: HardwareInfo) -> Check:
    if not hw.gpus:
        return Check("D07", "VRAM", "SKIP", "no GPU")
    g = hw.gpus[0]
    if g.vendor == "apple":
        return Check("D07", "VRAM", "PASS", f"{g.total_gib:.1f} GiB unified memory")
    if g.free_gib is None:
        return Check("D07", "VRAM", "PASS", f"{g.total_gib:.1f} GiB total")
    value = f"{g.total_gib:.1f} GiB total, {g.free_gib:.1f} GiB free"
    if g.free_gib >= 0.9 * g.total_gib:
        return Check("D07", "VRAM", "PASS", value)
    return Check("D07", "VRAM", "WARN", value, "another process holds GPU memory")


def _bf16_matmul_probe(device: str) -> bool | None:
    """On non-CUDA backends: does a bf16 matmul run and agree with fp32?"""
    try:
        import torch

        gen = torch.Generator().manual_seed(0)
        a = torch.randn(64, 64, generator=gen)
        ref = a @ a
        got = (a.to(device, torch.bfloat16) @ a.to(device, torch.bfloat16)).float().cpu()
        return bool(((got - ref).abs().max() / ref.abs().max()) < 0.05)
    except Exception:
        return None


def check_bf16(hw: HardwareInfo) -> Check:
    if hw.torch is None or not hw.gpus or hw.gpu_source != "torch":
        return Check("D08", "bf16", "SKIP", "needs torch with a usable GPU")
    if hw.accelerator in {"cuda", "rocm"}:
        if hw.real_bf16:
            return Check("D08", "bf16", "PASS", "native")
        return Check("D08", "bf16", "WARN", "emulated only", "fp16 autocast + fp32 master")
    ok = _bf16_matmul_probe(hw.accelerator)
    if ok:
        return Check("D08", "bf16", "PASS", f"bf16 matmul works on {hw.accelerator}")
    return Check(
        "D08",
        "bf16",
        "WARN",
        f"bf16 matmul failed on {hw.accelerator}",
        "fp16 autocast + fp32 master",
    )


def check_host_ram(hw: HardwareInfo) -> Check:
    value = f"{hw.host_ram_gib:.1f} GiB total, {hw.host_ram_available_gib:.1f} GiB available"
    if not hw.gpus or hw.gpus[0].vendor == "apple":
        return Check("D09", "Host RAM", "PASS", value)
    need = HOST_TO_GPU_RATIO * 0.92 * hw.gpus[0].total_gib
    if hw.host_ram_available_gib >= need:
        return Check("D09", "Host RAM", "PASS", value)
    note = f"host-offload routes limited (about {need:.0f} GiB suits this GPU)"
    if hw.wsl2:
        note += "; raise memory= in %UserProfile%\\.wslconfig, then run wsl --shutdown"
    return Check("D09", "Host RAM", "WARN", value, note)


def _hf_cache_dir() -> Path:
    try:
        from huggingface_hub import constants

        return Path(constants.HF_HUB_CACHE)
    except Exception:
        return Path.home() / ".cache" / "huggingface" / "hub"


def check_disk() -> Check:
    path = _hf_cache_dir()
    probe_path = path
    while not probe_path.exists() and probe_path != probe_path.parent:
        probe_path = probe_path.parent
    try:
        free = shutil.disk_usage(probe_path).free / 2**30
    except OSError as exc:
        return Check("D10", "Disk", "WARN", "not readable", str(exc))
    value = f"{free:.0f} GiB free for the Hugging Face cache"
    if free >= DISK_FLOOR_GIB:
        return Check("D10", "Disk", "PASS", value)
    return Check("D10", "Disk", "WARN", value, f"{DISK_FLOOR_GIB:.0f} GiB or more is advised")


def check_multi_gpu(hw: HardwareInfo) -> Check:
    n = len(hw.gpus)
    if n <= 1:
        return Check("D11", "Multi-GPU", "SKIP", f"{n} GPU")
    if hw.system == "Windows":
        return Check("D11", "Multi-GPU", "WARN", f"{n} GPUs", "Windows: single-GPU only")
    nccl: bool | None = None
    if hw.torch is not None:
        try:
            import torch.distributed as dist

            nccl = bool(dist.is_available() and dist.is_nccl_available())
        except Exception:
            nccl = None
    if nccl is False and hw.accelerator == "cuda":
        return Check("D11", "Multi-GPU", "WARN", f"{n} GPUs", "NCCL is not available")
    return Check("D11", "Multi-GPU", "PASS", f"{n} GPUs", "DDP from v0.1, FSDP2 from v0.2")


def check_core_packages() -> Check:
    parts: list[str] = []
    notes: list[str] = []
    worst: Status = "PASS"
    for pkg in ("transformers", "accelerate", "peft", "trl"):
        st = standing(pkg)
        status: Status
        if st.standing == "missing":
            parts.append(f"{pkg} missing")
            status = "WARN"
        else:
            parts.append(f"{pkg} {st.version}")
            if st.standing == "tested":
                status = "PASS"
            elif st.standing == "unsupported":
                status = "FAIL"
                notes.append(f"{pkg} {st.version} is not supported ({describe_range(pkg)})")
            else:
                status = "WARN"
                notes.append(f"{pkg} is {st.standing} than tested ({describe_range(pkg)})")
        if _RANK[status] > _RANK[worst]:
            worst = status
    if worst != "FAIL" and any(p.endswith("missing") for p in parts):
        notes.append("not needed by 0.0.1; v0.1 installs them")
    return Check("D12", "Packages", worst, ", ".join(parts), "; ".join(notes))


def check_extras(hw: HardwareInfo) -> Check:
    found = {
        name: installed_version(dist)
        for name, dist in (
            ("bitsandbytes", "bitsandbytes"),
            ("flash-attn", "flash-attn"),
            ("triton", "triton"),
            ("deepspeed", "deepspeed"),
        )
    }
    present = [f"{k} {v}" for k, v in found.items() if v]
    value = ", ".join(present) if present else "none installed"
    if not hw.gpus or hw.gpus[0].vendor not in {"nvidia", "amd"}:
        return Check("D13", "Extras", "SKIP", value)
    notes: list[str] = []
    status: Status = "PASS"
    if not found["bitsandbytes"]:
        status = "WARN"
        notes.append(
            'bitsandbytes missing: pip install "tailtune[bnb]" for lora on a 4-bit base (from v0.1)'
        )
    cc = hw.gpus[0].capability
    if found["flash-attn"] and cc is not None and cc < (8, 0):
        status = "WARN"
        notes.append("flash-attn does not support Turing; SDPA is used")
    return Check("D13", "Extras", status, value, "; ".join(notes))


def check_attention(hw: HardwareInfo) -> Check:
    if hw.torch is None or hw.accelerator not in {"cuda", "rocm"} or hw.gpu_source != "torch":
        return Check("D14", "Attention", "SKIP", "needs torch with a CUDA or ROCm GPU")
    try:
        import torch

        flash = bool(torch.backends.cuda.flash_sdp_enabled())
        mem = bool(torch.backends.cuda.mem_efficient_sdp_enabled())
    except Exception as exc:
        return Check("D14", "Attention", "WARN", "SDPA flags not readable", str(exc))
    value = f"SDPA flash {'on' if flash else 'off'}, memory-efficient {'on' if mem else 'off'}"
    if flash:
        return Check("D14", "Attention", "PASS", value)
    return Check("D14", "Attention", "WARN", value, "FlashAttention SDPA backend is off")


def check_mps(hw: HardwareInfo) -> Check:
    if hw.system != "Darwin":
        return Check("D15", "MPS", "SKIP", "not macOS")
    mac = _mac_major()
    value = f"macOS {platform.mac_ver()[0] or 'unknown'}, torch {hw.torch or 'missing'}"
    if mac is not None and mac >= 15 and _torch_at_least(hw, "2.15"):
        return Check("D15", "MPS", "PASS", value)
    return Check("D15", "MPS", "FAIL", value, "MPS needs macOS 15+ and torch 2.15+")


def check_xpu(hw: HardwareInfo) -> Check:
    if hw.accelerator != "xpu":
        return Check("D16", "XPU", "SKIP", "no Intel XPU")
    try:
        import torch

        torch.zeros(1, dtype=torch.float64, device="xpu")
        return Check("D16", "XPU", "PASS", "FP64 available")
    except Exception:
        return Check("D16", "XPU", "WARN", "no FP64 (Arc A-series?)", "no loss scaler")


# Qwen3-8B's 252 target matrices, for scaling a measured SVD time (07 section 8).
_QWEN3_8B_LAYER = (
    (4096, 4096),
    (1024, 4096),
    (1024, 4096),
    (4096, 4096),
    (12288, 4096),
    (12288, 4096),
    (4096, 12288),
)
_QWEN3_8B_SVD_WORK = 36 * sum(m * n * min(m, n) for m, n in _QWEN3_8B_LAYER)


def _svd_numbers() -> tuple[float, float] | None:
    """(seconds, orthonormality error) for a 1024x1024 fp32 SVD, or None without a backend."""
    n = 1024
    try:
        import torch

        gen = torch.Generator().manual_seed(0)
        a = torch.randn(n, n, generator=gen, dtype=torch.float32)
        start = time.perf_counter()
        u, _, _ = torch.linalg.svd(a, full_matrices=False)
        elapsed = time.perf_counter() - start
        err = float((u.T @ u - torch.eye(n)).abs().max())
        return elapsed, err
    except ImportError:
        pass
    try:
        import numpy as np

        rng = np.random.default_rng(0)
        a = rng.standard_normal((n, n), dtype=np.float32)
        start = time.perf_counter()
        u, _, _ = np.linalg.svd(a, full_matrices=False)
        elapsed = time.perf_counter() - start
        err = float(np.abs(u.T @ u - np.eye(n, dtype=np.float32)).max())
        return elapsed, err
    except ImportError:
        return None


def check_cpu_svd() -> Check:
    try:
        result = _svd_numbers()
    except Exception as exc:
        return Check("D17", "CPU SVD", "WARN", "failed to run", f"{type(exc).__name__}: {exc}")
    if result is None:
        return Check("D17", "CPU SVD", "SKIP", "needs torch or numpy")
    elapsed, err = result
    hours = elapsed * _QWEN3_8B_SVD_WORK / 1024**3 / 3600
    value = f"1024x1024 in {elapsed:.2f} s, max orthonormality error {err:.1e}"
    note = f"an 8B model's bases would take about {hours:.1f} h here [C]"
    if err > 1e-3:
        return Check("D17", "CPU SVD", "FAIL", value, "singular vectors are not orthonormal")
    if hours > SVD_SLOW_HOURS:
        return Check("D17", "CPU SVD", "WARN", value, note)
    return Check("D17", "CPU SVD", "PASS", value, note)


# ---- --deep -------------------------------------------------------------------------------


def _to_bf16(x: float) -> float:
    """Round an fp32 value to bf16 (round to nearest even), without torch."""
    bits = struct.unpack("<I", struct.pack("<f", x))[0]
    bits += 0x7FFF + ((bits >> 16) & 1)
    return struct.unpack("<f", struct.pack("<I", (bits >> 16) << 16 & 0xFFFFFFFF))[0]


def check_bf16_rounding() -> Check:
    """The precision trap behind D-12: a small update to a bf16 weight is rounded away."""
    w, step = 1.0, 1e-3
    after = _to_bf16(w + step)
    if after == w:
        return Check(
            "D19",
            "bf16 rounding",
            "PASS",
            f"bf16(1.0 + {step:g}) == 1.0",
            "why tailtune keeps fp32 master weights for constrained methods",
        )
    return Check(
        "D19",
        "bf16 rounding",
        "WARN",
        f"bf16(1.0 + {step:g}) == {after}",
        "unexpected rounding result",
    )


_GLOO_SCRIPT = """
import sys, torch, torch.distributed as dist
rank, path = int(sys.argv[1]), sys.argv[2]
dist.init_process_group("gloo", init_method="file://" + path, rank=rank, world_size=2)
t = torch.ones(1) * (rank + 1)
dist.all_reduce(t)
assert float(t) == 3.0
dist.destroy_process_group()
"""


def check_gloo(hw: HardwareInfo) -> Check:
    if hw.torch is None:
        return Check("D20", "gloo 2-rank", "SKIP", "needs torch")
    with tempfile.TemporaryDirectory() as tmp:
        store = Path(tmp) / "store"
        script = Path(tmp) / "gloo_check.py"
        script.write_text(_GLOO_SCRIPT, encoding="utf-8")
        env = {**os.environ, "PYTHONWARNINGS": "ignore"}
        start = time.perf_counter()
        procs = [
            # Fixed interpreter and a script we just wrote; no shell, no user input.
            subprocess.Popen(  # noqa: S603
                [sys.executable, "-I", str(script), str(r), store.as_posix()],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                env=env,
            )
            for r in (0, 1)
        ]
        errors: list[str] = []
        for p in procs:
            try:
                _, err = p.communicate(timeout=90)
            except subprocess.TimeoutExpired:
                p.kill()
                errors.append("timed out")
                continue
            if p.returncode != 0:
                lines = (err or b"").decode(errors="replace").strip().splitlines()
                errors.append(lines[-1] if lines else f"exit {p.returncode}")
        elapsed = time.perf_counter() - start
    if errors:
        return Check("D20", "gloo 2-rank", "FAIL", "all_reduce failed", "; ".join(errors))
    return Check("D20", "gloo 2-rank", "PASS", f"all_reduce across 2 ranks in {elapsed:.1f} s")


# ---- report -------------------------------------------------------------------------------


def run_doctor(*, deep: bool = False, hw: HardwareInfo | None = None) -> DoctorReport:
    from tailtune import __version__

    start = time.perf_counter()
    info = hw or probe()
    tier = classify(info)
    checks = [
        check_python(),
        check_os(info),
        check_hosted(info),
        check_torch(info),
        check_accelerator(info, tier),
        check_driver(info),
        check_vram(info),
        check_bf16(info),
        check_host_ram(info),
        check_disk(),
        check_multi_gpu(info),
        check_core_packages(),
        check_extras(info),
        check_attention(info),
        check_mps(info),
        check_xpu(info),
        check_cpu_svd(),
    ]
    if deep:
        checks += [check_bf16_rounding(), check_gloo(info)]
    verdict_status: Status = "FAIL" if tier.label == "Unsupported" else "PASS"
    checks.append(Check("D18", "Verdict", verdict_status, tier.label, tier.reason))
    return DoctorReport(
        version=__version__,
        checks=checks,
        tier=tier,
        hardware=info,
        elapsed_s=time.perf_counter() - start,
        notes=list(info.notes),
    )
