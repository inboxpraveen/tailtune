from __future__ import annotations

import json
from dataclasses import replace

import pytest

from tailtune._core.versions import standing
from tailtune.hardware import doctor as doc
from tailtune.hardware.probe import GpuInfo, HardwareInfo, probe


def _hw(**overrides: object) -> HardwareInfo:
    base = HardwareInfo(
        system="Linux",
        machine="x86_64",
        python="3.13.0",
        wsl2=False,
        hosted=None,
        accelerator="cuda",
        gpus=(GpuInfo(0, "NVIDIA A100-SXM4-80GB", 80.0, 79.0, (8, 0), "nvidia"),),
        host_ram_gib=256.0,
        host_ram_available_gib=240.0,
        torch="2.15.0+cu132",
        torch_error=None,
        cuda="13.2",
        hip=None,
        driver="580.65",
        real_bf16=True,
        gpu_source="torch",
    )
    return replace(base, **overrides)  # type: ignore[arg-type]


def _gpu(name: str, cc: tuple[int, int] | None, gib: float = 24.0) -> tuple[GpuInfo, ...]:
    return (GpuInfo(0, name, gib, gib, cc, "nvidia"),)


@pytest.mark.parametrize(
    ("overrides", "tier"),
    [
        ({}, "Tier 1"),
        ({"gpus": _gpu("Tesla T4", (7, 5), 15.0), "real_bf16": False}, "Tier 1 reduced"),
        ({"gpus": _gpu("NVIDIA B200", (10, 0), 180.0)}, "Tier 1 at release gate"),
        ({"gpus": _gpu("RTX PRO 6000", (12, 0), 96.0)}, "Tier 1 at release gate"),
        ({"gpus": _gpu("Tesla P100", (6, 0), 16.0)}, "Unsupported"),
        ({"gpus": _gpu("Tesla V100", (7, 0), 16.0)}, "Unsupported"),
        ({"system": "Windows", "gpus": _gpu("RTX 4090", (8, 9))}, "Tier 2"),
        ({"wsl2": True, "gpus": _gpu("RTX 4090", (8, 9))}, "Tier 2"),
        ({"machine": "aarch64", "gpus": _gpu("NVIDIA GB200", (10, 0), 186.0)}, "Tier 2"),
        ({"gpus": _gpu("Vera Rubin", (10, 7), 288.0)}, "Tier 3"),
        ({"accelerator": "cpu", "gpus": ()}, "Tier 1 numerics"),
        ({"system": "Darwin", "accelerator": "cpu", "gpus": ()}, "Unsupported"),
        ({"accelerator": "xpu", "gpus": (GpuInfo(0, "Arc", 16, None, None, "intel"),)}, "Tier 3"),
        (
            {
                "accelerator": "rocm",
                "gpus": (GpuInfo(0, "AMD Instinct MI300X", 192, 190, (9, 4), "amd"),),
            },
            "Tier 2",
        ),
        (
            {
                "accelerator": "rocm",
                "gpus": (GpuInfo(0, "Radeon RX 9070 XT", 16, 16, (12, 0), "amd"),),
            },
            "Tier 3",
        ),
        ({"gpus": _gpu("Mystery GPU", None)}, "Unknown"),
    ],
)
def test_tiers_follow_the_support_matrix(overrides: dict[str, object], tier: str) -> None:
    hw = _hw(**overrides)
    if overrides.get("system") == "Darwin":
        hw = replace(hw, machine="x86_64")
    assert doc.classify(hw).label == tier


def test_t4_verdict_promises_the_t4_limits() -> None:
    hw = _hw(gpus=_gpu("Tesla T4", (7, 5), 15.0), real_bf16=False)
    report = doc.run_doctor(hw=hw)
    assert report.tier.label == "Tier 1 reduced"
    assert "sst up to ~0.7B" in report.tier.promise
    bf16 = next(c for c in report.checks if c.id == "D08")
    assert bf16.status == "WARN"
    assert "fp16 autocast" in bf16.note


def test_all_checks_are_reported_in_order() -> None:
    report = doc.run_doctor(hw=_hw())
    ids = [c.id for c in report.checks]
    assert ids == [f"D{i:02d}" for i in range(1, 19) if i != 18] + ["D18"]


def test_deep_adds_its_checks() -> None:
    report = doc.run_doctor(hw=_hw(torch=None), deep=True)
    ids = [c.id for c in report.checks]
    assert "D19" in ids
    assert "D20" in ids
    gloo = next(c for c in report.checks if c.id == "D20")
    assert gloo.status == "SKIP"


def test_doctor_on_the_real_machine_makes_no_network_calls(no_network: None) -> None:
    report = doc.run_doctor()
    data = report.to_dict()
    json.dumps(data)  # serialisable
    assert data["checks"][0]["id"] == "D01"
    assert report.elapsed_s < 30


def test_missing_torch_is_a_result_not_a_crash(no_gpu: None) -> None:
    hw = probe()
    assert hw.torch is None
    report = doc.run_doctor(hw=hw)
    d04 = next(c for c in report.checks if c.id == "D04")
    assert d04.status == "FAIL"
    assert "not installed" in d04.value
    assert report.tier.label in {"Tier 1 numerics", "Unsupported"}


def test_old_driver_with_cuda_13_fails() -> None:
    check = doc.check_driver(_hw(driver="575.10", cuda="13.0"))
    assert check.status == "FAIL"
    assert doc.check_driver(_hw(driver="575.10", cuda="12.6")).status == "WARN"
    assert doc.check_driver(_hw(driver=None)).status == "WARN"


def test_host_ram_warning_mentions_wslconfig_on_wsl2() -> None:
    hw = _hw(wsl2=True, host_ram_gib=16.0, host_ram_available_gib=12.0)
    check = doc.check_host_ram(hw)
    assert check.status == "WARN"
    assert ".wslconfig" in check.note


def test_bf16_rounding_demonstration() -> None:
    assert doc._to_bf16(1.0 + 1e-3) == 1.0
    assert doc._to_bf16(1.0 + 2**-7) == 1.0 + 2**-7
    assert doc.check_bf16_rounding().status == "PASS"


@pytest.mark.parametrize(
    ("package", "version", "expected"),
    [
        ("torch", "2.15.0+cu132", "tested"),
        ("torch", "2.14.1+cu130", "tested"),
        ("torch", "2.16.0", "newer"),
        ("torch", "2.11.0", "unsupported"),
        ("transformers", "4.57.6", "unsupported"),
        ("transformers", "5.10.0", "older"),
        ("transformers", "5.22.2", "tested"),
        ("transformers", "5.23.0", "newer"),
        ("peft", "0.21.2", "tested"),
        ("trl", "1.17.1", "tested"),
    ],
)
def test_version_standing(package: str, version: str, expected: str) -> None:
    assert standing(package, version).standing == expected


def test_transformers_4_fails_the_package_check(monkeypatch: pytest.MonkeyPatch) -> None:
    from tailtune._core import versions

    fake = {"transformers": "4.57.6", "accelerate": "1.14.0", "peft": "0.21.2", "trl": "1.17.1"}
    monkeypatch.setattr(versions, "installed_version", lambda pkg: fake.get(pkg))
    assert doc.check_core_packages().status == "FAIL"
