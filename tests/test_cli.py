"""Exit-code contract (README, docs/plan/04-api-and-ux.md section 4.1)."""

from __future__ import annotations

import json

import pytest

from conftest import config_path
from tailtune import __version__
from tailtune.cli import main

QWEN3_8B = str(config_path("Qwen_Qwen3-8B"))


@pytest.mark.parametrize(
    ("argv", "code"),
    [
        (["--help"], 0),
        (["-h"], 0),
        (["--version"], 0),
        ([], 0),
        (["estimate", "--help"], 0),
        (["nosuchcommand"], 5),
        (["estimate"], 5),
        (["estimate", QWEN3_8B, "--bogus"], 5),
        (["estimate", QWEN3_8B, "--seq-len", "abc"], 5),
        (["estimate", QWEN3_8B, "--r", "1.5"], 5),
        (["estimate", QWEN3_8B, "--nproc", "2"], 5),
        (["estimate", QWEN3_8B, "--gpu", "GTX 9999"], 5),
        (["estimate", QWEN3_8B, "--gpu", "T4"], 0),
        (["estimate", QWEN3_8B, "--gpu", "T4", "--route", "sst/gpu"], 4),
        (["estimate", QWEN3_8B, "--gpu", "T4", "--method", "sst"], 4),
        (["estimate", QWEN3_8B, "--gpu", "T4", "--method", "lora"], 0),
        (["estimate", QWEN3_8B, "--gpu", "B200", "--route", "sst/host-bases"], 0),
    ],
)
def test_exit_codes(argv: list[str], code: int, no_network: None, no_gpu: None) -> None:
    assert main(argv) == code


def test_version_prints_the_version(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--version"]) == 0
    assert __version__ in capsys.readouterr().out


def test_usage_errors_use_the_catalogue_shape(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["estimate"]) == 5
    err = capsys.readouterr().err
    assert err.startswith("error[TT-C003]")
    assert "--help" in err


def test_estimate_table_is_labelled_as_computed(
    capsys: pytest.CaptureFixture[str], no_network: None
) -> None:
    assert main(["estimate", QWEN3_8B, "--gpu", "T4", "--host-ram", "12.7"]) == 0
    out = capsys.readouterr().out
    assert "not yet checked against hardware" in out
    assert "137.7 (117.1-172.2)" in out
    assert "sst/8bit-adam *" in out
    assert "method=auto -> lora on a 4-bit base" in out
    out.encode("ascii")  # plain ASCII, safe on any Windows code page


def test_estimate_json(capsys: pytest.CaptureFixture[str], no_network: None) -> None:
    assert main(["estimate", QWEN3_8B, "--gpu", "B200", "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    rows = {r["route"]: r for r in data["routes"]}
    assert rows["sst/host-bases"]["verdict"] == "borderline"
    assert data["auto"]["route"] == "osf"
    assert data["machine"]["source"] == "--gpu"


def test_refusal_prints_the_table_then_the_error(
    capsys: pytest.CaptureFixture[str], no_network: None
) -> None:
    assert main(["estimate", QWEN3_8B, "--gpu", "T4", "--route", "sst/gpu"]) == 4
    captured = capsys.readouterr()
    assert "sst/gpu" in captured.out
    assert captured.err.startswith("error[TT-H001]")


def test_quantized_model_is_refused(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["estimate", str(config_path("openai_gpt-oss-120b")), "--gpu", "B200"]) == 4
    assert "TT-T003" in capsys.readouterr().err


def test_doctor_runs_and_exits_zero(capsys: pytest.CaptureFixture[str], no_network: None) -> None:
    assert main(["doctor"]) == 0
    out = capsys.readouterr().out
    assert "Verdict:" in out
    out.encode("ascii")


def test_doctor_json(capsys: pytest.CaptureFixture[str], no_network: None) -> None:
    assert main(["doctor", "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["tailtune"] == __version__
    assert {c["id"] for c in data["checks"]} >= {f"D{i:02d}" for i in range(1, 19)}


def test_doctor_strict_exits_4_on_a_failed_check(
    no_gpu: None, capsys: pytest.CaptureFixture[str]
) -> None:
    # Without torch, D04 fails, so --strict must exit 4 while plain doctor exits 0.
    assert main(["doctor"]) == 0
    assert main(["doctor", "--strict"]) == 4
    assert "TT-H006" in capsys.readouterr().err


def test_internal_errors_exit_1(monkeypatch: pytest.MonkeyPatch) -> None:
    from tailtune.hardware import doctor

    def boom(**kwargs: object) -> None:
        raise RuntimeError("boom")

    monkeypatch.setattr(doctor, "run_doctor", boom)
    assert main(["doctor"]) == 1


def test_ctrl_c_exits_130(monkeypatch: pytest.MonkeyPatch) -> None:
    from tailtune.hardware import doctor

    def interrupt(**kwargs: object) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(doctor, "run_doctor", interrupt)
    assert main(["doctor"]) == 130


def test_quiet_estimate_prints_only_the_auto_line(
    capsys: pytest.CaptureFixture[str], no_network: None
) -> None:
    assert main(["-q", "estimate", QWEN3_8B, "--gpu", "T4"]) == 0
    out = capsys.readouterr().out.strip()
    assert out.startswith("method=auto -> lora on a 4-bit base")
    assert len(out.splitlines()) == 1


def test_quiet_doctor_keeps_the_verdict_and_failures(
    capsys: pytest.CaptureFixture[str], no_gpu: None
) -> None:
    assert main(["--quiet", "doctor"]) == 0
    out = capsys.readouterr().out
    assert "Verdict:" in out
    assert "torch" in out  # D04 fails without torch, so it stays
    assert "PASS" not in out


def test_auto_flags_unchecked_host_ram(
    capsys: pytest.CaptureFixture[str], no_network: None
) -> None:
    # Qwen3-1.7B on an L40S: auto picks sst/host-bases, which needs about 9.6 GiB host RAM.
    path = str(config_path("Qwen_Qwen3-1.7B"))
    assert main(["-q", "estimate", path, "--gpu", "L40S"]) == 0
    out = capsys.readouterr().out
    assert "bases streamed from host RAM" in out
    assert "not checked: pass --host-ram" in out
    assert main(["-q", "estimate", path, "--gpu", "L40S", "--host-ram", "64"]) == 0
    assert "not checked" not in capsys.readouterr().out
