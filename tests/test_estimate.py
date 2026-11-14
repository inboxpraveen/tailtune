"""Golden tests: the formula must reproduce docs/plan/07-hardware.md sections 9.2 to 9.4.

Values here are the plan's own budget figures for Qwen3 1.7B to 235B, computed by the
research budget scripts, which are not in this repository.
"""

from __future__ import annotations

import pytest

from conftest import load_config
from tailtune._core.errors import EXIT_REFUSED, EXIT_USAGE, ConfigError, DoesNotFitError
from tailtune.hardware.estimate import (
    AUTO_ORDER,
    GIB,
    ROUTE_LABEL,
    ROUTES,
    Calib,
    Estimate,
    Machine,
    activation_bytes,
    bases_bytes,
    estimate,
    estimate_shapes,
    refusal_for,
    resolve_machine,
    route_gib,
)
from tailtune.hardware.shapes import shapes_from_config

# (lora-4bit, lora-16bit, full, osf, sst/gpu, sst/host-bases, host, sst/8bit-adam,
#  sst/offload-state, host, sst/zero-offload, host), GiB
GOLDEN = {
    "Qwen_Qwen3-1.7B": (
        3.455,
        5.62,
        30.01,
        17.488,
        40.585,
        30.387,
        9.614,
        19.865,
        16.287,
        22.432,
        9.237,
        28.841,
    ),
    "Qwen_Qwen3-4B": (
        5.095,
        10.678,
        67.826,
        40.187,
        96.329,
        68.617,
        25.912,
        44.017,
        35.652,
        55.88,
        19.17,
        70.865,
    ),
    "Qwen_Qwen3-8B": (
        8.873,
        19.547,
        136.302,
        78.081,
        188.217,
        137.744,
        47.196,
        87.651,
        70.618,
        108.219,
        37.055,
        138.731,
    ),
    "Qwen_Qwen3-14B": (
        13.204,
        33.507,
        244.261,
        142.116,
        343.794,
        246.749,
        90.485,
        156.429,
        125.717,
        200.514,
        65.201,
        255.529,
    ),
    "Qwen_Qwen3-235B-A22B": (
        179.135,
        538.485,
        3856.02,
        2465.085,
        5766.348,
        3876.343,
        1736.661,
        2438.516,
        1949.607,
        3488.24,
        986.239,
        4364.029,
    ),
}


@pytest.mark.parametrize("name", sorted(GOLDEN))
def test_routes_match_the_research_budgets(name: str) -> None:
    s = shapes_from_config(load_config(name))
    g = GOLDEN[name]
    expect = {
        "lora-4bit": (g[0], 0.0),
        "lora-16bit": (g[1], 0.0),
        "full": (g[2], 0.0),
        "osf": (g[3], 0.0),
        "sst/gpu": (g[4], 0.0),
        "sst/host-bases": (g[5], g[6]),
        "sst/8bit-adam": (g[7], g[6]),
        "sst/offload-state": (g[8], g[9]),
        "sst/zero-offload": (g[10], g[11]),
    }
    for route, (gpu, host) in expect.items():
        got_gpu, got_host = route_gib(s, route)
        assert got_gpu == pytest.approx(gpu, abs=0.002), route
        assert got_host == pytest.approx(host, abs=0.002), route


def test_worked_example_terms() -> None:
    # 07 section 9.2, step by step.
    s = shapes_from_config(load_config("Qwen_Qwen3-8B"))
    assert 4 * s.p_total / GIB == pytest.approx(30.51, abs=0.01)
    assert bases_bytes(s.block, 0.99) / GIB == pytest.approx(0.59, abs=0.01)
    assert 4 * s.p_block / GIB == pytest.approx(0.72, abs=0.01)
    assert bases_bytes(s.targets, 0.99) / GIB == pytest.approx(21.3, abs=0.05)
    assert 4 * s.p_targets / GIB == pytest.approx(25.9, abs=0.05)
    assert activation_bytes(s, 1024) / GIB == pytest.approx(0.955, abs=0.001)
    assert bases_bytes(s.targets, 0.99, form="head") / GIB == pytest.approx(37.0, abs=0.05)


def test_tail_form_is_never_larger_than_head_form() -> None:
    s = shapes_from_config(load_config("Qwen_Qwen3-4B"))
    for r in (0.1, 0.5, 0.9, 0.99):
        assert bases_bytes(s.targets, r) <= bases_bytes(s.targets, r, form="head")


def test_activations_grow_with_tokens() -> None:
    s = shapes_from_config(load_config("Qwen_Qwen3-8B"))
    one = route_gib(s, "lora-16bit", tokens=1024)[0]
    two = route_gib(s, "lora-16bit", tokens=2048)[0]
    # 07 section 9.1: each extra 1,024 tokens adds about 1.05 GiB after fragmentation.
    assert two - one == pytest.approx(1.05, abs=0.01)


def _est(
    name: str,
    card: str,
    *,
    host_ram: float | None = None,
    method: str = "auto",
    route: str | None = None,
) -> Estimate:
    machine = resolve_machine(gpu=card, host_ram=host_ram)
    shapes = shapes_from_config(load_config(name))
    return estimate_shapes(shapes, machine=machine, method=method, route=route)


def test_b200_host_bases_is_borderline_and_auto_skips_it() -> None:
    est = _est("Qwen_Qwen3-8B", "B200", host_ram=512.0)
    row = est.route("sst/host-bases")
    assert row.gpu_gib < 0.92 * 180 < row.high_gib
    assert row.verdict == "borderline"
    assert est.route("sst/gpu").verdict == "does not fit"
    assert est.auto.route == "osf"
    assert "borderline" in est.auto.reason


def test_auto_picks_lora_4bit_on_a_t4_for_8b() -> None:
    est = _est("Qwen_Qwen3-8B", "T4")
    assert est.auto.route == "lora-4bit"
    assert "sst needs about 138 GiB" in est.auto.reason


def test_auto_picks_sst_when_it_fits_and_warns_on_small_models() -> None:
    est = _est("Qwen_Qwen3-0.6B", "RTX 4090")
    assert est.auto.route == "sst/gpu"
    assert "8.0 points at 1.7B" in est.auto.reason


def test_auto_never_picks_variants_offload_or_baselines() -> None:
    for card in ("T4", "RTX 4090", "RTX 5090", "L40S", "H100", "RTX PRO 6000", "H200", "B200"):
        for name in GOLDEN:
            est = _est(name, card, host_ram=4096.0)
            assert est.auto.route in (*AUTO_ORDER, None)
            assert est.auto.route not in ROUTE_LABEL


def test_auto_reports_when_nothing_fits() -> None:
    est = _est("Qwen_Qwen3-235B-A22B", "T4")
    assert est.auto.route is None
    assert est.auto.rule_id == "A7"


def test_no_gpu_means_no_verdict() -> None:
    s = shapes_from_config(load_config("Qwen_Qwen3-8B"))
    est = estimate_shapes(s, machine=Machine("cpu", None, 64.0, "doctor"))
    assert {r.verdict for r in est.routes} <= {"not checked", "does not fit"}
    assert est.auto.rule_id == "A1"
    assert refusal_for(est) is None


def test_host_ram_is_part_of_the_fit() -> None:
    est = _est("Qwen_Qwen3-8B", "B200", route="sst/host-bases", host_ram=32.0)
    row = est.route("sst/host-bases")
    assert row.verdict == "does not fit"
    err = refusal_for(est)
    assert isinstance(err, DoesNotFitError)
    assert err.exit_code == EXIT_REFUSED
    assert "host RAM" in err.message


def test_a_named_route_that_does_not_fit_is_refused() -> None:
    est = _est("Qwen_Qwen3-8B", "T4", route="sst/gpu")
    err = refusal_for(est)
    assert isinstance(err, DoesNotFitError)
    assert err.code == "TT-H001"


def test_a_borderline_named_route_runs() -> None:
    est = _est("Qwen_Qwen3-8B", "B200", route="sst/host-bases", host_ram=512.0)
    assert refusal_for(est) is None


def test_a_method_fits_when_any_of_its_routes_fits() -> None:
    est = _est("Qwen_Qwen3-8B", "T4", method="lora")
    assert [r.route for r in est.routes] == ["lora-4bit", "lora-16bit"]
    assert refusal_for(est) is None
    est = _est("Qwen_Qwen3-8B", "T4", method="sst")
    assert isinstance(refusal_for(est), DoesNotFitError)


def test_every_route_is_priced_in_auto_mode() -> None:
    est = _est("Qwen_Qwen3-4B", "H100")
    assert [r.route for r in est.routes] == list(ROUTES)


def test_band_is_asymmetric() -> None:
    c = Calib()
    assert c.band == (0.85, 1.25)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"r": 0.0},
        {"r": 1.0},
        {"r": 1.5},
        {"nproc": 2},
        {"seq_len": 0},
        {"micro_batch": 0},
        {"gpu": "B200", "vram": 80.0},
        {"gpu": "GTX 9999"},
        {"vram": -1.0},
        {"host_ram": 0.0},
        {"method": "freeze"},
        {"method": "magic"},
        {"route": "sst/turbo"},
        {"method": "lora", "route": "lora-4bit"},
    ],
)
def test_bad_inputs_are_usage_errors(kwargs: dict[str, object], no_network: None) -> None:
    with pytest.raises(ConfigError) as info:
        estimate("unused", **kwargs)  # type: ignore[arg-type]
    assert info.value.exit_code == EXIT_USAGE
    assert info.value.code == "TT-C003"


def test_estimate_from_a_local_config_needs_no_network(no_network: None, no_gpu: None) -> None:
    from conftest import config_path

    est = estimate(str(config_path("Qwen_Qwen3-8B")), gpu="T4", host_ram=12.7)
    assert est.route("sst/host-bases").gpu_gib == pytest.approx(137.744, abs=0.002)
    assert est.to_dict()["machine"]["name"] == "Tesla T4"


def test_estimate_on_this_machine_without_a_gpu(no_network: None, no_gpu: None) -> None:
    from conftest import config_path

    est = estimate(str(config_path("Qwen_Qwen3-0.6B")))
    assert est.machine.vram_gib is None
    assert est.auto.rule_id == "A1"
