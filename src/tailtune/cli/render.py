"""Plain-text rendering for the terminal. ASCII only, so redirected output never fails
to encode on Windows code pages."""

from __future__ import annotations

from tailtune.hardware.doctor import DoctorReport
from tailtune.hardware.estimate import Estimate, RouteEstimate


def _b(n: int) -> str:
    return f"{n / 1e9:.2f}B"


def _gib(x: float) -> str:
    return f"{x:g}" if float(x).is_integer() else f"{x:.1f}"


def doctor_text(report: DoctorReport, *, quiet: bool = False) -> str:
    lines = [f"tailtune {report.version} doctor", ""]
    for c in report.checks:
        if c.id == "D18" or (quiet and c.status != "FAIL"):
            continue
        note = f"  {c.note}" if c.note else ""
        lines.append(f"  {c.name:<13} {c.value:<58} {c.status:<4}{note}")
    lines.append("")
    lines.append(f"Verdict: {report.tier.label} ({report.tier.reason}).")
    lines.append(f"  Promised from v0.1: {report.tier.promise}.")
    for note in report.notes:
        lines.append(f"  Note: {note}")
    lines.append("  Next: tailtune estimate <model>")
    return "\n".join(lines)


def _here(row: RouteEstimate) -> str:
    return "-" if row.verdict == "not checked" else row.verdict


def _card(row: RouteEstimate) -> str:
    if row.smallest_card is None:
        return "none up to 288 GB"
    text = row.smallest_card
    if row.host_gib > 0:
        text += f" + {row.host_gib:.0f} GiB RAM"
    if row.smallest_card_borderline:
        text += " (borderline)"
    return text


def estimate_text(est: Estimate) -> str:
    c = est.calib
    m = est.machine
    lines = [
        f"Model     {est.model}, read from {est.source}",
        f"          {_b(est.p_total)} parameters; {est.n_targets} target matrices hold "
        f"{_b(est.p_targets)}; embeddings and head {_b(est.p_embed)}"
        + (" (tied)" if est.tied else ""),
    ]
    limit = m.limit_gib(c)
    if limit is None or m.vram_gib is None:
        machine = f"{m.name}. Pass --gpu NAME or --vram GIB to check fit"
    else:
        machine = f"{m.name}: limit {limit:.1f} GiB ({c.usable:.0%} of {_gib(m.vram_gib)} GiB)"
    if m.host_ram_gib is not None:
        avail = " available" if m.source == "doctor" else ""
        machine += f"; host RAM {m.host_ram_gib:.1f} GiB{avail}"
    elif m.vram_gib is not None:
        machine += "; host RAM not checked (pass --host-ram GIB)"
    source = {"doctor": "[from doctor]", "--gpu": "[--gpu]", "--vram": "[--vram]"}.get(m.source, "")
    lines.append(f"Machine   {machine}   {source}".rstrip())
    lines.append(
        f"Settings  {est.tokens:,} tokens per micro-batch ({est.seq_len:,} x {est.micro_batch}), "
        f"checkpointing on, {est.precision}, sst r={est.r:g}, osf rho={c.osf_rho:g}"
    )
    lines.append("")
    header = (
        f"{'Route':<20} {'Method':<7} {'GPU GiB (range)':>21}  {'Host GiB':>8}  "
        f"{'Here':<13} Smallest single card"
    )
    lines.append(header)
    marks = {"variant": " *", "untested": " **", "baseline": ""}
    for row in est.routes:
        name = row.route + (" (baseline)" if row.label == "baseline" else "")
        name += marks.get(row.label or "", "")
        rng = f"{row.gpu_gib:.1f} ({row.low_gib:.1f}-{row.high_gib:.1f})"
        host = f"{row.host_gib:.1f}" if row.host_gib > 0 else "-"
        lines.append(
            f"{name:<20} {row.method:<7} {rng:>21}  {host:>8}  {_here(row):<13} {_card(row)}"
        )
    lines.append("")
    labels = {row.label for row in est.routes}
    if "variant" in labels:
        lines.append("*  Optimizer variant: published SST results do not transfer to it.")
    if "untested" in labels:
        lines.append("** Untested route; excludes a 2 to 4 B/param host gradient buffer.")
    lines.append(f"All figures [C]: {est.label}.")
    lines.append(
        f"Ranges are -{1 - c.band[0]:.0%} / +{c.band[1] - 1:.0%} until calibrated. Excluded: "
        "load-time peaks, save-time gathers, the SVD transient and communication buffers."
    )
    lines.append("")
    lines.append(est.auto.reason)
    return "\n".join(lines)
