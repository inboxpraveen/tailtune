"""Typer app: parse arguments, call one API function, render the result.

Never imports torch at module scope, so `tailtune --help` stays fast.
"""

from __future__ import annotations

import json
from typing import Annotated

import typer

from tailtune import __version__

app = typer.Typer(
    name="tailtune",
    help="Fine-tune LLMs with far less forgetting, and prove it. "
    "0.0.1 ships `doctor` and `estimate`.",
    add_completion=False,
    no_args_is_help=True,
    pretty_exceptions_enable=False,
    rich_markup_mode=None,
    context_settings={"help_option_names": ["-h", "--help"]},
)


def _version(value: bool) -> None:
    if value:
        typer.echo(f"tailtune {__version__}")
        raise typer.Exit(0)


@app.callback()
def _root(
    ctx: typer.Context,
    version: Annotated[
        bool,
        typer.Option("--version", callback=_version, is_eager=True, help="Print the version."),
    ] = False,
    quiet: Annotated[
        bool, typer.Option("-q", "--quiet", help="Errors and the final summary only.")
    ] = False,
    verbose: Annotated[
        bool, typer.Option("-v", "--verbose", help="Tracebacks for internal errors.")
    ] = False,
    no_color: Annotated[
        bool, typer.Option("--no-color", help="No colour. Output is plain text already.")
    ] = False,
) -> None:
    """Fine-tune LLMs with far less forgetting, and prove it."""
    ctx.obj = {"quiet": quiet}


def _quiet(ctx: typer.Context) -> bool:
    obj = ctx.find_root().obj
    return bool(obj.get("quiet")) if isinstance(obj, dict) else False


@app.command()
def doctor(
    ctx: typer.Context,
    deep: Annotated[
        bool,
        typer.Option("--deep", help="Also run a 2-rank gloo test and a bf16 rounding check."),
    ] = False,
    strict: Annotated[
        bool, typer.Option("--strict", help="Exit 4 if any check failed. For CI and scripts.")
    ] = False,
    json_out: Annotated[bool, typer.Option("--json", help="Machine-readable output.")] = False,
) -> None:
    """Report whether this machine can run tailtune, and at what tier. No network calls."""
    from tailtune._core.errors import DoctorStrictError
    from tailtune.cli.render import doctor_text
    from tailtune.hardware.doctor import run_doctor

    report = run_doctor(deep=deep)
    if json_out:
        typer.echo(json.dumps(report.to_dict(), indent=2))
    else:
        typer.echo(doctor_text(report, quiet=_quiet(ctx)))
    if strict and report.failed:
        ids = ", ".join(c.id for c in report.failed)
        raise DoctorStrictError(
            f"doctor --strict: {len(report.failed)} check(s) failed ({ids}).",
            fix="Fix the FAIL rows above, or run without --strict to report only.",
        )


@app.command()
def estimate(
    ctx: typer.Context,
    model: Annotated[str, typer.Argument(metavar="MODEL", help="A Hub id or a local path.")],
    method: Annotated[str, typer.Option("--method", help="auto, lora, osf, sst or full.")] = "auto",
    route: Annotated[
        str | None, typer.Option("--route", help="One route, for example sst/host-bases.")
    ] = None,
    r: Annotated[float, typer.Option("--r", help="Preserved fraction for sst routes.")] = 0.99,
    seq_len: Annotated[int, typer.Option("--seq-len", help="Tokens per sequence.")] = 1024,
    micro_batch: Annotated[int, typer.Option("--micro-batch", help="Sequences per step.")] = 1,
    nproc: Annotated[int, typer.Option("--nproc", help="GPUs. 0.0.1 supports 1.")] = 1,
    gpu: Annotated[
        str | None, typer.Option("--gpu", help='Price a named card, for example "B200".')
    ] = None,
    vram: Annotated[
        float | None, typer.Option("--vram", help="Price a GPU of this many GiB.")
    ] = None,
    host_ram: Annotated[
        float | None, typer.Option("--host-ram", help="Available host RAM in GiB.")
    ] = None,
    revision: Annotated[
        str | None, typer.Option("--revision", help="Hub branch, tag or commit.")
    ] = None,
    json_out: Annotated[bool, typer.Option("--json", help="Machine-readable output.")] = False,
) -> None:
    """Print the memory budget per route before any weights load. Reads config.json only."""
    from tailtune.cli.render import estimate_text
    from tailtune.hardware.estimate import estimate as run_estimate
    from tailtune.hardware.estimate import refusal_for

    est = run_estimate(
        model,
        method=method,
        route=route,
        r=r,
        seq_len=seq_len,
        micro_batch=micro_batch,
        nproc=nproc,
        gpu=gpu,
        vram=vram,
        host_ram=host_ram,
        revision=revision,
    )
    if json_out:
        typer.echo(json.dumps(est.to_dict(), indent=2))
    elif _quiet(ctx):
        typer.echo(est.auto.reason)
    else:
        typer.echo(estimate_text(est))
    refusal = refusal_for(est)
    if refusal is not None:
        raise refusal
