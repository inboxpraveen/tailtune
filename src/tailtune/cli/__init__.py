"""Command-line entry point: exit-code mapping and Ctrl-C handling.

Click exits with 2 on a parse error, which would collide with tailtune's code 2
(data validation). Every parse error is remapped to 5 here.
"""

from __future__ import annotations

import contextlib
import sys
import traceback


def _configure_streams() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            with contextlib.suppress(OSError, ValueError):
                reconfigure(errors="replace")


def _click_like(exc: BaseException) -> bool:
    """A Click exception. Typer 0.28+ vendors Click, older Typer uses the click package,
    so the classes are matched by shape rather than imported."""
    return hasattr(exc, "format_message") and hasattr(exc, "exit_code")


def main(argv: list[str] | None = None) -> int:
    """Run the CLI and return its exit code."""
    from tailtune._core.errors import (
        EXIT_ERROR,
        EXIT_INTERRUPTED,
        EXIT_OK,
        EXIT_USAGE,
        InternalError,
        TailtuneError,
        UsageError,
    )
    from tailtune.cli.app import app

    _configure_streams()
    args = sys.argv[1:] if argv is None else argv
    verbose = "-v" in args or "--verbose" in args
    try:
        result = app(args=args, prog_name="tailtune", standalone_mode=False)
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        return EXIT_INTERRUPTED
    except TailtuneError as exc:
        print(exc.render(), file=sys.stderr)
        return exc.exit_code
    except Exception as exc:
        name = type(exc).__name__
        if name == "Abort":  # Click turns Ctrl-C at a prompt into Abort
            print("Interrupted.", file=sys.stderr)
            return EXIT_INTERRUPTED
        if name == "NoArgsIsHelpError" and _click_like(exc):
            print(exc.format_message())  # type: ignore[attr-defined]
            return EXIT_OK
        if _click_like(exc) and getattr(exc, "exit_code", None) == 2:
            ctx = getattr(exc, "ctx", None)
            path = getattr(ctx, "command_path", "tailtune")
            err = UsageError(
                f"Invalid usage: {exc.format_message()}",  # type: ignore[attr-defined]
                fix=f"Run `{path} --help` for the options.",
            )
            print(err.render(), file=sys.stderr)
            return EXIT_USAGE
        if _click_like(exc):
            print(f"error: {exc.format_message()}", file=sys.stderr)  # type: ignore[attr-defined]
            return EXIT_ERROR
        err = InternalError(
            f"Internal error: {name}: {exc}",
            fix="This is a bug. Rerun with --verbose and attach the output to an issue.",
        )
        print(err.render(), file=sys.stderr)
        if verbose:
            traceback.print_exc()
        return EXIT_ERROR
    if isinstance(result, int):
        return result
    return EXIT_OK


__all__ = ["main"]
