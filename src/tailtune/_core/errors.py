"""Exception hierarchy, stable error codes and the exit-code contract.

Exit codes (README contract, docs/plan/04-api-and-ux.md section 4.1):

    0    success
    1    internal or unexpected error, Hub unreachable, declined confirmation
    2    blocking data validation failure
    3    regression gate failed
    4    refused by a pre-flight check, including model access and policy refusals
    5    invalid configuration or usage
    130  interrupted with Ctrl-C

Only the classes that 0.0.1 raises are defined here. The full tree is in
docs/plan/02-architecture.md section 10.
"""

from __future__ import annotations

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_DATA = 2
EXIT_REGRESSION = 3
EXIT_REFUSED = 4
EXIT_USAGE = 5
EXIT_INTERRUPTED = 130

DOCS_ERRORS = (
    "https://github.com/inboxpraveen/tailtune/blob/main/docs/plan/04-api-and-ux.md"
    "#8-error-message-catalogue"
)


class TailtuneDeprecationWarning(FutureWarning):
    """Raised for deprecated tailtune behaviour. Subclasses FutureWarning so users see it."""


class TailtuneError(Exception):
    """Base class. Every error carries a stable code, a message, a cause and a fix."""

    exit_code: int = EXIT_ERROR
    default_code: str = "TT-X000"

    def __init__(
        self,
        message: str,
        *,
        code: str | None = None,
        cause: str | None = None,
        fix: str | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.code = code or self.default_code
        self.cause = cause
        self.fix = fix

    def render(self) -> str:
        """Format the error in the catalogue's shape, for stderr."""
        lines = [f"error[{self.code}] {self.message}"]
        if self.cause:
            lines.append(f"  cause  {self.cause}")
        if self.fix:
            lines.append(f"  fix    {self.fix}")
        lines.append(f"  docs   {DOCS_ERRORS}")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, object]:
        return {
            "code": self.code,
            "exit_code": self.exit_code,
            "message": self.message,
            "cause": self.cause,
            "fix": self.fix,
        }


class ConfigError(TailtuneError):
    """A bad value, an unknown key or an out-of-range option (TT-C001, TT-C003)."""

    exit_code = EXIT_USAGE
    default_code = "TT-C003"


class UsageError(ConfigError):
    """A command line that could not be parsed. Click's exit code 2 is remapped to 5."""

    default_code = "TT-C003"


class ModelAccessError(TailtuneError):
    """A model tailtune cannot access, or will not load under its policy (D-08)."""

    exit_code = EXIT_REFUSED


class GatedModelError(ModelAccessError):
    default_code = "TT-M001"


class MissingTokenError(ModelAccessError):
    default_code = "TT-M002"


class RemoteCodeRequiredError(ModelAccessError):
    default_code = "TT-M003"


class UnsafeWeightsError(ModelAccessError):
    default_code = "TT-M004"


class HubUnavailableError(TailtuneError):
    """The Hugging Face Hub could not be reached and nothing usable was cached."""

    exit_code = EXIT_ERROR
    default_code = "TT-M005"


class RefusedError(TailtuneError):
    """A pre-flight refusal. Nothing was loaded or changed."""

    exit_code = EXIT_REFUSED


class DoesNotFitError(RefusedError):
    default_code = "TT-H001"


class TargetTypeError(RefusedError):
    """Conv1D or quantized trainable targets (TT-T002, TT-T003)."""

    default_code = "TT-T003"


class UnsupportedModelError(RefusedError):
    """An architecture whose shapes this version cannot read reliably."""

    default_code = "TT-T004"


class DoctorStrictError(RefusedError):
    """`doctor --strict` found at least one failed check."""

    default_code = "TT-H006"


class InternalError(TailtuneError):
    """An invariant broke. Always a bug."""

    default_code = "TT-X001"


__all__ = [
    "EXIT_DATA",
    "EXIT_ERROR",
    "EXIT_INTERRUPTED",
    "EXIT_OK",
    "EXIT_REFUSED",
    "EXIT_REGRESSION",
    "EXIT_USAGE",
    "ConfigError",
    "DoctorStrictError",
    "DoesNotFitError",
    "GatedModelError",
    "HubUnavailableError",
    "InternalError",
    "MissingTokenError",
    "ModelAccessError",
    "RefusedError",
    "RemoteCodeRequiredError",
    "TailtuneDeprecationWarning",
    "TailtuneError",
    "TargetTypeError",
    "UnsafeWeightsError",
    "UnsupportedModelError",
    "UsageError",
]
