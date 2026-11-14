"""tailtune: fine-tune LLMs with far less forgetting, and prove it.

0.0.1 ships two tools that need no GPU and no torch:

- ``tailtune.doctor()`` reports what this machine can run, and at which support tier.
- ``tailtune.estimate(model)`` prices every training route in GPU and host memory,
  from the model's ``config.json`` alone.

Every memory figure is computed from shapes and has not yet been checked against
hardware. Gate 1 of the plan calibrates it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from tailtune._core.errors import TailtuneDeprecationWarning, TailtuneError

__version__ = "0.0.1"

if TYPE_CHECKING:
    from tailtune.hardware.doctor import DoctorReport
    from tailtune.hardware.estimate import Estimate


def doctor(*, deep: bool = False) -> DoctorReport:
    """Run the machine checks (D01 to D18). Makes no network calls."""
    from tailtune.hardware.doctor import run_doctor

    return run_doctor(deep=deep)


def estimate(model: str, **kwargs: Any) -> Estimate:
    """Estimate GPU and host memory per route. See ``tailtune.hardware.estimate.estimate``."""
    from tailtune.hardware.estimate import estimate as _estimate

    return _estimate(model, **kwargs)


__all__ = [
    "TailtuneDeprecationWarning",
    "TailtuneError",
    "__version__",
    "doctor",
    "estimate",
]
