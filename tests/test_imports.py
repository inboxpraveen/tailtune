"""Import hygiene: `tailtune --help` must never import torch (02 section 4.1, rule 4).

Each check runs in a fresh interpreter, so sys.modules reflects only what tailtune
imports. On a machine without torch these pass trivially; CI also runs them in a
leg with CPU torch installed, where they are meaningful.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

HEAVY = ("torch", "transformers", "peft", "trl", "numpy")


def _modules_after(code: str) -> set[str]:
    script = f"{code}\nimport sys\nprint('\\n'.join(sorted(sys.modules)))"
    out = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=True, timeout=120
    )
    return set(out.stdout.split())


@pytest.mark.parametrize(
    "code",
    [
        "import tailtune",
        "import tailtune.cli",
        "import tailtune.cli.app",
        "from tailtune.cli import main\ntry:\n    main(['--help'])\nexcept SystemExit:\n    pass",
        "from tailtune.cli import main\nmain(['estimate', '--help'])",
    ],
)
def test_no_heavy_imports(code: str) -> None:
    loaded = _modules_after(code)
    assert not [m for m in HEAVY if m in loaded]


def test_import_tailtune_does_not_touch_the_hub_client() -> None:
    assert "huggingface_hub" not in _modules_after("import tailtune")


def test_public_surface() -> None:
    import tailtune

    assert callable(tailtune.doctor)
    assert callable(tailtune.estimate)
    assert set(tailtune.__all__) >= {"doctor", "estimate", "TailtuneError", "__version__"}


def test_python_m_tailtune() -> None:
    out = subprocess.run(
        [sys.executable, "-m", "tailtune", "--version"],
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    assert out.returncode == 0
    assert out.stdout.startswith("tailtune ")
