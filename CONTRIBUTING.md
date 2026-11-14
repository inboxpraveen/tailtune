# Contributing to tailtune

Thank you for helping. tailtune is pre-alpha: 0.0.1 ships `tailtune doctor` and
`tailtune estimate`, and the [implementation plan](docs/plan/README.md) describes
everything else. Read the plan's index before proposing a large change, because
some decisions are locked (D-01 to D-25) and change only through a gate result or
an RFC.

## Set up

You need Python 3.11 or later and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/inboxpraveen/tailtune
cd tailtune
uv sync --group dev
uv run tailtune doctor
```

torch is not needed for 0.0.1. To exercise the torch code paths in `doctor`, add a
CPU build to the environment:

```bash
uv pip install "torch>=2.14,<2.16" --torch-backend=cpu
```

## Check your change

```bash
uv run ruff check .
uv run ruff format --check .
uv run pyright
uv run pytest
```

Tests that need the Hugging Face Hub are marked `network` and skipped by default.
Run them with `uv run pytest -m network`.

## Sign your commits (DCO)

tailtune uses the [Developer Certificate of Origin](https://developercertificate.org/)
instead of a CLA. Every commit must carry a `Signed-off-by:` line with your real
name and email, which `git commit -s` adds. By signing off you certify that you wrote
the change, or otherwise have the right to submit it under the Apache-2.0 licence.
A CI check rejects pull requests with unsigned commits.

To sign off commits you already made on a branch:

```bash
git rebase --signoff main
```

## Pull requests

- Keep each pull request to one change, with a test.
- Add a line under "Unreleased" in `CHANGELOG.md` for anything a user would notice.
- Do not copy code from other projects. If adapted code is unavoidable, its licence
  must allow it, and it is recorded in `THIRD_PARTY_NOTICES.md` in the same pull
  request.
- Label claims in docs and output: sourced, measured, computed or unverified. A
  memory figure is "computed" until it has been checked on real hardware.

## Adding a model architecture to `estimate`

`tailtune estimate` refuses architectures whose shape arithmetic has not been checked
(error TT-T004). To add one:

1. Add the `model_type` to `src/tailtune/hardware/shapes.py`, with any shape
   differences from the seven standard projections.
2. Add a trimmed `config.json` for a model with an Apache-2.0 or MIT licence to
   `tests/fixtures/configs/`.
3. Add the repository to `tests/test_shapes_network.py`, so the parameter count is
   checked against the model's safetensors headers, and run `uv run pytest -m network`.

## Where to ask

Questions go to [Discussions](https://github.com/inboxpraveen/tailtune/discussions),
not issues. See [SUPPORT.md](SUPPORT.md). Everyone taking part follows the
[Code of Conduct](CODE_OF_CONDUCT.md).
