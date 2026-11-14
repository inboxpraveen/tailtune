## What this changes

<!-- One or two sentences. Link the issue or discussion. -->

## How it was tested

<!-- Commands you ran and what they showed. For memory or numeric changes, say whether
the figures were computed or measured, and on what hardware. -->

## Checklist

- [ ] Every commit is signed off (`git commit -s`), per the DCO in CONTRIBUTING.md
- [ ] `uv run ruff check .`, `uv run ruff format --check .` and `uv run pyright` pass
- [ ] `uv run pytest` passes, and new behaviour has a test
- [ ] User-visible changes have a line under "Unreleased" in CHANGELOG.md
- [ ] No code copied from other projects, or its licence is recorded in THIRD_PARTY_NOTICES.md
- [ ] Claims in docs or output are labelled: sourced, measured, computed or unverified
