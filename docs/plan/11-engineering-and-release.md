# 11. Engineering and release

**What this document decides:**

- The repository meets the PyTorch Ecosystem layout from the first commit, plus the files a sponsor or security reviewer looks for.
- tailtune ships as a pure `py3-none-any` wheel built with hatchling. It declares tested dependency ranges, commits `uv.lock`, and has no CUDA extras.
- 0.0.1 is a real, small tool (`tailtune doctor` and `tailtune estimate`), published through Trusted Publishing behind a reviewer gate. A placeholder would not hold the name.
- Tests are built around a few invariants that fail silently when broken. They run on free CPU runners on every PR, on a free Kaggle T4 weekly, on Modal L4 nightly, and on 2 GPUs per release.
- Releases use 0.x SemVer with a written deprecation policy and a support table per release.
- Security comes first in the release pipeline. The OpenSSF Scorecard target is 7+ by Gate 5. There is no telemetry at launch.
- Docs build on Zensical with Material as a tested fallback, serve `llms.txt`, and test their notebooks with papermill.

Evidence labels are defined in the [plan index](README.md#evidence-labels). Package layout under `src/tailtune/` belongs to [02](02-architecture.md), the docs site structure and notebooks to [04](04-api-and-ux.md), hardware tiers to [07](07-hardware.md), FSDP2 and ZeRO to [08](08-distributed.md), the evaluation sandbox and statistics to [09](09-evaluation.md), and the GGUF environment to [10](10-export-and-deployment.md). Figures were retrieved on 2026-11-07 and 2026-11-08.

All code and configuration in this document are original, untested design proposals.

## 1. Repository layout

### Top level

```text
tailtune/
├── .github/
│   ├── CODEOWNERS
│   ├── dependabot.yml
│   ├── FUNDING.yml
│   ├── pull_request_template.md
│   ├── ISSUE_TEMPLATE/          bug.yml, model-support.yml, feature.yml, config.yml
│   ├── requirements/uv.txt      hash-pinned uv for CI bootstrap
│   └── workflows/               see section 5
├── benchmarks/                  regression workload and per-release results
├── docs/                        docs site source (structure in 04), plus docs/plan/
├── notebooks/                   generated from one template (04)
├── examples/                    short scripts, each exercised in CI
├── rfcs/                        accepted design records (12)
├── scripts/                     fixture generators, tiny-model builders, release helpers
├── src/tailtune/                package (layout in 02)
├── tests/                       see section 4
├── CHANGELOG.md
├── CITATION.cff
├── CODE_OF_CONDUCT.md
├── CONTRIBUTING.md
├── GOVERNANCE.md
├── LICENSE
├── MAINTAINERS.md
├── NOTICE
├── README.md
├── SECURITY.md
├── SUPPORT.md
├── THIRD_PARTY_NOTICES.md
├── mkdocs.yml
├── pyproject.toml
└── uv.lock
```

The research reports and notes currently sit untracked at the repo root. Where they live long term is the owner's call. This document does not move them.

### What each file is for

| File | Purpose | Gate |
|---|---|---|
| `LICENSE` | Apache-2.0 text. Already present | 0 |
| `NOTICE` | Apache-2.0 attribution notice, copied into the wheel | 0 |
| `THIRD_PARTY_NOTICES.md` | Licence notices for any adapted code, such as the mini_trainer reference noted in [03](03-methods.md). It starts empty and grows only through review (D-06) | 0 |
| `README.md` | Today's text is in [01](01-product-strategy.md). The standard for v0.2 is in section 10 | 0 |
| `CONTRIBUTING.md` | Setup with uv, test commands, DCO sign-off, the PR checklist, and how to add a model architecture | 0 |
| `CODE_OF_CONDUCT.md` | Choice and enforcement contact in [12](12-community-and-sponsorship.md) | 0 |
| `SECURITY.md` | Section 7 | 0 |
| `GOVERNANCE.md`, `MAINTAINERS.md` | BDFL model and maintainer path (D-24, [12](12-community-and-sponsorship.md)) | 0 |
| `.github/CODEOWNERS` | Required by the PyTorch layout. It also protects `.github/workflows/` | 0 |
| `CHANGELOG.md` | Hand-written. Section 6 | 0 |
| `CITATION.cff` | Makes the repo citable now. The tech report entry is added at Gate 3 | 0 |
| `pyproject.toml`, `uv.lock` | Section 2 | 0 |
| `.github/dependabot.yml` | Actions and Python updates. Section 7 | 0 |
| `.github/FUNDING.yml` | Points to GitHub Sponsors once it opens at v0.1 ([12](12-community-and-sponsorship.md)) | 2 |
| `SUPPORT.md` | Where to ask: Discussions, not issues. No private support by direct message | 0 |
| `mkdocs.yml` | Conservative config shared by Zensical and Material. Section 9 | 2 |
| `benchmarks/` | Regression workload definitions and results JSON per release | 2 |
| `rfcs/` | Accepted RFCs, one Markdown file each | 2 |

### PyTorch Ecosystem layout checklist

The PyTorch Ecosystem process lists measurable, functional and layout requirements [S](https://github.com/pytorch-fdn/tac/blob/main/docs/governance/PyTorch_Ecosystem_Process.md). The layout and functional items cost little, so meet them from day one. The measurable ones are milestones (M1 in [12](12-community-and-sponsorship.md)).

| Requirement | tailtune item | When |
|---|---|---|
| `LICENSE.md` | `LICENSE` exists. Whether the working group accepts `LICENSE` without the `.md` suffix is unverified [U]. Check before applying, and do not rename it before then | Gate 0 check |
| `README.md` with release cadence and criteria | A "Releases" section (section 6) | Gate 0 |
| `CONTRIBUTING.md` | As above | Gate 0 |
| `CODEOWNERS` | `.github/CODEOWNERS` | Gate 0 |
| `CODE_OF_CONDUCT.md` | The Linux Foundation code is named as the default. See [12](12-community-and-sponsorship.md) | Gate 0 |
| Documented technical governance | `GOVERNANCE.md` | Gate 0 |
| Permissive licence | Apache-2.0 | Done |
| Working CI for all declared architectures | Section 5. Declare only what CI runs | Gate 2 |
| Quick start | README | Gate 2 |
| Binary install via pip or Conda | PyPI from 0.0.1, conda-forge later | Gate 0 |
| 12-month maintenance commitment | Stated in `GOVERNANCE.md` | Gate 0 |
| Support for the latest two PyTorch releases | torch 2.14 and 2.15 tested (D-05) | Gate 0 |
| CLA or DCO on onboarding | DCO (D-06) | Gate 0 |
| At least 2 core maintainers, 200 stars, 30+ commits in 90 days | Milestone M1 | Later |

## 2. Python packaging

### `pyproject.toml`

This sample is for v0.1. The 0.0.1 variant follows below it. Version floors are the releases current on 2026-11-08 [S](https://pypi.org/pypi/transformers/json). Re-check every one before use.

```toml
# Design proposal, untested.
[build-system]
requires = ["hatchling>=1.33"]
build-backend = "hatchling.build"

[project]
name = "tailtune"
dynamic = ["version"]
description = "Fine-tune LLMs with far less forgetting, and prove it."
readme = "README.md"
license = "Apache-2.0"
license-files = ["LICENSE", "NOTICE"]
requires-python = ">=3.11"
authors = [{ name = "tailtune maintainers" }]
classifiers = [
  "Development Status :: 3 - Alpha",
  "License :: OSI Approved :: Apache Software License",
  "Programming Language :: Python :: 3 :: Only",
  "Programming Language :: Python :: 3.11",
  "Programming Language :: Python :: 3.12",
  "Programming Language :: Python :: 3.13",
]
# Tested ranges. Upper caps on the fast movers sit at the next minor not yet tested.
# compat-weekly opens a PR to lift a cap once that minor passes.
dependencies = [
  "torch>=2.14,<2.16",
  "transformers>=5.22,<5.23",
  "trl>=1.17,<1.18",
  "peft>=0.21.2,<0.22",
  "accelerate>=1.14,<2",
  "datasets>=5.1,<6",
  "safetensors",            # floor set from uv.lock at Gate 2
  "huggingface-hub",        # floor set from uv.lock at Gate 2
  "python-calamine>=0.9",
  "openpyxl>=3.1.5",
  "pyyaml",
  "psutil",
  # CLI and terminal libraries: as chosen in 02-architecture.md
]

[project.optional-dependencies]
bnb   = ["bitsandbytes"]     # 4-bit base for lora
flash = ["flash-attn"]       # optional attention kernel
vllm  = ["vllm"]             # fast eval backend; its torch pin must overlap ours
deepspeed = ["deepspeed"]    # zero2 reference backend, later zero3 (08)
docs  = ["docling", "markitdown"]   # PDF, DOCX and HTML ingestion (05), not the docs site
wandb = ["wandb"]

[project.scripts]
tailtune = "tailtune.cli:main"

[project.urls]
Source = "https://github.com/inboxpraveen/tailtune"

[dependency-groups]
test      = ["pytest>=8", "pytest-timeout", "hypothesis"]
lint      = ["ruff", "pyright"]
notebooks = ["papermill>=2.7", "ipykernel"]
docs-site = ["zensical", "mkdocs-material"]
dev       = [{ include-group = "test" }, { include-group = "lint" }]

[tool.hatch.version]
path = "src/tailtune/__init__.py"

[tool.hatch.build.targets.wheel]
packages = ["src/tailtune"]

[tool.hatch.build.targets.sdist]
include = ["src/", "tests/", "LICENSE", "NOTICE", "README.md", "CHANGELOG.md"]

[tool.ruff]
line-length = 100
target-version = "py311"
src = ["src", "tests"]

[tool.ruff.lint]
select = ["E", "F", "W", "I", "B", "UP", "S", "PT", "C4", "SIM", "RUF", "TID"]

[tool.ruff.lint.per-file-ignores]
"tests/**" = ["S101"]

# Enforce safetensors-only and safe YAML inside our own code.
[tool.ruff.lint.flake8-tidy-imports.banned-api]
"torch.load".msg = "Pickle checkpoints are refused. Use safetensors (SECURITY.md)."
"pickle".msg = "No pickle anywhere in tailtune. Use safetensors, Parquet or JSON."
"yaml.load".msg = "Use yaml.safe_load."

[tool.pyright]
include = ["src"]
pythonVersion = "3.11"
typeCheckingMode = "standard"
strict = ["src/tailtune/bases", "src/tailtune/methods", "src/tailtune/data"]
reportMissingTypeStubs = false

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "--strict-markers -m 'not gpu and not network and not slow'"
markers = [
  "gpu: needs a CUDA device",
  "network: needs the Hugging Face Hub",
  "slow: over 60 s on a CPU runner",
  "distributed: spawns 2 gloo ranks",
  "quarantine: known flaky, tracked in a linked issue",
]
filterwarnings = ["error::tailtune.TailtuneDeprecationWarning"]
```

Notes on the choices:

- **`docs` means documents.** The research report reserves `tailtune[docs]` for PDF, DOCX and HTML ingestion through Docling or MarkItDown, both MIT ([05](05-data-ingestion.md)). Docs-site tooling therefore goes in the `docs-site` dependency group, which is never published as an extra.
- **No CUDA extras (D-07).** Unsloth composes a large extras matrix, such as `cu126-torch2110`, with leaf extras pinning `torch==2.11.0+cu126` [S](https://raw.githubusercontent.com/unslothai/unsloth/main/pyproject.toml). tailtune keeps `torch` a plain range and lets the installer pick the build.
- **No GGUF extra.** GGUF conversion runs in its own environment, because llama.cpp's converter pins `transformers==4.57.6` and `torch==2.11.0` [S](https://github.com/ggml-org/llama.cpp/blob/master/requirements/requirements-convert_hf_to_gguf.txt). See [10](10-export-and-deployment.md).
- **pyright, not ty.** ty is beta at `0.0.x`, with breaking changes possible in any release [S](https://raw.githubusercontent.com/astral-sh/ty/main/README.md). It may run as a non-blocking extra job [J].
- **Banned APIs.** The Ruff `banned-api` table turns the safetensors-only rule (D-08) into a lint failure, so it does not rely on review.
- **Deprecation warnings.** `filterwarnings` turns tailtune's own deprecation warnings into test errors, so internal code never calls a deprecated path.

**0.0.1 variant [J].** 0.0.1 ships only `doctor` and `estimate`. It depends on `huggingface-hub`, `packaging`, `psutil` and `typer` (the CLI library chosen in [02](02-architecture.md)), and does not require torch or Transformers. `doctor` probes whatever torch, Transformers, TRL and PEFT are already installed and reports them against the supported ranges, so it describes the user's real environment rather than one it just installed. `estimate` is arithmetic on a downloaded `config.json`. The full dependency list arrives at v0.1.

### Why a pure wheel

A native wheel needs a build matrix per platform, Python version and GPU, which one person cannot sustain alongside new-model work (D-04). Triton kernels compile at runtime from Python source, so even a later Triton hot op keeps the wheel `py3-none-any` [S](https://api.github.com/repos/unslothai/unsloth/contents/unsloth/kernels). LlamaFactory reached 75,927 stars with no native code [S](https://api.github.com/repos/hiyouga/LLaMA-Factory/languages). Unsloth's CI shows the alternative's cost: workflows such as `prebuilt-cuda-wheels`, `windows-amsi-defender-differential-ci` and `release-desktop` [S](https://api.github.com/repos/unslothai/unsloth/contents/.github/workflows). The release job (section 5) fails if the built wheel is not `py3-none-any`.

### Installing torch

| Who | Command | Notes |
|---|---|---|
| Users, primary | `uv pip install tailtune --torch-backend=auto` | `--torch-backend=auto` queries the CUDA driver, AMD GPU versions and Intel GPU presence, picks the matching PyTorch index and falls back to CPU. It exists only in the `uv pip` interface [S](https://docs.astral.sh/uv/guides/integration/pytorch/) |
| Users, plain pip | `pip install tailtune` | Works when a suitable torch is already installed, as on Colab. Otherwise the docs tell users to install torch first using PyTorch's own instructions |
| Colab | `pip install tailtune` | Colab's snapshot lists `torch 2.14.1+cu130` [S](https://github.com/googlecolab/backend-info/blob/main/pip-freeze.gpu.txt). Because the floor is 2.14, pip keeps Colab's torch rather than replacing it. The daily `colab-resolve` job checks that this stays true |
| Contributors and CI | `uv sync --locked --group dev` | The lock resolves torch from PyPI. The uv guide shows explicit indexes, markers and `tool.uv.conflicts` for `uv sync` [S](https://docs.astral.sh/uv/guides/integration/pytorch/). Adopt that only if CPU runner time becomes a problem, and keep it out of published metadata |

Unsloth documents the same primary command. On Windows it tells users to install torchao first, because `--torch-backend` looks for it only on the PyTorch index, which has no recent Windows build, and a one-step install can silently resolve an older Unsloth [S](https://github.com/unslothai/unsloth/blob/main/README.md). tailtune has no torchao requirement, but the Windows install docs must be tested on a clean Windows machine before v0.1 [J].

The status of wheel-variant standards for GPU-specific wheels was not checked in the research [U]. Re-check before v1.0, because it could change this advice. Astral, which makes uv and Ruff, announced on 2026-03-19 that it would be acquired by OpenAI, with the tools to stay open source [S](https://simonwillison.net/2026/mar/19/openai-acquiring-astral/). The tools are permissively licensed, so the risk is low. Even so, never make uv the only working path [J].

### Dependency policy for fast movers

In 2026 so far, Transformers shipped 48 releases and TRL 32 [S](https://pypi.org/pypi/transformers/json), [S](https://pypi.org/pypi/trl/json). Those counts include patch releases, and the research did not split them by minor version. For comparison, Axolotl pins `transformers==5.22.0`, `trl==1.16.0` and `peft==0.21.2` exactly [S](https://raw.githubusercontent.com/axolotl-ai-cloud/axolotl/main/pyproject.toml). An application can do that. A library that users install next to other packages cannot.

The policy [J]:

1. **Floors** are the oldest version CI has tested. For Transformers that floor is v5 only, with no v4 compatibility layer (D-05).
2. **Caps.** For `transformers`, `trl` and `peft`, the cap sits at the next minor version not yet tested. `torch` is capped at the next minor after the two tested releases. Everything else is capped at the next major.
3. **Widening.** When `compat-weekly` passes on a new minor, it opens a PR that lifts the cap. Merging that PR ships a patch release within 7 days [J].
4. **Lockfile.** Commit `uv.lock`, and attach a `constraints-<version>.txt` exported from it to each GitHub Release as the known-good set.
5. **Breakage.** If a new upstream minor breaks tailtune, the cap stays. The nightly `transformers-main` job should have warned about it a week or more earlier.

**Support table.** This is published in the docs and linked from the README, with one row per release. The first rows are placeholders:

| tailtune | Python | torch | Transformers | TRL | PEFT | Notes |
|---|---|---|---|---|---|---|
| 0.0.1 | 3.11 to 3.13 | 2.14, 2.15 probed, not required | probed | probed | probed | `doctor`, `estimate` only |
| 0.1.0 | 3.11 to 3.13 | 2.14, 2.15 | 5.22.x | 1.17.x | 0.21.x | Placeholder until release |
| 0.1.x | same | same | widened by compat PRs | same | same | One row per patch that changes a range |

CI tests Python 3.11 and 3.13 on every PR and 3.12 weekly, following the hardware research's matrix [J].

## 3. The 0.0.1 release

### Why a real release, and why now

- **A pending publisher does not hold the name.** PyPI's docs say a pending publisher "does **not** create a project or reserve a project's name **until** it is actually used to publish". If someone registers the name first, the pending publisher is invalidated [S](https://docs.pypi.org/trusted-publishers/creating-a-project-through-oidc/).
- **An empty stub can be taken away.** PEP 541 lists "project is name squatting (package has no functionality or is empty)" as a condition for removal [S](https://peps.python.org/pep-0541/). The same PEP protects a reachable owner from reassignment, and its abandonment test needs no release in twelve months, so release at least once a year.
- **The name is open today.** `tailtune` returned 404 on PyPI at retrieval [S](https://pypi.org/pypi/tailtune/json).

### What 0.0.1 contains

| Command | Does | Output |
|---|---|---|
| `tailtune doctor` | Reports Python, OS, torch version and build, backend (CUDA, ROCm, MPS, CPU), driver, GPU name and compute capability, VRAM, host RAM, real bf16 support using `is_bf16_supported(including_emulation=False)` [S](https://github.com/pytorch/pytorch/blob/v2.15.0/torch/cuda/__init__.py#L244-L263), and installed Transformers, TRL and PEFT against the supported ranges. It names the D-17 tier and warns on unsupported setups such as Pascal or Python 3.10 | Terminal table plus `--json` |
| `tailtune estimate <model>` | Reads `config.json` from the Hub without weights and prints the memory budget per route (LoRA on a 4-bit base, LoRA on a 16-bit base, full, constrained) for the detected or named GPU | Every figure marked as computed and not yet checked against hardware, until Gate 1 |

Both are useful to someone today: an Unsloth user can check whether SST would fit their card before anything else exists. Neither needs a GPU to run.

### Step by step

One-time setup:

1. **Decide where the repo lives.** A Trusted Publisher is bound to an owner, a repository, a workflow file and an environment. If the repo will move to a `tailtune` GitHub org, create the org and transfer the repo **before step 4**. Otherwise you must edit the publisher after the move [J]. [13](13-roadmap.md) lists the org claim after the first publish, so this recommendation moves only the org step earlier.
2. **Secure the accounts.** Turn on 2FA for PyPI and GitHub, as the research recommends. Prefer a hardware key and keep recovery codes offline [J]. Create no PyPI API token at all.
3. **Create the `pypi` environment in GitHub.**
   - Add a required reviewer, which is the owner today.
   - Restrict deployments to tags matching `v*`.
   - Add a tag ruleset so only maintainers can create `v*` tags [J].
   - With one maintainer, the owner approves their own deployments. The gate is then a deliberate pause and a block on any compromise that only controls workflow code, not two-person control. Turn on "prevent self-review" once a second maintainer exists [J].
4. **Add the pending publisher on PyPI.** Use the "Publishing" sidebar link [S](https://docs.pypi.org/trusted-publishers/creating-a-project-through-oidc/) with:
   - project `tailtune`
   - the owner and repository from step 1
   - workflow `release.yml`
   - environment `pypi`
5. **Commit the workflows.** Add `ci.yml` and `release.yml` from section 5, with every action pinned by SHA.

Each release:

6. Open a release PR that bumps the version in `src/tailtune/__init__.py` to `0.0.1`, writes the `CHANGELOG.md` entry and adds the support-table row. Wait for CI to pass, then merge.
7. Tag `v0.0.1` on `main` and push the tag.
8. The `build` job builds the sdist and wheel, checks that the wheel is `py3-none-any` and matches the tag, and smoke-tests `tailtune doctor` from the built wheel.
9. Read the build log, then approve the `publish` job in the `pypi` environment. It uploads through Trusted Publishing, and with `pypa/gh-action-pypi-publish` attestations are "generated and uploaded automatically by default" [S](https://docs.pypi.org/attestations/producing-attestations/).

Verification, in order:

10. `https://pypi.org/pypi/tailtune/json` returns 200 and lists `0.0.1` with a wheel and an sdist.
11. Confirm the attestation exists for both files, as PyPI's attestation docs describe [S](https://docs.pypi.org/attestations/producing-attestations/).
12. Confirm the pending publisher became a normal publisher on the PyPI project.
13. In clean environments on Linux, macOS and Windows, run `pip install tailtune==0.0.1`, then `tailtune doctor`, then `tailtune estimate Qwen/Qwen3-0.6B`.
14. Create the GitHub Release with the changelog text and the built files.
15. Delete any PyPI token that exists. Restore the PyPI badge to the README, which [01](01-product-strategy.md) removed until this point.

This procedure is Gate 0's exit test: a real upload with an attestation.

### Namespace reservation

| Order | Namespace | State at retrieval | Action |
|---|---|---|---|
| 1 | PyPI `tailtune` | 404 [S](https://pypi.org/pypi/tailtune/json) | Real 0.0.1 as above |
| 2 | GitHub org `tailtune` | 404 [S](https://github.com/tailtune) | Create it. Decide on the repo transfer before step 4 above |
| 3 | Hugging Face org `tailtune` | 404 [S](https://huggingface.co/tailtune) | Create it. Also create `tailtune-testing` for tiny test models (section 4) |
| 4 | Domain | `tailtune.com` has nameservers at `squadhelp.com`, a domain marketplace, so it is likely parked for sale. `.ai`, `.dev`, `.org` and `.io` returned NXDOMAIN [S](https://dns.google/resolve?name=tailtune.com&type=NS), which does not prove they are available | Check with a registrar, then buy `.dev` or `.ai`. Do not buy `tailtune.com` at marketplace prices before traction [J], and never link to it |
| 5 | conda-forge | 404 [S](https://api.anaconda.org/package/conda-forge/tailtune) | Submit a recipe only after a real PyPI release exists |
| 6 | X handle | Not checked: needs login [U] | Check and claim |
| 7 | Discord | Not checked: needs login [U] | Claim the server name. Open it only on the trigger in [12](12-community-and-sponsorship.md) |
| — | npm `tailtune` | 404 [S](https://registry.npmjs.org/tailtune) | Leave it. An empty npm package would itself be squatting |

The research put conda-forge last. This plan moves it before X because it is a packaging channel, but it still waits for the release [J].

### Trademark

Trademark was not researched. India and USPTO fees, the Madrid route and a clearance search were not checked, and no search for conflicting "tailtune" marks was run [U]. The position is to defer paid filing until traction (M3 or M4 in [12](12-community-and-sponsorship.md)) [J]. Before spending on a logo or a paid domain, run a basic search of public trademark databases, and keep dated records of first public use, such as the 0.0.1 release date [J]. None of this is legal advice.

## 4. Test strategy

### Layout

```text
tests/
├── conftest.py      seeds, tiny-model factories, temporary run dirs, offline Hub cache
├── unit/            config resolution, exit codes, estimate formula, CLI parsing
├── numerics/        bases, projector maths, subspace distance, precision guards
├── methods/         each method on each tiny model, single process
├── distributed/     2-rank gloo DDP and FSDP2 (later ZeRO-2), spawned in-test
├── data/            the ingest pipeline against fixtures
│   └── fixtures/    generated by scripts/make_data_fixtures.py, committed
├── eval/            paired statistics against known answers, report rendering
├── export/          merged safetensors round trip; GGUF round trip (slow)
├── integration/     two-command flow on CPU with a tiny model
├── gpu/             marked gpu; Kaggle and Modal only
└── notebooks/       papermill runner and parameters
```

Distributed tests start their own 2 ranks with the gloo backend inside the test. A Transformers FSDP2 reproduction did the same on CPU, which shows the route works without GPUs [S](https://github.com/huggingface/transformers/issues/50424).

### Tiny random models per architecture

Peers keep one tiny model per architecture. TRL uses models such as `trl-internal-testing/tiny-Gemma4ForConditionalGeneration` [S](https://raw.githubusercontent.com/huggingface/trl/main/tests/test_sft_trainer.py), and Axolotl hosts `tiny-qwen3-129m` and similar under its own org [S](https://raw.githubusercontent.com/axolotl-ai-cloud/axolotl/main/tests/conftest.py).

tailtune keeps two kinds [J]:

| Kind | Built from | Used for |
|---|---|---|
| In-process | A reduced config in `tests/models/<model_type>.json`, such as 2 layers and hidden size 64, instantiated with a fixed seed. No network | Most method, numerics and distributed tests |
| Hub-hosted | `scripts/make_tiny_models.py` builds each one, pushes it to `tailtune-testing`, and tests pin it by commit hash. The CI cache is warm, and tests run with the Hub offline | Loading paths: checkpoint versus in-memory names (v5's `WeightConverter` can fuse at load), tokenizer and template, export round trip |

The architectures follow D-20 and D-15:

| `model_type` | Exercises | When |
|---|---|---|
| `qwen3` | Dense, tied embeddings (0.6B shape) | Gate 0 |
| `llama` | Dense, Llama-3.1 and SmolLM2 shape | Gate 2 |
| `granite` | Dense, untied | Gate 2 |
| `gemma4` | Composite, towers, tied, sliding window | Gate 2 |
| synthetic `[E,m,n]` MoE | 3D experts treated as E matrices, router frozen | Gate 2 |
| GPT-2-style Conv1D | Conv1D target refused (D-15) | Gate 2 |
| `qwen3_5` | Hybrid, fused QKV, MTP head frozen | Gate 4 |
| `gpt_oss` | MoE, quantized targets refused | Gate 4 |

Copy tokenizer files only from models whose licence allows redistribution, and check each one. Llama-licensed files need the review noted in [10](10-export-and-deployment.md).

### The invariants that matter most

These break silently, so they get the most test effort. Each method in `tailtune.methods` exposes an invariant check that the tests call, so OSF's factor-space rule and SST's weight-space rule are each tested on their own terms (D-11, [03](03-methods.md)). Tolerances are starting points [J] to tighten after Gate 1.

| ID | Invariant | How the test checks it | Runs on |
|---|---|---|---|
| I1 | The hook ran after the real optimizer step (D-13) | Record a target weight's checksum at `on_step_begin`. In `on_optimizer_step` it must already differ. With gradient accumulation 4 over 40 micro-batches, the hook fires exactly 10 times. Under Accelerate's DeepSpeed wrapper the real step happens inside `backward` [S](https://github.com/huggingface/accelerate/blob/main/src/accelerate/utils/deepspeed.py#L264-L312), so this test must also run under ZeRO-2 once that backend exists | Every backend |
| I2 | Skipped steps are skipped | Force a skipped step and check that projection does not run and W is unchanged. The next real step's cumulative projection from `W_ref` still holds I3 [S](https://github.com/huggingface/transformers/blob/main/src/transformers/trainer.py#L1934-L1941) | Single, and the Kaggle T4 fp16 leg |
| I3 | Frozen-subspace content is unchanged after N steps | After N=20 optimizer steps at a learning rate high enough to move weights, for every target with `d = W − W_ref`: `‖U_kᵀ d‖_F / ‖d‖_F ≤ 1e-5` and `‖d V_k‖_F / ‖d‖_F ≤ 1e-5` in fp32. Also `‖d‖_F > 0`, so a no-op cannot pass | `single`, `ddp`, `fsdp2`, and `zero2` once built, on CPU and on each GPU leg |
| I4 | Frozen parameters are untouched | Routers, conv kernels and `mtp` heads are bit-identical after N steps (D-15) | All backends |
| I5 | Precision guards hold (D-12) | A bf16 checkpoint loaded for a constrained method gives fp32 trainable targets. A low-precision trainable target raises. On a T4, fp16 autocast is selected | CPU, Kaggle T4 |
| I6 | No DDP drift | All ranks hold identical W after N steps | `ddp` |
| I7 | FSDP2 settings are explicit (D-19) | Resolved config sets the FSDP version, `reduce_dtype` and `use_reentrant=False`. A module called twice before `backward` gets the summed gradient, which guards against the silent failure in [S](https://github.com/huggingface/transformers/issues/50424) | `fsdp2` |

**Bases (D-14).** `torch.linalg.svd` returns singular vectors that differ across hardware [S](https://docs.pytorch.org/docs/2.15/generated/torch.linalg.svd.html), so never compare vectors element by element.

| ID | Check | Threshold [J] |
|---|---|---|
| B1 | Orthonormality: `max |UᵀU − I|` | ≤ 1e-5 in fp32 |
| B2 | Subspace distance against a float64 reference: `‖U Uᵀ − U_ref U_refᵀ‖₂` | ≤ 1e-4, on fixtures built with a relative spectral gap of at least 1e-2 at the cut |
| B3 | Hash determinism: same weights, r and parameter name give the same hash | Exact |
| B4 | Cache invalidation: a new weight revision misses the cache | Exact |
| B5 | Small spectral gap at the cut produces a warning, because the subspace is then ill-defined | Warning raised |

B2 runs on all three operating systems against the same stored float64 reference.

**Data fixtures (D-21).** `scripts/make_data_fixtures.py` generates each failure case the research reproduced, so every fixture can be reviewed as code. Each has a golden `data_report.json`:

| Fixture | Expected behaviour |
|---|---|
| Two data sheets plus a Notes tab | Every sheet listed, none auto-selected, clear message |
| Merged cells and a banner row | Merges reported and filled down, header row detected |
| Mixed-type column | Read as text, no `ArrowInvalid` |
| `00123`, a blank in an integer column, and the strings `NA`, `None`, `null` | Preserved exactly as the user sees them |
| Formula with no cached value | Warning with a cell count |
| cp1252 CSV, and a type flip halfway through a file | Detected encoding reported, no late failure |
| Double BOS, missing EOS, all labels masked | Blocking, exit code 2 |

Add Hypothesis property tests where they fit. For example, any cell string must come back from `read` unchanged as text [J].

**Logit round trip (Gate 2 exit test).**

1. Train a tiny model for 5 steps with `lora` and with `sst`.
2. Export merged safetensors and reload them with plain Transformers.
3. Compare logits on 8 fixed prompts. The maximum absolute difference must be at most 1e-5 in fp32 [J].
4. Check that key names match the base checkpoint, that config, tokenizer and chat template are present, and that the template hash matches `lineage.json`.

The GGUF leg converts in the isolated environment from [10](10-export-and-deployment.md) and loads the result in llama.cpp on CPU. It requires 100% top-1 token agreement on the fixed prompts for an f16 GGUF, and a logit tolerance set at Gate 2 [J]. Whether the pinned converter reads tokenizers written by v5 is untested, and the report calls that a real risk [S](https://github.com/ggml-org/llama.cpp/blob/master/requirements/requirements-convert_hf_to_gguf.txt). This test is what will catch it. It runs nightly and at release because it is slow.

**Cross-version checkpoints.** torch 2.14 cannot load distributed checkpoints written by 2.15 [S](https://github.com/pytorch/pytorch/releases/tag/v2.15.0). Never share distributed-checkpoint fixtures between torch matrix legs. Generate them inside each leg.

### Regression benchmark

The `benchmark` job runs a fixed workload on Modal L4:

- `lora` and `sst` on Qwen3-0.6B
- 200 optimizer steps, sequence length 1,024, a fixed seed, and a fixed 2,000-row dataset

It records median step time, peak VRAM (`max_memory_allocated`), final loss, the `estimate` prediction, and every dependency version to `benchmarks/results/<version>.json`. A release is blocked unless the changelog explains any of the following [J]:

- peak VRAM rose more than 5% against the previous release
- median step time rose more than 10%
- final loss moved more than 2% at the same seed and dependencies
- `estimate` under-predicted peak VRAM by more than 10%

The last one matters most, because `estimate` refuses runs (exit code 4) based on that prediction. README VRAM and time figures come only from this job's output files, and retention figures only from the preregistered benchmark ([09](09-evaluation.md)). The research recommended the same per-release record of time, peak VRAM and final loss.

### Flake handling [J]

- Seed everything. Use `torch.use_deterministic_algorithms(True)` in CPU numerics.
- Every test has a timeout: 300 s by default, 120 s for each distributed test.
- No automatic retries for unit, numerics or distributed tests on PRs. GPU and notebook jobs may retry once, and only for an infrastructure error such as a quota, a provisioning failure or a Hub outage. They log both attempts.
- A test that fails without a code change twice in 14 days gets the `quarantine` marker, a linked issue and a 14-day expiry. Quarantined tests still run nightly and report.
- I1 to I7 and B1 to B5 are never quarantined. A failure there blocks the release until it is fixed.

## 5. CI

### Matrix

Costs use Modal's listed rates: L4 $0.80/h, A100 80 GB $2.50/h, RTX PRO 6000 $3.03/h, and $30 a month of free credit on the Starter plan [S](https://modal.com/pricing). GitHub's own GPU runners cost $3.12/h and need a paid plan, so they are skipped [S](https://docs.github.com/en/billing/reference/actions-runner-pricing).

| Job | Trigger | Runner | Cost | What it protects |
|---|---|---|---|---|
| `lint` | Every PR, push to `main` | GitHub Ubuntu | $0 | Ruff, format, pyright, banned APIs, DCO sign-off |
| `test-cpu` | Every PR | Ubuntu: Python 3.11 and 3.13 × torch 2.14 and 2.15. macOS and Windows: locked versions, Python 3.11 and 3.13 | $0 | Unit, numerics, methods, data, eval, merged export. Tier 1 numerics on three OSes (D-17) |
| `distributed-cpu` | Every PR | Ubuntu, macOS, Windows | $0 | 2-rank gloo DDP and FSDP2, I1 to I7. FSDP2 over gloo on Windows is unverified [U]. If it fails at Gate 0, run DDP only on Windows, since Windows multi-GPU is unsupported anyway (D-17) |
| `notebooks-cpu` | PRs touching `notebooks/` or `src/`, nightly | Ubuntu | $0 | papermill with tiny-model parameters |
| `colab-resolve` | Daily | Ubuntu | $0 | tailtune's ranges still resolve against Colab's published package snapshot [S](https://github.com/googlecolab/backend-info/blob/main/pip-freeze.gpu.txt) |
| `docs` | PRs touching docs, push to `main` | Ubuntu | $0 | Strict builds on Zensical and on Material, link check, `llms.txt` generated |
| `codeql` | Every PR, weekly | Ubuntu | $0 | Static analysis (Scorecard SAST) |
| `scorecard` | Weekly, push to `main` | Ubuntu | $0 | Published score and badge |
| `kaggle-t4` | Weekly, manual | Kaggle T4 kernel driven from Actions | $0 within quota [U] | fp16 autocast with fp32 master. **More than 5 optimizer steps** (use 20), because under fp16 the GradScaler skips the first steps, so runs shorter than about 5 steps apply no updates [S](https://github.com/unslothai/unsloth/blob/main/.github/workflows/kaggle-t4-notebook-ci.yml). A control leg on `uv.lock` pins and a canary leg on latest releases. The T4 notebook end to end with `lora` and the report |
| `gpu-nightly` | Nightly on `main` | Modal L4 | About 10 h a month at $0.80, so about $8 [C], inside the $30 credit | Qwen3-0.6B per method, I1 to I5 in bf16, measured memory against `estimate` |
| `gpu-dispatch` | Maintainer `workflow_dispatch` on a reviewed ref | Modal L4 | From the same credit | A GPU check for a risky PR before merge |
| `transformers-main` | Nightly | Ubuntu CPU | $0 | Transformers `main` with latest TRL and PEFT. Failure opens or updates one issue and does not block |
| `compat-weekly` | Weekly | Ubuntu CPU | $0 | Floors and latest releases of each dependency. On a green new minor, opens a cap-lifting PR |
| `release-gate` | Manual, on the release PR head | Modal, 2 × L4 or 2 × A100 | About 2 h × 2 GPUs × $2.50, so about $10 per release [C] | DDP and FSDP2 on 2 GPUs, save and reload, I1 to I7. ZeRO-2 once built. ZeRO-3 only once it exists (D-19). Whether Modal's credit covers multi-GPU apps is unconfirmed [U] |
| `release-blackwell` | Manual, per release | Modal RTX PRO 6000 | About 1 to 2 h, so $3 to $6 [C] | Blackwell at the release gate (D-17) |
| `benchmark` | Release PR head, then the release tag | Modal L4 | About $1 [C] | Regression thresholds above, README numbers |
| `release` | Tag `v[0-9]+.[0-9]+.[0-9]+` | GitHub Ubuntu, `pypi` environment | $0 | Build, publish, attestations |

At one release a month, the cash cost is about $8 a month for nightly runs plus $14 to $17 per release [C]. That fits inside Modal's $30 credit only if the credit covers multi-GPU runs [U]. $30 buys about 37 L4-hours [C], so the nightly budget has room to grow. Kaggle's free tier carries most T4 risk. Its weekly quota of about 30 h is user-reported [U], its run-time limit is reported as both 9 h and 12 h [U], and its terms on automated kernels and multiple accounts were not read [U]. Use one account and keep the weekly job under 1 hour. Read Kaggle's terms before Gate 2.

Comparable projects converged on this layered pattern:

- Axolotl runs nightly GPU jobs on `[self-hosted, modal]` runners and installs Transformers from `main` [S](https://raw.githubusercontent.com/axolotl-ai-cloud/axolotl/main/.github/workflows/tests-nightly.yml).
- TRL runs `tests_latest` and `tests_transformers_branch` [S](https://api.github.com/repos/huggingface/trl/contents/.github/workflows).
- Unsloth's Kaggle job has control and canary legs [S](https://github.com/unslothai/unsloth/blob/main/.github/workflows/kaggle-t4-notebook-ci.yml).

That Kaggle file carries an `AGPL-3.0-only` header, so tailtune copies the pattern and none of the code (D-06).

### GPU runs and untrusted code

The Modal token and Kaggle credentials appear only in workflows triggered by `schedule`, `workflow_dispatch`, a push to `main`, or a tag. Never use `pull_request_target` with the PR's code checked out [J]. DeepSpeed's pattern is the model: select tests on a runner with no secrets, then run them on Modal [S](https://github.com/deepspeedai/DeepSpeed/blob/master/.github/workflows/modal-torch-latest.yml). For an outside PR, a maintainer reads the diff and then dispatches `gpu-dispatch` on that exact commit.

### Sample workflows

Replace every `<sha>` with the full 40-character commit SHA of the named release, and keep the version comment so updates stay readable. The only actions used are GitHub's own `actions/*` and `pypa/gh-action-pypi-publish`. uv is installed from a hash-pinned requirements file rather than a third-party action, which also counts toward Scorecard's pinned-dependency check.

```yaml
# .github/workflows/ci.yml (excerpt). Design proposal, untested.
name: ci
on:
  pull_request:
  push:
    branches: [main]

permissions:
  contents: read

concurrency:
  group: ci-${{ github.ref }}
  cancel-in-progress: true

jobs:
  test-cpu:
    name: cpu / ${{ matrix.os }} / py${{ matrix.python }} / torch ${{ matrix.torch }}
    runs-on: ${{ matrix.os }}
    timeout-minutes: 30
    strategy:
      fail-fast: false
      matrix:
        os: [ubuntu-latest]
        python: ["3.11", "3.13"]
        torch: ["2.14", "2.15"]
        include:
          - { os: macos-latest,   python: "3.11", torch: locked }
          - { os: macos-latest,   python: "3.13", torch: locked }
          - { os: windows-latest, python: "3.11", torch: locked }
          - { os: windows-latest, python: "3.13", torch: locked }
    steps:
      - uses: actions/checkout@<sha> # vX.Y.Z
        with:
          persist-credentials: false
      - uses: actions/setup-python@<sha> # vX.Y.Z
        with:
          python-version: ${{ matrix.python }}
      - name: Install uv from a hash-pinned file
        run: python -m pip install --require-hashes -r .github/requirements/uv.txt
      - name: Create the locked environment
        run: uv sync --locked --group test
      - name: Override torch for this leg
        if: matrix.torch != 'locked'
        run: uv pip install "torch~=${{ matrix.torch }}.0" --torch-backend=cpu
      - name: Tests
        run: uv run --no-sync pytest --timeout=300
      - name: Two-rank gloo tests
        run: uv run --no-sync pytest tests/distributed -m "distributed" --timeout=600
```

```yaml
# .github/workflows/release.yml. Design proposal, untested.
name: release
on:
  push:
    tags: ["v[0-9]+.[0-9]+.[0-9]+"]

permissions:
  contents: read

jobs:
  build:
    runs-on: ubuntu-latest
    timeout-minutes: 15
    steps:
      - uses: actions/checkout@<sha> # vX.Y.Z
        with:
          persist-credentials: false
      - uses: actions/setup-python@<sha> # vX.Y.Z
        with:
          python-version: "3.11"
      - run: python -m pip install --require-hashes -r .github/requirements/uv.txt
      - name: Build sdist and wheel
        run: uv build
      - name: Wheel is pure and matches the tag
        run: |
          v="${GITHUB_REF_NAME#v}"
          test -f "dist/tailtune-${v}-py3-none-any.whl"
          test -f "dist/tailtune-${v}.tar.gz"
      - name: Smoke test the built wheel
        run: |
          uv venv "$RUNNER_TEMP/smoke"
          uv pip install --python "$RUNNER_TEMP/smoke" dist/*.whl
          "$RUNNER_TEMP/smoke/bin/tailtune" doctor
      - uses: actions/upload-artifact@<sha> # vX.Y.Z
        with:
          name: dist
          path: dist/
          if-no-files-found: error

  publish:
    needs: build
    runs-on: ubuntu-latest
    timeout-minutes: 10
    environment:
      name: pypi              # required reviewer; deployments limited to v* tags
      url: https://pypi.org/project/tailtune/
    permissions:
      id-token: write         # Trusted Publishing. No other job has this.
    steps:
      - uses: actions/download-artifact@<sha> # vX.Y.Z
        with:
          name: dist
          path: dist/
      - uses: pypa/gh-action-pypi-publish@<sha> # vX.Y.Z
        # Attestations are generated and uploaded by default.
```

`--torch-backend=cpu` follows the uv guide's backend selection. Confirm the exact accepted values when building this [U]. A third job can create the GitHub Release with `contents: write` and attach the built files. Whether that also satisfies Scorecard's signed-releases check is covered in section 7.

## 6. Release engineering

### Versioning: 0.x SemVer

| Project | Scheme | Releases in 2026 so far |
|---|---|---|
| Unsloth | CalVer, 2026.11.3 | 107 [S](https://pypi.org/pypi/unsloth/json) |
| LiteLLM | SemVer, 1.108.2 | 274 [S](https://pypi.org/pypi/litellm/json) |
| vLLM | 0.x, 0.33.0 | 30 [S](https://pypi.org/pypi/vllm/json) |
| Axolotl | 0.x, 0.21.1 | 12 [S](https://pypi.org/pypi/axolotl/json) |

CalVer suits Unsloth's cadence, where a version signals freshness. It loses the API-contract signal that integrators and sponsors value [J]. tailtune uses `0.MINOR.PATCH`:

- **PATCH:** bug fixes, cap widening, docs. Never breaks anything.
- **MINOR:** features. Breaks something only after the deprecation period.
- **1.0:** at Gate 5, when the config schema is frozen.

The README states the cadence as one minor release every 4 to 8 weeks during 0.x, patches as needed, and compatibility patches within 7 days of a passing upstream minor [J].

### What counts as public API

1. Names in `tailtune.__all__` and the documented Python surface (`tt.read`, `tt.train`, `tt.wrap` and the returned objects).
2. CLI commands, flags and exit codes 0 to 5, and 130.
3. Run-directory file names and the schemas of `tailtune.yaml`, `mapping.yaml`, `data_report.json`, `targets.json`, `lineage.json` and `forgetting_report.json`. Each JSON and YAML file carries a `schema_version`.
4. Nothing else. Underscore modules are private.

### Deprecation policy [J]

The research's suggestion was a warning for at least two minor releases, a config-schema version field and a migration note [J]. Concretely:

- `TailtuneDeprecationWarning` subclasses `FutureWarning`, so beginners see it by default. The message names the removal version and the replacement.
- Old config keys are accepted with a warning and rewritten. The resolved `tailtune.yaml` in the run directory always uses the new keys, so users can copy the migration.
- Readers of report JSON accept the previous `schema_version` for two minor releases.
- Every deprecation appears under "Deprecated" in the changelog, with a migration snippet.

### Changelog

The research did not compare changelog tools such as towncrier, git-cliff or release-please, so there is no evidence-based pick. Start with a hand-written `CHANGELOG.md` [J]:

- An "Unreleased" section that every user-visible PR edits. The PR template has a checkbox for it.
- Per release: Added, Changed, Deprecated, Removed, Fixed and Security.
- **Compatibility:** the support-table row.
- **Claims:** any model and mode label that moved between "runs" and "retention benchmarked" (D-02).

Revisit tooling when a second maintainer joins.

### Release checklist

1. `compat-weekly` is green on the intended ranges. Note the state of `transformers-main`.
2. Open the release PR: version, changelog, support-table row, and a claims-label review against D-02.
3. All PR jobs are green.
4. Dispatch `release-gate`, `release-blackwell`, `kaggle-t4` and `benchmark` on the PR head, and link the run URLs in the PR. Benchmark thresholds pass, or the changelog explains why not.
5. Merge, tag on `main` and push the tag.
6. Read the `build` log, then approve the `pypi` deployment.
7. Verify as in section 3, steps 10 to 13, with the new version.
8. Publish the GitHub Release with the changelog, built files and `constraints-<version>.txt`. Deploy the docs.
9. If the release is broken, yank it on PyPI rather than deleting it, and ship a patch.
10. Announce notable releases following [12](12-community-and-sponsorship.md).

## 7. Security

### The case that sets the priority

On 2026-03-24, an attacker used a compromised LiteLLM maintainer's PyPI account to publish malicious versions 1.82.7 and 1.82.8. The chain began with the TeamPCP compromise of Trivy, an open-source security scanner [S](https://www.netspi.com/blog/executive-blog/ai-ml-pentesting/litellm-supply-chain-compromise/). The lesson is that a long-lived credential sat within reach of a third-party tool. tailtune's mitigations:

| Risk | Mitigation |
|---|---|
| Stolen long-lived PyPI token | No tokens at all. Trusted Publishing only (D-08) |
| A CI tool reads credentials | `id-token: write` exists only in the `publish` job. Scanners (CodeQL, Scorecard, anything added later) run in separate workflows with no secrets and no `id-token`. Default `permissions: contents: read` |
| A mutable action tag is moved | Every action pinned by full SHA |
| Workflow tampering | CODEOWNERS on `.github/`. Branch protection on `main`. No `pull_request_target` with PR checkout. No `${{ github.event.* }}` interpolated into `run:` |
| Malicious tag or deployment | Tag ruleset, required reviewer on `pypi`, deployments limited to tags |
| Account takeover | 2FA on GitHub and PyPI, with a hardware key preferred [J] |
| Dependency compromise | `uv.lock` committed. Lock upgrades only through reviewed PRs, never auto-merged [J] |
| Install-time code | Pure wheel, no build hooks, no post-install steps |

### OpenSSF Scorecard: 7+ by Gate 5

Peer scores from the Scorecard API: Transformers 6.5 (2026-11-02) [S](https://api.securityscorecards.dev/projects/github.com/huggingface/transformers), LiteLLM 5.7 (2026-11-07) [S](https://api.securityscorecards.dev/projects/github.com/BerriAI/litellm), Unsloth 4.9 (2026-09-01) [S](https://api.securityscorecards.dev/projects/github.com/unslothai/unsloth). No entry came back for Axolotl, vLLM, Ollama, LlamaFactory, llama.cpp, uv or TRL. A score above all three peers is a concrete, sponsor-facing number.

The research suggested 7+ is reachable "from day one" on a new repo. That is probably optimistic. A young repo with one maintainer is structurally limited on Code-Review, Contributors and Maintained (see the table) [J]. The target is therefore 7+ at Gate 5, matching [13](13-roadmap.md), with every check the owner controls passed by Gate 2.

The check list and scoring below come from Scorecard's own documentation and were **not verified in the research** [U]. At Gate 0, run the Scorecard action and read its per-check output, or query the API endpoint pattern above. Correct this table from that output.

| Check | How tailtune passes | Solo limit? |
|---|---|---|
| Binary-Artifacts | No binaries in git. If `.xlsx` fixtures are flagged, generate them at test time instead | No |
| Branch-Protection | Protect `main`: PRs required, status checks required, no force push, no deletion | Required approvals need a second maintainer |
| CI-Tests | Every change merges through a PR with CI | No |
| CII-Best-Practices | Earn the OpenSSF Best Practices "passing" badge | No |
| Code-Review | Changes reviewed before merge | **Yes**, until a second maintainer reviews |
| Contributors | Contributors from several organisations | **Yes**, grows with the community |
| Dangerous-Workflow | No `pull_request_target` with checkout, no script injection | No |
| Dependency-Update-Tool | `.github/dependabot.yml` | No |
| Fuzzing | Hypothesis property tests may count for Python [U] | No |
| License | `LICENSE` | No |
| Maintained | Recent commits and issue activity | **Yes** for roughly the first 90 days of the repo [U] |
| Packaging | The `pypa/gh-action-pypi-publish` workflow | No |
| Pinned-Dependencies | SHA-pinned actions, `--require-hashes` installs | No |
| SAST | CodeQL on every PR | No |
| Security-Policy | `SECURITY.md` | No |
| Signed-Releases | Signatures or provenance attached to GitHub Releases. Whether PyPI attestations count is unverified [U]. If they do not, attach the files' signature bundles to each GitHub Release | No |
| Token-Permissions | Top-level read-only, write per job | No |
| Vulnerabilities | No open advisories against dependencies. Keep the lock fresh | No |

Publish the badge from the first scored run and record the baseline in the changelog. A low score at Gate 0 is fine if the trend is visible [J].

### `SECURITY.md`

The file should contain [J]:

- **Supported versions:** the latest minor release during 0.x.
- **Reporting:** GitHub private vulnerability reporting (turn it on in the repo settings), with an email fallback. Never a public issue.
- **Response targets:** acknowledge within 7 days, assess within 14, and fix or mitigate high-severity issues within 30. Disclose in coordination, within 90 days at most.
- **In scope:**
  - the release pipeline
  - code execution through model, dataset or config loading
  - crafted spreadsheet or document input
  - escape from the evaluation sandbox
  - path traversal in run directories
  - the GGUF conversion environment
- **Out of scope:** upstream bugs (report them upstream, and tailtune tracks them) and model behaviour, about which tailtune makes no safety claim (D-02).
- **Process:** advisories through GitHub Security Advisories, with credit to reporters.
- **Defaults:** the security defaults below, listed in the file.

### Model loading

- **Safetensors only (D-08).** Load with `use_safetensors=True`. Refuse `.bin`, `.pt`, `.pth` and `.ckpt` checkpoints with a message that explains why and how to convert them in a separate environment. There is no escape hatch at v0.1 [J]. Bases caches and run directories hold safetensors, Parquet, JSON and YAML only. The Ruff `banned-api` rules enforce this in tailtune's own code.
- **`trust_remote_code=False` by default.** To enable it, the user passes a flag that names the repository and a full commit hash, never a branch. The value is recorded in `lineage.json`. Under `--yes` it is refused unless the config file also sets it [J].

The research read no CVEs or incidents for pickle or `trust_remote_code`, so this policy rests on reasoning, not cited cases.

### Sandboxing evaluated code

HumanEval runs code the model wrote. The report requires code execution to be sandboxed or marked "not run". The rules [J], with full detail in [09](09-evaluation.md):

- Never execute in the training process.
- Use one subprocess per sample with a timeout, a memory cap, no network, a temporary directory that is deleted afterwards, and an environment stripped of tokens and cloud credentials.
- Use container or namespace isolation where the platform offers it.
- Where no isolation is available, mark the benchmark "not run" rather than run it unprotected.

### Dependency updates

Unsloth, Axolotl and vLLM all keep `.github/dependabot.yml` [S](https://github.com/unslothai/unsloth/blob/main/.github/dependabot.yml), [S](https://github.com/axolotl-ai-cloud/axolotl/blob/main/.github/dependabot.yml), [S](https://github.com/vllm-project/vllm/blob/main/.github/dependabot.yml). tailtune uses Dependabot as follows:

- Weekly updates for `github-actions`, which keep the SHA pins current. Confirm that it rewrites SHA pins together with their version comments [U].
- Weekly grouped Python updates. Whether Dependabot updates `uv.lock` directly is unverified [U]. If it does not, `compat-weekly` refreshes the lock instead.
- `transformers`, `trl` and `peft` are excluded from Dependabot. The compatibility job owns them, so their caps move only on a green test run.

### Not covered by the research

These topics were not researched, so the plan makes no evidence-based claim about them:

- SBOM practice
- Renovate versus Dependabot
- pickle and `trust_remote_code` incidents
- GPU CI services beyond those named in section 5

## 8. Telemetry

**There is no telemetry at launch (D-08).** tailtune sends nothing about its users. Its only network calls are Hub downloads and package installs the user asked for, and a docs page lists each one.

The reasons:

- A solo project has no credibility buffer, and tailtune's product is trust in its numbers.
- The one sourced design is Axolotl's. It is on by default, opted out with `AXOLOTL_DO_NOT_TRACK=1` or `DO_NOT_TRACK=1`, uses PostHog, collects system, hardware, runtime, usage and sanitised error data, and pauses startup for 10 seconds to show a notice [S](https://raw.githubusercontent.com/axolotl-ai-cloud/axolotl/main/docs/telemetry.qmd). It is a known path, but not one to copy before trust exists.
- A review notes that Unsloth Desktop's no-telemetry claim has no independent audit [S](https://www.implicator.ai/unsloth-desktop-local-ai-training-mac-windows-linux/). So "no telemetry" is only worth as much as the code that proves it.

**What the research did not find:** any evidence about community reaction to projects that added telemetry. The question "what happened to projects that added it" is unanswered.

**Revisit only if all of these hold [J]:**

1. A concrete question cannot be answered from public signals: PyPI downloads, GitHub dependents, Hub tags, forgetting reports in model cards, or voluntary surveys.
2. A second maintainer exists, and the change goes through an RFC with at least a 30-day comment period ([12](12-community-and-sponsorship.md)).
3. It is opt-in, never opt-out. Every field is documented, the schema is public, a command shows exactly what would be sent, and aggregates are published.

## 9. Docs tooling

### Zensical with a Material fallback (D-09)

| Fact | Source |
|---|---|
| Material for MkDocs feature development has ended, and "critical maintenance continues through **May 5, 2027**" | Official Material blog [S](https://squidfunk.github.io/mkdocs-material/blog/2025/11/05/zensical/) |
| A third-party tracker gives end of life as 2026-11-05 | [releasebot](https://releasebot.io/updates/squidfunk/mkdocs-material). Conflicts with the official blog, which is treated as authoritative. mkdocs-material 9.8.1 shipped on 2026-11-07 [S](https://pypi.org/pypi/mkdocs-material/json), which supports the official date |
| 9.7.0 (November 2025) was described as the last Material release with new features | Secondary [S](https://pydevtools.com/handbook/reference/mkdocs-material/) |
| Zensical launched 2025-11-05, began building Material projects from existing `mkdocs.yml` in October 2026, and reached 0.1 on 2026-11-05 | [S](https://squidfunk.github.io/mkdocs-material/blog/2025/11/05/zensical/), [S](https://pypi.org/pypi/zensical/json) |

Peers use no single standard. By config file, Axolotl uses Quarto [S](https://github.com/axolotl-ai-cloud/axolotl/blob/main/_quarto.yml), uv uses MkDocs [S](https://github.com/astral-sh/uv/blob/main/mkdocs.yml) and LlamaFactory uses Sphinx [S](https://github.com/hiyouga/LlamaFactory/blob/main/docs/conf.py). TRL's `_toctree.yml` suggests Hugging Face's doc builder, inferred from the file name [S](https://github.com/huggingface/trl/blob/main/docs/source/_toctree.yml).

How to keep the fallback real [J]:

1. Keep `mkdocs.yml` conservative: the theme's built-in features, admonitions, code highlighting, search and an explicit `nav`. Add no third-party plugins until each one is confirmed to work on both builders. Any plugin for generating API reference has unknown Zensical compatibility [U].
2. The `docs` job builds with Zensical **and** with pinned Material on every docs PR, both in strict mode. Switching builders is then a one-line deploy change.
3. Re-evaluate at each Zensical minor release and before 2027-05-05.

### `llms.txt` and machine-readable docs

Adoption is mixed. `llms.txt` returned 200 on [Unsloth](https://docs.unsloth.ai/llms.txt), [uv](https://docs.astral.sh/uv/llms.txt), [Ollama](https://docs.ollama.com/llms.txt) and [LiteLLM](https://docs.litellm.ai/llms.txt), and 404 on [vLLM](https://docs.vllm.ai/llms.txt) and [Axolotl](https://docs.axolotl.ai/llms.txt). That makes it cheap differentiation rather than table stakes.

Generate `llms.txt` at build time from `nav`, with each page's title, one-line summary and link to its Markdown source. The `docs` job fails if it is missing.

The research also supports a position on docs for LLM-based coding tools. Axolotl's README has a section on CLI commands that print its docs for such tools [S](https://github.com/axolotl-ai-cloud/axolotl), and TRL runs a workflow named `sync-huggingface-skills` [S](https://api.github.com/repos/huggingface/trl/contents/.github/workflows). The plan [J]:

- Ship `llms.txt` at v0.1.
- Add a `tailtune docs <topic>` command that prints the installed version's docs as plain text by v0.3. Its strength is that it always matches the installed version.
- Defer anything heavier.

### Notebook testing

The main failure is that Colab's base image changes under a notebook. papermill is the runner: version 2.7.0 shipped 2026-02-27, while nbmake's last release was 1.5.5 on 2024-12-23 [S](https://pypi.org/pypi/papermill/json), [S](https://pypi.org/pypi/nbmake/json). papermill's parameter cells also let one notebook run in two modes:

| Layer | Where | What it shows |
|---|---|---|
| Tiny model on CPU, with `model`, `max_steps` and `data_path` parameters | `notebooks-cpu`, every PR and nightly | The code path and the notebook's own cells still work |
| Package resolution against Colab's snapshot | `colab-resolve`, daily | A Colab user's `pip install` will not break |
| Real run on a T4 | `kaggle-t4`, weekly | The fp16 path, memory and time figures in the notebook table |

Unsloth runs a similar pair: a daily `notebooks-ci` job that diffs a "Colab oracle" against committed snapshots, and real Kaggle T4 runs [S](https://raw.githubusercontent.com/unslothai/unsloth/main/.github/workflows/notebooks-ci.yml). Notebooks are generated from one template ([04](04-api-and-ux.md)).

## 10. README standard

### What a first screen contains

| Element | Seen in | tailtune rule |
|---|---|---|
| One-line promise | Unsloth, Axolotl, vLLM ("Easy, fast, and cheap LLM serving for everyone") [S](https://github.com/vllm-project/vllm) | D-01 tagline, unchanged |
| One numeric claim | Unsloth: "2x faster with 70% less VRAM with no accuracy loss" [S](https://github.com/unslothai/unsloth). A review notes those figures are Unsloth's own, measured on older library versions [S](https://www.implicator.ai/unsloth-desktop-local-ai-training-mac-windows-linux/) | One number, with hardware, seeds and a linked script that reproduces it. Paired interval wording per D-02 |
| Install and run in very few commands | vLLM: two commands. LlamaFactory: three CLI commands for train, chat and export [S](https://github.com/hiyouga/LlamaFactory) | Install, train, run |
| Model and notebook table | Unsloth: 12 rows of Model, Free Notebooks, Performance and Memory use | Model, notebook, method, peak VRAM, time, label |
| Day-N table | LlamaFactory: "Day 0" and "Day 1" model lists | Measured dates only, from Gate 4 |
| Dated updates | Axolotl: "Latest Updates" back to 2025/01 [S](https://github.com/axolotl-ai-cloud/axolotl) | Three most recent, linked |
| "Open in" badges | LlamaFactory: Colab, DSW, Spaces, ModelScope, Novita | Colab and Kaggle first |
| Badges | Optional: Unsloth uses none, Axolotl many | Only badges that report something real: PyPI, Python, licence, CI, Scorecard |

Runnable notebooks and dated news are what make a first screen work, more than badges.

### Draft skeleton at v0.2

Every figure in angle brackets is a placeholder. Nothing goes in until the `benchmark` job or the preregistered benchmark produces it. Today's README replacement is in [01](01-product-strategy.md), not here.

````markdown
<h1 align="center">tailtune</h1>
<p align="center"><b>Fine-tune LLMs with far less forgetting, and prove it.</b></p>
<p align="center">[PyPI] [Python 3.11+] [Apache-2.0] [CI] [OpenSSF Scorecard] [Docs]</p>

Fine-tuning teaches a model something new and can quietly damage what it already
knew. tailtune trains with the strongest protection your hardware allows and hands
back a running model with a before-and-after report on every run.

**Measured, not claimed.** On <MODEL> fine-tuned for <DOMAIN>, standard fine-tuning
lowered the worst retention benchmark by <X.X> points. With <METHOD> the change was
<Y.Y> ± <Z.Z> (paired 95% interval, <N> seeds, <GPU>).
[Reproduce it](<link to script>) · [All results, including where tailtune loses](<link>)

## Quickstart

    uv pip install tailtune --torch-backend=auto
    tailtune train faq.xlsx --model Qwen/Qwen3-0.6B
    tailtune run run-001

Already fine-tuned with another tool? `tailtune eval base/ finetuned/`

| Model | Notebook | Method on this GPU | Peak VRAM | Time | Status |
|---|---|---|---|---|---|
| Qwen3-0.6B | [Colab] [Kaggle] | sst | <GB> | <min> | retention benchmarked |
| Qwen3-8B | [Colab] | lora, 4-bit base | <GB> | <min> | runs |

## Latest updates
- <YYYY-MM-DD> <one line, linked>

## What you get
Data in · Training that fits · Forgetting report · GGUF and Ollama out

## Releases and support
Cadence, criteria, and the torch and Transformers support table: <link>

## Contributing · Sponsors · Citation · License
Sponsors are listed with what they fund. Any benchmark featuring a sponsor's
hardware or model says so beside the result.
````

## Implementation checklist

**Gate 0, Claim**

- [ ] Decide whether the repo moves to a `tailtune` org, and do it before the publisher is configured.
- [ ] Turn on 2FA for PyPI and GitHub. Create no PyPI token.
- [ ] Add `pyproject.toml` (0.0.1 variant), `uv.lock`, `src/` layout, Ruff, pyright.
- [ ] Add `NOTICE`, `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, `SECURITY.md`, `GOVERNANCE.md`, `MAINTAINERS.md`, `SUPPORT.md`, `CHANGELOG.md`, `CITATION.cff` and `.github/CODEOWNERS`.
- [ ] Build `tailtune doctor` and `tailtune estimate` with unit tests.
- [ ] Add `ci.yml` (lint, test-cpu), `release.yml`, `codeql`, `scorecard` and `dependabot.yml`, all pinned by SHA.
- [ ] Create the `pypi` environment with a reviewer, a tag restriction and a tag ruleset. Add the pending publisher.
- [ ] Publish 0.0.1 and verify it with steps 10 to 15.
- [ ] Claim the namespaces in order. Run a basic trademark database search.
- [ ] Turn on private vulnerability reporting. Run Scorecard and record the per-check baseline.
- [ ] Check whether the Ecosystem layout accepts `LICENSE`.
- [ ] Confirm whether FSDP2 over gloo works on Windows.

**Gate 1, Verify**

- [ ] Add the measured peak memory as fixtures for the `estimate` regression threshold.
- [ ] Set real tolerances for I3 and B2 from measured numerics.

**Gate 2, v0.1 Door**

- [ ] Full dependency list with floors from the lock. Support-table row for 0.1.0.
- [ ] Tiny models for `qwen3`, `llama`, `granite`, `gemma4`, synthetic MoE and Conv1D, both in-process and on `tailtune-testing`.
- [ ] I1 to I7 and B1 to B5 passing on `single` and `ddp`. Data fixture corpus with golden reports.
- [ ] Logit round trip for merged safetensors and GGUF.
- [ ] Add `distributed-cpu`, `notebooks-cpu`, `colab-resolve`, `transformers-main`, `compat-weekly`, `kaggle-t4`, `gpu-nightly`, `gpu-dispatch` and `benchmark`.
- [ ] Read Kaggle's terms.
- [ ] Docs on Zensical and Material in CI, plus `llms.txt`.
- [ ] Every Scorecard check the owner controls passes.

**Gate 3, v0.2 Proof**

- [ ] I1 to I7 on `fsdp2`. Add `release-gate` and `release-blackwell`.
- [ ] README rebuilt from the v0.2 skeleton with real numbers only.
- [ ] Tech report entry in `CITATION.cff`.

**Gate 4, v0.3 Reach**

- [ ] Tiny models for `qwen3_5` and `gpt_oss`.
- [ ] Day-N table with measured dates. `tailtune docs` command.

**Gate 5, v1.0**

- [ ] Config schema frozen. Deprecation policy applied to the 1.0 break.
- [ ] Second maintainer reviews, "prevent self-review" turned on, Scorecard 7+.
- [ ] Re-check Zensical against the 2027-05-05 Material date.

## Acceptance tests

1. `https://pypi.org/pypi/tailtune/json` returns 200, and both files of every release carry an attestation.
2. The built wheel's file name ends in `py3-none-any.whl`.
3. In clean environments on Linux, macOS and Windows, both `pip install tailtune` and `uv pip install tailtune --torch-backend=auto` give a working `tailtune doctor`.
4. On Colab, `pip install tailtune` keeps the preinstalled torch.
5. I1 to I7 pass on every implemented backend. B1 to B5 pass on three OSes.
6. The data fixture corpus matches its golden reports, and blocking cases exit with code 2.
7. The merged-safetensors logit round trip is within tolerance. The f16 GGUF gives 100% top-1 agreement.
8. Every workflow file pins every action by full SHA. Only the `publish` job has `id-token: write`. No workflow uses `pull_request_target`.
9. Ruff blocks `torch.load`, `pickle` and `yaml.load` in `src/`.
10. Scorecard reaches 7+ by Gate 5, with the per-check output recorded.
11. No release ships with an unexplained regression past the benchmark thresholds.
12. Zensical and Material both build the docs strictly, and `llms.txt` is served.

## Open questions

1. Does the PyTorch Ecosystem working group accept `LICENSE` without `.md`, and does it expect the Linux Foundation code of conduct specifically?
2. Does Modal's free credit cover multi-GPU apps? If not, each release gate costs about $10 in cash [C].
3. What do Kaggle's terms say about automated kernels, and what is the real weekly quota and run-time limit?
4. Does FSDP2 over gloo work on Windows runners?
5. Does PyPI attestation count toward Scorecard's signed-releases check, and how does a young solo repo score on the Maintained check?
6. Does Dependabot update `uv.lock` and SHA-pinned actions with their comments?
7. Will wheel-variant standards for GPU wheels change the install advice?
8. Is a hand-written changelog sustainable at the planned cadence? Changelog tools were not compared.
9. Trademark clearance, fees and timing were not researched.
10. SBOM practice, Renovate versus Dependabot, and pickle or `trust_remote_code` incidents were not researched.
