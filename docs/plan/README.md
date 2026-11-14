# tailtune implementation plan

This folder is the build plan for tailtune: what to build, in what order, on which hardware and models, and how to fund it. It is written for the project owner and for contributors. Only `tailtune doctor` and `tailtune estimate` exist so far, so everything else here is a design until a gate in the [roadmap](13-roadmap.md) proves it.

Background research: a full research report and its source notes, kept outside the repository for now. Facts in this plan were checked on 2026-11-07 and 2026-11-08. This field moves weekly, so re-check versions, prices and star counts before relying on them.

Where the research report and this plan disagree, the plan and its [decision changes](#decision-changes) win. Three known cases:

- SST at r=0.99 on 8B needs 186 GiB, not the report's 173, once tail-form bases and `W_ref` are counted ([07](07-hardware.md)).
- Modal's $30 credit buys about 37 L4-hours, not the report's 10 ([11](11-engineering-and-release.md)).
- Unsloth's YC S24 batch is unverified [U], although the report implies it is confirmed.

The memory budgets in [07](07-hardware.md) are reproduced exactly by `tailtune estimate`, and its golden tests pin them.

> **Plan status: complete draft, 2026-11-08.** All fourteen documents exist. Nothing has been measured on a GPU yet, so every figure marked [C] waits on Gate 1. The detail documents are long by design. Read this index and [01](01-product-strategy.md) first, then [13](13-roadmap.md), then whichever area you are building.

## The plan on one page

**The goal.** Make tailtune the go-to repository for fine-tuning: start from whatever data you have, finish with a running model, and need no ML expertise.

**The finding that shapes everything.** Anti-forgetting training alone cannot make tailtune the go-to tool, for three reasons:

1. **It is already available.** OSF ships in PEFT and in Red Hat's Training Hub. Unsloth could add any published method quickly.
2. **It does not fit on free GPUs.** SST on an 8B model holds about 131 GB of fp32 weights, gradients and Adam state before any bases [C]. Unsloth quotes 6 GB for the same model with QLoRA [S]. The models that do fit free GPUs are the ones where SST's published protection is weakest.
3. **The evidence is young.** SST is one preprint, about three weeks old at the time of writing, with no independent replication found. "Without forgetting" is not defensible: even under SST, a 1.7B model loses 8.0 points [S].

**The opening.** The research found no fine-tuning framework that reports retention by default. Nearly nobody who fine-tunes a model finds out what it lost. So tailtune's product is this bundle:

| What | Why it wins |
|---|---|
| **Data in:** a spreadsheet read as a real table, with a blocking validation report | Unsloth's training upload rejects xlsx. Hugging Face `datasets` has no spreadsheet loader. Only Oumi reads xlsx as a table, and it merges all sheets |
| **Training:** LoRA on small GPUs, SST and OSF where they fit, picked honestly | The tool says what your hardware can do before loading any weights |
| **Proof out:** a before-and-after forgetting report on every run, and for any model from any trainer | No trainer read in the research ships this, and it works for Unsloth users too |
| **Model out:** merged safetensors, GGUF and Ollama | Matches what beginners expect |

**The route to "go-to".** One developer cannot match Unsloth's kernels, day-zero model fixes or 467 notebooks in the short term, so the plan takes three stages:

| Stage | Versions | Target position | Entry condition |
|---|---|---|---|
| Wedge | 0.0.1 to v0.3 | The reference everyone cites for what fine-tuning breaks | Name claimed |
| Default | v1.0 | The default for anyone who cares whether their model still works | Published benchmark, second maintainer |
| Breadth | after v1.0 | The go-to repository for fine-tuning generally | Sustained compute and a funded or staffed team |

The stages and their entry conditions are a judgement call [J]. The breadth race is deferred, not dropped.

**The first move.** Do not start with framework code. Publish a useful 0.0.1 so the name is safe, correct the README, and run five cheap measurements that could reorder this plan ([Gates 0 and 1](13-roadmap.md)).

## Reading order

| # | Document | Answers |
|---|---|---|
| 01 | [Product strategy](01-product-strategy.md) | What are we competing on, and what do we leave alone? What should the README say? |
| 02 | [Architecture](02-architecture.md) | What is the language, stack and package layout? How do the modules fit together? |
| 03 | [Methods](03-methods.md) | How do SST, OSF and LoRA work, exactly enough to implement and test them? |
| 04 | [API and UX](04-api-and-ux.md) | What do users type? What do the CLI, Python API, config, errors and notebooks look like? |
| 05 | [Data ingestion](05-data-ingestion.md) | How does Excel, a Hugging Face dataset or anything else become training data safely? |
| 06 | [Models](06-models.md) | Which models are supported, when, and which architecture quirks break naive code? |
| 07 | [Hardware](07-hardware.md) | Which GPUs and platforms are supported, and how much memory does each route need? |
| 08 | [Distributed training](08-distributed.md) | How do we scale past one GPU, and in what order? |
| 09 | [Evaluation](09-evaluation.md) | How is forgetting measured credibly, and what is the public benchmark? |
| 10 | [Export and deployment](10-export-and-deployment.md) | How do trained models reach GGUF, Ollama and vLLM? |
| 11 | [Engineering and release](11-engineering-and-release.md) | How do we handle tooling, tests, CI, security and releases? |
| 12 | [Community and sponsorship](12-community-and-sponsorship.md) | How does the project grow, who governs it, and who pays? |
| 13 | [Roadmap](13-roadmap.md) | What work comes in what order, and what proves each gate? |
| 14 | [Risks and open questions](14-risks-and-open-questions.md) | What could invalidate this plan, and what remains unknown? |

## Evidence labels

Decisive numbers and claims in this plan carry one of these labels.

| Label | Meaning |
|---|---|
| **[S]** | Sourced. A primary source is linked nearby, or in the research report for this page's summary |
| **[C]** | Computed by us from sourced inputs. Not measured on real hardware |
| **[M]** | Measured by us in a small local test. The test's scope is stated |
| **[J]** | Our judgement. Reasonable people could choose differently |
| **[U]** | Unverified. Could not be confirmed, so do not rely on it |

No GPU training run or GPU memory measurement has been performed yet. Every memory budget in this plan is [C] until Gate 1.

## Locked decisions

Every document follows these. A decision changes only when a gate result or an owner decision overturns it, and the change is recorded here first.

### Product and claims

| ID | Decision | Detail |
|---|---|---|
| D-01 | The product is forgetting made visible and bounded. SST and OSF are the current remedies, not the identity. Tagline: "Fine-tune LLMs with far less forgetting, and prove it." | [01](01-product-strategy.md) |
| D-02 | Claims policy. Never say "no forgetting" or "without forgetting". Report "no regression detected at ±X points". Write "SST (Chavan et al., 2026)" in full on first use, because the terms collide with unrelated work. Each model and mode pair carries a separate "runs" or "retention benchmarked" label. Make no safety claim. Do not quote SST's overhead figures until we have measured them | [01](01-product-strategy.md), [09](09-evaluation.md) |
| D-03 | Do not compete on raw kernel speed, multi-node scale, RL breadth, synthetic data generation or a local chat app | [01](01-product-strategy.md) |

### Stack and licensing

| ID | Decision | Detail |
|---|---|---|
| D-04 | Python 3.11+ on PyTorch, as a pure wheel. No Rust or C++. Triton only for profiled hot ops, with a measured gain, a PyTorch fallback, an environment switch and a numerical-equivalence test | [02](02-architecture.md) |
| D-05 | A thin layer over Transformers v5 (no v4 compatibility layer), TRL, PEFT, Accelerate and `datasets`. Torch floor 2.14, also tested on 2.15. No dependency on Unsloth | [02](02-architecture.md) |
| D-06 | Apache-2.0 with DCO sign-off. Never vendor unlicensed, AGPL or LGPL code. Implement SST from the CC BY 4.0 paper, because its repo has no licence. Use PEFT for OSF. Do not copy Unsloth code unless the file is verified Apache at a pinned commit | [02](02-architecture.md), [11](11-engineering-and-release.md) |
| D-07 | uv, Ruff, pyright, hatchling and a `src/` layout. 0.x SemVer. Functional extras such as `[bnb]`, `[flash]` and `[docs]`, not a CUDA matrix. `[docs]` means document ingestion (PDF, DOCX, HTML). Docs-site tooling lives in a `docs-site` dependency group. Publish tested dependency ranges and commit `uv.lock`. Run a weekly compatibility job | [11](11-engineering-and-release.md) |
| D-08 | PyPI Trusted Publishing, Actions pinned by SHA, a reviewer gate on the `pypi` environment, safetensors only, `trust_remote_code=False` by default, and no telemetry at launch | [11](11-engineering-and-release.md) |
| D-09 | Docs on Zensical with a conservative `mkdocs.yml`, keeping Material as the fallback. Ship `llms.txt`. Test notebooks with papermill | [11](11-engineering-and-release.md) |

### Training core

| ID | Decision | Detail |
|---|---|---|
| D-10 | v0.1 methods: `lora` (including on a 4-bit base), `sst`, `osf` (the PEFT tuner), and the baselines `full` and `freeze`. `sst` ships with `r=0.99, K=None`. `K` is exposed but marked unverified. `tail-lora` is experimental and arrives by v0.3 | [03](03-methods.md) |
| D-11 | SST and OSF are different algorithms, and results do not transfer between variants. Docs always name the exact variant and its rank | [03](03-methods.md) |
| D-12 | Precision. The tensor the optimizer updates must be fp32 for constrained methods and for the `full` and `freeze` baselines. On ZeRO, a fp32 master behind bf16 module parameters satisfies this. Load with `dtype=torch.float32` explicitly, because Transformers v5 defaults to `dtype="auto"`. Refuse anything else. Compute under bf16 autocast, or fp16 autocast on T4 | [03](03-methods.md) |
| D-13 | Hook placement. Projection runs in `on_optimizer_step`, skips when `optimizer_step_was_skipped` is true, and uses the cumulative form from `W_ref`. Never use `on_pre_optimizer_step` | [03](03-methods.md), [08](08-distributed.md) |
| D-14 | Bases are offline artifacts. Initial SVD runs in fp32 on one CPU path everywhere. It is hashed, cached, keyed by parameter name, and tested by subspace distance. Refresh during training (`K` set) may use GPU SVD on CUDA or ROCm, with the same fp32 input and orthonormality checks. Sharing bases on the Hub waits on licence review | [03](03-methods.md) |
| D-15 | Targets. A role-based resolver with a structural fallback. Treat `[E,m,n]` tensors as E matrices. Freeze routers, conv kernels and `mtp` heads by default. Reject Conv1D targets and quantized trainable targets. `tailtune inspect` is mandatory, and `train` refuses below a coverage floor. Embedding policy is decided by the Gate 1 ablation, and an `sst_paper` preset keeps replication exact | [06](06-models.md) |
| D-16 | Modes. SFT and continued pretraining at launch. Then SDFT, then DPO and GRPO. No PPO | [06](06-models.md) |

### Reach

| ID | Decision | Detail |
|---|---|---|
| D-17 | Hardware tiers, in brief. **Tier 1:** Linux x86_64 NVIDIA Ampere, Ada and Hopper, with Blackwell at release gate. **Tier 1 reduced:** Linux T4 on Colab and Kaggle. **Tier 1 numerics:** CPU on Linux, macOS and Windows. **Tier 2:** Windows native and WSL2 single GPU, AMD Instinct single node, Apple MPS on macOS 15+, Linux aarch64. **Tier 3:** Radeon, Intel Arc, Data Center GPU Max and Gaudi, Rubin, Windows on Arm, MLX, TorchTPU, Ascend. **Unsupported:** Pascal, Volta, Maxwell, `torch_xla`, FSDP1, Windows multi-GPU, macOS 14, Intel Macs, Python 3.10. MPS needs torch 2.15 or later, above the general floor | [07](07-hardware.md) |
| D-18 | The honest free tier. On T4, offer LoRA on a 4-bit base up to about 8B with the report on. Paper-exact constrained tuning reaches only about 0.7B on Colab, or about 1B with 8-bit Adam [C]. Real SST on 8B means renting. On one B200, it fits with bases and `W_ref` held in host memory. On a 96 GB RTX PRO 6000, it fits only with optimizer state in host memory (slow and untested), or with 8-bit Adam. 8-bit Adam is a variant, so its results never carry the SST label (D-11). `tailtune estimate` prints all of this before any weights load | [07](07-hardware.md) |
| D-19 | Scale order. Single GPU and DDP first. Public SST on ZeRO-2 serves as the reference only. FSDP2 is the one default sharded path, with explicit version and `reduce_dtype`, and `use_reentrant=False`. ZeRO-3 comes last, and only on real demand | [08](08-distributed.md) |
| D-20 | Launch models. Public claim: Qwen3-8B, Qwen3-4B, Qwen3-0.6B, Llama-3.1-8B-Instruct, Gemma-4-E4B-it, Granite-4.2-8b. SmolLM2-135M is for CI only. Experimental after FSDP2: Qwen3.8-27B, gpt-oss-20b | [06](06-models.md) |
| D-21 | Data. One pipeline behind every surface: `sniff → read → normalize → detect → map → render → validate → freeze`. Spreadsheets are read as tables, every cell as text. Validation is on by default and blocking. A mapping file is always written, and never required as input | [05](05-data-ingestion.md) |
| D-22 | Evaluation. The report runs by default and accepts any merged model from any trainer. It uses paired statistics, always shows the worst benchmark, and supports user-supplied retention sets. The public benchmark is preregistered and has a learning gate | [09](09-evaluation.md) |
| D-23 | Export priority: merged safetensors, then GGUF in an isolated environment, then vLLM, then Ollama. TGI is not a target. The template hash travels with the model | [10](10-export-and-deployment.md) |

### Project

| ID | Decision | Detail |
|---|---|---|
| D-24 | BDFL with a written `GOVERNANCE.md`. The second maintainer is the key milestone | [12](12-community-and-sponsorship.md) |
| D-25 | Ask for compute before cash. Never sell roadmap control or benchmark placement. Disclose sponsors beside any benchmark featuring them. The training core stays Apache-2.0 | [12](12-community-and-sponsorship.md) |

## Shared vocabulary

All documents use these exact names.

**Modules** (under `src/tailtune/`):

| Module | Job |
|---|---|
| `tailtune.data` | The ingest pipeline and validation report |
| `tailtune.targets` | Role registry, structural fallback, coverage |
| `tailtune.bases` | Offline SVD, hashing, caching |
| `tailtune.methods` | `lora`, `sst`, `osf`, `full`, `freeze`, later `tail-lora`, behind one interface |
| `tailtune.backends` | Read and write weights per training mode: `single`, `ddp`, `fsdp2`, `zero2`, `zero3` |
| `tailtune.hardware` | Hardware probe and memory estimator |
| `tailtune.train` | Orchestration, the `Trainer` and `SFTTrainer` wrapper, run lineage |
| `tailtune.eval` | Base cache, fast report, paired statistics, regression gates |
| `tailtune.export` | Merged safetensors, GGUF, Modelfile |
| `tailtune.cli` | Commands only, with no logic of its own |

**Commands:** `tailtune doctor`, `estimate`, `inspect`, `train`, `eval`, `export`, `run`. `tailtune repair` is planned for v0.3, pending verification of DG-Hard.

**Python surface:**

```python
import tailtune as tt

# Two-call flow
data = tt.read("faq.xlsx", sheet="Q1", user="{question}", assistant="{answer}")
print(data.report())
run = tt.train(data, model="Qwen/Qwen3-0.6B", method="lora")
print(run.retention_report())
run.export("gguf")

# Wrapper for existing training code
trainer = tt.wrap(trainer, method="sst", r=0.99)
trainer.train()
```

**Method selection:** `method="auto"` is the default. It picks the strongest method that fits the estimated budget and prints one line explaining the choice. Its exact rule stays provisional until Gate 1.

**Run directory** (`runs/run-001/`):

| File | Contents |
|---|---|
| `tailtune.yaml` | The fully resolved configuration, including every default that was applied |
| `mapping.yaml` | The column mapping, reusable for later runs |
| `data.parquet` | The frozen training data |
| `data_report.json` | The data validation result |
| `targets.json` | The `inspect` coverage output |
| `lineage.json` | Base commit, method variant string, r, `K`, bases hash, coverage, data hash, template hash, applied and skipped optimizer steps, dependency versions, previous run |
| `FORGETTING_REPORT.md`, `forgetting_report.json` | The retention report |
| `model/` | The trained output |

**Exit codes:**

| Code | Meaning |
|---|---|
| 0 | Success |
| 1 | Failed for any other reason: an internal error, an export or conversion failure, out of memory during training, or a declined confirmation |
| 2 | Blocking data validation failure |
| 3 | Regression gate failed |
| 4 | Refused by a pre-flight check: estimated not to fit, coverage below the floor, a low-precision or quantized trainable target, an unsupported target or backend, a chat-template mismatch at training start, `run` or `eval`, or a model that cannot be loaded under policy (gated without access, missing token, remote code required, pickle-only weights). A template mismatch found by export's own self-check exits 1, since the artifact is wrong rather than the input refused |

| 5 | Invalid configuration or usage: a bad argument value, an unknown key, a command-line parse error, mismatched `eval` inputs, or no terminal attached without `--yes` |

`tailtune doctor` exits 0 even when it finds problems, since its job is to report them. With `--strict`, any failed check exits 4. Ctrl-C exits 130, the shell convention.

**Provisional defaults.** The detail documents set these judgement defaults [J]. Gate 1 or Gate 3 results may change them.

| Default | Value | Set in |
|---|---|---|
| Coverage floor | 0.90 of trainable parameters in targeted roles | [06](06-models.md) |
| Minimum free tail | 4 directions. Any matrix with fewer is frozen | [06](06-models.md) |
| Norms and scalar parameters | Frozen by default. The `sst_paper` preset trains them, as the paper does | [06](06-models.md) |
| Non-inferiority margin | 2 points per benchmark | [09](09-evaluation.md) |
| First benchmark release | Qwen3-8B and Gemma-4-E4B-it, with Granite-4.2-8b as fallback. Medical, coding and text-to-SQL. Thinking mode off | [09](09-evaluation.md) |

The paper's headline results are in thinking mode, so the Gate 1 replication cell uses thinking mode even though the benchmark default does not.

**Gates:** Gate 0 Claim, Gate 1 Verify, Gate 2 v0.1 Door, Gate 3 v0.2 Proof, Gate 4 v0.3 Reach, Gate 5 v1.0.

## Decision changes

| Date | Change | Reason |
|---|---|---|
| 2026-11-08 | D-14: GPU SVD is allowed for refresh. Initial bases stay CPU | A CPU-only rule made refresh impractical during training |
| 2026-11-08 | Exit code 4 now covers every pre-flight refusal | No code covered a precision or target refusal |
| 2026-11-08 | SDFT is scheduled at Gate 4, ahead of DPO and GRPO at Gate 5, as D-16 orders | D-16 named SDFT, but the roadmap had not scheduled it |
| 2026-11-08 | D-18 now gives the exact conditions for SST on a B200, a 96 GB card and a T4 | The budgets had left out `W_ref`, which the cumulative form in D-13 requires |
| 2026-11-08 | D-17 places Windows on Arm, Data Center GPU Max and Intel Macs, and sets the MPS torch floor | These platforms were missing |
| 2026-11-08 | D-07 defines `[docs]` as document ingestion | The name had been used for two different things |
| 2026-11-08 | `doctor` exit behaviour defined | No exit code fitted a diagnostic command |
| 2026-11-08 | A chat-template mismatch exits 4 | It is a pre-flight refusal, not an internal error |
| 2026-11-08 | Exit code 5 added for invalid configuration or usage | No code covered these errors |
| 2026-11-08 | D-20: Qwen3-0.6B is the claimed small model, and SmolLM2-135M is CI only | The choice had been left open |
| 2026-11-08 | Provisional defaults recorded | They were set in the detail documents but missing from this contract |
| 2026-11-08 | D-12 reworded around the optimizer-updated tensor, and extended to the `full` and `freeze` baselines | The literal wording would have blocked the ZeRO reference path, and the baselines need the same precision to be fair |
| 2026-11-08 | `lineage.json` gains the variant string, `K`, and applied and skipped step counts | These are needed to reproduce a run and to compare variants |
| 2026-11-08 | Exit codes made complete. Model access and weight-safety refusals use 4, a missing terminal uses 5, code 1 is redefined, and Ctrl-C is 130 | Several cases had fallen back to code 1, which meant "internal error" |

## Decisions the owner needs to make

| # | Decision | Recommendation |
|---|---|---|
| 1 | Accept the positioning in D-01 and replace the README's "without forgetting", "Scales" and `pip install` claims | Yes, before any public promotion. Wording is in [01](01-product-strategy.md) |
| 2 | Approve Gate 1 rental spend for the five measurements, and later about $300 to $700 for the first benchmark [C] | Yes, since these results decide what gets built next |
| 3 | Ask the SST authors about a licence, ZeRO-3, Muon and refresh code | Yes, and early. It costs one message |
| 4 | Stay an individual maintainer, or incorporate | Stay individual until milestone M3 or M4. Note that NVIDIA Inception requires a company ([12](12-community-and-sponsorship.md)) |
