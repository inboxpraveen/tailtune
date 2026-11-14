# 04. API and UX

**What this document decides:**

- The first run is two commands and one confirmation after install: `tailtune train faq.xlsx`, `[Y/n]`, then `tailtune run run-001`. Export to GGUF happens inside `train` by default.
- The primary install is `uv pip install tailtune --torch-backend=auto`. Plain `pip install tailtune` keeps working.
- Eight CLI commands share one implementation with the Python API. Every CLI command calls exactly one public Python function.
- The Python surface is `tt.read`, `tt.plan`, `tt.train`, `tt.wrap`, `tt.evaluate`, `tt.inspect`, `tt.estimate` and `tt.doctor`, plus the `Dataset`, `Plan`, `Run` and `RetentionReport` objects. The README's two-call flow works exactly as written.
- `method="auto"` follows the provisional rule from [07](07-hardware.md), and prints one line that explains its choice.
- Every failure has a stable code (`TT-xxxx`), an exact message, a cause and a fix. Exit codes follow the README contract, 0 to 5, plus 130 for Ctrl-C.
- One override flag spelling for a changed chat template: `--accept-template <sha256>` on `run` and `eval`, recorded in the output. `eval` also takes `--template base|ft` when the two models' templates differ ([09](09-evaluation.md)).
- Notebooks before v1.0 stay few, all generated from one template and all tested. There is no app before v1.0.

Evidence labels are defined in the [plan index](README.md#evidence-labels). Code samples are untested design proposals. Terminal output is illustrative, not measured: no part of tailtune exists yet. Package structure and the error hierarchy are in [02](02-architecture.md).

## 1. Principles from the Unsloth teardown

Unsloth is the default because of a bundle: free notebooks, a two-call API, one-line GGUF export and a large doc site ([01](01-product-strategy.md)). Some parts are worth matching, some can be beaten, and some should not be copied.

| Match | Evidence | tailtune's version |
|---|---|---|
| A two-call Python API | `FastLanguageModel.from_pretrained`, then `get_peft_model` [S](https://unsloth.ai/docs/basics/inference-and-deployment/vllm-guide.md) | `tt.read`, then `tt.train` |
| One-line GGUF export | `save_pretrained_gguf(...)` [S](https://unsloth.ai/docs/basics/inference-and-deployment/saving-to-gguf.md) | GGUF by default inside `train`, plus `run.export("gguf")` |
| An Ollama Modelfile written for you | "we automatically create a `Modelfile`" [S](https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/tutorial-how-to-finetune-llama-3-and-use-in-ollama.md) | Written next to the GGUF, with the training template |
| Free-GPU notebooks that run top to bottom | "Click Run all ... add your dataset, train, and deploy" [S](https://unsloth.ai/docs/get-started/unsloth-notebooks.md) | One tested T4 notebook at Gate 2 (section 9) |
| `{col}` and `[[optional {col}]]` column templates | Unsloth's `to_sharegpt` syntax [S](https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/datasets-guide.md) | Same syntax, reimplemented ([05](05-data-ingestion.md)) |
| `uv pip install ... --torch-backend=auto` | Unsloth's documented install [S](https://github.com/unslothai/unsloth/blob/5f1d55556fd82f889abfe17a446c3b3a97a2f42b/README.md) | The primary install (section 2) |
| `llms.txt` and a Markdown copy of every page | [S](https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/datasets-guide.md) | Both, from day one (D-09) |

| Beat | Evidence of the gap | tailtune's version |
|---|---|---|
| Spreadsheets read as tables | Studio's training upload takes `.csv .json .jsonl .parquet` only, and xlsx is flattened as a document [S](https://github.com/unslothai/unsloth/blob/5f1d55556fd82f889abfe17a446c3b3a97a2f42b/studio/backend/routes/data_recipe/seed.py) | Sheet list, header detection, merged cells, mapping preview |
| Validation that runs by default and blocks | `check_dataset` exists but is opt-in [S](https://github.com/unslothai/unsloth/blob/5f1d55556fd82f889abfe17a446c3b3a97a2f42b/unsloth/chat_templates.py) | Exit code 2 on a blocking check |
| Known silent failures caught | "All labels in your dataset are -100" and template-related gibberish after export are documented failures [S](https://unsloth.ai/docs/basics/troubleshooting-and-faqs.md) | TT-D012, plus a template hash checked at `run` and `eval` (TT-E001) and at export (TT-E003) |
| A retention report | None found in any trainer read | On every run, and for any model through `tailtune eval` |
| Honest hardware fit before loading weights | Not found | `tailtune estimate`, and `method="auto"` explains itself |
| One ingestion path | The legacy CLI sends any `.json` to a raw-text loader that finds no text in a `messages` file, by code reading [S](https://github.com/unslothai/unsloth/blob/5f1d55556fd82f889abfe17a446c3b3a97a2f42b/unsloth-cli.py) | One pipeline behind every surface (D-21) |

| Do not copy | Why |
|---|---|
| Any AGPL or LGPL file, including notebooks | D-06. The notebooks repo is LGPL [S](https://api.github.com/repos/unslothai/notebooks), and the CLI is mostly AGPL [S](https://github.com/unslothai/unsloth/tree/5f1d55556fd82f889abfe17a446c3b3a97a2f42b) |
| The desktop app and the widening scope (chat, RAG, diffusion) | D-03. The desktop pivot brought a new class of install bugs, and 581 bug-labelled issues were created after 2026-08-01 [S](https://github.com/unslothai/unsloth/issues?q=is%3Aissue+label%3Abug+created%3A%3E2026-08-01) |
| Speed headlines without a reproducible benchmark | D-02. No independent reproduction of Unsloth's claims was found |
| Patching modeling code | Upstream breakage is a top issue theme, for example [#1144](https://github.com/unslothai/unsloth/issues/1144) |
| A combinatorial CUDA extras matrix | D-07 |
| A "render your own `text` column first" step | The current Qwen3.8 recipe needs one [S](https://unsloth.ai/docs/models/qwen3.8/train.md). tailtune renders for the user |

## 2. Install

```bash
# Primary: uv picks the PyTorch build that matches this machine's driver
uv pip install tailtune --torch-backend=auto

# With LoRA on a 4-bit base (bitsandbytes)
uv pip install "tailtune[bnb]" --torch-backend=auto

# Plain pip keeps working
pip install tailtune
```

- `--torch-backend=auto` queries the CUDA driver, AMD GPU versions and Intel GPU presence, picks the matching PyTorch index, and falls back to CPU. It exists only in the `uv pip` interface. Projects using `uv sync` need explicit index configuration instead [S](https://docs.astral.sh/uv/guides/integration/pytorch/).
- Plain `pip` installs whatever torch build the default index serves. On Windows that may be a CPU-only build, which is why LlamaFactory tells Windows users to install the GPU build manually [S](https://github.com/hiyouga/LlamaFactory/blob/f6355ab6a426eedb800742ecf61896f7bff95ebb/README.md). `tailtune doctor` reports a CPU-only torch next to a GPU (TT-H004).
- **On Colab, do not reinstall torch.** The preinstalled `2.14.1+cu130` already meets the floor [S](https://github.com/googlecolab/backend-info/blob/main/pip-freeze.gpu.txt). Use `pip install "tailtune[bnb]"` without `--torch-backend`. bitsandbytes needs that install on Colab, per the same freeze list.
- Extras are functional (D-07): `[bnb]`, `[flash]`, `[vllm]`, `[docs]` (document ingestion, not the docs site) and `[wandb]`. The package metadata is in [11](11-engineering-and-release.md).
- Python 3.11 or later.

## 3. The first run

### 3.1 From `faq.xlsx` to a running model

```text
$ uv pip install tailtune --torch-backend=auto
$ tailtune train faq.xlsx
  ... plan shown ...
Start training? [Y/n] y
  ... training, report, export ...
$ tailtune run run-001
```

After install, that is two commands and one confirmation. The confirmation covers everything at once: the sheet, the mapping, the model, the method and the memory estimate.

The output below is **illustrative, not measured**. It shows a Colab T4 with no `--model` or `--method` given. Every number in it is a placeholder.

```text
$ tailtune train faq.xlsx                                        # illustrative output
tailtune 0.1.0                                          run  runs/run-001

Machine   Tesla T4, limit 13.8 GiB, fp16 compute (bf16 is emulated here)    Tier 1, reduced
Model     model=auto -> Qwen/Qwen3-4B @ 1a2b3c4  (largest default model for a 16 GB card)

Reading   faq.xlsx  (xlsx by file signature, 0.4 s)
  Sheets  Q1        1,204 rows x 4 columns    selected: the only sheet that looks like data
          Notes     2 rows x 1 column         skipped: not a table
          Archive   hidden                    skipped: hidden sheet
  Header  row 3 (2 title rows above it skipped)
  Merged  118 cells in "category", filled down

Mapping   confidence 0.90                         saved  runs/run-001/mapping.yaml
  user        {question}      exact header match
  assistant   {answer}        exact header match
  ignored     category (12 distinct values), internal_note

Sample 1 of 3, rendered with the model's chat template
  <|im_start|>user
  How do I reset my password?<|im_end|>
  <|im_start|>assistant
  Go to Settings, then Security, and choose Reset password.<|im_end|>

Checks    1,204 read, 1,201 kept, 3 dropped (2 empty answers, 1 exact duplicate)
          BOS ok   EOS ok   labels ok (41% of tokens trained)   p95 212 tokens, 0 over 2,048
          60 rows held out (5%) for the in-domain score
          details  runs/run-001/data_report.json

Method    method=auto -> lora on a 16-bit base. sst needs about 70 GiB of GPU memory for
          this model; the smallest single card is a 96 GB RTX PRO 6000.
Targets   252 matrices in 7 roles, all adapted                    runs/run-001/targets.json
Estimate  GPU 10.4 GiB (8.8 to 13.0) of 13.8; about 25 min training and 25 min report [computed]

Start training? [Y/n] y

Base report   no cached result for this revision; evaluating Qwen/Qwen3-4B (hf backend)  12 min
Training      225 optimizer steps  24 min  loss 1.92 -> 0.61  (223 applied, 2 skipped by the fp16 scaler)
Merged        runs/run-001/model/  safetensors, base key names
Report        evaluating the fine-tuned model  12 min

Forgetting report                                      runs/run-001/FORGETTING_REPORT.md
  In-domain   held-out answers  +38.5 points vs base                         learned
  Worst       IFEval            -3.1 points (95% interval ±2.7)              Minor
  Others      GSM8K, MMLU-Pro, MGSM, HumanEval, refusal sets:
              no regression detected at ±2.7 to ±4.8 points
  Drift       KL to the base on neutral text: 0.041
  Protection  lora: base weights frozen, adapters merged into model/

Exported      GGUF (q8_0) and Modelfile in runs/run-001/model/   template sha256:3f9a1c... stored

Next          tailtune run run-001
```

```text
$ tailtune run run-001                                           # illustrative output
Template  sha256:3f9a1c... matches the training template
Engine    ollama not found; using transformers on runs/run-001/model/
Type a message. /exit quits.
>>> How do I reset my password?
Go to Settings, then Security, and choose Reset password.
```

**Why this design** [J]:

- The method line is honest. On a T4, `auto` never implies that SST protection is running. It names the strongest route that did not fit and what would hold it (D-18).
- The report summary always names the worst benchmark and uses "no regression detected at ±X points" wording, never "no forgetting" (D-02, D-22).
- GGUF is `q8_0` by default. The converter emits f16, bf16 or q8_0 directly [S](https://github.com/ggml-org/llama.cpp/blob/master/convert_hf_to_gguf.py), while 4-bit types such as Q4_K_M need llama.cpp's separately built quantize tool. That route is owned by [10](10-export-and-deployment.md).
- File names follow the README contract: `mapping.yaml`, `data_report.json`, `targets.json` and `FORGETTING_REPORT.md`.

The research's own illustrative flow differs in three places, and this design does not follow it. It showed Qwen3-4B "fits your 8 GB GPU" with preservation on, which the computed memory table rules out. It used a `preserve="balanced"` argument rather than `method=`. And it wrote `faq.tailtune.yaml` and an HTML report rather than the README names.

### 3.2 The second entry point: check any fine-tune

[01](01-product-strategy.md) names the local tinkerer, who already fine-tunes with another tool, as the most reachable user. Their first run needs no training:

```text
$ tailtune eval Qwen/Qwen3-4B ./my-unsloth-finetune/              # illustrative output
Base       Qwen/Qwen3-4B @ 1a2b3c4      cached result found (0 min)
Candidate  ./my-unsloth-finetune  merged safetensors
Checks     same model ok   tokenizer ok   chat template matches base
Report     evaluating the candidate  12 min                    ./eval-2026-11-08-1012/

  Worst    HumanEval  -14.6 points (95% interval ±4.8)    Major
  Others   IFEval -2.0 (±2.7), GSM8K -0.7 (±3.9): no regression detected at those margins
  Report   ./eval-2026-11-08-1012/FORGETTING_REPORT.md
```

The checks before any GPU work are [09](09-evaluation.md)'s preconditions. Two models that are not a before-and-after pair exit 5 (TT-V001). Different chat templates exit 4 (TT-E004) unless `--template base` or `--template ft` says which one to use for both sides. The docs give this flow equal billing with the spreadsheet flow (section 11).

### 3.3 Steps compared with other tools

These counts come from reading each tool's docs, not from timed runs on clean machines. All counts **include install**, which is how the research counted the other tools. tailtune's row is a target, because nothing is built yet.

| Tool and path | Steps to a trained model | Steps to an exported, running model | Reads xlsx as a table | Source |
|---|---|---|---|---|
| **tailtune CLI (target)** | 3: install, `train`, confirm | 4: adds `tailtune run` | Yes | This document |
| Unsloth Desktop | 6 UI actions | About 10 | No: flattened as a document | [Studio export](https://unsloth.ai/docs/new/studio/export.md) |
| Unsloth notebook | About 6 | About 8 | Not in the code read | [Notebooks](https://unsloth.ai/docs/get-started/unsloth-notebooks.md) |
| LlamaFactory CLI | 7, including `dataset_info.json` registration | 9, plus GGUF conversion outside the tool | No | [README](https://github.com/hiyouga/LlamaFactory/blob/f6355ab6a426eedb800742ecf61896f7bff95ebb/README.md) |
| TRL CLI | 2, Hub data only | About 7: merge, get llama.cpp, convert, quantize, serve | No | [README](https://github.com/huggingface/trl/blob/4ae4d34a86f385511c85fa14caedc2178095d0b8/README.md) |
| Axolotl CLI (for comparison) | 6 | 7, with a llama.cpp checkout | No | [README](https://github.com/axolotl-ai-cloud/axolotl/blob/620c1efca4470c1bc2e4a5f46b4b106341e9dc41/README.md) |

TRL is shorter to a checkpoint. tailtune's target is shortest to a running model from a spreadsheet. A clean-room timing on Windows, Linux and Colab at Gate 2 would replace these counts with measurements.

## 4. CLI reference

### 4.1 Conventions

**Global options:** `--version`, `-h/--help`, `-q/--quiet` (errors and the final summary only), `-v/--verbose` (tracebacks, debug events), `--no-color` (also honours `NO_COLOR`). Where a command supports `--json`, the result goes to stdout as JSON, and progress goes to stderr.

**Run references.** Wherever a command takes `RUN`, it accepts a run id (`run-001`, resolved under the runs directory), a path to a run directory, or `latest`.

**Exit codes** (README contract):

| Code | Meaning |
|---|---|
| 0 | Success. `doctor` also exits 0 when it finds problems, unless `--strict` is passed |
| 1 | Internal or unexpected error, an export or conversion failure, running out of GPU memory (TT-H005), or a declined confirmation |
| 2 | Blocking data validation failure, including an ambiguous mapping or sheet under `--yes` |
| 3 | Regression gate failed. The run's artifacts are complete |
| 4 | Refused by a pre-flight check: estimated not to fit, coverage below the floor, a low-precision or quantized trainable target, an unsupported target or backend, or a chat-template mismatch at training start, `run` or `eval`. Also a model tailtune cannot access or will not load: gated, no token, remote code or pickle-only weights (TT-M001 to TT-M004). Also `doctor --strict` with any failed check |
| 5 | Invalid configuration or usage: a bad argument value, an unknown key, an unparseable command line, mismatched `eval` inputs, or no terminal attached without `--yes` (TT-C002) |
| 130 | Interrupted with Ctrl-C (shell convention, outside the README table) |

Typer and Click exit with 2 on a parse error, which would collide with code 2. `tailtune.cli` remaps every parse error to 5.

**Prompts and `--yes`.** The CLI asks at most one question in the normal case. It asks more only when it genuinely cannot decide.

| Situation | In a terminal | With `--yes` |
|---|---|---|
| Mapping confidence 0.85 or more ([05](05-data-ingestion.md)) | Preview, default Yes | Proceed |
| Mapping confidence 0.60 to 0.85 | Preview, and the user must type `y` | Exit 2 (TT-D003) unless `--mapping` or explicit `--user` and `--assistant` are given |
| Mapping confidence below 0.60, or a tie | Ask the user to pick columns | Exit 2 (TT-D003), listing candidates and the exact flags to pass |
| Several sheets look like data | Ask the user to pick | Exit 2 (TT-D002) unless `--sheet` or `--sheets` is given |
| Blocking data check | Exit 2 | Exit 2 |
| Warnings only | Shown, default Yes | Proceed. Warnings are recorded in `data_report.json` |
| `model=auto` or `method=auto` | Shown in the plan | Proceed. The choice is recorded in `tailtune.yaml` provenance |
| Estimated not to fit | Exit 4 | Exit 4 |
| No terminal attached and no `--yes` | Not applicable | Exit 5 (TT-C002), after printing the plan |

### 4.2 `tailtune doctor`

Reports whether this machine can run tailtune, and at what tier. The checks are specified in [07](07-hardware.md) section 14. It makes no network calls.

```text
tailtune doctor [--deep] [--strict] [--json]
```

| Option | Default | Effect |
|---|---|---|
| `--deep` | off | Adds a 2-rank gloo initialisation, a bf16 rounding demonstration and a short fp16 scaler test |
| `--strict` | off | Any FAIL row exits 4. For CI and setup scripts |
| `--json` | off | Machine-readable results, for bug reports and CI |

**Output:** one row per check (PASS, WARN or FAIL), then a verdict naming the tier and what is promised here. **Exit codes** (README contract): 0 whenever a report is produced, including one with FAIL rows, because the job of `doctor` is to report problems. With `--strict`, 4 if any check failed. 1 on a crash.

### 4.3 `tailtune estimate`

Prints the memory budget per route before any weights load (D-18). It reads the config and safetensors headers only. The formula is in [07](07-hardware.md).

```text
tailtune estimate MODEL [--method M | --route R] [--r 0.99] [--seq-len 1024] [--micro-batch 1]
                        [--nproc 1] [--gpu NAME | --vram GIB] [--host-ram GIB] [--revision REV] [--json]
```

| Argument or option | Default | Effect |
|---|---|---|
| `MODEL` | required | A Hub id or a local directory |
| `--method`, `--route` | all routes | Restrict the output, and set the exit code by its verdict |
| `--r` | 0.99 | Preserved fraction for `sst` routes |
| `--seq-len`, `--micro-batch`, `--nproc` | 1024, 1, 1 | Activation and sharding assumptions |
| `--gpu`, `--vram`, `--host-ram` | this machine | Price a different machine, for example `--gpu "RTX PRO 6000"` |
| `--revision` | `main` | Hub revision, resolved to a commit and printed |

**Output:** a route table with a point estimate, an uncertainty band, host GiB, the verdict here and the smallest single card. Then the `method=auto` line. Every figure is labelled computed. **Exit codes:** 0. 4 when `--method` or `--route` names a route that does not fit. 5 for an unknown route, GPU name, model id or value out of range (TT-C003). 4 for a gated model, a missing token or a checkpoint tailtune will not load (TT-M001 to TT-M004).

### 4.4 `tailtune inspect`

Shows which parameters a method would constrain, train freely or freeze, and the coverage (D-15). The role resolver is in [06](06-models.md).

```text
tailtune inspect MODEL [--method sst] [--r 0.99] [--preset sst_paper] [--coverage-floor 0.90]
                       [--role-action ROLE=ACTION]... [--freeze-unresolved] [--trust-remote-code]
                       [--show summary|groups|all] [--out FILE] [--revision REV] [--json]
```

| Option | Default | Effect |
|---|---|---|
| `--method` | `sst` | Which method's targeting rules to apply |
| `--r` | method default | Changes the free tail size per matrix |
| `--preset` | none | `sst_paper` applies the paper's exact scope, and sets the floor to 0 with coverage printed as a warning |
| `--coverage-floor` | `0.90` | The refusal floor. A changed value is recorded in lineage and the report header |
| `--role-action` | none | Override one row of 06's role policy, for example `norm=free`. Repeatable, and recorded |
| `--freeze-unresolved` | off | Freeze unresolved groups instead of refusing |
| `--trust-remote-code` | off | Allow models that need repository code (D-08). Recorded |
| `--show` | `summary` | `groups` lists each role group. `all` lists every parameter |
| `--out` | none | Writes the `targets.json` format to a file |

**Output:** each group's role, shape, count, free directions at r, and state, followed by the protected, free and frozen shares of trainable parameters, as in [06](06-models.md) section 4.10. Matrices with fewer than `min_free_tail` (4) free directions are frozen and counted. **Exit codes:** 0. 4 when coverage is below the floor, a group is unresolved, a constrained method would protect nothing, or a target is quantized or Conv1D (TT-T001 to TT-T003). 4 for model access errors (TT-M001 to TT-M004). 5 for a bad option value.

### 4.5 `tailtune train`

The full sequence is in [02](02-architecture.md) section 5.

```text
tailtune train [SOURCE] [options]
```

| Option | Default | Effect |
|---|---|---|
| `SOURCE` | required unless `--config` names one | File, directory, Hub dataset id or `-` for JSONL on stdin |
| `--model ID` | `auto` | Hub id or local directory. `auto` follows section 7.2 |
| `--method M` | `auto` | `auto`, `lora`, `sst`, `osf`, `full`, `freeze`, later `tail-lora` |
| `--r FLOAT` | 0.99 for `sst`, 0.5 for `osf` | Preserved fraction. For `osf`, passed to PEFT as `effective_rank` ([03](03-methods.md)) |
| `--config FILE` | none | A `tailtune.yaml`, for example from an earlier run |
| `--mapping FILE` | none | Reuse a `mapping.yaml`. Skips detection |
| `--sheet NAME`, `--sheets A,B` | auto | Sheet choice. `--sheets` combines only on matching headers |
| `--header-row N` | detect | 1-based sheet row |
| `--user T`, `--assistant T`, `--system T`, `--text T` | detect | Column templates, for example `--user "{question}[[ Context: {context}]]"` |
| `--set KEY=VALUE` | none | Any `tailtune.yaml` key, repeatable, for example `--set training.epochs=3` |
| `--preset NAME` | none | `sst_paper` reproduces SST (Chavan et al., 2026) exactly ([06](06-models.md)) |
| `--coverage-floor F`, `--role-action ROLE=ACTION`, `--freeze-unresolved`, `--trust-remote-code` | as `inspect` | Targeting overrides, each recorded in `tailtune.yaml` and `lineage.json` |
| `--from RUN` | none | Chain: start from that run's model ([02](02-architecture.md) section 8.2) |
| `--resume RUN` | none | Continue an interrupted run from its last checkpoint |
| `--runs-dir DIR` | `runs` | Where run directories go |
| `--name NAME` | `run-NNN` | Fixed run name. Exits 5 if it exists (TT-C004) |
| `--yes`, `-y` | off | Skip the confirmation (section 4.1) |
| `--dry-run` | off | Stop after the plan. Writes `mapping.yaml`, `data_report.json` and `targets.json` |
| `--eval SUITE` | `fast` | `fast`, `full` or `none`. `none` is recorded in lineage, and the report says only "Evaluation disabled by the user" ([09](09-evaluation.md)) |
| `--max-regression NAME=POINTS,...` | none | A regression gate, for example `ifeval=3,worst=5`. Comma list or repeatable. Keys follow 09 section 8.3 |
| `--gate-mode MODE` | `estimate` | `estimate`, `confident` or `strict` (09) |
| `--margin POINTS` | `2` | Non-inferiority margin, resolved into `tailtune.yaml` before training (09) |
| `--export LIST` | `merged,gguf` | Comma list, or `none` |
| `--nproc N` | 1 | GPUs. More than 1 uses `ddp` at v0.1 |
| `--backend NAME` | `auto` | `single`, `ddp`, later `fsdp2`, `zero2`, `zero3` ([08](08-distributed.md)) |
| `--accelerate-config FILE` | none | Use this Accelerate config instead of the one tailtune writes into the run directory |
| `--allow-cpu` | off | Train on CPU. Only sensible for tiny smoke-test models |

**Multi-GPU.** With `--nproc` above 1, `train` resolves everything in one process, asks once, writes the Accelerate config beside `tailtune.yaml`, and starts the workers through `accelerate launch` ([08](08-distributed.md) section 10).

**Output:** the run directory (README contract), the plan and the summary in section 3.1. **Exit codes:** 0 to 5 as in section 4.1. If no retention measurement could be produced at all, `train` exits 1 ([09](09-evaluation.md)).

### 4.6 `tailtune eval`

Produces a forgetting report for any merged model from any trainer (D-22). Report contents are in [09](09-evaluation.md).

```text
tailtune eval BASE CANDIDATE [--suite fast|full] [--full] [--cap-thinking] [--retention-set NAME=PATH]...
                             [--max-regression NAME=POINTS,...] [--gate-mode estimate] [--margin 2]
                             [--allow-missing] [--engine auto|vllm|hf] [--template base|ft]
                             [--accept-template SHA256] [--seed N] [--no-cache] [--sandbox auto]
                             [--out DIR] [--json]
tailtune eval RUN [--against base|parent|both]
tailtune eval --chain RUN
```

| Argument or option | Default | Effect |
|---|---|---|
| `BASE`, `CANDIDATE` | required | Hub ids or local directories with merged safetensors, or a PEFT adapter merged in memory ([09](09-evaluation.md) section 4.1) |
| `RUN` | | Re-evaluate a run against its recorded base, or its parent for a chain |
| `--chain RUN` | | Walk back to the original base and report every stage (09 section 9) |
| `--suite`, `--full` | `fast` | Benchmark suite. `--full` is short for `--suite full` |
| `--cap-thinking` | off | Caps thinking at 1,024 tokens, labelled on every affected row. It does not change the suite (09) |
| `--retention-set` | none | A named user set, repeatable, in 09's JSONL format |
| `--max-regression`, `--gate-mode`, `--margin`, `--allow-missing` | none, `estimate`, `2`, off | Regression gate and statistics settings (09 sections 5 and 8.3) |
| `--engine` | `auto` | vLLM where installed and supported, otherwise Hugging Face generation, with the same items and scorers |
| `--template` | none | When base and candidate templates differ, use this side's template for both, and record it |
| `--accept-template` | none | Accept a candidate whose template no longer matches its own tailtune record, by naming the new hash. Recorded in the report header ([10](10-export-and-deployment.md) section 8) |
| `--seed`, `--no-cache`, `--sandbox` | 0, off, `auto` | Reproducibility, cache and code-execution settings (09) |
| `--out` | `./eval-<timestamp>/` | Output directory. For `RUN`, the run directory |
| `--against` | `both` for chains, otherwise `base` | Comparison target |

**Output:** `FORGETTING_REPORT.md` and `forgetting_report.json`, plus the terminal summary. **Exit codes:** 0. 3 when a gate fails. 4 for a chat-template mismatch (TT-E001, TT-E004). 5 when the two models are not a before-and-after pair, or an input format is not accepted, such as GGUF (TT-V001). 1 on other errors.

### 4.7 `tailtune export`

Exports a run or a merged model directory (D-23). Formats and layout are in [10](10-export-and-deployment.md).

```text
tailtune export SOURCE [FORMAT ...] [--outtype q8_0] [--out DIR] [--ollama-name NAME]
                       [--push-to-hub REPO [--public] [--override-name-rule]]
```

| Argument or option | Default | Effect |
|---|---|---|
| `SOURCE` | required | `RUN` or a merged model directory |
| `FORMAT` | the run's `export.formats` | `merged`, `gguf`, `modelfile`, `ollama` (runs `ollama create`) |
| `--outtype` | `q8_0` | GGUF type. Types that need the quantize tool are listed in 10. A missing tool falls back to `q8_0` with a message |
| `--out` | inside `model/` | Output location |
| `--push-to-hub` | none | Upload to a private repository by default, with the base licence carried through ([10](10-export-and-deployment.md) section 10) |
| `--public` | off | Create the pushed repository as public |
| `--override-name-rule` | off | Push a derivative whose name breaks the base licence's naming rule, such as a Llama 3.1 derivative not named "Llama…". Without it the push is refused. The override is printed and written into the model card |

Export is idempotent: an existing output with matching hashes is reused. Export has no template override: its template check compares the files it wrote with `lineage.json`, so a failure means the artifact is wrong (TT-E003). **Exit codes:** 0. 1 on any export error, including TT-E003, following 10 section 8. 5 for a bad option value.

### 4.8 `tailtune run`

Chats with a trained model. This is a thin launcher, not a chat app (D-03).

```text
tailtune run SOURCE [--engine auto|ollama|transformers] [--prompt TEXT] [--system TEXT]
                    [--max-new-tokens 512] [--accept-template SHA256]
```

| Option | Default | Effect |
|---|---|---|
| `SOURCE` | required | `RUN`, a model directory or a `.gguf` file |
| `--engine` | `auto` | Ollama if it is on `PATH` and a GGUF exists, otherwise Transformers on the merged model |
| `--prompt` | none | Answer one prompt and exit, for scripts and notebooks |
| `--accept-template` | none | Run a model whose template no longer matches its record. The value must be the new hash, which the refusal prints, so an override is always deliberate. The banner then shows "template override accepted: sha256:…" |

It verifies the template hash against `tailtune_export.json` or the GGUF sidecar before the first message ([10](10-export-and-deployment.md) section 8). **Exit codes:** 0. 4 on a template mismatch without a matching `--accept-template` (TT-E001). 1 for a missing artifact or an engine failure.

### 4.9 `tailtune repair` (v0.3, pending verification)

```text
tailtune repair BASE FINETUNED --out DIR
```

This is planned for v0.3 only if DG-Hard verifies at Gate 4. It needs no data or training. It does not ship, and is not documented as available, before then ([03](03-methods.md)).

## 5. Python API

### 5.1 One implementation behind both surfaces

| CLI | Python | Module |
|---|---|---|
| `tailtune doctor` | `tt.doctor(deep=False) -> DoctorReport` | `tailtune.hardware` |
| `tailtune estimate MODEL` | `tt.estimate(model, ...) -> Estimate` | `tailtune.hardware` |
| `tailtune inspect MODEL` | `tt.inspect(model, ...) -> TargetReport` | `tailtune.targets` |
| `tailtune train SOURCE --dry-run` | `tt.plan(data, ...) -> Plan` | `tailtune.train` |
| `tailtune train SOURCE` | `tt.train(data, ...) -> Run` | `tailtune.train` |
| `tailtune eval BASE CAND` | `tt.evaluate(base, candidate, ...) -> RetentionReport` | `tailtune.eval` |
| `tailtune eval RUN` | `tt.Run.load(path).evaluate(...)` | `tailtune.train` |
| `tailtune export RUN gguf` | `tt.Run.load(path).export("gguf")` | `tailtune.export` |
| `tailtune run RUN --prompt P` | `tt.Run.load(path).generate(P)` | `tailtune.export` |

Every CLI flag becomes a keyword argument or an `overrides` entry with the same dotted key. A CLI-built and a Python-built run with the same inputs produce the same `tailtune.yaml`, apart from provenance tags ([02](02-architecture.md) acceptance test A4).

**The one deliberate difference: the Python API never prompts.** Notebook cells should not block on input. In Python, an ambiguous mapping raises `AmbiguousMappingError`, which lists the candidates, and `tt.plan` stands in for the confirmation.

### 5.2 `tt.read`

```python
def read(
    source: str | os.PathLike | Sequence[str | os.PathLike]
            | "datasets.Dataset" | "datasets.DatasetDict" | "pandas.DataFrame" | Sequence[Mapping[str, Any]],
    *,
    sheet: str | Sequence[str] | None = None,    # None: auto-select
    header_row: int | None = None,               # 1-based; None: detect
    system: str | None = None,                   # column templates, e.g. "{question}"
    user: str | None = None,
    assistant: str | None = None,
    text: str | None = None,                     # continued pretraining
    mapping: str | os.PathLike | None = None,    # reuse a mapping.yaml
    format: str = "auto",                        # auto|messages|sharegpt|alpaca|prompt_completion|preference|text
    hub_config: str | None = None,               # Hub dataset config
    split: str | None = None,
    revision: str | None = None,
    model: str | None = None,                    # tokenizer for render and token checks
    max_seq_length: int | None = None,
    on_overflow: str = "drop",                   # drop|truncate|raise
    eval_fraction: float | None = None,
    seed: int = 0,
) -> Dataset: ...
```

This follows [05](05-data-ingestion.md), with two renames decided here. The return type is called `Dataset`. The Hub dataset config parameter is `hub_config`, so that it cannot be confused with `tt.train(config=...)`, which takes a `tailtune.yaml`.

**Raises:** `UnsupportedFormatError`, `AmbiguousMappingError`, and `DataValidationError` when a blocking check fails among those that can run. `tt.read` itself never touches model weights.

### 5.3 `Dataset`

`Dataset` is immutable. Methods that change it return a new object.

```python
class Dataset:
    source: SourceInfo                 # path, kind, sha256, sheet list with chosen and skipped sheets
    mapping: Mapping                   # .templates, .confidence, .reasons, .save(path), .to_yaml()
    model: str | None                  # the tokenizer this dataset is rendered for, if any
    sha256: str                        # hash of the kept examples
    def __len__(self) -> int: ...      # kept examples
    def report(self) -> DataReport: ...
    def preview(self, n: int = 3) -> list[Sample]: ...   # messages, plus rendered text once a model is bound
    def for_model(self, model: str) -> "Dataset": ...    # render and token checks, tokenizer only
    def with_mapping(self, **templates: str) -> "Dataset": ...   # e.g. with_mapping(assistant="{answer}")
    def to_hf(self) -> "datasets.Dataset": ...           # messages format
    def to_pandas(self) -> "pandas.DataFrame": ...
```

**What `data.report()` can check before a model is chosen.** In the README flow, `tt.read(...)` is called without a model, so the tokenizer is not known yet.

| Available without a model | Needs a model (`model=` or `.for_model`) |
|---|---|
| Sheets, header row, merged cells, hidden content | Rendered samples with special tokens |
| Rows read, kept and dropped by reason, with sheet row numbers | BOS count, final EOS, share of labels trained |
| Duplicates, empty cells, role alternation | Token length distribution and over-length policy |
| Mapping, confidence and reasons | Template name, source and hash |
| Personal-data counts, row-count warnings | |

Without a model, the report shows "token checks: pending". `tt.train` completes them using the tokenizer alone, before any weights load. `DataReport` prints as a terminal table through `str()`, and has `.ok`, `.blocking`, `.warnings`, `.to_dict()` and `.save(path)`.

### 5.4 `tt.plan` and `Plan`

```python
def plan(data: Dataset | str | os.PathLike, *, model: str = "auto", method: str = "auto",
         r: float | None = None, config: str | os.PathLike | Mapping[str, Any] | None = None,
         overrides: Mapping[str, Any] | None = None, from_run: "Run | str | os.PathLike | None" = None,
         run_dir: str | os.PathLike | None = None) -> Plan: ...

class Plan:
    run_dir: Path
    model: str; revision: str
    method: str; route: str; reason: str          # reason is the one-line auto explanation
    estimate: Estimate; targets: TargetReport; data_report: DataReport
    config: TailtuneConfig                        # fully resolved, auto values replaced
    warnings: list[str]
    def __str__(self) -> str: ...                 # exactly the text the CLI shows before [Y/n]
```

A plan runs steps 1 to 8 of the train sequence and writes `mapping.yaml`, `data_report.json` and `targets.json`. `tt.train(plan)` executes it without re-resolving anything.

### 5.5 `tt.train`

```python
def train(
    data: Dataset | Plan | str | os.PathLike,
    *,
    model: str = "auto",
    method: str = "auto",
    r: float | None = None,
    config: str | os.PathLike | Mapping[str, Any] | None = None,
    overrides: Mapping[str, Any] | None = None,       # dotted keys, the same as --set
    from_run: "Run | str | os.PathLike | None" = None,
    run_dir: str | os.PathLike | None = None,
    report: bool = True,                              # False is the same as --eval none
    max_regression: Mapping[str, float] | None = None,
    coverage_floor: float | None = None,              # None: targets.coverage_floor (0.90)
    export: Sequence[str] | None = None,              # None: config default ["merged", "gguf"]
    nproc: int = 1,
    resume: bool = False,
) -> Run: ...
```

**Raises:** `ConfigError` (exit 5 in the CLI), `DataValidationError` (with `.report`), `AmbiguousMappingError`, `DoesNotFitError`, `CoverageBelowFloorError`, `TargetTypeError`, `LowPrecisionTargetError`, `UnsupportedBackendError`, `TemplateRefusedError`, `ModelAccessError` subclasses, and `RegressionGateError` (with `.report` and `.run`, raised only after the run is complete). The class names and their exit codes are in [02](02-architecture.md) section 10. The README call `tt.train(data, model="Qwen/Qwen3-0.6B", method="lora")` works unchanged.

### 5.6 `Run`

```python
class Run:
    id: str; path: Path
    status: Literal["complete", "regressed", "failed", "incomplete"]
    config: TailtuneConfig; lineage: Lineage; targets: TargetReport; data_report: DataReport
    model_path: Path
    @classmethod
    def load(cls, path: str | os.PathLike) -> "Run": ...
    def retention_report(self) -> RetentionReport: ...          # reads forgetting_report.json
    def export(self, fmt: Literal["merged", "gguf", "modelfile", "ollama"], *,
               outtype: str = "q8_0", out: str | os.PathLike | None = None,
               name: str | None = None) -> Path: ...              # idempotent
    def evaluate(self, *, suite: str = "fast", against: str = "base",
                 retention_sets: Mapping[str, str | os.PathLike] | None = None,
                 max_regression: Mapping[str, float] | None = None,
                 accept_template: str | None = None) -> RetentionReport: ...
    def generate(self, prompt: str, *, system: str | None = None, max_new_tokens: int = 256,
                 accept_template: str | None = None) -> str: ...      # same check as tailtune run
    def continue_with(self, data: Dataset | str | os.PathLike, **kwargs: Any) -> "Run": ...   # chain
```

`run.retention_report()` raises `ReportUnavailableError` when the run used `report=False`. `str()` of a `RetentionReport` prints the summary block from section 3.1, always including the worst benchmark.

### 5.7 `tt.wrap`

```python
def wrap(trainer: "transformers.Trainer", *, method: str = "auto", r: float | None = None,
         K: int | None = None, run_dir: str | os.PathLike | None = None, report: bool = True,
         max_regression: Mapping[str, float] | None = None, base_model: str | None = None,
         coverage_floor: float | None = None) -> "transformers.Trainer": ...
```

It returns the same trainer, with callbacks installed and a `tailtune_run` attribute. Inside `wrap`, `method="auto"` reads the model's state rather than the hardware: a LoRA or OSF PEFT model gets the report only, a dense fp32 model gets `sst` at `r=0.99`, and a bf16 model is refused (TT-P001). The refusal offers two fixes: reload with `dtype=torch.float32`, or train without `tt.wrap` and check the result with `tailtune eval`. Under D-12, the `full` and `freeze` baselines also need fp32, so neither is a way around the refusal. The full sequence and its unproven parts are in [02](02-architecture.md) section 6. Support for `SFTTrainer` is a goal until the TRL tests pass (01).

### 5.8 Other functions

```python
def evaluate(base: str | os.PathLike, candidate: str | os.PathLike, *, suite: str = "fast",
             retention_sets: Mapping[str, str | os.PathLike] | None = None,
             max_regression: Mapping[str, float] | None = None, gate_mode: str = "estimate",
             margin: float = 2.0, engine: str = "auto",
             template: Literal["base", "ft"] | None = None, accept_template: str | None = None,
             out: str | os.PathLike | None = None) -> RetentionReport: ...
def inspect(model: str, *, method: str = "sst", r: float | None = None,
            preset: str | None = None, revision: str | None = None) -> TargetReport: ...
def estimate(model: str, *, method: str | None = None, route: str | None = None, r: float | None = None,
             seq_len: int = 1024, micro_batch: int = 1, nproc: int = 1,
             gpu: str | None = None, vram_gib: float | None = None) -> Estimate: ...
def doctor(*, deep: bool = False) -> DoctorReport: ...
```

The evaluation function is `tt.evaluate`, never `tt.eval`, which would shadow the builtin and collide with the `tailtune.eval` module ([02](02-architecture.md) section 4.13).

## 6. `tailtune.yaml` schema

Every key appears in a run's `tailtune.yaml`, with its provenance ([02](02-architecture.md) section 7). The **Owner** column names the document that decides the final value. Defaults marked [J] are provisional until that document or a gate result confirms them. Four of them are also recorded in the README as provisional defaults: the 0.90 coverage floor, the minimum free tail of 4, frozen norms, and the 2-point non-inferiority margin. An unknown key exits 5 (TT-C001), and so does a value of the wrong type or out of range (TT-C003).

**Top level and run**

| Key | Type | Default | Owner |
|---|---|---|---|
| `schema_version` | int | `1` | 02 |
| `run.name` | str or null | `null` (auto-numbered `run-NNN`) | 02 |
| `run.runs_dir` | path | `runs` (env `TAILTUNE_RUNS_DIR`) | 02 |
| `run.seed` | int | `42` [J] | 02 |

**Model**

| Key | Type | Default | Owner |
|---|---|---|---|
| `model.id` | str | `auto` (section 7.2) | 04 |
| `model.revision` | str or null | `null`, resolved to a commit hash | 02 |
| `model.trust_remote_code` | bool | `false` | D-08 |
| `model.attn_implementation` | `auto`, `sdpa`, `flash_attention_2`, `eager` | `auto` (resolves to `sdpa`) | 02 |

**Data**

| Key | Type | Default | Owner |
|---|---|---|---|
| `data.source` | str, path or list | required | 05 |
| `data.sheet` | str, list or null | `null` (auto-select) | 05 |
| `data.header_row` | int or null | `null` (detect) | 05 |
| `data.hub_config`, `data.split`, `data.revision` | str or null | `null` | 05 |
| `data.format` | `auto`, `messages`, `sharegpt`, `alpaca`, `prompt_completion`, `preference`, `text` | `auto` | 05 |
| `data.mapping` | path or null | `null` | 05 |
| `data.system`, `data.user`, `data.assistant`, `data.text` | template or null | `null` (detected) | 05 |
| `data.include_hidden` | bool | `false` | 05 |
| `data.train_on` | `assistant`, `last_turn`, `all` | `assistant` | 05 |
| `data.system_policy` | `prepend_to_user`, `error` | `prepend_to_user` [J] | 05 |
| `data.eos` | `check`, `append` | `check`: a missing end token is blocking. `append` adds it to each assistant span | 05 |
| `data.max_seq_length` | int | `2048` [J] | 05 |
| `data.on_overflow` | `drop`, `truncate`, `raise` | `drop` (blocking above 10% dropped) [J] | 05 |
| `data.dedupe` | `exact`, `none` | `exact` [J] | 05 |
| `data.eval_fraction` | float or null | `null` (5% when 200 or more rows, at least 20 and at most 1,000) [J] | 05 |
| `data.pii_scan` | bool | `true` | 05 |

**Method**

| Key | Type | Default | Owner |
|---|---|---|---|
| `training.mode` | `sft`, `cpt` | `sft` (`cpt` when the format is `text`). Later: `sdft` at Gate 4, `dpo` and `grpo` at Gate 5 | D-16 |
| `method.name` | `auto`, `lora`, `sst`, `osf`, `full`, `freeze`, `tail-lora` | `auto` | 04, 07 |
| `method.preset` | null or `sst_paper` | `null` | 03, 06 |
| `method.sst.r` | float in (0, 1) | `0.99` | D-10 |
| `method.sst.K` | int or null | `null` (no refresh). Refresh is unverified | D-10 |
| `method.sst.form` | `auto`, `head`, `tail` | `auto` | 03 |
| `method.osf.r` | float in (0, 1] | `0.5` [J], PEFT's default, passed as `effective_rank` [S](https://huggingface.co/docs/peft/en/package_reference/osf). Not comparable with `sst` at 0.99 | 03 |
| `method.osf.post_step` | bool | `false`. `true` is the variant `osf/peft-hooks+post-step` | 03 |
| `method.lora.rank`, `method.lora.alpha` | int | `16`, `32` [J] | 03 |
| `method.lora.dropout` | float | `0.0` [J] | 03 |
| `method.lora.base_quant` | `auto`, `4bit`, `none` | `auto` (set by the route) | 07 |
| `method.freeze.blocks` | int or `auto` | `auto`: the last 25% of decoder blocks, rounded up [J] | 03 |

**Targets**

| Key | Type | Default | Owner |
|---|---|---|---|
| `targets.role_actions` | map of role to `constrain`, `adapter`, `free` or `frozen` | `{}`: 06's default policy. Attention and MLP projections are targeted. Routers, conv kernels, `mtp` heads, norms and scalar parameters are frozen. Embeddings and `lm_head` stay frozen until the Gate 1 ablation. `sst_paper` trains norms, embeddings and head, as the paper does | 06, D-15 |
| `targets.coverage_floor` | float | `0.90` of trainable parameters in targeted roles [J]. `sst_paper` sets 0 and warns | 06 |
| `targets.min_free_tail` | int | `4` [J]. A constrained matrix with fewer free directions is frozen | 06 |
| `targets.freeze_unresolved` | bool | `false`: unresolved groups refuse | 06 |

**Training**

| Key | Type | Default | Owner |
|---|---|---|---|
| `training.epochs` | float | `2`, as in the published SST recipe [S](https://github.com/Arnav0400/SST/blob/ff0c5bf/configs/model/qwen3-8b.yaml) | 03 |
| `training.max_steps` | int or null | `null` | 03 |
| `training.learning_rate` | float or `auto` | `auto`: the method default from 03. `sst` starts from the published 5e-5 [S](https://github.com/Arnav0400/SST/blob/ff0c5bf/configs/model/qwen3-8b.yaml) for Qwen3, which must be validated on other models. `osf` uses 5e-5 [J] | 03 |
| `training.effective_batch_size` | int | `16` [S](https://github.com/Arnav0400/SST/blob/ff0c5bf/configs/model/qwen3-8b.yaml) | 03 |
| `training.micro_batch_size` | int or `auto` | `auto` (largest that fits the estimate) | 07 |
| `training.lr_scheduler` | str | `cosine` | 03 |
| `training.warmup_steps` | int | `10`, as in the SST recipe [S] | 03 |
| `training.weight_decay` | float | `0.0` | 03 |
| `training.max_grad_norm` | float | `1.0` | 03 |
| `training.optimizer` | `auto`, `adamw_torch_fused`, `adamw_torch`, `adamw_8bit` | `auto` (fused on CUDA). `adamw_8bit` with `sst` is a variant whose results do not transfer | 02, 07 |
| `training.precision` | `auto`, `bf16`, `fp16` | `auto`: bf16 autocast where real bf16 exists, otherwise fp16. The tensor the optimizer updates is always fp32 for constrained methods and for the `full` and `freeze` baselines | D-12 |
| `training.gradient_checkpointing` | bool | `true`, always with `use_reentrant=False` | D-19 |
| `training.compile` | bool | `false` | 02 |
| `training.packing` | bool | `false` | 05 |
| `training.logging_steps` | int | `10` | 02 |
| `training.save_total_limit` | int | `1` | 02 |
| `training.report_to` | `none`, `wandb` | `none` | 11 |

**Backend, bases, evaluation, export**

| Key | Type | Default | Owner |
|---|---|---|---|
| `backend.name` | `auto`, `single`, `ddp`, `fsdp2`, `zero2`, `zero3` | `auto` (`single`, or `ddp` when `nproc` is above 1) | 08 |
| `backend.nproc` | int | `1` [J]. Multi-GPU is never started without being asked | 08 |
| `backend.fsdp2.reduce_dtype` | str | `float32`, always explicit | 08 |
| `backend.accelerate_config` | path or null | `null`: tailtune writes one into the run directory | 08 |
| `bases.cache_dir` | path | `~/.cache/tailtune/bases` (env `TAILTUNE_CACHE`) | 02 |
| `bases.placement` | `auto`, `gpu`, `host` | `auto` (set by the route) | 07 |
| `eval.enabled` | bool | `true` | D-22 |
| `eval.suite` | `fast`, `full` | `fast` | 09 |
| `eval.engine` | `auto`, `vllm`, `hf` | `auto` | 09 |
| `eval.retention_sets` | map of name to path | `{}` | 09 |
| `eval.margin` | float, points | `2.0` [J]: the non-inferiority margin per benchmark | 09 |
| `eval.max_regression` | map of step name to points | `{}` | 09 |
| `eval.gate_mode` | `estimate`, `confident`, `strict` | `estimate` [J] | 09 |
| `export.formats` | list | `[merged, gguf]` | 10 |
| `export.gguf_outtype` | str | `q8_0` | 10 |
| `export.ollama_name` | str or null | `null` (`tailtune-<run id>`) | 10 |
| `export.push_to_hub` | str or null | `null`. A pushed repository is private by default | 10 |
| `provenance` | map | Written by tailtune, ignored on input | 02 |

Environment variables are limited to paths and switches: `TAILTUNE_RUNS_DIR`, `TAILTUNE_CACHE`, `HF_TOKEN`, `NO_COLOR`, and later `TAILTUNE_TRITON` (D-04).

## 7. The provisional `method="auto"` rule

### 7.1 Decision table

This rule restates [07](07-hardware.md) section 15. It stays provisional until Gate 1 measurement 5 decides between OSF and SST [J]. The rules are checked in order, and the first match wins.

| Rule | Condition | Result |
|---|---|---|
| A0 | `method` set explicitly | Use it. Exit 4 if its verdict is "does not fit". Run with a warning if "borderline" |
| A1 | No supported accelerator, and no `--allow-cpu` | Exit 4 (TT-H002) |
| A2 | `sst/gpu` or `sst/host-bases` fits, host RAM included | `sst`, `r=0.99`, `K=None` |
| A3 | `osf` fits | `osf` (PEFT tuner, default `effective_rank`) |
| A4 | `lora-16bit` fits | `lora` on a 16-bit base |
| A5 | `lora-4bit` fits and bitsandbytes is installed | `lora` on a 4-bit base |
| A6 | `lora-4bit` would fit, but bitsandbytes is missing | Exit 4 (TT-H003), suggesting `tailtune[bnb]` |
| A7 | Nothing fits | Exit 4 (TT-H001) |

`auto` never selects `sst/8bit-adam`, `sst/offload-state`, `sst/zero-offload`, `full`, `freeze` or `tail-lora`. Those run only when named.

### 7.2 `model="auto"`

When no model is given, tailtune picks one of the launch models (D-20) by GPU memory limit, then applies `method="auto"` to it [J]:

| Rule | GPU limit | Model |
|---|---|---|
| M1 | 24 GiB or more | `Qwen/Qwen3-8B` |
| M2 | 12 to 24 GiB | `Qwen/Qwen3-4B` |
| M3 | Under 12 GiB, or CPU with `--allow-cpu` | `Qwen/Qwen3-0.6B` |

This keeps the first run short on free GPUs. The choice is printed and recorded.

### 7.3 The line it prints

The format is: the choice, then the strongest route that did not fit, with its size and the smallest card that would hold it. If the choice is constrained and the model is under 4B, a second line adds the published small-model result (07). Illustrative examples, with figures from 07's computed tables:

```text
method=auto -> lora on a 4-bit base. sst needs about 138 GiB of GPU memory for this model;
the smallest single card is a B200 180 GB.

method=auto -> osf (PEFT OSF, 50% preserved). sst needs about 138 GiB of GPU memory for this
model; the smallest single card is a B200 180 GB.

method=auto -> sst (r=0.99, fixed basis). This is the strongest route available.

method=auto -> sst (r=0.99, fixed basis). SST (Chavan et al., 2026) protects small models least:
Science dropped 8.0 points at 1.7B even under SST.
```

## 8. Error message catalogue

Every message has the same shape, printed to stderr:

```text
error[TT-D012] Every label is masked in 1,204 of 1,204 examples, so training would learn nothing.
  cause  The assistant turns could not be found in the rendered text.
  fix    Check the samples in runs/run-001/data_report.json, then pass --assistant "{answer}"
         or edit runs/run-001/mapping.yaml and rerun with --mapping runs/run-001/mapping.yaml.
  docs   <docs site>/reference/errors/#tt-d012
```

Text in `<angle brackets>` is filled in at run time. Warnings use `warning[...]` and do not change the exit code.

| Code | Exit | Message | Cause | Fix it suggests |
|---|---|---|---|---|
| TT-C001 | 5 | `Unknown key "<key>" in <source>. Did you mean "<nearest>"?` | A typo in `tailtune.yaml` or `--set` | Correct the key. `tailtune train --help` lists the sections |
| TT-C002 | 5 | `No terminal is attached, so tailtune cannot ask for confirmation. Nothing was trained.` | Running in a script, CI or a notebook shell cell without `--yes` | Review the plan printed above, then rerun with `--yes`, or use `tt.train` in Python |
| TT-C003 | 5 | `Invalid value for <option or key>: "<value>". <rule>.` For example: `Invalid value for --r: "1.5". r must be between 0 and 1, exclusive.` | A value of the wrong type, out of range, or not among the allowed names, including a parse error on the command line | Use a value the rule allows. `tailtune <command> --help` lists them |
| TT-C004 | 5 | `<runs dir>/<name> already exists. tailtune never overwrites a run.` | `--name` reused | Choose another name, omit `--name` to auto-number, or continue it with `--resume <name>` |
| TT-D001 | 2 | `<file> is not a format tailtune can read (detected: <kind>). Supported: xlsx, xlsm, xls, xlsb, ods, csv, tsv, json, jsonl, parquet, arrow, Hugging Face dataset ids.` | Unknown signature, or a file whose extension does not match its contents | Export to one of the listed formats. A `.csv` that is really xlsx is read as xlsx automatically |
| TT-D002 | 2 | `<n> sheets look like training data: <names>. tailtune will not guess which one you mean.` | Several data-like sheets, under `--yes` or in Python | Pass `--sheet <name>`, or `--sheets A,B` to combine sheets with matching headers |
| TT-D003 | 2 | `Could not map columns to roles with enough confidence (<score>). Candidates for assistant: <cols>.` | Headers that match no alias, or two equally likely columns | Pass `--user "{<col>}" --assistant "{<col>}"`, or edit and pass the written `mapping.yaml` |
| TT-D010 | 2 | `<n> of <m> examples start with two BOS tokens (<token><token>). The model would learn a pattern it never sees at inference.` | Text already rendered with a chat template was tokenized with special tokens added. Transformers warns that this duplicates tokens [S](https://github.com/huggingface/transformers/blob/0f53828e83f2212060233d2ea552cc76617b8712/docs/source/en/chat_templating.md) | Map the raw question and answer columns instead of a pre-rendered text column, or strip the leading BOS from that column |
| TT-D011 | 2 | `The last assistant turn does not end with EOS or the template's stop token in <n> of <m> examples. The trained model may never stop generating.` | A template or user template that omits the end-of-turn token. Truncated examples are exempt but counted | Pass `--set data.eos=append` to add the EOS token to each assistant span, or use a template that closes assistant turns ([05](05-data-ingestion.md)) |
| TT-D012 | 2 | `Every label is masked in <n> of <m> examples, so training would learn nothing.` | No assistant content was found to train on. This is the "All labels in your dataset are -100" failure [S](https://unsloth.ai/docs/basics/troubleshooting-and-faqs.md) | Check the samples, set `--assistant`, or set `--set data.train_on=all` for raw text |
| TT-D013 | 2 | `<pct>% of examples are longer than max_seq_length (<n> tokens) and would be dropped.` | Long answers, or a limit set too low | Raise `--set data.max_seq_length=<n>` (and re-run `estimate`), or choose `--set data.on_overflow=truncate` |
| TT-D014 | 2 | `<model> has no chat template, and this data is conversational.` | A base model with no template | Use the model's instruct variant, or pass a template ([05](05-data-ingestion.md)) |
| TT-D020 | warning | `Only <n> examples were kept. Fewer than 100 rarely teaches a model anything reliable.` | A small sheet. 100 rows is Unsloth's published minimum [S](https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/datasets-guide.md) | Add examples, or continue knowing the in-domain score will be noisy |
| TT-D021 | warning | `<n> formula cells have no saved value and were read as empty.` | A workbook written by a script, never opened and saved in Excel | Open the file in Excel or LibreOffice, save it, and rerun |
| TT-T001 | 4 | `Only <pct>% of trainable parameters in <model> are covered by targeting rules, below the <floor>% floor.` | A policy override left too much training unconstrained, or the structural fallback left matrix-shaped parameters unresolved. Each case names the parameters | Run `tailtune inspect <model> --show groups`. Then remove the override, accept the leak with `--coverage-floor 0.80` (recorded in lineage and the report), replicate the paper with `--preset sst_paper`, or freeze unresolved groups with `--freeze-unresolved`. Report models with unresolved groups ([06](06-models.md) section 4.9) |
| TT-T002 | 4 | `<param> is a Conv1D layer, which <method> does not support as a target.` | GPT-2-style layers. PEFT v0.21.2 would skip them silently and train nothing [S](https://github.com/huggingface/peft/blob/v0.21.2/src/peft/tuners/osf/model.py) | Use `lora`, or a model with `nn.Linear` projections |
| TT-T003 | 4 | `<n> target parameters are quantized (<dtype>). <method> cannot train quantized weights.` | A quantized checkpoint, such as MXFP4 experts ([06](06-models.md)) | Use the model's bf16 release, or `--method lora` on a 4-bit base |
| TT-T004 | 4 | `tailtune cannot read the shapes of "<model_type>" models yet.` | An architecture whose shape arithmetic has not been checked, or a multimodal config whose towers the text config does not describe. 0.0.1 accepts Llama, Mistral, Qwen2, Qwen3 and Qwen3-MoE | Open a model-support issue. Safetensors-header reading at v0.1 widens the set |
| TT-P001 | 4 | `<method> needs the weights the optimizer updates to be fp32, but <n> target matrices are <dtype>.` | In `tt.wrap`, a model loaded without `dtype`. Transformers v5 defaults to `dtype="auto"` [S](https://github.com/huggingface/transformers/releases/tag/v5.0.0). D-12 applies to constrained methods and to the `full` and `freeze` baselines | Load with `from_pretrained(..., dtype=torch.float32)`, or let `tt.train` load the model. To keep your own bf16 training unchanged, drop `tt.wrap` and run `tailtune eval <base> <your model>` afterwards |
| TT-P002 | 4 | `bf16 was requested, but <gpu> only emulates bf16.` | `training.precision=bf16` on a T4 | Remove the setting. On this GPU tailtune uses fp16 autocast, and the optimizer still updates fp32 weights |
| TT-H001 | 4 | `<method> does not fit: it needs about <x> GiB of GPU memory for <model>, and this GPU's limit is <y> GiB.` | The route's estimate exceeds the limit (computed, not measured) | Run `tailtune estimate <model>` for every route. Use `--method auto`, a smaller model, or a card with at least <card> |
| TT-H002 | 4 | `No supported GPU was found. CPU is a test platform for tailtune, not a training target.` | CPU-only machine, or an unsupported GPU (D-17) | Use a free Colab or Kaggle T4 notebook, or pass `--allow-cpu` for a tiny smoke test |
| TT-H003 | 4 | `lora on a 4-bit base would fit, but bitsandbytes is not installed.` | Missing extra | `uv pip install "tailtune[bnb]" --torch-backend=auto` |
| TT-H004 | 4 | `An NVIDIA GPU is present, but the installed torch is a CPU-only build.` | Plain pip resolved a CPU wheel | `uv pip install --reinstall torch --torch-backend=auto` |
| TT-H005 | 1 | `Out of GPU memory at optimizer step <n>. The estimate was <x> GiB (band <lo> to <hi>).` | The estimate was too low for this model or machine | Rerun with `--set training.micro_batch_size=1` or a smaller `max_seq_length`, and attach `logs/events.jsonl` to an issue so the formula can be corrected |
| TT-H006 | 4 | `doctor --strict: <n> check(s) failed (<ids>).` | At least one FAIL row under `--strict` | Fix the FAIL rows, or run without `--strict` to report only |
| TT-B001 | 4 | `This trainer uses FSDP version 1, which tailtune does not support.` | `fsdp_config` without `version: 2`. Accelerate still defaults to 1 [S](https://huggingface.co/docs/accelerate/concept_guides/fsdp1_vs_fsdp2) | Set FSDP version 2 explicitly |
| TT-B002 | 4 | `Multi-GPU training is not supported on Windows.` | Windows has no NCCL [S](https://docs.pytorch.org/docs/2.15/distributed.html) | Use `--nproc 1`, or WSL2 or Linux for multi-GPU |
| TT-M001 | 4 | `<model> is gated, and this account has not been granted access.` | Licence acceptance needed, as for Llama-3.1-8B-Instruct | Accept the licence on the model's Hugging Face page with the same account, then rerun |
| TT-M002 | 4 | `<model> requires a Hugging Face token, and none was found.` | `HF_TOKEN` is not set and no saved login exists | Set `HF_TOKEN`, or log in with the Hugging Face CLI. In notebooks, add `HF_TOKEN` to Colab or Kaggle secrets |
| TT-M003 | 4 | `<model> needs custom code from its repository, and tailtune does not run remote code by default.` | The model requires `trust_remote_code` (D-08) | Use a model supported natively by Transformers, or pass `--trust-remote-code` after reading that code. The choice is recorded, and such models never carry a launch claim ([06](06-models.md)) |
| TT-M004 | 4 | `<model> has no safetensors weights. tailtune does not load pickle files.` | Only `.bin` files (D-08) | Use a safetensors revision, or convert the weights yourself in an isolated environment. This code applies to `train`, `inspect`, `estimate`, `export` and `run`. In `eval`, the same condition is TT-V001, because format is one of 09's preconditions |
| TT-M005 | 1 | `Could not reach the Hugging Face Hub to read <model>, and its config is not cached.` | No network, a proxy, or a Hub outage. A cached config is used when one exists | Check the network or proxy settings, or pass a local model directory |
| TT-E001 | 4 | `The chat template in <path> (sha256:<a>) does not match the one used in training (sha256:<b>).` | At `run` or `eval`: tokenizer files were replaced or edited after training, or an engine was given different files. At training start: the tokenizer at the pinned revision no longer matches the data report. A wrong template is the most common cause of gibberish after export [S](https://unsloth.ai/docs/basics/troubleshooting-and-faqs.md) | At `run` or `eval`: restore the tokenizer and template files in `model/` from the base revision, since they are byte copies of it ([10](10-export-and-deployment.md) section 3), then re-export `gguf`. If the change is intended, pass `--accept-template <a>` instead, which is recorded. At training start there is no override: rerun from the source data so it is rendered with this tokenizer |
| TT-E002 | 1 | `The GGUF converter cannot map tensor <name>.` | Tensor names changed from the base's names, or the architecture is not in the converter [S](https://github.com/ggml-org/llama.cpp/blob/master/conversion/base.py) | Export `merged` and convert from that. Report the model if `merged` was already used |
| TT-E003 | 1 | `Export stopped: <file> in <out> has template sha256:<a>, but lineage.json records sha256:<b>. The export is incomplete.` | The template check at export ([10](10-export-and-deployment.md) section 8). Files in `model/` were edited after training, or tailtune wrote them wrongly. `tailtune_export.json` is not written | If you edited `model/`, restore the files from the base revision and export again. Otherwise this is a bug: report it with the two hashes |
| TT-E004 | 4 | `<base> and <candidate> use different chat templates (sha256:<a> and sha256:<b>).` | The fine-tune changed its template, so the two sides would be prompted differently | Pass `--template base` or `--template ft` to use one template for both sides. The choice is recorded in the report ([09](09-evaluation.md) section 4.1) |
| TT-V001 | 5 | `<base> and <candidate> cannot be compared: <check> failed (<detail>).` | One of [09](09-evaluation.md)'s preconditions: a GGUF or non-safetensors input, a different `model_type`, architecture, hidden size or layer count, or a tokenizer that differs beyond appended tokens | Pass the exact base the fine-tune started from, as merged safetensors. For a GGUF, evaluate the merged directory it came from. Only `eval` raises this code. Elsewhere, pickle-only weights are TT-M004 |
| TT-R001 | 3 | `Regression gate failed: <bench> dropped <x> points (limit <y>).` | The `--max-regression` gate was crossed | Read `FORGETTING_REPORT.md`. Try a stronger method, a lower learning rate or fewer epochs. The model was still saved |
| TT-W001 | Python only | `tt.wrap() must be called before trainer.train().` | `wrap` after training started | Build the trainer, call `tt.wrap`, then `train()` |

## 9. Notebooks

### 9.1 Which notebooks exist, and when

Unsloth's 467 notebooks [S](https://github.com/unslothai/notebooks/tree/d85d113eacfc8a87a7003e73ffb2a86403a94344/nb) are not the target. A large catalogue belongs to Stage 3, after v1.0 ([01](01-product-strategy.md)). Before then, the rule is that **every notebook runs in CI every week**, so the count is capped by the free test quota [J].

| Gate | Notebook | What it shows | Platforms |
|---|---|---|---|
| Gate 2 v0.1 Door | `spreadsheet-to-gguf` | Upload `faq.xlsx`, `lora` with the report, GGUF, test a prompt | Colab, Kaggle |
| Gate 3 v0.2 Proof | `check-any-fine-tune` | `tt.evaluate` on a public fine-tune of a small model | Colab, Kaggle |
| Gate 3 v0.2 Proof | `wrap-your-trainer` | `tt.wrap` on a tiny `Trainer` script | Colab, Kaggle |
| Gate 3 v0.2 Proof | `sst-on-a-rented-gpu` | `sst` on Qwen3-8B on a 96 GB or larger card | Generic Jupyter. Tested per release on a rental |
| Gate 4 v0.3 Reach | `small-constrained-on-t4` | `sst` or `osf` up to about 1B, labelled for what it is (D-18) | Colab, Kaggle |
| Gate 4 v0.3 Reach | `continue-a-model` | Chains with `--from`, the report against parent and root | Colab, Kaggle |
| Gate 4 onwards | Model-launch notebooks | One per hyped model release, published within 24 to 72 hours, with `inspect` and the report | Colab |

That gives about 6 notebook sources by v1.0, or about 11 rendered files, plus at most three live model-launch notebooks at a time [J]. A launch notebook that cannot be kept passing is archived, not left broken.

### 9.2 One template, generated

```text
notebooks/
  specs/<id>.yaml           title, platforms, model, method, data fixture, ordered list of cells
  cells/*.md.j2, *.py.j2    reusable cells: install, GPU check, upload, secrets, train, report, export, test
  colab/<id>.ipynb          generated, committed
  kaggle/<id>.ipynb         generated, committed
tools/gen_notebooks.py      renders specs with Jinja2 and writes .ipynb files with nbformat
```

- Each spec is a short list of cell ids plus parameters. Platform differences live only in the install, upload, secrets and output cells.
- Every notebook has a cell tagged `parameters`, holding `model`, `max_steps`, `data_path` and `tiny`, so papermill can run it in a cheap mode (D-09).
- CI regenerates every notebook and fails on any diff. Hand edits to `.ipynb` files are rejected.
- Notebook content is written from scratch. Unsloth's notebooks are LGPL and are not copied (D-06).

### 9.3 How they are tested

The jobs are defined in [11](11-engineering-and-release.md).

| Job | When | What it proves |
|---|---|---|
| Generator check | Every PR | Committed notebooks match the specs |
| `notebooks-cpu` | Every PR and nightly | papermill with a tiny model on CPU: the cells and the API still agree |
| `colab-resolve` | Daily | tailtune's pins resolve against Colab's published package snapshot [S](https://github.com/googlecolab/backend-info/blob/main/pip-freeze.gpu.txt) |
| `kaggle-t4` | Weekly | A real T4 run with 20 optimizer steps, because the fp16 scaler skips early ones [S](https://github.com/unslothai/unsloth/blob/main/.github/workflows/kaggle-t4-notebook-ci.yml). It refreshes the VRAM and time figures shown in each notebook |
| Manual Colab run | Before each release | Colab cannot be driven from CI in the same way [J] |

Budget check [C]: at about 20 to 40 minutes per notebook in its short T4 mode, 6 notebooks take 2 to 4 hours a week. That sits well inside Kaggle's reported weekly quota of about 30 hours, which is user-reported [U].

### 9.4 Colab specifics

- **Install:** `%pip install "tailtune[bnb]"`. Never reinstall torch, and never use `--torch-backend` here (section 2).
- **GPU check:** the first code cell runs `tailtune doctor`. Without a GPU it stops with "Runtime > Change runtime type > T4 GPU".
- **Data:** upload with `google.colab.files.upload()`, or mount Drive.
- **Token:** `HF_TOKEN` from Colab secrets, needed only for gated models.
- **Limits:** a runtime lasts at most 12 hours and resources are not guaranteed [S](https://research.google.com/colaboratory/faq.html). Host RAM is about 12.7 GB by folk knowledge only [U], which is why constrained tuning stops at about 1B here (D-18).
- **Output:** download the GGUF, or copy the run directory to Drive. The final cell runs `run.generate(...)` so the user sees the model answer.

### 9.5 Kaggle specifics

- **Accelerator:** a T4 is required. Kaggle's P100 cannot run torch 2.15 or later (D-17), so the GPU check cell stops with instructions to switch.
- **Internet:** must be switched on in the notebook settings for the install and the Hub.
- **Token:** `HF_TOKEN` from Kaggle's secrets.
- **Data:** attach `faq.xlsx` as a Kaggle dataset under `/kaggle/input/`. Write runs to `/kaggle/working/`.
- **Limits:** the maximum runtime is reported as 9 or 12 hours, and the sources conflict [U]. The weekly quota of about 30 hours is user-reported [U]. T4 x2 runtimes report about 29 to 30 GB of host RAM [U], which allows constrained tuning up to about 2B with host offload, as best effort (D-18).

## 10. UI position

**No app before v1.0** [J]. The reasons:

- D-03 rules out a local chat app, and a training UI drifts towards one.
- Desktop distribution has a large cost. Unsloth's workflows include `release-desktop` and `virustotal-installer-delta` [S](https://api.github.com/repos/unslothai/unsloth/contents/.github/workflows), and its desktop launch brought install and AppImage issues such as [#8463](https://github.com/unslothai/unsloth/issues/8463) and [#8546](https://github.com/unslothai/unsloth/issues/8546).
- One maintainer cannot keep a UI, a CLI and notebooks in step while also running the benchmark.
- The users in [01](01-product-strategy.md) all have a first contact without one. The spreadsheet owner gets a notebook, which already works as a UI with an upload button and visible output.
- Transformer Lab shows the risk: a GUI-first trainer, AGPL, with public development quiet since August 2026 [S](https://api.github.com/repos/transformerlab/transformerlab-app/commits?per_page=3).

**What would change this:**

1. Gate 2 user tests in which spreadsheet owners fail at the terminal step, at a rate the notebook does not fix.
2. A second maintainer with front-end skills. Unsloth's own split is one person on algorithms and one on everything else [S](https://un-supervised-learning-pod.simplecast.com/episodes/unsloth-chat).
3. Stage 3 funding ([01](01-product-strategy.md)).

If a UI is built, it is a separate local package that calls the same Python API. Its licence is decided then, while the training core stays Apache-2.0 (D-25).

**A guided wizard in the CLI?** Only the minimal one already described: the sheet picker, the column picker on low confidence, and the single confirmation. There is no separate `init` or wizard command before Gate 2 user tests [J]. `tailtune train` with no source prints a three-line hint (an example command, `--help`, and the notebook link) rather than starting a questionnaire.

## 11. Docs site

The site is built with Zensical from a conservative `mkdocs.yml` (D-09). It serves `llms.txt` and a Markdown copy of every page, and has one anchor per error code, which error messages link to.

```text
Home                                 promise, two flows side by side, model/notebook/VRAM/time table
Get started
  Install                            uv primary, pip, Colab, Kaggle, Windows, doctor
  Your first fine-tune from a spreadsheet
  Check any fine-tune with tailtune eval       (equal billing with the spreadsheet flow)
  Free GPU notebooks
Guides
  Data
    Spreadsheets                     sheets, headers, merged cells, hidden content
    CSV, JSON, Parquet and Hub datasets
    Column templates and mapping.yaml
    Reading the data report
  Training
    Choosing a method, and what method=auto does
    LoRA, including on a 4-bit base
    SST (Chavan et al., 2026)        exact variant, r, what is and is not protected
    OSF through PEFT
    Baselines: full and freeze
    Continued pretraining
    Continuing a model (chains)
    Wrapping an existing Trainer or SFTTrainer
  Hardware
    What fits: tailtune estimate
    Colab and Kaggle
    Windows and WSL2
    Renting a GPU for SST
    Multi-GPU
  Evaluation
    Reading the forgetting report
    Regression gates in CI
    Your own retention set
  Export and serving
    Merged safetensors, GGUF, Ollama, vLLM
Concepts
  What fine-tuning forgets
  SST and OSF are different algorithms
  Coverage: protected, free and frozen
  What "no regression detected at ±X points" means
Migrate
  From Unsloth
  From AutoTrain
  From TRL or plain Trainer
Reference
  CLI                                generated from the Typer app
  Python API                         generated from docstrings
  tailtune.yaml                      generated from the schema
  Run directory and lineage.json
  Exit codes
  Errors (TT-xxxx)
  Support matrix                     torch, Transformers, platforms, per release
  Model labels                       "runs" and "retention benchmarked" per model and mode
Benchmark
  Results, preregistration, how to reproduce, how to submit
Project
  Contributing, governance, security, changelog, sponsors
```

**From Unsloth.** A concept map, followed by "keep Unsloth, add the report":

| In Unsloth | In tailtune |
|---|---|
| `FastLanguageModel.from_pretrained(...)` and `get_peft_model(...)` | `tt.train(data, model=..., method="lora")` |
| `to_sharegpt` with `{col}` and `[[...]]` | The same template syntax in `tt.read(user=..., assistant=...)` |
| `train_on_responses_only` | The default, `data.train_on: assistant` |
| `check_dataset` | `data.report()`, which runs by default and blocks |
| `save_pretrained_gguf(...)` | Default `train` output, or `run.export("gguf")` |
| A fine-tune you already have | `tailtune eval <base> <your merged model>` |

**From AutoTrain.** AutoTrain says it is no longer maintained and recommends Axolotl, TRL or `transformers.Trainer` [S](https://github.com/huggingface/autotrain-advanced/blob/552dfe8c245067113ada4b5f918fb9af95d57dc4/README.md), none of which is no-code. The page shows the two-command flow, reading the user's existing CSV or xlsx without conversion, the Colab notebook as the no-install path, and a table of AutoTrain task names against tailtune modes, with "not supported" stated plainly.

## Implementation checklist

**Gate 0 Claim**
- [ ] `tailtune doctor` and `tailtune estimate` with the output formats in sections 4.2 and 4.3, `--json`, and `doctor --strict`.
- [ ] Exit-code mapping 0 to 5 and 130, the parse-error remap to 5, and the error-message format, including TT-C001, TT-C003 and TT-H001 to TT-H004.
- [ ] The install section of the README and docs, with `uv pip install tailtune --torch-backend=auto` first.

**Gate 1 Verify**
- [ ] Update the `method="auto"` table (section 7) from measurement 5, and recalibrate the estimate bands shown in the plan.

**Gate 2 v0.1 Door**
- [ ] `tailtune inspect`, `train`, `eval`, `export` and `run`, with every option in section 4.
- [ ] `tt.read`, `Dataset`, `tt.plan`, `Plan`, `tt.train`, `Run`, `tt.wrap`, `tt.evaluate`, `tt.inspect`, `tt.estimate` and `tt.doctor`.
- [ ] The `--yes` behaviour table, the TTY check and TT-C002.
- [ ] Template checks with exit 4 at training start, `run` and `eval`, the `--accept-template` and `--template` overrides, and their records in the output. TT-E003 at export, with 10.
- [ ] Every catalogue entry in section 8, each with a test that triggers it and a docs anchor.
- [ ] The `tailtune.yaml` schema in section 6, with JSON Schema export for the reference page.
- [ ] The notebook generator, `spreadsheet-to-gguf` in Colab and Kaggle renders, `notebooks-cpu` and `kaggle-t4`.
- [ ] Docs: Get started, the Data and Training guides, the Reference section, and the Unsloth and AutoTrain migration pages.
- [ ] A clean-room timing of the first run on Windows, Linux and Colab, replacing the doc-counted table.

**Gate 3 v0.2 Proof**
- [ ] `check-any-fine-tune`, `wrap-your-trainer` and `sst-on-a-rented-gpu` notebooks.
- [ ] Benchmark pages and the model-labels table.

**Gate 4 v0.3 Reach**
- [ ] `tailtune repair`, only if DG-Hard verifies.
- [ ] `small-constrained-on-t4`, `continue-a-model` and the first model-launch notebook.
- [ ] `training.mode: sdft`.

**Gate 5 v1.0**
- [ ] Freeze `schema_version` 1 and the public Python signatures, with a deprecation policy for later changes.
- [ ] `training.mode: dpo` and `grpo`.
- [ ] Re-decide the UI position against the three conditions in section 10.

## Acceptance tests

| # | Test | Passes when |
|---|---|---|
| U1 | First run | On a Colab T4, `faq.xlsx` gives a running model after install with exactly two commands and one `[Y/n]`, and the logit round trip passes (Gate 2 exit test) |
| U2 | README surface | The README's Python block runs unchanged on a tiny model, and `data.report()` before a model shows "token checks: pending" |
| U3 | One implementation | CLI and Python runs with the same inputs give identical resolved configs apart from provenance tags |
| U4 | `--yes` | Each row of the section 4.1 table gives the stated behaviour and exit code, using fixture files |
| U5 | Non-TTY | `tailtune train fixture.xlsx < /dev/null` prints the plan and exits 5 with TT-C002, and trains nothing |
| U6 | Error catalogue | Every TT code has a fixture that triggers it, the exact message text matches this catalogue, and the docs anchor exists |
| U7 | `auto` rule | A parametrised test over model sizes and card sizes reproduces the section 7.1 table. The excluded routes are never chosen |
| U8 | `auto` line | The printed line names the choice, the strongest unfit route and the smallest card, plus the small-model line when it applies |
| U9 | Schema | Every key in section 6 exists in the schema with the stated type and default, including coverage floor 0.90, `min_free_tail` 4, frozen norms and margin 2. An unknown key exits 5 with TT-C001 |
| U10 | Notebooks | The generator check passes, every notebook runs in `notebooks-cpu`, and `spreadsheet-to-gguf` passes `kaggle-t4` |
| U11 | Python never prompts | With an ambiguous fixture, `tt.train` raises `AmbiguousMappingError` and never reads stdin |
| U12 | Template check | Editing one character of the chat template in `model/` makes `tailtune run` exit 4 with TT-E001, and `--accept-template <new hash>` lets it run with the override shown. `tailtune export` on the same run exits 1 with TT-E003 |
| U13 | `eval` on any model | `tailtune eval` on a base and a merged model built with plain Transformers produces both report files |
| U14 | `doctor` exit | On a machine with a FAIL row, `tailtune doctor` exits 0 and `tailtune doctor --strict` exits 4 |
| U15 | Code 5 | An unknown key, `--r 1.5`, an unknown flag, and `tailtune eval` on two unrelated architectures each exit 5, with TT-C001, TT-C003, TT-C003 and TT-V001 |

## Open questions

1. *(Resolved in the plan index: model access and pickle-only refusals use 4, a missing terminal uses 5, a declined confirmation uses 1, and Ctrl-C uses 130.)*
2. **`model="auto"` defaults.** The M1 to M3 thresholds and the choice of Qwen3 sizes are a judgement. Gate 2 user tests should check whether a 4B first run on a T4 is too slow.
3. *(Resolved in the plan index: a mismatch found by export's own self-check exits 1, and every other template mismatch exits 4.)*
4. **GGUF by default.** Building the converter environment on first export adds download time on Colab. If Gate 2 timings show this hurts the first run, `export.formats` could default to `[merged]`, with `tailtune run` building GGUF on demand.
5. *(Resolved: 05 now uses `tailtune.Dataset` and `hub_config`.)*
6. **The CLI wizard.** Should a minimal guided mode appear if Gate 2 user tests show people stuck at the first command?
7. **Notebook quota.** Kaggle's weekly quota is user-reported. If it is lower, the "every notebook weekly" rule shrinks the catalogue further.
8. *(Resolved: the thinking cap is spelled `--cap-thinking` in both 04 and 09.)*
