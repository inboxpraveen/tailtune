# 01. Product strategy

**What this document decides:**

- tailtune competes on showing and reducing what fine-tuning breaks, not on having a subspace algorithm or on being faster than Unsloth.
- It wins users through three things: spreadsheet ingestion, a default retention report, and a beginner flow that ends in a running model.
- It reaches "go-to" in three stages, and enters the breadth race only once it has trust, a second maintainer and compute.
- The README and the GitHub description must change now, because today they promise things the evidence does not support.

Evidence labels are defined in the [plan index](README.md#evidence-labels). Figures were retrieved on 2026-11-07 and 2026-11-08.

## 1. Who tailtune is for

No user research exists yet, so these profiles are a judgement [J]. Testing them is part of the plan.

| User | Has | Wants | First contact with tailtune |
|---|---|---|---|
| **Spreadsheet owner** | Domain knowledge, an Excel file, free Colab or one consumer GPU | A model that answers like their documents, without learning ML | `tailtune train faq.xlsx` with `lora` and the report |
| **Local model tinkerer** | Unsloth experience, a GGUF habit, a 16 to 32 GB GPU | To know whether their last fine-tune damaged the model | `tailtune eval base/ ft/` on a model trained with another tool |
| **ML engineer** | An existing `Trainer` pipeline, rented or company GPUs | A regression check in CI, and less damage on production models | `tt.wrap(trainer, ...)` and `--max-regression` |
| **Researcher** | Clusters and a method to test | A credible, reusable forgetting benchmark | The public benchmark and its repro command |

The local tinkerer is the most reachable group [J]. `tailtune eval` needs no change to their training tool, and a surprising result is worth sharing. That makes it the best early growth path, even though the spreadsheet flow is the more distinctive pitch.

## 2. Why Unsloth is the default, and which parts can be matched

Unsloth has 79,153 stars and 885,608 PyPI downloads a month [S](https://api.github.com/repos/unslothai/unsloth). It holds that position through a bundle rather than one feature.

| Part of the bundle | Evidence | Can one developer match it? |
|---|---|---|
| Free notebooks | 467 notebook files across Colab, Kaggle and AMD [S](https://github.com/unslothai/notebooks/tree/d85d113eacfc8a87a7003e73ffb2a86403a94344/nb) | The pattern yes, the count no |
| A two-call API | `FastLanguageModel`, `get_peft_model` | Yes |
| Fixed, pre-quantized models under one Hub namespace | Trust built over years | Not soon |
| One-line GGUF and Ollama export | [Saving to GGUF](https://unsloth.ai/docs/basics/inference-and-deployment/saving-to-gguf.md) | Yes |
| Speed kernels | Headline now about 1.5x speed and 50% memory on Qwen3.8 by its own guide [S](https://unsloth.ai/docs/models/qwen3.8/train.md). No independent reproducible benchmark was found | No, and not worth attempting |
| Docs | 201 English pages [S](https://unsloth.ai/docs/llms.txt) | Partly, by being narrower |
| Day-zero model fixes | A very fast release cadence | No |

Two recent shifts leave room beside it:

- **Its focus is widening.** The repo now describes itself as a "Local UI to run and train LLMs and diffusion models" [S](https://api.github.com/repos/unslothai/unsloth).
- **Its licence is less simple than its badge.** Package metadata says Apache-2.0, but AGPL text appears in 126 of 206 Python files in `unsloth/` and 83 of 84 in `unsloth_cli/` at the commit read [S](https://github.com/unslothai/unsloth/tree/5f1d55556fd82f889abfe17a446c3b3a97a2f42b). So tailtune must not depend on it (D-05, D-06), and a fully permissive stack has real value to companies. This is an engineering reading, not legal advice.

## 3. The landscape

| Tool | Stars | Interface | Excel as a table | GGUF | Biggest weakness [J] |
|---|---|---|---|---|---|
| [Unsloth](https://api.github.com/repos/unslothai/unsloth) | 79,153 | Python, CLI, UI | No | Yes | Mixed licence, widening focus |
| [LlamaFactory](https://api.github.com/repos/hiyouga/LlamaFactory) | 75,927 | CLI, YAML, WebUI | No | None found | Mandatory dataset registration, large backlog |
| [TRL](https://api.github.com/repos/huggingface/trl) | 19,685 | Python, CLI | No | No | A library, not a product |
| [ms-swift](https://api.github.com/repos/modelscope/ms-swift) | 16,058 | CLI, Web UI, Python | No | Modelfile only | ModelScope-first, broad surface |
| [Axolotl](https://api.github.com/repos/axolotl-ai-cloud/axolotl) | 12,672 | YAML, CLI | No | Yes | Config-first, no beginner path |
| [Oumi](https://api.github.com/repos/oumi-ai/oumi) | 9,429 | YAML, CLI, Python | Merges all sheets | Not implemented | Broad platform, low adoption |
| [MLX-LM](https://api.github.com/repos/ml-explore/mlx-lm) | 7,585 | CLI, Python | No | Limited | Apple only |
| [NeMo AutoModel](https://api.github.com/repos/NVIDIA-NeMo/Automodel) | 1,079 | YAML, CLI, Python | No | No | Cluster-first |
| [Training Hub](https://api.github.com/repos/Red-Hat-AI-Innovation-Team/training_hub) | 105 | Python | No | None found | Tiny adoption, JSONL only |

Two once-obvious options are gone. [AutoTrain](https://github.com/huggingface/autotrain-advanced/blob/552dfe8c245067113ada4b5f918fb9af95d57dc4/README.md) says it is no longer maintained, and sends its no-code users to tools that are not no-code. [torchtune](https://github.com/meta-pytorch/torchtune/blob/bd2a0fc7c31430972728494fa01aaeeb0ebf1ba1/README.md) says the same, though it is not archived.

## 4. What is already taken

The current README implies that anti-forgetting training is the differentiator. It is not.

| Tool | What it offers | Since or status |
|---|---|---|
| PEFT | `OSFConfig` as one tuner | First commit 2025-10-23, release v0.18.0 [S](https://github.com/huggingface/peft/pull/2685) |
| Red Hat Training Hub | First-class `osft()`, multi-node [S](https://github.com/Red-Hat-AI-Innovation-Team/training_hub/blob/1eb3d9b2034362883d7ecf9d1e69dab99c315a61/docs/README.md) | Productized in OpenShift AI [S](https://docs.redhat.com/en/documentation/red_hat_openshift_ai_self-managed/3.4/html/customize_models_for_gen_ai_and_agentic_ai_applications/train-the-model-by-using-your-prepared-data_custom-models) |
| TRL | Experimental SDFT, which uses distillation rather than subspaces [S](https://github.com/huggingface/trl/blob/4ae4d34a86f385511c85fa14caedc2178095d0b8/docs/source/sdft_trainer.md) | Experimental |
| Unsloth | Could add a published method quickly [J] | Not present at the commit read |

OSF also does not clearly win on PEFT's own benchmark, which is one single-task proxy on Llama-3.2-3B:

| Run | Test accuracy | Forgetting (lower is better) | Train time | Peak VRAM |
|---|---|---|---|---|
| [OSF rank 128](https://github.com/huggingface/peft/blob/f482b819536f475da8d4a304cd7a7e37cae59351/method_comparison/MetaMathQA/results/osf--llama-3.2-3B-rank128.json) | 42.9% | 0.0166 | 1,539 s | 29.9 GB |
| [LoRA rank 32](https://github.com/huggingface/peft/blob/f482b819536f475da8d4a304cd7a7e37cae59351/method_comparison/MetaMathQA/results/lora--llama-3.2-3B-rank32.json) | 47.8% | 0.0500 | 942 s | 22.3 GB |
| [SuperTuning s974](https://github.com/huggingface/peft/blob/f482b819536f475da8d4a304cd7a7e37cae59351/method_comparison/MetaMathQA/results/supertuning--llama-3.2-3B-s974-topk.json) | 43.9% | 0.0097 | 942 s | 21.6 GB |

The conclusion: never market "first to offer". Treat every method as replaceable, and let measured results decide what `auto` selects.

## 5. Where to compete

| Area | State today | Rating | Plan |
|---|---|---|---|
| Default retention report | PEFT's metric lives inside a [benchmark script](https://github.com/huggingface/peft/blob/f482b819536f475da8d4a304cd7a7e37cae59351/method_comparison/MetaMathQA/run.py). [CapTrack](https://github.com/thomsonreuters/captrack) is a separate evaluator. No trainer read in the research ships one | **Open** | **Compete: this is the product** |
| Spreadsheet as a table | Oumi merges all sheets, Unsloth Studio treats xlsx as a document, `datasets` has no loader | **Open** | **Compete: the front door** |
| Sequential model lineage | PEFT documents a [manual loop](https://huggingface.co/docs/peft/en/package_reference/osf). Unsloth advises [combining datasets](https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/datasets-guide.md) | **Open** | **Compete** |
| Anti-forgetting in a beginner flow | PEFT is a primitive. Training Hub is JSONL only with no export | **Open** | **Compete** |
| Anti-forgetting at low VRAM | PEFT OSF needs real `nn.Linear` weights | **Open and hard** | Benchmark `tail-lora`, promise nothing |
| Permissive beginner stack | Unsloth's CLI and UI are AGPL, but LlamaFactory and ms-swift are Apache | **Partly open** | Keep it true, mention it lightly |
| Pre-flight data validation | Exists, but opt-in and partial | **Partly open** | Make it default and blocking |
| Template drift at export | The named top cause of [post-export gibberish](https://unsloth.ai/docs/basics/troubleshooting-and-faqs.md) | **Partly open** | Template hash through to export |
| Orphaned AutoTrain users | No no-code destination | **Open but small** | One migration page |
| Kernel speed, day-zero breadth, multi-node, RL, synthetic data, chat app | Well covered by funded teams | **Closed** | Do not compete (D-03) |

**How long each lead lasts** [J]. Excel ingestion probably buys months, since Unsloth Studio already parses xlsx and has heuristic column mapping. The method buys nothing. These assets compound and are slow to copy:

- the benchmark corpus and its published losses
- per-model cached bases
- a reputation for honest results
- a report that works on anyone's model, which a competing trainer has less reason to build and less credibility in judging

### Where tailtune sits

No existing tool fills more than two of these columns well. tailtune's planned position is the only one that fills all five. Today it fills none, because no code exists yet.

| Tool | Easy for beginners | Prevents forgetting | Shows what was forgotten | Excel as a table | Ends in GGUF or Ollama |
|---|---|---|---|---|---|
| Unsloth | Yes, the best | No | No | No | Yes |
| LlamaFactory | Medium | No | No | No | Modelfile only |
| Axolotl | No, config-heavy | No | No | No | Yes |
| PEFT | No, a library | Yes, OSF | No | No | No |
| Red Hat Training Hub | No, Python and JSONL only | Yes, OSFT | No | No | No |
| CapTrack | Evaluation only | No | Yes | Not applicable | Not applicable |
| Oumi | Medium | No | No | Merges all sheets | No |
| **tailtune, planned** | Yes | Yes, SST and OSF | Yes, on every run | Yes | Yes |

**How strong each part of the gap is** [J]:

| Strength | Part of the gap | Why |
|---|---|---|
| Strong | The forgetting report | It also works on models from other trainers |
| Medium | Excel ingestion | A real gap, but easy to copy, and demand is unproven |
| Weak | The method | Already shipped elsewhere |
| Unknown | Demand for the report | Untested, which is why the demand test comes early ([13](13-roadmap.md)) |

## 6. Positioning

**Tagline:** Fine-tune LLMs with far less forgetting, and prove it.

**One paragraph:** Fine-tuning teaches a model something new and can quietly damage what it already knew. On one published benchmark, standard fine-tuning cut a model's prior capability score from 81.0 to 37.6. tailtune starts from your spreadsheet or dataset, trains with the strongest protection your hardware allows, and hands back a running model with a before-and-after report on every run.

**A note on that figure:** it comes from SST (Chavan et al., 2026), Table 5, on Qwen3-8B across four domains in thinking mode, and is author-reported [S](https://arxiv.org/html/2610.22293). Wherever it appears, name the source and say no independent replication has been found. Replace it with tailtune's own measurement as soon as one exists.

### Claims policy

D-02 sets the rules. These examples apply them.

| Do not write | Write instead | Why |
|---|---|---|
| "Fine-tune without forgetting" | "with far less forgetting, and prove it" | Under SST a 1.7B model still loses 8.0 points [S](https://arxiv.org/html/2610.22293) |
| "No forgetting detected" | "No regression detected at ±2.7 points on IFEval" (using the run's actual figure) | A fast report detects large regressions and cannot certify none |
| "Keeps your model safe" | Report the refusal metrics, make no claim | Even benign fine-tuning raised one model's harmfulness rate from 0.3% to 16.1% [S](https://ar5iv.labs.arxiv.org/html/2310.03693). SST's follow-up showed mixed results [S](https://arnav0400.github.io/subspace-tuning-emergent-misalignment/) |
| "1.5% overhead" | Our measured number, or nothing | The paper's figure could not be reconciled with its public code ([03](03-methods.md)) |
| "Works with any optimizer" | "Tested with AdamW" until Muon has been run | Muon is absent from the public SST code [S](https://github.com/Arnav0400/SST/blob/ff0c5bf/README.md) |
| "Scales to multi-node" | Name each path after it has run end to end | Public SST is ZeRO-2 only [S](https://github.com/Arnav0400/SST/blob/ff0c5bf/sst/config.py#L45-L51), and dense FSDP2 projection is unimplemented anywhere |
| "Faster than Unsloth" | Say nothing about speed | tailtune is not competing on speed (D-03) |
| "Supports model X" | "Model X: runs" or "Model X: retention benchmarked" | Keeps claims from spreading to pairs nobody measured |

### The name

Keep `tailtune`. It is short and easy to spell, and it was unclaimed on PyPI, conda-forge, Hugging Face, npm and as a GitHub org at retrieval [S](https://pypi.org/pypi/tailtune/json). No trademark clearance search has been run [U].

Keep one precision issue in mind. "Tail" is only loosely accurate: on a rectangular matrix at r=0.99, nearly all free output directions sit in the left null space, not in a low-energy tail [S](https://arxiv.org/html/2610.22293). The name can stay informal, but technical docs must say "the complement of the top-k singular subspaces".

## 7. What to change now

### README claims

| Today's README says | Verdict | Change to |
|---|---|---|
| "Fine-tune LLMs without forgetting." | Indefensible as an absolute | "Fine-tune LLMs with far less forgetting, and prove it." |
| "freezes the top of each weight matrix's singular spectrum and trains only the low-energy tail" | Only 7 matrix classes are targeted, 15.2% of Qwen3-8B's parameters train unconstrained [C], and OSF preserves far less than 99% | Name the targeted matrices and state r per method |
| "keeps its coding, reasoning and instruction-following" | Holds only on Qwen3 benchmarks, and is author-reported | Cite the paper, point to `tailtune eval` |
| "Drop-in: one wrapper around HF Trainer or TRL's SFTTrainer" | Plausible by design, no published TRL test | State as a goal until tested |
| "Optimizer-agnostic ... AdamW, Muon and others" | Tested on Adam and Muon only, and Muon is absent from public code | "Tested with AdamW and Muon" once tailtune has run both |
| "Scales: single GPU to multi-node with FSDP and DeepSpeed ZeRO-3" | Unsupported today | Roadmap wording. Claim each path after an end-to-end run |
| `pip install tailtune` and a PyPI badge | PyPI returns 404 | Remove until 0.0.1 is published |
| Python 3.10+ badge | Current torch, Transformers and TRL need 3.11 [S](https://pypi.org/pypi/transformers/json) | 3.11+ |

### Replacement README for today

This is the proposed text until v0.1 ships. It promises only what is being built. The README standard for later versions is in [11](11-engineering-and-release.md).

```markdown
<h1 align="center">tailtune</h1>
<p align="center"><b>Fine-tune LLMs with far less forgetting, and prove it.</b></p>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache--2.0-blue" alt="License"></a>
  <img src="https://img.shields.io/badge/python-3.11%2B-blue" alt="Python">
  <img src="https://img.shields.io/badge/status-pre--alpha-orange" alt="Status">
</p>

Fine-tuning teaches a model something new and can quietly damage what it
already knew. tailtune is being built to show you exactly what changed on
every run, and to reduce the damage as far as your hardware allows.

**Status: pre-alpha.** Nothing is ready to use yet. Follow the
[implementation plan](docs/plan/README.md) to see what is being built and why.

What we are building:

- **Start from what you have.** Excel read as a real table, plus CSV, JSON,
  Parquet and Hugging Face datasets, with a validation report before training.
- **A forgetting report on every run.** Before-and-after scores on coding,
  instruction following, maths, knowledge, languages and refusals.
  It will also work on a model trained with any other tool.
- **Protection that fits your hardware.** LoRA on small GPUs. SST
  (Chavan et al., 2026) and OSF where they fit. The tool tells you which
  applies before it loads any weights.
- **A running model at the end.** Merged safetensors, GGUF and Ollama.
```

### GitHub description and topics

The current repo description says "without catastrophic forgetting" and "from a single GPU to multi-node". Replace it with:

> Fine-tune LLMs with far less forgetting, and prove it. Start from Excel or any dataset, get a before-and-after report on every run, export to GGUF. Early development.

Add these topics: `fine-tuning`, `llm`, `catastrophic-forgetting`, `continual-learning`, `subspace-tuning`, `lora`, `gguf`, `huggingface`.

Both changes are public, so they need the owner's approval.

## 8. The route to go-to

The target is still to be the go-to repository. Sequencing is what changes. Each stage funds the next with trust, contributors or compute. The thresholds below are a judgement [J]. The funding ladder M0 to M4 in [12](12-community-and-sponsorship.md) is the detailed version.

### Stage 1: Wedge (0.0.1 to v0.3)

**Goal:** become the reference people cite for what fine-tuning breaks.

| Ship | Measure |
|---|---|
| `eval` on any model, Excel flow, `lora`, `sst`, `osf`, GGUF | Reports appearing in other people's model cards and posts |
| Preregistered benchmark including the rows where tailtune loses | Citations, replies from method authors, outside submissions |
| A "what a stock fine-tune breaks" post for each hyped model | Inbound traffic per post |

### Stage 2: Default (v1.0)

**Goal:** the default choice for anyone who cares whether their model still works.

| Ship | Measure |
|---|---|
| Stable config schema, FSDP2, DPO and GRPO, embeddings | Named production users, `--max-regression` in others' CI |
| Second maintainer, PyTorch Ecosystem listing, Scorecard 7+ | Bus factor above 1, sponsor conversations |

### Stage 3: Breadth (after v1.0)

**Goal:** the go-to for fine-tuning generally, including people who never think about forgetting.

Enter only with a funded or staffed team and sustained compute [J]. Reopen D-03 then. Candidates, roughly in order of value:

1. **Larger notebook catalogue**, generated from one template ([04](04-api-and-ux.md)), never hand-maintained.
2. **Day-zero model support** for the most-used families.
3. **A `tailtune` Hub namespace** of cached bases, then fixed templates.
4. **More modes and multimodal training.**
5. **Triton kernels for profiled hot ops** (D-04).
6. **A local UI**, only if the evidence shows people need it ([04](04-api-and-ux.md)).

## 9. What happens if the evidence goes against us

The strategy is designed to survive bad results. The first benchmark is the launch story either way.

| Outcome | What it means | Response |
|---|---|---|
| SST replicates on Qwen3-8B | tailtune is the first independent confirmation of a striking result | Publish, name it in `auto`, contact the authors |
| SST replicates only partly | The honest gap becomes the story | Publish the disagreement, narrow where `auto` selects it |
| SST fails to replicate | tailtune is the source that caught it | Publish, keep the report and data flow, promote whichever method measures best |
| Overhead is far above the paper's claim | SST is impractical on some hardware | Update `estimate`, limit SST to models where it is measured to fit |
| `tail-lora` works on a 4-bit base | Prevention becomes available on hardware beginners own | Make it the free-tier headline after benchmarking |
| `tail-lora` does not work | Cheap hardware gets diagnosis, rented hardware gets the cure | Say exactly that in the docs |
| Unsloth ships a retention report | The feature lead closes | Lean on independence, the benchmark corpus and `eval` on any model. Credit them |
| Unsloth or PEFT adds SST | The method lead closes, which was never the plan's basis | Support their output in `eval`, and compare it in the benchmark |
| Nobody uses Excel input | The door is the wrong door | Keep the feature, since it is cheap, and lead with `eval` |
| Nobody cares about the report | The core thesis is wrong | Re-evaluate at M2. See [14](14-risks-and-open-questions.md) |

The last row matters most. Demand for a forgetting report has not been tested. A cheap test exists before much is built: publish one `eval` result on a popular public fine-tune at Gate 2, and see whether anyone responds. Note the evaluation should be of models, not a criticism of their authors, and the results checked carefully before posting.

## 10. Non-goals before v1.0

- A chat, agent or RAG application.
- Our own speed kernels, unless profiling justifies a specific hot op (D-04).
- Synthetic data generation. Connect to [Data Designer](https://pypi.org/pypi/data-designer/json) instead.
- A hosted service.
- Multi-node as a headline. It is supported when proven, not marketed.
- Beating anyone's model count.

## Implementation checklist

**Gate 0, Claim**
- [ ] Owner approves the tagline, README and GitHub description.
- [ ] Replace `README.md`, the description and topics.
- [ ] Add a one-line pointer from the README to this plan.

**Gate 1, Verify**
- [ ] Publish the measurements, including any disagreement with the paper.
- [ ] Review the claims table against the results.

**Gate 2, v0.1 Door**
- [ ] Publish one `eval` result on a popular public fine-tune as a demand test.
- [ ] Publish the AutoTrain migration page.
- [ ] Update the README to the [11](11-engineering-and-release.md) standard.

**Gate 3, v0.2 Proof**
- [ ] Replace the SST headline figure with tailtune's own benchmark numbers.
- [ ] Publish the rows where tailtune loses alongside the rest.

**Gate 5, v1.0**
- [ ] Reassess D-03 and decide whether to enter Stage 3.

## How we will know it is working

| Signal | Stage 1 target [J] |
|---|---|
| Forgetting reports appearing in model cards not written by the maintainer | At least 10 by the end of Gate 4 |
| Outside benchmark submissions or corrections | At least 1 by the end of Gate 4 |
| A method author engaging with published results | At least 1 |
| Stars | See the M0 to M4 ladder in [12](12-community-and-sponsorship.md). Stars are a lagging signal, not the goal |

## Open questions

1. Do users recognise forgetting when it happens, and will they act on a report? Nothing in the research answers this.
2. How many people really want Excel input? GitHub shows only weak signals, with 7 issue hits for "excel" in Unsloth and few of them about training [S](https://github.com/unslothai/unsloth/issues?q=is%3Aissue+excel).
3. Could a proprietary or minor tool already offer a retention report? The "open" rating rests on absence in the sources read.
4. Reddit could not be read directly during the research, so user sentiment comes from GitHub issues and docs alone.
