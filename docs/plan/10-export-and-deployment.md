# 10. Export and deployment

**What this document decides:**

- Export priority follows D-23: merged safetensors, then GGUF in an isolated environment, then vLLM, then Ollama. TGI is not a target.
- Merged safetensors always use the base checkpoint's key names and shapes, with its config, tokenizer and chat template intact. Every other path builds on that directory.
- Each method reaches dense weights by a stated route, in fp32. The result is then cast once to the base's storage dtype, and the update lost in that cast is measured and reported.
- GGUF conversion runs in a separate, pinned virtual environment, never in the training environment. The default preset is `q8_0`, because the converter alone can produce it.
- The template hash defined in [05](05-data-ingestion.md#template-hash) travels with every artifact and is checked at export and at load.
- A small low-rank adapter export for no-refresh SST is an untested hypothesis. It ships only if a logit-match gate passes (Gate 4).
- Model cards are generated, carry the forgetting summary and lineage, and flag licence duties for legal review.

Evidence labels are defined in the [plan index](README.md#evidence-labels). All code and output in this document are untested design proposals or illustrations. Licence remarks are engineering notes, not legal advice.

## 1. Priority order

| # | Target | Why this position | Verified at |
|---|---|---|---|
| 1 | Merged safetensors | Every other path converts from it. The GGUF converter fails on any tensor it cannot map, `Can not map tensor {name}` [S](https://github.com/ggml-org/llama.cpp/blob/master/conversion/base.py), so key names must match the base exactly. [09](09-evaluation.md) evaluates this artifact | Gate 2 |
| 2 | GGUF | GGUF builds take 17.4% of downloads in the top 300 text-generation models, by one vendor-maintained snapshot [S](https://dev.to/ai_openfree_b23025ef075cf/qwen-still-takes-half-of-open-llm-downloads-what-changed-in-the-hugging-face-top-300-since-1gbn). Beginners expect it | Gate 2 |
| 3 | vLLM | The evaluation harness depends on it, so it must work anyway (D-22) | Gate 2, through the harness |
| 4 | Ollama | A Modelfile on top of the GGUF. Mostly documentation [S](https://github.com/ollama/ollama/blob/main/docs/import.mdx) | Gate 2 |
| 5 | Low-rank delta adapters | No-refresh SST only. A hypothesis (section 9) | Gate 4, if it passes |
| 6 | MLX | Tier 3 (D-17). Documented only | Gate 4, docs |
| 7 | SGLang | Its LoRA documentation could not be retrieved [U] | Gate 5, docs |
| — | TGI | "now in maintenance mode" [S](https://github.com/huggingface/text-generation-inference/blob/main/README.md) | Never |

## 2. Layout of `model/`

[02](02-architecture.md) leaves this layout to this document. Everything stays inside the run directory's `model/` entry, so the README's run-directory table is unchanged.

```text
model/
  config.json, generation_config.json      from the base, dtype field updated only
  model-0000N-of-0000M.safetensors         base key names; index file if sharded
  tokenizer files, chat template            byte copies from the base revision
  README.md                                 generated model card (section 10)
  LICENSE, NOTICE                           when the base licence requires them
  tailtune_export.json                      hashes, dtype, method, file manifest
  adapter/                                  lora and tail-lora only: PEFT adapter files
  gguf/
    run-001.q8_0.gguf
    run-001.q8_0.gguf.tailtune.json         sidecar: template hashes, converter commit
    Modelfile
    convert.log
```

`tailtune_export.json` holds `template_hash`, `chat_template_sha256`, the save dtype, the method and its variant, the base ID and commit, the delta-survival figures (section 3), and a SHA-256 for every file. That manifest is what `lineage.json`'s model hash covers ([02](02-architecture.md)).

## 3. Merged safetensors

### Contract

| Item | Rule |
|---|---|
| Tensor names and shapes | Exactly the base checkpoint's set: none added, none missing, none renamed |
| Tied embeddings | Stored as the base stored them. If the base holds both keys, write both with identical values. Qwen3-0.6B stores both [S](https://huggingface.co/Qwen/Qwen3-0.6B) |
| Towers and `mtp` heads | Written unchanged under their checkpoint keys ([06](06-models.md)). Stale `mtp` heads produce a warning |
| `config.json` | Copied from the base. Only the dtype field changes |
| Tokenizer and template | Byte copies of the base revision's files. The tokenizer is never re-saved through Transformers v5 (section 4). A template override or user template is written in the base's file layout, and the export says so |
| Serialization | safetensors only (D-08). v5 removed `safe_serialization=False` [S](https://raw.githubusercontent.com/huggingface/transformers/main/MIGRATION_GUIDE_V5.md) |
| Shards | Transformers v5 raised the default shard size from 5 GB to 50 GB [S](https://github.com/huggingface/transformers/releases/tag/v5.0.0). Match the base's shard size where practical, otherwise 5 GB [J] |

Why the key rule is hard: v5's `WeightConverter` can fuse separate q, k and v projections into one tensor at load [S](https://raw.githubusercontent.com/huggingface/transformers/main/MIGRATION_GUIDE_V5.md). In-memory names can then differ from checkpoint keys. [06](06-models.md) records both key spaces, with slices. Export writes checkpoint keys. Whether `save_pretrained` reverses every conversion is unverified [U]. So:

1. Save with `save_pretrained`.
2. Read the safetensors headers of the output and of the base revision.
3. Compare names, shapes and dtypes. On any difference, rewrite the state dict under the recorded checkpoint keys, using the recorded slices.
4. Check again. A second failure is a key-mapping error (exit 1, [02](02-architecture.md) step 17).

### How each method reaches merged form

| Method | Trained state | Route to dense weights, computed in fp32 | Extra artifact |
|---|---|---|---|
| `lora`, 16-bit base | Adapter A and B | Reload the base in fp32, then PEFT `merge_and_unload()`: W + (α/r)·B·A | `model/adapter/` |
| `lora`, 4-bit base | Adapter trained against NF4 weights | Merge into the **original 16-bit base** from the pinned revision, never into dequantized NF4 weights [J] | `model/adapter/` |
| `sst` | Dense fp32 master weights (D-12) | None needed | None at v0.1. Section 9 at Gate 4 |
| `osf` | PEFT factors `U_l, S_l, V_l`, with the base layer kept | W_base + (U_l S_l V_l − U⁰S⁰V⁰). PEFT's forward adds exactly this δ, as a rank-2r product [S](https://github.com/huggingface/peft/blob/f482b819536f475da8d4a304cd7a7e37cae59351/src/peft/tuners/osf/layer.py#L317-L355) | None. δ is not small at PEFT's default rank |
| `full`, `freeze` | Dense fp32 | None needed | None |
| `tail-lora` (v0.3, experimental) | Adapter | As `lora` ([03](03-methods.md)) | `model/adapter/` |

Two cases need care:

- **OSF merging.** PEFT raises `NotImplementedError("OSF models do not support unmerging")` [S](https://github.com/huggingface/peft/blob/f482b819536f475da8d4a304cd7a7e37cae59351/src/peft/tuners/osf/model.py). Whether `merge_and_unload()` merges OSF correctly is unverified [U]. tailtune computes W_base + δ itself and requires the largest absolute difference from PEFT's merge to be at most 1e-5 times max |W| [J]. If PEFT cannot merge, tailtune's own result is used, confirmed by a logit check ([03](03-methods.md)).
- **LoRA on a 4-bit base.** The merged model is not the model that was trained, because training saw quantized base weights. The size of that gap is unmeasured [U]. Export reports the logit KL between the training-time model (4-bit base plus adapter) and the merged model on the fixed prompts in section 4. The forgetting report evaluates the merged artifact, and says so.

### Precision on save

Constrained methods train fp32 master weights (D-12). Export casts once, to the base's **storage dtype**, read from the base's safetensors headers. The `torch_dtype` field is not used, because llama.cpp's own converter warns that "some finetunes lie" about it [S](https://github.com/ggml-org/llama.cpp/blob/master/convert_hf_to_gguf.py).

| Save dtype | When | Consequence |
|---|---|---|
| Base storage dtype, usually bf16 | Default ([02](02-architecture.md) step 17) | Same size as the base, and what converters expect. bf16 keeps 8 significand bits, so a weight's total change survives only if it is larger than about half the bf16 spacing at that weight [C]. Small tail updates can vanish, and rounding adds noise in protected directions ([03](03-methods.md)) |
| float32 | `dtype="float32"` | Exact. Twice the bf16 size. The converter accepts f32 [S](https://github.com/ggml-org/llama.cpp/blob/master/conversion/base.py) |
| float16 | Opt-in | Finer significand than bf16 but a narrower range. Some recent families force float32 because fp16 produces NaNs [S](https://github.com/unslothai/unsloth/blob/main/unsloth/models/loader.py#L213-L223) |

**Delta survival.** For every trained matrix, export computes the relative error of the delivered update: ε = ‖(cast(W_ft) − W_base) − ΔW‖_F / ‖ΔW‖_F, where ΔW = W_ft − W_base in fp32. The median and maximum of ε go into `tailtune_export.json` and the model card. If the median exceeds 0.1, export warns and recommends float32 [J]. Gate 2 measures how much retention this costs, which is an open question in [03](03-methods.md).

## 4. GGUF

### The isolated environment

llama.cpp's converter pins `transformers==4.57.6`, `torch==2.11.0`, `gguf>=0.1.0`, `sentencepiece>=0.1.98,<0.3.0` and `protobuf>=4.21.0,<5.0.0` [S](https://github.com/ggml-org/llama.cpp/blob/master/requirements/requirements-convert_hf_to_gguf.txt). tailtune needs Transformers v5 and torch 2.14 or later (D-05). The two cannot share an environment. `tailtune.export.isolated_env` manages a separate one:

| Aspect | Design [J] |
|---|---|
| Pin | Each tailtune release names one llama.cpp commit. The weekly compatibility job (D-07) bumps it only after the base conversions below pass |
| Location | `<user cache>/tailtune/gguf/<commit>/`, holding `src/` (llama.cpp at that commit, fetched as an archive and checked against a recorded SHA-256), `venv/` and `env.json` (Python version, resolved package versions) |
| Creation | `uv venv`, then install the converter requirements file with CPU-only torch. Conversion needs no GPU. Which Python versions the pinned torch supports must be checked at pin time [U] |
| Invocation | A subprocess: `<venv>/python -I <src>/convert_hf_to_gguf.py <model_dir> --outfile <path> --outtype <type>`, run from an empty temporary directory, with output captured to `model/gguf/convert.log` |
| Reuse | Reused while `env.json` matches. Rebuilt on mismatch. `tailtune doctor` reports its state |
| Bring your own | `TAILTUNE_LLAMA_CPP_DIR` points at an existing checkout, much as Axolotl uses `$LLAMA_CPP_DIR` [S](https://github.com/axolotl-ai-cloud/axolotl/blob/620c1efca4470c1bc2e4a5f46b4b106341e9dc41/docs/export.qmd). The sidecar records "user-supplied converter" and the commit if it can read one |
| Containers | Both Docker images carry the environment pre-built ([07](07-hardware.md)) |
| Licence | Fetched at run time, never vendored. Its licence is recorded as [U] in [02](02-architecture.md). Confirm it at each pin |

### The tokenizer risk

Whether the pinned converter, running on Transformers 4.57.6, reads tokenizer files written by Transformers v5 is untested. v5 changed the on-disk tokenizer layout, so the risk is real [U]. Some bases are already written by 5.x versions: the Qwen3.8 and Gemma 4 configs carry `5.8.0.dev0` and `5.5.0.dev0` [S](https://huggingface.co/Qwen/Qwen3.8-27B/blob/main/config.json).

Mitigations:

1. Export never re-saves the tokenizer. It byte-copies the base's files, so tailtune introduces no new v5 layout.
2. Adding tokens is refused at v0.1, because embeddings are frozen by default (D-15). So a byte copy is always possible.
3. A weekly CI job converts each launch model's **unmodified base** at the pinned commit. A base that fails marks GGUF as unsupported for that model in the support matrix, until llama.cpp catches up.

### Presets

The converter itself writes f32, f16, bf16, q8_0, tq1_0, tq2_0 and auto [S](https://github.com/ggml-org/llama.cpp/blob/master/convert_hf_to_gguf.py). K-quants need the compiled `llama-quantize` tool. A missing tool is a known Unsloth complaint [S](https://github.com/unslothai/unsloth/issues/748).

| Preset | Made by | Status [J] |
|---|---|---|
| `q8_0` | Converter | **Default.** Needs no compiled tool. Matches `export_gguf` in [02](02-architecture.md) |
| `f16`, `bf16`, `f32` | Converter | Opt-in. `f16` is used by the round trip |
| `q4_k_m` | `llama-quantize`, from an `f16` or `bf16` intermediate | Recommended for small GPUs. Unsloth's documented example uses it [S](https://unsloth.ai/docs/basics/inference-and-deployment/saving-to-gguf.md), and Axolotl's uses `Q4_K_M,Q8_0` [S](https://github.com/axolotl-ai-cloud/axolotl/blob/620c1efca4470c1bc2e4a5f46b4b106341e9dc41/docs/export.qmd). Becomes the default once getting `llama-quantize` is reliable on Tier 1 platforms |
| `q5_k_m`, `q6_k` | `llama-quantize` | Opt-in |
| `tq1_0`, `tq2_0` | Converter | Not offered |

How `llama-quantize` is obtained on each platform is an open question: a local build from the pinned source needs a compiler, and prebuilt binaries for that exact commit are unverified [U].

Quantization is far coarser than bf16 rounding [J]. Whether a quantized GGUF keeps a small SST update is unknown [U]. When a llama.cpp runtime is available, export reports KL(merged ‖ GGUF) next to KL(base ‖ merged) on the fixed prompts. If the first is as large as the second, export warns that quantization may have erased the fine-tune [J].

### Failure behaviour

| Failure | Detected by | Behaviour |
|---|---|---|
| Key-space drift | Header comparison before conversion | Internal error (exit 1) with a bug-report message. Conversion is not attempted |
| `Can not map tensor X`, where X is in the base key set | Parsing `convert.log` | "llama.cpp at commit C cannot convert this architecture's tensor X." Merged safetensors stay. Exit 1 |
| `Can not map tensor X`, where X is not in the base key set | Same | Internal error: tailtune added a tensor. Exit 1 |
| Unknown architecture | Converter error. Architectures register with `@ModelBase.register` [S](https://github.com/ggml-org/llama.cpp/blob/master/docs/development/HOWTO-add-model.md) | Names the commit and suggests a newer one. Exit 1 |
| Tokenizer load failure | Converter error | Points to the base-conversion job's result for that model. Exit 1 |
| Missing `llama-quantize` | Before quantizing | Falls back to `q8_0`, with a message. Exit 0 |

In every case, earlier artifacts stay ([02](02-architecture.md) step 19).

### CI logit round trip

This matches [11](11-engineering-and-release.md) and is the Gate 2 exit test.

1. Train a tiny model for 5 steps with each of `lora`, `sst`, `osf` and `full`.
2. Export merged fp32 safetensors and reload them with plain Transformers. On 8 fixed prompts, rendered with the stored template, the largest absolute logit difference must be at most 1e-5 [J].
3. Check that key names match the base, that the config, tokenizer and template are present, and that the template hash matches `lineage.json`.
4. **Non-trivial training guard:** KL(base ‖ trained) on the same prompts must exceed a floor set at Gate 2 [J]. Otherwise an export that silently dropped the update would pass. Unsloth has an issue titled "Applying LoRA doesn't change output" [S](https://github.com/unslothai/unsloth/issues/2009).
5. Convert to `f16` GGUF in the isolated environment, and load it in llama.cpp on CPU:
   - token IDs for the rendered prompts must equal Transformers' IDs, which catches a duplicated or missing BOS;
   - top-1 agreement must be 100%;
   - a logit tolerance is set at Gate 2 [J].
6. Convert to `q8_0` and `q4_k_m`, and report top-1 agreement and KL. These are not gated at v0.1.

Which llama.cpp runtime interface the test calls is open [U]. The test runs nightly and at release, because it is slow ([11](11-engineering-and-release.md)).

## 5. vLLM

vLLM serves the merged directory as is, and is the fast backend of `tailtune eval` through the `[vllm]` extra ([02](02-architecture.md)). An illustrative command, whose spelling should be confirmed against vLLM's docs [U]:

```text
vllm serve runs/run-001/model
```

| Point | Rule |
|---|---|
| Template | `model/` carries the training template byte for byte. `tailtune eval` verifies the hash before starting vLLM. That vLLM's chat endpoint uses the directory's template by default is unverified [U] |
| Windows | vLLM "does not support Windows directly but only via WSL or Linux" [S](https://unsloth.ai/docs/get-started/install/windows-installation.md). Evaluation falls back to Hugging Face generation ([09](09-evaluation.md)) |
| LoRA serving | Only for models that implement `SupportsLoRA`. `max_lora_rank` defaults to 16, and `max_loras` to 1 [S](https://github.com/vllm-project/vllm/blob/main/vllm/config/lora.py) |
| Adapter rank | Any exported adapter above rank 16 needs `max_lora_rank` raised. The model card's serving snippet states the value |
| torch pin | If vLLM pins a torch outside tailtune's range, evaluation may need its own environment too ([02](02-architecture.md)) |

## 6. Ollama

Ollama imports a Safetensors directory or a GGUF file through `FROM`. It does not quantize a GGUF on import, and the import page mentions no adapter route [S](https://github.com/ollama/ollama/blob/main/docs/import.mdx). Unsloth [S](https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/tutorial-how-to-finetune-llama-3-and-use-in-ollama.md), LlamaFactory [S](https://github.com/hiyouga/LlamaFactory/blob/f6355ab6a426eedb800742ecf61896f7bff95ebb/src/llamafactory/train/tuner.py) and ms-swift all write a Modelfile, so tailtune does too.

An illustrative Modelfile. The directive and comment syntax should be confirmed against Ollama's docs at Gate 2 [U]:

```text
# Generated by tailtune 0.1.0 from runs/run-001
# template_hash sha256:7d2a...  chat_template_sha256 sha256:e3b0...
FROM ./run-001.q8_0.gguf
PARAMETER stop "<|im_end|>"
```

- The stop tokens are the template's stop set ([05](05-data-ingestion.md#eos)).
- At v0.1, the Modelfile writes no hand-made `TEMPLATE` block [J]. Ollama is expected to use the template embedded in the GGUF, but that is unverified [U].
- A nightly CI check on Linux runs `ollama create` and greedy generation on 3 prompts, and compares the first 32 tokens with llama.cpp's greedy output from the same GGUF [J]. A model family that fails gets an explicit `TEMPLATE` from a small, tested table kept in tailtune.

## 7. SGLang, MLX and TGI

- **SGLang.** Unsloth documents an SGLang deployment page [S](https://unsloth.ai/docs/llms.txt). SGLang's own LoRA documentation could not be retrieved, so adapter support is unverified [U]. tailtune documents serving the merged directory only, with no CI, at Gate 5.
- **MLX** is Tier 3 (D-17). Document `mlx_lm.convert` from the merged directory [S](https://github.com/ml-explore/mlx-lm/blob/main/README.md). Whether MLX accepts each new architecture after a fine-tune is untested [U]. Docs only, at Gate 4.
- **TGI** is not a target (D-23), because it is in maintenance mode [S](https://github.com/huggingface/text-generation-inference/blob/main/README.md).

## 8. Template hash at export and at load

Unsloth's troubleshooting page says poor results or "gibberish, endless/infinite generations" after export most often come from an "incorrect chat template", or from the engine adding or omitting a start token [S](https://unsloth.ai/docs/basics/troubleshooting-and-faqs.md). The two hashes from [05](05-data-ingestion.md#template-hash) target exactly that.

| When | What is compared | On mismatch |
|---|---|---|
| Freeze | Both hashes computed and written to `data_report.json` | — |
| Training start | Tokenizer at the pinned revision, against the data report | Refused, exit 4 |
| Merged export | Hashes recomputed from `model/`'s own files, against `lineage.json` | Export fails, exit 1. `tailtune_export.json` is not written, so the artifact counts as incomplete |
| GGUF export | If the GGUF carries a chat template string, its SHA-256 against `chat_template_sha256` [U: whether the converter embeds one, and how to read it]. The sidecar is always written | A mismatch blocks. No embedded template gives a warning |
| Modelfile | Hash written as a comment. Behaviour checked in CI (section 6) | — |
| `tailtune run`, `tailtune eval` | Recomputed from the artifact, against `tailtune_export.json` or the sidecar | Refused, with both hashes and the likely cause. An override flag (spelling in [04](04-api-and-ux.md)) is recorded in the output |
| External engines | The model card lists both hashes and the stop tokens | — |

A template mismatch is a pre-flight refusal, so it exits with code 4 (see the [plan index](README.md#shared-vocabulary)).

## 9. The small low-rank delta: an untested hypothesis

**The idea.** With no refresh (`K=None`), SST keeps W − W_ref = Π(W − W_ref) after every step, including decoupled weight decay, because it projects the whole applied update ([03](03-methods.md)). Every target's delta then has rank at most t = min(d_out, d_in) − k. A truncated SVD of the fp32 delta would give a standard LoRA adapter with `lora_B = U_t S_t^½`, `lora_A = S_t^½ V_tᵀ`, r = t and `lora_alpha = r`. That works with PEFT, `convert_lora_to_gguf.py` and multi-LoRA servers, with no new format.

**Size** [C]:

- For a [4096, 12288] matrix at r=0.99: k = ⌊0.99·4096⌋ = 4055 and t = 41. The factors hold (4096 + 12288) × 41 = 671,744 values against 50,331,648, about **1.3%**.
- A square 4096 × 4096 matrix needs 2 × 4096 × 41 against 16,777,216, about 2.0%.
- Smaller matrices, such as key and value projections, have smaller tails.
- A hidden size above about 6,400 gives t above 64.

**Why it is only a hypothesis:**

- The bound has not been tested numerically, and no source confirms or refutes it.
- bf16 storage breaks exactness, so the delta must come from fp32 master weights. Rounding under mixed precision and DeepSpeed is unmeasured.
- Refresh (R10, R100) breaks the bound, because each period can add new directions. Refreshed models need merged export.
- Any trainable parameter outside the projection, such as unfrozen embeddings or norms, gives a dense delta. The LoRA format can carry that only through `modules_to_save`, and `convert_lora_to_gguf.py` rejects anything that is "Not a lora_A or lora_B tensor" [S](https://github.com/ggml-org/llama.cpp/blob/master/convert_lora_to_gguf.py).
- MoE experts stored as 3D tensors have no adapter target in current LoRA tooling, and fused tensors depend on what the converter expects.

**Procedure and gates** [J]:

1. Refuse unless `method="sst"`, `K=None`, `inspect` shows every trainable parameter is a projected 2D target, and fp32 masters are available.
2. For each target, compute ΔW in fp32 and its singular values on the CPU path from D-14. Require σ_{t+1}/σ_1 ≤ 1e-5. This is the direct test of the hypothesis.
3. Factor at rank t. Different matrices have different t: pad every target to the largest t with zero singular values, which is exact, or use PEFT per-module ranks if they are confirmed [U].
4. Require the reconstruction error ‖ΔW − BA‖_F / ‖ΔW‖_F ≤ 1e-4 for every matrix.
5. **Logit-match gate.** On the fixed prompts, base plus adapter (PEFT, fp32) against the merged fp32 model: 100% top-1 agreement and a largest absolute logit difference of at most 1e-4.
6. Write `model/adapter/` with the pinned base ID and commit, and the checkpoint key space. The card states the `max_lora_rank` vLLM needs. vLLM's default of 16 is below t = 41 [S](https://github.com/vllm-project/vllm/blob/main/vllm/config/lora.py).

The same rank t defines the capacity-matched LoRA arm in the public benchmark ([09](09-evaluation.md)).

**Licence implications.** A delta is not a licence escape. A delta trained from a model's weights plausibly remains a derivative. Llama 3.1's licence uses "derivative works" without defining it [S](https://dev.meta.ai/llama/llama3_1/license). Apply the base's naming and notice terms to deltas too, and get legal review before advertising delta sharing as a way to share work on gated models.

## 10. Model cards and `--push-to-hub`

### What the card contains

`model/README.md` is generated at export.

| Section | Content | From |
|---|---|---|
| Metadata | Base licence identifier, base model ID, tags including `tailtune`. The exact Hub metadata keys for the base-model relation are unverified [U]. The Hub's `base_model:finetune:` filter shows such relations exist [S](https://huggingface.co/models-json?other=base_model:finetune:Qwen/Qwen3-8B&withCount=true) | `lineage.json` |
| Training | Method with exact variant and rank (D-11). "SST (Chavan et al., 2026)" written in full on first use (D-02). Data summary: format, row count, `data_hash`. Never the data | `tailtune.yaml`, `data_report.json` |
| Coverage | Share of trainable parameters protected, free and frozen | `targets.json` |
| Forgetting summary | Worst benchmark first. Paired deltas with intervals. The wording "no regression detected at ±X points". The backend used, which artifact and dtype was evaluated, and the "runs" or "retention benchmarked" label for this model and mode (D-02) | `forgetting_report.json` ([09](09-evaluation.md)) |
| Export precision | Save dtype, delta survival ε, GGUF presets with any KL figures | `tailtune_export.json` |
| Lineage | Base commit, bases hash, r, coverage, data hash, template hash, dependency versions, previous run | `lineage.json` |
| Serving | Both template hashes, stop tokens, snippets for Transformers, vLLM (with `max_lora_rank` where needed), llama.cpp and Ollama | Export |
| Licence | The base licence's name and link, and its obligations as printed at export | Licence table below |
| Limitations | Stale `mtp` heads, untouched vision towers with no claim on vision retention, and no safety claim (D-02) | [06](06-models.md) |

A lint step refuses any card containing "no forgetting" or "without forgetting" (D-02).

### Licence duties for launch models

**Every row needs legal review before tailtune automates it.** Compliance stays with the user ([06](06-models.md)).

| Base (D-20) | Licence | What export does [J] |
|---|---|---|
| Qwen3-8B, Qwen3-4B, Qwen3-0.6B | Apache-2.0, not gated [S](https://huggingface.co/Qwen) | Copy the licence identifier |
| SmolLM2-135M | Apache-2.0 [S](https://huggingface.co/HuggingFaceTB) | Same |
| Gemma-4-E4B-it | Apache-2.0, not gated [S](https://huggingface.co/google) | Same |
| Granite-4.2-8b | Apache-2.0 [S](https://huggingface.co/ibm-granite) | Same |
| gpt-oss-20b, Qwen3.8-27B (experimental) | Apache-2.0 [S](https://huggingface.co/openai) | Same |
| Llama-3.1-8B-Instruct | Llama 3.1 Community License, gated. Redistributing a derivative requires a copy of the licence, a prominent "Built with Llama", "Llama" at the start of the model name, and a Notice file. Companies above 700M monthly active users must request a licence [S](https://dev.meta.ai/llama/llama3_1/license) | Card shows "Built with Llama". Copy the base repo's licence and notice files if it ships them. If it does not, refuse to push until the user supplies them. Refuse a repo name that does not start with "Llama" unless the user overrides it explicitly |

Other families with derivative conditions, such as MiniMax, Kimi K3 and GLM-5.3, are not launch models. Their terms are printed from [06](06-models.md) if they are used.

### Push flow

1. **Preconditions:**
   - export is complete, with `tailtune_export.json` present and the template hash verified;
   - the forgetting report exists, or its absence is stated on the card;
   - the licence check passes;
   - a Hub token is available.
2. **Create the repository private by default** [J]. Unsloth's CLI also exposes `--private` [S](https://github.com/unslothai/unsloth/blob/5f1d55556fd82f889abfe17a446c3b3a97a2f42b/unsloth_cli/commands/export.py) (an AGPL file: described here, not copied).
3. **Upload** `model/`, except `gguf/` and `adapter/`, together with `FORGETTING_REPORT.md`, `forgetting_report.json` and a copy of `lineage.json` with local paths removed [J].
4. **Never upload** `data.parquet`, `data_report.json` (it holds rendered samples and row numbers) or `mapping.yaml` (column names can be sensitive). Spreadsheets often hold private business data.
5. **GGUF and adapter files** go to sibling repositories, `<repo>-GGUF` and `<repo>-adapter` [J]. Unsloth's `push_to_hub_gguf` is a precedent [S](https://unsloth.ai/docs/basics/inference-and-deployment/saving-to-gguf.md).
6. **Print the URLs.**

## 11. `tailtune run`

`tailtune run <run-dir | model-dir | file.gguf>` is a smoke test that proves the exported model works. It is not a chat app (D-03). The engine is `auto`, `ollama` or `transformers` ([02](02-architecture.md)).

1. **Resolve the artifact.** A run directory points to `model/`.
2. **Verify the template hash** (section 8). A mismatch is refused.
3. **Pick an engine.** `auto` uses Ollama when it is installed and `model/gguf/Modelfile` exists. Otherwise it uses a Transformers REPL on the merged safetensors, on GPU if present, or on CPU with a speed warning.
4. **Ollama path.** Create the Ollama model from the Modelfile if it does not exist, named after the run, then hand over to Ollama's interactive prompt, or send a single request for `--prompt`.
5. **Transformers path:**
   - render with the stored template and its kwargs;
   - use the base's `generation_config.json` defaults;
   - stream tokens until a token in the stop set appears;
   - support only `/reset` and `/exit` [J].
6. **Print one header line:** artifact path, engine, dtype or quantization, and "template hash verified".
7. **Exit codes:** 0 on normal exit. 1 for a missing artifact (with the export command to run) or an engine failure. 4 for a template mismatch, as in section 8.

There are no tools, no server mode, no saved history and no UI.

## 12. Which paths are verified at which gate

| Path | Gate 2, v0.1 Door | Gate 3, v0.2 Proof | Gate 4, v0.3 Reach | Gate 5, v1.0 |
|---|---|---|---|---|
| Merged safetensors, every v0.1 method | Round trip on tiny models in CI | Six launch models, merged models published with the benchmark | Experimental models | Per-release support matrix |
| GGUF `f16`, `q8_0` | Round trip on a tiny model, nightly | Launch-model base conversions, weekly | Experimental models | Matrix |
| GGUF `q4_k_m` | Reported, not gated | Gated once a tolerance is set | | |
| Ollama Modelfile | Behaviour check on a tiny model | Launch models | | |
| vLLM, merged weights | Through `tailtune eval` on the nightly Linux GPU job | Launch models | | |
| LoRA adapter, `model/adapter/` | PEFT reload test | `convert_lora_to_gguf.py` test | `tail-lora` | |
| Template hash checks | Every leg above | | | |
| Model card and push | Card generated and linted | Push used for benchmark models | | |
| Low-rank delta | | | Experiment. Ships only if the section 9 gates pass | |
| MLX, SGLang | | | MLX docs | SGLang docs |
| TGI | Never | | | |

## Implementation checklist

**Gate 0, Claim**

- [ ] Nothing ships. Record the llama.cpp commit to pin first.

**Gate 1, Verify**

- [ ] Convert each launch model's unmodified base at the pinned commit, which tests the tokenizer risk early.
- [ ] Check PEFT's `merge_and_unload()` on an OSF model against W_base + δ.

**Gate 2, v0.1 Door**

- [ ] `export.merged`: merge routes per method, fp32 merge, cast to the base storage dtype, header-based key-space check with rewrite, byte copies of the tokenizer and template, `tailtune_export.json`, delta survival.
- [ ] `export.isolated_env`: pin, fetch, verify, create, reuse, `TAILTUNE_LLAMA_CPP_DIR`, `doctor` status.
- [ ] `export.gguf`: presets, failure parsing, sidecar, `q8_0` fallback.
- [ ] `export.modelfile`: stop tokens, hash comment.
- [ ] `export.verify`: hash checks at every stage in section 8, and the logit round trip in section 4.
- [ ] `export.serve`: `tailtune run` with both engines.
- [ ] Model card generator, licence table, wording lint.
- [ ] CI: merged round trip on every PR (CPU). GGUF and Ollama legs nightly. vLLM through the eval job.

**Gate 3, v0.2 Proof**

- [ ] Weekly base-conversion job for the six launch models.
- [ ] `--push-to-hub` with private default, exclusions and sibling repositories.
- [ ] `convert_lora_to_gguf.py` test for `lora` adapters.
- [ ] Set the `q4_k_m` tolerance and the quantization-erasure warning from measured data.

**Gate 4, v0.3 Reach**

- [ ] Low-rank delta experiment and gates (section 9). Publish the result either way.
- [ ] Export for Qwen3.8-27B and gpt-oss-20b, including the MXFP4 key space.
- [ ] MLX documentation.

**Gate 5, v1.0**

- [ ] SGLang documentation.
- [ ] Export path per model in the support matrix of each release.

## Acceptance tests

1. **Gate 2 exit, export:** merged safetensors, then GGUF in an isolated environment, then Ollama. The logit round trip passes, and the template hash is verified ([13](13-roadmap.md)).
2. For each of `lora`, `sst`, `osf`, `full` and `freeze` on a tiny model, the exported tensor names, shapes and count equal the base checkpoint's exactly.
3. The tokenizer files in `model/` are byte-identical to the base revision's.
4. Editing one character of `model/`'s chat template makes `tailtune run` refuse, and exit with code 4.
5. Installing the converter environment leaves the training environment's `pip freeze` unchanged.
6. An export that silently dropped the update fails the non-trivial training guard.
7. The model card for a Llama base contains "Built with Llama", and pushing it under a name that does not start with "Llama" is refused without an override.
8. `--push-to-hub` never uploads `data.parquet`, `data_report.json` or `mapping.yaml`.
9. `tailtune run runs/run-001 --prompt "Hello"` answers and exits with code 0, on Linux, macOS and Windows, with the Transformers engine.

## Open questions

1. **Run-directory layout.** GGUF, Modelfile and adapter files sit under `model/` to avoid changing the README table. The owner may prefer a separate `export/` entry, which would need recording in the [plan index](README.md) first.
2. How should `llama-quantize` be obtained on Linux, macOS and Windows?
3. Which llama.cpp runtime interface should the round trip and the quantization check call?
4. Does the pinned converter read each launch model's tokenizer? The Gate 1 base conversions answer this.
5. Does a quantized GGUF keep a small SST update? How much retention does a bf16 export lose against fp32 ([03](03-methods.md))?
6. Does PEFT merge OSF correctly?
7. How large is the gap between a LoRA trained on a 4-bit base and its 16-bit merge?
8. Should template verification have its own command instead of living under `run`? That would change the shared vocabulary, so it is an owner decision. The exit code is settled: a mismatch is a pre-flight refusal, code 4.
9. [13](13-roadmap.md)'s Gate 2 export package lists merged, GGUF and Ollama, but not vLLM. This document verifies vLLM at Gate 2 through the evaluation harness, following D-23. The roadmap may want to say so.
10. Does Ollama use the GGUF's embedded template for each launch family, or do some families need an explicit `TEMPLATE`?
11. Legal review is needed on: Llama notice text and naming enforcement, delta sharing, and the licence table as a whole.
