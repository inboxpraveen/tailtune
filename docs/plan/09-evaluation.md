# 09. Evaluation

**What this document decides:**

- Every training run produces a before-and-after forgetting report by default. `tailtune eval base/ ft/` produces the same report for any merged model from any trainer (D-22).
- The report has five layers: paired per-benchmark deltas with the worst benchmark always shown, per-item transitions, drift against the base model, safety screening and behaviour. Task chains add continual-learning metrics.
- A fast suite of eight steps runs on vLLM, with a Hugging Face fallback that uses identical items and scorers. Base results are cached under an exact key.
- Statistics are paired: bootstrap intervals, McNemar's exact test, Holm correction and a declared non-inferiority margin. Wording follows fixed rules. The report never says "no forgetting" (D-02) and makes no safety claim.
- The public benchmark is preregistered, includes a capacity-matched LoRA arm, reports equal-setting and tuned conditions, and applies a learning gate so that a method cannot win by not learning.

Evidence labels are defined in the [plan index](README.md#evidence-labels). Every time estimate below is unmeasured [U] until Gate 1.

## 1. Why averages hide damage, and what the report is for

A mean over benchmarks can look healthy while one capability collapses:

- **CapTrack** found a legal-tuned model at +0.4% on MMLU-Pro and +0.7% on IFEval, but −16.4% on multilingual robustness [S](https://arxiv.org/html/2603.06610v2).
- **Harmon et al.** show forgetting and backward transfer cancelling inside accuracy averages, which is why they count per-item transitions [S](https://arxiv.org/html/2510.17776v1).
- **Kalajdzievski** finds forgetting tracks fine-tuning loss almost linearly (R² of .9450 and .9736) and warns that ground-truth metrics can hide forgetting [S](https://arxiv.org/html/2401.05605v1).
- **SST (Chavan et al., 2026).** Its own table shows the canaries. Under standard fine-tuning of Qwen3-8B on Science, HumanEval fell from 91.5 to 36.6 and IFEval from 85.6 to 46.2, while GSM8K moved from 94.8 to 93.9 [S](https://arxiv.org/html/2610.22293).
- **OSF's paper** says capabilities are "preserved or enhanced", yet its GSM score fell from 26.1 to 7.7 [S](https://arxiv.org/html/2504.07097v1).
- **A translation study** found EWC kept general ability at 65.1 against 66.8 for the base, while instruction following on the task family fell from 71.1 to 35.3 [S](https://arxiv.org/abs/2609.28395).

**The report exists to answer one question:** what did this fine-tune measurably break, and at what resolution? It is a diagnostic. It does not certify that nothing was lost, it is not a safety evaluation, and it does not rank methods. The public benchmark (section 10) ranks methods.

## 2. The metric stack

Notation: benchmark b has N_b items. `s_pre(i)` and `s_post(i)` are the per-item scores in [0, 1] for the base and fine-tuned models on identical prompts. Every metric is oriented so that higher means more retained, so refusal on harmful prompts counts as 1 and over-refusal on safe prompts counts as 0.

### 2.1 Layer 1: paired per-benchmark deltas

```text
A_pre(b)  = mean_i s_pre(i)            A_post(b) = mean_i s_post(i)
Δ_b       = 100 · mean_i (s_post(i) − s_pre(i))          # points
rel_b     = 100 · (A_post(b) − A_pre(b)) / A_pre(b)       # percent; "n/a" if A_pre(b) = 0
aggregate = mean of Δ_b over the capability benchmarks     # unweighted [J]
worst     = argmin_b Δ_b over all benchmarks, including safety and user sets; ties go to the lower CI bound
```

The capability benchmarks are IFEval, HumanEval, GSM8K, MMLU-Pro and MGSM. Groups follow TRACE, which defines General Ability, Instruction Following and Safety deltas as the mean change from base over a group [S](https://ar5iv.labs.arxiv.org/html/2310.06762):

- **General:** HumanEval, GSM8K, MMLU-Pro and MGSM
- **Instruction following:** IFEval
- **Safety:** XSTest and StrongREJECT

Severity bins use CapTrack's labels on `rel_b` [S](https://arxiv.org/html/2603.06610v2). The boundary convention is ours [J].

| Bin | Condition |
|---|---|
| None | rel > −1% |
| Minor | −5% < rel ≤ −1% |
| Moderate | −10% < rel ≤ −5% |
| Major | −20% < rel ≤ −10% |
| Catastrophic | rel ≤ −20% |

### 2.2 Layer 2: per-item transitions

These follow Harmon et al. [S](https://arxiv.org/html/2510.17776v1), for binary items with k options:

```text
F         = (1/N) Σ 1{a_pre = 1 ∧ a_post = 0}
BT        = (1/N) Σ 1{a_pre = 0 ∧ a_post = 1}
F_chance  = [(1 − ā_pre)/(k − 1)] · (1 − ā_post)
BT_chance = (1 − ā_pre) · [(1 − ā_post)/(k − 1)]
F_true    = max(F − F_chance, 0)        BT_true = max(BT − BT_chance, 0)
F_max     = max((k·ā_pre − 1)/(k − 1), 0)    BT_max = max((k·ā_post − 1)/(k − 1), 0)
```

The chance-adjusted terms are [U]. The research read these formulas through a summarising fetch. The paper's worked example (80% to 70%, adjusted "about 6%") does not match them: with k=4 and no backward transfer, they give F_chance = 0.02 and F_true = 0.08. **Checking both against the PDF is a Gate 1 blocker.** Until then, the report prints F and BT, and labels F_true "pending verification".

For open-ended benchmarks (GSM8K, HumanEval, IFEval, MGSM), the chance terms tend to 0 as k grows, so F_true = F, labelled "unadjusted" [J]. Where the option count varies by item, k is read per item. The authors warn that F_true can reflect elicitation failure rather than lost knowledge, and the report repeats that caveat.

### 2.3 Layer 3: drift

Drift is the KL divergence from the base model's next-token distribution, on fixed neutral text. It needs no judge and is the lowest-noise signal. SST uses KL(π_base ‖ π_θ) on WikiText-2 [S](https://arxiv.org/html/2610.22293):

```text
KL = (1/|T|) Σ_t Σ_v p_base(v | x_<t) · [log p_base(v | x_<t) − log p_θ(v | x_<t)]
CE_base_greedy = −(1/|T|) Σ_t log p_θ(argmax_v p_base(v | x_<t) | x_<t)
```

The default corpus is 1,000 windows of 256 tokens [J] from WikiText-2, downloaded at run time. Its licence has not been confirmed [U]. The second metric follows Kalajdzievski's cross-entropy against the base model's predictions, read here as its greedy token [J].

Drift always runs as a teacher-forced forward pass in Hugging Face Transformers, on both engines, with log-softmax in fp32. Full-vocabulary base logits are too large to cache, so the cache stores the base model's top 32 log-probabilities per position [J], plus their total mass. The cached estimator merges the remaining vocabulary into one bucket. Merging categories cannot increase KL, so this is a **lower bound** on the true KL, and the report labels it that way. `--drift exact` loads both models and computes the full sum.

### 2.4 Layer 4: safety screening

| Metric | Source | Score per item |
|---|---|---|
| Over-refusal | [XSTest](https://huggingface.co/datasets/Paul/XSTest) safe prompts | 1 if the model complies |
| Refusal on unsafe prompts | XSTest unsafe prompts | 1 if the model refuses |
| Harm refusal | [StrongREJECT](https://huggingface.co/datasets/walledai/StrongREJECT) | 1 if the model refuses |

The default detector is a refusal-phrase baseline. A local open classifier is optional: [gpt-oss-safeguard](https://huggingface.co/openai/gpt-oss-safeguard-20b), [Shieldstral](https://huggingface.co/mistralai/Shieldstral-1.0-3B) or [WildGuard](https://huggingface.co/allenai/wildguard), none of which the research validated. A cloud judge runs only on explicit opt-in. The report names the detector, labels the section a screening signal, and makes no safety claim (D-02). Benign fine-tuning raised Llama-2-7b-Chat's harmfulness rate from 0.3% to 16.1% [S](https://ar5iv.labs.arxiv.org/html/2310.03693), and no SST safety evaluation was found, so this layer is never optional in the fast suite.

### 2.5 Layer 5: behaviour

Behaviour metrics cost nothing extra, because they are read from the generations already made. Every one is given before and after:

- mean and median output tokens per benchmark, and the ratio of the two
- truncation rate: the share of generations that hit `max_new_tokens`
- empty-response rate
- think-tag rate: the share with a complete reasoning block, on thinking models
- JSON parse rate on IFEval items whose instruction requires JSON output [J]

Tool-call fidelity is not in the fast suite. It enters through user sets with the `json_schema` scorer, or through `--full`.

### 2.6 Chain metrics

For a chain of stages 1..T, R_{i,j} is the in-domain score on task j's held-out set after stage i.

```text
GEM [U]:    ACC = (1/T) Σ_{i=1..T} R_{T,i}
            BWT = (1/(T−1)) Σ_{i=1..T−1} (R_{T,i} − R_{i,i})
            FWT = (1/(T−1)) Σ_{i=2..T} (R_{i−1,i} − b̄_i)
TRACE:      BWT_t = (1/t) Σ_{i=1..t−1} (R_{t,i} − R_{i,i})          # note 1/t, not 1/(T−1)
Chaudhry:   f_j^k = max_{l ∈ {1..k−1}} a_{l,j} − a_{k,j}   for j < k
            F_k   = (1/(k−1)) Σ_{j=1..k−1} f_j^k
            I_k   = a_k^* − a_{k,k}
```

Sources are [GEM](https://ar5iv.labs.arxiv.org/html/1706.08840), [TRACE](https://ar5iv.labs.arxiv.org/html/2310.06762) and [Chaudhry et al.](https://ar5iv.labs.arxiv.org/html/1801.10112). The GEM equations were reconstructed by the retrieval tool from a garbled page, so check them against the PDF before implementing [U].

- **BWT normalisation.** The primary BWT uses GEM's 1/(T−1). The report also prints TRACE's 1/t, because OSF's figures appear to follow TRACE, and it states the normalisation beside each number.
- **FWT reference.** GEM defines b̄_i at random initialisation. tailtune substitutes the base model's score on task i and says so [J].
- **Intransigence.** I_k needs a_k^*, a reference model trained on task k alone. It is computed only when that run is linked, and is otherwise reported as "not computed".

## 3. The fast report suite

| Step | Items | Signal | Licence and gating | Est. time [U] |
|---|---|---|---|---|
| Drift | 1,000 windows | KL to base | WikiText-2, licence unconfirmed [U] | under 1 min |
| [IFEval](https://huggingface.co/datasets/google/IFEval) | 541, all | Instruction following, a key canary | Apache-2.0, open | 2 min |
| [HumanEval](https://huggingface.co/datasets/openai/openai_humaneval) | 164, all | Coding, a key canary, sandboxed | MIT, open | 1 min |
| [GSM8K](https://huggingface.co/datasets/openai/gsm8k) | 300, stratified by reference solution length [J] | Maths | MIT, open | 30 s |
| [MMLU-Pro](https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro) | 500, stratified by category | Knowledge | MIT, open | 1.5 min |
| [MGSM](https://huggingface.co/datasets/juletxara/mgsm) | 220, 20 per language | Multilingual | CC-BY-SA-4.0, open; text not redistributed | 30 s |
| XSTest | 450 rows (Inspect lists 250) | Over-refusal and refusal | CC-BY-4.0, open | 1 min |
| StrongREJECT | 324 | Harm refusal | MIT in the [official repo](https://github.com/alexandrasouly/strongreject); Hub mirror auto-gated | 1 min |
| Behaviour | from the steps above | Length, truncation, think tags, JSON | n/a | none |

The suite totals about 8 to 10 minutes per model for non-thinking models, and likely above 15 for thinking models [U]. The research's assumptions were a bf16 8B-class model in vLLM on a 24 GB GPU, at about 2,000 output tokens per second aggregate and 200 to 350 output tokens per item. A cached base halves the cost. The Hugging Face fallback and T4 GPUs are slower by an unmeasured factor [U].

**Benchmark text at run time.** tailtune ships no benchmark text. Each committed index list (section 6) names a dataset id, config, split and pinned revision. At run time, `datasets.load_dataset` downloads that revision using the user's own Hub token. StrongREJECT comes from the official MIT repository at a pinned commit, falling back to the gated mirror. The steps that follow:

1. tailtune selects the listed indices and checks each item's SHA-256.
2. A gated dataset the user has not accepted becomes "not run: gated" and links to its card.
3. A hash mismatch becomes "not run: item hash mismatch". It is never silently substituted.

Gated and share-alike text is never redistributed.

**Thinking models.** These rules apply:

- **Same mode both sides.** The template's thinking switch is held identical before and after. The default is the template's own default, recorded in the cache key.
- **Token caps.** The cap is 4,096 new tokens for thinking [J], matching SST's maximum sequence length, against per-benchmark caps of 512 to 1,024 without thinking [J].
- **Scoring.** The answer is extracted after the closing delimiter in the registry ([06](06-models.md)). A generation truncated inside the reasoning block scores 0 and counts toward the truncation rate.
- **`--cap-thinking`.** Caps thinking at 1,024 tokens. It is named this way so that it is not confused with `--suite fast`. The report then carries "thinking capped at 1,024 tokens; not comparable with uncapped runs" on every affected row.
- **Lost thinking.** If the fine-tune stops producing reasoning blocks, the think-tag row flags it as a behaviour change, because accuracy alone would hide that.

`--full` runs the full item sets plus GPQA-Diamond ([CC-BY-4.0, auto-gated](https://huggingface.co/datasets/Idavidrein/gpqa), 198 items), estimated at 1 to 3 hours [U].

## 4. The `tailtune.eval` engine

### 4.1 Any merged model from any trainer

`tailtune eval base/ ft/` takes two directories or Hub ids. This puts the people behind the Hub's 147,002 `unsloth` and 274,691 `peft` tags in the audience [S](https://huggingface.co/models-json?other=sft&withCount=true). Preconditions are checked before any GPU work. Violations are usage errors, reported with the code defined in [04](04-api-and-ux.md).

| Check | Pass | Otherwise |
|---|---|---|
| Format | safetensors only, `trust_remote_code=False` unless passed (D-08) | Refuse. GGUF is not accepted, because export fidelity is tested separately ([10](10-export-and-deployment.md)) |
| Same model | Same `model_type`, `architectures`, hidden size and layer count | Refuse. This is not a before-and-after pair |
| Tokenizer | Identical token-to-id map and special tokens | Warn if the fine-tune only appends tokens. Refuse otherwise |
| Chat template | Identical template hash | Refuse unless `--template base` or `--template ft`. The chosen template is used for both sides and recorded |
| Adapter only | `adapter_config.json` without full weights | Merge in memory with PEFT and say so. Warn if `base_model_name_or_path` names a different base |
| Generation config | Uses the base's settings for both sides, and stops on any EOS id from either | Recorded |
| Precision | Both sides run in bf16, or fp16 on T4, whatever the stored dtype [J] | Recorded |

### 4.2 Engines

A single `Engine` protocol has two implementations. `VLLMEngine` is the default. `HFEngine` is used where vLLM is unavailable, including native Windows [S](https://unsloth.ai/docs/get-started/install/windows-installation.md), MPS and CPU. Both take the same rendered prompt strings, already templated by tailtune, so no engine applies its own template. Both return the same `Sample` record: item id, prompt hash, completion text, output token count, finish reason, engine and version. Scorers consume only `Sample` records, so both engines share item lists and scorers by construction.

The rules:

- **One engine per comparison.** Base and fine-tune always use the same engine. If the base cache was built on a different engine, the base is re-run.
- **Separate process.** The engine runs in its own process, so GPU memory is fully released before training starts [J].
- **Base first.** `tailtune train` runs the base evaluation before training, so a broken setup fails before GPU hours are spent.
- **Header line.** The report prints the engine, its version and the elapsed time.

### 4.3 Generation settings

The defaults are greedy decoding (temperature 0, top-p 1), a fixed seed and no system prompt beyond the template's default. Instruct and chat models are prompted zero-shot. Base models without a chat template use few-shot prompts, as Harmon et al. did [S](https://arxiv.org/html/2510.17776v1). All settings are identical before and after, and all enter the cache key.

GPU greedy decoding is not guaranteed to be bitwise reproducible across batch shapes or engine versions [J]. Gate 2 therefore runs the base model twice with different batch orders and publishes the observed item flip rate. That is the engine-noise floor.

### 4.4 Extraction and scoring

| Benchmark | Extraction | Score |
|---|---|---|
| IFEval | Response after any reasoning block | Prompt-level strict accuracy [J] |
| HumanEval | First fenced code block, else the whole response | pass@1 in the sandbox |
| GSM8K, MGSM | Last number after the answer marker, else the last number | Exact numeric match |
| MMLU-Pro | Answer letter by regex after chain of thought | Exact letter. k read per item |
| XSTest, StrongREJECT | Response after any reasoning block | Detector in section 2.4 |

Extractors are versioned functions. A version change invalidates the cache. Scorers are tailtune code, ported from or checked against lm-evaluation-harness [S](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/interface.md) and Inspect Evals [S](https://ukgovernmentbeis.github.io/inspect_evals/), with notices kept. Each source file's licence is confirmed under D-06 before porting. A parity test runs lm-evaluation-harness with `--log_samples` and `--samples <committed indices>` and requires identical per-item scores on the same completions. tailtune never uses `--limit`, which the harness documents as "for testing only" and which takes a leading slice. Its own `--smoke` runs 20 items and stamps "SMOKE RUN: not a measurement" across the report.

### 4.5 Sandboxing HumanEval

Model-written code runs in a separate process group. It gets a temporary directory, a 10 second limit per problem [J], CPU and memory limits via `resource` on POSIX, and no inherited credentials in its environment. The process group is killed on timeout.

`--sandbox auto` picks a container when Docker is available, and otherwise the subprocess sandbox on Linux and macOS. On native Windows, or where neither is available, HumanEval is "not run: sandbox unavailable" unless the user passes `--sandbox subprocess --allow-code-execution`. The report then states that code ran without isolation.

### 4.6 The base cache

Base results live under `~/.cache/tailtune/eval/<key>/`, with per-item JSONL, drift top-k arrays and a summary. The key is the SHA-256 of this canonical JSON (design proposal):

```json
{
  "model": {"id": "str", "revision": "sha | content-hash of weights, config and tokenizer files"},
  "tokenizer_sha256": "hex", "template_sha256": "hex", "system_prompt": "str|null",
  "thinking": "on|off|template-default", "thinking_cap": "int|null",
  "suite": {"id": "fast", "version": "int",
            "benchmarks": [{"name": "str", "dataset_revision": "sha", "index_list_sha256": "hex",
                            "extractor_version": "str", "scorer_version": "str",
                            "max_new_tokens": "int", "few_shot": "int"}]},
  "drift": {"corpus_sha256": "hex", "windows": 1000, "window_tokens": 256, "top_k": 32},
  "engine": {"name": "vllm|hf", "version": "str", "dtype": "bf16|fp16", "max_model_len": "int"},
  "decoding": {"temperature": 0, "top_p": 1, "seed": "int"},
  "detector": {"name": "refusal-phrases|<model id>", "version": "str"},
  "tailtune_eval_version": "str"
}
```

A partial cache resumes benchmark by benchmark. `--no-cache` forces a fresh base run.

### 4.7 `--full` and CapTrack

`--full captrack` delegates to [CapTrack](https://github.com/thomsonreuters/captrack), which is Apache-2.0 and is invoked as `run_captrack.py --base --adapted` on merged checkpoints. It runs in its own isolated environment, like the GGUF converter. Its judge-based metrics stay off unless the user configures and opts into a judge. Its 28 benchmarks (about 19.3k samples) are estimated to take hours [U]. The project states no runtime. tailtune imports its outputs into a separate "CapTrack (full)" report section and does not reimplement it.

## 5. Statistics

**Paired bootstrap.** Draw B = 10,000 resamples [J] of item indices with replacement. For each, compute Δ_b as the mean of s_post − s_pre, then take the 2.5th and 97.5th percentiles. The aggregate resamples each benchmark independently (stratified). The seed is recorded. Drift and behaviour metrics get bootstrap intervals over windows or items.

**McNemar's exact test** applies to binary benchmarks. Let b = #(pre correct, post wrong) and c = #(pre wrong, post correct), with n = b + c. Then `p = min(1, 2 · Σ_{x=0}^{min(b,c)} C(n,x) · 0.5^n)`, and p = 1 when n = 0.

**Holm correction** runs over the m binary benchmarks in the report, user sets included, at α = 0.05. Sort the p-values ascending, reject H_(i) while `p_(i) ≤ α/(m − i + 1)`, and stop at the first failure. The report prints m. Drift and behaviour are descriptive and are not tested.

**Non-inferiority margin.** δ = 2 points by default [J], set with `--margin`. A benchmark is non-inferior when its paired 95% lower bound exceeds −δ. The margin is resolved into `tailtune.yaml` before training. If a later `tailtune eval` uses a different margin, the report says "margin changed from ‹a› to ‹b› after training".

**Resolution.** The report states each benchmark's resolution as its paired half-width h = (u − l)/2, rounded up to one decimal. Planning half-widths at 95%, assuming about 80% accuracy, where d is the share of items whose correctness differs between pre and post [C]:

| N | Single score | Unpaired difference | Paired d=5% | Paired d=10% | Paired d=20% |
|---|---|---|---|---|---|
| 164 | ±6.1 | ±8.7 | ±3.4 | ±4.8 | ±6.8 |
| 300 | ±4.5 | ±6.4 | ±2.5 | ±3.6 | ±5.1 |
| 541 | ±3.4 | ±4.8 | ±1.9 | ±2.7 | ±3.8 |
| 1,000 | ±2.5 | ±3.5 | ±1.4 | ±2.0 | ±2.8 |
| 2,000 | ±1.8 | ±2.5 | ±1.0 | ±1.4 | ±2.0 |

These use `1.96·√(p(1−p)/N)`, `√2` times that, and `1.96·√(d/N)`. A fast report can detect large regressions and cannot certify their absence. Miller's error-bars paper is the reference to read before this section is final. Only its abstract was read [U](https://arxiv.org/abs/2411.00640).

**Seeds.** A user run has one training seed, and the report says so. Published results use three training seeds, each shown with its own interval, plus the mean and range. No pooled interval is computed over three seeds [J].

**Wording rules.** Every benchmark sentence comes from exactly one of these templates:

| State | Condition | Template |
|---|---|---|
| Regression | Holm-adjusted p < 0.05 and Δ < 0 | "Regression on ‹b›: −‹d› points [‹l›, ‹u›], significant after Holm correction over ‹m› benchmarks (‹bin›)." |
| Improvement | Holm-adjusted p < 0.05 and Δ > 0 | "Improvement on ‹b›: +‹d› points [‹l›, ‹u›]." |
| Non-inferior | Not significant, and l > −δ | "No regression detected on ‹b› at ±‹h› points; the lower bound ‹l› is inside the −‹δ›-point margin." |
| Inconclusive | Not significant, and l ≤ −δ | "No regression detected on ‹b› at ±‹h› points. This item set cannot rule out a drop of ‹δ› points." |
| Not run | Section 8.5 | "‹b› was not run: ‹reason›." |

Allowed elsewhere: "The worst benchmark was ‹b› at −‹d› points." "N of M steps ran." "One training seed; training noise is not measured."

Forbidden anywhere, including model card blocks:

- "no forgetting" and "without forgetting"
- "retained all capabilities"
- "safe" or "safety preserved"
- "equivalent to the base"
- "not significant, so nothing was lost"
- "retention N%" with no benchmark named
- any sentence that cites the aggregate without the worst benchmark beside it

A unit test greps the rendered report for these phrases.

## 6. Contamination controls

1. **Exact match.** Normalise each text with NFKC, casefolding and collapsed whitespace, then compare SHA-256 hashes of each eval item's prompt and reference against each training example's user turns, assistant turns and full rendered text.
2. **Near-duplicates.** Use MinHash over word 5-gram shingles with 128 permutations and LSH, flagging Jaccard ≥ 0.8 [J]. The MinHash code must pass D-06's licence rule.
3. **Containment.** Build an index of every eval item's word 13-grams, which is cheap because the suite is small. Flag a training example that contains at least 50% of an item's 13-grams [J]. This catches short items embedded in long documents, which Jaccard misses.
4. **Reporting.** The report prints overlap counts per benchmark. When any are non-zero, it shows Δ both on all items and with flagged items excluded. The cache stays valid, because exclusion is computed from item ids.
5. **Known adjacency warnings** fire on the training dataset's Hub id or on overlap hits:

   | Training data | Warn about |
   |---|---|
   | MedMCQA or medical QA | MMLU and MMLU-Pro medical subjects |
   | Legal sets (BarExam QA, LegalBench) | MMLU professional law |
   | Code sets | HumanEval and MBPP |
   | Reasoning traces or maths sets | GSM8K and MGSM |

6. **Committed index lists.** `src/tailtune/eval/suites/fast-v1/<benchmark>.json` holds the dataset id, config, split, pinned revision, sampling rule and seed, row indices, per-item SHA-256 and a list hash. It holds no text. Lists change only with a suite version bump.
7. **Fixed protocol.** Items, template, system prompt, thinking flag and extractor are fixed before training. Users tune only on validation splits, never on eval items.

## 7. User-supplied retention sets

A general suite misses task-family losses, as the translation study shows [S](https://arxiv.org/abs/2609.28395). Users add sets with `--retention-set name=path.jsonl`, which can repeat, or under `eval.retention_sets` in `tailtune.yaml`. Format (design proposal), one JSON object per line:

```json
{"id": "q-0001",
 "messages": [{"role": "user", "content": "..."}],
 "scorer": "exact | normalized_exact | numeric | regex | choice | contains | json_schema",
 "target": "...", "choices": ["A", "B"], "answer": "B", "pattern": "...", "schema": {}}
```

`prompt` may replace `messages`. Fields unused by the chosen scorer are ignored. A `python` scorer runs only with `--allow-user-scorer`, inside the HumanEval sandbox.

The file's SHA-256 enters the cache key. Each set appears as `user:<name>` in its own report section, joins the Holm family, competes for "worst benchmark" and can be gated. Below 100 items the report adds the set's half-width in bold. In a task chain, each earlier stage's held-out set is added automatically (section 9).

## 8. Outputs

### 8.1 `FORGETTING_REPORT.md` template

Illustrative only. Every number is a placeholder, and no evaluation has been run.

````markdown
# Forgetting report

**Run** runs/run-NNN · **Base** org/model@<commit> · **Fine-tune** runs/run-NNN/model
**Method** sst (SST (Chavan et al., 2026)), r=0.99, K=None · **Coverage** NN.N% protected, N.N% free, NN.N% frozen (floor NN%)
**Suite** fast v1 · **Engine** vllm ‹version› · **Elapsed** NN min · **Seeds** 1 (training noise not measured)
**Steps run** N of M · **Margin** δ = 2.0 points · **Holm family** m = N

## Headline
Worst benchmark: **‹benchmark› −D.D points [−L.L, −U.U]** (‹bin›).
Aggregate capability delta: −D.D points [−L.L, +U.U].
‹One sentence per regression, from the wording templates.›

| Benchmark | Items | Base | Fine-tune | Δ points [95% CI] | Resolution | Bin | F | BT | F_true | p (Holm) | Statement |
|---|---|---|---|---|---|---|---|---|---|---|---|
| IFEval | 541 | NN.N | NN.N | −D.D [−L.L, +U.U] | ±H.H | Minor | N.N | N.N | pending | 0.NN | Inconclusive |
| HumanEval | 164 | NN.N | NN.N | ... | ±H.H | ... | ... | ... | unadjusted | ... | ... |
| GSM8K | 300 | ... |
| MMLU-Pro | 500 | ... |
| MGSM | 220 | ... |
| XSTest over-refusal | ‹n› | ... |
| StrongREJECT refusal | 324 | ... |
| user:‹name› | ‹n› | ... |

Groups (TRACE style): General −D.D · Instruction following −D.D · Safety −D.D

## Drift
KL(base ‖ fine-tune), lower bound from cached top-32: 0.NNN [0.NNN, 0.NNN] · CE to base greedy: N.NN (base N.NN)

## Safety screening (not a safety evaluation)
Detector: refusal phrases v‹n›. Over-refusal ‹b›→‹a›%. Harm refusal ‹b›→‹a›%. No safety claim is made.

## Behaviour
| Benchmark | Mean tokens (base → ft) | Truncated | Empty | Think-tag rate | JSON parse |
|---|---|---|---|---|---|

## Contamination
Exact N · near-duplicate N · containment N. Adjacency warnings: ‹list or none›.

## Not run
| Step | Reason | Effect |
|---|---|---|
| HumanEval | sandbox_unavailable | Excluded from worst benchmark and aggregate; gate humaneval cannot be evaluated |

## What this report cannot show
‹Fixed paragraph from section 12.›

## Reproduce
`tailtune eval org/model@<commit> runs/run-NNN/model --suite fast --seed N` · cache key ‹hex› · lineage.json
````

### 8.2 `forgetting_report.json`

```json
{
  "schema_version": 1, "tailtune_version": "str", "created_utc": "iso8601",
  "base": {"id": "str", "revision": "str"}, "finetune": {"path": "str", "content_sha256": "hex"},
  "run": {"lineage": "path|null", "method": "str", "r": "str|null", "coverage": "float|null",
          "seeds": "int"},
  "suite": {"id": "str", "version": "int"}, "engine": {"name": "str", "version": "str"},
  "cache_key": "hex", "margin_points": "float", "holm_m": "int",
  "steps": [{
    "name": "str", "group": "general|instruction|safety|user|drift", "status": "ran|not_run",
    "not_run_reason": "str|null", "items": "int", "k": "int|null",
    "base_score": "float", "ft_score": "float", "delta_points": "float",
    "ci95": ["float", "float"], "half_width": "float", "rel_percent": "float|null",
    "bin": "None|Minor|Moderate|Major|Catastrophic|n/a",
    "F": "float", "BT": "float", "F_true": "float|null", "BT_true": "float|null",
    "F_true_status": "pending_verification|verified|unadjusted",
    "mcnemar": {"b": "int", "c": "int", "p": "float", "p_holm": "float"},
    "statement": "regression|improvement|non_inferior|inconclusive|not_run",
    "sentence": "str", "contamination": {"exact": "int", "near": "int", "contained": "int"},
    "samples_path": "str"
  }],
  "drift": {"kl_lower_bound": "float", "kl_ci95": ["float", "float"], "exact": "bool",
            "ce_base_greedy": {"base": "float", "ft": "float"}},
  "behaviour": [{"step": "str", "mean_tokens": ["float", "float"], "truncated": ["float", "float"],
                 "empty": ["float", "float"], "think_tag": ["float|null", "float|null"],
                 "json_parse": ["float|null", "float|null"]}],
  "aggregate": {"delta_points": "float", "ci95": ["float", "float"]},
  "worst": {"step": "str", "delta_points": "float", "ci95": ["float", "float"]},
  "gate": {"spec": "str|null", "mode": "estimate|confident|strict", "passed": "bool|null",
           "failures": [{"key": "str", "limit": "float", "observed": "float", "reason": "str"}]},
  "chain": "object|null"
}
```

### 8.3 `--max-regression`

`--max-regression ifeval=3,humaneval=5,worst=5,aggregate=2,user:faq=3`. The keys are step names, `user:<name>`, `aggregate`, `worst` and `all`, which applies to every step. Values are in points. The gate is evaluated after all outputs are written. Any failure exits with **code 3**. Within `tailtune train`, the trained model is still saved, and `run.retention_report().gate.passed` is `False`.

| `--gate-mode` | A key fails when |
|---|---|
| `estimate` (default) [J] | Δ < −limit |
| `confident` | upper CI bound < −limit, meaning it is confident the drop exceeds the limit |
| `strict` | lower CI bound < −limit, a non-inferiority check |

When a step's half-width exceeds its limit, the report warns: "this gate cannot tell a ‹limit›-point drop from noise on ‹N› items". A gated step that did not run fails the gate unless `--allow-missing` is passed [J]. Fail closed.

### 8.4 Model card summary block

`--push-card` writes this between markers, replacing any earlier block (illustrative, placeholders):

```markdown
<!-- tailtune:forgetting-report:start -->
**Forgetting report (tailtune ‹version›, fast suite v1, 1 training seed).**
Base org/model@<commit>. Method ‹method›, coverage NN.N% protected.
Worst benchmark: ‹benchmark› −D.D points [−L.L, −U.U]. Aggregate −D.D points.
N of M steps ran. Safety screening only; no safety claim. [Full report](FORGETTING_REPORT.md)
<!-- tailtune:forgetting-report:end -->
```

### 8.5 Steps that could not run

Every step ends as `ran` or `not_run`, with one of these reasons:

- `sandbox_unavailable`
- `gated_access_denied`
- `download_failed`
- `item_hash_mismatch`
- `engine_error`
- `timeout`
- `template_mismatch`
- `skipped_by_user`
- `out_of_memory`

A step that did not run is excluded from the aggregate and from the worst-benchmark search. The headline always reads "N of M steps ran". If drift and every capability step failed, the report says "No retention measurement was produced", and `tailtune train` exits 1. `--eval none` writes a report that says only "Evaluation disabled by the user."

## 9. Task chains and lineage

`lineage.json` links each run to its previous run. `tailtune eval --chain runs/run-003` walks back to the original base and builds the R matrix from cached evaluations, keyed by each model's content hash. Each stage's held-out in-domain set becomes a retention set for later stages automatically.

The chain report shows:

1. the fast suite against the **original base** for every stage, not only the previous one
2. ACC, GEM BWT, TRACE BWT and Chaudhry F_k, with normalisations stated
3. each stage's method, r and coverage
4. intransigence where a single-task reference run is linked

SST's sequential experiments cover only two- and three-stage chains [S](https://arxiv.org/html/2610.22293). Longer chains are labelled "beyond published evidence". Tail capacity across stages is described in [03](03-methods.md).

## 10. The public benchmark

### 10.1 Preregistration

Before any run, a dated commit in the benchmark repository fixes:

- hypotheses
- models, domains, dataset revisions and hashes
- arms, conditions, seeds and the trial budget
- the primary endpoint: aggregate retention Δ and the worst benchmark at the matched-learning checkpoint [J]
- secondary metrics
- the learning gate
- exclusion rules: a diverged run is repeated once with the next seed, and both are reported
- a promise to publish every result

Changes after that are dated commits labelled "deviation", and they are listed in the paper.

### 10.2 Models and domains

- **Core models:** Qwen3-8B, Gemma-4-E4B-it and Granite-4.2-8b.
- **Stretch:** Llama-3.1-8B-Instruct, Qwen3.8-27B and gpt-oss-20b, after FSDP2.
- **Expected-loss row:** Qwen3-1.7B, where SST loses 8.0 points [S](https://arxiv.org/html/2610.22293).
- **Thinking mode:** off for all models in the benchmark [J], for comparability across families. The Gate 1 replication cell uses thinking, as the paper did.

| Domain | Train | In-domain test | Licence | Note |
|---|---|---|---|---|
| Medical | [MedMCQA](https://huggingface.co/datasets/openlifescienceai/medmcqa) | Held-out split | Apache-2.0 | Direct SST comparison. MMLU medical adjacency |
| Legal | [BarExam QA](https://huggingface.co/datasets/reglab/barexam_qa) or [LegalBench](https://huggingface.co/datasets/nguha/legalbench) | Held-out split | CC-BY-SA-4.0 / CC-BY-4.0 | MMLU law adjacency |
| Coding | [Self-OSS-Instruct](https://huggingface.co/datasets/bigcode/self-oss-instruct-sc2-exec-filter-50k) | Held-out split, execution filter | ODC-By | Not HumanEval or MBPP |
| Text-to-SQL | [Gretel text-to-SQL](https://huggingface.co/datasets/gretelai/synthetic_text_to_sql) | SQL execution match | Apache-2.0 | Format fidelity |
| Non-English | [Aya dataset](https://huggingface.co/datasets/CohereLabs/aya_dataset), one or two languages | Native held-out | Apache-2.0 | Per-language volume unchecked [U] |

Licences are Hub card tags. Upstream licensing and suitability need a closer look before preregistration.

### 10.3 Arms and conditions

| # | Arm | Purpose |
|---|---|---|
| 1 | Base (no training) | Perfect retention, zero learning. Makes the gate visible |
| 2 | SFT, standard full fine-tune | Reference for forgetting and for the gate |
| 3 | LoRA r16 | What most users run |
| 4 | **Capacity-matched LoRA**, rank t per matrix (41 at width 4096), on SST's targets | No-refresh SST's per-matrix update also has rank ≤ t. This is the first control a sceptic asks for. Close prior art is [MiLoRA](https://arxiv.org/abs/2406.09044) |
| 5 | Freeze (last N blocks) | Cheap baseline |
| 6 | OSF via PEFT at its defaults, versions cited | The shipped alternative |
| 7 | [OPLoRA](https://arxiv.org/html/2510.13003v2) at k=0.99d | Matches SST's own comparison table |
| 8 | SST r=0.99, no refresh | Paper default |
| 9 | SST r=0.99 with R100, plus SDFT+SST on one domain | Refresh is labelled unverified (D-10) |

Every arm uses identical target sets and prints its `inspect` coverage, because unequal coverage is the obvious confound. There are two conditions, and both are always reported:

- **Equal-setting:** one shared recipe for every arm, SST's published one, as SST did [S](https://arxiv.org/html/2610.22293). This disadvantages LoRA, and the table says so.
- **Tuned:** each method gets 4 trials [J] over a preregistered learning-rate grid, chosen by validation-split in-domain score only. SST's authors note the constrained optimum "need not equal that of unconstrained SFT".

Seeds 0, 1 and 2 [J] are used throughout, and evaluation is greedy.

### 10.4 The learning gate

```text
LR_m = (ID_m − ID_base) / (ID_SFT − ID_base)
```

- **Inputs.** ID_SFT is the mean final in-domain score of the SFT arm over its seeds. In-domain scores are evaluated at every 10% of training steps [J].
- **Matched checkpoint.** Each run is compared at the first checkpoint where LR_m ≥ 0.9, and the full retention suite is evaluated there and at the final checkpoint.
- **Did not learn.** A run that never reaches 0.9 is listed as "did not learn" and is not ranked.
- **SFT did not learn.** If SFT's in-domain gain is not significant (its paired CI includes 0), the domain gets no gate and no ranking.
- **Pareto plot.** In-domain score against aggregate retention accompanies every table.

Forgetting tracks fine-tuning loss [S](https://arxiv.org/html/2401.05605v1), so this pairing is not optional. Where SST's in-domain score is below SFT's, as in the paper's own Pathology row (80.4 against 77.3), it is shown plainly.

### 10.5 Cost

The research derived these from SST's measured 1.73 s per step for full-parameter bf16 Qwen3-8B fine-tuning on eight L40S GPUs at batch 16 [S](https://arxiv.org/html/2610.22293). That is a standard fine-tuning step rate. The cost tables assume:

- the same step rate for every arm and model, with LoRA arms really cheaper
- **SST arms cost an unmeasured extra**, which Gate 1 measurement 2 settles. The paper's overhead figures are not used (D-02)
- tailtune's own backend may run at a different speed from the paper's setup [U]
- SVD, storage and failed runs are excluded

**First release, 90 runs:**

| Item | Value |
|---|---|
| Design | 2 models (Qwen3-8B and Gemma-4-E4B-it, with Granite-4.2-8b as the fallback) [J] × 3 domains (medical, coding, text-to-SQL) [J] × 5 trained arms (2, 3, 4, 6, 8) × 3 seeds. Base is evaluated once per model |
| Steps per run | 2,000 examples × 3 epochs / 16 = 375 |
| Wall per run on 8×L40S | 0.18 h, which is 1.44 GPU-h [C] |
| Training | about 130 GPU-h [C] |
| Evaluation | 90 to 270 GPU-h at 1 to 3 GPU-h per full-suite evaluation [U]. Evaluating both the final and the matched checkpoint can double this [C] |
| Money | roughly $300 to $700 [C], at listed RTX PRO 6000 or L40S training rates and L4 evaluation rates ([RunPod](https://www.runpod.io/pricing), [Lightning](https://lightning.ai/pricing)), assuming equal GPU-hours across those cards |

The first release runs the equal-setting condition. A sensitivity check outside the 90 runs costs about 17 GPU-h [C]: three learning rates for SFT and SST, on one domain, one seed and both models.

**Full plan, 405 runs** (3 models × 5 domains × 9 arms × 3 seeds):

| Scenario | Steps per run | Wall on 8×L40S | GPU-h per run | Training GPU-h [C] |
|---|---|---|---|---|
| Lean: 2,000 × 3 epochs | 375 | 0.18 h | 1.44 | about 583 |
| Mid: 5,000 × 2 epochs | 625 | 0.30 h | 2.4 | about 972 |
| Full: 9,000 × 5 epochs | 2,812 | 1.35 h | 10.8 | about 4,374 |

- **Evaluation** adds 405 to 1,215 GPU-h [U].
- **Base arm.** Arm 1 needs no training, so these figures are on the high side.
- **Tuned condition.** It multiplies training on arms 2 to 9 by up to 4. At the lean size, that is up to about 2,074 GPU-h for those 360 runs [C].
- **No currency figure** is computed for the full plan [U].

### 10.6 Publication, repro and outside runs

- **Publication.** A tech report, a Hugging Face Space and a results dataset. Each cell includes per-item JSONL, configs, base commit hashes, dataset hashes, seeds, `targets.json`, `lineage.json` and training logs. Merged models are released for some cells where the base licence permits.
- **Tables** follow the layout below (symbols only) and add a severity bin per cell. Sponsors are disclosed beside any result featuring them (D-25).
- **One-command repro.** `uv run repro.py qwen3-8b/medical/sst-r0.99/seed-0` in the benchmark repository pins tailtune and `uv.lock`, and calls `tailtune train --config cells/<cell>/tailtune.yaml` then `tailtune eval`. `repro.py --rescore <cell>` re-scores the published per-item logs with no GPU.
- **Outside submissions** arrive as pull requests to the results dataset. Each carries results files, configs, `lineage.json`, per-item JSONL and `tailtune doctor` output, with DCO sign-off. CI re-scores the logs automatically. Rows are labelled "external, unverified" until a maintainer reruns one cell, after which they read "external, reproduced".

| Model / task | Method | ID (Δ vs base) | LR | Aggregate Δ [95% CI] | Worst benchmark | F_true | BT_true | IFEval Δ | HumanEval Δ | Over-refusal Δ | Harm refusal Δ | Multilingual Δ | KL | Coverage | Seeds |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ‹model› / ‹domain› | ‹arm› | x (+y) | 0.NN | −z [a, b] | ‹name› −w | … | … | … | … | … | … | … | … | NN% | 3 |

### 10.7 Where tailtune is expected to lose

These rows are reported, not hidden:

- **Small models with narrow tails.** Qwen3-1.7B (SST −8.0) and 4B (−3.6).
- **Tasks that need large behaviour changes.** Qwen2.5-7B-Instruct needed r=0.90 on Science, because the r=0.99 tail was too small [S](https://arxiv.org/html/2610.22293).
- **Domains where constrained in-domain scores trail SFT.** Examples are Pathology and FinQA at r=0.99.
- **Cheaper matches.** Capacity-matched LoRA may match SST at matched learning for far less memory.
- **Refresh.** It costs prior capability.
- **Safety.** It may drift on any arm.

### 10.8 If SST fails to replicate

1. **Find the source.** Separate implementation error from non-replication by checking against the Gate 1 reference run of the public repository. If the reference reproduces and tailtune does not, that is a tailtune bug, and it is fixed before anything is published.
2. **Contact the authors.** If neither reproduces, send them the per-item logs and configs, and give them 14 days [J] to respond before publication.
3. **Publish the disagreement** with the exact variant named: "SST (Chavan et al., 2026) as implemented in tailtune ‹version›, r=0.99, no refresh, did not reproduce ‹cell›".
4. **Update the product.** Update `method="auto"` and the README, and keep the report and data flow unchanged ([01](01-product-strategy.md)).

## 11. Leaderboard

The leaderboard is a Gate 4 item, built as a Hugging Face Space over a results dataset rather than waiting to join an existing board. Existing boards rank downloads [S](https://theopenweights.com/leaderboards/most-downloaded).

- **Rows.** Each row is a (base, fine-tune, method) triple scored with a named suite version and engine.
- **Columns** align with Harmon's F_true (once verified) and CapTrack's bins.
- **Sorting.** The default sort is the worst benchmark, never the mean.
- **Two tables.** Benchmark rows carry a learning ratio. Third-party fine-tunes have no known in-domain score, so they appear in a separate unranked table of per-benchmark deltas, which evaluates models, not their authors.
- **No paid placement** (D-25).
- **Outreach.** Approach the Harmon and CapTrack authors, whose tools can serve as a second opinion.

## 12. Honest limits

The report cannot show:

- **That nothing was lost.** At ±2 to ±5 points per benchmark, it detects large regressions only.
- **Capabilities outside the suite.** These include tool use, long context, multi-turn dialogue, most languages, domain knowledge the user cares about unless supplied, and vision or audio inputs.
- **Safety.** Refusal-phrase screening is not a safety evaluation.
- **Knowledge versus elicitation.** It cannot separate knowledge loss from changes in elicitation or formatting. Extraction choices move results [S](https://arxiv.org/html/2510.17776v1).
- **Training noise, in a one-seed user run.**
- **Results across engines.** vLLM and Hugging Face numbers are not comparable with each other. Only before and after on the same engine are.
- **All contamination.** Paraphrases below the thresholds pass unflagged.
- **Unseen futures.** Benchmark items are public, so future base models may have trained on them.

## Implementation checklist

**Gate 0, Claim**
- [ ] Keep README and description wording within section 5's rules ([01](01-product-strategy.md)).

**Gate 1, Verify**
- [ ] Check Harmon's F_true formulas and worked example against the PDF. This blocks implementing F_true.
- [ ] Check the GEM, Chaudhry and TRACE formulas against the PDFs.
- [ ] Read Miller's paper in full and revise section 5 if needed.
- [ ] Measure fast-suite time per step on one 24 GB GPU and one T4, then replace the [U] estimates.
- [ ] Confirm the licences of WikiText-2, LiveBench, HellaSwag and WinoGrande, and the XSTest, HarmBench and AdvBench item counts.

**Gate 2, v0.1 Door**
- [ ] Build `tailtune.eval`: the engine protocol, vLLM and Hugging Face engines in a subprocess, scorers with parity tests, the sandbox, drift, the cache and committed index lists.
- [ ] Add `tailtune eval base/ ft/` with every precondition check.
- [ ] Generate statistics, wording templates, both report files, the gate with exit code 3, and the card block.
- [ ] Add the contamination scan and user retention sets.
- [ ] Publish the engine-noise flip rate.
- [ ] Publish one `eval` result on a popular public fine-tune as the demand test ([13](13-roadmap.md)).

**Gate 3, v0.2 Proof**
- [ ] Commit the preregistration, run the 90 runs, apply the learning gate and publish with losses included.
- [ ] Release the benchmark repository with `repro.py` and `--rescore`.

**Gate 4, v0.3 Reach**
- [ ] Ship the leaderboard Space, the CapTrack `--full` backend, the chain report and the 4-bit `tail-lora` rows.

**Gate 5, v1.0**
- [ ] Run the full 405-run plan if funded, evaluate DPO and GRPO, and freeze suite v1.

## Acceptance tests

1. A tiny model passed as both base and fine-tune, scored by the Hugging Face engine on CPU, gives Δ = 0, F = BT = 0, p = 1 and KL = 0 on every step, and the statement "Non-inferior" everywhere. CPU is used so the test is deterministic. On GPU the residual is the engine-noise flip rate from section 4.3.
2. A synthetic fine-tune that fails 20% of HumanEval items the base passed, with everything else unchanged, shows HumanEval as the worst benchmark, a "Regression" statement and gate exit code 3 under `humaneval=5`.
3. The same items and completions scored through the vLLM and Hugging Face code paths give identical per-item scores.
4. The scorer parity test matches lm-evaluation-harness per item, on logged samples, for IFEval and GSM8K.
5. A second run with an unchanged base hits the cache, and changing any field in section 4.6 misses it.
6. On a machine without Docker on native Windows, HumanEval reports `sandbox_unavailable`, the headline reads "N of M steps ran", and a HumanEval gate fails without `--allow-missing`.
7. A tokenizer or template mismatch is refused before any GPU work.
8. Training data containing a GSM8K test item verbatim is flagged by the exact check, and an item embedded in a long document is flagged by the containment check.
9. The forbidden-phrase test passes on every rendered fixture report.

## Open questions

1. Do Harmon's chance-adjusted formulas hold as summarised? This is a Gate 1 check.
2. Is the 2-point margin and the `estimate` gate default right for users, given half-widths of ±2 to ±5 points?
3. Which detector should be the default for safety screening? None of the candidates was validated.
4. Should the benchmark use thinking mode on Qwen3? It raises cost and changes response length.
5. Is Gemma-4-E4B-it ready to be the second first-release model by preregistration time, or should Granite-4.2-8b take its place?
6. What exit code should an incomparable `eval` input pair get? The README's table has none, so this defers to [04](04-api-and-ux.md).
