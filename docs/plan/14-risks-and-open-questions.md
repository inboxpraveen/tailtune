# 14. Risks and open questions

**What this document decides:**

- Which risks could invalidate the plan.
- The early signal that would reveal each risk, and the planned response.
- What the research could not establish.

Likelihood and impact ratings are judgements [J]. Each risk has a gate where it must be checked again.

## 1. Risk register

### Evidence and science

| # | Risk | Likelihood | Impact | Early signal | Mitigation | Check at |
|---|---|---|---|---|---|---|
| R1 | SST does not replicate, or replicates only on long-trace thinking models | Medium | High | Gate 1 reference run misses Table 5 by more than its own variance | Publish either way. The report stays the product, and the method is swappable ([01](01-product-strategy.md#9-what-happens-if-the-evidence-goes-against-us)) | Gate 1 |
| R2 | SST's real overhead is far above the paper's figures. The 26.2 ms projection claim implies about 4.2 peta multiply-adds per second under the head form [C], and the refresh claim sits 40 to 60 times below other reported SVD costs | High | Medium | Gate 1 timing | Use the tail form. Quote only measured numbers. Limit `auto` to sizes where SST is measured to fit | Gate 1 |
| R3 | Every memory budget is wrong, because all are computed and none measured | Medium | High | Gate 1 peak-memory runs on 0.3B and 1.7B | Calibrate `estimate`. Show its uncertainty until calibrated | Gate 1 |
| R4 | Free-tier users get the weakest protection. Models that fit free GPUs are where SST is weakest: −8.0 at 1.7B against −0.9 at 8B [S](https://arxiv.org/html/2610.22293) | Certain | Medium | Already known | Say so plainly. Free tier means `lora` plus the report. Benchmark `tail-lora` | Gate 2 |
| R5 | `tail-lora` on a 4-bit base gives no real retention, because NF4 error may swamp the tail of the spectrum | Unknown | Medium | Gate 4 benchmark | Label it experimental. Publish the result as new evidence either way | Gate 4 |
| R6 | A fast report raises false alarms, or misses real damage. IFEval's 541 items resolve only about ±2.7 points paired [C] | Medium | Medium | Seed-to-seed variance in Gate 3 | Paired statistics, Holm correction, the "no regression detected at ±X" wording, and user-supplied retention sets | Gate 3 |
| R7 | Fine-tuned models become less safe, and tailtune is blamed. Benign fine-tuning raised one model's harmfulness rate from 0.3% to 16.1% [S](https://ar5iv.labs.arxiv.org/html/2310.03693) | Medium | High | Refusal metrics in reports | Make no safety claim (D-02). Always report refusal and over-refusal | Every release |

### Engineering

| # | Risk | Likelihood | Impact | Early signal | Mitigation | Check at |
|---|---|---|---|---|---|---|
| R8 | Silent training failures from three known traps: bf16 rounding away projected steps, the hook firing before DeepSpeed's real step, and reentrant checkpointing under FSDP2 dropping gradients | High if untested | High | Invariant tests | Locked rules D-12, D-13 and D-19. Frozen-subspace invariant tests on every backend | Gate 2 |
| R9 | Dense FSDP2 projection is harder than designed, since no public implementation exists | Medium | Medium | 2-rank gloo prototype on CPU | Treat it as new engineering. Single GPU on a B200 is the fallback for 8B | Gate 3 |
| R10 | Upstream churn breaks tailtune weekly. Transformers shipped 48 releases and TRL 32 in 2026 so far [S](https://pypi.org/pypi/transformers/json) | Certain | Medium | Weekly compatibility job | Tested ranges, `uv.lock`, nightly against Transformers `main` | Weekly |
| R11 | GGUF export breaks. llama.cpp's converter pins `transformers==4.57.6` and `torch==2.11.0`, and whether it reads v5-written tokenizers is untested | Medium | High | First CI logit round trip | An isolated conversion environment, plus the round trip test as a release gate | Gate 2 |
| R12 | Platform floors move. `torch_xla` will be archived on 2026-11-15, FSDP1 is deprecated, and CUDA 13 wheels need driver 580+ | Certain | Low | Release notes | Already in the Unsupported tier. Colab's driver is unverified [U] | Each torch release |
| R13 | A supply-chain compromise like LiteLLM's malicious 1.82.7 and 1.82.8 [S](https://www.netspi.com/blog/executive-blog/ai-ml-pentesting/litellm-supply-chain-compromise/) | Low | Severe | Scorecard alerts | Trusted Publishing, SHA-pinned Actions, a reviewer gate, and no scanner near the credentials (D-08) | Gate 0 |

### Product and competition

| # | Risk | Likelihood | Impact | Early signal | Mitigation | Check at |
|---|---|---|---|---|---|---|
| R14 | Nobody wants a forgetting report. Demand is untested | Unknown | Severe | Response to the first published `eval` result | Run the demand test at Gate 2, and re-evaluate the thesis at M2 | Gate 2, M2 |
| R15 | Nobody wants Excel input. GitHub shows weak signals only | Unknown | Low | Usage questions, issues | The feature is cheap, so keep it. Lead with `eval` if Excel draws no interest | Gate 3 |
| R16 | Unsloth or PEFT ships a retention report or SST | Medium | Medium | Their release notes | The moat is independence, the benchmark corpus and `eval` on anyone's model. Credit them and compare in the benchmark | Ongoing |
| R17 | An overreaching claim damages trust before launch. The current README already contains three | High if unchanged | High | Today | Apply the [01](01-product-strategy.md#7-what-to-change-now) rewrite at Gate 0 | Gate 0 |

### Legal and licensing

These are engineering readings, not legal advice.

| # | Risk | Likelihood | Impact | Early signal | Mitigation | Check at |
|---|---|---|---|---|---|---|
| R18 | Code contamination. SST's repo has no licence, Unsloth mixes AGPL into Apache-labelled files, and mini_trainer's licence file and metadata disagree (Apache-2.0 versus MIT) | Medium | High | Licence checks in review | Implement SST from the CC BY 4.0 paper. Never vendor AGPL, LGPL or unlicensed code. Keep both mini_trainer notices (D-06) | Every PR |
| R19 | Shared bases or published deltas count as derivative works of gated or restrictively licensed weights | Unknown | High | Legal review | Start Hub sharing only with Apache-2.0 models, and get legal review before covering Llama and similar | Before any Hub sharing |
| R20 | Benchmark text cannot be redistributed. Some sets are gated or share-alike | Medium | Medium | Licence read per set | Download at run time, and commit only index lists and hashes | Gate 2 |
| R21 | Name conflict. No trademark clearance search has been run [U] | Low | Medium | Clearance search | Run one before paid promotion. Defer filing until traction | M2 |

### Sustainability

| # | Risk | Likelihood | Impact | Early signal | Mitigation | Check at |
|---|---|---|---|---|---|---|
| R22 | The maintainer burns out, with a bus factor of 1 | High | Severe | Response times, missed weekly jobs | Narrow scope (D-03). Recruit the second maintainer (D-24) | M1 |
| R23 | Donations do not sustain the project. Unsloth has 13 current GitHub sponsors at 79k stars [S](https://github.com/sponsors/unslothai) | High | Medium | Sponsor counts | Ask for compute first. Follow the ladder in [12](12-community-and-sponsorship.md) | M2 |
| R24 | Programmes are closed or ineligible for an individual in India, and the tax treatment is unknown | Medium | Medium | Applications | Use GitHub Sponsors, which is officially open in India. Get a chartered accountant to check GST, FCRA and remittance | M1 |

## 2. Open questions

### Science

1. Does SST's protection hold outside Qwen3 and outside thinking mode? Instruct models show weaker gains, about −5 on Qwen2.5-7B-Instruct [S](https://arxiv.org/html/2610.22293).
2. How much learning flows through unconstrained embeddings and norms, and does freezing them hurt in-domain scores?
3. How does gradient clipping on the full gradient interact with the projected step? This has not been studied.
4. How long can a sequential chain run before tail capacity runs out? SST's published chains have only two and three stages.
5. Do SST or OSF retain anything under DPO, ORPO, KTO or GRPO? No paper evaluates this.
6. What learning rate is right under the constraint? The authors say it "need not equal that of unconstrained SFT".

### Questions for the SST authors

1. Can the repo get a licence?
2. Is SST meant to run on ZeRO-3 or ZeRO-2? The paper says ZeRO-3 and the repo maps SST to ZeRO-2.
3. Will the refresh and Muon code be released?
4. Tables 3 and 4 report 78.9 and 80.0 for what looks like the same setting. Which is correct?
5. How was the 26.2 ms projection time measured?

### Engineering

1. Does Modal's free credit cover multi-GPU apps? [U]
2. What are Kaggle's current quotas, RAM and terms of use for CI? [U]
3. Does bitsandbytes 8-bit Adam work on bf16 parameters under FSDP2 or DeepSpeed? [U]
4. Is the host-offload route on T4 fast enough, and stable in fp16 on newer models? [U]
5. Is a no-refresh SST delta really low-rank in practice, so that it can export as a small LoRA?
6. What is Apple's exact threadgroup memory limit for MPS SVD? [U]

### Product and demand

1. Do users recognise forgetting, and will they act on a report?
2. How many people want Excel input?
3. Does a proprietary or minor tool already offer a retention report? The research could only show absence in what it read.

## 3. What the research could not read

These gaps limit confidence in parts of this plan.

| Source | Why it matters |
|---|---|
| r/LocalLLaMA, read directly | User sentiment comes from GitHub issues and docs only |
| The ICLR full text of OSFT, the SST authors' X thread and blog | Some method details rest on abstracts and search summaries |
| Unsloth's Excel Colab notebook | What Unsloth tells Excel users exactly |
| Kaggle's documentation | CI quotas are from secondary sources |
| Several model licences: Nemotron, OpenMDW, `qwen-community-1.0`, MiniMax M2.x, Mistral's ecosystem licence | Phase 2 model choices |
| Funding programmes: Mozilla, Chan Zuckerberg, Tidelift, Thanks.dev, Open Source Collective's India route | The full funding map |
| Compute programmes: Lambda, Hugging Face, AWS, Microsoft, CoreWeave, RunPod, IndiaAI | The full compute map |
| Any statement from a corporate sponsor about what it requires | The sponsorship checklist is judgement |

## 4. When to revisit this plan

- After Gate 1. Any of the five measurements can reorder the roadmap.
- When a new paper replicates or contradicts SST.
- When Unsloth, PEFT or TRL ships a retention report or SST.
- When a major Transformers or torch release breaks an integration point.
- At each funding milestone, M1 to M4.
