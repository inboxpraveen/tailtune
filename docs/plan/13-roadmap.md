# 13. Roadmap

**What this document decides:** the order of work, and the test that ends each gate. A gate closes when its exit test passes, not on a date. Gate 1 results can reorder everything after it.

This sequence is a synthesis of the research, not a sourced timeline [J]. Effort estimates assume one experienced developer working full time, and they are rough [J].

| Gate | Ship | Exit test | Rough effort [J] |
|---|---|---|---|
| 0, Claim | PyPI name through Trusted Publishing, orgs, corrected README, repo checklist, `doctor`, `estimate` | Real upload with attestation | 1 to 2 weeks |
| 1, Verify | Five measurements, published even if they disagree with the paper | One SST cell reproduced, or a documented disagreement | 2 to 4 weeks |
| 2, v0.1 Door | Data pipeline with Excel, `inspect`, `lora`, `sst` and `osf` on single GPU and DDP, fast report, merged, GGUF and Ollama export, one T4 notebook | Two commands and one confirmation from xlsx to a running model, and the logit round trip passes | 8 to 12 weeks |
| 3, v0.2 Proof | Preregistered 90-run benchmark, FSDP2, six launch models, public launch timed to a model release | Results published, losses included | 6 to 10 weeks |
| 4, v0.3 Reach | Refresh, SDFT, 4-bit `tail-lora` benchmark, `tailtune repair` if DG-Hard verifies, the two experimental models, leaderboard, measured Day-N table | Each new mode labelled "runs" or "retention benchmarked" | 8 to 12 weeks |
| 5, v1.0 | Stable config schema, ZeRO-3 if demanded, embeddings, DPO and GRPO, second maintainer, PyTorch Ecosystem | Scorecard 7+, support matrix per release | Depends on contributors |

## Gate 0, Claim

Do these in order. The name is unprotected: a pending publisher does not reserve it, and an empty placeholder can be removed under PEP 541.

1. Owner approves the positioning and README wording in [01](01-product-strategy.md).
2. Decide whether the repo moves to a `tailtune` GitHub org, and move it now if so. A Trusted Publisher is tied to the repo's owner and name, so this must happen before steps 7 and 8. See [11](11-engineering-and-release.md).
3. Replace the README, GitHub description and topics.
4. Set up the `src/` layout, uv, Ruff, pyright and hatchling, with `requires-python >=3.11`.
5. Build `tailtune doctor`, which reports Python, torch, CUDA or other backend, driver, VRAM, host RAM and real bf16 support using `including_emulation=False`.
6. Build `tailtune estimate`, which reads a model's config without downloading weights and prints a memory budget per route, labelled as not yet calibrated against hardware until Gate 1.
7. Set up the GitHub workflow with Actions pinned by SHA and a reviewer gate on the `pypi` environment.
8. Publish a real 0.0.1 through Trusted Publishing, and verify the attestation.
9. Claim the Hugging Face org, plus the GitHub org if step 2 did not. Claim a `.ai` or `.dev` domain after a registrar check, since `tailtune.com` points at a domain marketplace.
10. Message the SST authors about a licence, ZeRO-3, Muon and refresh code.

## Gate 1, Verify

These five measurements are cheap, and any of them could change the plan.

| # | Measurement | Protocol | What it decides |
|---|---|---|---|
| 1 | Reproduce one Table 5 cell | Run the public SST repo, unmodified, as a reference. It has no licence, so run it but copy nothing | Whether the headline can be repeated at all |
| 2 | Real projection and refresh overhead | Time head and tail forms per optimizer step on real shapes | Whether SST's overhead figures can be quoted, and SST's practical size limit |
| 3 | Peak memory | Measure a 0.3B and a 1.7B model on each route, and correct the `estimate` formula | Every budget in this plan, all computed until now |
| 4 | Embedding-freeze ablation | Compare in-domain and retention scores with embeddings frozen and unfrozen. On Qwen3-8B, 15.2% of parameters train unconstrained [C] | The default embedding policy (D-15) |
| 5 | PEFT OSF against SST at equal r | Same model, data, targets and r. No fair comparison on shipped code exists | What `method="auto"` prefers |

Publish all five, including any disagreement with the paper. Then review the locked decisions and record any changes in the [index](README.md).

## Gate 2, v0.1 Door

Dependencies run top to bottom, except that the data and evaluation work can proceed in parallel with methods.

**Recommended order: build the report first** [J].

1. `tailtune eval base/ ft/`, the forgetting report for any model from any trainer.
2. The demand test: publish one report on a popular public fine-tune.
3. The data pipeline with Excel.
4. `lora` on free GPUs.
5. `sst` and `osf` on rented GPUs.
6. Export and the notebook.

The reasons for this order:

- The report is the differentiator, and the cheapest piece to build, since it needs no training internals.
- It works on models trained with Unsloth or Axolotl, so it can win users before the trainer is mature.
- It tests the riskiest assumption, that people want to see what they lost (R14 in [14](14-risks-and-open-questions.md)), within weeks rather than months.

| Work package | Done when |
|---|---|
| `tailtune.data`: `sniff → read → normalize → detect → map → render → validate → freeze` | Fixture corpus passes. Blocking checks catch a double BOS, missing EOS and all-masked labels, with exit code 2 |
| `tailtune.targets` and `tailtune inspect` | Snapshot tests pass on one tiny model per launch architecture. `train` refuses below the coverage floor with exit code 4 |
| `tailtune.bases` | Offline fp32 SVD, hashed and cached. Tests use subspace distance, not vector equality |
| `tailtune.methods`: `lora`, `full`, `freeze`, `osf`, `sst` | Frozen-subspace content is unchanged after N steps. The hook runs after the real step and skips when `optimizer_step_was_skipped` is true |
| `tailtune.backends`: `single`, `ddp` | The same invariants hold on both, with a DDP drift check |
| `tailtune.eval` and `tailtune eval base/ ft/` | Accepts a merged model from any trainer. Uses paired statistics and always shows the worst benchmark. Exit code 3 on a failed gate |
| `tailtune.export` | Merged safetensors, then GGUF in an isolated environment, then vLLM (verified through the evaluation harness), then Ollama. Logit round trip passes, and the template hash is verified |
| `tailtune.train`, `tt.wrap`, CLI | Two-command flow runs. TRL wrapping is claimed only after a test passes |
| One T4 notebook, tested | Runs end to end on a free GPU with `lora` and the report |
| Demand test | One `eval` result published on a popular public fine-tune, with results checked carefully first |

## Gate 3, v0.2 Proof

1. Preregister the benchmark in a dated commit before any runs.
2. Run the first release: 90 runs (2 models, 3 domains, 5 arms, 3 seeds). This costs about 130 GPU-hours of training plus 90 to 270 of evaluation, or roughly $300 to $700 [C]. Those hours are optimistic for projected arms.
3. Apply the learning gate, so that a method which did not learn cannot win on retention.
4. Build the `fsdp2` backend. Dense FSDP2 projection has no public implementation, so treat it as new engineering.
5. Bring the six launch models in D-20 to their labelled status.
6. Launch alongside a model release, with a working notebook and reproducible numbers. Do not launch between releases.

## Gates 4 and 5

**Gate 4** adds refresh under its unverified label, SDFT through TRL's experimental trainer with a "runs" label until benchmarked, the first published 4-bit `tail-lora` benchmark, `tailtune repair` if DG-Hard verifies, the experimental models after FSDP2, a leaderboard, and a Day-N table containing measured dates only.

**Gate 5** freezes the config schema, adds embeddings, DPO and GRPO, and adds ZeRO-3 only on real demand. It also needs the second maintainer and a PyTorch Ecosystem listing. Apply as soon as the bars are met, which is milestone M1 in [12](12-community-and-sponsorship.md), rather than waiting for Gate 5. The application requires at least 2 core maintainers, 200 stars and 30+ commits in 90 days [S](https://github.com/pytorch-fdn/tac/blob/main/docs/governance/PyTorch_Ecosystem_Process.md).

## Open questions

1. Gate 1's rental budget is roughly $50 to $150 [J]. This assumes about 10 to 20 GPU-hours on one 96 to 180 GB card at the rates in [07](07-hardware.md), and the larger share goes to the reference run and the two 8B comparisons. It has not been costed in detail, so check it against current quotes before renting.
2. Effort estimates have not been tested against real progress. Revise them after Gate 1.
3. Whether a solo maintainer can keep up with the 48 Transformers releases shipped so far in 2026 is untested. The weekly compatibility job will show this early.
