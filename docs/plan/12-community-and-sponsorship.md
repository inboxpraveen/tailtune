# 12. Community and sponsorship

**What this document decides:**

- Growth comes from model releases and correct findings, not steady marketing. tailtune's repeatable move is a post on "what a stock fine-tune of this model breaks", published 24 to 72 hours after a hyped release.
- v0.2 launches alongside a model release, in a fixed order across a fixed set of channels.
- The project runs as a BDFL (benevolent dictator for life: one named lead who has the final say) with a written `GOVERNANCE.md`, DCO sign-off, no CLA, GitHub Discussions before Discord, and the second maintainer as the key milestone (D-24).
- Donations will not pay a salary. Ask for compute before cash, with asks tied to signals on an M0 to M4 ladder. Sponsors never buy roadmap control or benchmark placement, and the training core stays Apache-2.0 (D-25).
- Benchmark conflict-of-interest rules are written down before the first sponsor arrives.

Evidence labels are defined in the [plan index](README.md#evidence-labels). The GitHub description, topics and today's README text are in [01](01-product-strategy.md), and the gate order in [13](13-roadmap.md). Figures were retrieved on 2026-11-07 and 2026-11-08. Nothing here is tax or legal advice.

## 1. How comparable projects grew

### Size today

| Project | Stars [S] | Contributors, including anonymous [S] | PyPI downloads a month [S] |
|---|---|---|---|
| [Ollama](https://api.github.com/repos/ollama/ollama) | 186,260 | 629 | n/a |
| [llama.cpp](https://api.github.com/repos/ggml-org/llama.cpp) | 133,656 | 2,210 | n/a |
| [vLLM](https://api.github.com/repos/vllm-project/vllm) | 96,520 | 4,024 | not collected |
| [Unsloth](https://api.github.com/repos/unslothai/unsloth) | 79,153 | 442 | [885,608](https://pypistats.org/api/packages/unsloth/recent) |
| [LlamaFactory](https://api.github.com/repos/hiyouga/LLaMA-Factory) | 75,927 | 302 | not collected |
| [LiteLLM](https://api.github.com/repos/BerriAI/litellm) | 62,874 | 1,921 | not collected |
| [Axolotl](https://api.github.com/repos/axolotl-ai-cloud/axolotl) | 12,672 | 272 | [8,818](https://pypistats.org/api/packages/axolotl/recent) |

For scale, the libraries underneath are far larger: TRL has [2,540,661](https://pypistats.org/api/packages/trl/recent) downloads a month, PEFT [8,262,388](https://pypistats.org/api/packages/peft/recent) and Transformers [96,323,619](https://pypistats.org/api/packages/transformers/recent) [S].

### When the stars arrived

These are monthly star events from GH Archive, a public log of GitHub activity, queried through [ClickHouse's public playground](https://play.clickhouse.com/?user=play) [S].

| Project | Spike months (events) | Baseline nearby |
|---|---|---|
| llama.cpp | 2023-03: 15,229. 2023-04: 8,256 | 2,000 to 4,000 afterwards |
| Ollama | 2023-07: 3,552. 2023-11: 8,574. 2023-12: 8,826. 2024-04: 10,259. 2025-01: 10,704. 2025-02: 13,227 | 4,000 to 8,000 in 2024 |
| vLLM | 2023-06: 2,698. 2025-02: 3,818. 2025-03: 3,192 | 1,200 to 2,000 |
| Unsloth | 2023-12: 1,844. 2024-04: 2,642. 2024-05: 2,609. 2025-01: 2,644. 2025-02: 9,396. 2025-03: 3,237. 2026-03: 960 | 600 to 1,500 |
| LlamaFactory | 2024-01: 2,293. 2024-04: 5,116. 2025-02: 3,195. 2025-03: 3,069 | 1,500 to 2,000 |
| Axolotl | 2023-12: 937. 2024-01: 828. 2024-03: 789 | 200 to 290 a month in 2025 |

**Caveats that limit what this shows:**

- **GH Archive undercounts.** Unsloth's monthly sums total about 45.8k against 79,153 actual stars. The series can time spikes but not size them.
- **Recent data has gaps.** Counts for every repo collapse from around 2026-05, for example Unsloth at 10 in 2026-07, which is a collection artefact. A softer drop from about 2025-10 is probably partly the same.
- **No exact dates.** GitHub's stargazer timestamp API returned 404 for every repo tried, so exact milestone dates are missing.
- **Renames split some series.** No spike month in the table straddles a rename.
- **Timing does not prove cause.** Every link below between an event and a spike is an inference from timing.

### Events near the spikes

| Date | Event | What the star data shows |
|---|---|---|
| March 2024 | Unsloth publicly lists Gemma bugs: missing `<bos>`, a typo in `<end_of_turn>`, float32 needs for layernorm and RoPE [S](https://x.com/danielhanchen/status/1765446273661075609), [HN](https://news.ycombinator.com/item?id=39671146). A 2024 conference talk later cited 8 Gemma bugs, a Phi-3 sliding-window issue and Llama 3 tokenization issues [S](https://www.ai.engineer/worldsfair/2024/schedule/daniel-han-tba) | **No March 2024 spike.** The next spikes, in April and May 2024, overlap the Llama 3 period (inferred, not sourced) and GitHub Accelerator selection in May 2024 [S](https://github.blog/news-insights/company-news/2024-github-accelerator-meet-the-11-projects-shaping-open-source-ai/). The data does not isolate the bug-finding effect |
| 2025-01-27 | Unsloth's r/LocalLLaMA post of a 1.58-bit DeepSeek-R1 dynamic GGUF, 131 GB, with MoE layers at 1.5 bits and attention at 4 to 6 bits because naive 1.5-bit gave gibberish [S](https://redlib.ssps.io/r/LocalLLaMA/comments/1ibbloy/158bit_deepseek_r1_131gb_dynamic_gguf/m9kdti7) | 9,396 events the next month, its largest spike by far. Every top project except Axolotl spiked from January to March 2025. Axolotl barely moved: 245 in January, 286 in February |
| March 2025, April 2026 | Gemma 3 float16 overflow fix [S](https://x.com/danielhanchen/status/1902396261875249346). Gemma 4 fixes, stated to affect all implementations [S](https://github.com/unslothai/unsloth/discussions/4921) | No large spike visible, but the 2026 data is unreliable |
| 2024 | LlamaFactory's ACL demo paper [S](https://aclanthology.org/2024.acl-demos.38), and a Day-0 support table in its README [S](https://github.com/hiyouga/LlamaFactory) | A citation channel. Not tied to one spike |

Unsloth's YC profile, seen only through search because a direct fetch returned just its title, cites "20+ bugs fixed" alongside downloads and stars [S](https://www.ycombinator.com/companies/unsloth-ai). So bug fixing is at least used as a credential, even if the star data cannot show it as a growth driver.

### What this supports, and what it does not

| Supported by the data, as timing | Not established |
|---|---|
| Stars arrive in bursts near model releases | What caused the spikes for Ollama, vLLM and llama.cpp. Their launch threads were not read |
| Unsloth's largest burst followed a model-release post that solved a concrete problem | Whether bug-finding posts moved stars |
| Axolotl, with configuration-first breadth and no free-notebook hook, missed the 2025 wave and has about 6x fewer stars than Unsloth. Its PyPI count understates it, because its README also leads with Docker and cloud providers | The cause of the April 2024 rise |
| | Exact milestone dates |
| | Anything about Reddit sentiment. Reddit could not be read directly, so user sentiment comes from GitHub issues and docs only |

## 2. The growth plan for one maintainer

A solo maintainer cannot offer day-zero support across model families. Unsloth ships at a cadence of 107 PyPI releases so far in 2026 [S](https://pypi.org/pypi/unsloth/json). The plan therefore uses levers that need insight rather than budget [J].

### 2.1 The model-release post

**The format:** "What a stock fine-tune of `<model>` breaks", published 24 to 72 hours after a hyped release, with one working notebook. Only a project whose product is the forgetting report can write this post. Each new architecture is both a test of `tailtune inspect` and a launch moment.

**Built in advance (by Gate 3) [J]:**

- `tailtune inspect` working from a config on the meta device, without downloading weights ([06](06-models.md)).
- A fixed small fine-tuning dataset and a fixed fast-report suite ([09](09-evaluation.md)), so the only variable is the model.
- A notebook template ([04](04-api-and-ux.md)) and a post template with the D-02 wording already in it.
- A rental account with a budget cap.

**The runbook:**

| Hours after release | Step | Output |
|---|---|---|
| 0 to 4 | Read the model card, licence and config. Run `tailtune inspect`: roles, shapes, coverage, and quirks such as fused QKV, 3D experts, MTP heads or vision towers | Go or no-go. An unsupported shape is itself a finding |
| 4 to 24 | Run a stock LoRA fine-tune on the fixed dataset, plus `sst` or `osf` if it fits. Run `tailtune eval base/ ft/`, with two seeds where budget allows | `FORGETTING_REPORT.md`, per-item JSON |
| 24 to 48 | Write the post: what broke, the worst benchmark, "no regression detected at ±X points" where true, and the "runs" or "retention benchmarked" label. Test the notebook on Kaggle or Colab | Draft, notebook |
| 48 to 72 | Publish. See section 2.7 for channels | Post, notebook, results |

**Rules [J]:**

- Correctness beats speed. If the post is not right by 72 hours, publish when it is.
- Write at most two of these posts a month, to protect build time.
- Record the actual rental cost in each post.
- Evaluate models, not their authors (as [01](01-product-strategy.md) also says).

### 2.2 Being publicly right about bugs

This is the highest-leverage content a solo developer can produce, and it needs insight rather than budget [J]. tailtune's own checks will surface this kind of finding:

- a template that drifts at export
- labels that are all masked
- a double BOS
- a tuner that silently trains nothing, as PEFT's OSF v0.21.2 appears to do on Conv1D targets by source reading [S](https://github.com/huggingface/peft/blob/v0.21.2/src/peft/tuners/osf/model.py)

**Rules [J]:**

1. Reproduce on two seeds or two setups, and reduce to a minimal repro.
2. File the upstream issue first, then post and link to it.
3. Describe the bug and credit whoever fixes it.
4. Never claim "all implementations" without testing them.
5. If a finding turns out wrong, correct it publicly in the same channels.

### 2.3 A citable tech report

At Gate 3, publish the preregistered benchmark as a tech report on arXiv, and add it to `CITATION.cff`. LlamaFactory's ACL 2024 demo paper is the precedent for a framework gaining a citation channel [S](https://aclanthology.org/2024.acl-demos.38). Offer co-authorship to outside contributors who run benchmark arms [J]. A peer-reviewed demo track can come later [J].

### 2.4 Migration pages

| Page | Audience | Why |
|---|---|---|
| From AutoTrain | No-code users | AutoTrain's README says it is no longer maintained and sends users to tools that are not no-code [S](https://github.com/huggingface/autotrain-advanced/blob/552dfe8c245067113ada4b5f918fb9af95d57dc4/README.md) |
| Check a model you trained with Unsloth | The local tinkerer in [01](01-product-strategy.md) | `tailtune eval` works on any merged model. The Hub has 147,002 models tagged `unsloth` and 274,691 tagged `peft` [S](https://huggingface.co/models-json?other=sft&withCount=true) |
| From PEFT `OSFConfig` | ML engineers | Names the exact variant and rank (D-11) |

Structure and placement belong to [04](04-api-and-ux.md). The AutoTrain page is a Gate 2 item in [01](01-product-strategy.md).

### 2.5 The SST authors' thread

The lead author announced SST on X. The research could read that thread only through a search-engine summary [U]. Per that summary, the lead author framed SST as controlling where weight updates happen. When a reply claimed the work had solved catastrophic forgetting, they answered that "solved" is a stretch and that the team plans to look for failure cases ([X profile](https://x.com/ArnavChavan6)).

That is the honest angle for tailtune: a benchmark that publishes failure cases is what the authors said they want. The order [J]:

1. **Gate 0:** a private message about a licence, ZeRO-3, Muon and refresh code (step 9 in [13](13-roadmap.md)).
2. **After Gate 1:** once a Table 5 cell is reproduced or a disagreement is documented, send the result to the authors privately and give them 7 days to respond.
3. **Then:** reply publicly in the thread with the numbers, the repro command and any failure cases, framed as an independent replication. Write "SST (Chavan et al., 2026)" in full (D-02).

Do not tag the authors publicly before there are results.

### 2.6 Launch mistakes to avoid

The research names four mistakes:

- launching with no working notebook
- unreproducible benchmarks
- launching between model releases
- a long install

This plan adds four of its own [J]:

- Claims beyond D-02, or a README that the user's first run contradicts.
- Launching before Discussions, issue templates and good first issues are ready.
- Launching without 7 clear days to answer questions.
- Posting identical text everywhere. Each channel gets its own framing.

### 2.7 v0.2 launch runbook

**Preconditions.** All must be true:

- Gate 3's exit test has passed.
- The README has been rebuilt to the [11](11-engineering-and-release.md) standard, with real numbers only.
- Notebooks are green on Kaggle and Colab.
- Docs are live with `llms.txt`, and the Scorecard badge is up.
- GitHub Sponsors is open.
- At least 5 good first issues are open.
- The tech report has been submitted.

**Launch window.** Ship v0.2 within 72 hours of the first hyped model release after the preconditions are met, using the model-release post as the vehicle. If none comes within 6 weeks, publish the tech report and benchmark quietly and keep waiting for a release to launch on [J].

| When | Step | Channel |
|---|---|---|
| T−14 days | Pre-brief every method author whose method appears, with the same content at the same time: results, repro, an invitation to send corrections | Private email to the SST authors, PEFT maintainers and Red Hat's Training Hub team |
| T−7 days | Freeze. Run the release checklist from [11](11-engineering-and-release.md) to step 4. Submit the tech report so its link exists at launch | arXiv |
| T−1 day | Draft each channel's text. Stage the results dataset and Space | Private |
| T0 | Model releases. Run the 2.1 runbook | — |
| Launch, step 1 | Publish v0.2 | PyPI, GitHub Release |
| Step 2 | Make the docs, benchmark page, results dataset and Space public | Docs site, Hugging Face |
| Step 3 | Long-form post: the model's breakage plus the benchmark, losses included | Docs blog (the canonical copy) |
| Step 4 | Announcement, pinned | GitHub Discussions |
| Step 5 | Fine-tuned model cards carrying `FORGETTING_REPORT.md` | Hugging Face |
| Step 6 | Thread. Reply to the SST authors' thread as in 2.5 | X |
| Step 7 | Technical post: numbers, notebook, repro | r/LocalLLaMA, where Unsloth's 2025 post ran |
| Step 8 | "Show HN", only if the benchmark result is notable [J] | Hacker News |
| T+1 to T+7 days | Answer every issue and Discussion within 24 hours. Ship a patch if needed. Fold accepted corrections into the results | GitHub |
| T+14 days | Retrospective: stars, downloads, new issues, outside forgetting reports, what to change | Discussions |

Channel norms on Reddit were not read and are unverified [U].

## 3. GitHub repo metadata

The current description says "without catastrophic forgetting" and "from a single GPU to multi-node". Neither meets D-02. The replacement description and topics are in [01](01-product-strategy.md), section 7, and need the owner's approval. This document adds only rules for later updates [J]:

1. The description opens with the benefit, the D-01 tagline, before any feature.
2. Never put a number in the description. Numbers need sources, and the README carries them.
3. Status words match the README's status badge, so "Early development" goes when the badge changes.
4. Add a topic only when its feature ships. Candidates are `ollama` and `excel` at v0.1, then `pytorch` and `transformers`.
5. Set the website field to the docs site once the domain exists ([11](11-engineering-and-release.md)). Add a social preview image showing the tagline.
6. Turn on Discussions, private vulnerability reporting and the Sponsor button (`FUNDING.yml`, at v0.1).
7. Re-check the description and topics against D-02 in every release's claims review.

## 4. Governance and contribution

### Evidence versus judgement

| Topic | Evidence | Judgement |
|---|---|---|
| BDFL with written policies | Two people carried Unsloth to 79k stars [S](https://github.com/sponsors/unslothai). Axolotl's sidebar has Code of Conduct, Contributing and Security, but no governance section [S](https://github.com/axolotl-ai-cloud/axolotl) | That a written BDFL model suits this stage |
| DCO, not CLA | vLLM requires DCO sign-off and carries a `DCO` file [S](https://raw.githubusercontent.com/vllm-project/vllm/main/docs/contributing/README.md) | No CLA, because no relicensing is planned |
| Second maintainer | PyTorch Ecosystem needs at least 2 core maintainers [S](https://github.com/pytorch-fdn/tac/blob/main/docs/governance/PyTorch_Ecosystem_Process.md) | Recruitment method |
| Community profile | vLLM and LlamaFactory 100%, Unsloth and Axolotl 87%, llama.cpp and LiteLLM 75%, Ollama 62% [S](https://api.github.com/repos/vllm-project/vllm/community/profile) | Target 100% at Gate 0 |
| Code of conduct | Unsloth, Axolotl, vLLM and LlamaFactory use Contributor Covenant. None was detected for Ollama, llama.cpp or LiteLLM [S](https://api.github.com/repos/vllm-project/vllm/community/profile) | Choice below |
| RFC process, burnout, Discord versus Discussions | **Not researched.** No sourced comparison exists | Everything below on these topics |

### `GOVERNANCE.md`

The contents [J]:

1. **Roles.**
   - **BDFL:** the project owner, named.
   - **Maintainers:** merge, review and release rights.
   - **Contributors:** everyone else.
2. **Decisions.**
   - Lazy consensus on PRs: a non-trivial PR with no objection after 72 hours may merge.
   - Disagreements are discussed in public. The BDFL decides and writes down the reason.
   - Locked decisions (D-01 to D-25) change only through a gate result or an owner decision, recorded first in [the plan index](README.md).
3. **Becoming a maintainer.**
   - About 10 merged non-trivial PRs over at least 3 months.
   - Regular reviews of other people's PRs.
   - Ownership of an area, such as `tailtune.data`, `tailtune.eval` or a backend.
   - Following the code of conduct.
   - A maintainer nominates, and the BDFL confirms.
4. **Stepping down.** An emeritus status, with access removed after 6 months of inactivity.
5. **Continuity.**
   - Once a second maintainer exists, they become co-owner of the PyPI project, the GitHub and Hugging Face orgs and the domain.
   - If the BDFL is unreachable for 60 days, the maintainers may act on the project's behalf.
   - Credentials and recovery steps live in a private runbook that both maintainers hold.
6. **Commitment.** A 12-month maintenance commitment, which the PyTorch Ecosystem process requires.
7. **Money and conflicts.** Links to the sponsor rules (section 8) and benchmark rules (section 12).
8. **Changing governance.** Through an RFC. Move to a small steering group once there are 3 or more maintainers.

### DCO and CODEOWNERS

- **DCO (D-06).** Every commit carries `Signed-off-by:` (`git commit -s`), and a check runs on every PR. Decide on DCO versus CLA before the first outside contribution, because adding a CLA later is hard. With DCO and no CLA, contributions cannot be relicensed without their authors, and that is what makes the Apache-2.0 promise in section 11 credible.
- **CODEOWNERS.** Today: the owner for `*`. Later: per-area owners for `src/tailtune/data/`, `src/tailtune/eval/` and so on, with `.github/` always needing a maintainer review.

### RFC process [J]

**When an RFC is needed:**

- a change to a locked decision
- a breaking change to the public API or config schema
- a new method or a new required dependency
- telemetry
- licensing, governance or sponsor policy

**How it works:**

1. Open a Discussion in an "RFC" category using a template: problem, proposal, evidence with labels, alternatives, migration, and who will maintain it.
2. Comments stay open for 14 days, or 30 for telemetry and licensing.
3. The BDFL decides and records the outcome.
4. An accepted RFC is committed as `rfcs/NNNN-title.md`, and any D-xx change is recorded in the plan index.

### Good first issues [J]

- **Labels:** `good first issue` and `help wanted`.
- **Content:** every issue names the file, the expected change, the test to add and the command to run it.
- **Rules:** the maintainer never fixes these issues themselves, and keeps 5 to 10 open from v0.1 on. First-time contributors get a reply within 48 hours.

Good candidates:

- a role mapping for a new architecture in `tailtune.targets`, with its tiny-model test
- a data fixture for a spreadsheet failure
- a `doctor` probe for a platform
- a migration or how-to page

### Discussions first, Discord later [J]

GitHub Discussions comes first because it is searchable and indexed. The categories are Q&A, Show your report, Ideas, RFC and Announcements.

For comparison, Axolotl uses Discord plus a support email, while vLLM uses Issues, a forum, Slack, Security Advisories and a collaboration email [S](https://github.com/vllm-project/vllm).

Open Discord when either of these is true:

- Discussions sees more than about 30 new threads a week for 4 weeks.
- A second person can moderate it.

Claim the server name early ([11](11-engineering-and-release.md)).

### Code of conduct

Adopt Contributor Covenant, which four peers use, with the owner as enforcement contact and a second contact once there is a second maintainer [J]. The PyTorch layout names the Linux Foundation code as the default [S](https://github.com/pytorch-fdn/tac/blob/main/docs/governance/PyTorch_Ecosystem_Process.md). Whether it accepts another code is unverified [U], so confirm before applying at M1. Switching costs little.

### Recruiting the second maintainer

The second maintainer is the key milestone (D-24). Three things depend on them:

- the PyTorch Ecosystem listing
- the main fix for the bus factor (the risk that the project stops if one person stops)
- the Code-Review check in Scorecard ([11](11-engineering-and-release.md))

Unsloth's sponsors page describes its team as "2 brothers" [S](https://github.com/sponsors/unslothai). One of them has described the split as one brother on algorithms and the other on front end, website and everything else [S](https://un-supervised-learning-pod.simplecast.com/episodes/unsloth-chat). That suggests a non-ML partner who covers docs, web, community and release operations is valuable, and may be easier to find than a second ML specialist [J].

The steps [J]:

1. From v0.1, label the areas that need an owner.
2. Invite contributors with 3 or more merged PRs to help triage.
3. Offer area ownership, with a stated time expectation of about 4 hours a week.
4. Share credentials only after 3 months of maintainer work.
5. When money arrives, spend first on benchmark compute, then on a stipend for the second maintainer.

### Burnout mitigations [J]

No burnout data was researched. These are judgement:

- **Scope.** The D-03 non-goals, plus a polite "out of scope" reply template.
- **Intake.** Issue templates require `tailtune doctor --json` and a minimal repro. Support happens only in Discussions, never by direct message ([11](11-engineering-and-release.md), `SUPPORT.md`).
- **Rhythm.** Two fixed triage slots a week rather than constant watching. Releases follow the [11](11-engineering-and-release.md) cadence. Model-release posts are capped at two a month.
- **No day-zero promise.** The README never promises support on release day.
- **Breaks.** Announce absences in Discussions.
- **Backlog rule.** If untriaged issues pass 50, pause feature work until the count is back under 20.

## 5. Funding reality

**No comparable project has sustained itself on donations.** They raised venture money, joined a larger company, or stayed lightly funded.

| Project | Facts | Label |
|---|---|---|
| Unsloth | 13 current and 24 past GitHub sponsors at 79k stars, with no tiers or goals on the page. Past sponsors include @github and @elevenlabs, current ones @comet-ml [S](https://github.com/sponsors/unslothai) | [S] |
| Unsloth | YC S24 and about $500K to $540K [U](https://www.employbl.com/companies/unsloth), or $630K [U](https://www.everydev.ai/developers/unsloth), from aggregators. The YC page did not render its data | [U] |
| Unsloth | GitHub Accelerator 2024, cohort of 11 [S](https://github.blog/news-insights/company-news/2024-github-accelerator-meet-the-11-projects-shaping-open-source-ai/). The $40K figure comes from [CB Insights](https://www.cbinsights.com/investor/github-accelerator) only | Cohort [S], amount [U] |
| Axolotl | a16z Open Source AI Grant, second batch, 2023-12-13 [S](https://a16z.com/announcing-our-latest-open-source-ai-grants). No named sponsors at 12.7k stars. The README asks sponsors to email | Grant amount unknown [U] |
| LlamaFactory | Two named supporters, Warp and SerpApi, at 76k stars. Warp uses a tracked link [S](https://github.com/hiyouga/LlamaFactory) | Amounts unknown |
| vLLM / Inferact | $150M seed at an $800M valuation, January 2026, co-led by a16z and Lightspeed [S](https://techcrunch.com/2026/01/22/inference-startup-inferact-lands-150m-to-commercialize-vllm/) | [S] |
| Ollama | $65M Series B on 2026-07-09 led by Theory Ventures, $88M in total [S](https://lasvegassun.com/news/2026/jul/09/ollama-raises-65m-series-b-funding-to-grow-its-ope/). Earlier rounds per [Dealroom](https://dealroom.co/companies/ollama-com/) | Series B [S], earlier [U] |
| llama.cpp / ggml.ai | Joined Hugging Face, announced 2026-02-20, terms undisclosed [S](https://hn.jakechampion.name/reader/https://github.com/ggml-org/llama.cpp/discussions/19759) | [S] |
| LiteLLM / BerriAI | YC W23, about $1.6M seed, $5M+ ARR per one profile, with conflicting dates [U](https://www.jaredwatkins.com/research/compute-marketplaces/litellm/) | [U] |
| vLLM sponsors | In an older docs snapshot: 5 cash sponsors and 19 compute sponsors [S](https://docs.vllm.ai/en/v0.10.1.1/community/sponsors.html) | [S] |

**What attracted large funding.** The large outcomes went to inference and runtime projects with broad usage. Coverage of Inferact cites Amazon as a user [S](https://techcrunch.com/2026/01/22/inference-startup-inferact-lands-150m-to-commercialize-vllm/) and more than 400,000 GPUs running vLLM, a secondary figure. If the aggregator figures are right, Unsloth's raise is roughly 300 times smaller than Inferact's [U].

Unsloth now describes its repo as a "Local UI to run and train LLMs and diffusion models" [S](https://api.github.com/repos/unslothai/unsloth). That move fits the reading that fine-tuning alone is a narrow base for venture funding [J]. Fine-tuning runs less often than inference, on fewer machines, by fewer people.

**What follows for tailtune [J]:**

- Donations cover tools and small bills, not a salary.
- Compute is what vendors give most readily: vLLM's list has about four times as many compute sponsors as cash sponsors. So ask for compute first (D-25).
- Stars alone do not bring sponsors. Axolotl has 12.7k stars and no named sponsors, so outreach must be active.

## 6. Sponsorship readiness

No corporate sponsor stated their requirements anywhere in the research. This checklist is built from published programme bars and from what funded projects chose to cite. Every item except the PyTorch bars is judgement [J].

**What a sponsor sees in the first minute [J]:**

1. The README first screen: one reproducible number with the losses linked ([11](11-engineering-and-release.md)).
2. Security signals: the Scorecard badge, Trusted Publishing and attestations.
3. Maturity signals: the release cadence, the support table and the changelog.
4. A real team: more than one maintainer.
5. Sponsor rules: what a sponsor gets, and what it never gets.

| Item | Why a sponsor cares | Gate |
|---|---|---|
| Trusted Publishing and attestations on every release | Supply-chain risk | 0 |
| Scorecard 7+ and badge, `SECURITY.md`, private reporting | Comparable across vendors, and above all three peers measured | 2 for every check the owner controls, 5 for 7+ |
| Safetensors only, `trust_remote_code` off | Safe to run in their environment | 2 |
| Reproducible benchmark with hardware and script, losses included | Independent verifiability is the differentiator | 3 |
| Support table for torch and Transformers per release | Integrators need it | 2 |
| Public roadmap and dated changelog | Predictability | 0 |
| Day-N table with measured dates | Responsiveness | 4 |
| `GOVERNANCE.md`, `CODEOWNERS`, DCO, second maintainer named | Bus factor | 0, then 5 |
| "Used by" logos, shown only with permission. `CITATION.cff` | Social proof | 3 |
| One-page pitch (section 10) | Their first read | M2 |
| Legal entity decision | Gates NVIDIA Inception and invoicing | M3 to M4 |

## 7. Milestone ladder

The ladder is triggered by signals, not dates. Apply to a programme when its bar is met, whichever gate the build is at. The research synthesis places the PyTorch Ecosystem application at Gate 5 in [13](13-roadmap.md), but if M1's bars are met earlier, apply then. Thresholds are judgement except the PyTorch Ecosystem bars [S](https://github.com/pytorch-fdn/tac/blob/main/docs/governance/PyTorch_Ecosystem_Process.md).

| Milestone | Signal | Approach | Ask |
|---|---|---|---|
| **M0** | Name secured, real 0.0.1 | Nobody yet | Nothing. Open GitHub Sponsors at v0.1, without tiers |
| **M1** | 200 stars, 30+ commits in 90 days, second maintainer [S] | PyTorch Ecosystem, FOSS United, AMD | Ecosystem listing. A FOSS United project grant. AMD cloud access for Tier 2 ROCm evidence ([07](07-hardware.md)) |
| **M2** | 1k stars, and a model-release post that lands: at least 3 outside forgetting reports or new-user issues within two weeks [J] | GitHub Secure Open Source Fund, FLOSS/fund, GPU clouds. Publish the tier page | Security-fund work on the Scorecard items. A FLOSS/fund grant. Credits for the first 90-run benchmark, about $300 to $700 [C] |
| **M3** | 5k stars, about 50k monthly downloads | Developer-tool sponsors (as Warp sponsors LlamaFactory), a16z-type grants, model labs | Named tier sponsors. A grant. Pre-release model access for 2.1 posts, disclosed in each post |
| **M4** | 15k stars, named production users | Cash sponsors, cloud vendors | Multi-year sponsorship. Decide on a company or accelerator. NVIDIA Inception requires one |

The plan index recommends staying an individual maintainer until M3 or M4.

## 8. Sponsor tiers

Publish these at M2. All prices are illustrative judgement [J]. GitHub Sponsors charges 0% on sponsorships from personal accounts and up to 6% from organisations (3% card, 3% GitHub service). Paying by invoice avoids the card fee [S](https://docs.github.com/en/sponsors/getting-started-with-github-sponsors/about-github-sponsors).

| Tier | Per month [J] | What it does |
|---|---|---|
| Backer | $5 to $25 | Name in `BACKERS.md` |
| Bronze | $100 to $250 | Small logo in the docs |
| Silver | $500 to $1,000 | README logo, thanks in release notes |
| Gold | $2,500 to $5,000 | Top README logo, one integration example maintained in `examples/`, a quarterly call to hear their needs. Decisions still go through the RFC process |
| Compute partner | In kind | CI credit acknowledged, an "Open in" badge, a provider guide in the docs |

**What no tier ever buys (D-25):**

- roadmap control, or a faster RFC
- benchmark placement, ordering, inclusion or exclusion
- the choice of a default method or of what `auto` selects
- review or veto of results before publication
- removal of unfavourable results
- exclusivity, such as being the only provider listed
- any statement that tailtune endorses their product
- user data, of which tailtune collects none ([11](11-engineering-and-release.md))

Every sponsor is listed with what they fund, and any benchmark that features a sponsor's hardware, model or product says so beside the result.

## 9. Channels

### Funding and status programmes

"India" means an individual maintainer resident in India.

| Channel | Status at retrieval | Terms found | India | Label | When |
|---|---|---|---|---|---|
| [GitHub Sponsors](https://docs.github.com/en/sponsors/getting-started-with-github-sponsors/about-github-sponsors) | Open. The Matching Fund is closed | 0% from personal accounts, up to 6% from organisations. A W-8BEN is needed, with PAN as tax ID [S](https://docs.github.com/en/sponsors/receiving-sponsorships-through-github-sponsors/tax-information-for-github-sponsors) | Yes, since 2022-05-23, with a bank account and tax residence in India [S](https://github.blog/changelog/2022-05-23-github-sponsors-is-now-available-in-india). INR payout after Stripe KYC is claimed only by a third party [U](https://karboncard.com/blog/github-sponsors-india-payouts) | [S], payout [U] | v0.1 |
| [GitHub Secure Open Source Fund](https://github.com/open-source/github-secure-open-source-fund) | Rounds recur. 2026 terms and deadline unconfirmed [U] | $10,000 per project in the original round, $1.25M across 125 projects [S](https://github.blog/news-insights/company-news/announcing-github-secure-open-source-fund/). A third round reported 67 projects and $670,000 [S](https://helpnetsecurity.com/?p=317390) | Yes, through Sponsors regions, for a valid licence [S] | [S], 2026 [U] | M2, after Scorecard work |
| [FLOSS/fund](https://floss.fund/) (Zerodha) | Application form live. Current intake unconfirmed | "$1 million a year", "for individuals and orgs, big and small". First tranche $325k to 9 projects on 2025-05-30 [S]. Final 2025 tranche of $675,000 reported [S](https://www.opensourceforu.com/2025/10/zerodha-announces-final-675000-tranche-for-global-open-source-projects-advocates-sovereign-foss-fund/). A $10K to $100K range appears only in a digest [U]. FAQ unread | Global and open to individuals. India is not named specifically | [S], sizes [U] | M2, with real users |
| [FOSS United](https://fossunited.org/grants/project) | Grants listed | Project grants of ₹1,00,000, ₹3,00,000 and ₹4,00,000 [S] | Likely. Eligibility rules unread [U] | [S], eligibility [U] | M1 |
| [a16z Open Source AI Grants](https://a16z.com/announcing-our-latest-open-source-ai-grants) | Batches on 2023-08-30, 2023-12-13 (including Axolotl) and 2025-06-26 [S](https://a16z.com/advancing-open-source-ai-through-benchmarks-and-bold-experimentation/). Current intake unconfirmed | Amounts unknown | Unverified [U] | [S], terms [U] | M3. The most relevant precedent |
| [Open Source Collective](https://oscollective.org/about/) | Active. A US 501(c)(6) non-profit that manages money for projects. vLLM has used Open Collective [S](https://docs.vllm.ai/en/v0.10.1.1/community/sponsors.html) | Fees and eligibility not on the page read | **Not researched** | [U] | When a sponsor needs invoices |
| [Polar](https://polar.sh/docs/merchant-of-record/supported-countries) | Open, as a merchant of record | India on its payout list through Stripe Connect Express. Fees and any donation product not stated | Payouts yes [S] | [S] | Only when selling something |
| [PyTorch Ecosystem](https://github.com/pytorch-fdn/tac/blob/main/docs/governance/PyTorch_Ecosystem_Process.md) | Open. Status, not cash | 2+ core maintainers, 200 stars, 30+ commits in 90 days, the latest two torch releases, governance, CI. Rechecked every 6 months | Not location-bound as stated | [S] | M1 |
| [PyTorch foundation-hosted](https://github.com/pytorch-fdn/foundation-hosted) | Open | Must already be an Ecosystem project, with a TAC sponsor and assets transferred to the Linux Foundation. "Blog Candidate" is one outcome [S](https://pytorch.org/projects/host-your-project/) | — | [S] | After v1.0 [J] |
| GitHub Accelerator | Unsloth was in the 2024 cohort [S](https://github.blog/news-insights/company-news/2024-github-accelerator-meet-the-11-projects-shaping-open-source-ai/). Current status not researched | $40K is secondary [U] | Not researched | [U] | Check at M2 |
| [NLnet NGI Zero Commons Fund](https://nlnet.nl/commonsfund/) | **Closed.** The final call closed 2026-06-01. Other calls (Restack, CodeSupply) appear to close 2027-01-03, terms unread [U] | €5,000 to €50,000 | n/a | [S] | No |
| [NumFOCUS](https://numfocus.org/projects-overview) | **Paused** new applications | Needs public governance and a roadmap | n/a | [S] | No |
| [Sovereign Tech Fund](https://www.sovereign.tech/programs/fund) | Open | Work over €50,000, technology "vital to the development of other software", "not looking for user-facing applications", no prototypes | Whether individuals qualify is not stated [U] | [S] | Poor fit |

### Compute programmes

| Programme | Terms found | India individual | Label | Use for tailtune |
|---|---|---|---|---|
| [AMD Developer Cloud](https://www.amd.com/en/developer/resources/cloud-access.html) | "Cost Free (with approval)" for independent developers, startups and open-source contributors, at AMD's discretion. Program perks list $100 in credits with a 90-day expiry [S](https://developer.amd.com/member-perks/). One third party says 30 days [U]. Credits arrive within 3 business days of approval [S](https://amd.com/en/developer/resources/technical-articles/2026/how-to-claim-amd-cloud-credits.html) | Independent developers are named. Country rules not read [U] | [S] | Tier 2 ROCm evidence at M1. The only verified individual amount found |
| [NVIDIA Inception](https://www.nvidia.com/en-us/startups/) | "Your company needs to be incorporated", under 10 years old, with a website and an active founder. Free, no equity. Consulting firms excluded | **No**, not as an individual | [S] | Only after incorporating |
| [Google TPU Research Cloud](https://sites.research.google/trc/about/) | Cloud TPUs at no charge across 1,000+ devices, PyTorch listed. Work must be shared publicly. Duration and eligibility not stated | Not stated [U] | [S] | Poor fit now: TorchTPU is Tier 3 and `torch_xla` unsupported (D-17) |
| [Modal](https://modal.com/pricing) | $30 a month free on Starter. Academics up to $10k. No open-source programme on the page | Not stated [U] | [S] | Nightly and release GPU CI ([11](11-engineering-and-release.md)) |
| Kaggle | Free T4. A weekly quota of about 30 h is user-reported, and terms were not read | — | [U] | Weekly T4 CI, user notebooks |
| [Hugging Face Spaces GPU grants](https://huggingface.co/docs/hub/spaces-gpus) | Community GPU grants apply to Spaces | Not stated | [S] | Hosting the benchmark Space and leaderboard |
| [TensorWave](https://go.tensorwave.com/open-source-developer-program) | Open-source developer program on AMD hardware, credits reportedly up to $250,000 | Not stated | [U] | MI300X job if granted |
| [CloudRift AI Grant](https://grantedai.com/grants/cloudrift-ai-grant-gpu-credits-cloudrift-a1b2c3d4) | $100 to $1,000 per a grant listing | Not stated | [U] | Small benchmark runs |
| [Cirun](https://cirun.io/) | Free for open source, but runs on your own cloud account | — | [S] | Only once credits exist |

Do not plan around five-figure credits before traction [J]. A `tailtune[<provider>]` extra or a provider guide is a natural thing to offer in return for credits. Axolotl's `modal`, `baseten` and `coreweave` extras show the pattern [S](https://raw.githubusercontent.com/axolotl-ai-cloud/axolotl/main/pyproject.toml), though whether credits changed hands was not established.

### Not researched, and therefore open

The research did not cover the following. No statement is made about their status, terms or India availability:

- Mozilla
- Chan Zuckerberg Initiative
- Tidelift
- Thanks.dev
- Open Source Collective's India payout route
- Lambda
- Hugging Face's general open-source funding and community grants beyond Spaces
- AWS open-source credits
- Microsoft for Startups
- CoreWeave
- RunPod
- Nebius
- IndiaAI compute subsidies
- NVIDIA academic grants
- Buy Me a Coffee and Ko-fi
- Linux Foundation LFX

A provider appearing in vLLM's compute-sponsor list does not mean it runs an open programme.

## 10. One-page sponsor pitch

**Outline:**

1. **The problem.** Fine-tuning quietly damages what a model knew, and almost nobody measures it. Until tailtune has its own numbers, cite SST (Chavan et al., 2026), Table 5, as author-reported with no independent replication found [S](https://arxiv.org/html/2610.22293).
2. **What tailtune is.** The D-01 tagline. Data in, training that fits, a forgetting report on every run for any model, and a running model out.
3. **Proof.** The benchmark, losses included, with a one-command repro.
4. **Traction.** Stars, monthly downloads from pypistats, forgetting reports in other people's model cards, and outside benchmark submissions.
5. **Trust.** Scorecard score against the peer figures (6.5, 5.7, 4.9), Trusted Publishing, governance and the maintainers.
6. **The ask.** A specific amount mapped to output, for example "credits worth N GPU-hours run the next benchmark release of M runs".
7. **Tiers and rules.** Section 8, including what no tier buys.
8. **Contact.**

**Claims allowed by milestone [J]:**

| Milestone | May claim | Must not claim yet |
|---|---|---|
| M0 to M1 | An open-source tool being built to report what fine-tuning breaks. `doctor` and `estimate` exist. The plan is public | Any retention number of our own. User counts. "The reference" |
| M2 | Benchmark results with losses (after Gate 3). Download counts. Scorecard score. "Independent replication of SST", only if Gate 1 reproduced a cell | "Used by X" without permission. SST overhead figures before tailtune measures them |
| M3 | Outside reports and citations, linked | Production use not confirmed in writing |
| M4 | Named production users, with permission | — |

**Never, at any milestone (D-02, D-03):**

- "no forgetting" or "without forgetting"
- any safety claim
- "faster than Unsloth"
- "first to offer" anti-forgetting training
- a scaling path that has not run end to end
- an SST figure without its attribution
- a result shaped by a sponsor

## 11. Business model options

These are the models seen among comparable projects:

| Model | Example |
|---|---|
| Open core with an AGPL UI | Unsloth: Apache-2.0 core, AGPL-3.0 Studio [S](https://github.com/unslothai/unsloth/blob/main/README.md) |
| Mixed licence with a commercial product | LiteLLM, licence reported by GitHub as NOASSERTION [S](https://api.github.com/repos/BerriAI/litellm). Its revenue figure is unverified [U] |
| Venture-backed company around the project | Inferact, Ollama |
| Absorbed by a larger company | ggml.ai into Hugging Face. Astral's acquisition by OpenAI, announced, with closing unconfirmed [S](https://openai.com/zh-Hant/index/openai-to-acquire-astral/) |
| Paid support by contact | Axolotl's README email [S](https://github.com/axolotl-ai-cloud/axolotl) |
| Sponsored developer tools as advertising | Warp in LlamaFactory |

**The order to consider them [J]:**

1. **Sponsorship**, in kind first and then cash (sections 7 and 8).
2. **Paid support or training contracts**, bounded and disclosed. Examples: team onboarding, custom retention sets, and help wiring `--max-regression` into a company's CI.
3. **A hosted or UI product**, only at M4 and only with a team. Hosted training was not researched, so its viability is unassessed. The category's largest peer moved into a local desktop product rather than hosted training.

**The promise (D-25).** The `tailtune` package stays Apache-2.0. That covers every module named in the [plan index](README.md#shared-vocabulary), including `tailtune.eval` and the forgetting report. No feature moves from open to paid. Any commercial product is a separate package built on the public API.

DCO without a CLA means contributors' code cannot be relicensed without them. DCO alone also does not stop the owner licensing their own new code differently, which is how Unsloth split off its UI. Any such product would therefore live in its own repository and would never take code out of the core.

## 12. Conflict-of-interest rules for benchmarks

These go into the benchmark README and `GOVERNANCE.md` before the first sponsor [J], except where marked as a locked decision:

1. **Disclose.** Sponsors are disclosed beside any benchmark that features their hardware, model or product (D-25).
2. **Preregister.** The benchmark is preregistered in a dated commit before any runs (D-22), and settings do not change afterwards without a public, dated amendment.
3. **Equal access.** No sponsor sees results early. Pre-briefs go to every affected method author at the same time, with the same content.
4. **No sponsor edits.** Sponsors cannot request reruns, exclusions or setting changes. Corrections arrive as public issues and apply to every arm.
5. **Own methods labelled.** tailtune's own methods, such as `tail-lora` and its SST implementation, are labelled as tailtune's. The benchmark includes settings where tailtune is expected to lose, plus the capacity-matched LoRA control ([09](09-evaluation.md)).
6. **Donated hardware.** Results record the GPU and provider. tailtune publishes no rankings of hardware vendors.
7. **Maintainer interests.** Maintainers list their employment and any paid relationship with vendors in `MAINTAINERS.md`, and step back from choosing settings for arms involving those vendors.
8. **Leaderboard.** Inclusion criteria are public, anyone can submit a run that follows the repro, and placement is never sold.
9. **Corrections log.** Corrections are public and dated.
10. **Pre-release access.** A model-release post written with early access from a lab says so in its first paragraph.

## 13. Tax and legal

- GitHub Sponsors needs a W-8BEN, and Indian residents can use their PAN as tax ID [S](https://docs.github.com/en/sponsors/receiving-sponsorships-through-github-sponsors/tax-information-for-github-sponsors). Settlement in INR is not officially confirmed [U].
- The research did **not** confirm GST treatment, whether FCRA applies to foreign donations, or remittance paperwork such as Form 15CA/CB and FIRC. Consult a chartered accountant before accepting any money.
- Whether to incorporate affects NVIDIA Inception and invoicing (section 7).
- Licence remarks in this plan are engineering notes, not legal advice.

## Implementation checklist

**Gate 0, Claim**

- [ ] Write `GOVERNANCE.md`, `MAINTAINERS.md`, `CODE_OF_CONDUCT.md`, `CONTRIBUTING.md` (with DCO), `SUPPORT.md` and `CODEOWNERS`. Bring the GitHub community profile to 100%.
- [ ] Turn on Discussions with the five categories, and add the issue templates.
- [ ] Apply the [01](01-product-strategy.md) description and topics once the owner approves them.
- [ ] Send the private message to the SST authors.
- [ ] Claim the X handle and the Discord server name.

**Gate 1, Verify**

- [ ] Send results privately to the SST authors, wait 7 days, then reply publicly in their thread.

**Gate 2, v0.1 Door**

- [ ] Consult a chartered accountant, then open GitHub Sponsors and add `FUNDING.yml`.
- [ ] Keep 5 to 10 good first issues open. Add the RFC category, template and `rfcs/`.
- [ ] Publish the migration pages: AutoTrain, checking an Unsloth fine-tune, PEFT `OSFConfig`.
- [ ] Build the model-release post template, fixed dataset and notebook template, and publish the first post.
- [ ] Apply to FOSS United and AMD when M1's signals are met.

**Gate 3, v0.2 Proof**

- [ ] Publish the tech report on arXiv and add it to `CITATION.cff`.
- [ ] Add the conflict-of-interest rules to the benchmark README.
- [ ] Execute the v0.2 launch runbook. Hold the T+14 retrospective.
- [ ] At M2: publish the tier page and the one-page pitch, and apply to the GitHub Secure Open Source Fund and FLOSS/fund.

**Gate 4, v0.3 Reach**

- [ ] Publish the leaderboard with public inclusion rules, and the Day-N table with measured dates.
- [ ] Recruit a second maintainer actively. Decide on Discord against its trigger.

**Gate 5, v1.0**

- [ ] Name the second maintainer as co-owner of every namespace. Turn on "prevent self-review".
- [ ] Apply to the PyTorch Ecosystem, or earlier if M1's bars were met.
- [ ] Make the incorporation decision at M3 or M4.

## How we will know it is working

All targets are judgement [J].

| Signal | Target |
|---|---|
| Model-release posts published | 1 to 2 a month from Gate 2 |
| Forgetting reports in model cards not written by the maintainer | At least 10 by the end of Gate 4 (as in [01](01-product-strategy.md)) |
| Outside benchmark submissions or corrections | At least 1 by the end of Gate 4 |
| A method author engaging with published results | At least 1 |
| Contributors with 3 or more merged PRs | 3 by Gate 4 |
| Median first response to new issues | Under 48 hours |
| Second maintainer named | By Gate 5 |
| Ladder signals | M1 by Gate 3, M2 by Gate 4 |
| Sponsor conversations started | At least 3 by M2 |
| Untriaged issues | Never above 50 for more than 2 weeks |

## Open questions

1. What do corporate sponsors actually require? No sponsor stated requirements in the research.
2. Does GitHub Sponsors settle in INR for an individual, and how are GST, FCRA and remittance handled? A chartered accountant needs to confirm this.
3. Are FOSS United's eligibility rules, the GitHub Secure Open Source Fund's 2026 terms, FLOSS/fund's grant sizes and a16z's intake compatible with an individual in India?
4. Is r/LocalLLaMA still the right channel? Reddit could not be read during the research.
5. Can a non-ML partner be recruited before an ML one, and would that satisfy the PyTorch "core maintainer" bar?
6. Does the PyTorch working group require the Linux Foundation code of conduct?
7. Is hosted training viable as a business? Not researched.
8. None of the programmes in the "not researched" list has been assessed.
