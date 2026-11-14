# 06. Models

**What this document decides:**

- Which models tailtune supports, in what order, and with what claim. Eight launch models follow D-20: six form the public claim, and two stay experimental until FSDP2 lands.
- Training targets come from a role resolver over the loaded module tree, not from a name list. It scopes to the text sub-model, uses a registry keyed by `model_type`, falls back on structure, treats `[E,m,n]` tensors as E matrices, and records both in-memory and checkpoint names (D-15).
- `tailtune inspect` is mandatory. `train` refuses with exit code 4 when coverage falls below a floor of 90% [J], when any matrix-shaped parameter is unresolved, or when nothing would train.
- Routers, conv kernels, `mtp` heads, norms, non-matrix scalars and vision or audio towers are frozen by default. Embeddings and `lm_head` stay frozen provisionally until the Gate 1 ablation decides. An `sst_paper` preset reproduces SST (Chavan et al., 2026) exactly.
- Every model and mode pair carries a "runs" or "retention benchmarked" label (D-02, D-16).
- Pre-computed bases go on the Hub only after legal review, starting with Apache-2.0 and MIT models.

Evidence labels are defined in the [plan index](README.md#evidence-labels). Hub figures come from API queries run on 2026-11-07 and 2026-11-08. Tensor names and shapes were read from safetensors headers and `config.json` in each repository, not from model cards.

## 1. The market in November 2026

A tool built around Llama-shaped models would miss most of the 2026 market. In one snapshot of the text-generation top 300 by downloads on 4 November 2026, the family shares were Qwen 50.9%, GPT/OPT 15.4%, DeepSeek 6.7%, Llama 5.3%, GLM 3.4%, Nemotron 3.2% and Gemma 2.5% [S](https://dev.to/ai_openfree_b23025ef075cf/qwen-still-takes-half-of-open-llm-downloads-what-changed-in-the-hugging-face-top-300-since-1gbn). By size, models under 3B held 30.9%, 3B to 10B 24.6%, 10B to 70B 27.3% and above 70B 12.2%. By format, full precision held 60.1%, GGUF 17.4% and FP8 8.3%.

Treat these shares with care. The author's organisation maintains the leaderboard the figures come from and says so, the figures come from its own snapshots, and gpt2's traffic is suspected to be automated. Raw downloads mislead in any case. `openai-community/gpt2` is the most downloaded text-generation repository at 31.3M, ahead of Qwen3-0.6B at 21.4M [S](https://huggingface.co/api/models?pipeline_tag=text-generation&sort=downloads&direction=-1&limit=60). The Hub counts every `GET` and `HEAD` request on the query file [S](https://huggingface.co/docs/hub/models-download-stats). The ranking also hides composite models. Qwen3.8-27B (7.8M) and gemma-4-31B-it (7.4M) are tagged `image-text-to-text`, so they never appear in it [S](https://huggingface.co/api/models/Qwen/Qwen3.8-27B).

**Fine-tune tags are a better signal of what people fine-tune.** These are Hub `base_model` tag counts on 2026-11-08 [S](https://huggingface.co/models-json?other=base_model:finetune:Qwen/Qwen3-8B&withCount=true):

| Base model | Tagged fine-tunes |
|---|---|
| Llama-3.1-8B-Instruct | 3,305 |
| Qwen3-8B | 2,304, plus 2,450 tagged adapters |
| Llama-3.2-3B-Instruct | 2,049 |
| Qwen3.5-9B | 1,293 |
| Qwen3-4B | 1,231 |
| Qwen3.8-27B | 843 |
| gpt-oss-20b | 575 |
| gemma-4-E4B-it, E2B-it, 26B-A4B-it | 461, 438, 236 |
| SmolLM3-3B | 178 |
| phi-4 | 130 |
| Mistral-Small-3.2-24B | 67 |
| granite-4.1-8b | 28 |

Tagging is voluntary, so these counts undercount. Downloads exclude ModelScope and private mirrors. Tool tags from the same endpoint are `peft` 274,691, `unsloth` 147,002 and `axolotl` 50,022, which is the audience `tailtune eval` serves ([09](09-evaluation.md)).

## 2. Families

DL is 30-day downloads. Parameter counts come from Hub safetensors metadata. "OSF map" means PEFT's OSF target mapping, which covers 17 model types (llama, llama4, mistral, mixtral, gemma, gemma2, gemma3_text, qwen2, qwen3, phi, gpt2, bloom, opt, gptj, gpt_neox, falcon, gpt_bigcode). Unlisted types fall back to `all-linear` [S](https://github.com/huggingface/peft/blob/main/src/peft/utils/constants.py). The Unsloth column comes from its catalogue, which has no formal support matrix. Treat GGUF-only rows there as inference-oriented [S](https://unsloth.ai/docs/get-started/all-our-models).

| Family | Top models: params (DL) | Licence terms that affect fine-tuning or redistribution | Attention and expert shape | Unsloth | OSF map | tailtune phase |
|---|---|---|---|---|---|---|
| [Qwen](https://huggingface.co/Qwen) | Qwen3-0.6B 0.75B (21.4M); Qwen3-14B 14.8B (13.0M); Qwen3.8-27B 27.8B (7.8M); Qwen3-8B 8.2B (7.0M); Qwen3-4B 4.0B (6.4M); Qwen3.5-9B 9.65B (6.3M); Qwen3.6-35B-A3B 35.95B (4.3M) | Apache-2.0, not gated. Qwen3.8-Flash-Next uses `qwen-community-1.0` and Qwen3.8-2.4T uses `qwen3.8-max` [U] | Qwen3 dense, tied at 0.6B. Qwen3.8 hybrid with fused QKV, a vision tower and `mtp`. Qwen3.6 has 3D experts and a shared expert | Qwen3 and Qwen3.8-27B in 4-bit or NVFP4; Qwen3.5 GGUF only | `qwen3` yes; Qwen3.5 and later no | 1, 2, 3 |
| [Gemma](https://huggingface.co/google) | 31B-it 31.3B (7.4M); 26B-A4B 25.8B (5.6M); E4B 8.0B (4.2M); E2B 5.12B (3.2M); 12B 11.96B (1.8M) | Gemma 4 Apache-2.0, not gated. Gemma 3 uses `gemma` terms with a manual gate [U] | Composite with vision and audio towers, tied, sliding window. 26B has 3D experts and a router | Gemma 4 in 4-bit or NVFP4 | `gemma3_text` yes; `gemma4` no | 1, 2 |
| [Llama](https://huggingface.co/meta-llama) | 3.1-8B-Instruct 8.0B (4.3M); 3.2-1B 1.2B; 3.2-3B 3.2B; 3.3-70B 70.6B; 4-Scout 108.6B | Community licences with a manual gate. A redistributed derivative needs the licence copy, "Built with Llama", "Llama" at the start of its name and a Notice file. Above 700M MAU a separate licence is needed. "Derivative works" is not defined [S](https://dev.meta.ai/llama/llama3_1/license) | Dense | 4-bit | `llama` yes | 1, 2 |
| [Muse](https://huggingface.co/meta-models) | Muse-Glimmer-30B 29.78B; Spark-Open-816B 822B | Glimmer Apache-2.0. Spark uses `muse-model-license` with a gate [U]. The org's link to Meta was not verified | Glimmer: attention `gate_proj` collides with MLP `gate_proj` | Glimmer 4-bit | no | 2 |
| [Mistral](https://huggingface.co/mistralai) | 7B-Instruct-v0.3 7.25B (1.7M); Ministral-3-14B 13.95B; Small-3.2-24B 24.0B; Large-4-1T | Mostly Apache-2.0. Large 4 uses a modified-MIT ecosystem licence [U](https://huggingface.co/mistralai/Mistral-Large-4-1T-A52B) | Not read in the research | Small, Magistral, Devstral, Ministral 3 in 4-bit | `mistral` yes; `mistral3` no | 2 |
| [Phi](https://huggingface.co/microsoft) | Phi-4-mini 3.84B; phi-4 14.66B; Phi-3.5-mini 3.82B | MIT, not gated | Phi-4-mini has fused `qkv_proj` and `gate_up_proj` | Phi-4, Phi-3 | `phi` yes; `phi3` no | 2 |
| [GLM](https://huggingface.co/zai-org) | 5.3-Flash 321.3B (6.4M); 5.3 753.3B; 4.5-Air 110.5B; 4.7-Flash 31.2B | Mostly MIT. GLM-5.3 permits fine-tuning and derivatives, but MaaS operators above $10B revenue need Z.AI security review [S](https://huggingface.co/zai-org/GLM-5.3/blob/main/LICENSE) | Per-expert modules, a router named `mlp.gate`, factorised attention, official FP8 | GGUF only | no | 2, 3, multi-node |
| [DeepSeek](https://huggingface.co/deepseek-ai) | V3.2 685.4B; V4-Flash-0731 304.2B; V4.1-Flash 763.2B; R1-Distill-Qwen-1.5B 1.8B | MIT, not gated | Bare `layers.N` prefix, factorised attention, FP8 and int8 official weights | V4 GGUF only | no | 3, multi-node |
| [Kimi](https://huggingface.co/moonshotai) | K3 2.8T; K2.5 and K2.6 1.0T; Kimi-Linear-48B-A3B 49.1B | K3: MaaS above $20M revenue needs an agreement, and products above 100M MAU or $20M monthly revenue must display "Kimi K3" [S](https://huggingface.co/moonshotai/Kimi-K3/blob/main/LICENSE). K2.x modified-MIT [U]. Kimi-Linear MIT | K3 weights could not be scanned [U]. Custom code | K3 4-bit; K2.x GGUF only | no | 2, multi-node |
| [gpt-oss](https://huggingface.co/openai) | 20b 20.9B (5.9M); 120b 116.8B (2.4M) | Apache-2.0, not gated | MoE with MXFP4 experts, sinks, sliding window, untied | 20B, 120B | no | 1 experimental, 3 |
| [Granite](https://huggingface.co/ibm-granite) | 4.1-3b 3.4B; 4.1-8b 8.8B; 4.2-8b 8.8B | Apache-2.0, not gated | Dense, untied | Not on the page read | no (`granite`) | 1 |
| [SmolLM](https://huggingface.co/HuggingFaceTB) | SmolLM2-135M 134.5M (2.5M); SmolLM3-3B 3.1B | Apache-2.0, not gated | Not read beyond size [U] | SmolLM2 | `model_type` not recorded [U] | 1, 2 |
| [Nemotron](https://huggingface.co/nvidia) | Nano-4B 4.0B (1.9M); Nano-30B-A3B 31.6B; Super-120B 123.6B; Ultra-550B 560.5B | Nemotron Open Model Licence and OpenMDW-1.1 [U] | `backbone.*` prefix, Mamba hybrid, custom code, untied | Super NVFP4; others GGUF only | no (`nemotron_h`) | 2 |
| [MiniMax](https://huggingface.co/MiniMaxAI) | M2.5 228.7B; M3 427B | M3: the grant is non-commercial. Commercial use, explicitly including fine-tuned deployments, needs "Built with MiniMax M3", plus written authorisation above $20M yearly revenue [S](https://huggingface.co/MiniMaxAI/MiniMax-M3/blob/main/LICENSE). M2.x modified-MIT [U] | 256 per-expert modules, FP8, custom code | Not on the page read | no | 3, multi-node |
| Others | [Olmo-3-7B](https://huggingface.co/allenai/Olmo-3-7B-Instruct) 7.3B; [Apertus-8B](https://huggingface.co/swiss-ai/Apertus-8B-Instruct-2509) 8.1B; [LFM3-1.7B](https://huggingface.co/LiquidAI/LFM3-1.7B) | Olmo and Apertus Apache-2.0. LFM Open License v1.0 allows commercial use only below $10M annual revenue [S](https://huggingface.co/LiquidAI/LFM3-1.7B/blob/main/LICENSE) | LFM3: hybrid conv and attention, tied, custom code | n/a | no | 2, 3 |

Fully fine-tuned weights carry the base model's licence. Deltas derived from them plausibly do too, and whether bases count as derivative works is unknown. Both need legal review (section 8). tailtune copies the base licence identifier into the model card and prints the obligations above at export. Compliance stays with the user ([10](10-export-and-deployment.md)).

## 3. The quirks catalogue

SST and PEFT's OSF were designed around Llama-shaped `q/k/v/o/gate/up/down` `nn.Linear` matrices [S](https://arxiv.org/html/2610.22293). Most 2026 flagships break that assumption. Names and shapes below come from safetensors headers [S]. N is a layer or expert index.

| Quirk | Concrete names and shapes | What naive code does wrong | What tailtune does |
|---|---|---|---|
| Fused QKV | [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) `linear_attn.in_proj_qkv.weight [10240,5120]`; [Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) `in_proj_qkv [8192,2048]`; [Phi-4-mini](https://huggingface.co/microsoft/Phi-4-mini-instruct) `qkv_proj [5120,3072]` and `gate_up_proj [16384,3072]`; [Nemotron Nano 4B](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16) `mixer.in_proj [17504,3136]`; legacy `c_attn`, `query_key_value`, `Wqkv` | A `q_proj` name list matches nothing, so the matrix trains unconstrained. One SVD shares the preserved rank across Q, K and V | Role `attn_fused` or `mlp_fused`. The SVD unit is the checkpoint tensor, so a tensor fused in the checkpoint gets one whole-matrix SVD. Block-split SVD is an experiment, since no paper read evaluated either option |
| Fused at load by Transformers v5 | The migration guide's own example fuses `q_proj`, `k_proj` and `v_proj` into `qkv_proj` with `Concatenate(dim=0)` [S](https://raw.githubusercontent.com/huggingface/transformers/main/MIGRATION_GUIDE_V5.md) | In-memory names differ from checkpoint keys. Bases keyed on one space miss the other, and a whole-matrix SVD silently departs from the paper | Record both key spaces (section 4.6). When the checkpoint has separate tensors, use one SVD per row block |
| 3D MoE experts | [gemma-4-26B-A4B](https://huggingface.co/google/gemma-4-26B-A4B-it) `experts.down_proj [128,2816,704]` and `experts.gate_up_proj [128,1408,2816]`; Qwen3.6 `mlp.experts.gate_up_proj [256,1024,2048]` and `down_proj [256,2048,512]`; [Inkling-Small](https://huggingface.co/thinkingmachines/Inkling-Small) `mlp.experts.w13_weight [256,4096,4096]` | An `ndim == 2` or `nn.Linear` filter skips them silently, so the experts, which are most of the model, train unconstrained | Treat `[E,m,n]` as E matrices: one SVD per expert, bases keyed `name[e]`, and per-expert free tails. Report the share of experts routed on the training data |
| Per-expert modules | [MiniMax-M2.5](https://huggingface.co/MiniMaxAI/MiniMax-M2.5) `block_sparse_moe.experts.N.w1.weight [1536,3072]`, 256 experts in 62 layers; [GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash) `mlp.experts.N.{gate,up,down}_proj` | The matrix count explodes. Experts with no gradient on a batch still move through Adam momentum or weight decay | Role `expert`. Projecting the applied update still holds for unrouted experts. Bases are batched per layer |
| Routers with too small a tail | gpt-oss `mlp.router.weight [32,2880]`; Gemma 4 `router.proj.weight [128,2816]`; GLM `mlp.gate.weight`, 288-way; LFM3 attention gate `[16,2048]` | At r=0.99, k=⌊0.99·32⌋=31 leaves one free direction, so the router is "protected" but cannot learn, and the coverage figure misleads | Role `router`, frozen by default (D-15). Any constrained matrix with a free tail below `min_free_tail` is frozen too (section 4.8) |
| Name collisions | [Muse-Glimmer-30B](https://huggingface.co/meta-models/Muse-Glimmer-30B) `self_attn.gate_proj [4096,6656]` beside `mlp.gate_proj [19968,6656]`; [LFM3](https://huggingface.co/LiquidAI/LFM3-1.7B) `self_attn.gate_proj [16,2048]` while its MLP is `feed_forward.w1/w2/w3`; GLM `mlp.gate` (router) beside `gate_proj` | Suffix matching merges an attention gate into the MLP role, and misses LFM3's MLP entirely | Patterns are anchored on whole path segments (`(^|\.)mlp\.gate_proj\.weight$`). A snapshot test fails if any parameter matches more than one registry rule |
| Already factorised | [DeepSeek-V4-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731) `attn.wq_a [1024,4096]` then `wq_b [32768,1024]`, `wo_a [8192,4096]`, `wo_b [4096,8192]`; GLM `q_a_proj`, `q_b_proj`, `kv_a_proj_with_mqa` | A per-factor tail differs from the tail of the product, and the bottleneck makes "low energy" ambiguous | Role `factor`, frozen under constrained methods. Support is not claimed until a sweep exists |
| GPT-2 `Conv1D` | `transformers.pytorch_utils.Conv1D` stores weight as (in, out) [S](https://github.com/huggingface/peft/blob/main/src/peft/tuners/lora/layer.py) | U and V swap roles. PEFT's OSF on `main` raises `TypeError` for anything but `nn.Linear` [S](https://github.com/huggingface/peft/blob/main/src/peft/tuners/osf/model.py). By source reading, v0.21.2 skips them silently and leaves zero trainable parameters [S](https://github.com/huggingface/peft/blob/v0.21.2/src/peft/tuners/osf/model.py) | Rejected as a constrained target (D-15) with a clear error. GPT-2 survives only as a CI fixture |
| Depthwise `nn.Conv1d` | Qwen3.8 `linear_attn.conv1d.weight [10240,1,4]`; Nemotron `mixer.conv1d [9728,1,4]`; LFM3 `conv.conv.weight [2048,1,7]` | 3D, but not experts. Matrix SVD is meaningless, so they train unconstrained | Role `conv`, frozen by default (D-15). Recognised by the shape `[C,1,K]` |
| Tied embeddings | `tie_word_embeddings: true` on [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) (both keys stored at `[151936,1024]`), Gemma 4, LFM3. False on Qwen3.8-27B, gpt-oss-20b, Granite 4.2, Nemotron | The update is projected or counted twice. Freezing one alias leaves the other live | Group parameters by storage identity, count once, and freeze or train the pair together. A secondary source says v5 ties weights even when both keys exist [U](https://depscope.dev/breaking/pypi/transformers), so `inspect` compares the two stored tensors and warns if they differ |
| MXFP4 and FP8 official weights | gpt-oss experts as MXFP4 `down_proj_blocks [32,2880,90,16] U8`, `quantization_config: mxfp4`; MiniMax-M2.5 F8_E4M3 with `weight_scale_inv`; DeepSeek-V4 F8_E4M3 attention with F8_E8M0 `scale`, int8 experts | SVD of quantized codes is meaningless, and a fine update cannot be stored back losslessly | Role `quantized`. Constrained methods dequantize at load if the pinned Transformers version supports it for training, which is unverified for MXFP4 [U]. Otherwise they refuse and name a BF16 alternative where one exists, as with [GLM-5.3-Flash-BF16](https://huggingface.co/zai-org/GLM-5.3-Flash-BF16) |
| Composite `*ForConditionalGeneration` | Qwen3.8-27B and Gemma 4 load as composite classes. In v5, vision-language models lose the `model.language_model` shortcut for `model.model.language_model` or `get_decoder()`, and Qwen-VL configs nest under `config.text_config` [S](https://raw.githubusercontent.com/huggingface/transformers/main/MIGRATION_GUIDE_V5.md) | Loading a text-only class drops tower tensors, so export and GGUF conversion break. Reading `config.hidden_size` at the top level fails | Load the composite class, scope through `get_decoder()`, and read `text_config` (section 6.6) |
| Hybrid attention | Qwen3.8-27B `layer_types`: 48 `linear_attention` and 16 `full_attention` layers, with `linear_attn.*` and `self_attn.*` in different layers; Nemotron Nano 4B: Mamba in 21 layers, MLP in 17, attention in 4; LFM3: 13 conv and 11 attention | Coverage computed on layer 0 and multiplied by the layer count is wrong | Count roles per layer against `layer_types`, and report per-layer coverage |
| Sliding window | gpt-oss-20b window 128 (12 sliding, 12 full); gemma-4-E2B window 512 (28 and 7); gemma-4-26B window 1024 (25 and 5) | No matrix changes. The risk is evaluation using a different attention path before and after | Record the window in `targets.json`. Evaluation runs base and fine-tune through the same engine and attention implementation ([09](09-evaluation.md)) |
| `mtp` heads | Qwen3.8 `mtp.layers.N.*`, `mtp.fc.weight [5120,10240]`; Inkling `model.mtp.*`; Gemma `*-assistant` companion repos | Not in the standard loss, so they go stale and speculative decoding silently degrades | Role `mtp`, frozen (D-15). Saved unchanged at export, with a warning that they are stale |
| Vision and audio towers | Gemma 4 `model.vision_tower.*`, `model.audio_tower.*`, `model.embed_vision.embedding_projection.weight`; Qwen3.8 `model.visual.blocks.N.attn.qkv.weight [3456,1152]`; Gemma vision wraps `q_proj.linear.weight`, and its audio tower has clip scalars such as `k_proj.input_min` | A global `q_proj` match pulls tower matrices into training, and retention on images or audio is never measured | Everything outside the text scope gets role `tower` and is frozen. No retention claim is made for vision or audio inputs |
| Non-matrix state and prefixes | gpt-oss `self_attn.sinks [64]`; DeepSeek `hc_attn_fn [24,16384]`; Gemma `layer_scalar`; prefixes `backbone.*`, `model.llm.*`, bare `layers.N.*` | Regexes anchored on `model.layers` miss whole models | Role `scalar` is frozen. Patterns are relative to the text scope, never to `model.layers` |

Some models also need custom code: `auto_map` appears on Nemotron Nano 4B, LFM3, MiniMax-M2.5 and Kimi-K3. Module classes can then change without notice. D-08 keeps `trust_remote_code=False`, so these models run only through `--trust-remote-code` and never appear in a launch claim.

One useful identity covers orientation. Write W = U S Vᵀ, so Wᵀ = V S Uᵀ. SST's two-sided projection of Wᵀ is then exactly the transpose of the projection of W, because k=⌊r·min(m,n)⌋ is symmetric in m and n. An orientation mix-up therefore cannot corrupt the SST constraint. It does corrupt PEFT OSF, LoRA and adapter export, which need `(out, in)`. So the resolver records orientation anyway.

## 4. The role resolver

### 4.1 Contract

```text
resolve(model | meta_model, config, header | None, method, r, policy) -> Resolution
```

- **Inputs:** a loaded model or one built on the meta device from its config, the safetensors header (names, shapes, dtypes, read by range request), the method and its rank parameter, and the policy overrides.
- **Output:** one `Group` per parameter storage, with role, action, reason, shapes, SVD units, and in-memory and checkpoint names, plus totals, coverage and an optional refusal.
- **Weight-free:** the resolver never reads weight values, except to compare tied tensors on models small enough to do so. So `tailtune inspect` runs on a 700B model in seconds on a laptop.
- **Deterministic:** the same config, header, registry version and policy give byte-identical `targets.json`.

### 4.2 Roles and default actions

| Role | `lora` | `sst`, `osf` (default policy) | `sst_paper` preset | `full` |
|---|---|---|---|---|
| `attn_q`, `attn_k`, `attn_v`, `attn_o`, `mlp_gate`, `mlp_up`, `mlp_down` | adapter | constrain | constrain | free |
| `attn_fused`, `mlp_fused`, `linear` (structural) | adapter | constrain | free (the paper constrains names only) | free |
| `expert` (3D or per-module) | frozen at launch [J] | constrain (`sst`); refuse (`osf`, which needs `nn.Linear`) | constrain | free |
| `attn_gate`, `router`, `factor`, `conv`, `mtp`, `scalar`, `ssm` | frozen | frozen | free, except `mtp` | free |
| `norm` | frozen | frozen [J] | free | free |
| `embedding`, `head` | frozen | frozen until Gate 1 (section 5) | free | free |
| `tower` | frozen | frozen | frozen | frozen |
| `quantized` | frozen base under 4-bit LoRA | dequantize or refuse | refuse | dequantize or refuse |
| `unresolved` | refuse | refuse | refuse | refuse |

`freeze` trains the last N blocks and uses the same roles to report what moved. A user may override any row with `--role-action <role>=<action>`. Every override is written to `tailtune.yaml` and `lineage.json`, and printed in the report.

### 4.3 Registry data structure

Untested design proposal:

```python
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Literal

class Role(StrEnum):
    ATTN_Q = "attn_q"; ATTN_K = "attn_k"; ATTN_V = "attn_v"; ATTN_O = "attn_o"
    ATTN_FUSED = "attn_fused"; ATTN_GATE = "attn_gate"
    MLP_GATE = "mlp_gate"; MLP_UP = "mlp_up"; MLP_DOWN = "mlp_down"; MLP_FUSED = "mlp_fused"
    EXPERT = "expert"; ROUTER = "router"; FACTOR = "factor"; LINEAR = "linear"
    EMBEDDING = "embedding"; HEAD = "head"; NORM = "norm"; CONV = "conv"; SSM = "ssm"
    SCALAR = "scalar"; MTP = "mtp"; TOWER = "tower"; QUANTIZED = "quantized"
    UNRESOLVED = "unresolved"

@dataclass(frozen=True)
class Rule:
    id: str                              # stable, e.g. "gemma4.expert.down"
    pattern: str                         # regex on the name relative to the text scope, segment-anchored
    role: Role
    ndim: int | None = None              # required tensor rank, if any
    layer_types: tuple[str, ...] = ()    # restrict to layers of these config.layer_types values
    fused_blocks: tuple[str, ...] = ()   # roles of row blocks when fused in memory only

@dataclass(frozen=True)
class ArchEntry:
    model_types: tuple[str, ...]         # matched against text_config.model_type, then model_type
    rules: tuple[Rule, ...]              # first match wins; overlaps fail the snapshot test
    tower_prefixes: tuple[str, ...] = () # full-name prefixes outside the text scope
    expert_layout: Literal["e_out_in", "e_in_out"] | None = None
    reasoning_delimiters: tuple[str, str] | None = None   # used by tailtune.eval
    expect: dict[Role, str] = field(default_factory=dict) # count expressions over config
    status: Literal["launch", "experimental", "phase2", "phase3"] = "phase2"
    verified_against: str = ""           # Transformers version and header date
```

### 4.4 Example entries

These are original, untested design proposals. Names quoted in section 3 come from headers. Anything marked "confirm" must be checked against the header and modelling code at implementation time.

```python
QWEN3 = ArchEntry(
    model_types=("qwen3",),
    rules=(
        Rule("qwen3.q", r"(^|\.)self_attn\.q_proj\.weight$", Role.ATTN_Q, 2),
        Rule("qwen3.k", r"(^|\.)self_attn\.k_proj\.weight$", Role.ATTN_K, 2),
        Rule("qwen3.v", r"(^|\.)self_attn\.v_proj\.weight$", Role.ATTN_V, 2),
        Rule("qwen3.o", r"(^|\.)self_attn\.o_proj\.weight$", Role.ATTN_O, 2),
        Rule("qwen3.gate", r"(^|\.)mlp\.gate_proj\.weight$", Role.MLP_GATE, 2),
        Rule("qwen3.up", r"(^|\.)mlp\.up_proj\.weight$", Role.MLP_UP, 2),
        Rule("qwen3.down", r"(^|\.)mlp\.down_proj\.weight$", Role.MLP_DOWN, 2),
    ),
    expect={Role.ATTN_Q: "num_hidden_layers", Role.MLP_DOWN: "num_hidden_layers"},
    reasoning_delimiters=("<think>", "</think>"),
    status="launch",
)

GEMMA4 = ArchEntry(
    model_types=("gemma4",),             # confirm the text_config.model_type value
    tower_prefixes=("model.vision_tower.", "model.audio_tower.", "model.embed_vision."),
    rules=(
        Rule("gemma4.router", r"(^|\.)router\.proj\.weight$", Role.ROUTER, 2),
        Rule("gemma4.expert.gate_up", r"(^|\.)experts\.gate_up_proj$", Role.EXPERT, 3),
        Rule("gemma4.expert.down", r"(^|\.)experts\.down_proj$", Role.EXPERT, 3),
        Rule("gemma4.scalar", r"(^|\.)layer_scalar$", Role.SCALAR),
        # Text attention and dense MLP names: confirm against the header.
        Rule("gemma4.q", r"(^|\.)self_attn\.q_proj\.weight$", Role.ATTN_Q, 2),
        Rule("gemma4.v", r"(^|\.)self_attn\.v_proj\.weight$", Role.ATTN_V, 2),
    ),
    expert_layout="e_out_in",            # inferred from checkpoint shapes; confirm in memory
    status="launch",                     # E4B; 26B-A4B is phase 2
)

GPT_OSS = ArchEntry(
    model_types=("gpt_oss",),
    rules=(
        Rule("gpt_oss.router", r"(^|\.)mlp\.router\.weight$", Role.ROUTER, 2),
        Rule("gpt_oss.sinks", r"(^|\.)self_attn\.sinks$", Role.SCALAR, 1),
        Rule("gpt_oss.mxfp4", r"_(blocks|scales)$", Role.QUANTIZED),
        # Dequantized expert names and layout depend on the loader: confirm. [U]
        # Attention projections under self_attn.*: confirm against the header.
    ),
    status="experimental",
)

QWEN3_8_HYBRID = ArchEntry(
    model_types=("qwen3_5",),            # assumed for Qwen3.8-27B; confirm from config.json [U]
    tower_prefixes=("model.visual.",),
    rules=(
        Rule("q38.lin.qkv", r"(^|\.)linear_attn\.in_proj_qkv\.weight$", Role.ATTN_FUSED, 2,
             layer_types=("linear_attention",)),
        Rule("q38.lin.conv", r"(^|\.)linear_attn\.conv1d\.weight$", Role.CONV, 3,
             layer_types=("linear_attention",)),
        Rule("q38.mtp", r"^mtp\.", Role.MTP),  # outside the decoder; matched on the full name
        # Remaining linear_attn.* and self_attn.* names: confirm against the header.
    ),
    status="experimental",
)
```

### 4.5 Matching algorithm

1. **Load the config.** Read `text_config` if present, otherwise the top-level config. Take the registry key from `text_config.model_type`, falling back to `model_type`. If no entry exists, continue in structural-only mode and mark `registry: none` in the output.
2. **Find the text scope.** Call `model.get_decoder()` and record its module path as `text_prefix`, for example `model.language_model.`. Identify the input and output embeddings through `get_input_embeddings()` and `get_output_embeddings()`, since `lm_head` usually sits outside the decoder. Every other parameter outside `text_prefix` is a `tower` if it matches `tower_prefixes`, an `mtp` if a rule says so, and otherwise also a `tower`, frozen and listed.
3. **Enumerate storage.** Walk `named_parameters(remove_duplicate=False)` and group names by storage identity. The first name in module order becomes canonical, and the others become aliases. Compare the storage groups with `tie_word_embeddings`. A mismatch is a warning on `inspect` and a refusal on `train`.
4. **Assign identity roles.** The input embedding group gets `embedding`, and the output embedding gets `head`. A tied pair gets both roles on one group.
5. **Assign quantized roles.** A non-floating dtype (U8, I8, F8 formats in the header) or a quantization companion name (`_blocks`, `_scales`, `weight_scale_inv`, `scale`) gets `quantized`.
6. **Apply registry rules.** Take the name relative to `text_prefix` and try the rules in order. A rule matches only when its pattern, `ndim` and `layer_types` all hold. The layer index is parsed from the first integer path segment. First match wins, and the `Rule.id` is recorded as `matched_by`.
7. **Apply structural rules** to anything still unmatched:
   - 0D or 1D: `norm` if the name contains `norm`, otherwise `scalar`.
   - 3D of shape `[C,1,K]`: `conv`.
   - 3D of shape `[E,m,n]`, where E equals the expert count in the config and `min(m,n) >= 256`: `expert`.
   - 2D with one dimension equal to the expert count: `router`.
   - 2D on a `Conv1D` module: `conv1d_transposed`, which is refused for constrained methods.
   - 2D with `min(m,n) >= 256` [J]: `linear`.
   - 2D below 256: `scalar`, frozen.
   - Anything else, including 3D tensors whose leading dimension is not the expert count and all 4D or higher tensors: `unresolved`.
8. **Check counts.** For each registry `expect` expression, compare the role count per layer type with the config. A mismatch means a rename upstream, and it fails the snapshot test and warns in `inspect`.
9. **Apply the method policy** from section 4.2 and any overrides, giving each group `constrain`, `adapter`, `free`, `frozen` or `refuse`, with a reason.
10. **Build SVD units** for `constrain` groups (section 4.6 and 4.7), and compute k, t and free values (section 4.8). Freeze units with `t < min_free_tail`. Mark a group `frozen` when all its units are frozen.
11. **Compute totals and coverage**, check the refusal conditions (section 4.9), and write `targets.json`.

### 4.6 Key spaces and load-time fusing

Each group records `name` (in memory) and `checkpoint_keys`. Each key comes with an optional slice, as in `{"key": "…self_attn.q_proj.weight", "rows": [0, 4096]}`. The bases cache and the Hub use checkpoint keys, which stay stable across Transformers versions. Training uses in-memory names. Export writes checkpoint keys, because the GGUF converter fails on unmapped tensors ([10](10-export-and-deployment.md)).

The mapping comes first from the conversion list the v5 loader applied. The loader is built around `WeightConverter` [S](https://github.com/huggingface/transformers/releases/tag/v5.0.0), but the public accessor for that list has not been identified [U]. The fallback compares in-memory shapes with header shapes:

- An exact name and shape match gives an identity mapping.
- An in-memory tensor whose rows equal the sum of several header tensors with matching columns, in the same module, gives a concatenation in header order, which must be confirmed by a registry `fused_blocks` hint.
- Anything else is ambiguous, gets role `unresolved` and is refused.

**The SVD unit is the checkpoint tensor** [J]. When v5 fuses separate q, k and v at load, tailtune runs three SVDs on row-block views of the fused parameter. That keeps results identical to the paper's per-matrix setup and independent of the Transformers version. PEFT OSF works on the in-memory `nn.Linear`, so on such models `osf` and `sst` use different SVD units. `inspect` says so.

### 4.7 3D tensors

A `[E,m,n]` parameter yields E units, `name[0]` to `name[E-1]`, each an m×n matrix with its own bases from batched SVD (D-14 covers precision and caching). Projection applies per unit to the applied update, as described in [03](03-methods.md). Unrouted experts still move through Adam momentum and weight decay, and projection of the applied update still bounds them [S](https://arxiv.org/html/2610.22293). During training, a forward hook on each router counts how often each expert is selected. `lineage.json` stores the routed fraction per layer, and the report prints it. `expert_layout` records whether the in-memory layout is `[E,out,in]` or `[E,in,out]`. The shapes in section 3 suggest `[E,out,in]` in checkpoints. Whether Gemma 4 experts are raw `nn.Parameter` in memory is unconfirmed [U].

### 4.8 Free tail size

For each SVD unit of shape m×n at preserved ratio r:

```text
q           = min(m, n)
k           = floor(r · q)          # computed with exact rationals from the decimal string of r
t           = q − k                 # free tail size; bounds the rank of the unit's update
free_values = (m − k) · (n − k)     # dimension of the complement both projections leave
```

Use `fractions.Fraction(str(r))`, because floating point gives `0.99 * 100 = 98.99999999999999`, which would floor to 98 instead of 99. If r is outside (0, 1), the config is rejected before any work (see [04](04-api-and-ux.md) for the error code). For `osf`, k comes from PEFT's preserved rank, which defaults to 50% of the smaller dimension [S](https://github.com/huggingface/peft/blob/main/src/peft/tuners/osf/config.py).

**Minimum free tail: `min_free_tail = 4`** [J]. A constrained unit with t < 4 is frozen with reason `tail_too_small`. It would otherwise cost bases, projection and Adam state while barely being able to change, and it would inflate the "protected" share. The value is a judgement. Gate 2 should log how many units it freezes on each launch model.

Worked values at r=0.99, computed from header or config shapes [C]:

| Unit | Shape | k | t | Free values | At `min_free_tail=4` |
|---|---|---|---|---|---|
| Qwen3-8B `q_proj` | 4096×4096 | 4,055 | 41 | 1,681 | constrain |
| Qwen3-8B `k_proj` | 1024×4096 | 1,013 | 11 | 33,913 | constrain |
| Qwen3-8B `gate_proj` | 12288×4096 | 4,055 | 41 | 337,553 | constrain |
| Qwen3.8-27B `in_proj_qkv` | 10240×5120 | 5,068 | 52 | 268,944 | constrain |
| gemma-4-26B expert `down_proj` (one of 128) | 2816×704 | 696 | 8 | 16,960 | constrain |
| Qwen3.6 expert `down_proj` (one of 256) | 2048×512 | 506 | 6 | 9,036 | constrain |
| Muse-Glimmer `k_proj` | 256×6656 | 253 | 3 | 19,209 | frozen |
| gpt-oss router | 32×2880 | 31 | 1 | 2,849 | frozen by role anyway |

Across Qwen3-8B's seven target types, these shapes give 39.0M free values out of 6.95B constrained parameters, or 0.56%. That matches the research's figure [C]. On the rectangular `gate_proj`, almost all free directions sit in the left null space, so docs say "the complement of the top-k singular subspaces", not "the low-energy tail".

### 4.9 Coverage, the floor and the refusal

Over the text sub-model plus embeddings and head, with tied storage counted once:

```text
protected = parameters in constrained units          (requires_grad, projected)
free      = trainable parameters with no constraint  (requires_grad, not projected)
frozen    = parameters with requires_grad = False    (towers count separately, not here)
coverage  = protected / (protected + free)
```

Coverage measures leakage: the share of what trains that is constrained. **The default floor is 0.90** [J]. Under the default policy, every dense launch model should sit at or near 100%, so anything below 90% means something is training unconstrained that the user did not ask for. For scale, the paper's scope on Qwen3-8B leaves 15.2% of parameters unconstrained, giving 84.8% coverage [C]. Coverage applies to `sst`, `osf` and `tail-lora`. For `lora`, `full` and `freeze`, `inspect` prints the same table and no floor applies.

`train` refuses with **exit code 4**, before loading weights where possible, when any of the following holds:

1. Coverage is below the floor.
2. Any group is `unresolved`, unless `--freeze-unresolved` is passed.
3. A constrained method would have zero protected parameters.
4. A trainable target is quantized or `Conv1D`.

You can override the floor with `--coverage-floor 0.80`, `coverage_floor: 0.80` in `tailtune.yaml`, or `tt.train(..., coverage_floor=0.8)`. The value and where it came from go into `lineage.json`, and the report header shows it. The `sst_paper` preset sets the floor to 0 and prints coverage as a warning.

The refusal looks like this (illustrative, placeholder numbers):

```text
Refused: coverage NN.N% is below the floor of 90.0% (method sst, r=0.99).
N.NNB of N.NNB trainable parameters would train with no constraint:
  embedding  model.embed_tokens.weight  [V, H]   N.NNB   (policy override: embeddings=free)
  head       lm_head.weight             [V, H]   N.NNB
  norm       NN tensors                          N.NNM
Choose one:
  1. Remove the override and keep embeddings frozen.
  2. Accept the leak explicitly: --coverage-floor 0.80   (recorded in lineage and the report)
  3. Replicate the paper exactly: --preset sst_paper
Exit code 4.
```

### 4.10 `tailtune inspect` and `targets.json`

`tailtune inspect <model> [--method sst] [--r 0.99] [--revision SHA] [--json]` reads the config and header only. It writes `targets.json` and prints this (illustrative output, placeholder numbers):

```text
Model      example-org/example-8b @ <commit>     model_type <type>    registry <type> v1
Scope      text prefix "model."   towers: none   tied: no   quantized: none
Key space  in-memory = checkpoint (no load-time fusion)
Method     sst  r=0.99  min_free_tail=4  floor 90.0%

Role        Action     Units  Params   Unit shape     k      t    Free values
attn_q      constrain  NN     N.NNB    [m, n]         kkkk   tt   N.NNM
attn_k      constrain  NN     N.NNB    [m, n]         kkkk   tt   N.NNM
mlp_gate    constrain  NN     N.NNB    [m, n]         kkkk   tt   N.NNM
...
embedding   frozen     1      N.NNB    [V, H]         -      -    -
norm        frozen     NN     N.NNM    [H]            -      -    -

Protected NN.N%   Free N.N%   Frozen NN.N%   of N.NNB text parameters
Coverage NN.N% (floor 90.0%): OK
Units frozen for tail_too_small: N    Unresolved: 0
Warnings: none
```

The `targets.json` schema (design proposal):

```json
{
  "schema_version": 1,
  "tailtune_version": "str",
  "registry_version": "str",
  "model": {
    "id": "str", "revision": "sha", "model_type": "str", "text_model_type": "str|null",
    "architectures": ["str"], "text_prefix": "str", "tie_word_embeddings": "bool",
    "quantization": "str|null", "layer_types": ["str"], "sliding_window": "int|null",
    "trust_remote_code": "bool", "config_sha256": "hex", "header_sha256": "hex"
  },
  "method": {"name": "sst|osf|lora|full|freeze|tail-lora", "r": "decimal-string|null",
             "min_free_tail": "int", "preset": "str|null", "overrides": {"role": "action"}},
  "groups": [{
    "name": "str", "aliases": ["str"],
    "checkpoint_keys": [{"key": "str", "rows": "[int,int]|null", "cols": "[int,int]|null"}],
    "role": "str", "matched_by": "registry:<rule id>|structural:<rule>|identity",
    "action": "constrain|adapter|free|frozen|refuse", "reason": "str",
    "shape": ["int"], "dtype": "str", "layer": "int|null", "layer_type": "str|null",
    "numel": "int",
    "units": [{"index": "int|null", "shape": ["int", "int"], "k": "int", "t": "int",
               "free_values": "int", "frozen_reason": "str|null"}]
  }],
  "totals": {"text_params": "int", "protected": "int", "free": "int", "frozen": "int",
             "tower_params": "int", "unresolved": "int"},
  "coverage": {"value": "float", "floor": "float", "floor_source": "default|cli|config|preset",
               "passed": "bool", "per_layer": [{"layer": "int", "value": "float"}]},
  "warnings": ["str"],
  "refusal": {"exit_code": 4, "reasons": ["str"]}
}
```

Coverage and per-layer coverage are copied into `lineage.json` and into every forgetting report ([09](09-evaluation.md)).

## 5. Embedding and `lm_head` policy

SST constrains `{q,k,v,o,gate,up,down}_proj` and leaves embeddings and norms to the optimizer. It does not address `lm_head` [S](https://arxiv.org/html/2610.22293). Under that scope on Qwen3-8B, 15.2% of parameters train unconstrained. The embedding and head matrices hold about 1.245B values, against 39.0M free values inside the constrained matrices. Free-to-free degrees of freedom fall from 38.86% of the target matrices at r=0.5 to 0.56% at r=0.99 [C](https://huggingface.co/Qwen/Qwen3-8B/raw/main/config.json). On Qwen3.8-27B, `lm_head [248320,5120]` alone is about 1.27B parameters, roughly 4.6% of the model [C]. Gemma 4's `embed_tokens [262144,2816]` is about 738M [C]. So in sst_paper mode, the unconstrained route may carry more of the learning, and more of the forgetting, than the protected one. No ablation has tested this.

**Gate 1 ablation** (roadmap measurement 4). The protocol:

- Run SST at r=0.99 with no refresh, on the Gate 1 model and one domain.
- Compare (a) embeddings, head and norms free, as in the paper, against (b) all three frozen.
- Add (c), with only norms free, if budget allows.
- Use the same data, seed and settings for each, and score both in-domain and on the full retention suite.

The decision rule [J]: make frozen the default if in-domain accuracy under (b) is at least the accuracy under (a) minus the larger of 1 point and the in-domain 95% half-width. Otherwise make free the default, revisit the 0.90 floor in the README's locked decisions, and record the change there first. Repeat the ablation on a tied small model before claiming the result for small models, since embeddings are a much larger share there.

Until Gate 1 reports, the provisional default is frozen embeddings and head. New tokens go through a `trainable_tokens`-style mechanism rather than unfreezing the whole matrix [S](https://github.com/huggingface/peft/tree/main/src/peft/tuners). Tied pairs are frozen or trained together.

**`sst_paper` preset.** It exists for exact replication and does the following:

- Targets by the seven name suffixes only.
- Leaves embeddings, `lm_head` and norms trainable without constraint. `lm_head` training follows from full-parameter training when nothing freezes it. Confirm this against the public code at Gate 1 [U].
- Uses r=0.99 with `K=None`.
- Sets the coverage floor to 0.
- Takes the training recipe from [03](03-methods.md).

It warns when the model is not one the paper evaluated (Qwen3 1.7B to 32B, Qwen2.5-0.5B and 7B-Instruct, Olmo-3-7B-Think).

## 6. Launch tiers

### 6.1 Launch models (D-20)

| Model | Params | Licence | Why | Shape exercised | Claim at launch |
|---|---|---|---|---|---|
| Qwen3-8B | 8.2B | Apache-2.0 | Direct SST replication anchor, 2,304 fine-tunes | Dense baseline, untied | Public |
| Qwen3-4B | 4.0B | Apache-2.0 | Smaller budgets. SST's −3.6 at 4B must be shown honestly [S](https://arxiv.org/html/2610.22293) | Dense | Public |
| Qwen3-0.6B or SmolLM2-135M | 0.75B / 134.5M | Apache-2.0 | CI and laptop smoke test. Expect the weakest retention (−8.0 at 1.7B) | Tied embeddings | Public |
| Llama-3.1-8B-Instruct | 8.0B | Llama 3.1 community, gated | Most fine-tuned base, 3,305 fine-tunes. Tests the token, gating and licence-notice flow | Dense instruct | Public |
| Gemma-4-E4B-it | 8.0B | Apache-2.0 | Composite class in a hot family, absent from every subspace paper | Towers, tied, sliding window | Public |
| Granite-4.2-8b | 8.8B | Apache-2.0 | Enterprise audience, independent lineage. OSF used Granite 8B for SVD analysis [S](https://arxiv.org/html/2504.07097v1) | Dense, untied | Public |
| Qwen3.8-27B | 27.8B | Apache-2.0 | Most downloaded Qwen3.8 model, third among Qwen repos in the snapshot; 843 fine-tunes | Hybrid, fused QKV, vision tower, `mtp` | Experimental after FSDP2 |
| gpt-oss-20b | 20.9B | Apache-2.0 | Real MoE test, 575 fine-tunes | MoE, MXFP4, sinks, sliding window | Experimental after FSDP2 |

D-20 leaves the small-model choice open. The recommendation [J] is to claim **Qwen3-0.6B** publicly, because it is the second most downloaded text-generation model, shares the Qwen3 code path with the anchor and exercises tied embeddings. SmolLM2-135M would then serve as a nightly real-weights CI fixture with no public claim. The owner should confirm this. The experimental pair waits for FSDP2 because a 27B constrained run needs about eight 96 GiB cards by the memory formula [C] ([07](07-hardware.md), [08](08-distributed.md)). Neither shape appears in any published subspace result.

### 6.2 Phase 2 (after v0.3, toward v1.0) [J]

| Models | Reason |
|---|---|
| Qwen3.5 small (0.8B to 9B), Qwen3-14B, Qwen3.6-35B-A3B, Gemma-4-26B-A4B | Popular. The MoE pair brings 3D experts into the claim |
| Gemma-4-31B, Gemma-4-12B (`gemma4_unified`), Gemma 3 | Family completion. Gemma 3 terms need reading first |
| Llama-3.2-1B and 3B, Llama-3.3-70B | 2,049 fine-tunes on 3.2-3B. 70B tests a single-node sharded path |
| Mistral-Small-3.2, Ministral 3 | `mistral3` type, Apache-2.0 |
| Phi-4-mini, phi-4 | MIT. Fused `qkv_proj` and `gate_up_proj` checkpoints |
| Olmo-3-7B Instruct and Think, Apertus-8B, SmolLM3-3B | Fully open lineages. Olmo-3-7B-Think is SST's stress test |
| Muse-Glimmer-30B | The name-collision case |
| Nemotron-3-Nano 4B and 30B | Mamba hybrid, but custom code |
| GLM-4.7-Flash, Kimi-Linear-48B | Mid-size routes into large families |
| Sentence Transformers embeddings, Qwen3-Embedding, Qwen3-Reranker, Granite embeddings | Small, cheap SVD, and clean retention metrics (rank correlation, nDCG). SVD-tail methods are untested on encoders, so a fresh sweep is needed |

### 6.3 Phase 3 (after v1.0) [J]

| Models or area | Reason |
|---|---|
| GLM-5.3-Flash (BF16 repo), DeepSeek-V4-Flash, MiniMax-M2.5, Inkling-Small, Hy3, gpt-oss-120b, Qwen3.5-122B, LFM3 | Large MoE, factorised attention, FP8, custom code or restrictive licences. LoRA routes come first |
| Vision tower training | Only once multimodal retention can be measured |
| TTS and ASR, starting with Orpheus and Qwen3-TTS | Need WER and speaker-similarity retention metrics. Several top TTS repos report no `model_type` |

### 6.4 Too big until multi-node sponsorship

Constrained methods need at least 16 bytes per parameter before bases ([07](07-hardware.md)). An 8×180 GB node holds 1.44 TB, so above roughly 90B parameters a constrained run cannot fit one node even before bases and activations [C]. The following wait for multi-node sponsorship and an end-to-end multi-node run (D-19):

- DeepSeek-V3.2 (685.4B), R1 and V4.1-Flash (763.2B)
- GLM-5 and GLM-5.3 (about 753B)
- Kimi K2.5 and K2.6 (1.0T) and K3 (2.8T)
- MiniMax M3 (427B)
- Muse-Spark-Open (822B), Mistral-Large-4 and Inkling (952.4B)
- Nemotron Ultra (560.5B)
- Constrained runs on any phase 3 model above about 90B

### 6.5 Non-goals

- Quantized-only releases such as Ternary-Bonsai's GGUF and MLX 2-bit builds. They have no float source weights.
- Fine-tuning QAT repositories such as `gemma-4-*-qat-w4a16-ct`.
- GPT-2 beyond the `Conv1D` CI fixture.

### 6.6 Deferring vision at training level, not at loading level

Qwen3.5 and later, and Gemma 4, are composite. A text-only loader would drop tower tensors, so the merged export would no longer match the base's key set, and conversion would fail. So tailtune always loads the full class and trains only the text scope. Untested design proposal:

```python
import torch
from transformers import AutoConfig

def load_for_training(repo: str, revision: str, policy, resolver):
    config = AutoConfig.from_pretrained(repo, revision=revision)
    cls = composite_or_causal_class(config)       # the class named in config.architectures
    model = cls.from_pretrained(repo, revision=revision,
                                dtype=policy.load_dtype)   # torch.float32 for constrained methods (D-12)
    res = resolver.resolve(model, config, header=None, method=policy.method,
                           r=policy.r, policy=policy)
    for name, param in model.named_parameters():
        param.requires_grad_(res.action(name) in ("constrain", "adapter", "free"))
    towers = [n for n, p in model.named_parameters() if res.role(n) == "tower"]
    assert not any(p.requires_grad for n, p in model.named_parameters() if n in towers)
    return model, res
```

The same rule applies across the pipeline:

- `tailtune.data` refuses image and audio columns with a message naming the phase.
- `tailtune.bases` skips towers.
- `tailtune.export` writes tower tensors byte-identical to the base under their checkpoint keys.
- A CI test asserts that tower hashes are unchanged after training.
- `tailtune.eval` uses text prompts only, and the report says retention on image and audio inputs was not measured.
- Tower weights still count in `tailtune estimate`.

## 7. Modes and support labels (D-16)

Hub tag counts on 2026-11-08 were `trl` 163,368, `sft` 85,467, `grpo` 25,759, `dpo` 13,798, `ppo` 4,326, `orpo` 1,297, `reward-trainer` 1,075 and `kto` 444 [S](https://huggingface.co/models-json?other=sft&withCount=true). Tools write these tags automatically, so they are an imperfect measure.

TRL's `trl/trainer` on `main` has SFT, DPO, GRPO, KTO, reward, RLOO and distillation trainers. `trl/experimental` includes `orpo`, `kto`, `sdft`, `online_dpo` and `gkd`. Neither lists a PPO trainer [S](https://github.com/huggingface/trl/tree/main/trl/trainer).

| Mode | Trainer | Phase | Note |
|---|---|---|---|
| SFT | TRL `SFTTrainer` | Launch | The case SST evaluated |
| Continued pretraining | Same code path, raw `text` | Launch | Stresses fixed tail capacity on large token volumes |
| SDFT | TRL experimental `sdft` | Next | Whether it matches the SST paper's SDFT variant is unchecked [U] |
| DPO, GRPO | TRL `DPOTrainer`, `GRPOTrainer` | v1.0 (roadmap Gate 5) | No paper evaluates SST or OSF with them. DPO needs a reference model |
| ORPO, KTO | TRL experimental | On request | Low use |
| PPO | none | Not planned | TRL lists no trainer |

Projection wraps the optimizer step, so other TRL trainers should run mechanically. That is not evidence of retention. So there are two labels per model and mode pair, both required by D-02:

- **runs:** a CI test passes on the tiny random model of that architecture. A real-weights smoke run of at least 20 optimizer steps also completes with `targets.json`, a fast report and a merged export, recorded in the support matrix with the tailtune and Transformers versions.
- **retention benchmarked:** the pair appears in a published, preregistered benchmark with at least three seeds, with the learning gate passed ([09](09-evaluation.md)). The label names the methods covered.

Here is the planned status at the Gate 3 public launch, assuming every gate passes. **Nothing is benchmarked today.** The second benchmark model is a judgement shared with [09](09-evaluation.md).

| Model | SFT | CPT | SDFT | DPO | GRPO |
|---|---|---|---|---|---|
| Qwen3-8B | retention benchmarked (target: `full`, `lora`, capacity-matched LoRA, `osf`, `sst`) | runs | not offered | not offered | not offered |
| Gemma-4-E4B-it | retention benchmarked (target, same arms) | runs | not offered | not offered | not offered |
| Qwen3-4B | runs | runs | not offered | not offered | not offered |
| Qwen3-0.6B | runs | runs | not offered | not offered | not offered |
| Llama-3.1-8B-Instruct | runs | runs | not offered | not offered | not offered |
| Granite-4.2-8b | runs | runs | not offered | not offered | not offered |
| Qwen3.8-27B | not yet (Gate 4) | not yet | not offered | not offered | not offered |
| gpt-oss-20b | not yet (Gate 4) | not yet | not offered | not offered | not offered |

## 8. Pre-computed bases on the Hub

**Value.** A first constrained run must compute an SVD of every target matrix in fp32 on the CPU path (D-14). The research expects this to take hours per model [U]. A shared cache per model revision removes that wait, and it is a slow-to-copy community asset.

**Size.** On Qwen3-8B at r=0.99, the head form holds 9.94B entries, or 39.8 GB in fp32. That is more than the 16.4 GB bf16 checkpoint. Storing the tail on square sides cuts it to about 22.9 GB [C]. The full thin SVD (U, S and V for every matrix) is 10.04B entries, about 40.2 GB, which is only about 1% more than the r=0.99 head form, and it serves any r [C]. The proposal [J] is to publish the full thin SVD and let the local cache derive the compact r-specific form ([03](03-methods.md)).

**How it works:**

1. Use one Hub repository per base model and revision, sharded per layer. A `manifest.json` records the model id and commit, the checkpoint key for each unit, shapes, the SVD driver and dtype, the torch version, a per-file SHA-256 and a whole-set hash.
2. On a cache miss, tailtune downloads only the shards for the resolved units and verifies the hashes.
3. It recomputes one randomly chosen small unit locally and checks the subspace distance against tolerance (D-14).
4. It writes the bases hash to `lineage.json`.

A revision mismatch is a hard miss, never a fallback.

**Licence risk.** Singular vectors are computed from the weights, so bases are plausibly derivative works, the same as deltas. The Llama 3.1 licence uses "derivative works" without defining it, and attaches naming and notice terms to redistributed derivatives [S](https://dev.meta.ai/llama/llama3_1/license). Kimi K3, GLM-5.3 and MiniMax M3 attach conditions to derivatives. **Flag this for legal review before any upload (D-14).**

After review, start with permissive, ungated bases:

- **Start with:** Qwen3 0.6B, 4B and 8B, Gemma-4-E4B-it, Granite-4.2-8b and gpt-oss-20b (all Apache-2.0), then Phi (MIT), Olmo 3 and Apertus (Apache-2.0).
- **Users compute locally:** Llama, Gemma 3 and any model with unread terms ([U] in section 2).
- **Never upload:** bases for licences whose commercial or derivative conditions conflict with free redistribution, such as LFM3 and MiniMax M3, unless legal review says otherwise.

## 9. Adding a model, and the Day-N process

### 9.1 Checklist for a new model

1. Run `tailtune inspect <repo>` header-only, and save the output in the tracking issue.
2. Record the licence, gate, `auto_map` and quantization, and stop on any of the non-goals.
3. Write or extend the registry entry. Quote every pattern's source tensor from the header, and set `expect` counts.
4. Commit a shrunk config (2 layers, small widths, 4 experts if MoE, the same `layer_types` pattern, tie flag and towers), the full config and the header JSON as fixtures.
5. Add resolver snapshot tests (section 10) and make them pass with zero `unresolved` groups.
6. Train the tiny model for 10 steps under each method. Check the invariants in [03](03-methods.md), plus frozen and tower tensors unchanged.
7. Run an export round trip on the tiny model ([10](10-export-and-deployment.md)).
8. Run a real-weights smoke run, then set the "runs" label for SFT and CPT in the support matrix.
9. Update the families table and the status table.

### 9.2 Day-N for a hyped release

Full day-zero support is out of reach for one maintainer. What is reachable is a post only this project can write, within 24 to 72 hours of release [J]:

1. **Hours 0 to 4.** Run `tailtune inspect` header-only, then publish the quirks, coverage and unresolved groups in an issue. This needs no GPU.
2. **Hours 4 to 24.** Patch the registry if needed and land steps 3 to 5 of the checklist.
3. **Hours 24 to 72.** Run a stock fine-tune: `lora` with defaults on a fixed, licence-clean reference dataset chosen once and reused for every Day-N post. Then run `tailtune eval base/ ft/`.
4. Publish "What a stock fine-tune of ‹model› breaks", with the report, per-item logs, the exact commands and one working notebook.

The post states one seed, the fast-suite resolution and that no safety claim is made. It never says "retention benchmarked". It evaluates models, not their authors.

### 9.3 The Day-N table

The table holds measured times only. Every cell links to its evidence:

- the repository creation time as recorded by the Hub
- the commit time of the `inspect` issue or registry PR
- the post's publication time
- the notebook commit

A cell stays blank until the event happens. Planned or estimated dates are never entered, and nothing is backfilled from memory. The elapsed-hours column is computed from two filled cells.

| Model | Hub repo created (UTC) | `inspect` published | Stock fine-tune report | Notebook | Hours to report | tailtune version | Label |
|---|---|---|---|---|---|---|---|
| | | | | | | | |

## 10. Tests

| Test | What it checks | Runs on |
|---|---|---|
| Tiny random model per architecture | Built from the committed shrunk config on CPU, it resolves with zero `unresolved` groups. Ten training steps keep frozen, tower and `mtp` tensors bitwise unchanged and the protected subspace intact | Every PR |
| Resolver snapshot, tiny model | `targets.json` equals the committed golden file | Every PR |
| Resolver snapshot, real architecture | Built from the committed full config on the meta device with no weights, it gives counts, roles, k, t and coverage equal to the golden file | Every PR |
| Header-only snapshot | The committed header JSON, with no modelling code, gives the same roles for all registry-matched names | Every PR |
| Rule overlap | No parameter in any fixture matches more than one rule | Every PR |
| Arithmetic | `Fraction` flooring (r=0.99 with q=100 gives k=99), t, free values and coverage summing to 1 | Every PR |
| Synthetic quirks | A Muse-style `self_attn.gate_proj` plus `mlp.gate_proj` collision, a `[E,m,n]` parameter, a `[C,1,K]` conv, a uint8 parameter, a GPT-2 `Conv1D` (refused) and a tied pair (counted once) | Every PR |
| Refusals | Exit code 4 and message text for floor, unresolved, zero-protected and quantized cases | Every PR |
| Upstream drift | Snapshot tests against Transformers `main`. A diff opens an issue | Weekly |

## Implementation checklist

**Gate 0, Claim**
- [ ] Share the config reader between `estimate` and `inspect`. Neither downloads weights.

**Gate 1, Verify**
- [ ] Run the embedding-freeze ablation (section 5) and record the default in the README.
- [ ] Confirm `lm_head` behaviour in the public SST code for the `sst_paper` preset.
- [ ] Check whether the pinned Transformers dequantizes MXFP4 for training.
- [ ] Find the accessor for the v5 conversion list, or accept the shape-inference fallback.
- [ ] Confirm whether Gemma 4 experts are raw parameters in memory, and their layout.
- [ ] Open the legal review on bases and deltas.

**Gate 2, v0.1 Door**
- [ ] Build `tailtune.targets`: roles, rules, structural fallback, key spaces, 3D units, free tail and coverage.
- [ ] Add registry entries for `qwen3`, `llama`, `gemma4` (E4B) and Granite 4.2, with fixtures and snapshots.
- [ ] Build `tailtune inspect`, `targets.json`, and the exit code 4 refusals.
- [ ] Add the `sst_paper` preset.
- [ ] Add tiny-model CI for every launch architecture, plus GPT-2 as a `Conv1D` fixture.

**Gate 3, v0.2 Proof**
- [ ] Bring the six public launch models to "runs", and the two benchmark models to "retention benchmarked".
- [ ] Publish the support matrix with both labels.

**Gate 4, v0.3 Reach**
- [ ] Bring Qwen3.8-27B and gpt-oss-20b to "runs (experimental)" on FSDP2.
- [ ] Add router counting for MoE.
- [ ] Publish the first Hub bases after legal review.
- [ ] Publish the Day-N table with measured dates only.

**Gate 5, v1.0**
- [ ] Add the phase 2 models, embedding and reranker support, and the DPO and GRPO labels.

## Acceptance tests

1. On Qwen3-8B's config and header, `inspect` reports 252 constrained units (7 per layer × 36), 39.0M free values at r=0.99 and coverage of 100% under the default policy. It reports 84.8% under `sst_paper` [C].
2. On Qwen3.8-27B's header, `in_proj_qkv` is `attn_fused` in exactly 48 layers, `conv1d` is frozen, `mtp.*` is frozen, `model.visual.*` is `tower`, and nothing is `unresolved`.
3. On gemma-4-26B-A4B's header, the two expert tensors yield 128 units each per layer, and `router.proj` is frozen.
4. On gpt-oss-20b, `sst` either dequantizes and resolves the experts, or refuses with exit code 4 and names the cause. It never trains with the experts silently skipped.
5. A GPT-2 tiny model under `osf` or `sst` exits 4 with a `Conv1D` message.
6. On Qwen3-0.6B, the tied pair appears as one group and is counted once.
7. Training a tiny Gemma 4 model leaves every tower tensor's SHA-256 unchanged, and export writes the base's full key set.

## Open questions

1. Should the public small model be Qwen3-0.6B or SmolLM2-135M? The recommendation is Qwen3-0.6B, pending the owner's decision.
2. Are a 0.90 floor and `min_free_tail = 4` right? Revisit both after Gate 2 logs on the launch models.
3. Whole-matrix or block-split SVD for checkpoint-fused tensors? No study compares them.
4. Is per-expert SVD on 3D experts sensible at all? SST reports only Qwen3, Qwen2.5 and Olmo 3.
5. When does SDFT ship? D-16 orders it before DPO and GRPO, but the roadmap does not schedule it.
6. Are bases and deltas derivative works under each licence? This needs legal review.
