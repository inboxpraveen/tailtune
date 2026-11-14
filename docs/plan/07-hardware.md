# 07 Hardware

**What this document decides.**

- **Tiers.** It sets what each tier in D-17 promises, per method and per platform, and the evidence a platform needs to move up or down.
- **One canonical path for bases.** Bases are computed by SVD in fp32 on the CPU on every platform (D-14). GPU SVD exists on CUDA and ROCm, but tailtune does not use it by default.
- **The memory model.** It gives one formula that `tailtune estimate` implements, with every assumption stated. SST's budget includes `W_ref`, the fp32 copy of the starting weights, because D-13 locks the cumulative projection form.
- **The honest hardware map.** `lora` on a 4-bit base reaches about 8B on a free T4. Paper-exact SST on an 8B model needs a B200 with bases and `W_ref` streamed from host memory. A 96 GB RTX PRO 6000 can also run it, but only with AdamW state held in host memory, or with 8-bit Adam, which is a variant. Nothing on the free tier comes close.
- **Two Gate 0 commands.** It specifies `tailtune doctor` and `tailtune estimate`, and the Gate 1 calibration that turns the estimate from computed to measured.

No GPU training run or GPU memory measurement has been made. Every memory figure here is computed [C] and stays that way until Gate 1. All memory is in GiB (2^30 bytes) unless marked GB. Following the research notes, a card's label such as "24 GB" is read as 24 GiB. On real hardware, `estimate` uses the byte count the device reports instead.

Related documents: precision rationale in [03-methods.md](03-methods.md), sharding in [08-distributed.md](08-distributed.md), CI pipelines in [11-engineering-and-release.md](11-engineering-and-release.md), and gates in [13-roadmap.md](13-roadmap.md).

---

## 1. Tiers

### 1.1 What each tier promises

| Tier | Promise | CI | Release impact |
|---|---|---|---|
| **Tier 1** | Every v0.1 method (`lora`, `lora` on a 4-bit base, `osf`, `sst`, `full`, `freeze`) runs within the limits `estimate` prints. Calibrated memory figures are published for the platform | GPU jobs nightly or per release, as listed in §2 | A failure blocks the release |
| **Tier 1 at release gate** | As Tier 1, but tested once per release rather than nightly | Per release | A failure blocks the release |
| **Tier 1 reduced** | As Tier 1, but only for the methods, model sizes and precision mode named in the row | Weekly smoke test | A failure blocks the release |
| **Tier 1 numerics** | Correctness of SVD, projection, hooks and 2-rank gloo FSDP2 on tiny models. Not a training target | Every PR | A failure blocks the merge |
| **Tier 2** | Installs, and is expected to train. Known limits are documented. GPU behaviour is checked by hand or by a smoke test | CPU jobs, smoke tests or donated credits | Install failures block the release. Training bugs are fixed best effort |
| **Tier 3** | Community. `doctor` recognises the platform and prints its known limits. PRs are welcome if they add no burden to Tier 1 code paths | None | None |
| **Unsupported** | `doctor` reports FAIL with the reason. No fixes | None | None |

The tier assignments are a judgement [J], built on the upstream facts below.

### 1.2 Moving between tiers

A platform is **promoted** when all of the following hold [J]:

1. A CI route exists that runs on every release (Tier 1) or at least monthly (Tier 2), and is funded for at least two releases.
2. Gate 1-style memory calibration has been run on the platform, and `estimate` lies within the tolerance in §15.
3. Each promised method has completed one end-to-end run there, including the projection invariant tests: frozen-subspace content unchanged after N steps, and projection run after the real optimizer step.
4. Someone is named to answer issues for that platform: the maintainer, a second maintainer, or a sponsor.

A platform is **demoted** when upstream drops it (a torch wheel, CUDA or ROCm release), when its CI route is lost for two consecutive releases, or when a silently wrong result there stays unfixed for a release.

Every move is recorded in D-17 in [README.md](README.md) first, as the README requires for any change to a locked decision.

---

## 2. Support matrix (D-17)

Promise codes: **P** promised and tested; **R** promised, tested at release gate; **L** promised up to the stated size limit; **B** best effort; **C** community; **N** not offered. "SVD" is where tailtune computes bases. It is the CPU everywhere, per D-14.

| Platform | OS | Tier | `lora` on 4-bit | `lora` | `osf` | `sst` | SVD | GPUs | CI route |
|---|---|---|---|---|---|---|---|---|---|
| NVIDIA Ampere (A100, RTX 30, A10) | Linux x86_64 | 1 | P | P | P | P | CPU | Single, DDP at Gate 2. FSDP2 at Gate 3. ZeRO-3 last (D-19) | A100 in the 2-GPU release gate |
| NVIDIA Ada (RTX 40, L4, L40S) | Linux x86_64 | 1 | P | P | P | P | CPU | As above | Modal L4 nightly |
| NVIDIA Hopper (H100, H200) | Linux x86_64 | 1 | P | P | P | P | CPU | As above | One H100 run per minor release [J] |
| NVIDIA Blackwell (B200, RTX 50, RTX PRO 6000) | Linux x86_64 | 1 at release gate | R | R | R | R | CPU | As above | Modal RTX PRO 6000 or B200 per release |
| NVIDIA T4 (Colab, Kaggle) | Linux x86_64 | 1 reduced | L (about 8B) | L (about 4B) | L (about 1B) | L (about 0.7B with AdamW; 1.1B with 8-bit Adam, a variant) | CPU | Single only. Kaggle T4 x2 not promised | Kaggle T4 kernel weekly |
| CPU only | Linux, macOS, Windows | 1 numerics | N | N (tiny tests only) | N | N (tiny tests only) | CPU | 2-rank gloo for tests | Free GitHub runners, every PR |
| NVIDIA GeForce, RTX, workstation | Windows native | 2 | B | B | B | B | CPU | Single only (no NCCL) | Windows CPU job in PRs; GPU by hand |
| NVIDIA GeForce, RTX | WSL2 | 2 | B | B | B | B (host RAM limit) | CPU | Single only | By hand |
| AMD Instinct (MI300X, MI325X, MI350 class) | Linux x86_64 | 2 | B | B | B | B | CPU | Single node | AMD or TensorWave credits, when granted |
| Apple MPS, macOS 15+, torch 2.15+ | macOS arm64 | 2 | B | B | B (small) | B (small; no 8-bit Adam) | CPU | Single device | macOS runner smoke test |
| NVIDIA GB200, DGX Spark | Linux aarch64 | 2 | B | B | B | B | CPU | Single node | None until sponsored |
| AMD Radeon RDNA 3/4 | Linux, Windows | 3 | C | C | C | C | CPU | Single | None |
| Intel Arc, Data Center GPU Max (XPU) | Linux, Windows | 3 | C | C | C | C | CPU (forced) | Single | None |
| Intel Gaudi 2/3 | Linux | 3 | C | C | C | C (no bitsandbytes optimizers) | CPU | Single | None |
| NVIDIA Rubin | Linux | 3 (watch) | C | C | C | C | CPU | n/a | None |
| MLX | macOS | 3 | N | N | N | N | n/a | n/a | None |
| TorchTPU | Linux | 3 (watch) | C | C | C | C | CPU | n/a | None |
| Huawei Ascend | Linux | 3 | C | C | C | C | CPU | n/a | None |
| Pascal, Volta, Maxwell (incl. Kaggle P100, V100) | any | Unsupported | | | | | | | |
| `torch_xla` | any | Unsupported | | | | | | | |
| FSDP1 | any | Unsupported | | | | | | | |
| Windows multi-GPU | Windows | Unsupported | | | | | | | |
| macOS 14 | macOS | Unsupported | | | | | | | |
| Python 3.10 | any | Unsupported | | | | | | | |

Three platforms are not named in D-17. The research put Windows on Arm (RTX Spark) at Tier 3 and Intel Macs at unsupported, and PyTorch 2.15 publishes macOS wheels only as `macosx_14_0_arm64` [S](https://pypi.org/pypi/torch/2.15.0/json). This document treats them that way until the owner records a decision. Intel Data Center GPU Max sits with Arc because both use the same XPU backend.

---

## 3. NVIDIA by generation

| Generation | Examples | Compute capability | VRAM | Native bf16 | FlashAttention package | Tier |
|---|---|---|---|---|---|---|
| Turing | T4 | 7.5 | 16 GB nominal, about 15 usable | No | Not supported upstream; community fork only | 1 reduced |
| Ampere | A100, RTX 3090, A10 | 8.0, 8.6 | A100 40/80; 3090 24 | Yes | FA2 | 1 |
| Ada | RTX 4090, L4, L40S | 8.9 | 4090 and L4 24; L40S 48 | Yes | FA2 | 1 |
| Hopper | H100, H200 | 9.0 | H100 80 (NVL 94); H200 141 or 143 | Yes | FA2, FA3, FA4 | 1 |
| Blackwell data centre | B200, GB200 | 10.0 | 180 (B200); about 186 (GB200) | Yes | FA4 | 1 at release gate (GB200 is aarch64, so Tier 2) |
| Blackwell workstation and consumer | RTX 5090, RTX PRO 6000 | 12.0 | 32; 96 | Yes | Not stated in the FA README | 1 at release gate |
| DGX Spark (GB10) | DGX Spark | 12.1 | 128 unified, 273 GB/s | Yes | Not stated | 2 (aarch64) |
| Rubin | Vera Rubin | 10.7 | 288 HBM4, reported | Not confirmed | Not stated | 3 (watch) |

Sources: [NVIDIA CUDA GPUs](https://developer.nvidia.com/cuda-gpus), [flash-attention](https://github.com/Dao-AILab/flash-attention), [HF Spaces GPUs](https://huggingface.co/docs/hub/spaces-gpus), [DGX B200](https://www.nvidia.com/en-us/data-center/dgx-b200/), [RTX PRO 6000](https://www.nvidia.com/en-us/products/workstations/professional-desktop-gpus/rtx-pro-6000/), [RTX 5090](https://www.nvidia.com/en-us/geforce/graphics-cards/50-series/rtx-5090/), [DGX Spark](https://www.nvidia.com/en-us/products/workstations/dgx-spark/) [S]. Rubin memory comes from press coverage of NVIDIA specs ([Tom's Hardware](https://www.tomshardware.com/tech-industry/artificial-intelligence/nvidia-delivers-first-vera-rubin-ai-gpu-samples-to-customers-88-core-vera-cpu-paired-with-rubin-gpus-with-288-gb-of-hbm4-memory-apiece)) [U].

**CUDA and driver minimums.**

- PyTorch 2.15 ships CUDA 13.2 (stable) and 13.4 (experimental) wheels and drops cu126 and cu130. CUDA 13 wheels need driver 580 or newer and a Turing or newer GPU [S](https://github.com/pytorch/pytorch/releases/tag/v2.15.0).
- PyTorch 2.14 shipped CUDA 12.6, 13.0 and 13.2 [S](https://github.com/pytorch/pytorch/blob/main/RELEASE.md).
- The 2.15 blog calls 13.4 "GA wheels", while RELEASE.md calls it experimental [S](https://pytorch.org/blog/pytorch-2-15-release-blog/). Treat 13.2 as the tested build.
- CUDA 13.5 features need driver R620+, but existing CUDA 13.x applications run on 580+ [S](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html).
- The practical driver floor is therefore **580** on every supported generation [C]. The minimum driver for torch 2.14's cu126 wheels was not collected [U].

**bf16 and emulation.** `torch.cuda.is_bf16_supported()` returns True on a T4 because it falls back to testing whether a bf16 tensor can be created. Only `including_emulation=False` gives the honest answer [S](https://github.com/pytorch/pytorch/blob/v2.15.0/torch/cuda/__init__.py#L244-L263). The slowdown from emulated bf16 on a T4 is unmeasured [U]. tailtune never trains in emulated bf16.

**FlashAttention and SDPA.** FA2 covers Ampere, Ada and Hopper, FA3 is Hopper only, FA4 covers Hopper and Blackwell, and Turing is not supported [S](https://github.com/Dao-AILab/flash-attention). PyTorch's SDPA works on every backend and is tailtune's default. The `flash-attn` package is an optional `[flash]` extra (D-07). PyTorch 2.15 enables the FlashAttention SDPA backend in Windows CUDA builds [S](https://github.com/pytorch/pytorch/releases/tag/v2.15.0).

**What people use them for.** No survey of fine-tuning hardware was found [U]. The proxies are:

- The T4 dominates free notebooks.
- Vendor buying guides call the 4090 the default consumer card and a used 3090 the budget pick. These vendors sell or rent GPUs.
- Projects test on T4, A10 and L40S (bitsandbytes), L4 (Accelerate, PEFT), L40S (TRL, DeepSpeed), and H100 and B200 (Axolotl) ([bnb](https://github.com/bitsandbytes-foundation/bitsandbytes/blob/main/.github/workflows/test-runner.yml), [PEFT](https://github.com/huggingface/peft/blob/main/.github/workflows/nightly.yml), [Axolotl](https://github.com/axolotl-ai-cloud/axolotl/blob/main/.github/workflows/gpu-slow-jobs.yml)).
- DGX Spark's page says it can "Fine-tune AI models up to 70 billion parameters" [S](https://www.nvidia.com/en-us/products/workstations/dgx-spark/). By §9, that holds for `lora` on a 4-bit base, not for full-parameter SST.

**Rubin.** CUDA 13.4 wheels add "experimental support for NVIDIA Rubin GPUs" [S](https://pytorch.org/blog/pytorch-2-15-release-blog/). Rack samples have shipped to customers, with volume production reported for H2 2026 ([Windows Report](https://windowsreport.com/nvidia-ships-first-vera-rubin-vr200-ai-rack-samples-to-customers/), secondary) [U]. None of the rental pricing pages read lists Rubin. It stays at Tier 3 until it can be rented and the cu134 build is stable.

**Why Pascal, Volta and Maxwell are unsupported.** CUDA 13 removed them [S](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html), and PyTorch 2.15 tells their users to stay on 2.14 [S](https://github.com/pytorch/pytorch/releases/tag/v2.15.0). torch 2.14 with cu126 can still target them. However, tailtune's 2.14 floor exists only because Colab still ships it, and it will rise. These GPUs also lack native bf16, and a platform that works only on the floor version would block every future floor raise [J]. This rules out Kaggle's P100 too [C].

**Turing is next.** No source says NVIDIA or PyTorch will drop Turing, but it is the oldest remaining architecture. The free-tier plan must not assume the T4 is permanent [J].

---

## 4. AMD

**Versions.** The current ROCm is 10.1.0 (2026-10-05) [S](https://rocm.docs.amd.com/en/latest/release/versions.html). PyTorch 2.15's ROCm wheels are built on AMD's TheRock packages and install ROCm as a pip dependency. They drop ROCm 7.2 and the gfx900/gfx906 kernels [S](https://pytorch.org/blog/pytorch-2-15-release-blog/). RELEASE.md lists ROCm 7.14 as the stable build for 2.15, while the blog announces ROCm 10.0 wheels. These two primary pages disagree, so pin to what the installed torch reports rather than to AMD's latest release [J].

**Hardware.** ROCm 7.14 lists MI355X, MI350X, MI325X, MI300X, MI300A, MI250X, MI250, MI210 and MI100, plus Radeon PRO and consumer RX 9000 and 7000 parts. Radeon is limited to specific Ubuntu and RHEL releases [S](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html). Reported memory is 192 GB for the MI300X, 256 GB for the MI325X and 288 GB for the MI350X/MI355X, from press sources ([The Register](https://www.theregister.com/2025/06/12/amd_mi355x/)) [U].

**What is claimed, and what has been shown independently:**

| Item | Claimed | Shown independently |
|---|---|---|
| bitsandbytes | ROCm "out of preview; it is now considered stable" in 0.50.0; Windows ROCm wheels; ROCm 10.0 build in 0.50.2 [S](https://github.com/bitsandbytes-foundation/bitsandbytes/releases) | No |
| Accelerate | v1.14.0 headlined "AMD ROCm support" [S](https://github.com/huggingface/accelerate/releases) | No |
| FlashAttention | Composable Kernel backend for MI200 to MI355 and RDNA 3/4; Triton backend [S](https://github.com/Dao-AILab/flash-attention) | No |
| Unsloth | RDNA 3/3.5/4 on Windows and Linux, and MI300X; uses SDPA, not FA2, on AMD [S](https://unsloth.ai/docs/get-started/install/amd) | No |
| DeepSpeed | Runs an MI200 workflow; the most-tested AMD GPUs are MI100 and MI200 [S](https://github.com/deepspeedai/DeepSpeed/blob/master/README.md) | No |
| LLM fine-tuning performance on Instinct | None found | None found [U] |
| Radeon on Windows | "PyTorch on Windows updated with ROCm 7.2.1" for Radeon 9000 and some 7000 parts. Training support is not stated [S](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/index.html) | One user report: a CIFAR-10 ResNet18 run took 748 s on an RX 9070 XT, then 200 to 240 s on nightlies, against 74 s on an RTX 4090. The reporter later called it "still well behind Linux" [S](https://github.com/ROCm/TheRock/issues/2510). It is a CNN, not an LLM |

On ROCm, "stable" means the APIs pass tests. It is not a claim of performance parity [J].

**The `gesvdj` limitation.** On ROCm, `torch.linalg.svd` runs natively but hard-codes `gesvdj` and ignores the `driver=` argument [S](https://github.com/pytorch/pytorch/blob/v2.15.0/aten/src/ATen/native/cuda/linalg/BatchLinearAlgebraLib.cpp#L657-L662). The source warns that `gesvdj` "may be numerically unstable for large sized matrix". Because D-14 puts SVD on the CPU everywhere, this does not affect tailtune. It is one more reason not to offer GPU SVD as an option.

**Fit.** By §9, an MI300X at 192 GB would hold `sst/host-bases` for an 8B model (137.7 GiB) but not `sst/gpu` (188.2 GiB) [C]. An MI325X or MI350 would hold either.

**How AMD Instinct earns Tier 1** [J]:

1. Apply for AMD developer cloud access [S](https://www.amd.com/en/developer/resources/cloud-access.html) and the TensorWave open-source program [S](https://go.tensorwave.com/open-source-developer-program).
2. Run a per-release MI300X job:
   - unit tests on ROCm;
   - `lora` on a 4-bit base with bitsandbytes on ROCm;
   - `sst` and `osf` on a ~1.7B model with the projection invariants;
   - Gate 1 memory calibration;
   - one 2-GPU FSDP2 run after Gate 3.
3. Keep that job green for two consecutive releases, then record the promotion in D-17.

Radeon stays at Tier 3 until a contributor with the hardware takes on the same checklist.

---

## 5. Apple, Intel, TPU, Ascend, CPU and aarch64

### Apple MPS (Tier 2)

**Works:**

- MPS on macOS 15 or later. PyTorch 2.15 dropped macOS 14 for MPS [S](https://pytorch.org/blog/pytorch-2-15-release-blog/).
- bitsandbytes 4-bit and int8 on MPS with torch 2.9 or later [S](https://github.com/bitsandbytes-foundation/bitsandbytes/releases).
- bf16 on MPS through Accelerate needs torch 2.6, and fp16 needs torch 2.8 [S](https://github.com/huggingface/accelerate/releases).

**Does not work, or must be forced:**

- **Require torch 2.15 on MPS.** Its notes fix "crashes and wrong results in MPS `linalg.svd`", SVD "returning non-orthonormal `U`/`Vh`", and "silent wrong results for large MPS tensors" in common ops [S](https://github.com/pytorch/pytorch/releases/tag/v2.15.0). This raises the D-05 floor of 2.14 on this one platform. Treat 2.14 MPS results as untrusted.
- **SVD goes to the CPU.** MPS SVD runs natively only if `max(m,n)*min(m,n)*4` bytes fits in threadgroup memory. Otherwise it falls back to the CPU with a warning, and it accepts float32 only [S](https://github.com/pytorch/pytorch/blob/v2.15.0/aten/src/ATen/native/mps/operations/LinearAlgebra.mm#L2256-L2296). A 1024×4096 matrix needs about 16.8 MB, so every LLM weight falls back [C]. Apple's exact limit was not confirmed [U].
- **No float64** ("Metal Shading Language has no double type") [S](https://docs.pytorch.org/docs/2.15/notes/mps.html). Reference checks in float64 run on the CPU.
- **No 8-bit optimizers** in bitsandbytes on MPS [S](https://github.com/bitsandbytes-foundation/bitsandbytes/releases). The `sst/8bit-adam` route does not exist here.
- **Fused AdamW on MPS is a prototype** that "supports torch.float32 and torch.float16" [S](https://docs.pytorch.org/docs/2.15/generated/torch.optim.AdamW.html).
- **No distributed promise.**
- **Evaluation falls back to Hugging Face generation**, because vLLM does not run there. See [09-evaluation.md](09-evaluation.md).

**Memory.** Unified memory means host offload saves nothing: GPU and host bytes come from the same pool. §9.5 therefore uses total bytes on Macs. PyTorch's default MPS high-watermark ratio is 1.7, where 1.0 is `recommendedMaxWorkingSetSize` [S](https://docs.pytorch.org/docs/stable/mps_environment_variables.html). The research assumed 75% of unified memory is usable by the GPU. That figure has no source [U]. Throughput is unmeasured, so a memory fit on a Mac does not mean the run is practical.

### MLX (Tier 3)

MLX is a separate framework, not a PyTorch backend. mlx-lm supports `lora`, `dora` and `full` fine-tuning [S](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/LORA.md). Its `linalg::svd` runs on the CPU stream only [S](https://github.com/ml-explore/mlx/blob/main/mlx/linalg.cpp#L269-L271). It is listed as a possible export or companion path later, not a training target.

### Intel XPU: Arc, Max (Tier 3) and Gaudi (Tier 3)

**XPU.** PyTorch XPU supports Data Center GPU Max, Arc A and B series and Core Ultra graphics, on Windows 11, Ubuntu and WSL2. `GradScaler` needs FP64, which Arc A-series lacks [S](https://docs.pytorch.org/docs/2.15/notes/get_start_xpu.html). SVD falls back to the CPU (`_linalg_svd.U` is on the fallback list) [S](https://github.com/intel/torch-xpu-ops/blob/main/src/ATen/native/xpu/XPUFallback.cpp#L362). bitsandbytes 0.48.0 added Arc and Max support for NF4 and int8 [S](https://github.com/bitsandbytes-foundation/bitsandbytes/releases). Only inference benchmarks and vendor training claims were found. XPU stays at Tier 3 because SVD offers no gain and no independent training evidence exists [J].

**Gaudi.** bitsandbytes has no optimizers on Gaudi [S](https://github.com/bitsandbytes-foundation/bitsandbytes/releases). Intel's roadmap is weak: Falcon Shores was cancelled for commercial release, and Crescent Island is inference-focused ([Tom's Hardware](https://tomshardware.com/tech-industry/semiconductors/intel-chip-roadmap-2026-2028)). Phoronix reports neglected Gaudi 3 software ([Phoronix](https://www.phoronix.com/review/intel-crescent-island)) [U].

### TPU (TorchTPU Tier 3; `torch_xla` unsupported)

`pytorch/xla` "is deprecated and replaced by TorchTPU". It will be archived on 2026-11-15, and its last stable release is 2.8.0 [S](https://github.com/pytorch/xla). TorchTPU was introduced in April 2026 and open-sourced around October 2026 ([Google Developers Blog](https://developers.googleblog.com/en/torchtpu-running-pytorch-natively-on-tpus-at-google-scale/), [InfoQ](https://infoq.com/news/2026/10/google-meta-torchtpu)). Its GA status, its SVD support and any Hugging Face integration are unverified [U].

### Huawei Ascend (Tier 3)

Liger Kernel lists Ascend NPU CI, and DeepSpeed lists Ascend as contributed hardware with upstream validation marked "No" [S](https://github.com/linkedin/Liger-Kernel). The Ascend PyTorch repository was not read. The tier entry is a note only.

### CPU only (Tier 1 numerics)

The CPU is where every correctness test runs: SVD, the projector, hook order, and 2-rank gloo FSDP2. The Transformers FSDP2 checkpointing bug was reproduced this way [S](https://github.com/huggingface/transformers/issues/50424). The CPU is also the canonical SVD path (§8). It is not a training target, and no CPU training numbers were gathered.

### Linux aarch64 (Tier 2)

torch publishes `manylinux_2_28_aarch64` wheels [S](https://pypi.org/pypi/torch/2.15.0/json), and bitsandbytes publishes `manylinux_2_24_aarch64` [S](https://pypi.org/pypi/bitsandbytes/0.50.2/json). GB200 and DGX Spark live here. There is no CI until someone sponsors it.

---

## 6. Version floors

These move weekly. Transformers shipped 48 releases and TRL 32 in 2026 so far [S](https://pypi.org/pypi/transformers/json), so the weekly compatibility job in D-07 is mandatory. Only the Python, torch and Transformers floors are locked (D-04, D-05). The rest are proposals, which [11-engineering-and-release.md](11-engineering-and-release.md) replaces with tested ranges and a committed `uv.lock`.

| Component | Floor | Latest seen (release date) | Why this floor | Source, retrieved 2026-11-07/08 |
|---|---|---|---|---|
| Python | 3.11 (locked) | torch 2.15 supports 3.11 to 3.15 | torch 2.15, Transformers 5.22 and TRL 1.17 require 3.11 | [torch](https://pypi.org/project/torch/2.15.0/), [Transformers](https://pypi.org/project/transformers/5.22.2/), [TRL](https://pypi.org/project/trl/1.17.1/) |
| torch | 2.14 (locked); tested on 2.14 and 2.15 | 2.15.0 (2026-10-28) | Colab's 2026-11-04 snapshot lists `2.14.1+cu130` | [Colab freeze](https://github.com/googlecolab/backend-info/blob/main/pip-freeze.gpu.txt), [RELEASE.md](https://github.com/pytorch/pytorch/blob/main/RELEASE.md) |
| torch on MPS | 2.15 | 2.15.0 | Fixes silently wrong MPS results | [release notes](https://github.com/pytorch/pytorch/releases/tag/v2.15.0) |
| CUDA (from the torch wheel) | 13.2 with torch 2.15; 12.6 to 13.2 with 2.14 | Toolkit 13.5 | What PyTorch builds | [release notes](https://github.com/pytorch/pytorch/releases/tag/v2.15.0), [CUDA notes](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html) |
| NVIDIA driver | 580 | n/a | Required by CUDA 13 wheels | [release notes](https://github.com/pytorch/pytorch/releases/tag/v2.15.0) |
| ROCm (from the torch wheel) | 7.14 with torch 2.15 | 10.1.0 (2026-10-05) | What PyTorch ships, not AMD's latest | [ROCm versions](https://rocm.docs.amd.com/en/latest/release/versions.html), [RELEASE.md](https://github.com/pytorch/pytorch/blob/main/RELEASE.md) |
| Transformers | 5.x only (locked) | 5.22.2 (2026-11-06) | v5 defaults, no v4 layer (D-05) | [PyPI](https://pypi.org/project/transformers/5.22.2/), [v5.0.0](https://github.com/huggingface/transformers/releases/tag/v5.0.0) |
| Accelerate | 1.14 | 1.15.0 (2026-09-09) | 1.14 added ROCm support and excluded `Params4bit` from `fully_shard`; 1.15 fixed tied embeddings on torch 2.13+ | [releases](https://github.com/huggingface/accelerate/releases) |
| bitsandbytes (`[bnb]`) | 0.50 | 0.50.2 (2026-08-27) | ROCm stable, CUDA 13.2 wheels, MPS 4-bit | [releases](https://github.com/bitsandbytes-foundation/bitsandbytes/releases) |
| TRL | Tested against latest [J] | 1.17.1 (2026-10-30) | No hardware-driven floor found | [PyPI](https://pypi.org/project/trl/1.17.1/) |
| PEFT | Tested against latest [J] | 0.21.2 (2026-10-01) | OSF arrived in 0.18.0. 0.21.2 skips Conv1D targets silently, so tailtune rejects them itself (D-15) | [PyPI](https://pypi.org/project/peft/0.21.2/) |
| DeepSpeed | Not a core dependency [J] | 0.19.9 (2026-10-29) | The ZeRO-2 reference runs the public SST repo in its own environment (D-19) | [PyPI](https://pypi.org/project/deepspeed/0.19.9/) |

---

## 7. Precision rules per platform

The reasons are in [03-methods.md](03-methods.md). In short:

- bf16 rounds away updates smaller than about `2^-9·|w|`, and projected steps are tiny.
- Transformers v5 loads with `dtype="auto"` by default [S](https://github.com/huggingface/transformers/releases/tag/v5.0.0).
- So D-12 requires fp32 master weights for constrained methods, an explicit `dtype=torch.float32`, and refusal of low-precision trainable targets.

| Platform | Master weights (`sst`, `osf`, `full`) | Autocast | Loss scaler | bf16 check | Families forced to float32 |
|---|---|---|---|---|---|
| NVIDIA Ampere, Ada, Hopper, Blackwell | fp32 | bf16 | None | `is_bf16_supported(including_emulation=False)` returns True | None needed |
| NVIDIA T4 | fp32 | fp16 | `GradScaler` on the fp32 masters. Never fp16 weights: the scaler raises "Attempting to unscale FP16 gradients." [S](https://github.com/pytorch/pytorch/blob/v2.15.0/torch/amp/grad_scaler.py#L275) | Returns False with the flag. The default call says True through emulation | Gemma 3 and 3n, Gemma 4, GLM-4 MoE, gpt-oss, Qwen3.5, Qwen3-MoE: compute in fp32 and double the activation term [S](https://github.com/unslothai/unsloth/blob/main/unsloth/models/loader.py#L213-L223) |
| AMD Instinct, Radeon | fp32 | bf16 | None | The function's behaviour on ROCm was not checked [U]. `doctor` runs a bf16 matmul probe against fp32 | None known |
| Apple MPS | fp32 | fp32 by default for constrained methods; bf16 opt-in [J] | None | MPS probe | None known. No float64 |
| Intel XPU | fp32 | bf16 [U] | Not on Arc A-series (no FP64) | Probe | None known |
| CPU | fp32; float64 for reference tests | None | None | n/a | n/a |

**`lora` on every platform.** The frozen base is 4-bit or 16-bit. PEFT casts adapter weights to float32 by default [S](https://huggingface.co/docs/peft/main/en/package_reference/peft_model). On a T4, the 4-bit compute dtype is fp16.

**Two T4 traps.** Under fp16 the scaler skips the first steps, so a run shorter than about 5 steps may apply no update [S](https://github.com/unslothai/unsloth/blob/main/.github/workflows/kaggle-t4-notebook-ci.yml). Every T4 test must run more than 5 optimizer steps. Gemma-4-E4B-it is a launch model (D-20) on the forced-fp32 list, so on a T4 it runs `lora` with fp32 compute. `estimate` must apply that cost.

---

## 8. SVD support per backend

| Backend | Native SVD for LLM-size matrices | tailtune route | Notes |
|---|---|---|---|
| CUDA | Yes: `gesvdj` by default, falling back to `gesvd`; also `gesvda` [S](https://docs.pytorch.org/docs/2.15/generated/torch.linalg.svd.html) | CPU | GPU SVD is available but not used by default |
| ROCm | Yes, `gesvdj` only [S](https://github.com/pytorch/pytorch/blob/v2.15.0/aten/src/ATen/native/cuda/linalg/BatchLinearAlgebraLib.cpp#L657-L662) | CPU | Never offer GPU SVD here |
| MPS | Only below the threadgroup-memory gate | CPU | float32 only |
| XPU | No; on the CPU fallback list | CPU | |
| MLX | No; CPU stream | n/a | |
| CPU | Yes (LAPACK or MKL) | **Canonical path** | float32 and float64 |
| TorchTPU, Ascend | Unverified [U] | CPU | |

**Why one CPU path (D-14).** Singular vectors "are not unique", and "different hardware and software may compute different singular vectors" [S](https://docs.pytorch.org/docs/2.15/generated/torch.linalg.svd.html). One path makes the bases hash meaningful across platforms. GPU SVD could become an opt-in accelerator on CUDA later, but only with `driver="gesvd"` and only after it passes the subspace-distance test against the CPU path [J].

**Required upcast.** `torch.linalg.svd` accepts float32 and float64 only, and the CUDA kernel's dispatch excludes fp16 and bf16 [S](https://github.com/pytorch/pytorch/blob/v2.15.0/aten/src/ATen/native/cuda/linalg/BatchLinearAlgebraLib.cpp#L336). PEFT's OSF upcasts the same way [S](https://github.com/huggingface/peft/blob/main/src/peft/tuners/osf/utils.py). `tailtune.bases` always does `W.to("cpu", torch.float32)` and calls `torch.linalg.svd(W, full_matrices=False)`. The reduced form is enough: the short side gets a full basis, so its tail can be stored, and the long side keeps its head `k` vectors. Use full SVD at high r, never a randomized routine, because randomized routines target leading components (see [03-methods.md](03-methods.md)).

**Checks on every basis** (tolerances are [J], to be confirmed at Gate 1):

1. Orthonormality: `max|QᵀQ − I| ≤ 1e-4` for every stored factor, accumulated in float64.
2. Reconstruction: `‖W − U S Vᵀ‖_F / ‖W‖_F ≤ 1e-5`.
3. Cross-platform: subspace distance `‖sin Θ‖₂ ≤ 1e-3` between the cached basis and a fresh CPU run on CI. Skip the test where `σ_k` and `σ_{k+1}` are within `1e-6` relative, and report that gap, because at the cut the subspace itself is ill-defined.

**Cost.** Timings were measured on one laptop as a scoping test, not a benchmark: NumPy 2.3.5 with MKL, 24 threads, float32, random matrices, reduced SVD [M]. A 4096×4096 matrix took 26 to 32 s, and a 14336×4096 matrix took 48 s. Scaled across whole models [C]:

| Model | ~2B | ~4B | ~8B | ~14B | ~27B | ~70B | ~120B MoE |
|---|---|---|---|---|---|---|---|
| CPU SVD pass, hours | 0.2 to 0.4 | 0.5 to 1.2 | 1.5 to 3.5 | 3.7 to 8.4 | 7.2 to 16.6 | 30.6 to 70.4 | 18.5 to 42.5 |

The peak transient for the largest 8B linear layer is 0.28 GiB [C]. Embeddings are not targeted, which avoids the largest single allocation. On a rented B200, a CPU SVD pass leaves the GPU idle. 1.5 to 3.5 hours at $5.98 to $7.99 per hour wastes $9 to $28 [C], so compute the bases before renting.

**Caching.**

- **Key:** model repo, revision commit, checkpoint tensor name and in-memory name, shape, source dtype, `r`, and storage form (head or tail per side).
- **Value:** fp32 tensors in safetensors, under the user cache directory. Bases are never stored in bf16, because that makes the projector inexact.
- **Recorded data:** torch version, LAPACK backend and check results go in a sidecar. The content hash goes into `lineage.json`.
- **Disk:** at r=0.99, Qwen3-8B's bases need 39.8 GB in head form and 22.9 GB with tails stored on the square sides [C].
- **Hub sharing:** waits on licence review (D-14).

---

## 9. The memory model

### 9.1 Formula

```
V_gpu  = R + f · (W + G + O + B_gpu + W_ref,gpu + L + A)        GiB
V_host = O_host + W_host + G_host + B_host + W_ref,host         GiB
Fit:     V_gpu ≤ u · VRAM   and   V_host ≤ available host RAM
```

| Term | Definition | Default (to be calibrated) |
|---|---|---|
| `P` | All parameters, tied tensors counted once | From the safetensors header (§14) |
| `P_t` | Parameters in target matrices (D-15 roles); `[E,m,n]` counted as E matrices | From the resolver |
| `R` | Runtime reserve: CUDA context and allocator baseline | 1 GiB |
| `f` | Allocator fragmentation | 1.10 |
| `u` | Usable fraction of VRAM | 0.92 |
| `W` | Weights. `full`, `sst` and `osf` dense: 4·P. `lora` 16-bit: 2·P. `lora` 4-bit: 0.5·P_linear + 2·P_embed+head | |
| `G` | Gradients, fp32: 4·P on trainable parameters | |
| `O` | AdamW m and v, fp32: 8·P. 8-bit Adam: 2.03·P (two uint8 values plus two float32 `absmax` per 256-element block [S](https://github.com/bitsandbytes-foundation/bitsandbytes/blob/main/bitsandbytes/optim/optimizer.py#L511-L529)). 0 if on host | |
| `B` | Bases, fp32. Head form: Σ (m+n)·k·4. Tail form: Σ [m·min(k, m−k) + n·min(k, n−k)]·4, with `k = ⌊r·min(m,n)⌋`. The research notes' ρ=0.5 rows use (m+n+1)·k·4 | Tail form for `sst` |
| `W_ref` | fp32 copy of the starting target weights for the cumulative form (D-13): 4·P_t | Required by `sst` |
| `L` | LoRA rank 16 on all seven linears: Σ(m+n)·16, at 16 bytes each (fp32 weight, gradient, Adam) | |
| `A` | Activations, checkpointing on: T·H·2·(N+12) + 2·T·V·2 bytes. Off: T·H·2·12·N + 2·T·V·2 | T = 2×512 = 1,024 tokens |

**Which parts sit on GPU or host, per route:**

| Route ID | Method | On GPU | On host |
|---|---|---|---|
| `lora-4bit` | `lora` | 4-bit base, fp32 adapters with their gradients and Adam | Nothing |
| `lora-16bit` | `lora` | 16-bit base, fp32 adapters with their gradients and Adam | Nothing |
| `full` | `full` (baseline) | fp32 W, G, AdamW | Nothing |
| `osf` | `osf` (PEFT as coded, ρ=0.5, all-linear) | Base W, factors, initial copies, factor gradients and Adam, high bases on non-square matrices, all in bf16 [C]. Factor space, not dense projection | Nothing |
| `sst/gpu` | `sst` | fp32 W, G, AdamW, bases, `W_ref` | Nothing |
| `sst/host-bases` | `sst` | fp32 W, G, AdamW; one block's bases and `W_ref` at a time | Bases and `W_ref` |
| `sst/8bit-adam` | `sst` **variant** | fp32 W, G, 8-bit Adam; one block's bases and `W_ref` | Bases and `W_ref` |
| `sst/offload-state` | `sst` | fp32 W and G; one block's bases and `W_ref` | AdamW state (8·P), bases, `W_ref` |
| `sst/zero-offload` | `sst`, untested | 16-bit W and G; one block's bases and `W_ref` | fp32 master and AdamW (12·P), bases, `W_ref`, plus a host gradient buffer (2 to 4 B/param, not included) |

**Excluded from every route:** load-time peaks, full-state-dict gathers at save time, the SVD transient (§8), the data pipeline, pinned staging buffers, and communication buffers.

`sst/8bit-adam` is a variant. SST's results were obtained with AdamW and Muon, not with 8-bit Adam, so its results do not transfer (D-11). The route must be labelled as a variant wherever it is printed, and `method="auto"` never picks it.

**Host streaming has a cost.** `sst/host-bases` moves bases and `W_ref` over the host link once per optimizer step: about 47 GiB for Qwen3-8B [C]. At an assumed 20 GB/s effective link, which comes from the research notes and is not measured, that is about 2.5 s per optimizer step [C]. Gradient accumulation spreads the cost, because projection runs once per optimizer step (D-13). One option reduces it further [J]: with `K=None` and a bf16 checkpoint, `W_ref` equals the bf16 weights exactly, so it can be stored in bf16 without loss. That halves `W_ref` to about 13 GiB. Refresh (`K` set) breaks this, because `W_ref` then becomes an fp32 value.

**Tokens per batch.** At 1,024 tokens with checkpointing, activations are small (0.83 GiB at 8B). For Qwen3-8B, each additional 1,024 tokens per micro-batch adds about 1.05 GiB after fragmentation [C]. Logits account for 0.58 GiB of that, because of the 151,936-token vocabulary. Turning checkpointing off raises 8B activations from 0.83 to 3.5 GiB at 1,024 tokens [C].

### 9.2 Worked example: `sst` on Qwen3-8B, r=0.99, route `sst/host-bases`, one B200

Inputs from the model's [config](https://huggingface.co/Qwen/Qwen3-8B/raw/main/config.json): hidden size H=4096, N=36 layers, vocabulary V=151,936, untied head. Parameters: P = 8.19B in total, P_t = 6.95B across 252 target matrices (36×7), and 1.24B in embeddings and head [C].

| Step | Term | Arithmetic | GiB |
|---|---|---|---|
| 1 | fp32 weights | 4 × 8.19e9 B | 30.51 |
| 2 | fp32 gradients | same | 30.51 |
| 3 | AdamW m and v | 8 × 8.19e9 B | 61.02 |
| | Subtotal | 16 B/param; this is the README's "about 131 GB" in decimal GB | 122.05 |
| 4 | One block's tail bases on GPU (all 36 blocks total 21.3 GiB, on host) | Tail form, r=0.99 | 0.59 |
| 5 | One block's `W_ref` on GPU (all blocks total 25.9 GiB, on host) | 4 × block target parameters | 0.72 |
| 6 | Activations, checkpointing on, T=1,024 | 1024·4096·2·(36+12) = 0.375; plus 2·1024·151936·2 = 0.580 | 0.955 |
| 7 | Sum before overheads | 122.05 + 0.59 + 0.72 + 0.955 | 124.31 |
| 8 | × fragmentation 1.10, + 1 GiB reserve | 124.31 × 1.1 + 1 | **137.7** |
| 9 | B200 limit | 0.92 × 180 | 165.6: the point estimate fits, with 27.9 GiB spare. The +25% band reaches 172.1, so `estimate` reports **borderline** and `method="auto"` does not choose it (§15) |
| 10 | Host RAM | 21.3 + 25.9, plus pinned staging | **47.2** |

For comparison, the `sst/gpu` route holds everything on the GPU: (122.05 + 21.3 + 25.9 + 0.955) × 1.1 + 1 = **188.2 GiB**. That does not fit a B200, at 1.14 times the limit [C]. The report's "about 173 GiB at 8B" used head-form bases on Llama-3.1-8B and no `W_ref`. On the same Llama-3.1-8B shapes, this model gives 186.3 GiB.

### 9.3 Tables at ρ=0.5 (the OSF range)

These figures are taken from the research notes. All are [C]. Columns show GPU GiB, or GPU / host GiB. The dense ρ=0.5 columns describe weight-space projection with the preserved rank fraction at 0.5. The notes compute them **without** `W_ref`. If such a route uses the cumulative form, add the `W_ref` column: ×1.1 on the GPU, or as-is on the host. Shapes come from real configs: [Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B/raw/main/config.json), [Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B/raw/main/config.json), [Llama-3.1-8B](https://huggingface.co/unsloth/Llama-3.1-8B/raw/main/config.json), [Qwen3-14B](https://huggingface.co/Qwen/Qwen3-14B/raw/main/config.json), [Gemma-3-27B](https://huggingface.co/unsloth/gemma-3-27b-it/raw/main/config.json) (text only), [Llama-3.1-70B](https://huggingface.co/unsloth/Meta-Llama-3.1-70B/raw/main/config.json), [gpt-oss-120b](https://huggingface.co/openai/gpt-oss-120b/raw/main/config.json) and [Qwen3-235B](https://huggingface.co/Qwen/Qwen3-235B-A22B/raw/main/config.json).

| Size (shapes) | `lora-4bit` | `lora-16bit` | `full` fp32 + AdamW | `full` + 8-bit Adam | Dense ρ=.5, all on GPU | Dense ρ=.5, 8-bit Adam, bases on host | Dense ρ=.5, state and bases on host | ZeRO-style offload | `osf` (PEFT as coded) | `W_ref` |
|---|---|---|---|---|---|---|---|---|---|---|
| ~2B (Qwen3-1.7B) | 3.5 | 5.6 | 30.0 | 19.5 | 34.2 | 19.6 / 3.8 | 16.1 / 16.6 | 9.0 / 23.1 | 17.5 | 5.2 |
| ~4B (Qwen3-4B) | 5.1 | 10.7 | 67.8 | 43.2 | 77.8 | 43.5 / 9.1 | 35.1 / 39.1 | 18.7 / 54.1 | 40.2 | 13.5 |
| ~8B (Llama-3.1-8B) | 8.3 | 19.1 | 133.5 | 84.4 | 153.5 | 85.0 / 18.1 | 68.4 / 78.0 | 35.4 / 107.9 | 76.0 | 26.0 |
| ~14B (Qwen3-14B) | 13.2 | 33.5 | 244.3 | 153.9 | 282.2 | 154.9 / 34.5 | 124.2 / 144.6 | 63.7 / 199.6 | 142.1 | 49.2 |
| ~27B (Gemma-3-27B text) | 20.8 | 60.1 | 445.6 | 280.4 | 514.4 | 281.6 / 62.5 | 225.4 / 263.8 | 114.7 / 364.4 | 264.7 | 95.4 |
| ~70B (Llama-3.1-70B) | 45.9 | 151.1 | 1159.6 | 728.1 | 1355.2 | 730.5 / 177.8 | 583.8 / 703.5 | 294.7 / 966.3 | 712.5 | 255.0 |
| ~120B MoE (gpt-oss-120b) | 84.8 | 262.5 | 1916.5 | 1202.2 | 2389.6 | 1215.3 / 430.2 | 972.4 / 1300.3 | 493.9 / 1735.4 | 1426.0 | 430.8 |
| ~235B MoE (Qwen3-235B) | 179.1 | 538.5 | 3856.0 | 2418.2 | 4516.3 | 2425.2 / 600.3 | 1936.3 / 2351.9 | 972.9 / 3227.6 | 2465.1 | 871.2 |

### 9.4 Tables at SST's r=0.99

These are computed by the §9.1 formula, with tail-form fp32 bases and fp32 `W_ref` [C]. Columns show GPU GiB, or GPU / host GiB. `lora` and `full` do not depend on r, so they appear only in §9.3.

| Size (shapes) | Bases, head form | Bases, tail form | `W_ref` | `sst/gpu` | `sst/host-bases` | `sst/8bit-adam` (variant) | `sst/offload-state` | `sst/zero-offload` (untested) |
|---|---|---|---|---|---|---|---|---|
| ~2B (Qwen3-1.7B) | 7.6 | 4.4 | 5.2 | 40.6 | 30.4 / 9.6 | 19.9 / 9.6 | 16.3 / 22.4 | 9.2 / 28.8 |
| ~4B (Qwen3-4B) | 18.0 | 12.4 | 13.5 | 96.3 | 68.6 / 25.9 | 44.0 / 25.9 | 35.7 / 55.9 | 19.2 / 70.9 |
| ~8B (Llama-3.1-8B) | 35.9 | 21.9 | 26.0 | 186.3 | 135.2 / 47.9 | 86.1 / 47.9 | 69.4 / 107.8 | 36.5 / 137.7 |
| ~8B (Qwen3-8B) | 37.0 | 21.3 | 25.9 | 188.2 | 137.7 / 47.2 | 87.7 / 47.2 | 70.6 / 108.2 | 37.1 / 138.7 |
| ~14B (Qwen3-14B) | 68.4 | 41.3 | 49.2 | 343.8 | 246.7 / 90.5 | 156.4 / 90.5 | 125.7 / 200.5 | 65.2 / 255.5 |
| ~27B (Gemma-3-27B text) | 123.8 | 87.9 | 95.4 | 647.2 | 448.9 / 183.3 | 283.7 / 183.3 | 227.5 / 384.5 | 116.9 / 485.1 |
| ~70B (Llama-3.1-70B) | 352.1 | 214.3 | 255.0 | 1675.7 | 1166.0 / 469.3 | 734.5 / 469.3 | 587.8 / 994.9 | 298.7 / 1257.7 |
| ~120B MoE (gpt-oss-120b) | 851.5 | 10.4 | 430.8 | 2401.8 | 1929.9 / 441.2 | 1215.7 / 441.2 | 972.8 / 1311.3 | 494.2 / 1746.4 |
| ~235B MoE (Qwen3-235B) | 1187.8 | 865.5 | 871.2 | 5766.3 | 3876.3 / 1736.7 | 2438.5 / 1736.7 | 1949.6 / 3488.2 | 986.2 / 4364.0 |

**gpt-oss-120b caveat.** Its experts are stored as MXFP4 codes, where SVD means nothing [S](https://huggingface.co/openai/gpt-oss-20b). D-15 also refuses quantized trainable targets. These rows assume dequantised square experts, which is why the tail-form bases are so small, and they assume routers are frozen. They are a scale reference, not a supported route.

### 9.5 Smallest single GPU per route

The table gives the smallest card size that holds the point estimate under the 92% rule [C]. Card sizes:

- 8: 8 GB consumer
- 15: T4
- 24: RTX 3090, 4090, L4
- 32: RTX 5090
- 40 and 80: A100
- 48: L40S
- 80: H100
- 96: RTX PRO 6000
- 141: H200
- 180: B200
- 288: MI350X/MI355X class [U]

"None" means no single card up to 288 GiB.

| Route | ~2B | ~4B | ~8B | ~14B | ~27B | ~70B | ~120B MoE |
|---|---|---|---|---|---|---|---|
| `lora-4bit` | 8 | 8 | 15 | 15 (13.2 of 13.8) | 24 | 80 | 96 |
| `lora-16bit` | 8 | 15 | 24 | 40 | 80 | 180 | 288 |
| `full` | 40 | 80 | 180 | 288 | None | None | None |
| `osf` (PEFT as coded) | 24 | 48 | 96 | 180 | 288 | None | None |
| Dense ρ=.5, all on GPU (no `W_ref`) | 40 | 96 | 180 | None | None | None | None |
| `sst/gpu` | 48 | 141 | 288 | None | None | None | None |
| `sst/host-bases` | 40 | 80 | 180 | 288 | None | None | None |
| `sst/8bit-adam` (variant) | 24 | 48 | 96 (margin 0.6 to 2.2 GiB) | 180 | None | None | None |
| `sst/offload-state` | 24 | 40 | 80 (plus 108 GiB host) | 141 | 288 | None | None |
| `sst/zero-offload` (untested) | 15 | 24 | 40 to 48 | 80 | 141 | None | None |

**Mac unified memory.** Host offload saves nothing on unified memory, so only all-in-memory routes count. The table gives the smallest Mac configuration in GB, assuming 75% usable [U] and then the 92% rule [C]. `sst/8bit-adam` does not exist on MPS.

| Route | ~2B | ~4B | ~8B | ~14B | ~27B | ~70B | ~120B MoE |
|---|---|---|---|---|---|---|---|
| `lora-4bit` | 16 | 16 | 16 | 24 | 32 | 96 | 128 |
| `lora-16bit` | 16 | 16 | 32 | 64 | 96 | 256 | 512 |
| `full` | 48 | 128 | 256 | 512 | None | None | None |
| `osf` | 32 | 64 | 128 | 256 | 512 | None | None |
| `sst/gpu` | 64 | 256 | 512 | 512 | None | None | None |

DGX Spark's 128 GB of unified memory behaves the same way. How much of it the GPU can use was not established [U].

**Multi-GPU.** The table gives the minimum number of cards for `sst` at r=0.99, fully sharded with FSDP2 (Gate 3, D-19), as host-bases / all-on-GPU [C]. The per-rank budget is `R + f·(state/n + one unsharded bf16 block + A)`. It ignores communication buffers. It also shards bases with the weights, while the FSDP2 local right projection needs `V` on every rank, which raises the counts on small cards (see [08-distributed.md](08-distributed.md)).

| Size | 24 GiB | 32 GiB | 48 GiB | 80 GiB | 96 GiB | 141 GiB | 180 GiB |
|---|---|---|---|---|---|---|---|
| ~2B | 2 / 2 | 2 / 2 | 1 / 1 | 1 / 1 | 1 / 1 | 1 / 1 | 1 / 1 |
| ~4B | 4 / 6 | 3 / 4 | 2 / 3 | 1 / 2 | 1 / 2 | 1 / 1 | 1 / 1 |
| ~8B | 8 / 12 | 6 / 8 | 4 / 6 | 2 / 3 | 2 / 3 | 2 / 2 | 1 / 2 |
| ~14B | 16 / 24 | 12 / 16 | 8 / 12 | 4 / 6 | 3 / 4 | 2 / 3 | 2 / 3 |
| ~27B | 32 / 48 | 24 / 32 | 12 / 16 | 8 / 12 | 6 / 8 | 4 / 6 | 3 / 4 |
| ~70B | 128 / 128 | 64 / 96 | 48 / 48 | 24 / 32 | 16 / 24 | 12 / 16 | 8 / 12 |
| ~120B MoE | >128 | >128 / 128 | 96 / 96 | 48 / 48 | 32 / 32 | 24 / 24 | 16 / 16 |

The research found SST results only up to 32B, and its authors call hundreds of billions untested [S](https://arxiv.org/html/2610.22293).

### 9.6 What does not fit, and by how much

The table divides the point estimate by the usable limit. 1.0x means just at the limit [C].

| Model and route | 8 GB card (7.4) | T4 (13.8) | 24 GB (22.1) | 32 GB (29.4) |
|---|---|---|---|---|
| ~8B `lora-4bit` | 1.1x, no | 0.6x, fits | fits | fits |
| ~8B `lora-16bit` | 2.6x | 1.4x | 0.9x, fits | fits |
| ~8B `full` | 18x | 9.7x | 6.0x | 4.5x |
| ~8B `osf` | 10x | 5.5x | 3.4x | 2.6x |
| ~8B `sst/host-bases` (Qwen3-8B) | 19x | 10x | 6.2x | 4.7x |
| ~8B `sst/offload-state` (Qwen3-8B) | 9.6x | 5.1x | 3.2x | 2.4x |
| ~8B `sst/zero-offload` (Qwen3-8B) | 5.0x | 2.7x | 1.7x | 1.3x |
| ~4B `sst/host-bases` | 9.3x | 5.0x | 3.1x | 2.3x |
| ~4B `sst/zero-offload` | 2.6x | 1.4x | 0.9x, fits | 0.7x, fits |
| ~2B `sst/host-bases` | 4.1x | 2.2x | 1.4x | 1.0x |
| ~2B `sst/8bit-adam` (variant) | 2.7x | 1.4x | 0.9x, fits | fits |
| ~2B `sst/zero-offload` | 1.3x | 0.7x, fits (29 GiB host) | fits | fits |

In plain terms:

- **An 8B model cannot run SST on any consumer card, by any route,** including the 32 GB RTX 5090. On a T4 it is 10 times over.
- **On an 8 GB card,** `lora-4bit` holds up to about 4B, and `sst` with AdamW holds about 0.3B (P_max 0.32B) [C].
- **An RTX 3090 or 4090** runs SST with AdamW up to about 1.2B. With the untested ZeRO-style offload it reaches 4B, but needs 71 GiB of host RAM [C].

Single-card ceilings for 8B `sst` [C]:

| Card | 8B `sst` |
|---|---|
| A100 40 | Only `sst/zero-offload`, at the limit (36.5 to 37.1 against 36.8), with about 138 GiB host RAM |
| A100 80, H100 80 | `sst/offload-state` (70.6), with 108 GiB host RAM |
| H200 141 | `sst/8bit-adam` comfortably. `sst/host-bases` misses (137.7 against 129.7) |
| B200 180 | `sst/host-bases` (137.7). Not `sst/gpu` (188.2) |
| 2× RTX PRO 6000, FSDP2 | `sst/host-bases`, at 71 GiB per rank |

**D-18 qualified.** D-18 names a 96 GB RTX PRO 6000 or a B200 for real SST on 8B. Both work, with conditions:

- **B200:** fits paper-exact SST with bases and `W_ref` streamed from host.
- **RTX PRO 6000:** fits paper-exact AdamW only through `sst/offload-state` (70.6 GiB on the GPU plus about 108 GiB of host RAM, slow and untested). Otherwise it needs the 8-bit Adam variant, which clears the limit by only 0.6 GiB on Qwen3-8B at 1,024 tokens. That margin disappears at about 2,048 tokens.
- **Two RTX PRO 6000 cards with FSDP2** fit cleanly once Gate 3 ships.

---

## 10. The T4 verdict

A free T4 is a real fine-tuning machine for LoRA and a demonstration machine for SST.

| On a free T4 (13.8 GiB usable) | Ceiling [C] | Status |
|---|---|---|
| `lora` on a 4-bit base | About 8B (8.3 to 8.9 GiB). 14B fits on paper at 13.2 GiB, too tight to promise | Promised up to about 8B (D-18) |
| `lora` on a 16-bit base | About 4B (10.7 GiB) | Promised up to about 4B |
| `osf` (PEFT as coded) | About 1.3B, by linear scaling of the ~2B figure (17.5 GiB) | Promised up to about 1B |
| `sst/host-bases`, paper-exact AdamW | 0.71B | Promised up to about 0.7B |
| `sst/8bit-adam` (variant) | 1.14B on GPU memory. Host needs about 6 to 7 GiB for bases and `W_ref` | The "about 1B on Colab" in D-18. Labelled a variant |
| `sst/zero-offload` on Kaggle (untested) | About 1.5 to 2.0B, limited by host RAM | Best effort only |
| Any 8B constrained route | 10 times over | Not offered |

**Qwen3-0.6B, a launch model, sits on the edge.** At the 0.75B the report lists, `sst/host-bases` needs about 14.1 GiB against 13.8 [C]. Counting the tied embedding once could pull it under. `estimate` settles this from the safetensors header, and Gate 1 measures it. SmolLM2-135M fits easily.

**Practical ceilings by platform:**

- **Colab** gives "at most 12 hours, depending on availability", publishes no limits, and guarantees no resources [S](https://research.google.com/colaboratory/faq.html). Its host RAM is folk knowledge at about 12.7 GB [U], which rules out every host-offload route above about 0.6 to 0.75B. The constrained ceiling on Colab is therefore about 1.1B with 8-bit Adam and about 0.7B with AdamW [C].
- **Kaggle** offers T4 or P100 (the P100 is unusable on torch 2.15+). Session limits are reported as both 12 h and 9 h, and the weekly quota of about 30 h is user-reported ([gpuperhour](https://gpuperhour.com/blog/free-cloud-gpus-and-credits), secondary) [U]. T4 x2 is reported as two 15 GB GPUs with about 29 to 30 GB of host RAM ([llcuda](https://llcuda.github.io/guides/kaggle-setup), secondary) [U]. Host RAM caps the offload route at about 2B.

**What a free-tier user gets:**

- `lora` on a 4-bit base on models up to about 8B, with the forgetting report on by default. That is diagnosis.
- Constrained tuning on models up to about 1B. That is prevention at a size with the weakest evidence. SST's Science deltas are −0.9 at 14B and 8B, −3.6 at 4B and −8.0 at 1.7B [S](https://arxiv.org/html/2610.22293), and the research found SST results only down to 1.7B.
- The docs must say so in one sentence: on free hardware tailtune measures forgetting well, but prevents it only on small models where the published protection is weakest. Real SST means renting.

The 4-bit `tail-lora` benchmark (Gate 4) is the experiment that could change this; see [01-product-strategy.md](01-product-strategy.md).

---

## 11. The cheap paid route

Prices are USD per GPU-hour for a single GPU [S]:

| GPU (VRAM) | RunPod community / secure, 2026-11-04 | Lambda, effective 2026-11-16 | Vast.ai on-demand min / median, 2026-11-08 | Vast.ai interruptible min / median | Modal |
|---|---|---|---|---|---|
| T4 (16) | not listed | not listed | 0.10 / 0.15 | 0.06 / 0.06 | 0.59 |
| RTX 3090 (24) | 0.22 / 0.50 | not listed | 0.10 / 0.20 | 0.08 / 0.16 | not listed |
| RTX 4090 (24) | 0.34 / 0.89 | not listed | 0.35 / 0.46 | 0.22 / 0.34 | not listed |
| L4 (24) | 0.44 / 0.59 | not listed | not sampled | not sampled | 0.80 |
| RTX 5090 (32) | 0.69 / 1.19 | not listed | 0.44 / 0.60 | 0.27 / 0.48 | not listed |
| A100 40 | no price returned | 1.99 | 0.34 / 0.67 | 0.14 / 0.34 | 2.10 |
| A100 80 | 1.19 / 1.79 (PCIe) | 3.19 (8x only) | mixed sample | mixed sample | 2.50 |
| RTX PRO 6000 (96) | 1.69 / 2.49 | not listed | 0.80 / 1.46 | 0.27 / 1.07 | 3.03 |
| H100 (80) | 2.69 / 3.99 (SXM); 1.99 / 2.89 (PCIe) | 4.89 (SXM); 3.79 (PCIe) | 2.20 / 2.50 (SXM) | 0.67 / 1.74 | 3.95 |
| H200 (141) | 3.59 / 5.29 | not listed | 2.99 / 4.41 | 1.32 / 2.63 | 4.54 |
| B200 (180) | 5.98 / 7.99 | 7.99 | 7.50 / 10.01 (n=2) | 5.94 / 7.98 (n=2) | 6.25 |

Sources: [RunPod](https://www.runpod.io/pricing), [Lambda](https://lambda.com/pricing), [Vast.ai bundles API](https://console.vast.ai/api/v0/bundles/), [Modal](https://modal.com/pricing).

**Caveats:**

- The Vast.ai figures are a single one-day snapshot of a small market sample. The API returned at most 64 offers per query, and only two B200s. Treat them as a floor, not a quote.
- Lambda's prices take effect on 2026-11-16, after retrieval. Current Lambda prices were visible only after login.
- RunPod spot prices are unverified [U].

**What a typical fine-tune costs [C].** Training hours are an assumption [J], because no tailtune run has been timed. For scale: SST's reported standard fine-tuning rate (1.73 s per step on 8 L40S at batch 16) implies about 1.4 GPU-hours for 2,000 examples over 3 epochs on an 8B model. Projected arms will be slower. Storage and failed runs are excluded.

| Job | Hardware | Assumed hours | Cost |
|---|---|---|---|
| `lora` 4-bit, 8B | RTX 4090, RunPod | 1 | $0.34 to $0.89 |
| `lora` 16-bit, 8B | RTX 3090 or 4090 | 1 | $0.22 to $0.89 |
| `sst/host-bases`, Qwen3-8B, bases precomputed | 1× B200 | 2 | $12 to $16 |
| Same, with the SVD pass run on the rental's CPU | 1× B200 | +1.5 to 3.5 | +$9 to $28 |
| `sst/8bit-adam` (variant), 8B | 1× RTX PRO 6000 | 2 (step time unmeasured) | $1.60 to $5.00 |
| `sst/offload-state`, 8B | 1× H100 80 PCIe or SXM | 2 × an unknown host-step slowdown | from $4 to $8, before slowdown |
| `sst/host-bases`, 8B, FSDP2 (Gate 3) | 2× RTX PRO 6000 | 2 | $6.80 to $10 |
| Forgetting report, non-thinking model | 24 GB GPU | 8 to 10 min (estimate, see [09-evaluation.md](09-evaluation.md)) | under $0.20 |

The first public benchmark budget ($300 to $700 [C]) is in [13-roadmap.md](13-roadmap.md).

---

## 12. Windows, WSL2, Linux, containers and free compute

| Feature | Linux | WSL2 | Windows native |
|---|---|---|---|
| CUDA training, single GPU | Yes | Yes, GeForce and RTX only; "TCC mode and data center GPUs are not supported" [S](https://docs.nvidia.com/cuda/wsl-user-guide/index.html) | Yes |
| NCCL multi-GPU, FSDP2 | Yes | Not verified [U]; GPUs cannot be filtered by index | No: Windows has every backend "but NCCL" [S](https://docs.pytorch.org/docs/2.15/distributed.html) |
| bitsandbytes | Yes | Yes (Linux wheel) | Yes: `win_amd64` wheels, tested on T4 [S](https://pypi.org/pypi/bitsandbytes/0.50.2/json) |
| Triton and Inductor GPU kernels | Yes | Yes | No: torch depends on `triton` only on Linux [S](https://pypi.org/pypi/torch/2.15.0/json); community fork only |
| FlashAttention through SDPA | Yes | Yes | Yes from torch 2.15 |
| `flash-attn` package | Yes | Yes | "Might work" [S](https://github.com/Dao-AILab/flash-attention) |
| DeepSpeed | Yes | Probably [U] | Builds with `build_win.bat`; no AIO, so no NVMe offload [S](https://github.com/deepspeedai/DeepSpeed/blob/master/README.md) |
| Host-offload routes | Yes | VM gets 50% of Windows RAM by default [S](https://learn.microsoft.com/en-us/windows/wsl/wsl-config); pinned memory is limited | Yes, untested |
| vLLM for the report | Yes | Yes | No; fall back to Hugging Face generation [S](https://unsloth.ai/docs/get-started/install/windows-installation.md) |
| Data loading | Normal | Normal | Unsloth sets `dataset_num_proc=1` "to avoid a crashing issue" [S](https://unsloth.ai/docs/get-started/install-and-update/windows-installation); spawn behaviour not re-verified [U] |

**What breaks where:**

- **WSL2:** host RAM. A 64 GB laptop gets a 32 GB VM by default, which is below the 47 GiB that `sst/host-bases` needs for 8B [C]. `doctor` detects WSL2 and prints the `.wslconfig` fix.
- **Windows native:** multi-GPU, which is unsupported, and Triton kernels.
- **Linux:** nothing structural. Linux is the only first-class OS.

A clean single-GPU Windows path for data ingestion and `lora` is still worth building, because Axolotl sends Windows users to WSL2 or Docker and LlamaFactory requires a manual torch install (see the research report).

**Docker plan** [J]:

- **Images:** publish one CUDA image and one ROCm image per release.
- **CUDA image:** built on an official PyTorch CUDA image matching the tested torch, as TRL's CI does with `pytorch/pytorch` images [S](https://github.com/huggingface/trl/blob/main/.github/workflows/tests.yml).
- **ROCm image:** Ubuntu plus the ROCm torch wheels, which install ROCm as a pip dependency in 2.15 [S](https://pytorch.org/blog/pytorch-2-15-release-blog/). AMD's `rocm/pytorch-training` images are being deprecated in favour of `rocm/primus`.
- **Contents:** both images include `[bnb]`. Both carry the GGUF converter in its own virtual environment (D-23, [10-export-and-deployment.md](10-export-and-deployment.md)).
- **GPU access:** the NVIDIA Container Toolkit 1.20.1 defaults to just-in-time CDI and includes WSL2 CDI fixes [S](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/release-notes.html).
- **Windows users:** Docker on Windows runs inside WSL2, so it inherits the 50% RAM default.

**Free compute:**

| Service | What you get | Limits | Source quality |
|---|---|---|---|
| Google Colab free | One GPU, user-reported as a T4 | "At most 12 hours, depending on availability". No published limits. RAM about 12.7 GB [U] | Primary FAQ; GPU type and RAM are folk knowledge |
| Kaggle | T4, T4 x2 or P100 | 12 h or 9 h per session (conflict); about 30 h a week [U] | Secondary ([gpuperhour](https://gpuperhour.com/blog/free-cloud-gpus-and-credits), which sells rentals) |
| Lightning | "50* free GPU hours"; 5 free credits plus 25 with a card | Credits expire after 12 months; 4-hour Studio restarts | Primary [S](https://lightning.ai/pricing); terms unclear, so do not document around them |
| Hugging Face ZeroGPU | Shared RTX PRO 6000 | Reported 5 min/day, Gradio Spaces only. Not for training [U] | Secondary |
| SageMaker Studio Lab | Closed to new customers [U] | n/a | Secondary |
| Modal Starter | $30 a month in credits | Covers about 10 L4 hours | Primary [S](https://modal.com/pricing) |
| AMD developer cloud | Cost-free access, discretionary | Application | Primary [S](https://www.amd.com/en/developer/resources/cloud-access.html) |

---

## 13. Performance features per platform

The projection runs in `on_optimizer_step` on fp32 masters, between steps (D-13). Attention kernels and fused optimizers do not care what happens between steps, so most features are compatible. The conflicts are listed in the last column.

| Feature | Where it works | Gain claimed, and by whom | Conflict with post-step projection |
|---|---|---|---|
| SDPA | All backends | Baseline | None |
| FlashAttention 2/3/4 | FA2: Ampere to Hopper. FA3: Hopper. FA4: Hopper and Blackwell. ROCm through CK or Triton. Not Turing | Speed and memory claimed by the FA authors; no independent 2026 benchmark found [U] | None: attention does not touch weights |
| Triton kernels | Linux; compute capability 8.0+, ROCm 6.2+ [S](https://github.com/triton-lang/triton) | D-04 allows them only with a measured gain. Inductor on a T4 is unverified [U] | None for weights; check each kernel |
| Liger Kernel | Linux; CUDA and ROCm | 20% throughput and 60% memory, by Liger, on LLaMA 3-8B, 8 A100s, FSDP1 [S](https://github.com/linkedin/Liger-Kernel). Small at 1,024 tokens, where 8B activations are about 0.8 GiB [C] | None for weights. Its patched forward reads `lm_head` weights directly, bypassing PEFT's `modules_to_save` [S](https://github.com/linkedin/Liger-Kernel/blob/main/src/liger_kernel/transformers/model/llama.py#L158-L162) |
| `torch.compile` | Linux CUDA primarily | Workload-dependent, unmeasured | Backward hooks break the graph and fire together after the compiled backward. Later hook changes go unnoticed [S](https://github.com/pytorch/pytorch/blob/v2.15.0/docs/source/user_guide/torch_compiler/torch.compiler_nn_module.md). This affects `osf`, whose gradient hooks are risky under compile. In-place post-step edits sit outside the graph, which is untested |
| Gradient checkpointing | All | Essential: 8B activations fall from 3.5 to 0.83 GiB at 1,024 tokens [C] | Under FSDP2, `use_reentrant=True` silently loses gradients [S](https://github.com/huggingface/transformers/issues/50424). Force `use_reentrant=False` (D-19) |
| Fused AdamW | CUDA, XPU, HPU; MPS as a prototype | Not the default; foreach uses more peak memory [S](https://docs.pytorch.org/docs/2.15/generated/torch.optim.AdamW.html) | None |
| 8-bit Adam (bitsandbytes) | CUDA, ROCm, XPU; not MPS or Gaudi | "75% less GPU memory", by bitsandbytes [S](https://huggingface.co/docs/bitsandbytes/main/en/optimizers). Saves about 6 B/param at fp32 master [C] | None mechanically. It changes the optimizer, so it is a variant for `sst`. Compatibility with DTensor sharding is unverified [U] |
| DeepSpeed offload | Linux; no NVMe on Windows | Large VRAM saving, large step-time cost [S](https://www.deepspeed.ai/docs/config-json/) | Works only through the `safe_*` APIs. ZeRO-2 is the reference only (D-19) |
| Unsloth kernels | n/a (no dependency, D-05) | "2x-5x faster with 80% less VRAM" circulates in vendor-derived material. Unsloth's own Qwen3.8 guide now says about 1.5x and 50% [S](https://unsloth.ai/docs/models/qwen3.8/train.md) | Not used |

---

## 14. `tailtune doctor` specification

`doctor` lives in `tailtune.hardware`, and `tailtune.cli` only formats its output.

**Behaviour:**

- Imports nothing heavy until each check needs it. A missing torch is a result, not a crash.
- Makes no network calls (D-08).
- Finishes in under 10 seconds without `--deep` [J].
- `--json` writes the same results as machine-readable data, so `estimate`, bug reports and CI can read them.

| ID | Check | Detection | PASS | WARN | FAIL |
|---|---|---|---|---|---|
| D01 | Python | `sys.version_info` | 3.11 or later | Newer than the tested set | 3.10 or earlier |
| D02 | OS and architecture | `platform`; WSL2 from `/proc/sys/kernel/osrelease` | Linux x86_64 | Windows native, WSL2, aarch64 (Tier 2) | Intel Mac (no torch wheel) |
| D03 | Hosted service | Platform environment markers; exact markers confirmed at Gate 0 [U] | n/a | Colab or Kaggle: print the quota caveats | n/a |
| D04 | torch | `torch.__version__`, `torch.version.cuda`, `torch.version.hip` | 2.14 or 2.15 | Newer than tested | Below 2.14, or not installed |
| D05 | Accelerator and tier | `torch.cuda.get_device_capability`; on ROCm, device properties; `torch.backends.mps.is_available()`; `torch.xpu.is_available()` | CC 8.0 to 9.0, and 10.0 or 12.0 on x86_64 | CC 7.5 (Tier 1 reduced); Tier 2 or 3 platforms; CPU only | CC below 7.5 (Pascal, Volta, Maxwell) |
| D06 | Driver | NVML if installed, else `nvidia-smi --query-gpu=driver_version` | 580 or later | Not readable (WSL2 NVML limits) | Below 580 with a CUDA 13 torch build |
| D07 | VRAM | `torch.cuda.mem_get_info()` per device | Free ≥ 90% of total | Less: another process holds memory | n/a |
| D08 | Real bf16 | `torch.cuda.is_bf16_supported(including_emulation=False)`; on other backends, a bf16 matmul probe against fp32 | Native | Emulated only (T4): fp16 autocast will be used | n/a |
| D09 | Host RAM | psutil total and available. On WSL2, compare with the Windows total if readable | Enough for `sst/host-bases` at the largest promised size on this tier | Below that; on WSL2, print the `.wslconfig` fix | n/a |
| D10 | Disk | `shutil.disk_usage` on the Hugging Face cache and the bases cache | ≥ 60 GiB free [J] | Less | n/a |
| D11 | Multi-GPU | `torch.cuda.device_count()`, `torch.distributed.is_nccl_available()` | n/a | More than one GPU on Windows: single-GPU only | n/a |
| D12 | Core packages | `importlib.metadata` versions | Transformers 5.x, Accelerate ≥ 1.14 | Untested newer versions | Transformers 4.x |
| D13 | Optional extras | bitsandbytes version and backend; `flash-attn`; `triton`; `deepspeed` | Present where useful | bitsandbytes missing ("install `tailtune[bnb]` for `lora` on a 4-bit base"); `flash-attn` on Turing (not supported) | n/a |
| D14 | Attention | SDPA backend flags | Available | FA SDPA backend off | n/a |
| D15 | MPS | `platform.mac_ver()`, torch version | macOS 15+, torch 2.15+ | n/a | macOS 14, or torch 2.14 on MPS |
| D16 | XPU | FP64 support probe [U] | Present | Absent (Arc A): no loss scaler | n/a |
| D17 | CPU SVD | Times a 1024×1024 fp32 SVD and checks orthonormality | Passes | Slow: print the expected 8B SVD hours | Orthonormality fails |
| D18 | Verdict | Combines D02, D05, D07 and D09 | Prints the tier, what is promised here, and "run `tailtune estimate <model>`" | | |

`--deep` adds a 2-rank gloo initialisation, a bf16 rounding demonstration and a short fp16 scaler test. The exit code is an open question.

Illustrative output on a Colab T4:

```text
$ tailtune doctor                                   # illustrative output
tailtune 0.0.1 doctor

  Platform    Linux x86_64, Google Colab detected                  WARN  no guaranteed quota
  Python      3.13                                                 PASS
  torch       2.14.1+cu130                                         PASS  tested: 2.14, 2.15
  GPU 0       Tesla T4, compute capability 7.5                     PASS  Tier 1 reduced
  Driver      580 or later                                         PASS
  VRAM        15.0 GiB total, 14.7 GiB free                        PASS
  bf16        emulated only                                        WARN  fp16 autocast + fp32 master
  Host RAM    12.7 GiB total, 11.0 GiB available                   WARN  host-offload routes limited
  Disk        71 GiB free                                          PASS
  Packages    transformers 5.22.1, accelerate 1.15.0, peft 0.21.2  PASS
  Extras      bitsandbytes missing                                 WARN  pip install "tailtune[bnb]"
              flash-attn not supported on Turing; SDPA in use
  CPU SVD     1024x1024 in 0.2 s, orthonormal                      PASS

Verdict: Tier 1 reduced.
  Promised here: lora on a 4-bit base up to ~8B, lora up to ~4B,
  osf up to ~1B, sst up to ~0.7B (~1.1B with 8-bit Adam, a variant).
  Next: tailtune estimate <model>
```

---

## 15. `tailtune estimate` specification

**Inputs:**

- A model ID or local path, plus a revision.
- `--method`: `auto` by default, or `lora`, `osf`, `sst`, `full` or `freeze`.
- `--r`: 0.99 by default for `sst`. `osf` uses ρ=0.5.
- LoRA rank: 16.
- Tokens per micro-batch: 1,024 by default.
- Checkpointing: on.
- Optimizer: `adamw`, or `adamw-8bit` as an explicit opt-in.
- `--gpus`.
- Hardware, from `doctor`'s probe by default, or overridden with `--vram`, `--host-ram` or `--gpu B200` for planning.
- `--json`.

**Reading parameter counts without downloading weights.** The order is a design proposal [J]:

1. **The safetensors header (primary).** Fetch `model.safetensors.index.json`, then range-read the header of each shard: an 8-byte length followed by JSON. This gives every stored tensor's name, dtype and shape for a few kilobytes per shard. It is the only source that shows fused QKV, 3D expert tensors and MXFP4 storage, which is how the research found those quirks (see [06-models.md](06-models.md)). A Hugging Face Hub client helper for this may exist. Its name and behaviour are to be confirmed at Gate 0 [U].
2. **`config.json` (fallback).** Use the shape arithmetic in the research notes, reading `text_config` for composite models. 0.0.1 ships this path only. Its parameter counts agree with the safetensors headers to within 0.02% on Qwen3-8B, Qwen3-30B-A3B, Qwen2.5-0.5B and 7B, Mistral-Nemo and Llama-3.1-8B [M] (`tests/test_shapes_network.py`).
3. **A meta-device skeleton (last resort, for unknown architectures).** Instantiate the model on the meta device, which allocates no memory, with `trust_remote_code=False` (D-08).

Tensors then pass through the same `tailtune.targets` resolver as `inspect`, so the estimate and the coverage report agree. Tied tensors must be counted once: Qwen3-0.6B ties its embeddings but its checkpoint also stores `lm_head.weight`, so a naive sum of header shapes overstates P by 0.16B (21%) [M]. Gated models need a token for any of these sources.

**Formula implementation.** The following is an untested design proposal:

```python
# Untested design proposal: core of tailtune.hardware.estimate
from dataclasses import dataclass

GIB = 2**30

@dataclass(frozen=True)
class Shapes:
    targets: tuple[tuple[int, int], ...]  # every target matrix; [E,m,n] already split into E
    block: tuple[tuple[int, int], ...]    # target matrices of the largest decoder block
    p_total: int                          # all parameters, tied tensors counted once
    hidden: int
    layers: int
    vocab: int

@dataclass(frozen=True)
class Calib:                              # Gate 1 replaces these values
    reserve_gib: float = 1.0
    frag: float = 1.10
    live_tensors: float = 12.0
    usable: float = 0.92
    band: tuple[float, float] = (0.85, 1.25)

def bases_bytes(mats, r, form="tail"):
    total = 0
    for m, n in mats:
        k = int(r * min(m, n))
        if form == "head":
            total += (m + n) * k
        else:  # per side, store whichever of head or tail is smaller
            total += m * min(k, m - k) + n * min(k, n - k)
    return 4 * total                      # fp32 only; bf16 bases make the projector inexact

def activation_bytes(s, tokens, act_bytes=2, c=Calib()):
    hidden = tokens * s.hidden * act_bytes * (s.layers + c.live_tensors)
    logits = 2 * tokens * s.vocab * act_bytes
    return hidden + logits

def sst_route(s, route, r=0.99, tokens=1024, act_bytes=2, c=Calib()):
    p = s.p_total
    p_t = sum(m * n for m, n in s.targets)
    p_blk = sum(m * n for m, n in s.block)
    b_all, b_blk = bases_bytes(s.targets, r), bases_bytes(s.block, r)
    w_ref, w_ref_blk = 4 * p_t, 4 * p_blk
    gpu, host = {
        "sst/gpu":           (16 * p + b_all + w_ref,       0),
        "sst/host-bases":    (16 * p + b_blk + w_ref_blk,   b_all + w_ref),
        "sst/8bit-adam":     (10.03 * p + b_blk + w_ref_blk, b_all + w_ref),
        "sst/offload-state": (8 * p + b_blk + w_ref_blk,    8 * p + b_all + w_ref),
    }[route]
    a = activation_bytes(s, tokens, act_bytes, c)
    return c.reserve_gib + c.frag * (gpu + a) / GIB, host / GIB

def verdict(gpu_gib, host_gib, vram_gib, ram_gib, c=Calib()):
    limit = c.usable * vram_gib
    if gpu_gib > limit or host_gib > ram_gib:
        return "does not fit"
    return "borderline" if gpu_gib * c.band[1] > limit else "fits"
```

The first draft of this sketch called a route "does not fit" only when the low end of its band exceeded the limit, so a point estimate up to 18% over the limit was "borderline" and would run when named. That contradicts the reason for the asymmetric band below. 0.0.1 implements the rule above: the point estimate must fit, and the upper end decides between fits and borderline.

`act_bytes` is 4 on a T4 for the forced-fp32 families in §7. The `lora`, `full`, `freeze` and `osf` routes follow §9.1 in the same way. `freeze` uses `W = 2·P_frozen + 4·P_train`, with gradients and optimizer state on `P_train` only.

**Output.** One row per route, showing:

- the point estimate and its uncertainty range;
- host GiB;
- the verdict on this machine;
- the smallest single card that holds the point estimate.

Every figure carries the roadmap's label: computed, uncalculated-against-hardware until Gate 1.

The following is illustrative output for Qwen3-8B on a Colab T4:

```text
$ tailtune estimate Qwen/Qwen3-8B                   # illustrative output
Model     Qwen/Qwen3-8B @ <revision>, read from safetensors headers (no weights downloaded)
          8.19B parameters; 252 target matrices hold 6.95B; embeddings and head 1.24B
Machine   Tesla T4: limit 13.8 GiB (92% of 15.0); host RAM 12.7 GiB      [from doctor]
Settings  1,024 tokens per micro-batch, checkpointing on, fp16 autocast, fp32 master

Route                Method   GPU GiB (range)        Host GiB   Here           Smallest single card
lora-4bit            lora       8.9 (7.6-11.1)          -       fits           T4 15 GB
lora-16bit           lora      19.5 (16.6-24.4)         -       does not fit   24 GB
osf                  osf       78.1 (66.4-97.6)         -       does not fit   96 GB
sst/host-bases       sst      137.7 (117.0-172.1)     47.2      does not fit   B200 180 GB
sst/8bit-adam *      sst       87.7 (74.5-109.6)      47.2      does not fit   96 GB (borderline)
sst/offload-state    sst       70.6 (60.0-88.3)      108.2      does not fit   80 GB + 108 GiB RAM
full (baseline)      full     136.3 (115.9-170.4)       -       does not fit   B200 180 GB

* Optimizer variant: published SST results do not transfer to it.
All figures [C]: computed from shapes, uncalculated-against-hardware until Gate 1.
Ranges are -15% / +25% until calibrated.

method="auto" -> lora on a 4-bit base. sst needs about 138 GiB of GPU memory for this
model; the smallest single card is a B200 180 GB.
```

**How it feeds `method="auto"`.** The rule is provisional until Gate 1 measurement 5 decides between OSF and SST [J]:

1. Rank the candidates by protection: `sst` (routes `sst/gpu` and `sst/host-bases` only), then `osf`, then `lora` on a 16-bit base, then `lora` on a 4-bit base.
2. Choose the first candidate whose verdict is **fits**, with host RAM included.
3. Never choose `sst/8bit-adam`, `sst/offload-state`, `sst/zero-offload`, `full` or `freeze`. These run only when named explicitly.
4. Print one line naming the choice, the strongest route that did not fit, and the smallest card that would hold it. When the chosen constrained method runs on a model below 4B, add the published small-model delta (for example, −8.0 points at 1.7B).

**What it refuses.** A refusal exits with code 4 and happens before any weights load:

- an explicitly requested method or route whose verdict is **does not fit**;
- a route whose host need exceeds available RAM;
- low-precision trainable targets (D-12);
- quantized trainable targets such as MXFP4 experts (D-15);
- multi-GPU on Windows;
- bf16 autocast on a GPU without native bf16.

A **borderline** route runs when requested, with a warning. `train` then records peak memory after the first optimizer step in `lineage.json`, which also feeds calibration. A `--ignore-estimate` override, recorded in the lineage, is an open question.

**Uncertainty.** Until Gate 1, every figure shows a −15% / +25% band [J]. The band is asymmetric because the activation constant 12 is a rough guess and the allocator behaviour is unmeasured. Underestimating causes a crash mid-run, which is worse than a cautious refusal. After Gate 1, the band becomes the largest observed error plus 5 points.

---

## 16. Calibration plan (Gate 1, measurement 3)

**Hardware:**

- One rented RTX PRO 6000 96 GB. At 1.7B every route fits, the largest being `sst/gpu` at 40.6 GiB [C].
- One Kaggle T4, for the fp16 path at 0.3B.
- Modal L4, later, for nightly assertions.

At $0.80 to $2.49 per hour, a day of measurements costs under $60 [C].

**Models:** Qwen3-1.7B, which matches the ~2B rows in §9, and one dense model of about 0.3B, chosen at Gate 1 [J].

**Protocol:**

1. For every route in §9.1 that the method code supports at that point, record `torch.cuda.mem_get_info()` before CUDA initialisation, after load, after the first backward, after the first optimizer step and after projection. Also record `max_memory_allocated()` and `max_memory_reserved()`, and host RSS for the offload routes.
2. Sweep tokens per micro-batch at 512, 1,024, 2,048 and 4,096 with checkpointing on, and at 1,024 with it off.
3. Run more than 5 optimizer steps everywhere, and more than 20 on the T4.

**How the formula is corrected:**

- `R` becomes the measured baseline after CUDA initialisation.
- `f` becomes the ratio of reserved to allocated at peak.
- The activation constants come from a linear fit over the token sweep. This includes the `12`, and checks whether the logits term needs a factor for the loss implementation.
- Weight, gradient, optimizer, bases and `W_ref` terms are exact by construction and are verified against tensor sizes, not fitted.
- The constants go into a versioned `calibration.json`, keyed by GPU family, torch version and date, with T4 fp16 kept separate.
- **Acceptance:** every route's predicted peak reserved memory is within ±10% of the measured value at both sizes [J]. If it is not, keep the wider band and publish the residuals.

**External check.** Recompute `lora-4bit` and `lora-16bit` for 8B, 14B, 27B and 70B and publish them beside Unsloth's minimums [S](https://unsloth.ai/docs/get-started/fine-tuning-for-beginners/unsloth-requirements):

| | 8B | 14B | 27B | 70B |
|---|---|---|---|---|
| Unsloth QLoRA, GB | 6 | 8.5 | 22 | 41 |
| tailtune `lora-4bit` today, GiB [C] | 8.3 | 13.2 | 20.8 | 45.9 |
| Unsloth 16-bit LoRA, GB | 22 | 33 | 64 | 164 |
| tailtune `lora-16bit` today, GiB [C] | 19.1 | 33.5 | 60.1 | 151.1 |

The 16-bit column lands within about 13%. The 4-bit column runs high on small models because tailtune keeps embeddings and head at bf16 and applies LoRA to all seven linears. Unsloth's figures are vendor "absolute minimums" that depend on its kernels, so a match is a sanity check, not a validation. Explain the gaps; do not tune them away.

**Spot check at 8B.** The Gate 1 SST reference run on a B200 records peak memory for `sst/host-bases` on Qwen3-8B, the one large-model data point.

---

## 17. Hardware CI summary

Details are in [11-engineering-and-release.md](11-engineering-and-release.md).

| Trigger | Where | What runs |
|---|---|---|
| Every PR | Free GitHub Linux, macOS and Windows runners | SVD and projection numerics, hook order, 2-rank gloo FSDP2; Python 3.11 and 3.13; torch 2.14 and 2.15 |
| Weekly | Kaggle T4 kernel | fp16 path, more than 5 optimizer steps, control and canary legs. Copy the pattern from Unsloth's workflow, not its AGPL-headed code |
| Nightly | Modal L4, inside the $30 monthly credit | Small real model; memory asserted against `estimate` |
| Per release | Modal 2× L4 or A100; one RTX PRO 6000 or B200 | DDP and later FSDP2, save and reload, projection invariants, Blackwell gate |
| When credits land | AMD MI300X | The Tier 1 promotion job in §4 |

---

## 18. Top gaps

1. **No GPU memory has been measured.** Every figure in §9 to §15 is computed. Gate 1 fixes this for 0.3B and 1.7B, and only partly for 8B.
2. **The activation constant is a guess,** especially for MoE, where FFN memory scales with active experts.
3. **Host-streaming step time is unknown.** The 2.5 s per step for `sst/host-bases` rests on an assumed link speed. The offload routes' step times are unmeasured.
4. **Neither SVD time nor projection time has been measured on a GPU host.** The CPU timings come from one laptop running NumPy. SST's own overhead figures may not be quoted (D-02).
5. **Free-tier facts are weak.** Colab RAM, Colab's GPU type, and Kaggle's session and weekly limits rest on folk or secondary sources.
6. **AMD has no independent LLM fine-tuning benchmark,** and `gesvdj` accuracy on ROCm is unmeasured.
7. **Several T4 and compile questions are unverified:** emulated bf16 slowdown on a T4, Inductor on a T4, `torch.compile` with in-place post-step edits, and 8-bit Adam with DTensor.
8. **WSL2 multi-GPU, Windows `DataLoader` behaviour and WSL2 overhead** were not verified.
9. **Apple's threadgroup-memory limit and the usable share of unified memory** are unconfirmed.
10. **Rubin and TorchTPU** have no rental listing or GA confirmation.

---

## Implementation checklist

**Gate 0, Claim**
- [ ] `tailtune.hardware.probe` with checks D01 to D18, `--json`, and no network calls.
- [ ] `tailtune doctor` CLI formatting, with tier verdict text taken from §2.
- [ ] Safetensors-header reader with range requests, plus the `config.json` and meta-device fallbacks.
- [ ] `tailtune.hardware.estimate` implementing §9.1 for every route ID, with tail-form bases and `W_ref` for `sst`.
- [ ] Uncertainty band, verdicts, smallest-card lookup, and the computed, uncalculated-against-hardware label on every figure.
- [ ] Exit code 4 refusals from §15.
- [ ] Unit tests reproducing the §9.3 and §9.4 tables from stored configs to 0.1 GiB.
- [ ] A docs page with the support matrix and the free-tier sentence from §10.

**Gate 1, Verify**
- [ ] The calibration run in §16; `calibration.json`; published residuals.
- [ ] The Unsloth comparison table, recomputed and published.
- [ ] Timings of `torch.linalg.svd` on the rental's CPU for Qwen3-1.7B and Qwen3-8B.
- [ ] Measured host-streaming time per optimizer step for `sst/host-bases`.
- [ ] Peak memory recorded on the 8B SST reference run.
- [ ] Qwen3-0.6B measured on a T4 to settle the edge case.

**Gate 2, v0.1 Door**
- [ ] `tailtune.bases` on the CPU path: upcast, reduced SVD, orthonormality and reconstruction checks, subspace-distance test, cache keys, and hash into `lineage.json`.
- [ ] The §7 precision rules enforced: `dtype=torch.float32`, the T4 fp16 scaler path, and fp32 compute for the forced families.
- [ ] `method="auto"` using `estimate`, with its one-line explanation.
- [ ] `train` recording peak memory after the first optimizer step.
- [ ] The T4 notebook; the weekly Kaggle T4 job; Modal L4 nightly memory assertions.
- [ ] The CUDA Docker image.
- [ ] Windows native single-GPU `lora` smoke test (by hand); the WSL2 `.wslconfig` guidance in `doctor` and the docs.

**Gate 3, v0.2 Proof**
- [ ] Sharded rows in `estimate` for FSDP2, with `V` replication accounted for.
- [ ] The 2-GPU release gate; the Blackwell release run.
- [ ] The ROCm Docker image; an AMD credits application and the MI300X job.
- [ ] macOS MPS smoke test with torch 2.15.

**Gate 4, v0.3 Reach**
- [ ] `tail-lora` routes in `estimate`.
- [ ] The AMD Instinct promotion review.
- [ ] Linux aarch64 notes; Tier 3 community pages.

**Gate 5, v1.0**
- [ ] A support matrix published per release.
- [ ] ZeRO-3 rows in `estimate` if ZeRO-3 ships.
- [ ] A re-check of every version floor.

## Acceptance tests

1. `estimate` on stored configs reproduces §9.3 and §9.4 to within 0.1 GiB, including Qwen3-8B at 137.7 GiB on the GPU and 47.2 GiB on the host for `sst/host-bases`.
2. `estimate` on a Hub model downloads no weight shards. A network trace shows only the index, config and header range requests.
3. On a T4, `doctor` reports emulated-only bf16, and `train` uses fp16 autocast with fp32 masters and completes more than 5 optimizer steps with nonzero updates.
4. `train --method sst` for Qwen3-8B on a T4 exits with code 4 before loading weights and prints the smallest fitting card.
5. `method="auto"` never selects `sst/8bit-adam`, `sst/offload-state` or `sst/zero-offload` in a parametrised test across all §9 sizes and card sizes.
6. Bases computed twice on the CPU hash identically. A basis computed on another platform passes the subspace-distance test, and orthonormality holds for every stored factor.
7. On MPS with torch 2.14, `doctor` fails D15. With 2.15, SVD runs on the CPU and the MPS path is never called.
8. On Windows with two GPUs, `doctor` warns that only one GPU is supported, and `train --gpus 2` exits with code 4.
9. On WSL2 with default settings, `doctor` warns about host RAM and prints the `.wslconfig` fix.
10. After Gate 1, predicted peak reserved memory is within ±10% of the measured value for every route at 0.3B and 1.7B.
11. Every number `estimate` prints carries the computed label until a matching calibration entry exists.

## Open questions

1. **Exit codes for `doctor`.** The README's codes 0 to 4 are locked and none covers "this environment cannot train". Options are: exit 0 always and leave the verdict to `--json`; reuse 4 under a `--strict` flag; or add a new code through the README. The recommendation is 0, plus `--strict` mapping to 4, pending an owner decision.
2. **Should `train` refuse on unsupported platforms** (Pascal, macOS 14), or only warn? A refusal needs an exit code; see question 1.
3. **Should `W_ref` be stored in bf16 when `K=None`?** It is lossless for bf16 checkpoints and saves about 13 GiB at 8B, but it adds a code path. Alternatively, D-13's cumulative form could rebuild `W_ref`'s frozen part from head-form bases plus `S`. That trades 25.9 GiB of `W_ref` for 15.7 GiB more bases. Both are for [03-methods.md](03-methods.md) to rule on.
4. **Should `--ignore-estimate` exist** while the estimate is uncalibrated?
5. **Which 0.3B model** to calibrate on?
6. **Should Windows on Arm, Intel Macs and Intel Max** be written into D-17 explicitly?
7. **Can Modal's free credit cover the multi-GPU release gate?** This is unconfirmed.
8. **Does the uncertainty band need to be per component** rather than one scalar, once Gate 1 shows where the error sits?
