# 03. Methods

**What this document decides:**

- `sst` is tailtune's own implementation of SST (Chavan et al., 2026), written from the paper's Eq. 3 and Algorithm 1. It applies the cumulative form `W ← W_ref + Q_U (W − W_ref) Q_V` once after each real optimizer step, on fp32 master weights, in `on_optimizer_step`. It ships with `r=0.99, K=None`. `K` (refresh) is exposed but marked unverified (D-10, D-13).
- `osf` wraps PEFT's `OSFConfig`. tailtune adds the guards PEFT lacks. A post-step factor projection is available as a separately named option.
- `lora` (including on a 4-bit base), `full` and `freeze` are the baselines and the free-tier route. `tail-lora` is experimental and arrives at Gate 4.
- Bases are offline fp32 artifacts. They are computed once on CPU, hashed, cached, keyed by parameter name, and checked by subspace distance (D-14).
- Every method sits behind one interface. Every method states its exact variant and rank. Results never transfer between variants (D-11).
- Five Gate 1 measurements, each with a decision rule, settle the embedding policy, overhead claims, memory terms and what `method="auto"` prefers.

Evidence labels are defined in the [plan index](README.md#evidence-labels). Source code was read at these commits: SST `ff0c5bf`, PEFT `f482b81`, mini_trainer `fd5b552`. Nothing in this document has been run on a GPU. All code here is an original, untested design proposal.

## 1. Names and citations

"SST" and "subspace tuning" are overloaded terms:

- [Subspace-Tuning](https://github.com/Chongjie-Si/Subspace-Tuning) is an unrelated PEFT survey framework (Apache-2.0).
- SubT is an unrelated audio-language method ([arXiv 2606.18560](https://arxiv.org/abs/2606.18560)).

So every document writes "SST (Chavan et al., 2026)" in full on first use (D-02).

| Method | Exact citation | Licence of the text | Code |
|---|---|---|---|
| SST | Arnav Chavan, Kusha Sareen, Rishabh Tiwari, Abhranil Chandra, Johan Obando-Ceron, Kurt Keutzer, Scott Niekum, Inderjit S. Dhillon, Devvrit Khatri. "Towards Continual Learning: Mitigating Forgetting With Subspace Tuning." arXiv:2610.22293v1, 15 October 2026 [S](https://arxiv.org/abs/2610.22293) | CC BY 4.0 [S](https://arxiv.org/html/2610.22293) | [Arnav0400/SST](https://github.com/Arnav0400/SST) has no licence. Run it, never copy it (D-06) |
| OSF / OSFT | Nayak, Killamsetty, Han, Bhandwaldar, Chanda, Xu, Wang, Pareja, Silkin, Eyceoz, Srivastava. "Sculpting Subspaces: Constrained Full Fine-Tuning in LLMs for Continual Learning." arXiv:2504.07097v1, 9 April 2025, which calls the method "Adaptive SVD" [S](https://arxiv.org/abs/2504.07097). ICLR 2026, where it is called "Orthogonal Subspace Fine-Tuning (OSFT)" [S](https://iclr.cc/virtual/2026/poster/10006812) | Not checked | PEFT tuner, Apache-2.0 [S](https://github.com/huggingface/peft/blob/f482b819536f475da8d4a304cd7a7e37cae59351/src/peft/tuners/osf/layer.py#L166-L210) |

The SST repo describes itself as the official implementation, pinned to `torch==2.9.0` and `transformers==4.57.1` [S](https://github.com/Arnav0400/SST/blob/ff0c5bf/pyproject.toml). Its own documentation says the public launcher "currently implements the fixed-basis, no-refresh SST variant" [S](https://github.com/Arnav0400/SST/blob/ff0c5bf/sst/methods/README.md).

## 2. The variants, and why results do not transfer

Every method here splits a weight change `ΔW` with the top-k singular subspaces of a reference weight. With `P_U = U_k U_kᵀ`, `Q_U = I − P_U` (and the same for V):

```
ΔW = P_U ΔW P_V  (block 1: inside both protected subspaces)
   + P_U ΔW Q_V  (block 2)
   + Q_U ΔW P_V  (block 3)
   + Q_U ΔW Q_V  (block 4: the free block)
```

SST keeps only block 4 of the applied step [S](https://arxiv.org/html/2610.22293). The variants differ in three ways: which blocks they remove, at what point in the step, and what the optimizer's moments see.

| Variant | Trains | Removes | Where | Adam moments see | Basis timing | Default preserved rank | Licence | In tailtune |
|---|---|---|---|---|---|---|---|---|
| SST (Chavan et al., 2026) | Dense `W` | Blocks 1 to 3 of the applied step | After the optimizer step | Full gradient | Once, with optional refresh every K optimizer steps | r=0.99 | Paper CC BY 4.0, repo unlicensed | `sst`, reimplemented |
| OSF paper Eq. 4 | Dense `W` | Block 1 of the gradient only | Before the step | Projected gradient | Once per task | Adaptive, `mrr=0.1, trr=0.8` (about 80% mean) [S](https://arxiv.org/html/2504.07097v1) | [MIT research code](https://github.com/Red-Hat-AI-Innovation-Team/orthogonal-subspace-learning), frozen since 2025 | Not offered |
| "Sculpting" in SST's tables | Dense `W` gradient hook | Block 1 only | Gradient | Projected gradient | Once | As configured | SST repo, unlicensed [S](https://github.com/Arnav0400/SST/blob/ff0c5bf/sst/methods/sculpting.py#L32-L101) | Not offered |
| PEFT OSF | Factors `U_l, S_l, V_l` | Factor-gradient components along `U_high`, `V_high` (hooks on `U_l`, `V_l` only) | During backward | Projected factor gradient | At `get_peft_model`; re-wrap per task | 50% [S](https://huggingface.co/docs/peft/en/package_reference/osf) | Apache-2.0 | `osf` |
| mini_trainer OSFT | Factors plus non-targets | Factor gradients before the step, factor values after it | Both | Projected factor gradient | Once at init | 75% [S](https://github.com/Red-Hat-AI-Innovation-Team/training_hub/blob/1eb3d9b/docs/api/functions/osft.md) | Apache `LICENSE.md` but MIT metadata [S](https://github.com/Red-Hat-AI-Innovation-Team/mini_trainer/blob/fd5b552/pyproject.toml) | Reference only |
| tailtune `osf` with `post_step=True` | PEFT factors | PEFT hooks plus a post-step factor projection | Both | Projected factor gradient | At wrap | As configured | tailtune, from the equations | Option, gated on Gate 1 measurement 5 |

Results do not transfer between these variants, for five reasons:

1. **Enforcement point.** SST states that Adam's diagonal preconditioning does not commute with the projector, so projecting the gradient differs from projecting the update [S](https://arxiv.org/html/2610.22293). In the paper's toy study (Qwen2.5-0.5B, 60 AdamW steps, r=0.99), unconstrained training put 98% of the change in protected directions, with a probe-NLL rise of +13.0. Gradient-only projection still left 69% there (+0.72). Projecting the applied update left none (+0.13) [S](https://arxiv.org/html/2610.22293). A NumPy toy in the research reproduced the mechanism: 512×512, 400 Adam steps, a linear task, not a language model. It left 70.21% outside the free block under two-sided gradient projection, against 0% under update projection [M]. Red Hat found the same leak independently [S](https://github.com/Red-Hat-AI-Innovation-Team/mini_trainer/issues/90).
2. **Blocks removed.** The block-1-only variants never remove blocks 2 and 3. So SST's "Sculpting" row is not what PEFT or mini_trainer ship. Those two confine change to the same two complements as SST's block 4.
3. **Parameterisation.** Dense `W` trains with full Adam state. Factors train a much smaller set of parameters with different step scaling.
4. **Rank.** Defaults range from 50% to 99%. On Qwen3-8B's target matrices, the free-to-free degrees of freedom are 38.86% of target values at r=0.5, 12.88% at 0.8, 6.00% at 0.9 and 0.56% at 0.99 [C](https://huggingface.co/Qwen/Qwen3-8B/raw/main/config.json).
5. **Evidence.** A fair comparison of OSF and SST on shipped code does not exist. Gate 1 measurement 5 runs one.

**Rule (D-11).** Docs, logs, `lineage.json` and every report name the variant string and the rank. One example: `sst/applied-step/cumulative/fixed-basis r=0.99`.

## 3. SST specification

### 3.1 Notation and k

For a target matrix `W ∈ R^{m×n}` (`m = d_out`, `n = d_in`, the `nn.Linear` weight layout):

| Symbol | Meaning |
|---|---|
| `W` | The fp32 master weight that the optimizer updates |
| `W_ref` | The reference weight: the weights at the start of training, or at the last refresh |
| `s = min(m, n)` | The short dimension |
| `k` | Number of protected singular directions |
| `t = s − k` | Size of the short-side tail |
| `U_k (m×k)`, `V_k (n×k)` | Top-k left and right singular vectors of `W_ref` |
| `T_U`, `T_V` | The last `t` singular vectors on a short side, from the thin SVD |
| `Π(X) = Q_U X Q_V` | The projection onto block 4 |

The paper sets `k = ⌊r·s⌋` [S](https://arxiv.org/html/2610.22293). The public code clamps the result to `[1, s−1]` [S](https://github.com/Arnav0400/SST/blob/ff0c5bf/sst/methods/sst.py#L63-L66), which guarantees at least one free direction. tailtune keeps the same rule, computed exactly:

- Parse `r` as a decimal string into a `fractions.Fraction`, then compute `k = (r.numerator * s) // r.denominator`. Floating-point `floor(0.99 * s)` can land one below the exact value for some `s`.
- Clamp the result to `[1, s−1]`.
- If `s < 2`, refuse the target with `TargetRefused`.
- If `r ≤ 0` or `r > 1`, raise `ConfigError`.
- If `r = 1`, clamp to `s−1` and warn that one free direction remains, and that the user may want to freeze the target instead.

Worked values for Qwen3-8B at r=0.99: `s=4096` gives `k=4055, t=41`. `s=1024` gives `k=1013, t=11`. A width-32 router gives `k=31`, leaving one free direction. D-15 freezes routers by default.

### 3.2 Eq. 3 and Algorithm 1, in our notation

The projection is `Π(ΔW) = Q_U ΔW Q_V`, which keeps block 4 only [S](https://arxiv.org/html/2610.22293). The optimizer and its state are never modified.

```
input: W_0, r, K (None means no refresh), optimizer O with state s_0
W_ref ← W_0;  compute U_k, V_k from W_ref
for t = 0, 1, 2, ...:                      # t counts applied optimizer steps
    g_t ← ∇L(W_t)                          # the full gradient, never projected
    (W̃, s_{t+1}) ← O(W_t, g_t, s_t)        # unmodified optimizer step
    W_{t+1} ← W_t + Π(W̃ − W_t)             # keep block 4 of the applied step
    if K is set and (t+1) mod K == 0:
        W_ref ← W_{t+1};  recompute U_k, V_k from W_ref
```

Parameters outside the target set, such as embeddings and norms, are "left to the optimizer" in the paper [S](https://arxiv.org/html/2610.22293). Section 9 covers tailtune's policy for them.

### 3.3 The cumulative form

tailtune implements `W_{t+1} ← W_ref + Π(W̃ − W_ref)` (D-13). With fixed bases, this equals the per-step form. Assume by induction that `W_t − W_ref = Π(W_t − W_ref)`. Then:

```
Π(W̃ − W_ref) = Π(W̃ − W_t) + Π(W_t − W_ref) = Π(W̃ − W_t) + (W_t − W_ref)
```

The paper states the same equivalence [S](https://arxiv.org/html/2610.22293). The cumulative form is chosen for three reasons:

- **It needs no pre-step snapshot.** Under DeepSpeed no pre-step hook exists, because the real step runs inside `backward` (section 3.5).
- **It is idempotent.** Re-applying it after a step that changed nothing is a no-op.
- **It self-corrects.** Each application re-projects the whole change, so floating-point error in protected directions never accumulates beyond one application.

It costs one fp32 `W_ref` per target: 27.8 GB on Qwen3-8B [C]. **The bf16-exact option.** When `K=None` and the checkpoint was bf16, every `W_ref` value is exactly representable in bf16, because the fp32 values were upcast from bf16. tailtune may then store `W_ref` in bf16, at 13.9 GB [C]. It does so only after checking `W_ref_bf16.float().equal(W_start)` per tensor, and never when `K` is set. This is our design and has not been tested.

### 3.4 Head form, tail form, and which side gets which

On a short side, the thin SVD returns a complete orthonormal basis `[U_k, T]`. So `I − U_k U_kᵀ = T Tᵀ`, the identity PEFT also uses [S](https://github.com/huggingface/peft/blob/f482b819536f475da8d4a304cd7a7e37cae59351/src/peft/tuners/osf/layer.py#L176-L182). On a long side, the complement also contains the null space of `W`. The thin SVD does not return a basis for that, so that side must use the head form `X − U_k(U_kᵀ X)`.

**Rule.** Store the tail on a short side and the head on a long side. A square matrix stores two tails. Apply the tail side first, so the long-side head acts on a thin residual.

| Shape | Left side | Right side | Steps | Free coordinates kept |
|---|---|---|---|---|
| Square (`m = n`) | Tail | Tail | `C = T_Uᵀ D T_V`; `D' = T_U C T_Vᵀ` | `C`, size t×t |
| Wide (`m < n`) | Tail | Head | `Y = T_Uᵀ D`; `Y ← Y − (Y V_k) V_kᵀ`; `D' = T_U Y` | `Y`, size t×n |
| Tall (`m > n`) | Head | Tail | `X = D T_V`; `X ← X − U_k (U_kᵀ X)`; `D' = X T_Vᵀ` | `X`, size m×t |

Qwen3-8B at r=0.99, with all values computed from the [config](https://huggingface.co/Qwen/Qwen3-8B/raw/main/config.json) [C]:

| Matrix | Shape (out×in) | k / t | Head-form multiply-adds | Tail-first multiply-adds | Free block |
|---|---|---|---|---|---|
| `q_proj`, `o_proj` | 4096×4096 | 4055 / 41 | 272 G each | 1.4 G each | 41×41 |
| `k_proj`, `v_proj` | 1024×4096 | 1013 / 11 | 17 G each | 0.18 G each | 11×3083 |
| `gate_proj`, `up_proj` | 12288×4096 | 4055 / 41 | 816 G each | 8.2 G each | 8233×41 |
| `down_proj` | 4096×12288 | 4055 / 41 | 816 G | 8.2 G | 41×8233 |
| All 252 matrices, per step | | | about 109 T | about 1.0 T | 39.0M values |

In NumPy on one CPU, the tail form matched the head form to a relative 2 to 5e-5 in fp32. It also ran about 40x faster [M] (single run, random matrices). The head form is kept only as a test oracle.

The table also bounds capacity. With no refresh, each matrix's total change has rank at most `t`: 41 for most Qwen3-8B targets and 11 for `k_proj` and `v_proj` [C]. A LoRA at rank `t` is therefore the natural capacity-matched control ([09](09-evaluation.md)).

### 3.5 Step order

The Hugging Face Trainer runs `on_pre_optimizer_step`, `optimizer.step()` and `on_optimizer_step` inside the gradient-sync block, after clipping [S](https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/trainer.py#L2704-L2742). Under Accelerate's DeepSpeed integration, `DeepSpeedOptimizerWrapper.step()` is `pass`. The real `engine.step()` runs inside `backward` at accumulation boundaries [S](https://github.com/huggingface/accelerate/blob/main/src/accelerate/utils/deepspeed.py#L264-L312).

| # | Event | tailtune action |
|---|---|---|
| 1 | Forward and backward on each of the G micro-batches, under autocast | None. No gradient hooks |
| 2 | Accumulation boundary. With fp16, the scaler unscales gradients | None |
| 3 | Gradients are clipped to `max_grad_norm` using the full gradient norm | None. The norm includes protected components (section 4) |
| 4 | `on_pre_optimizer_step` | Nothing (D-13). Under DeepSpeed it already fires after the real step |
| 5 | `optimizer.step()`, or `scaler.step()` with fp16, which may skip | None |
| 6 | `on_optimizer_step` | If `accelerator.optimizer_step_was_skipped` is true, count the skip and return. Otherwise, project every target in the cumulative form, let the backend sync any working copy ([08](08-distributed.md)), count the applied step, and refresh if due |
| 7 | The Trainer's own bookkeeping. It reads `optimizer_step_was_skipped` itself [S](https://github.com/huggingface/transformers/blob/main/src/transformers/trainer.py#L1934-L1941) | Report statistics at logging steps |

Step 6 runs once per optimizer step, never per micro-batch. `K` therefore counts optimizer steps, as the paper defines it. tailtune counts only applied (non-skipped) steps. The tailtune callback must be the last callback that modifies weights. A run that adds an EMA or weight-averaging callback after it is refused with `HookPlacementError`.

### 3.6 Skipped AMP steps

A skipped fp16 step does not change `W` or the optimizer state. tailtune skips projection on such steps to save the work and to keep refresh cadence on real steps. The skip flag is not trusted for correctness. If a backend reports it wrongly, the cumulative form still returns the same `W`, because `W − W_ref` already lies in block 4. The cost of a wrong flag is one wasted projection or one avoided projection, never a wrong weight. Under fp16 the scaler skips the first steps, so tests run more than 5 steps [S](https://github.com/unslothai/unsloth/blob/main/.github/workflows/kaggle-t4-notebook-ci.yml).

### 3.7 Callback and projector sketch

```python
# Design proposal. Untested. Original code written from Eq. 3 and Algorithm 1 of
# SST (Chavan et al., 2026), arXiv:2610.22293 (CC BY 4.0). Nothing here is copied
# from the SST repository, which has no licence.
from __future__ import annotations
from dataclasses import dataclass
from typing import Literal
import torch
from transformers import TrainerCallback

@dataclass(frozen=True)
class SideBasis:
    form: Literal["tail", "head"]
    vecs: torch.Tensor            # fp32, orthonormal columns: (dim, t) for tail, (dim, k) for head

@dataclass(frozen=True)
class MatrixBasis:
    name: str                     # checkpoint parameter name, plus "#e" for expert e
    shape: tuple[int, int]        # (d_out, d_in)
    r: str                        # exact decimal string, e.g. "0.99"
    k: int
    left: SideBasis               # acts on d_out
    right: SideBasis              # acts on d_in
    sigma: torch.Tensor           # fp32 singular values sigma_1..sigma_{k+1}

@torch.no_grad()
def project_free(d: torch.Tensor, b: MatrixBasis) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (Q_U d Q_V, free coordinates) for an fp32 matrix d."""
    L, R = b.left, b.right
    if L.form == "tail" and R.form == "tail":           # square
        c = (L.vecs.T @ d) @ R.vecs
        return L.vecs @ c @ R.vecs.T, c
    if L.form == "tail":                                 # wide: short left side
        y = L.vecs.T @ d
        y = y - (y @ R.vecs) @ R.vecs.T
        return L.vecs @ y, y
    if R.form == "tail":                                 # tall: short right side
        x = d @ R.vecs
        x = x - L.vecs @ (L.vecs.T @ x)
        return x @ R.vecs.T, x
    y = d - L.vecs @ (L.vecs.T @ d)                      # head on both sides: test oracle only
    y = y - (y @ R.vecs) @ R.vecs.T
    return y, y

@torch.no_grad()
def apply_cumulative(w, w_ref, b, prev_coords=None):
    d = w - w_ref.float()                                # cumulative change, fp32
    raw_sq = d.square().sum()
    d_proj, coords = project_free(d, b)
    w.copy_(w_ref).add_(d_proj)                          # W <- W_ref + Q_U (W - W_ref) Q_V
    kept_sq = coords.square().sum()                      # equals ||d_proj||_F^2
    stats = {"drift_sq": kept_sq, "removed_sq": (raw_sq - kept_sq).clamp_min(0.0)}
    if prev_coords is not None:
        stats["step_sq"] = (coords - prev_coords).square().sum()
    return stats, coords

class SubspaceProjectionCallback(TrainerCallback):
    def __init__(self, engine, accelerator, refresh_every=None):
        self.engine = engine              # bases, W_ref and a tailtune.backends adapter
        self.accelerator = accelerator    # taken from the wrapped Trainer
        self.refresh_every = refresh_every
        self.applied_steps = 0
        self.skipped_steps = 0
        self.last_stats = None

    def on_optimizer_step(self, args, state, control, **kwargs):
        if self.accelerator.optimizer_step_was_skipped:
            self.skipped_steps += 1
            return
        self.last_stats = self.engine.project_all()      # every target, fixed order, no_grad
        self.applied_steps += 1
        if self.refresh_every and self.applied_steps % self.refresh_every == 0:
            self.engine.refresh()                         # unverified path, section 4.3
```

`engine.project_all()` iterates the backend's matrix views in a fixed name order ([08](08-distributed.md)). The projector never sees a backend.

### 3.8 What every run logs

`Π` is an orthogonal projector in the Frobenius inner product, so `‖A‖² = ‖ΠA‖² + ‖A − ΠA‖²`. The free coordinates have the same norm as the projected change, because the stored bases are orthonormal. These two facts give three exact per-step metrics with no weight snapshot:

| Metric | Formula | Cost |
|---|---|---|
| `sst/drift_norm` | `‖W − W_ref‖_F` over targets, from `kept_sq` | Free |
| `sst/removed_norm` | `‖(I − Π)(W̃ − W_ref)‖`, the protected part the optimizer tried to write this step | One reduction |
| `sst/projected_step_norm` | `‖coords_t − coords_{t−1}‖`, the change actually applied | Keeps the previous coordinates: about 58M fp32 values (231 MB) on Qwen3-8B [C] |

The Trainer's full-gradient norm is logged alongside. The removed norm subtracts nearly equal squares, so treat values near zero as noise. This logging scheme is our design and is untested.

## 4. Hyperparameters and refresh

### 4.1 Defaults

| Setting | `sst` default | `sst_paper` preset | Evidence |
|---|---|---|---|
| `r` | 0.99 | 0.99 | [S](https://arxiv.org/html/2610.22293) |
| `K` | None | None | Refresh is [U]: no public code |
| Learning rate | 5e-5 | 5e-5 | [S](https://github.com/Arnav0400/SST/blob/ff0c5bf/configs/model/qwen3-8b.yaml) for Qwen3. [U] on other models |
| Effective batch | 16 | 16 | [S] |
| Schedule | Cosine, 10 warmup steps, 2 epochs | Same | [S] |
| Weight decay | 0 | 0 | [S] |
| `max_grad_norm` | 1.0 | 1.0 | [S] |
| Max length | 4096 | 4096 | [S] |
| Optimizer | AdamW | AdamW | Muon is in the paper but absent from public code [S](https://github.com/Arnav0400/SST/blob/ff0c5bf/README.md) |
| Embeddings, `lm_head` | Gate 1 measurement 4 decides. Provisionally frozen [J] | Trainable, unconstrained | Section 9 |

The paper used these values for every method and domain. It says they were "not selected separately for each method or domain" [S](https://arxiv.org/html/2610.22293).

### 4.2 Learning rate: guidance, and why not to inherit it

Do not reuse the learning rate from a user's earlier SFT run, from PEFT's OSF advice (1e-5 to 1e-4 [S](https://huggingface.co/docs/peft/en/package_reference/osf)) or from Red Hat's examples (5e-6 to 2e-5 [S](https://github.com/Red-Hat-AI-Innovation-Team/training_hub/blob/1eb3d9b/docs/api/functions/osft.md)). There are four reasons:

- **The authors say so.** The constrained optimum "need not equal that of unconstrained SFT" [S](https://arxiv.org/html/2610.22293).
- **The applied step is far smaller.** Adam scales the step to the full gradient, then `Π` discards most of it. In the research's toy at r=0.99, the projected step's per-entry RMS was about 1% of the learning rate (5.4e-7 against 5e-5) [M] (NumPy, not a language model). The effective learning rate differs from SFT's.
- **Optimizers differ by orders of magnitude.** The authors' follow-up sweeps Muon two orders of magnitude above AdamW [S](https://arnav0400.github.io/subspace-tuning-emergent-misalignment/).
- **The paper's own sweep is mixed.** SST was more robust across learning rates, but SFT reached the highest in-domain score at the lowest learning rate [S](https://arxiv.org/html/2610.22293).

Keep 5e-5 as the starting point, labelled as validated only for the Qwen3 recipe. The benchmark's per-method tuned arm ([09](09-evaluation.md)) is where tailtune learns better defaults.

### 4.3 Weight decay, clipping and refresh

**Weight decay.** AdamW's decoupled decay is part of the applied step, so `Π` drops its protected component. On targets, only the free block decays. Unconstrained parameters decay in full. The default stays 0.

**Clipping.** The clip norm is computed on the full gradient, protected components included. A large protected gradient can therefore shrink a small useful free step. This is unstudied. Keep 1.0 as the paper did, and always log `sst/projected_step_norm` and `sst/removed_norm` (section 3.8).

**Refresh semantics.** When `K` is set, after every `K` applied steps, `W_ref ← W` and the bases are recomputed from the new `W_ref`. The paper uses an eigendecomposition of the smaller Gram matrix [S](https://arxiv.org/html/2610.22293). Refresh trades retention for learning. On Qwen3-8B, R10 lifts mean in-domain to 79.5 against 75.7 with no refresh, and lowers mean prior capability to 74.9 against 79.6 [S](https://arxiv.org/html/2610.22293).

**Refresh status: unverified.** No public code reproduces the R10 or R100 rows. A Gram route squares the condition number, so fp32 `eigh` is likely unreliable in exactly the tail directions that matter [J]. Gate 4 adds refresh under the unverified label. It requires fp32 `W_ref`, never the bf16-exact option, and saves `W_ref` and bases in checkpoints ([08](08-distributed.md)). D-14 keeps initial bases on the CPU path, where a full pass on an 8B model takes 0.5 to 3.5 hours [C], but allows refresh to use GPU SVD on CUDA or ROCm with the same fp32 input and orthonormality checks; [08](08-distributed.md) specifies how, and Gate 4 measures whether K=10 or K=100 is then practical on 8B models.

**A cheaper exact refresh. This is a candidate for Gate 4, untested.** It rests on two preconditions:

- The bases are exact singular vectors of `W_ref`, so the cross blocks of `W_ref` vanish.
- `D = W − W_ref` lies in block 4.

Under those preconditions, `W = U_k S_k V_kᵀ + (free part)` is an orthogonal direct sum, and the free part has a compact form:

| Shape | Free part |
|---|---|
| Square | `T_U (S_t + C) T_Vᵀ`, with `C` from section 3.4 |
| Tall | `M T_Vᵀ`, with `M = W T_V` (m×t) |
| Wide | `T_U N`, with `N = T_Uᵀ W` (t×n) |

The new top-k subspace is then the top k of `σ(S_k) ∪ σ(free core)`.

- If no free singular value exceeds `σ_k`, the bases do not change and only `W_ref` moves. Moving `W_ref` alone leaves the projection unchanged.
- If some do, the swapped directions come from an SVD of a t×t core, or from a thin QR plus a t×t SVD.
- On a short side, the head vectors that leave the protected set must be rebuilt as `v = W_refᵀ u / σ`. The numerical stability of that step is unknown.

Never use this route to support claims about the paper's R10 or R100 results. Benchmark it against a fresh CPU SVD by subspace distance, re-anchor with a full SVD at intervals, and keep the paper's Gram route as the reference.

## 5. Precision

### 5.1 The trap and its evidence

- **Format arithmetic.** bf16 has an 8-bit significand, so an update smaller than about `2^-9·|w|` rounds away [C]. Adam's per-step moves are close to the learning rate, so at a learning rate of 1e-5, a pure-bf16 weight above about 5e-3 in magnitude stops moving [C]. Projected steps are about 100x smaller still (section 4.2).
- **Toy result.** NumPy, 1024×1024, 100 Adam-like steps at a learning rate of 5e-5, not a language model [M]:
  - With fp32 masters, `ΔW` matched the ideal projected trajectory (cosine 1.000).
  - With in-place bf16, `‖ΔW‖` was 0.0001 to 0.0002 against an ideal of 0.0057 to 0.0060, with cosine 0.007 to 0.023.
  - Of the change that remained, 93% to 99% sat outside the free block.
- **Loader default.** Transformers v5 defaults `from_pretrained` to `dtype="auto"` [S](https://github.com/huggingface/transformers/releases/tag/v5.0.0). A bf16 checkpoint therefore silently becomes pure-bf16 trainable weights.
- **Public SST.** It avoids the trap only because DeepSpeed holds fp32 masters. Its fallback, `parameter.data.float()`, takes the rounding path with no warning [S](https://github.com/Arnav0400/SST/blob/ff0c5bf/sst/methods/sst.py#L103).

### 5.2 Design: fp32 masters

D-12 says "refuse low-precision trainable targets". This document interprets it precisely: **the tensor the optimizer updates for a target must be fp32.** The literal reading would wrongly block the ZeRO paths, whose module parameters are bf16 while their masters are fp32. The README's D-12 now uses this wording. The same rule covers the `full` and `freeze` baselines, so comparisons are fair.

| Backend | Optimizer-updated tensor | Compute | Check |
|---|---|---|---|
| `single`, `ddp` | fp32 parameter. tailtune loads with `dtype=torch.float32` explicitly | bf16 autocast, or fp16 autocast with `GradScaler` on T4 | Parameter dtype |
| `fsdp2` | fp32 sharded storage | `MixedPrecisionPolicy(param_dtype=bf16, reduce_dtype=fp32)`, as mini_trainer does [S](https://github.com/Red-Hat-AI-Innovation-Team/mini_trainer/blob/fd5b552/src/mini_trainer/setup_model_for_training.py#L438-L440) | Local shard dtype |
| `zero2`, `zero3` | DeepSpeed fp32 master. Module parameters may be bf16 | bf16 | `safe_get_full_fp32_param` returns fp32 |

**Refused, with `PrecisionRefused`:**

- a target whose optimizer-updated tensor is not fp32
- bases in any dtype other than fp32
- a bf16 `W_ref`, unless the exactness check passes and `K=None`
- quantized or integer targets (D-15)
- `tt.wrap` on a model already loaded in bf16, with no fp32 masters behind it

`tt.wrap` refuses rather than upcasting in place, because the optimizer may already reference the bf16 tensors. The error message shows the exact `from_pretrained(..., dtype=torch.float32)` call to use.

The CLI exits with code 4 for this refusal, which the [plan index](README.md#shared-vocabulary) defines to include low-precision and quantized trainable targets.

**bf16 and T4 rules:**

- Detect bf16 with `torch.cuda.is_bf16_supported(including_emulation=False)`. The default call returns True on a T4 through emulation [S](https://github.com/pytorch/pytorch/blob/v2.15.0/torch/cuda/__init__.py#L244-L263).
- On a T4, pure fp16 weights cannot use the scaler, which raises on fp16 gradients [S](https://github.com/pytorch/pytorch/blob/v2.15.0/torch/amp/grad_scaler.py#L275). fp32 masters with fp16 autocast are the only stable route. Some recent families need float32 on a T4 because of fp16 NaNs [S](https://github.com/unslothai/unsloth/blob/main/unsloth/models/loader.py#L213-L223).
- tailtune does not offer fp16 autocast on bf16-capable hardware for constrained methods, so that each hardware class has one tested route [J].
- `full` and `freeze` use the same fp32-master rule, so they remain fair baselines [J]. D-12 itself names only constrained methods.

**Export caveat.** A merged bf16 export rounds each weight to half an ulp. That rounding can erase small tail changes and adds noise in protected directions. The retention report therefore evaluates the exported artifact, not the fp32 weights ([09](09-evaluation.md), [10](10-export-and-deployment.md)). Gate 2 measures the gap between them.

## 6. Bases

### 6.1 Computation

- **Routine.** Bases are computed offline from the checkpoint files, before any training wrapper exists (D-14). The canonical path is `torch.linalg.svd(W.float(), full_matrices=False)` on CPU.
- **Input dtype.** The routine accepts only float32 and float64 (and complex types) [S](https://docs.pytorch.org/docs/2.15/generated/torch.linalg.svd.html), so bf16 weights are upcast first.
- **Drivers.** The `driver` argument works only on CUDA. A future CUDA path would use `gesvd`, which the docs recommend "when precision matters". It would never use `gesvdj` on large matrices, which the source warns "may be numerically unstable" [S](https://github.com/pytorch/pytorch/blob/v2.15.0/aten/src/ATen/native/cuda/linalg/BatchLinearAlgebraLib.cpp#L657-L680).
- **Full SVD only at high r.** Randomized and truncated routines target leading components, which defeats their purpose when 99% of the spectrum is wanted. OPLoRA attributes nonzero leakage to its low-rank SVD [S](https://arxiv.org/html/2510.13003v2).
- **Compute once, then distribute.** Singular vectors "are not unique", and different hardware "may compute different singular vectors" [S](https://docs.pytorch.org/docs/2.15/generated/torch.linalg.svd.html). So bases are computed once and distributed, never recomputed per rank.

### 6.2 Checks run on every new basis set, failing with `BasesInvalid`

Every tolerance below is [J], to be calibrated at Gate 1 measurement 2.

1. **Ordering.** Singular values are non-negative and descending.
2. **Orthonormality.** For each stored side, `max|BᵀB − I| ≤ 1e-4`.
3. **Reconstruction.** `‖W − U S Vᵀ‖_F / ‖W‖_F ≤ 1e-5`, computed before the head is dropped on short sides.
4. **Tail energy.** For each stored tail, `|‖Tᵀ W‖_F² − Σ_{i>k} σ_i²| / ‖W‖_F² ≤ 1e-5`. This proves the stored tail really is the bottom of the spectrum.
5. **Gap record.** Record the relative gap `g = (σ_k − σ_{k+1}) / σ_1`. If `g < 1e-6`, mark the matrix `boundary_degenerate`. Its protected subspace is then not well defined across hardware, and cross-machine comparison tests skip it.

### 6.3 Subspace-distance testing

Never compare vectors for equality. Compare projectors: `d(A, B) = ‖P_A − P_B‖₂ = sin θ_max`. Two complementary subspaces of equal dimension share their nonzero principal angles, so compute the distance on the smaller side:

- For tails: `d = sqrt(1 − σ_min(T_Aᵀ T_B)²)`.
- For long-side heads: `d = sqrt(1 − σ_min(U_Aᵀ U_B)²)`, which is affordable offline.

The test tolerance is `max(1e-4, 100·ε·σ_1/(σ_k − σ_{k+1}))` [J], following the perturbation bound that scales with the inverse gap.

### 6.4 Size and time

All figures are for Qwen3-8B at r=0.99:

| Item | Entries | fp32 size |
|---|---|---|
| Head form, both sides | 9.94B | 39.8 GB [C] |
| Tail on short sides, head on long sides (shipped form) | 5.72B | 22.9 GB [C] |
| `W_ref` | 6.946B | 27.8 GB, or 13.9 GB bf16-exact [C] |
| bf16 checkpoint, for scale | | 16.4 GB [C] |
| Public SST: bases plus `W_ref` held per process | | 67.5 GB per rank, about 540 GB for 8 ranks [C] |

A full CPU SVD pass on an 8B model takes between about 0.5 and 3.5 hours [C]. Two local estimates in the research disagree by 3 to 6x. A 4096×4096 SVD took 13 s in one and 26 to 32 s in the other, on different thread counts. Gate 1 measurement 2 settles this.

### 6.5 Hash, cache key and keying

- **Lookup key.** The key is the tuple `(format_version, model_id, model_revision, param_name, expert_index, source_weight_sha256, r)`. Here `k` is derived from `r` and `s`. `source_weight_sha256` hashes the raw checkpoint tensor bytes in their stored dtype.
- **Provenance.** Implementation details are recorded in a manifest, not in the key: torch version, LAPACK backend, thread count, routine, `full_matrices=False`, side forms, singular values, gaps and check results. Valid bases are interchangeable for lookup, while the content hash identifies the exact set.
- **Bases hash.** SHA-256 over the canonical manifest JSON (sorted keys) plus every tensor's bytes, in sorted name order. It goes into `lineage.json`.
- **Storage.** safetensors only (D-08), one file per decoder layer plus `manifest.json`, under a per-model, per-revision, per-r cache directory. Sharing on the Hub waits on licence review (D-14), because bases derive from model weights.
- **Keying.** Bases are keyed by parameter name. They are never keyed by object identity, because parameter objects captured before FSDP2 or DeepSpeed wrapping go stale ([08](08-distributed.md)). [06](06-models.md) records both the in-memory and the checkpoint name. An `[E, m, n]` tensor is keyed as `name#e` per expert.
- **On load.** Names and shapes must match the resolved targets, and `source_weight_sha256` must match the loaded checkpoint. On a mismatch, the bases are stale: recompute them, or refuse in offline mode.

## 7. OSF via PEFT

### 7.1 Wrapping

```python
# Design proposal. Untested.
from peft import OSFConfig, get_peft_model

def wrap_osf(model, targets, r: float):
    refuse_unsupported_layers(model, targets)          # Conv1D and quantized layers, see 7.2
    cfg = OSFConfig(target_modules=sorted(targets.module_names), effective_rank=r)
    peft_model = get_peft_model(model, cfg)
    assert_wrapped_exactly(peft_model, targets)        # no silent skips, no extra matches
    assert_trainable_count(peft_model, targets.osf_factor_count(r))
    record_preserved_ranks(peft_model, targets)        # actual k per module, into targets.json
    return peft_model
```

`effective_rank` is the preserved rank. `None` means 50% of the smaller dimension, and a float in (0, 1] is a fraction [S](https://huggingface.co/docs/peft/en/package_reference/osf). tailtune passes `r` as that fraction and reads the resulting `k` back from each wrapped layer, because the rounding rule was not checked [U]. tailtune always passes the resolver's full module names. PEFT's default mapping falls back to all-linear for `qwen3_5`, `gemma4`, `gpt_oss` and `granite` [S](https://github.com/huggingface/peft/blob/main/src/peft/utils/constants.py).

### 7.2 Known defects and the guard for each

| Defect | Evidence | Guard |
|---|---|---|
| Conv1D targets are skipped silently in release v0.21.2, leaving zero trainable parameters. `main` raises `TypeError` | [test_osf.py](https://github.com/huggingface/peft/blob/f482b819536f475da8d4a304cd7a7e37cae59351/tests/test_osf.py), [v0.21.2 model.py](https://github.com/huggingface/peft/blob/v0.21.2/src/peft/tuners/osf/model.py) | The resolver rejects Conv1D (D-15). `assert_wrapped_exactly` and the trainable count catch any skip |
| bitsandbytes `Linear4bit` and `Linear8bitLt` subclass `nn.Linear` [S](https://github.com/bitsandbytes-foundation/bitsandbytes/blob/a616886cacf86701514f0cb1d6be760f967481c0/bitsandbytes/nn/modules.py#L504) and pass PEFT's type check [S](https://github.com/huggingface/peft/blob/f482b819536f475da8d4a304cd7a7e37cae59351/src/peft/tuners/osf/model.py#L75-L79). PEFT then runs SVD on the packed data and casts back to its dtype [S](https://github.com/huggingface/peft/blob/f482b819536f475da8d4a304cd7a7e37cae59351/src/peft/tuners/osf/layer.py#L110-L113). By reading only, this should fail with a generic integer-gradient error | Source reading | Before wrapping, refuse any target whose layer class comes from bitsandbytes, or whose weight is not floating fp32, with `TargetRefused: quantized` |
| The docs call OSF "memory-efficient", but the source keeps the base layer, factors, initial copies and, for non-square matrices, the high bases | [layer.py](https://github.com/huggingface/peft/blob/main/src/peft/tuners/osf/layer.py) | `estimate` uses the layout as coded ([07](07-hardware.md)) |
| A DTensor-aware projection helper exists, but the live hook uses plain matrix products | [utils.py](https://github.com/huggingface/peft/blob/f482b819536f475da8d4a304cd7a7e37cae59351/src/peft/tuners/osf/utils.py#L61-L110) | `osf` supports `single` and `ddp` only until tested under FSDP2 |
| Gradient hooks plus Adam can still leak into protected directions. This is the same mechanism as section 2, in factor space | [issue #90](https://github.com/Red-Hat-AI-Innovation-Team/mini_trainer/issues/90) | Offer the post-step option (7.4). Gate 1 measurement 5 measures the leak |
| Weight decay shrinks `U_l` and `V_l` away from orthonormal, changing `δ` even with zero gradient | Reasoned from the factor parameterization [J] | Factor weight decay defaults to 0 |
| Backward hooks change timing under `torch.compile` | [compile docs](https://github.com/pytorch/pytorch/blob/v2.15.0/docs/source/user_guide/torch_compiler/torch.compiler_nn_module.md) | `osf` refuses `torch.compile` until a test passes [J] |
| The docs' basic example uses a learning rate of 3e-4, outside their own advice of 1e-5 to 1e-4 | [docs](https://huggingface.co/docs/peft/en/package_reference/osf) | Default 5e-5 [J] |

Pin a tested PEFT range (D-07). Export merging for OSF is [U]. If PEFT cannot merge, compute `W + U_l S_l V_l − U⁰ S⁰ V⁰` directly and confirm with a logit check.

### 7.3 Defaults and rank non-comparability

`osf` defaults to PEFT's `r=0.5` until Gate 1 measurement 5 decides otherwise [J]. At r=0.5, free-to-free degrees of freedom are 38.86% of target values on Qwen3-8B, against 0.56% for `sst` at 0.99 [C]. An `osf` run and an `sst` run at their defaults are therefore different experiments. Reports print the variant and rank beside every number. On PEFT's own single-task benchmark, OSF at rank 128 cut forgetting to 0.0166 against LoRA r32's 0.0500. Accuracy fell from 47.8% to 42.9%, and training time rose from 942 s to 1,539 s [S](https://github.com/huggingface/peft/blob/f482b819536f475da8d4a304cd7a7e37cae59351/method_comparison/MetaMathQA/results/osf--llama-3.2-3B-rank128.json).

### 7.4 Optional post-step projection

With `osf(post_step=True)`, after each real optimizer step and in the same `on_optimizer_step` slot as SST, tailtune applies two updates:

```
U_l ← U_l − U_high (U_highᵀ U_l)
V_l ← V_l − (V_l V_highᵀ) V_high
```

Here `V_l` holds row vectors. On a square factor, `I − U_high U_highᵀ` equals `U_l⁰ U_l⁰ᵀ`, the initial low factor, which PEFT stores. These equations follow mini_trainer's post-step projection [S](https://github.com/Red-Hat-AI-Innovation-Team/mini_trainer/blob/fd5b552/src/mini_trainer/osft_utils.py#L541-L647), merged in [PR #91](https://github.com/Red-Hat-AI-Innovation-Team/mini_trainer/pull/91). tailtune implements them from the math and copies no mini_trainer code. Docs credit mini_trainer and the SST argument.

**Licence handling.** No code is ported, so no notice is legally required. If code is ever ported, keep both the Apache-2.0 `LICENSE.md` notice and the MIT metadata notice until upstream resolves the conflict.

The option reads PEFT's internal buffers, so a version-pinned test asserts that their names exist. Its variant string is `osf/peft-hooks+post-step`. Its moments still see projected factor gradients, which differs from SST's full-gradient moments.

## 8. `lora`, `full`, `freeze`, `tail-lora`

| Method | Definition | Defaults | Status |
|---|---|---|---|
| `lora` | PEFT LoRA on the roles from [06](06-models.md). On a 4-bit base: NF4 through `quantization_config`, since v5 dropped the old loader arguments [S](https://github.com/huggingface/transformers/blob/main/MIGRATION_GUIDE_V5.md). Adapters stay fp32, PEFT's default [S](https://huggingface.co/docs/peft/main/en/package_reference/peft_model). The base is frozen, so D-12 does not apply to it | Rank 16, alpha 32, dropout 0 [J]. Per-role ranks allowed, so a capacity-matched control can use rank `t` (41, and 11 on `k_proj` and `v_proj` for Qwen3-8B [C]) | Gate 2. The free-tier route (D-18): about 8.3 GiB at 8B [C] |
| `full` | Standard full fine-tuning of every parameter except the classes D-15 freezes by default. fp32 masters, bf16 autocast | Same recipe as `sst` | Gate 2 baseline. 8B needs about 133.5 GiB [C] |
| `freeze` | Partial-layer full tuning: the last N decoder blocks train with fp32 masters, and everything else stays frozen, including embeddings | N = 25% of blocks, rounded up [J] | Gate 2 baseline. The definition is ours and an open question |
| `tail-lora` | An adapter confined to the tail on a quantized base | Section 8.1 | Experimental, Gate 4 |

LoRA has no protection mechanism. Low-rank protection holds "only insofar as LoRA learns less" [S](https://arxiv.org/abs/2609.28395). On PEFT's benchmark, LoRA r32 forgot more than full fine-tuning: 0.0500 against 0.0300 [S](https://github.com/huggingface/peft/blob/f482b819536f475da8d4a304cd7a7e37cae59351/method_comparison/MetaMathQA/results/lora--llama-3.2-3B-rank32.json). The report is what makes `lora` honest.

### 8.1 `tail-lora`: feasible, unproven, labelled experimental

**The maths works.** The model computes `W_eff = Q(W) + ΔW`. Here `Q(W)` is the quantized base, which cannot be updated directly, and `ΔW` is the adapter. Three candidate forms:

| Form | Feasible set | Bases needed | Known evidence |
|---|---|---|---|
| Core `U_t C V_tᵀ` | Exactly SST's block 4 on square matrices. Drops the null space on rectangular ones | Tails only, about 0.2 GB on 8B, but only about 311k trainable values [C] | None |
| OPLoRA `Q_U B A Q_V` | Block 4, as an adapter | Tail plus long-side head, about 11.4 GB bf16 on 8B, 2.8x the 4.1 GB 4-bit model [C] | On a bf16 base at `k=0.99d`, prior capability averaged 73.05 against SST's 79.6 [C](https://arxiv.org/html/2610.22293) |
| MiCA (`init_lora_weights="mica"`) | One-sided, so block 3 stays open | Small | In PEFT [S](https://github.com/huggingface/peft/blob/main/src/peft/tuners/lora/config.py). Unsloth marks it 16-bit only [S](https://github.com/unslothai/unsloth/pull/6879) |

Bases come from the full-precision checkpoint, offline, before quantization (D-14). This is cleaner than dequantizing.

**What is unknown:**

- whether NF4 error swamps the tail of the spectrum, which nobody has measured
- how much Adam in factor space costs. SST says OPLoRA's scheme "mis-scales the step" [S](https://arxiv.org/html/2610.22293)
- retention on any quantized base

The protection holds relative to `Q(W)`, which already departs from `W`. Every `tail-lora` result is labelled "experimental". It carries "runs" until the Gate 4 benchmark, and "retention benchmarked" only after it.

## 9. Target scope

Public SST constrains 2D parameters whose names end in `.{q,k,v,o,gate,up,down}_proj.weight` [S](https://github.com/Arnav0400/SST/blob/ff0c5bf/sst/methods/sst.py#L21-L44). On Qwen3-8B, that leaves 15.2% of parameters training unconstrained [C]. The embedding and head matrices hold about 1.245B values, against only 39.0M free values inside the constrained matrices [C]. No ablation tests how much in-domain learning flows through them. Describe what is free as "the complement of the top-k singular subspaces", not as "the low-energy tail". On `gate_proj` at r=0.99, 8,192 of the 8,233 free output directions lie in the left null space [C].

**Embedding policy.** Gate 1 measurement 4 decides it (D-15). Until then, `sst` and `osf` provisionally freeze embeddings, `lm_head` and norms [J]. This closes a forgetting route outside the claim and matches PEFT, which freezes everything except its factors [S](https://github.com/huggingface/peft/blob/f482b819536f475da8d4a304cd7a7e37cae59351/src/peft/tuners/osf/model.py#L116-L120). With tied embeddings, freezing one freezes both.

**`sst_paper` preset.** This preset is for exact replication. It goes through [06](06-models.md)'s role resolver like everything else, then asserts that the resolved set equals the paper's suffix rule over 2D parameters. If the two differ, as with fused QKV or extra suffix matches, it refuses with exit code 4 and prints the difference. Embeddings, `lm_head` and norms stay trainable and unconstrained. The hyperparameters are those in section 4.1.

Role resolution, `[E, m, n]` experts, routers, fused projections and coverage floors belong to [06](06-models.md). This document only consumes the resolved `TargetSet`. Every report prints the protected, free and frozen shares from `tailtune inspect`.

## 10. The method interface

```python
# Design proposal. Untested.
from dataclasses import dataclass
from typing import ClassVar, Literal, Protocol, Sequence
import torch.nn as nn
from transformers import TrainerCallback

@dataclass(frozen=True)
class VariantSpec:
    method: str            # "sst", "osf", "lora", "full", "freeze", "tail-lora"
    variant: str           # e.g. "sst/applied-step/cumulative/fixed-basis"
    citation: str          # e.g. "SST (Chavan et al., 2026), arXiv:2610.22293"
    preserved: str | None  # "r=0.99", or None for an unconstrained method
    implementation: str    # e.g. "tailtune 0.1.0" or "peft 0.21.x OSFConfig"
    status: Literal["stable", "experimental", "unverified"]

class Method(Protocol):
    name: ClassVar[str]
    backends: ClassVar[frozenset[str]]       # subset of {"single", "ddp", "fsdp2", "zero2", "zero3"}

    def variant(self) -> VariantSpec: ...
    def check(self, plan: "TrainingPlan") -> Sequence["Finding"]: ...          # before weights load
    def memory_terms(self, shapes: "ModelShapes", plan: "TrainingPlan") -> "MemoryTerms": ...
    def prepare(self, model: nn.Module, targets: "TargetSet", ctx: "MethodContext") -> nn.Module: ...
    def callbacks(self, ctx: "MethodContext") -> Sequence[TrainerCallback]: ...
    def state_dict(self) -> dict: ...                                          # saved with checkpoints
    def load_state_dict(self, state: dict) -> None: ...
    def merged(self, model: nn.Module) -> nn.Module: ...                       # plain weights for export
```

**Contract:**

- `check` runs from config and shapes alone, so `estimate` and `train` refuse before weights load.
- `memory_terms` feeds [07](07-hardware.md) with the terms the method adds: `W_ref`, bases, factors, adapters and the free-coordinate cache.
- `prepare` freezes and wraps, and must leave the trainable set equal to what `check` predicted.
- Projection maths lives in `tailtune.methods` and receives matrix views from `tailtune.backends` ([08](08-distributed.md)).

**Adding a method:**

1. Write a `VariantSpec` with a citation, and confirm the licence (D-06).
2. Implement the Protocol and add memory terms.
3. Add a CPU numerics test and the frozen-subspace invariant test, if the method is constrained.
4. Declare the supported backends.
5. Add a row to the variants table.
6. Ship it labelled "runs". It becomes "retention benchmarked" only after a benchmark arm.

The registry stays in-tree before v1.0 [J], so claims stay under review.

## 11. Sequential tasks and lineage

`lineage.json` records the base commit, bases hash, r, coverage, data hash, template hash, dependency versions and the previous run. This document asks the owners of [02](02-architecture.md) and [04](04-api-and-ux.md) to add the variant string, `K`, and the applied and skipped step counts. For a chained run, `W_ref` is the previous run's output and the bases are recomputed from it, as PEFT advises when re-wrapping per task [S](https://huggingface.co/docs/peft/en/package_reference/osf). Retention is reported against both the original base and the previous run.

**Capacity budget.** Treat free capacity as a budget that shrinks across tasks. Each report prints free-to-free capacity: 0.56% of target values at r=0.99 on Qwen3-8B [C].

**What is unknown:**

- **Whether earlier tasks survive.** Each run's change has rank at most `t`, so the next run's top-k subspace barely moves. The next task competes for the same tail and may overwrite the previous task's learning. How much it overwrites is unmeasured.
- **Which budget rule works.** OSF's progressive rule freezes the top `(i−1)/n` for task i of n [S](https://arxiv.org/html/2504.07097v1). That suits low preserved ranks. At r=0.99 the tail is 41 directions per square matrix, so protecting each run's change outright could exhaust it after one task [J].
- **Long chains.** SST's sequential experiments cover only two- and three-stage chains [S](https://arxiv.org/html/2610.22293). Long chains remain a question to measure, not a promise.

## 12. Adjacent methods, ranked

| Rank | Method | Mechanism | Status of evidence | Cost and code | tailtune plan |
|---|---|---|---|---|---|
| 1 | [DG-Hard](https://arxiv.org/abs/2605.20296) | Donoho-Gavish hard threshold on `W_ft − W_base` | Abstract only, unreplicated [U] | No data and no training. Code status unverified | `tailtune repair`, Gate 4, only if verification passes |
| 2 | [RPSFT](https://arxiv.org/abs/2605.10973) | Loss term `λ‖U₀ᵀ(W−W₀)V₀‖²_F`, k=768, λ=1 | One group. +6.0% GPU-hours, peak 52.7 against 32.4 GB [S] | [Repo](https://github.com/jinhangzhan/RPSFT) has no licence, so reimplement. Avoids the rounding trap | After Gate 3 [J] |
| 3 | [OPLoRA](https://arxiv.org/abs/2510.13003), [MiCA](https://arxiv.org/abs/2604.01694) | Tail-confined adapters | Section 8.1 | MiCA is in PEFT | `tail-lora`, Gate 4 |
| 4 | [SDFT](https://arxiv.org/abs/2601.19897) composed with SST | Self-distilled targets | SDFT+SST kept prior capability but lowered in-domain on all four domains [S](https://arxiv.org/html/2610.22293) | Largest footprint | Comes after SFT and continued pretraining (D-16). The roadmap does not yet name a gate |
| 5 | [EWC](https://arxiv.org/abs/2505.05946) | Fisher-weighted anchor | General 65.1 against SFT's 55.8 on 8B, but instruction following fell from 71.1 to 35.3 [S](https://arxiv.org/abs/2609.28395) | Needs auxiliary data | Not planned |
| 6 to 8 | [MoFO](https://arxiv.org/abs/2407.20999), SPD, [LaLoRA](https://arxiv.org/abs/2512.17720) | Optimizer-level or regularizer | Abstracts only [U] | Licences unverified or absent | Watch |
| 9 to 12 | [Safe LoRA](https://arxiv.org/abs/2405.16833), [Lisa](https://arxiv.org/abs/2405.18641), [LwF](https://arxiv.org/abs/1606.09282), [SPG](https://arxiv.org/abs/2306.14775) | Safety or distillation | Off-mission, or weak evidence at LLM scale | | Not planned |

## 13. Inconsistencies in the SST paper, and questions for the authors

From reading v1 [S](https://arxiv.org/html/2610.22293):

1. Tables 3 and 4 disagree on the same setting: Qwen3-8B Science SFT+SST reaches 78.9 in one and 80.0 in the other, with identical in-domain 80.1.
2. Table 4's no-refresh and R100 rows have identical Science prior-capability values.
3. Table 6's base (MMLU-Pro 59.7, mean 78.0) differs from the other tables' base (73.0, 81.0).
4. "Higher new-task accuracy at every size" fails at 1.7B without refresh (67.1 against 69.5).
5. Appendix B.11 says in-domain accuracy declines at r=0.99, while section 3 says stronger protection improves both.
6. The text says ZeRO-3, but the repo maps SST to ZeRO-2 [S](https://github.com/Arnav0400/SST/blob/ff0c5bf/sst/config.py#L45-L51).
7. Section 4.1 calls Sculpting a factored method, but the repo's baseline hooks the dense gradient.
8. **Internal only, never to be quoted (D-02).** The overhead estimates do not reconcile:
   - The stated projection time implies about 4.2 peta multiply-adds per second under the head form the public code uses.
   - The implied 25 ms per refreshed matrix sits far from OPLoRA's 5.5-minute SVD initialisation [S](https://arxiv.org/html/2510.13003v2).

**Questions to send at Gate 0:**

- Can the repo carry a licence?
- Can the ZeRO-3, Muon (code and learning rates) and refresh code be shared?
- How does refresh handle the Gram route's precision, near-equal singular values at k, and resume?
- How was projection time measured, and was the tail form used?
- Are seeds or confidence intervals available?
- Were embeddings and `lm_head` deliberately left unconstrained, and was this ablated?

## 14. Gate 1 measurements, as they concern methods

The roadmap defines all five ([13](13-roadmap.md)). Every threshold below is [J], set before the runs and published with the results.

**1. Reproduce one Table 5 cell.**
- **Protocol.** Run the public repo unmodified at `ff0c5bf`, with its pinned versions and ZeRO-2 config (D-19). Run it but copy nothing. The cell is SFT+SST with no refresh, on Qwen3-8B in thinking mode, Science domain. Science appears in Tables 2 to 5, which gives the most cross-checks; if its data is far larger than FinQA's, use FinQA instead. Run the SFT arm too if the budget allows.
- **Hardware.** The public code holds bases and `W_ref` per process, so an 8-rank node needs about 540 GB of host RAM [C]. Choose the node accordingly.
- **Record.** Step time, peak GPU memory and host RAM.
- **Decision.** "Reproduced" means mean prior capability within 3 points of the paper's value and in-domain within 3 points. Otherwise publish the disagreement, keep `sst` out of `auto`'s first choice, and apply [01](01-product-strategy.md)'s outcome table.

**2. Real projection and refresh overhead.**
- **Protocol.** Use a single-GPU tailtune prototype written from this document. Run the CPU SVD on all 252 Qwen3-8B matrices from real weights; this also builds the bases cache for measurements 4 and 5. Time the following with CUDA events, 5 warm-up runs and the median of 20:
  - head and tail forms per optimizer step, with bases on the GPU and streamed from pinned host memory
  - a plain SFT step on the same GPU
  - the Gram `eigh` route in fp32 and fp64, with its subspace distance from the direct SVD
- **Decisions:**
  - Publish tailtune's numbers and never the paper's.
  - If streamed tail-form projection adds 10% or less to the SFT step, bases on host become the single-GPU default.
  - If fp32 Gram exceeds the subspace tolerance, ban the fp32 Gram route.
  - Refresh at K is offered in `auto` only if its amortised cost stays under 50% of step time.

**3. Peak memory.**
- **Protocol.** On a 0.3B and a 1.7B model, measure each method's own terms against `memory_terms`: `W_ref`, bases, the coordinate cache, PEFT OSF as coded, and LoRA on a 4-bit base.
- **Decision.** Any term off by more than 10% is corrected in [07](07-hardware.md)'s formula.

**4. Embedding-freeze ablation.**
- **Protocol.** Use the same prototype, validated against measurement 1 on identical settings. There are two `sst` arms at r=0.99 and `K=None` that differ only in non-targets:
  - (A) `sst_paper`, with embeddings, `lm_head` and norms trainable
  - (B) targets only

  Run on Qwen3-8B on one B200 with bases and `W_ref` in host memory (D-18), or on Qwen3-4B if the budget forces it, with that caveat stated. Use two seeds if affordable. Record in-domain score, prior capability with paired statistics, and the share of `‖ΔW‖` in each parameter group.
- **Decision.** If B's in-domain score is no more than 1 point below A's and B's prior capability is not lower beyond the paired interval, targets-only becomes the default. Otherwise the paper-like policy stays, and every report prints the unconstrained share.

**5. PEFT OSF against SST at equal r.**
- **Protocol.** Reuse arm B of measurement 4 as the SST arm, because PEFT also freezes all non-targets. Add PEFT OSF at `effective_rank=0.99` with the same model, data, targets, learning rate and schedule. Optional arms are `osf(post_step=True)` at 0.99 and PEFT's default 0.5. Record the in-domain score, prior capability, peak memory, step time and measured protected-direction leak (`‖U_kᵀ δ‖`, `‖δ V_k‖`).
- **Decisions:**
  - `auto` prefers whichever method has the higher prior capability without losing more than 1 point of in-domain score.
  - If OSF matches SST within the paired interval at lower memory, `auto` uses OSF wherever SST does not fit.
  - If PEFT's hooks leak measurably and the post-step option removes the leak, the option becomes `osf`'s default under its own variant name.

## Implementation checklist

**Gate 0, Claim**
- [ ] Send the section 13 questions to the SST authors (roadmap item 9).
- [ ] Implement `memory_terms` for `lora`, `full`, `freeze`, `sst` and `osf`, so that `estimate` can print per-route budgets.
- [ ] Write the variant strings and citations into the docs.

**Gate 1, Verify**
- [ ] Prototype projector and callback, single GPU, written from this document only.
- [ ] Measurements 1 to 5 as specified in section 14, published including any disagreement.
- [ ] Record the decisions on embedding policy, `osf` default rank and `auto` preference in the plan index.

**Gate 2, v0.1 Door**
- [ ] `tailtune.bases`: CPU fp32 SVD, mixed side forms, checks 6.2, hash, cache, name keying.
- [ ] `tailtune.methods` Protocol and registry. `sst` with the cumulative form, tail-first projection, skip handling and logging (3.8).
- [ ] Precision guard (section 5), including the `including_emulation=False` detection and the T4 fp16 route.
- [ ] `osf` with every guard in 7.2. `lora` on 16-bit and 4-bit bases. `full` and `freeze`.
- [ ] The `sst_paper` preset, with its resolver-equality assertion.
- [ ] Validate tailtune `sst` on one B200, with bases and `W_ref` in host memory, against the measurement 1 reference.
- [ ] Measure the gap between bf16 export and fp32 retention.

**Gate 3, v0.2 Proof**
- [ ] `sst` on `fsdp2` ([08](08-distributed.md)).
- [ ] `osf(post_step=True)` if measurement 5 supports it.
- [ ] Per-role LoRA ranks for the capacity-matched benchmark arm.

**Gate 4, v0.3 Reach**
- [ ] Refresh (`K`) under the unverified label. Benchmark the incremental candidate against a fresh SVD.
- [ ] `tail-lora`, and its 4-bit benchmark.
- [ ] `tailtune repair` if DG-Hard verifies.
- [ ] Measure sequential chains of more than three stages.
- [ ] SDFT composition (D-16). Its gate still needs to be recorded in the roadmap.

**Gate 5, v1.0**
- [ ] Freeze the method config schema.
- [ ] Decide on out-of-tree method registration.

## Acceptance tests

| # | Test | Pass condition |
|---|---|---|
| 1 | Head form against tail-first form on random fp32 matrices of all three shape classes, including odd sizes | Relative difference ≤ 1e-5. The result is idempotent, and its protected blocks are ≤ 1e-6 relative |
| 2 | `k` computation over `s` from 2 to 20,000 and `r` in {0.5, 0.8, 0.9, 0.99, 1.0} | Equals the exact rational floor, clamped to [1, s−1]. `s < 2` is refused |
| 3 | Frozen-subspace invariant: 50 AdamW steps on a tiny model on CPU | `‖U_kᵀ(W − W_ref)‖ / ‖W − W_ref‖ ≤ 1e-5` on every target, and likewise on the right side |
| 4 | Cumulative and per-step forms agree on fixed bases over 50 steps | Relative difference ≤ 1e-5 |
| 5 | Hook placement: a fake loop records the order. A fake DeepSpeed-style wrapper has `step()` as a no-op and steps inside `backward` | Projection always follows the real step. It never fires per micro-batch. Nothing runs in `on_pre_optimizer_step` |
| 6 | Skipped step: force `optimizer_step_was_skipped`. Also call projection twice | No work on a skipped step. Weights unchanged by a second application within 1e-7 relative |
| 7 | Precision guard | bf16 targets, bf16 bases and `dtype="auto"` loads are refused with the fix in the message. A T4 profile selects fp16 with the scaler |
| 8 | Basis checks | A corrupted basis fails 6.2. Recomputing with a different thread count passes the subspace tolerance |
| 9 | OSF guards | A GPT-2 Conv1D target and a bitsandbytes layer are refused before wrapping. A silently skipped target fails `assert_wrapped_exactly` |
| 10 | `sst_paper` preset on Qwen3-8B's module tree | Resolver set equals the suffix rule. 252 matrices. Embeddings trainable |
| 11 | Logging identities | `drift²` + `removed²` = raw² within 1e-5 relative. The step norm equals the snapshot-based norm on a tiny model |
| 12 | Reference parity (Gate 2, GPU) | tailtune `sst` within ±2 points of the measurement 1 reference on prior capability and in-domain score [J] |

## Open questions

1. **Refresh route.** D-14 now allows GPU SVD on CUDA or ROCm for refresh. Gate 4 must still choose between the owner-rank GPU SVD route in [08](08-distributed.md) and the incremental route, which is untested. ROCm's `gesvdj` may fail the orthonormality checks.
2. *(Resolved: precision refusals exit with code 4.)*
3. Which domain and budget for measurement 1, and is the SFT arm affordable?
4. Should `osf` default to PEFT's 0.5 or to 0.99 after measurement 5?
5. How should `freeze` be defined: by last N blocks, by roles, or as the hardware notes' partial-layer layout?
6. Does tail learning survive chains of more than three tasks, and which capacity rule works at r=0.99?
7. Does NF4 quantization error swamp the tail spectrum, which decides whether `tail-lora` can work?
8. Is the incremental refresh numerically stable when directions swap in and out of the protected set?
9. How much retention does a bf16 export lose against the fp32 weights?
10. The answers to the questions for the authors in section 13.
