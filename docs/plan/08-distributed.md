# 08. Distributed training

**What this document decides:**

- **Scale order (D-19).**
  - `single` and `ddp` come first (Gate 2).
  - `fsdp2` is the one default sharded path (Gate 3).
  - `zero3` comes last, and only on real demand (Gate 5).
  - Public SST on DeepSpeed ZeRO-2 serves as the reference run only.
  - FSDP1 is refused.
- **`tailtune.backends` hands the projector row blocks of each target matrix and a reducer.** The projector maths in `tailtune.methods` never imports `torch.distributed` or DeepSpeed.
- **On FSDP2, projection is a local multiply on each rank's `Shard(0)` rows plus one bucketed all-reduce of small coefficient matrices per layer.** The tail is applied first, after `reshard()`. On Qwen3-8B this moves about 134 MB per step, against 27.8 GB for full reassembly [C].
- **Initial bases are computed once, offline on the CPU path, and distributed.** Every rank verifies the same hash before training starts. Refresh may use GPU SVD on CUDA or ROCm (D-14). Each refreshed matrix is computed once on one owner rank and then distributed.
- **Two pieces are new engineering, not a port.** Dense weight-space projection under FSDP2 and any distributed refresh have no public implementation anywhere. tailtune would be the first, so both are labelled untested designs until their tests pass.

The method itself (Eq. 3, the cumulative form, head and tail forms, notation) is specified in [03](03-methods.md). Evidence labels are defined in the [plan index](README.md#evidence-labels). No distributed code has been run during the research. Every design and every byte count below is a computation or a judgement until Gate 3.

## 1. Modes

| Mode | Weight form | Projection route | Communication per step on Qwen3-8B at r=0.99 | Status |
|---|---|---|---|---|
| `single` | Plain fp32 tensor | In place under `no_grad`, using the tail-first forms in [03](03-methods.md) | None | Gate 2. Build first |
| `ddp` | A full fp32 copy on every rank | The same, run independently on each rank | None, plus a periodic drift-check fingerprint of a few kilobytes | Gate 2 |
| `fsdp2` | `DTensor` with `Shard(0)` and fp32 sharded storage [S](https://docs.pytorch.org/docs/2.15/distributed.fsdp.fully_shard.html) | Local multiply, then one all-reduce of a small coefficient matrix per target, bucketed per layer, after `reshard()` | About 33.5M fp32 values, 134 MB (variant A), or about 161 MB (variant B) [C] | Gate 3. The default sharded path. Untested design |
| `zero2` | bf16 module parameters plus partitioned fp32 masters | `safe_get_full_fp32_param`, project, `safe_set_full_fp32_param`, then sync the bf16 copy [S](https://deepspeed.readthedocs.io/en/latest/zero3.html) | About 27.8 GB of full fp32 reassembly [C] | The only public SST path [S](https://github.com/Arnav0400/SST/blob/ff0c5bf/sst/config.py#L45-L51). Reference run at Gate 1. A tailtune adapter only if a user needs one |
| `zero3` | Flat 1-D `ds_tensor` slices whose rows do not align with shards [S](https://github.com/deepspeedai/DeepSpeed/blob/master/deepspeed/runtime/zero/partition_parameters.py#L1855-L1858) | Gather, project and set each weight, and also write `ds_tensor` | A weight-sized all-gather per matrix, about 27.8 GB per step [C] | Gate 5, on demand, release-gated |
| FSDP1 | `FlatParameter` views | Would need `summon_full_params()` for each module | High | Unsupported (D-17) |

For scale: under FSDP2 with `reduce_dtype=float32`, the gradient reduce-scatter alone carries 8.19B values, or 32.8 GB, every step [C]. The FSDP2 projection adds about 0.4% to that traffic [C]. It is small in bytes but made of many small messages, so its latency still has to be measured.

## 2. The `tailtune.backends` contract

A backend answers three questions:

- Which rows of each target matrix does this rank hold, and where can they be written?
- How are partial sums combined across ranks?
- What must be synced after a write?

The projector receives row blocks and a reducer. It never sees DTensors, DeepSpeed objects or process groups. That keeps the method-by-backend matrix small enough for one maintainer.

```python
# Design proposal. Untested.
from dataclasses import dataclass
from typing import ClassVar, Iterator, Protocol, Sequence
import torch

@dataclass
class MatrixView:
    name: str                    # checkpoint parameter name, plus "#e" for expert e
    global_shape: tuple[int, int]
    rows: slice                  # rows of the global (d_out, d_in) matrix held here
    local: torch.Tensor          # fp32, writeable, aliases the tensor the optimizer updates
    ref: torch.Tensor            # W_ref for the same rows (fp32, or bf16-exact, see 03)

class Reducer(Protocol):
    world_size: int
    def all_reduce_sum(self, bucket: list[torch.Tensor]) -> None: ...   # in place, fp32, one call per bucket

class WeightBackend(Protocol):
    mode: ClassVar[str]          # "single" | "ddp" | "fsdp2" | "zero2" | "zero3"

    def bind(self, model, accelerator, names: Sequence[str]) -> None: ...   # after wrapping, by name
    def begin_edit(self) -> None: ...                    # fsdp2: reshard(); others: nothing
    def buckets(self) -> Iterator[list[MatrixView]]: ...  # one bucket per decoder layer, fixed order
    def reducer(self) -> Reducer: ...
    def end_edit(self, bucket: list[MatrixView]) -> None: ...  # write back, sync working copies
    def check_consistency(self) -> dict: ...             # ddp drift, hash agreement
    def state_dict(self) -> dict: ...
    def load_state_dict(self, state: dict) -> None: ...
```

The row-block projector is written once. With `rows` covering the whole matrix and a reducer of world size 1, it reduces exactly to `project_free` in [03](03-methods.md).

```python
# Design proposal. Untested. Written from Eq. 3 of SST (Chavan et al., 2026), CC BY 4.0.
@torch.no_grad()
def local_coefficients(v, b):
    """Phase 1: this rank's partial coefficients. The sum over ranks happens in the reducer."""
    d = v.local - v.ref.float()                       # local rows of W - W_ref
    lrows = b.left.vecs[v.rows]                       # left basis rows matching this shard
    if b.left.form == "tail":                         # square or wide: reduce T_U^T D
        part = lrows.T @ d                            # (t, n)
        if b.right.form == "tail":
            part = part @ b.right.vecs                # square: (t, t)
        return part, None
    x = d @ b.right.vecs                              # tall: (m_local, t), stays on this rank
    return lrows.T @ x, x                             # (k, t), a partial of U_k^T X

@torch.no_grad()
def apply_reduced(v, b, coeff, x):
    """Phase 2: after the all-reduce, write W <- W_ref + Q_U (W - W_ref) Q_V locally."""
    lrows = b.left.vecs[v.rows]
    if b.left.form == "tail" and b.right.form == "tail":       # square
        d_proj = lrows @ coeff @ b.right.vecs.T
    elif b.left.form == "tail":                                 # wide: right head, replicated (variant A)
        y = coeff - (coeff @ b.right.vecs) @ b.right.vecs.T
        d_proj = lrows @ y
    else:                                                       # tall: left head rows are local
        d_proj = (x - lrows @ coeff) @ b.right.vecs.T
    v.local.copy_(v.ref).add_(d_proj)

def project_all(backend, bases):
    backend.begin_edit()
    for bucket in backend.buckets():                  # identical order on every rank
        parts = [local_coefficients(v, bases[v.name]) for v in bucket]
        backend.reducer().all_reduce_sum([c for c, _ in parts])
        for v, (c, x) in zip(bucket, parts):
            apply_reduced(v, bases[v.name], c, x)
        backend.end_edit(bucket)
```

**Rules every backend obeys:**

1. **Bind by name, after wrapping.** `bind` runs after `accelerator.prepare` and after `fully_shard`. It resolves parameters by name, and it strips the `module.` prefix added by DDP and the `_orig_mod.` prefix added by `torch.compile`. It refuses `_fsdp_wrapped_module.`, which marks FSDP1.
2. **Order collectives identically on every rank.** Every rank must issue the same collectives in the same order. `bind` sorts target names, hashes the list together with the bases hash, and all-gathers the hashes. If they disagree, training refuses to start with `BackendMismatch` instead of hanging.
3. **Write only between steps.** All writes happen inside `on_optimizer_step`, never between forward and backward. FSDP2 does not support that [S](https://docs.pytorch.org/docs/2.15/distributed.fsdp.fully_shard.html).
4. **Keep the reduction in fp32.**
5. **Bucket per decoder layer.** The default is one all-reduce per decoder layer, 36 on Qwen3-8B [J]. That balances latency against transient memory. mini_trainer measured batching 224 coefficient all-reduces into one at 7.1 ms against 12.4 ms, on 2 H200s [S](https://github.com/Red-Hat-AI-Innovation-Team/mini_trainer/pull/72).
6. **Reduce scalar statistics once.** The logging metrics in [03](03-methods.md) are reduced in one scalar all-reduce at logging steps only.

The callback, skip handling and step order are identical on every backend, as specified in [03](03-methods.md). On every backend, `on_optimizer_step` is the correct point, including under DeepSpeed, where the real step has already run inside `backward` [S](https://github.com/huggingface/accelerate/blob/main/src/accelerate/utils/deepspeed.py#L264-L312).

## 3. `single` and `ddp`

**`single`.** Each view covers every row of `W`, the reducer is trivial, and `end_edit` does nothing. Parameters are fp32, and compute runs under bf16 autocast, or fp16 with the scaler on a T4 (D-12).

**`ddp`.** Every rank holds a full copy and projects it independently, with no communication. DDP synchronises gradients, not parameters, so any non-determinism lets ranks drift apart without anyone noticing. Two examples are a different matrix-multiply algorithm on a heterogeneous node and a non-deterministic reduction. The drift check below is our design, and both thresholds are [J], to be calibrated:

| Item | Specification |
|---|---|
| Fingerprint per target | Sum of the fp32 bit pattern viewed as int32, accumulated in int64, plus a float64 sum of values. That is two numbers per target, about 4 KB for 252 targets |
| When | At `bind`, after DDP's initial broadcast. Every 200 applied steps [J]. Before every checkpoint save. At the end of training |
| Collective | One all-gather of the fingerprint vector |
| On a match | Nothing |
| On a mismatch | Broadcast rank 0's tensor for the mismatching targets only, and compute the relative difference. If it is at most 1e-6 [J], overwrite from rank 0, warn and increment `ddp/drift_repairs`. Otherwise raise `DriftError`, which indicates a bug such as a hook missing on one rank or different data |
| Also checked at `bind` | The `W_ref` hash and the bases hash agree on all ranks |

Splitting projection work across DDP ranks and broadcasting the results is a later optimisation. It is not part of v0.1.

## 4. FSDP2: the design

### 4.1 What FSDP2 provides

These facts are sourced from the [`fully_shard` docs](https://docs.pytorch.org/docs/2.15/distributed.fsdp.fully_shard.html) unless noted:

- **Sharding.** `fully_shard` converts each parameter to a `DTensor` sharded on dim 0, with new sharded storage [S](https://github.com/pytorch/pytorch/blob/v2.9.0/torch/distributed/fsdp/_fully_shard/_fsdp_param.py#L258-L276). Fully qualified names do not change, because the module is not wrapped.
- **Optimizer.** The optimizer is built from the DTensors and steps on them. `param_dtype` sets only the unsharded compute dtype, and "the optimizer uses the sharded parameters in their original dtype". With fp32 storage, the shard the optimizer updates is therefore the fp32 master.
- **Reading and writing.** `to_local()` gives the shard, and `full_tensor()` all-gathers the full value. Before editing sharded parameters, the docs advise calling `reshard()` on every FSDP module on every rank. Modifying parameters between forward and backward is unsupported.
- **Calls.** Call `model(input)`, not `model.forward(input)`, or register the method with `register_fsdp_forward_method`.
- **What is not guaranteed.** The docs do not explicitly bless editing weights after the optimizer step, so the invariant is tested rather than assumed.

### 4.2 The maths

Under `Shard(0)`, rank `ρ` holds rows `R_ρ` of the `d_out × d_in` weight. Write `D_ρ = W[R_ρ] − W_ref[R_ρ]`, and let `B[R_ρ]` be the matching rows of a left basis. The left projection mixes rows, so it needs a sum over ranks. The right projection acts within rows, so it is local whenever the right basis is present on the rank. Applying the tail first makes the summed object small. Notation follows [03](03-methods.md): `t` is the short-side tail size and `k` the protected rank.

**Square matrices: `q_proj`, `o_proj` (4096×4096).** Left tail `T_U` is row-sharded, and right tail `T_V` is replicated.

```
C   = Σ_ρ  T_U[R_ρ]ᵀ (D_ρ T_V)          (t×t, one all-reduce)
D'_ρ = T_U[R_ρ] C T_Vᵀ                   (local)
```

**Wide matrices, `m < n`: `k_proj`, `v_proj` (1024×4096) and `down_proj` (4096×12288).** Left tail `T_U` is row-sharded, and right head `V_k` is replicated in variant A.

```
Y   = Σ_ρ  T_U[R_ρ]ᵀ D_ρ                 (t×n, one all-reduce)
Y'  = Y − (Y V_k) V_kᵀ                   (replicated compute)
D'_ρ = T_U[R_ρ] Y'                        (local)
```

**Tall matrices, `m > n`: `gate_proj`, `up_proj` (12288×4096).** Left head `U_k` is row-sharded, and right tail `T_V` is replicated.

```
X_ρ = D_ρ T_V                            (local, m_ρ×t)
G   = Σ_ρ  U_k[R_ρ]ᵀ X_ρ                 (k×t, one all-reduce)
D'_ρ = (X_ρ − U_k[R_ρ] G) T_Vᵀ           (local)
```

Each rank then writes `W[R_ρ] ← W_ref[R_ρ] + D'_ρ` into its fp32 shard. Concatenating the `D'_ρ` gives exactly `Q_U D Q_V`, because `Q_U D = Σ_ρ B[R_ρ](B[R_ρ]ᵀ D_ρ)` for a tail basis `B`, and `D − U_k(U_kᵀ D)` for a head. Non-distributed equivalence was checked in the research's NumPy tests. The distributed version is untested [C].

### 4.3 Communication on Qwen3-8B at r=0.99 [C]

| Matrix (per layer) | Case | All-reduce payload per matrix | Values |
|---|---|---|---|
| `q_proj`, `o_proj` (2) | Square | t×t = 41×41 | 1,681 each |
| `k_proj`, `v_proj` (2) | Wide | t×n = 11×4096 | 45,056 each |
| `gate_proj`, `up_proj` (2) | Tall | k×t = 4055×41 | 166,255 each |
| `down_proj` (1) | Wide | t×n = 41×12288 | 503,808 |
| One layer | | | 929,792 |
| 36 layers | | | 33.5M values, 134 MB fp32 |

The full fp32 reassembly that ZeRO-2 performs moves 6.946B values (27.8 GB), about 200x more [C]. The research's hardware notes also sketched a head-first variant whose all-reduce is `k×n` per matrix: 4055×4096 for `q_proj` alone. This document adopts tail-first, because it is about 100x smaller on square matrices.

### 4.4 Variant B: column-sharded right heads

Variant A replicates `V_k` on wide matrices. On Qwen3-8B that is 8.5 GB per rank, mostly `down_proj`'s 12288×4055 head [C]. Variant B shards those rows by column chunk `J_j` instead:

1. Replace the `Y` all-reduce with a reduce-scatter by column, so rank `j` holds `Y[:, J_j]`.
2. All-reduce `P = Σ_j Y[:, J_j] V_k[J_j, :]` (t×k).
3. Compute `Y'[:, J_j] = Y[:, J_j] − P V_k[J_j, :]ᵀ` locally.
4. All-gather `Y'`, so every rank holds all of it.

A reduce-scatter plus an all-gather carry the same volume as one all-reduce. The extra cost is therefore only the t×k all-reduce: about 6.8M values, or 27 MB, per step [C]. Build variant A first, because it has fewer collectives and is easier to test. Switch to B when `estimate` says replicated right bases do not fit [J].

### 4.5 Per-rank memory [C]

Qwen3-8B at r=0.99, all in GB:

| Item | Formula | N=2 | N=4 | N=8 |
|---|---|---|---|---|
| fp32 weights, gradients and AdamW state | 131/N | 65.5 | 32.8 | 16.4 |
| `W_ref` fp32 (bf16-exact, `K=None` only) | 27.8/N (13.9/N) | 13.9 (6.9) | 6.9 (3.5) | 3.5 (1.7) |
| Left bases, row-sharded | 14.4/N | 7.2 | 3.6 | 1.8 |
| Right bases, variant A (replicated) | 8.5 | 8.5 | 8.5 | 8.5 |
| Right bases, variant B | 0.1 + 8.4/N | 4.3 | 2.2 | 1.2 |
| **Total, variant A with fp32 `W_ref`** | | 95.1 | 51.8 | 30.2 |
| **Total, variant B with bf16-exact `W_ref`** | | 83.9 | 42.1 | 21.1 |

These totals exclude:

- activations, about 1 GiB at 1,024 tokens with checkpointing
- one unsharded bf16 block, about 0.4 GiB
- communication buffers
- the 10% fragmentation factor in [07](07-hardware.md)

On paper, 4 × 80 GB fits either variant. 2 × 96 GB fits neither once the fragmentation factor is applied. Holding `W_ref` and bases in host memory and streaming them per layer moves those rows off the GPU, at a transfer cost that Gate 1 measurement 2 quantifies. Compute per rank is about `1.0T/N` multiply-adds, plus about 0.15T of replicated `Y'` work under variant A [C].

### 4.6 The edit protocol, and why captured objects go stale

At each applied step:

1. **Check the skip flag.** If `optimizer_step_was_skipped`, return ([03](03-methods.md)).
2. **Reshard.** `begin_edit` calls `reshard()` on every FSDP module on every rank, as the docs advise. This frees any unsharded copy, so the next forward all-gathers from the edited shards.
3. **Project.** For each layer bucket: phase 1 on `to_local()` shards, one fp32 all-reduce, then phase 2 writes into the same local tensors in place.
4. **No low-precision sync.** No step is needed, because the next forward casts the sharded fp32 storage to `param_dtype` during its all-gather. This is inferred from the docs and must be tested, so it is [U].

**Why parameters are keyed by name.** `fully_shard` replaces each parameter with a new DTensor parameter that has new storage. Any object captured before wrapping still exists, but nothing reads it. This includes a dict keyed by parameter, a list built for the bases, and an optimizer created before sharding. Writes to such objects change nothing, and nothing raises. So:

- bases and views are keyed by name;
- `bind` resolves names only after sharding;
- at the first projection, `bind` asserts each resolved parameter is a `DTensor` with a `Shard(0)` placement, whose global shape equals the basis shape;
- `tt.wrap` checks that the optimizer's parameters are those same DTensors.

**Uneven shards.** When `d_out` is not divisible by the world size, the shards are uneven, and some ranks may hold fewer rows or none. mini_trainer pads for uneven DTensor shards [S](https://github.com/Red-Hat-AI-Innovation-Team/mini_trainer/blob/fd5b552/src/mini_trainer/osft_utils.py#L650-L781). tailtune derives `rows` with a helper whose rule is [U]. A test proves the helper on every supported torch version by comparing `full_tensor()[rows]` with `to_local()` on odd shapes. A rank with zero rows still joins every collective, contributing a zero partial.

**Mixed precision and checkpointing.**

- Load the model in fp32.
- Use `MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)`, as mini_trainer does [S](https://github.com/Red-Hat-AI-Innovation-Team/mini_trainer/blob/fd5b552/src/mini_trainer/setup_model_for_training.py#L438-L440). The reduce dtype is fp32 because constrained steps are small [J]. It must be set explicitly, because of the 2.15 default change (section 8).
- Gradient checkpointing always uses `use_reentrant=False`.
- FSDP2 on a T4, which means fp16 with sharding, is not supported in v0.2 [J].

**MoE.** `Shard(0)` on an `[E, m, n]` expert tensor splits the expert dimension, so each expert matrix sits whole on one rank. Its projection is local with a trivial reducer [C]. MoE stays experimental ([06](06-models.md)).

**Other methods.** PEFT's OSF has a DTensor-aware helper, but its live hook does not call it [S](https://github.com/huggingface/peft/blob/main/src/peft/tuners/osf/utils.py). So `osf` stays on `single` and `ddp` until a test shows otherwise. For `lora` on a 4-bit base under FSDP2, Accelerate v1.14 excluded `Params4bit` from `fully_shard` to prevent QLoRA crashes [S](https://github.com/huggingface/accelerate/releases). That route is not a v0.2 priority.

## 5. `zero2`: the reference path

Public SST runs here, and only here. Its paper says ZeRO-3, but its config maps `sst` to ZeRO-2 [S](https://github.com/Arnav0400/SST/blob/ff0c5bf/sst/config.py#L45-L51). At Gate 1, tailtune runs the public repo unmodified as the reference and copies nothing.

If a user later needs tailtune's own `zero2` backend, it is a small adapter. For each target, in sorted order and on all ranks:

1. `full = safe_get_full_fp32_param(p)`. This is collective, and every rank receives the full fp32 tensor [S](https://deepspeed.readthedocs.io/en/latest/zero3.html). At SST's pinned v0.18.4, ZeRO-1 and ZeRO-2 reassemble it with an all-reduce over a full zero-filled buffer [S](https://github.com/deepspeedai/DeepSpeed/blob/v0.18.4/deepspeed/utils/tensor_fragment.py#L87-L95).
2. Project with a view covering all rows and a trivial reducer, so every rank does the same compute.
3. `safe_set_full_fp32_param(p, projected)`. The full-form calls "must be called by all processes" or they hang [S](https://deepspeed.readthedocs.io/en/latest/zero3.html).
4. Copy the projected value, cast to the parameter dtype, into the full bf16 module parameter that ZeRO-2 keeps. Without this, the next forward would use the unprojected value.

The cost is about 27.8 GB of reassembly per step [C], with every rank projecting every matrix. The public code also keeps bases and `W_ref` in every process: about 67.5 GB per rank, and about 540 GB of host RAM for 8 ranks [C]. tailtune's adapter would hold one copy per node in host shared memory instead [J]. It also asserts that `safe_get_full_fp32_param` returns fp32, which is the precision rule from [03](03-methods.md).

## 6. `zero3`: gather, set and `ds_tensor`

**What differs from ZeRO-2:**

- **Partitions are flat 1-D slices.** Each parameter is cut as `one_dim_param.narrow(0, start, partition_size)` [S](https://github.com/deepspeedai/DeepSpeed/blob/master/deepspeed/runtime/zero/partition_parameters.py#L1855-L1858), so partition boundaries ignore rows and there is no local shortcut.
- **Partitioned parameters hold an empty `param.data`** [S](https://github.com/deepspeedai/DeepSpeed/blob/v0.18.4/deepspeed/runtime/zero/partition_parameters.py#L309). Public SST copies a full tensor into `param.data` at line 120 and should raise on ZeRO-3, by reading only [S](https://github.com/Arnav0400/SST/blob/ff0c5bf/sst/methods/sst.py#L120).
- **The setter writes only this rank's fp32 partition** [S](https://github.com/deepspeedai/DeepSpeed/blob/v0.18.4/deepspeed/runtime/zero/stage3.py#L2490-L2507). The bf16 `ds_tensor` that the next forward gathers would keep the unprojected value until the following step.

The design, for each target in sorted order:

1. `full = safe_get_full_fp32_param(p)`, which all-gathers.
2. Project the full matrix.
3. Call `safe_set_full_fp32_param(p, projected)`.
4. Copy this rank's slice of `projected.to(bf16).view(-1)`, at its partition offset, into `p.ds_tensor`, honouring the padding.

The offset rule is [U] for each DeepSpeed version. A test proves it by gathering the parameter after the write and comparing it with the projected value. With `offload_param`, `ds_tensor` lives on the host and the write goes there.

**Supporting rules:**

- **`W_ref`.** Every rank needs `W_ref` for full matrices, so it lives in host shared memory per node: 27.8 GB, or 13.9 GB bf16-exact [C].
- **Bases.** ZeRO-3 "shards parameters during initialization", so `TrainingArguments` must be created before the model loads [S](https://huggingface.co/docs/transformers/main/en/deepspeed). Bases therefore come from the checkpoint files, which D-14 requires anyway.
- **Cost.** About 27.8 GB of gathers per step, plus host-to-GPU transfers under offload [C].
- **Status.** The Transformers docs advise ZeRO-3 only "when your model doesn't fit across GPUs with ZeRO-2" [S](https://huggingface.co/docs/transformers/main/en/deepspeed). Build it at Gate 5, only when a user with a real model needs it.

## 7. Why FSDP1 is unsupported

- FSDP1 flattens each wrapped module's parameters into one `FlatParameter` and shards the flat buffer. A matrix may be wholly, partly or not at all inside a rank's shard [S](https://github.com/pytorch/pytorch/blob/v2.9.0/torch/distributed/fsdp/_flat_param.py#L151-L155), with boundaries cut at arbitrary offsets.
- Per-matrix projection would need `summon_full_params()`, which means a gather, for every module at every step [S](https://docs.pytorch.org/docs/2.15/fsdp.html).
- FSDP1 does not support gradient accumulation outside `no_sync()` with CPU offload [S](https://docs.pytorch.org/docs/2.15/fsdp.html).
- PyTorch states that "FSDP1 is deprecated" [S](https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html).

`bind` refuses any model containing a `FlatParameter` or FSDP1 wrapper names, and any Accelerate config with `fsdp_version` 1. The message tells the user to set version 2. The run exits with code 4, a pre-flight refusal. The same code covers any other unsupported backend for a method, such as `osf` on `fsdp2`.

## 8. Accelerate and Trainer traps

| Trap | Effect | tailtune rule | Source |
|---|---|---|---|
| Under Accelerate's DeepSpeed wrapper, `step()` is `pass`, and `engine.step()` runs inside `backward` | `on_pre_optimizer_step` fires after the real step. A step post-hook on the wrapper never fires. Hooks on the inner optimizer would see flat partitions (inferred, untested) | Project in `on_optimizer_step` only (D-13). Test with a fake wrapper in CI and with real DeepSpeed at release | [deepspeed.py](https://github.com/huggingface/accelerate/blob/main/src/accelerate/utils/deepspeed.py#L264-L312) |
| Accelerate defaults to `fsdp_version` 1 "to not break existing code". Trainer defaults to 2 | The same job runs FSDP1 or FSDP2 depending on the launcher | Write version 2 explicitly into both the Accelerate config and `fsdp_config`. At `bind`, assert that parameters are DTensors | [Accelerate](https://huggingface.co/docs/accelerate/concept_guides/fsdp1_vs_fsdp2), [training_args.py](https://github.com/huggingface/transformers/blob/main/src/transformers/training_args.py#L2826-L2827) |
| From torch 2.15, `reduce_dtype=None` reduces in the gradient dtype. With fp32 parameters and bf16 compute, that doubles reduce-scatter bytes | Behaviour differs between torch 2.14 and 2.15 | Always set `reduce_dtype=torch.float32` explicitly | [2.15 notes](https://github.com/pytorch/pytorch/releases/tag/v2.15.0) |
| Reentrant checkpointing under FSDP2 | A layer run twice before `backward()` gets gradients from the last forward only, and nothing raises. A CPU repro measured a relative gradient error of 0.405 | Force `use_reentrant=False`. Refuse an override under `fsdp2` | [#50424](https://github.com/huggingface/transformers/issues/50424), [PyTorch #203482](https://github.com/pytorch/pytorch/issues/203482) |
| Distributed checkpoints written by 2.15 cannot be loaded by 2.14 | Resume fails across the version floor | Record the torch version in checkpoint metadata. Refuse cross-version resume with a clear message. Never share checkpoint fixtures across versions in CI | [2.15 notes](https://github.com/pytorch/pytorch/releases/tag/v2.15.0) |
| "Accelerate ignores the `deepspeed` argument in TrainingArguments" | DeepSpeed config silently unused | Configure DeepSpeed only through the Accelerate config | [Transformers](https://huggingface.co/docs/transformers/main/en/deepspeed) |
| ZeRO-3 shards parameters during initialization | The model must load after `TrainingArguments` | Fix the loader order. Take bases from files | [Transformers](https://huggingface.co/docs/transformers/main/en/deepspeed) |
| torch 2.15 makes `nccl` the rewritten backend. The `nccl-legacy` and `TORCH_DIST_USE_NCCL2=0` rollbacks "will be removed in the next release" | Possible regressions | `tailtune doctor` reports the backend. Document the rollback for 2.15 only | [2.15 notes](https://github.com/pytorch/pytorch/releases/tag/v2.15.0) |
| `torch.compile` with hooks: backward hooks fire late, and `skip_nnmodule_hook_guards` hides later hook changes | Gradient-hook methods misbehave | `sst` installs no hooks, and its post-step edit sits outside the graph (untested). `osf` refuses compile until tested | [compile docs](https://github.com/pytorch/pytorch/blob/v2.15.0/docs/source/user_guide/torch_compiler/torch.compiler_nn_module.md) |
| Windows supports "all collective communications backends but NCCL" | No multi-GPU on Windows | Unsupported (D-17) | [torch.distributed](https://docs.pytorch.org/docs/2.15/distributed.html) |

## 9. Bases distribution, refresh and resume

### 9.1 Distribution

Initial bases are computed once, offline, on the CPU path (D-14, [03](03-methods.md)). Refreshed bases are computed once per matrix on its owner rank (9.2). No basis is ever computed independently on every rank, because different hardware "may compute different singular vectors" [S](https://docs.pytorch.org/docs/2.15/generated/torch.linalg.svd.html). mini_trainer likewise computes once and broadcasts [S](https://github.com/Red-Hat-AI-Innovation-Team/mini_trainer/blob/fd5b552/src/mini_trainer/osft_utils.py#L1383-L1612).

At `bind`:

- Each rank loads only what it needs: its rows of the left bases, plus the replicated right tails and (in variant A) the right heads.
- If partial reads prove impractical, one loader per node reads the files into host shared memory and the local ranks map them [J].
- Every rank checks the manifest's bases hash, and the per-tensor source-weight hashes against the loaded checkpoint, then joins the hash all-gather from section 2.

Across nodes, the cache sits on a shared filesystem or is copied to each node, and the same check applies.

### 9.2 Refresh under sharding

Distributed refresh is unsolved everywhere. mini_trainer's SVD path runs only at initialisation and goes through rank 0 [S](https://github.com/Red-Hat-AI-Innovation-Team/mini_trainer/blob/fd5b552/docs/distributed_initialization.md).

**Refresh rules out the CPU path.** A full CPU SVD pass on an 8B model takes 0.5 to 3.5 hours [C], against a 1.73 s SFT step in the paper's setup [S](https://arxiv.org/html/2610.22293). D-14 therefore allows GPU SVD for refresh on CUDA or ROCm, with fp32 input and the same orthonormality checks. Initial bases stay on the CPU path.

**The baseline refresh route is owner-rank GPU SVD.** It is untested and a Gate 4 item.

1. Assign each target to an owner rank, balancing `m·n·min(m, n)` across ranks [J].
2. Gather each target's full fp32 value onto its owner only, not onto every rank. That is about 27.8 GB of total traffic per refresh on Qwen3-8B [C].
3. On the owner, run `torch.linalg.svd` in fp32. On CUDA, use the `gesvd` driver, which the docs recommend "when precision matters" [S](https://docs.pytorch.org/docs/2.15/generated/torch.linalg.svd.html). ROCm hard-codes `gesvdj` [S](https://github.com/pytorch/pytorch/blob/v2.15.0/aten/src/ATen/native/cuda/linalg/BatchLinearAlgebraLib.cpp#L657-L662), which the source warns "may be numerically unstable for large sized matrix" [S](https://github.com/pytorch/pytorch/blob/v2.15.0/aten/src/ATen/native/cuda/linalg/BatchLinearAlgebraLib.cpp#L657-L680). On ROCm, a failed check is therefore an expected event, not a rare one.
4. Run the [03](03-methods.md) checks: ordering, orthonormality, reconstruction and tail energy. If any check fails, keep the previous bases for that matrix, log `refresh/rejected`, and retry at the next refresh.
5. Scatter left-basis rows to their row owners. Broadcast the right tails, and the right heads under variant A, or scatter their column chunks under variant B.

GPU SVD time on LLM-sized matrices has not been measured anywhere in the research. Gate 4 measures it before R10 or R100 is offered.

**A cheaper candidate is the incremental refresh in [03](03-methods.md).** It is also untested and a Gate 4 item, benchmarked against the owner-rank route by subspace distance. Under sharding it needs only small objects:

| Shape | What refresh needs | Cost under sharding |
|---|---|---|
| Square | The core `S_t + C` | `C` is already all-reduced by projection, so this is free |
| Tall | `M_ρ = W[R_ρ] T_V`, then a tall-skinny QR | All-gather the t×t R factors, then a t×t SVD |
| Wide | `N = T_Uᵀ W` | One t×n all-reduce, the same size as projection |

**Rules for any refresh route:**

- Every swap decision is computed from replicated, reduced data, or made on rank 0 and broadcast, so all ranks keep identical bases.
- `W_ref ← W` is a local shard copy in fp32. The bf16-exact option is invalid once refresh is on.
- The new bases hash is recorded in the checkpoint.
- Refresh runs after projection in the same `on_optimizer_step`.
- Refresh counts applied steps.

### 9.3 Resume: what must be saved

| State | `K=None` | `K` set | Where, under `fsdp2` |
|---|---|---|---|
| Model and optimizer | Saved | Saved | Distributed checkpoint, sharded |
| Applied and skipped step counts, last refresh step, drift-repair count | Saved | Saved | Rank 0 |
| Variant string, `r`, `k` per target, target-list hash, backend mode, world size, torch version | Saved | Saved | Rank 0 |
| Bases | Hash only. Reload from the cache and verify | Full bases and singular values | Left bases sharded with the checkpoint. Right bases saved once |
| `W_ref` | Hash only. Rebuild from the base checkpoint and verify per-tensor hashes | Full fp32 | Sharded |
| Previous free coordinates, used for the step-norm metric | Optional | Optional | Local |

**On resume:**

- Refuse if the torch minor version differs.
- Refuse if the world size differs, because resharding on resume is untested [J].
- Refuse if a cached basis is missing, unless the user allows recompute. A recompute is recorded as a new bases hash in `lineage.json`.
- Before the first step, check that the protected content of `W − W_ref` is within the [03](03-methods.md) tolerance on every target. A failure means `W_ref` or the bases do not match the weights.

## 10. Launchers and multi-node

| Launcher | Use | Status |
|---|---|---|
| `accelerate launch` | Primary. `tailtune train` writes the fully resolved Accelerate config into the run directory, beside `tailtune.yaml`. Flags are specified in [04](04-api-and-ux.md) | Gate 2 for DDP, Gate 3 for FSDP2 |
| `torchrun` | For `tt.wrap` users with their own scripts. The same `bind` checks apply | Gate 2 |
| `deepspeed` | The Gate 1 reference run of the public repo only, then `zero3` | Gate 1, then Gate 5 |
| Slurm | A tailtune template, modelled on Accelerate's `submit_multinode.sh` and `submit_multinode_fsdp.sh` examples [S](https://github.com/huggingface/accelerate/tree/main/examples/slurm) but written by us | Untested until a multi-node run |
| Kubernetes | A documentation recipe on Kubeflow Trainer, which lists PyTorch, Hugging Face and DeepSpeed with JobSet and LeaderWorkerSet orchestration [S](https://www.kubeflow.org/docs/components/trainer/overview/). Details on the page read were thin [U] | Gate 5, docs only |

**Multi-node evidence and limits:**

- **No SST run spans multiple nodes in any source read.** SST's largest published model is 32B, and its authors call hundreds of billions untested [S](https://arxiv.org/html/2610.22293).
- **mini_trainer's multi-node fixes landed in August 2025** [S](https://github.com/Red-Hat-AI-Innovation-Team/mini_trainer/commits/main/src/mini_trainer/osft_utils.py). They are in factor space, not dense weights.
- **Per-weight gathers scale badly across nodes.** MatrixFSDP reports that existing systems "reconstruct matrices at every optimizer step". Its owner-rank placement cut optimizer-step latency 4.2x on one node and 54.6x on eight nodes against stock FSDP2-Muon. These figures are self-reported, on 64 A100s, with no code found [S](https://arxiv.org/abs/2607.05895). The `zero2` and `zero3` routes are exactly such gathers, so they suit single-node research runs only.
- **The FSDP2 design moves little data but sends many messages.** 36 bucketed all-reduces per step make inter-node latency the risk. Measure it before any multi-node claim.
- **Product limits.** D-03 rules out competing on multi-node scale. [01](01-product-strategy.md) names each path only after an end-to-end run. The experimental 27B models need about eight 96 GiB cards by the memory formula [C], so they stay on one node.

## 11. Test plan

| Tier | Where | What |
|---|---|---|
| Every PR | Free GitHub runners, CPU, gloo | Projector numerics. Hook order with a fake Trainer loop and a fake DeepSpeed-style wrapper. 2-rank DDP. 2-rank and 3-rank FSDP2 with uneven shapes. Frozen-subspace invariants on every CPU-runnable backend. Torch 2.14 and 2.15 |
| Weekly | Kaggle T4 kernel | `single` with fp16 AMP for more than 5 steps, plus control and canary legs. Copy the pattern from Unsloth's workflow, never the code, because the file carries an AGPL header [S](https://github.com/unslothai/unsloth/blob/main/.github/workflows/kaggle-t4-notebook-ci.yml) |
| Nightly | Modal L4, single GPU | Small real model, invariants, memory against `estimate` |
| Before every release | Modal 2×L4 or A100, or one RTX PRO 6000 | `fsdp2` on 2 GPUs with NCCL, save and reload, invariants, real-DeepSpeed hook order, and `zero3` once it exists. About $10 per release [C](https://modal.com/pricing). Whether Modal's credit covers multi-GPU apps is [U] |

The 2-rank gloo FSDP2 route is proven workable by the Transformers repro of #50424 [S](https://github.com/huggingface/transformers/issues/50424). Whether DeepSpeed's CPU path runs ZeRO tests in CI is [U]. If it does not, ZeRO tests are release-gated only.

**The FSDP2 CPU suite.** Tiny models with odd row counts such as 7, 13 and 30, so shards are uneven and one rank can be empty. Run on 2 and 3 ranks:

1. **Views.** `rows` matches `full_tensor()` slicing, and a write to `to_local()` is visible in `full_tensor()` after the edit [U until it passes].
2. **Equivalence.** The FSDP2 projection of identical weights equals the single-process projection.
3. **Invariant.** After 20 AdamW steps, the frozen-subspace invariant holds.
4. **Reshard.** After projection, the next forward's output equals that of an unsharded reference model carrying the projected weights.
5. **Staleness.** Writing to a parameter object captured before `fully_shard` changes nothing, and `bind` rejects such objects.
6. **No hang.** Mismatched target lists across ranks raise `BackendMismatch` within a timeout.
7. **Checkpointing.** The #50424 pattern under `use_reentrant=False` matches non-FSDP gradients, and `use_reentrant=True` is refused.
8. **Config.** Generated configs carry FSDP version 2 and an fp32 `reduce_dtype`, and an FSDP1 model is refused.
9. **Resume.** Saving and loading on the same torch version restores weights bit for bit, along with the counters, and the resume check passes. A checkpoint marked with another version is refused.
10. **Byte count.** The bytes counted per step on a scaled mock equal the formula in 4.3.

**Frozen-subspace invariant on every backend.** One parametrised test body covers `single`, `ddp`, `fsdp2` and, later, `zero2` and `zero3`. It trains N≥20 applied steps, then checks `‖U_kᵀ(W − W_ref)‖ / ‖W − W_ref‖ ≤ 1e-5` and the right-side equivalent on every target, reading `W` through each backend's own full-value path.

## 12. Build order

| # | Step | Gate | Rationale |
|---|---|---|---|
| 1 | Projector maths and bases as pure CPU code | 1 and 2 | Correctness on the cheapest substrate. Gate 1 measurement 2 needs it |
| 2 | Reference run: public SST on ZeRO-2, unmodified | 1 | The fastest route to a reference result. Copies nothing (D-06) |
| 3 | `single` backend and callback | 2 | Every later backend is checked against it |
| 4 | Validate tailtune `single` on one B200, with bases and `W_ref` in host memory (D-18), against step 2 | 2 | Proves the implementation before any sharding |
| 5 | `ddp` with the drift check | 2 | No new maths, only drift risk |
| 6 | `fsdp2` variant A: CPU gloo first, then 2 GPUs | 3 | The only design that stays cheap at scale (D-19). **Dense FSDP2 projection has no public implementation, so this is new engineering** |
| 7 | `fsdp2` variant B, if `estimate` or Gate 3 memory requires it | 3 or 4 | Removes 8.5 GB of replicated right bases per rank [C] |
| 8 | Refresh, single GPU first, then distributed, with GPU SVD (D-14) | 4 | **Distributed refresh has no public implementation anywhere, so this is new engineering.** It is unverified even on one GPU |
| 9 | `zero3`, then a `zero2` adapter only if asked | 5 | Gathers whole weights, with no local shortcut. Only on demand |

## Implementation checklist

**Gate 0, Claim**
- [ ] `tailtune doctor` reports the distributed backend, NCCL availability and GPU count. On Windows it reports single-GPU only.
- [ ] `tailtune estimate` prints per-rank `fsdp2` budgets for variants A and B, and the host RAM for the ZeRO-2 reference run.

**Gate 1, Verify**
- [ ] Run the public SST repo unmodified on ZeRO-2 for measurement 1, on a node with about 540 GB of host RAM [C], or with fewer ranks.
- [ ] Measure tail-form projection time on real shapes (measurement 2). This feeds the FSDP2 budget.

**Gate 2, v0.1 Door**
- [ ] The `tailtune.backends` Protocol, `MatrixView` and `Reducer`. Add an import rule that keeps `torch.distributed` and DeepSpeed out of `tailtune.methods`.
- [ ] `single` and `ddp` backends. The DDP drift check. Hash agreement at `bind`.
- [ ] Hook-order tests with a fake DeepSpeed-style wrapper. The CPU DDP suite on every PR.
- [ ] Validate `single` against the reference on one B200, with bases and `W_ref` in host memory (D-18).

**Gate 3, v0.2 Proof**
- [ ] `fsdp2` variant A with `reshard()`, name binding, uneven-shard helper and per-layer buckets.
- [ ] The CPU FSDP2 suite (section 11) and the release-gated 2-GPU run.
- [ ] Resume for `fsdp2`, with version and world-size refusals.
- [ ] Explicit FSDP version 2, an fp32 `reduce_dtype` and `use_reentrant=False`, written into generated configs.
- [ ] Accelerate config output and a Slurm template.
- [ ] Variant B, if memory requires it.

**Gate 4, v0.3 Reach**
- [ ] Refresh on `single`, then on `fsdp2`, under the unverified label: owner-rank GPU SVD with the [03](03-methods.md) checks, and the incremental candidate benchmarked against it. Measure GPU SVD time first.
- [ ] Run the experimental models (D-20) on `fsdp2`, labelled "runs".

**Gate 5, v1.0**
- [ ] `zero3` with `ds_tensor` writes, only on real demand. A `zero2` adapter only if a user asks for one.
- [ ] Kubernetes recipe. A multi-node claim only after an end-to-end run.

## Acceptance tests

| # | Test | Pass condition |
|---|---|---|
| 1 | Import boundary | `tailtune.methods` imports neither `torch.distributed` nor `deepspeed` |
| 2 | `single` against `project_free` in [03](03-methods.md) | Bit-identical on CPU |
| 3 | `ddp`, 2 gloo ranks, 20 steps | Fingerprints identical. An injected 1e-8 perturbation is repaired. An injected 1e-3 perturbation raises `DriftError` |
| 4 | `fsdp2` equivalence, 2 and 3 ranks, uneven shapes | `full_tensor()` after projection matches the single-process result within 1e-6 relative |
| 5 | `fsdp2` invariant after 20 steps | Within 1e-5 relative on every target and side |
| 6 | `fsdp2` reshard and staleness | The next forward uses projected weights. Writes to pre-wrap objects are rejected at `bind` |
| 7 | Rank mismatch | `BackendMismatch` within 60 s, with no hang |
| 8 | Configs and refusals | FSDP version 2 and fp32 `reduce_dtype` present. FSDP1 refused. `use_reentrant=True` refused under `fsdp2` |
| 9 | Resume | Same-version round trip is bit-identical, and cross-version resume is refused |
| 10 | Communication count | Counted all-reduce values equal the section 4.3 formula on a scaled mock |
| 11 | Release-gated GPU | A 2-GPU NCCL `fsdp2` run of a small model, 50 steps: invariant holds, save and reload succeed, and the retention report runs |
| 12 | DeepSpeed hook order, at release, once `zero2` or `zero3` exists | Projection always follows `engine.step()`. The invariant holds after N steps |

All tolerances are [J], to be calibrated alongside [03](03-methods.md)'s tests.

## Open questions

1. Does an in-place write to `to_local()` persist into the next forward's all-gather on torch 2.14 and 2.15? The `fully_shard` docs do not say so explicitly.
2. Which right-basis variant should be the default for `fsdp2`, given measured memory and the latency of the extra collective?
3. Which distributed refresh route should ship: owner-rank GPU SVD, the incremental route, or the paper's Gram route? How fast is GPU SVD on LLM-sized matrices, and how often does ROCm's `gesvdj` fail the checks?
4. Can resume change world size safely through distributed-checkpoint resharding?
5. How much latency do 36 bucketed all-reduces per step add across nodes?
6. Do PEFT OSF's gradient hooks behave correctly under DDP, and do they work at all under FSDP2?
7. Are 8-bit optimizers compatible with DTensor parameters? No evidence either way was found.
8. What is the `ds_tensor` partition offset rule for each DeepSpeed version, and is `optimizer_step_was_skipped` reliable under DeepSpeed fp16? The cumulative form makes the second question harmless for correctness.
9. Does Modal's free credit cover multi-GPU release tests?
10. Should `W_ref` and bases sit in host shared memory with memory-mapped safetensors, or in a torch shared-memory tensor?
