"""Memory budget per route, before any weights load (D-18).

Implements the formula in docs/plan/07-hardware.md section 9.1:

    V_gpu  = R + f * (W + G + O + B_gpu + W_ref,gpu + L + A)
    V_host = O_host + W_host + G_host + B_host + W_ref,host
    fit:     V_gpu <= u * VRAM  and  V_host <= available host RAM

Every figure is computed from shapes [C] and has not been checked against
hardware. Gate 1 (measurement 3) replaces the Calib defaults.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from tailtune._core.errors import ConfigError, DoesNotFitError
from tailtune._core.hub import load_config
from tailtune.hardware import gpus
from tailtune.hardware.shapes import Matrix, Shapes, shapes_from_config

GIB = 2**30

Verdict = Literal["fits", "borderline", "does not fit", "not checked"]

ROUTES: tuple[str, ...] = (
    "lora-4bit",
    "lora-16bit",
    "osf",
    "sst/gpu",
    "sst/host-bases",
    "sst/8bit-adam",
    "sst/offload-state",
    "sst/zero-offload",
    "full",
)
ROUTE_METHOD: dict[str, str] = {
    "lora-4bit": "lora",
    "lora-16bit": "lora",
    "osf": "osf",
    "sst/gpu": "sst",
    "sst/host-bases": "sst",
    "sst/8bit-adam": "sst",
    "sst/offload-state": "sst",
    "sst/zero-offload": "sst",
    "full": "full",
}
ROUTE_LABEL: dict[str, str] = {
    "sst/8bit-adam": "variant",
    "sst/zero-offload": "untested",
    "full": "baseline",
}
METHODS: tuple[str, ...] = ("auto", "lora", "osf", "sst", "full")
# method="auto" ranks by protection and never picks variants, offload routes or baselines.
AUTO_ORDER: tuple[str, ...] = ("sst/gpu", "sst/host-bases", "osf", "lora-16bit", "lora-4bit")
SMALL_MODEL_PARAMS = 4_000_000_000


@dataclass(frozen=True)
class Calib:
    """Constants of the memory model. Gate 1 replaces these values."""

    reserve_gib: float = 1.0
    frag: float = 1.10
    live_tensors: float = 12.0
    usable: float = 0.92
    band: tuple[float, float] = (0.85, 1.25)
    lora_rank: int = 16
    osf_rho: float = 0.5
    adam8_bytes: float = 2.03  # two uint8 states plus two fp32 absmax per 256-element block


DEFAULT_CALIB = Calib()


@dataclass(frozen=True)
class Machine:
    name: str
    vram_gib: float | None
    host_ram_gib: float | None
    source: str
    bf16: bool | None = None

    def limit_gib(self, c: Calib) -> float | None:
        return None if self.vram_gib is None else c.usable * self.vram_gib


@dataclass(frozen=True)
class RouteEstimate:
    route: str
    method: str
    gpu_gib: float
    low_gib: float
    high_gib: float
    host_gib: float
    verdict: Verdict
    reason: str | None
    smallest_card: str | None
    smallest_card_borderline: bool
    label: str | None


@dataclass(frozen=True)
class AutoChoice:
    method: str | None
    route: str | None
    rule_id: str
    reason: str


@dataclass(frozen=True)
class Estimate:
    model: str
    source: str
    model_type: str
    p_total: int
    p_targets: int
    n_targets: int
    p_embed: int
    tied: bool
    tokens: int
    seq_len: int
    micro_batch: int
    r: float
    precision: str
    machine: Machine
    routes: tuple[RouteEstimate, ...]
    auto: AutoChoice
    requested: str | None
    calib: Calib = field(default_factory=Calib)
    label: str = "computed from shapes, not yet checked against hardware (calibration is Gate 1)"

    def route(self, name: str) -> RouteEstimate:
        for r in self.routes:
            if r.route == name:
                return r
        raise KeyError(name)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---- formula ----------------------------------------------------------------------------


def bases_bytes(mats: tuple[Matrix, ...], r: float, form: str = "tail") -> int:
    """fp32 bases. Tail form stores, per side, whichever of head or tail is smaller."""
    total = 0
    for m, n in mats:
        k = int(r * min(m, n))
        if form == "head":
            total += (m + n) * k
        else:
            total += m * min(k, m - k) + n * min(k, n - k)
    return 4 * total


def activation_bytes(s: Shapes, tokens: int, act_bytes: int = 2, c: Calib = DEFAULT_CALIB) -> float:
    """Checkpointing on: one saved input per layer plus live tensors, and logits with grad."""
    hidden = tokens * s.hidden * act_bytes * (s.layers + c.live_tensors)
    logits = 2 * tokens * s.vocab * act_bytes
    return hidden + logits


def lora_params(mats: tuple[Matrix, ...], rank: int) -> int:
    return sum((m + n) * rank for m, n in mats)


def osf_bytes(mats: tuple[Matrix, ...], rho: float, dt: int = 2) -> int:
    """PEFT OSF as coded: base W, factors, initial copies, factor grads and Adam, in bf16."""
    total = 0
    for m, n in mats:
        rank = min(m, n)
        k = int(rho * rank)
        factors = (m + n + 1) * (rank - k)
        total += m * n * dt + factors * dt * 3 + factors * 2 * dt
        if m != n:
            total += (m + n) * k * dt  # high bases are stored only for non-square matrices
    return total


def route_bytes(s: Shapes, route: str, r: float, c: Calib = DEFAULT_CALIB) -> tuple[float, float]:
    """(GPU bytes before overheads, host bytes) for one route."""
    p = s.p_total
    p_t = s.p_targets
    if route == "lora-4bit":
        return 0.5 * p_t + 2 * s.p_embed + 16 * lora_params(s.targets, c.lora_rank), 0.0
    if route == "lora-16bit":
        return 2 * p + 16 * lora_params(s.targets, c.lora_rank), 0.0
    if route == "full":
        return 16 * p, 0.0
    if route == "osf":
        return osf_bytes(s.targets, c.osf_rho) + 2 * s.p_embed, 0.0
    b_all = bases_bytes(s.targets, r)
    b_blk = bases_bytes(s.block, r)
    w_ref, w_ref_blk = 4 * p_t, 4 * s.p_block
    table = {
        "sst/gpu": (16 * p + b_all + w_ref, 0.0),
        "sst/host-bases": (16 * p + b_blk + w_ref_blk, b_all + w_ref),
        "sst/8bit-adam": ((8 + c.adam8_bytes) * p + b_blk + w_ref_blk, b_all + w_ref),
        "sst/offload-state": (8 * p + b_blk + w_ref_blk, 8 * p + b_all + w_ref),
        "sst/zero-offload": (4 * p + b_blk + w_ref_blk, 12 * p + b_all + w_ref),
    }
    if route not in table:
        raise ConfigError(f'Invalid value for --route: "{route}". Choose from {", ".join(ROUTES)}.')
    gpu, host = table[route]
    return float(gpu), float(host)


def route_gib(
    s: Shapes,
    route: str,
    *,
    r: float = 0.99,
    tokens: int = 1024,
    act_bytes: int = 2,
    c: Calib = DEFAULT_CALIB,
) -> tuple[float, float]:
    """(V_gpu GiB, V_host GiB) for one route."""
    gpu, host = route_bytes(s, route, r, c)
    a = activation_bytes(s, tokens, act_bytes, c)
    return c.reserve_gib + c.frag * (gpu + a) / GIB, host / GIB


def verdict(
    gpu_gib: float, host_gib: float, m: Machine, c: Calib = DEFAULT_CALIB
) -> tuple[Verdict, str | None]:
    """Fit against a machine.

    A point estimate above the limit does not fit. A point estimate inside the limit
    whose +25% band crosses it is borderline: it runs when named explicitly, with a
    warning, and method="auto" skips it. Underestimating crashes mid-run, which is
    worse than a cautious refusal.
    """
    limit = m.limit_gib(c)
    host_short = m.host_ram_gib is not None and host_gib > m.host_ram_gib
    if limit is None:
        return "not checked", "no GPU to check against" + (
            "; host RAM is also short" if host_short else ""
        )
    if host_short:
        return "does not fit", (
            f"needs {host_gib:.1f} GiB of host RAM; {m.host_ram_gib:.1f} GiB available"
        )
    if gpu_gib > limit:
        return "does not fit", f"{gpu_gib:.1f} GiB against a {limit:.1f} GiB limit"
    if gpu_gib * c.band[1] > limit:
        return "borderline", (
            f"range reaches {gpu_gib * c.band[1]:.1f} GiB against a {limit:.1f} GiB limit"
        )
    return "fits", None


def _smallest(gpu_gib: float, c: Calib) -> tuple[str | None, bool]:
    card = gpus.smallest_card(gpu_gib, c.usable)
    if card is None:
        return None, False
    return card.name, gpu_gib * c.band[1] > c.usable * card.vram_gib


# ---- inputs ------------------------------------------------------------------------------


def _check_inputs(
    *,
    method: str,
    route: str | None,
    r: float,
    seq_len: int,
    micro_batch: int,
    nproc: int,
    gpu: str | None,
    vram: float | None,
    host_ram: float | None,
) -> None:
    if route is not None and method != "auto":
        raise ConfigError("Invalid value for --route: pass either --method or --route, not both.")
    if method == "freeze":
        raise ConfigError(
            'Invalid value for --method: "freeze". freeze is estimated from v0.1, once '
            "target selection exists."
        )
    if method not in METHODS:
        raise ConfigError(
            f'Invalid value for --method: "{method}". Choose from {", ".join(METHODS)}.'
        )
    if route is not None and route not in ROUTES:
        raise ConfigError(f'Invalid value for --route: "{route}". Choose from {", ".join(ROUTES)}.')
    if not 0.0 < r < 1.0:
        raise ConfigError(f'Invalid value for --r: "{r}". r must be between 0 and 1, exclusive.')
    if not 1 <= seq_len <= 1_048_576:
        raise ConfigError(
            f'Invalid value for --seq-len: "{seq_len}". It must be between 1 and 1048576.'
        )
    if micro_batch < 1:
        raise ConfigError(
            f'Invalid value for --micro-batch: "{micro_batch}". It must be 1 or more.'
        )
    if nproc != 1:
        raise ConfigError(
            f'Invalid value for --nproc: "{nproc}". 0.0.1 estimates a single GPU only; '
            "sharded estimates arrive with FSDP2 (Gate 3)."
        )
    if gpu is not None and vram is not None:
        raise ConfigError("Invalid value for --vram: pass either --gpu or --vram, not both.")
    if gpu is not None and gpus.lookup(gpu) is None:
        raise ConfigError(
            f'Invalid value for --gpu: "{gpu}". Known cards: {", ".join(gpus.card_names())}. '
            "For anything else, pass --vram GIB."
        )
    if vram is not None and not 0 < vram <= 4096:
        raise ConfigError(f'Invalid value for --vram: "{vram}". It must be between 0 and 4096 GiB.')
    if host_ram is not None and not 0 < host_ram <= 65536:
        raise ConfigError(
            f'Invalid value for --host-ram: "{host_ram}". It must be between 0 and 65536 GiB.'
        )


def resolve_machine(
    gpu: str | None = None, vram: float | None = None, host_ram: float | None = None
) -> Machine:
    """The machine to price: a named card, a VRAM size, or this machine via the probe."""
    if gpu is not None:
        card = gpus.lookup(gpu)
        if card is None:
            raise ConfigError(
                f'Invalid value for --gpu: "{gpu}". Known cards: {", ".join(gpus.card_names())}. '
                "For anything else, pass --vram GIB."
            )
        bf16 = card.vram_gib != 15  # the T4 is the only listed card without native bf16
        return Machine(card.name, card.vram_gib, host_ram, "--gpu", bf16)
    if vram is not None:
        return Machine(f"{vram:g} GiB GPU", vram, host_ram, "--vram", None)
    from tailtune.hardware.probe import probe

    hw = probe()
    ram = host_ram if host_ram is not None else hw.host_ram_available_gib
    if not hw.gpus:
        return Machine("this machine (no GPU found)", None, ram, "doctor", None)
    g = hw.gpus[0]
    bf16 = hw.real_bf16
    if bf16 is None and g.capability is not None:
        bf16 = g.capability >= (8, 0)
    return Machine(g.name, g.total_gib, ram, "doctor", bf16)


def _requested_routes(method: str, route: str | None) -> tuple[str, ...]:
    if route is not None:
        return (route,)
    if method == "auto":
        return ROUTES
    return tuple(r for r in ROUTES if ROUTE_METHOD[r] == method)


def choose_auto(
    routes: dict[str, RouteEstimate], p_total: int, r: float, *, host_checked: bool = True
) -> AutoChoice:
    """method="auto": the strongest protection whose verdict is "fits" (07 section 15)."""
    usable = [n for n in AUTO_ORDER if n in routes]
    if not usable:
        return AutoChoice(None, None, "A0", "method=auto was not evaluated for this selection.")
    if all(routes[n].verdict == "not checked" for n in usable):
        return AutoChoice(
            None,
            None,
            "A1",
            "method=auto needs a GPU to decide. Pass --gpu NAME or --vram GIB to plan for one.",
        )
    chosen = next((n for n in usable if routes[n].verdict == "fits"), None)
    if chosen is None:
        last = routes[usable[-1]]
        card = last.smallest_card or "no single card up to 288 GB"
        return AutoChoice(
            None,
            None,
            "A7",
            f"method=auto -> nothing fits here. The lightest route, {last.route}, needs about "
            f"{last.gpu_gib:.1f} GiB; the smallest single card is {card}.",
        )
    names = {
        "sst/gpu": f"sst (r={r:g}, fixed basis, all on GPU)",
        "sst/host-bases": f"sst (r={r:g}, fixed basis, bases streamed from host RAM)",
        "osf": "osf (PEFT OSF, 50% preserved)",
        "lora-16bit": "lora on a 16-bit base",
        "lora-4bit": "lora on a 4-bit base",
    }
    parts = [f"method=auto -> {names[chosen]}."]
    # The strongest method that was skipped, quoted at its lightest auto route.
    skipped = usable[: usable.index(chosen)]
    stronger = [n for n in skipped if ROUTE_METHOD[n] != ROUTE_METHOD[chosen]]
    if stronger:
        top = ROUTE_METHOD[stronger[0]]
        sk = min(
            (routes[n] for n in stronger if ROUTE_METHOD[n] == top), key=lambda row: row.gpu_gib
        )
        if sk.verdict == "borderline":
            parts.append(
                f"{top} ({sk.route}) is borderline here: about {sk.gpu_gib:.0f} GiB, with a "
                f"range up to {sk.high_gib:.0f} GiB."
            )
        else:
            card = sk.smallest_card or "no single card up to 288 GB"
            parts.append(
                f"{top} needs about {sk.gpu_gib:.0f} GiB of GPU memory for this model; "
                f"the smallest single card is {card}."
            )
    else:
        parts.append("This is the strongest method available.")
    if routes[chosen].host_gib > 0 and not host_checked:
        parts.append(
            f"It also needs about {routes[chosen].host_gib:.0f} GiB of host RAM, which was not "
            "checked: pass --host-ram GIB."
        )
    if ROUTE_METHOD[chosen] in {"sst", "osf"} and p_total < SMALL_MODEL_PARAMS:
        parts.append(
            "Constrained methods protect small models least: in SST's published results "
            "(Chavan et al., 2026), Science dropped 8.0 points at 1.7B."
        )
    return AutoChoice(ROUTE_METHOD[chosen], chosen, "A2" if stronger else "A3", " ".join(parts))


def estimate_shapes(
    s: Shapes,
    *,
    model: str = "model",
    source: str = "config.json",
    method: str = "auto",
    route: str | None = None,
    r: float = 0.99,
    seq_len: int = 1024,
    micro_batch: int = 1,
    machine: Machine | None = None,
    calib: Calib | None = None,
) -> Estimate:
    """Estimate every requested route for already-resolved shapes."""
    c = calib or Calib()
    m = machine or Machine("unspecified", None, None, "none")
    tokens = seq_len * micro_batch
    rows: dict[str, RouteEstimate] = {}
    # auto needs its candidates priced even when the table shows a subset.
    names = _requested_routes(method, route)
    for name in dict.fromkeys((*names, *AUTO_ORDER)):
        gpu, host = route_gib(s, name, r=r, tokens=tokens, c=c)
        v, why = verdict(gpu, host, m, c)
        card, card_border = _smallest(gpu, c)
        rows[name] = RouteEstimate(
            route=name,
            method=ROUTE_METHOD[name],
            gpu_gib=gpu,
            low_gib=gpu * c.band[0],
            high_gib=gpu * c.band[1],
            host_gib=host,
            verdict=v,
            reason=why,
            smallest_card=card,
            smallest_card_borderline=card_border,
            label=ROUTE_LABEL.get(name),
        )
    auto = choose_auto(rows, s.p_total, r, host_checked=m.host_ram_gib is not None)
    precision = (
        "bf16 autocast, fp32 master weights"
        if m.bf16 is not False
        else "fp16 autocast, fp32 master weights (no native bf16)"
    )
    return Estimate(
        model=model,
        source=source,
        model_type=s.model_type,
        p_total=s.p_total,
        p_targets=s.p_targets,
        n_targets=len(s.targets),
        p_embed=s.p_embed,
        tied=s.tied,
        tokens=tokens,
        seq_len=seq_len,
        micro_batch=micro_batch,
        r=r,
        precision=precision,
        machine=m,
        routes=tuple(rows[n] for n in names),
        auto=auto,
        requested=route or (None if method == "auto" else method),
        calib=c,
    )


def estimate(
    model: str,
    *,
    method: str = "auto",
    route: str | None = None,
    r: float = 0.99,
    seq_len: int = 1024,
    micro_batch: int = 1,
    nproc: int = 1,
    gpu: str | None = None,
    vram: float | None = None,
    host_ram: float | None = None,
    revision: str | None = None,
) -> Estimate:
    """Read a model's config (no weights) and estimate memory per route.

    Raises ConfigError (exit 5) for bad inputs, ModelAccessError or RefusedError
    (exit 4) for models tailtune will not estimate, and HubUnavailableError (exit 1).
    """
    _check_inputs(
        method=method,
        route=route,
        r=r,
        seq_len=seq_len,
        micro_batch=micro_batch,
        nproc=nproc,
        gpu=gpu,
        vram=vram,
        host_ram=host_ram,
    )
    src = load_config(model, revision)
    shapes = shapes_from_config(src.config, src.ref)
    machine = resolve_machine(gpu, vram, host_ram)
    return estimate_shapes(
        shapes,
        model=src.label,
        source="config.json (no weights downloaded)",
        method=method,
        route=route,
        r=r,
        seq_len=seq_len,
        micro_batch=micro_batch,
        machine=machine,
    )


def refusal_for(est: Estimate) -> DoesNotFitError | None:
    """The exit-4 error when an explicitly requested method or route cannot fit."""
    if est.requested is None:
        return None
    if any(row.verdict in {"fits", "borderline", "not checked"} for row in est.routes):
        return None
    lightest = min(est.routes, key=lambda row: row.gpu_gib)
    limit = est.machine.limit_gib(est.calib) or 0.0
    card = lightest.smallest_card or "no single card up to 288 GB"
    if lightest.reason and "host RAM" in lightest.reason:
        message = f"{est.requested} does not fit: {lightest.route} {lightest.reason}."
    else:
        message = (
            f"{est.requested} does not fit: it needs about {lightest.gpu_gib:.1f} GiB of GPU "
            f"memory for this model, and this GPU's limit is {limit:.1f} GiB."
        )
    return DoesNotFitError(
        message,
        cause="The route's estimate exceeds the limit (computed, not measured).",
        fix=f"Use --method auto, a smaller model, or a card with at least {card}.",
    )
