"""Matrix shapes from a model's config.json.

This is the config fallback from docs/plan/07-hardware.md section 15. It follows
the shape arithmetic behind every budget in section 9 (the research notes'
budget scripts). Reading safetensors headers, which also reveals fused and
quantized tensors, replaces it at v0.1.

Only architectures whose arithmetic has been checked are accepted. Anything else
is refused rather than estimated wrongly.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, NoReturn

from tailtune._core.errors import (
    RemoteCodeRequiredError,
    TargetTypeError,
    UnsupportedModelError,
)

Matrix = tuple[int, int]

# Decoder-only families with the seven standard projections per block
# (q, k, v, o, gate, up, down). Counts were checked against safetensors headers
# on the Hub (tests/test_shapes_network.py).
DENSE_TYPES = frozenset({"llama", "mistral", "qwen2", "qwen3"})
MOE_TYPES = frozenset({"qwen3_moe"})
SUPPORTED_TYPES = DENSE_TYPES | MOE_TYPES


@dataclass(frozen=True)
class Shapes:
    """Shapes of one model, as tailtune's memory model needs them."""

    model_type: str
    targets: tuple[Matrix, ...]
    """Every target matrix. A [E, m, n] expert tensor counts as E matrices."""
    block: tuple[Matrix, ...]
    """Target matrices of the largest decoder block (streamed one at a time)."""
    embed: tuple[Matrix, ...]
    """Input embedding, plus the output head when it is not tied."""
    hidden: int
    layers: int
    vocab: int
    tied: bool
    experts: int | None = None

    @property
    def p_targets(self) -> int:
        return sum(m * n for m, n in self.targets)

    @property
    def p_block(self) -> int:
        return sum(m * n for m, n in self.block)

    @property
    def p_embed(self) -> int:
        return sum(m * n for m, n in self.embed)

    @property
    def p_total(self) -> int:
        """All parameters except norm and bias vectors (well under 0.1% here)."""
        return self.p_targets + self.p_embed


def _int(cfg: Mapping[str, Any], key: str) -> int:
    value = cfg.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise UnsupportedModelError(
            f'config.json has no usable "{key}" (found {value!r}), so tailtune cannot read '
            "this model's shapes."
        )
    return value


def _opt_int(cfg: Mapping[str, Any], key: str) -> int | None:
    value = cfg.get(key)
    return value if isinstance(value, int) and not isinstance(value, bool) and value > 0 else None


def _refuse_unsupported(ref: str, model_type: object, cfg: Mapping[str, Any]) -> NoReturn:
    if "auto_map" in cfg and model_type not in SUPPORTED_TYPES:
        raise RemoteCodeRequiredError(
            f"{ref} needs custom code from its repository, and tailtune does not run remote "
            "code by default.",
            cause="Its config.json has an auto_map entry and a model_type Transformers does "
            "not ship (D-08).",
            fix="Use a model supported natively by Transformers.",
        )
    names = ", ".join(sorted(SUPPORTED_TYPES))
    raise UnsupportedModelError(
        f'tailtune cannot read the shapes of "{model_type}" models yet.',
        cause=f"0.0.1 estimates only architectures whose arithmetic has been checked: {names}.",
        fix="Open a model-support issue with the model id. Hybrid, multimodal and MLA "
        "architectures arrive with safetensors-header reading at v0.1.",
    )


def shapes_from_config(cfg: Mapping[str, Any], ref: str = "this model") -> Shapes:
    """Build Shapes from a config.json mapping, or refuse with a tailtune error."""
    quant = cfg.get("quantization_config")
    if quant:
        method = quant.get("quant_method", "unknown") if isinstance(quant, Mapping) else "unknown"
        raise TargetTypeError(
            f"{ref} stores quantized weights ({method}). Constrained methods cannot train "
            "quantized weights, and 0.0.1 estimates only unquantized checkpoints.",
            cause="config.json has a quantization_config entry (D-15).",
            fix="Use the model's bf16 release.",
        )
    if "text_config" in cfg or "vision_config" in cfg:
        raise UnsupportedModelError(
            f"{ref} is a composite (multimodal) model, which 0.0.1 does not estimate.",
            cause="Its vision or audio tower is not described by the text config, so the "
            "parameter count would be too low.",
            fix="Use a text-only release of the model, or wait for v0.1.",
        )
    model_type = cfg.get("model_type")
    if not isinstance(model_type, str) or model_type not in SUPPORTED_TYPES:
        _refuse_unsupported(ref, model_type, cfg)

    hidden = _int(cfg, "hidden_size")
    layers = _int(cfg, "num_hidden_layers")
    heads = _int(cfg, "num_attention_heads")
    kv_heads = _opt_int(cfg, "num_key_value_heads") or heads
    head_dim = _opt_int(cfg, "head_dim") or hidden // heads
    vocab = _int(cfg, "vocab_size")
    q_width = heads * head_dim
    kv_width = kv_heads * head_dim
    attention: list[Matrix] = [
        (q_width, hidden),  # q_proj
        (kv_width, hidden),  # k_proj
        (kv_width, hidden),  # v_proj
        (hidden, q_width),  # o_proj
    ]

    experts: int | None = None
    blocks: list[list[Matrix]] = []
    if model_type in MOE_TYPES:
        experts = _int(cfg, "num_experts")
        moe_ffn = _int(cfg, "moe_intermediate_size")
        step = _opt_int(cfg, "decoder_sparse_step") or 1
        mlp_only = set(cfg.get("mlp_only_layers") or [])
        dense_ffn = 0
        if mlp_only or step > 1:
            dense_ffn = _int(cfg, "intermediate_size")
        for i in range(layers):
            is_moe = i not in mlp_only and (i + 1) % step == 0
            if is_moe:
                ffn = [(moe_ffn, hidden), (moe_ffn, hidden), (hidden, moe_ffn)] * experts
                blocks.append([*attention, *ffn, (experts, hidden)])  # router
            else:
                blocks.append(
                    [*attention, (dense_ffn, hidden), (dense_ffn, hidden), (hidden, dense_ffn)]
                )
    else:
        ffn_width = _int(cfg, "intermediate_size")
        mlp: list[Matrix] = [(ffn_width, hidden), (ffn_width, hidden), (hidden, ffn_width)]
        blocks = [[*attention, *mlp] for _ in range(layers)]

    # Transformers' PretrainedConfig defaults tie_word_embeddings to True when absent.
    tied = bool(cfg.get("tie_word_embeddings", True))
    embed: list[Matrix] = [(vocab, hidden)] if tied else [(vocab, hidden), (vocab, hidden)]
    targets = tuple(m for block in blocks for m in block)
    largest = max(blocks, key=lambda b: sum(m * n for m, n in b))
    return Shapes(
        model_type=model_type,
        targets=targets,
        block=tuple(largest),
        embed=tuple(embed),
        hidden=hidden,
        layers=layers,
        vocab=vocab,
        tied=tied,
        experts=experts,
    )
