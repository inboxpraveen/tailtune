"""Named cards for `estimate --gpu` and the "smallest single card" column.

Sizes follow docs/plan/07-hardware.md sections 3 and 9.5, where a card's nominal
size is used as its GiB budget before the 92% usable rule. The T4 is listed at
its usable 15 rather than its nominal 16. AMD sizes are from press sources [U].
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Card:
    name: str
    vram_gib: float
    aliases: tuple[str, ...] = ()


CARDS: tuple[Card, ...] = (
    Card("8 GB consumer GPU", 8, ("8GB",)),
    Card("Tesla T4", 15, ("T4",)),
    Card("RTX 3090", 24, ("3090",)),
    Card("RTX 4090", 24, ("4090",)),
    Card("L4", 24),
    Card("A10", 24),
    Card("RTX 5090", 32, ("5090",)),
    Card("A100 40GB", 40, ("A100-40", "A10040GB", "A10040")),
    Card("L40S", 48),
    Card("A100 80GB", 80, ("A100", "A100-80", "A10080GB", "A10080")),
    Card("H100", 80, ("H100 80GB",)),
    Card("H100 NVL", 94),
    Card("RTX PRO 6000", 96, ("RTXPRO6000", "PRO6000")),
    Card("H200", 141),
    Card("B200", 180),
    Card("MI300X", 192),
    Card("MI325X", 256),
    Card("MI350X", 288),
    Card("MI355X", 288),
)

# The ladder for "smallest single card", one representative per size (07 section 9.5).
LADDER: tuple[Card, ...] = (
    CARDS[0],  # 8
    CARDS[1],  # T4 15
    Card("24 GB (RTX 4090, L4)", 24),
    CARDS[6],  # RTX 5090 32
    CARDS[7],  # A100 40GB
    CARDS[8],  # L40S 48
    Card("80 GB (H100, A100 80GB)", 80),
    CARDS[12],  # RTX PRO 6000 96
    CARDS[13],  # H200 141
    CARDS[14],  # B200 180
    Card("288 GB (MI355X class)", 288),
)


def _key(text: str) -> str:
    text = text.upper().replace("NVIDIA", "").replace("TESLA", "").replace("GEFORCE", "")
    return re.sub(r"[^A-Z0-9]", "", text)


def lookup(name: str) -> Card | None:
    """Find a card by name, ignoring case, spaces, hyphens and vendor prefixes."""
    wanted = _key(name)
    if not wanted:
        return None
    for card in CARDS:
        if wanted in {_key(card.name), *(_key(a) for a in card.aliases)}:
            return card
    return None


def smallest_card(gpu_gib: float, usable: float) -> Card | None:
    """The smallest card on the ladder whose usable memory holds `gpu_gib`."""
    for card in LADDER:
        if gpu_gib <= usable * card.vram_gib:
            return card
    return None


def card_names() -> list[str]:
    return [c.name for c in CARDS]
