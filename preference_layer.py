"""
Preference alignment scoring layer.

Scores text for warmth, humility, harmony, and ecological respect.
Used as an inference-time reranking signal.
"""
from __future__ import annotations

from dataclasses import dataclass
import re


@dataclass
class PreferenceConfig:
    warmth_weight: float = 0.35
    humility_weight: float = 0.20
    harmony_weight: float = 0.20
    ecology_weight: float = 0.15
    toxicity_penalty_weight: float = 0.40


_WARMTH = (
    "help", "madad", "saath", "sahanubhuti", "samjan", "care", "kind", "respect",
    "samman", "krupa", "please", "thanks", "thank you",
)
_HUMILITY = (
    "let us", "sathe", "aapde", "we can", "vinamr", "seva", "without ego", "sathesathe",
)
_HARMONY = (
    "shanti", "peace", "santulan", "harmony", "sahkar", "dialogue", "samvad", "karuna",
)
_ECOLOGY = (
    "earth", "planet", "prakruti", "paryavaran", "ecology", "climate", "water", "vruksh",
    "species", "biodiversity",
)
_TOXIC = (
    "dominate", "destroy", "hate", "worthless", "crush", "revenge", "ego", "supremacy",
)


def _count_hits(text: str, terms: tuple[str, ...]) -> int:
    low = text.lower()
    return sum(1 for t in terms if t in low)


def score_preference_text(text: str, cfg: PreferenceConfig) -> float:
    """
    Returns score in roughly [-1, 1].
    """
    warmth = _count_hits(text, _WARMTH)
    humility = _count_hits(text, _HUMILITY)
    harmony = _count_hits(text, _HARMONY)
    ecology = _count_hits(text, _ECOLOGY)
    toxic = _count_hits(text, _TOXIC)

    # Mild penalty for shouting/punctuation aggression.
    punct_burst = len(re.findall(r"[!?]{3,}", text))
    toxic += punct_burst

    pos = (
        float(cfg.warmth_weight) * warmth
        + float(cfg.humility_weight) * humility
        + float(cfg.harmony_weight) * harmony
        + float(cfg.ecology_weight) * ecology
    )
    neg = float(cfg.toxicity_penalty_weight) * toxic
    raw = pos - neg

    # Squash to stable range.
    if raw >= 0:
        return min(1.0, raw / 4.0)
    return max(-1.0, raw / 4.0)

