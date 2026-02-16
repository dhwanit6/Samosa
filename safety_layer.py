"""
Rule-based safety layer for Gujarati / Roman Gujarati / English chat.

This layer is lightweight and deterministic for edge deployment.
"""
from __future__ import annotations

from dataclasses import dataclass
import re


@dataclass
class SafetyConfig:
    block_threshold: float = 0.70
    monitor_threshold: float = 0.35


@dataclass
class SafetyDecision:
    blocked: bool
    score: float
    categories: list[str]
    message: str


_PATTERNS: dict[str, list[str]] = {
    "self_harm": [
        r"\bkill myself\b",
        r"\bsuicide\b",
        r"\bself harm\b",
        r"\bmari ja(va|u)\b",
        r"\bpotane nuksan\b",
    ],
    "violence": [
        r"\bmake bomb\b",
        r"\bbuild bomb\b",
        r"\bhow to kill\b",
        r"\bweapon\b",
        r"\bhathiyar\b",
        r"\bmari nakh\b",
    ],
    "hate": [
        r"\bhate (them|him|her)\b",
        r"\bgenocide\b",
        r"\bkill all\b",
        r"\bcleanse\b",
    ],
    "illegal": [
        r"\bhack\b",
        r"\bphishing\b",
        r"\bfraud\b",
        r"\bsteal password\b",
        r"\bcredit card dump\b",
    ],
    "sexual_minors": [
        r"\bchild porn\b",
        r"\bminor sex\b",
        r"\bunderage\b.*\bsex\b",
    ],
}

_WEIGHTS = {
    "self_harm": 0.95,
    "violence": 0.95,
    "hate": 0.90,
    "illegal": 0.80,
    "sexual_minors": 1.00,
}


def _detect_categories(text: str) -> list[str]:
    low = text.lower()
    hits: list[str] = []
    for cat, pats in _PATTERNS.items():
        for p in pats:
            if re.search(p, low):
                hits.append(cat)
                break
    return hits


def _score_from_categories(categories: list[str]) -> float:
    if not categories:
        return 0.0
    return min(1.0, max(_WEIGHTS.get(c, 0.5) for c in categories))


def safe_refusal(language_mode: str = "gujlish") -> str:
    mode = language_mode.lower().strip()
    if mode == "gujarati":
        return (
            "માફ કરશો, આ પ્રકારની વિનંતીમાં હું મદદ કરી શકતો નથી. "
            "હું સુરક્ષિત અને કલ્યાણકારી માર્ગદર્શન આપી શકું છું."
        )
    if mode == "roman":
        return (
            "Maaf karjo, aa prakar ni vinanti ma hu madad kari shakto nathi. "
            "Hu surakshit ane hitkari margdarshan api shakun chu."
        )
    return (
        "Maaf karjo, hu aa request ma direct madad kari shakto nathi. "
        "Pan hu safe ane positive alternative ma jarur help karish."
    )


def evaluate_text(text: str, cfg: SafetyConfig) -> SafetyDecision:
    categories = _detect_categories(text)
    score = _score_from_categories(categories)
    blocked = score >= float(cfg.block_threshold)
    if blocked:
        msg = "blocked_high_risk"
    elif score >= float(cfg.monitor_threshold):
        msg = "monitor_medium_risk"
    else:
        msg = "allow_low_risk"
    return SafetyDecision(
        blocked=blocked,
        score=float(score),
        categories=categories,
        message=msg,
    )

