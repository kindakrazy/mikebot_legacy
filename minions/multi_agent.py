from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .minions_base import MinionDecision


# ---------------------------------------------------------------------------
# Blended decision output
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BlendedDecision:
    """
    Final fused decision produced by the multi-agent vote.

    This is the unified decision object consumed by the portfolio optimizer.
    It is intentionally simple, stable, and fully deterministic.
    """
    action: str               # "long", "short", "hold", "exit"
    score: float              # normalized blended score
    confidence: float         # blended confidence
    symbol: Optional[str]     # symbol targeted (if any)
    meta: Dict[str, float]    # per-minion weighted contributions


# ---------------------------------------------------------------------------
# Multi-agent blending logic
# ---------------------------------------------------------------------------

def blended_vote(
    decisions: List[MinionDecision],
    weights: Dict[str, float],
    personality: Optional[Any] = None,
) -> BlendedDecision:
    """
    Modern multi-agent vote.

    Inputs:
      - decisions: list of MinionDecision objects
      - weights: per-minion weight mapping from config
      - personality: optional personality profile that can bias scoring

    Output:
      - BlendedDecision (action, score, confidence, symbol, meta)
    """

    # No decisions → neutral hold
    if not decisions:
        return BlendedDecision(
            action="hold",
            score=0.0,
            confidence=0.0,
            symbol=None,
            meta={},
        )

    # Normalize weights
    total_weight = sum(weights.get(d.minion_name, 1.0) for d in decisions)
    if total_weight <= 0:
        total_weight = 1.0

    # Accumulators
    long_score = 0.0
    short_score = 0.0
    hold_score = 0.0
    exit_score = 0.0

    meta: Dict[str, float] = {}

    # ----------------------------------------------------------------------
    # Aggregate weighted scores
    # ----------------------------------------------------------------------
    for d in decisions:
        w = weights.get(d.minion_name, 1.0)
        contribution = d.score * w
        meta[d.minion_name] = contribution

        if d.action == "long":
            long_score += contribution
        elif d.action == "short":
            short_score += contribution
        elif d.action == "exit":
            exit_score += contribution
        else:
            hold_score += contribution

    # ----------------------------------------------------------------------
    # Personality shaping (optional)
    # ----------------------------------------------------------------------
    if personality is not None:
        # These attributes are intentionally generic; PersonalityManager
        # maps them to real behavioral profiles.
        long_score *= getattr(personality, "aggression", 1.0)
        short_score *= getattr(personality, "caution", 1.0)
        hold_score *= getattr(personality, "noise_tolerance", 1.0)
        exit_score *= getattr(personality, "risk_aversion", 1.0)

    # ----------------------------------------------------------------------
    # Determine final action
    # ----------------------------------------------------------------------
    scores = {
        "long": long_score,
        "short": short_score,
        "exit": exit_score,
        "hold": hold_score,
    }

    final_action = max(scores, key=scores.get)
    final_score = scores[final_action] / max(total_weight, 1e-9)

    # ----------------------------------------------------------------------
    # Confidence blending
    # ----------------------------------------------------------------------
    confidences = [
        d.confidence * weights.get(d.minion_name, 1.0)
        for d in decisions
    ]
    final_confidence = sum(confidences) / max(total_weight, 1e-9)

    # ----------------------------------------------------------------------
    # Symbol selection
    # ----------------------------------------------------------------------
    # If any minion targets a symbol, prefer that symbol.
    symbols = [d.symbol for d in decisions if d.symbol]
    final_symbol = symbols[0] if symbols else None

    return BlendedDecision(
        action=final_action,
        score=float(final_score),
        confidence=float(final_confidence),
        symbol=final_symbol,
        meta=meta,
    )