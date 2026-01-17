# C:\mikebot\minions\neural_decision_layer.py

from __future__ import annotations
import logging
from dataclasses import dataclass, replace
from typing import List, Dict, Any, Union
import numpy as np

# Import BlendedDecision since that is what flows through here now
from .multi_agent import BlendedDecision
from .minions_base import MinionDecision

logger = logging.getLogger(__name__)

@dataclass
class NeuralDecisionConfig:
    smoothing_window: int = 10
    min_confidence: float = 0.3
    max_memory: int = 500

class NeuralDecisionLayer:
    """
    Temporal smoothing + confidence shaping for multi-agent outputs.
    Updated to handle BlendedDecision objects.
    """

    def __init__(self, config: NeuralDecisionConfig) -> None:
        self.config = config
        self._history: List[Dict[str, Any]] = []

    def process(self, decision: Union[BlendedDecision, MinionDecision]) -> Union[BlendedDecision, MinionDecision]:
        # Skip smoothing if it's a hold or invalid
        if decision.score == 0.0 and decision.action == "hold":
            return decision

        # Extract current raw values
        score = decision.score
        conf = decision.confidence

        self._history.append({"score": score, "confidence": conf})
        if len(self._history) > self.config.max_memory:
            self._history = self._history[-self.config.max_memory :]

        smoothed_score = self._smoothed_score()
        effective_conf = self._effective_confidence()

        # Gate low-confidence decisions
        if effective_conf < self.config.min_confidence:
            # Return a neutralized copy
            if isinstance(decision, BlendedDecision):
                return replace(decision, action="hold", score=0.0, confidence=effective_conf)
            # Fallback for MinionDecision
            return replace(decision, action="hold", score=0.0, confidence=effective_conf)

        # Update metadata and score
        new_meta = decision.meta.copy()
        new_meta["smoothed_score"] = smoothed_score
        new_meta["effective_confidence"] = effective_conf
        
        # Determine new action based on smoothed score
        new_action = decision.action
        if abs(smoothed_score) < 0.1:
            new_action = "hold"
        elif smoothed_score > 0:
            new_action = "long"
        else:
            new_action = "short"

        if isinstance(decision, BlendedDecision):
            return replace(decision, action=new_action, score=smoothed_score, confidence=effective_conf, meta=new_meta)
        
        return replace(decision, action=new_action, score=smoothed_score, confidence=effective_conf, meta=new_meta)

    def _smoothed_score(self) -> float:
        window = self._history[-self.config.smoothing_window :]
        scores = np.array([h["score"] for h in window], dtype=float)
        return float(scores.mean()) if len(scores) else 0.0

    def _effective_confidence(self) -> float:
        window = self._history[-self.config.smoothing_window :]
        confs = np.array([h["confidence"] for h in window], dtype=float)
        return float(np.clip(confs.mean(), 0.0, 1.0)) if len(confs) else 0.0