# mikebot/config/strategy_config.py

from __future__ import annotations
import json
from pathlib import Path
from typing import Dict

CONFIG_PATH = Path(__file__).parent / "strategy_toggles.json"

def load_strategy_toggles() -> Dict[str, bool]:
    if CONFIG_PATH.exists():
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}

def save_strategy_toggles(toggles: Dict[str, bool]) -> None:
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(toggles, f, indent=2)