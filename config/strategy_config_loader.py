import json
from pathlib import Path
from copy import deepcopy

CONFIG_PATH = Path(__file__).parent / "strategy_configs.json"


class StrategyConfigLoader:
    def __init__(self):
        self._configs = self._load()
        self._defaults = self._extract_defaults()

    # ------------------------------------------------------------------
    # Internal loading
    # ------------------------------------------------------------------

    def _load(self):
        if CONFIG_PATH.exists():
            try:
                with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}

    def _extract_defaults(self):
        """
        Extract the __defaults__ block if present.
        If not present, treat the current configs as defaults.
        """
        if "__defaults__" in self._configs:
            return deepcopy(self._configs["__defaults__"])

        # If no defaults block exists, treat current configs as defaults
        defaults = deepcopy(self._configs)
        self._configs["__defaults__"] = defaults
        self._atomic_save()
        return defaults

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(self, strategy_name: str) -> dict:
        """
        Return a deep copy so callers cannot mutate internal state accidentally.
        """
        return deepcopy(self._configs.get(strategy_name, {}))

    def get_defaults(self, strategy_name: str = None):
        """
        Return default configs.
        If strategy_name is None, return all defaults.
        """
        if strategy_name is None:
            return deepcopy(self._defaults)
        return deepcopy(self._defaults.get(strategy_name, {}))

    def update(self, strategy_name: str, new_values: dict):
        """
        Update config for a strategy and save atomically.
        """
        self._configs[strategy_name] = new_values
        self._atomic_save()

    def reset(self, strategy_name: str):
        """
        Reset a single strategy to its default values.
        """
        if strategy_name in self._defaults:
            self._configs[strategy_name] = deepcopy(self._defaults[strategy_name])
            self._atomic_save()

    def reset_all(self):
        """
        Reset all strategies to defaults.
        """
        for name, values in self._defaults.items():
            self._configs[name] = deepcopy(values)
        self._atomic_save()

    # ------------------------------------------------------------------
    # Atomic save
    # ------------------------------------------------------------------

    def _atomic_save(self):
        tmp_path = CONFIG_PATH.with_suffix(".tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(self._configs, f, indent=2)
        tmp_path.replace(CONFIG_PATH)
