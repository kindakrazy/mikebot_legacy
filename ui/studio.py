# mikebot/ui/studio.py

from __future__ import annotations

from pathlib import Path
import tkinter as tk
from tkinter import ttk

from mikebot.ui.data_tab import DataTab
from mikebot.ui.developer_tab import DeveloperSettingsTab
from mikebot.ui.lineage_tab import LineageTab
from mikebot.ui.strategies_tab import StrategiesTab
from mikebot.ui.strategy_config_tab import StrategyConfigTab

from mikebot.ui.system_state_tab import SystemStateTab
from mikebot.ui.strategy_engine_tab import StrategyEngineTab
from mikebot.ui.model_learning_tab import ModelLearningTab
from mikebot.ui.experience_tab import ExperienceTab

from mikebot.config.strategy_config import load_strategy_toggles


def build_notebook(root: tk.Misc) -> ttk.Notebook:
    notebook = ttk.Notebook(root)
    notebook.pack(fill="both", expand=True)

    # Repo root (mikebot/)
    here = Path(__file__).resolve()
    repo_root = here.parent.parent

    # Data tab
    data_tab = DataTab(notebook)
    notebook.add(data_tab, text="Data")

    # Strategies + Strategy Config
    toggles = load_strategy_toggles()
    strategies_tab = StrategiesTab(notebook, toggles)
    notebook.add(strategies_tab.frame, text="Strategy Toggles")

    strat_cfg_tab = StrategyConfigTab(notebook)
    notebook.add(strat_cfg_tab.frame, text="Strategy Parameters")

    # Lineage
    lineage_tab = LineageTab(notebook, repo_root)
    notebook.add(lineage_tab, text="Lineage")

    # Developer Settings
    dev_tab = DeveloperSettingsTab(notebook)
    notebook.add(dev_tab, text="Developer")

    # New V4 Cockpit Tabs
    system_tab = SystemStateTab(notebook)
    notebook.add(system_tab, text="System")

    engine_tab = StrategyEngineTab(notebook)
    notebook.add(engine_tab, text="Engine")

    learning_tab = ModelLearningTab(notebook)
    notebook.add(learning_tab, text="Learning")

    experience_tab = ExperienceTab(notebook)
    notebook.add(experience_tab, text="Experience")

    return notebook


def main() -> None:
    root = tk.Tk()
    root.title("Mikebot Studio – V4 Cockpit")
    root.geometry("1600x900")

    style = ttk.Style(root)
    try:
        style.theme_use("vista")
    except Exception:
        pass

    build_notebook(root)

    root.mainloop()


if __name__ == "__main__":
    main()