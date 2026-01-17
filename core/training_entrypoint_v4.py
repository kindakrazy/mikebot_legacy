from __future__ import annotations

from typing import Optional

from mikebot.core.training_pipeline_v4 import TrainingConfigV4, TrainingPipelineV4


def build_default_config() -> TrainingConfigV4:
    """
    Construct a default TrainingConfigV4.

    This keeps the entrypoint thin: all real defaults live in
    TrainingConfigV4 itself.
    """
    return TrainingConfigV4()


def run_training(config: Optional[TrainingConfigV4] = None) -> None:
    """
    High-level training entrypoint used by scripts and tooling.
    """
    cfg = config or build_default_config()
    pipeline = TrainingPipelineV4(cfg)
    pipeline.run()


if __name__ == "__main__":
    run_training()