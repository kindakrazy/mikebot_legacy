from __future__ import annotations

from typing import List

from mikebot.data.candle_schema import Candle
from mikebot.features.feature_builder_v4 import FeatureBuilderV4
from mikebot.features.feature_interface import Feature
from mikebot.features import raw_math_feature_v4 as raw_math_mod


def _resolve_raw_math_feature_cls() -> type[Feature]:
    """
    Resolve the raw math feature class in a tolerant way.

    Older snapshots may expose `RawMathFeature`, newer ones `RawMathFeatureV4`.
    """
    if hasattr(raw_math_mod, "RawMathFeatureV4"):
        return getattr(raw_math_mod, "RawMathFeatureV4")
    if hasattr(raw_math_mod, "RawMathFeature"):
        return getattr(raw_math_mod, "RawMathFeature")
    raise ImportError("No RawMathFeature(V4) class found in raw_math_feature_v4 module")


def default_feature_pipeline() -> List[Feature]:
    """
    Build the default feature pipeline used by TrainingPipelineV4.

    Returns a list of instantiated Feature objects.
    """
    raw_cls = _resolve_raw_math_feature_cls()
    raw_feature: Feature = raw_cls()  # type: ignore[call-arg]

    # You can extend this list with additional features as needed.
    return [raw_feature]


def build_feature_builder_v4() -> FeatureBuilderV4:
    """
    Convenience helper to construct a FeatureBuilderV4 wired to the default pipeline.
    """
    pipeline = default_feature_pipeline()
    return FeatureBuilderV4(features=pipeline, candle_type=Candle)