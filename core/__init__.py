# mikebot/core/__init__.py

# Option A: completely empty, just mark as a package
# (safest for circulars)

# Option B: re-export *types only* if you really want convenience imports,
# but do NOT import submodules at import time.

# from .candle_engine import CandleEngine, SymbolRegistry
# from .feature_builder import FeatureBuilder, FeatureConfig
# from .meta_trainer import MetaTrainer