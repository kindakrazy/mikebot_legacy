# mikebot/models/model_loader_v4.py

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any, Optional


class ModelLoaderV4:
    """
    Responsible for loading v4 model artifacts from disk.

    This loader is intentionally generic:

    - If a framework-specific loader exists (e.g., TensorFlow SavedModel),
      it will be used automatically.
    - Otherwise, the model is loaded via pickle.
    - Metadata is always loaded from metadata.json.

    The saver determines how the model was written; the loader mirrors it.
    """

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(
        self,
        model_id: str,
        *,
        expect_metadata: bool = True,
    ) -> tuple[Any, Optional[dict]]:
        """
        Load a model and its metadata.

        Parameters
        ----------
        model_id:
            The UUID of the model directory.

        expect_metadata:
            If True, metadata.json must exist. If False, metadata is optional.

        Returns
        -------
        (model_object, metadata_dict_or_None)
        """
        model_dir = self.root / model_id
        if not model_dir.exists():
            raise FileNotFoundError(f"Model directory not found: {model_dir}")

        model_path = model_dir / "model.bin"
        meta_path = model_dir / "metadata.json"

        # Load metadata
        metadata = None
        if meta_path.exists():
            with open(meta_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
        elif expect_metadata:
            raise FileNotFoundError(f"Metadata file missing: {meta_path}")

        # Framework-native loader?
        # (We check for known patterns but do not import heavy frameworks.)
        if self._is_tensorflow_saved_model(model_dir):
            return self._load_tensorflow_model(model_dir), metadata

        # Fallback: pickle
        if not model_path.exists():
            raise FileNotFoundError(f"Model artifact missing: {model_path}")

        with open(model_path, "rb") as f:
            model = pickle.load(f)

        return model, metadata

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _is_tensorflow_saved_model(self, model_dir: Path) -> bool:
        """
        Detect TensorFlow SavedModel format.
        """
        return (model_dir / "saved_model.pb").exists()

    def _load_tensorflow_model(self, model_dir: Path) -> Any:
        """
        Load a TensorFlow SavedModel without assuming TF is installed.

        If TensorFlow is unavailable, this raises a clear error.
        """
        try:
            import tensorflow as tf  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "TensorFlow model detected but TensorFlow is not installed."
            ) from e

        return tf.keras.models.load_model(str(model_dir))