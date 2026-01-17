# mikebot/models/model_saver_v4.py

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any, Optional


class ModelSaverV4:
    """
    Responsible for serializing and saving model objects for v4.

    This class is intentionally generic:
    - If the model has a `.save(path)` method (e.g., TensorFlow, PyTorch),
      it will be used.
    - Otherwise, the model is pickled.
    - Metadata is written alongside the model in a JSON file.

    The caller (training orchestrator or registry) decides where the model
    should be saved and provides the target directory.
    """

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def save_model(
        self,
        model: Any,
        model_id: str,
        metadata: Optional[dict] = None,
    ) -> Path:
        """
        Save a model and its metadata to disk.

        Parameters
        ----------
        model:
            The model object to serialize.

        model_id:
            Unique identifier for the model (UUID from registry).

        metadata:
            Optional metadata dict to save alongside the model.

        Returns
        -------
        Path to the saved model artifact.
        """
        model_dir = self.root / model_id
        model_dir.mkdir(parents=True, exist_ok=True)

        model_path = model_dir / "model.bin"
        meta_path = model_dir / "metadata.json"

        # Try framework-native save first
        if hasattr(model, "save") and callable(getattr(model, "save")):
            # e.g., TensorFlow, Keras, some PyTorch wrappers
            model.save(str(model_path))
        else:
            # Fallback: pickle
            with open(model_path, "wb") as f:
                pickle.dump(model, f)

        # Write metadata
        if metadata is None:
            metadata = {}

        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        return model_path

    # ------------------------------------------------------------------
    # Loading (optional convenience)
    # ------------------------------------------------------------------

    def load_model(self, model_id: str) -> Any:
        """
        Load a previously saved model.

        This uses pickle unless the model directory contains a framework-
        specific loader (not implemented here — v4 loader handles that).

        Returns
        -------
        The deserialized model object.
        """
        model_dir = self.root / model_id
        model_path = model_dir / "model.bin"

        if not model_path.exists():
            raise FileNotFoundError(f"Model artifact not found: {model_path}")

        with open(model_path, "rb") as f:
            return pickle.load(f)