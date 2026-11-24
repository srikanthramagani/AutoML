"""
ModelExporter
=============
Responsibility: Serialize and save the final complete pipeline to disk using joblib.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional
import joblib


class ModelExporter:
    """
    Export a trained pipeline to disk as a single complete artifact.
    """

    def __init__(self, export_dir: str = "models", compress: int = 3) -> None:
        self.export_dir = export_dir
        self.compress = compress
        Path(export_dir).mkdir(parents=True, exist_ok=True)

    def export(
        self,
        pipeline: Any,
        model_name: str,
        metadata: Optional[Dict[str, Any]] = None,
        filename: str = "trained_pipeline",
    ) -> str:
        """
        Save the complete pipeline to a .pkl file and save a companion metadata JSON.
        """
        pkl_path = os.path.join(self.export_dir, f"{filename}.pkl")
        json_path = os.path.join(self.export_dir, f"{filename}_metadata.json")

        # Export pipeline
        joblib.dump(pipeline, pkl_path, compress=self.compress)

        # Export companion metadata
        meta = {
            "model_name": model_name,
            "artifact_file": f"{filename}.pkl",
            "exported_at": datetime.utcnow().isoformat() + "Z",
            **(metadata or {}),
        }
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        print(f"[ModelExporter] Saved trained pipeline -> {pkl_path}")
        print(f"[ModelExporter] Saved pipeline metadata -> {json_path}")
        return os.path.abspath(pkl_path)

    def load(self, pkl_path: str) -> Any:
        """
        Load an exported pipeline from disk.
        """
        return joblib.load(pkl_path)
