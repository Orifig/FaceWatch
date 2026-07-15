"""
face_database.py
Face enrollment database — stores embeddings, matches faces
"""

import json
import os
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Optional
import logging

logger = logging.getLogger("face_database")

_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(_DIR, "face_database.json")
RECOGNITION_THRESHOLD = 0.95


class FaceDatabase:
    def __init__(self, db_path: str = DB_PATH):
        self.db_path = Path(db_path)
        self.data = self._load()

    def _load(self) -> dict:
        if self.db_path.exists():
            with open(self.db_path, "r") as f:
                return json.load(f)
        return {}

    def reload(self):
        """Reload database from disk — call after enrollment to pick up new faces."""
        self.data = self._load()

    def _save(self):
        with open(self.db_path, "w") as f:
            json.dump(self.data, f, indent=2)

    def add_person(self, name: str) -> bool:
        if name in self.data:
            return False
        self.data[name] = {
            "label_id": len(self.data),
            "embeddings": [],
            "timestamps": [],
            "created": datetime.now().isoformat(),
            "sample_count": 0
        }
        self._save()
        logger.info(f"Added person: {name}")
        return True

    def add_embedding(self, name: str, embedding: np.ndarray) -> bool:
        if name not in self.data:
            self.add_person(name)

        self.data[name]["embeddings"].append(embedding.tolist())
        self.data[name]["timestamps"].append(datetime.now().isoformat())
        self.data[name]["sample_count"] = len(self.data[name]["embeddings"])
        self._save()
        return True

    def delete_person(self, name: str) -> bool:
        if name not in self.data:
            return False
        del self.data[name]
        self._save()
        return True

    def get_mean_embedding(self, name: str) -> Optional[np.ndarray]:
        if name not in self.data or not self.data[name]["embeddings"]:
            return None
        embeddings = np.array(self.data[name]["embeddings"])
        mean = np.mean(embeddings, axis=0)
        norm = np.linalg.norm(mean)
        if norm > 0:
            mean = mean / norm
        return mean

    def match(self, embedding: np.ndarray, threshold: float = RECOGNITION_THRESHOLD):
        """
        Match embedding against all enrolled faces.
        Returns (name, similarity) or (None, best_similarity)
        """
        best_name = None
        best_sim = 0.0

        for name in self.data:
            mean_emb = self.get_mean_embedding(name)
            if mean_emb is None:
                continue

            sim = float(np.dot(embedding, mean_emb) /
                        (np.linalg.norm(embedding) * np.linalg.norm(mean_emb) + 1e-8))

            if sim > best_sim:
                best_sim = sim
                best_name = name if sim >= threshold else None

        return best_name, best_sim

    def list_people(self) -> list:
        return [
            {
                "name": name,
                "samples": self.data[name]["sample_count"],
                "created": self.data[name]["created"]
            }
            for name in self.data
        ]

    def get_stats(self) -> dict:
        return {
            "total_people": len(self.data),
            "total_embeddings": sum(p["sample_count"] for p in self.data.values())
        }


# Global database instance
db = FaceDatabase()
