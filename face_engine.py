"""
face_engine.py
Core engine: YuNet detection + face encoder + cosine matching
Swap your trained model by replacing MODEL_PATH
"""

import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import Optional
import logging

logger = logging.getLogger("face_engine")

# ============================================================================
# MODEL PATH — SWAP THIS WHEN YOUR TRAINED MODEL IS READY
# ============================================================================

import os
_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(_DIR, "models", "face_encoder.pth")
YUNET_PATH = os.path.join(_DIR, "models", "face_detection_yunet_2023mar.onnx")

# Recognition threshold (cosine similarity, 0-1)
# Higher = stricter matching
# Fallback encoder needs HIGH threshold (0.85+) to avoid false matches
# Your trained model should use 0.55-0.65
RECOGNITION_THRESHOLD = 0.95


# ============================================================================
# FACE ENCODER MODEL
# Matches your trained architecture exactly
# ============================================================================

class FaceEncoder(nn.Module):
    """
    YOUR custom face encoder (triplet loss trained).
    512-dim embedding from 128x128 RGB input.
    Swap MODEL_PATH above when your trained model is ready.
    """
    def __init__(self, embedding_dim=512):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 64x64

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 32x32

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 16x16

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 8x8

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.fc = nn.Linear(256, embedding_dim)

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = F.normalize(x, p=2, dim=1)
        return x


# ============================================================================
# PRETRAINED FALLBACK (used while your model trains)
# Using OpenCV DNN as a zero-weight embedding fallback
# ============================================================================

class PretrainedFallbackEncoder:
    """
    Lightweight fallback using pixel statistics as embeddings.
    Used for testing before your trained model is ready.
    Replace MODEL_PATH with your face_encoder.pth to switch.
    """
    def __init__(self):
        self.embedding_dim = 512
        logger.warning("Using fallback encoder — swap MODEL_PATH with your trained model!")

    def encode(self, face_rgb: np.ndarray) -> np.ndarray:
        """Encode a 128x128 RGB face crop into 512-dim vector."""
        face = face_rgb.astype(np.float32) / 255.0

        # Build embedding from spatial statistics across regions
        h, w = face.shape[:2]
        features = []

        # 4x4 grid of mean/std per channel
        for row in range(4):
            for col in range(4):
                patch = face[
                    row * h // 4:(row + 1) * h // 4,
                    col * w // 4:(col + 1) * w // 4
                ]
                for c in range(3):
                    features.append(patch[:, :, c].mean())
                    features.append(patch[:, :, c].std())

        # Pad/truncate to 512
        features = np.array(features, dtype=np.float32)
        result = np.zeros(512, dtype=np.float32)
        n = min(len(features), 512)
        result[:n] = features[:n]

        # Normalize to unit sphere
        norm = np.linalg.norm(result)
        if norm > 0:
            result = result / norm

        return result


# ============================================================================
# FACE ENGINE
# ============================================================================

class FaceEngine:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.detector = None
        self.encoder = None
        self.use_torch = False

        self._load_detector()
        self._load_encoder()

    def _load_detector(self):
        """Load YuNet face detector."""
        yunet_path = Path(YUNET_PATH)

        if not yunet_path.exists():
            logger.warning(f"YuNet model not found at {yunet_path}")
            logger.warning("Run: python3 download_models.py")
            self.detector = None
            return

        self.detector = cv2.FaceDetectorYN.create(
            model=str(yunet_path),
            config="",
            input_size=(320, 320),
            score_threshold=0.6,
            nms_threshold=0.3,
            top_k=5000,
            backend_id=cv2.dnn.DNN_BACKEND_OPENCV,
            target_id=cv2.dnn.DNN_TARGET_CPU
        )
        logger.info("✓ YuNet detector loaded")

    def _load_encoder(self):
        """Load face encoder — your trained model or fallback."""
        model_path = Path(MODEL_PATH)

        if model_path.exists():
            try:
                model = FaceEncoder(embedding_dim=512)
                state_dict = torch.load(str(model_path), map_location=self.device)
                model.load_state_dict(state_dict)
                model.eval()
                model.to(self.device)
                self.encoder = model
                self.use_torch = True
                logger.info(f"✓ Trained encoder loaded from {model_path}")
            except Exception as e:
                logger.warning(f"Failed to load trained encoder: {e}")
                logger.warning("Using fallback encoder")
                self.encoder = PretrainedFallbackEncoder()
                self.use_torch = False
        else:
            logger.warning(f"No trained model at {model_path} — using fallback encoder")
            logger.warning("Place your face_encoder.pth in ./models/ to activate trained model")
            self.encoder = PretrainedFallbackEncoder()
            self.use_torch = False

    def reload_encoder(self):
        """
        Hot-swap encoder at runtime.
        Call this after placing your trained model in ./models/
        """
        self._load_encoder()

    def detect_faces(self, frame: np.ndarray) -> list:
        """
        Detect faces in a BGR frame using YuNet.
        Returns list of [x, y, w, h, confidence] dicts.
        """
        if self.detector is None:
            return []

        h, w = frame.shape[:2]
        self.detector.setInputSize((w, h))
        _, detections = self.detector.detect(frame)

        if detections is None:
            return []

        results = []
        for det in detections:
            x, y, fw, fh = int(det[0]), int(det[1]), int(det[2]), int(det[3])
            conf = float(det[14])
            # Filter out unreasonably small detections
            if fw < 20 or fh < 20:
                continue
            results.append({
                "x": x, "y": y, "w": fw, "h": fh,
                "confidence": conf
            })

        return results

    def crop_face(self, frame: np.ndarray, det: dict, margin: float = 0.15) -> Optional[np.ndarray]:
        """
        Crop and resize face to 128x128 RGB.
        """
        x, y, w, h = det["x"], det["y"], det["w"], det["h"]

        margin_px = int(margin * max(w, h))
        x1 = max(0, x - margin_px)
        y1 = max(0, y - margin_px)
        x2 = min(frame.shape[1], x + w + margin_px)
        y2 = min(frame.shape[0], y + h + margin_px)

        if x2 <= x1 or y2 <= y1:
            return None

        crop = frame[y1:y2, x1:x2]
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        face_128 = cv2.resize(crop_rgb, (128, 128), interpolation=cv2.INTER_LINEAR)

        return face_128

    def encode_face(self, face_rgb: np.ndarray) -> np.ndarray:
        """
        Encode 128x128 RGB face into 512-dim embedding.
        Works with both trained model and fallback.
        """
        if self.use_torch:
            tensor = torch.from_numpy(face_rgb).float() / 255.0
            tensor = tensor.permute(2, 0, 1).unsqueeze(0).to(self.device)
            with torch.no_grad():
                embedding = self.encoder(tensor)
            return embedding.squeeze(0).cpu().numpy()
        else:
            return self.encoder.encode(face_rgb)

    def cosine_similarity(self, v1: np.ndarray, v2: np.ndarray) -> float:
        """Compute cosine similarity between two unit vectors."""
        return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8))

    def is_trained_model_loaded(self) -> bool:
        """Check if trained model is active."""
        return self.use_torch

    def get_model_info(self) -> dict:
        """Return info about which model is active."""
        return {
            "trained_model_loaded": self.use_torch,
            "model_path": MODEL_PATH,
            "model_exists": Path(MODEL_PATH).exists(),
            "device": str(self.device)
        }


# Global engine instance
engine = FaceEngine()
