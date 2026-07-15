"""
camera_manager.py
Multi-camera manager — RTSP streams via OpenCV
"""

import cv2
import threading
import numpy as np
import time
import logging

logger = logging.getLogger("camera_manager")

# ============================================================================
# CAMERA SOURCES — update with your camera IPs and credentials
# ============================================================================

CAMERA_SOURCES = {
    "cam1": {
        "name": "Camera 1",
        "source": "rtsp://USERNAME:PASSWORD@CAMERA_IP:554/stream1",
        "fallback": 0  # USB webcam index if RTSP fails
    },
    # Add more cameras:
    # "cam2": {
    #     "name": "Camera 2",
    #     "source": "rtsp://USERNAME:PASSWORD@CAMERA_IP_2:554/stream1",
    #     "fallback": 1
    # },
}

DISPLAY_WIDTH = 1280
DISPLAY_HEIGHT = 720
TARGET_FPS = 15


class CameraStream:
    def __init__(self, camera_id: str, config: dict):
        self.camera_id = camera_id
        self.config = config
        self.cap = None
        self.frame = None
        self.running = False
        self._lock = threading.Lock()
        self._thread = None
        self.fps = 0
        self.connected = False
        self._last_frame_time = 0
        self._connect()

    def _connect(self):
        source = self.config.get("source", "")
        if source.startswith("rtsp://"):
            cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            if cap.isOpened():
                ret, _ = cap.read()
                if ret:
                    self.cap = cap
                    self.connected = True
                    logger.info(f"✓ RTSP connected: {self.camera_id}")
                    return
                cap.release()

        fallback = self.config.get("fallback", 0)
        logger.warning(f"RTSP failed for {self.camera_id}, trying USB {fallback}")
        cap = cv2.VideoCapture(fallback)
        if cap.isOpened():
            self.cap = cap
            self.connected = True
            logger.info(f"✓ USB connected: {self.camera_id}")
        else:
            logger.error(f"No camera for {self.camera_id}")
            self.connected = False

    def start(self):
        if not self.connected:
            return
        self.running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

    def _capture_loop(self):
        frame_count = 0
        start_time = time.time()

        while self.running:
            if not self.cap or not self.cap.isOpened():
                self.connected = False
                break

            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.05)
                continue

            frame = cv2.resize(frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT),
                             interpolation=cv2.INTER_LINEAR)

            with self._lock:
                self.frame = frame
                self._last_frame_time = time.time()

            frame_count += 1
            elapsed = time.time() - start_time
            if elapsed >= 1.0:
                self.fps = round(frame_count / elapsed, 1)
                frame_count = 0
                start_time = time.time()

    def get_frame(self) -> np.ndarray:
        with self._lock:
            if self.frame is None:
                return np.zeros((DISPLAY_HEIGHT, DISPLAY_WIDTH, 3), dtype=np.uint8)
            return self.frame.copy()

    def stop(self):
        self.running = False
        if self.cap:
            self.cap.release()
        self.connected = False

    def get_status(self) -> dict:
        stale = (time.time() - self._last_frame_time) > 3.0 if self._last_frame_time else True
        return {
            "camera_id": self.camera_id,
            "name": self.config["name"],
            "connected": self.connected and not stale,
            "fps": self.fps
        }


class CameraManager:
    def __init__(self):
        self.streams = {}
        self._init_cameras()

    def _init_cameras(self):
        for cam_id, config in CAMERA_SOURCES.items():
            stream = CameraStream(cam_id, config)
            stream.start()
            self.streams[cam_id] = stream

    def get_frame(self, camera_id: str) -> np.ndarray:
        if camera_id not in self.streams:
            return np.zeros((DISPLAY_HEIGHT, DISPLAY_WIDTH, 3), dtype=np.uint8)
        return self.streams[camera_id].get_frame()

    def get_all_status(self) -> list:
        return [s.get_status() for s in self.streams.values()]

    def list_cameras(self) -> list:
        return [
            {"id": cam_id, "name": config["name"]}
            for cam_id, config in CAMERA_SOURCES.items()
        ]

    def stop_all(self):
        for stream in self.streams.values():
            stream.stop()


camera_manager = CameraManager()


def _load_config_sources():
    """Load camera sources from config.json if it exists, else use defaults above."""
    import json
    from pathlib import Path
    config_path = Path(__file__).parent / "config.json"
    if config_path.exists():
        try:
            with open(config_path) as f:
                cfg = json.load(f)
            if cfg.get("cameras"):
                return cfg["cameras"]
        except Exception:
            pass
    return CAMERA_SOURCES


# Override with config.json if present
CAMERA_SOURCES = _load_config_sources()
