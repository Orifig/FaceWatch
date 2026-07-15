"""
ptz_controller.py
PTZ camera controller via ONVIF (tested with Tapo C216)
Ultra-slow patrol sweep — move, stop and scan, move again
"""

import threading
import time
import logging
from enum import Enum

logger = logging.getLogger("ptz_controller")

# ============================================================================
# CAMERA CONFIG — update with your camera IPs and credentials
# ============================================================================

CAMERAS = {
    "cam1": {
        "name": "Camera 1",
        "ip": "CAMERA_IP",       # e.g. 192.168.1.100
        "port": 2020,             # ONVIF port (Tapo default: 2020)
        "user": "USERNAME",       # Camera local username
        "password": "PASSWORD"    # Camera local password
    },
    # Add more cameras:
    # "cam2": {
    #     "name": "Camera 2",
    #     "ip": "CAMERA_IP_2",
    #     "port": 2020,
    #     "user": "USERNAME",
    #     "password": "PASSWORD"
    # },
}

# Sweep timing — tune these to your room size
SWEEP_STEP_DELAY = 4.0     # Seconds to pause and scan at each position
SWEEP_MOVE_TIME = 1.5      # Seconds of movement between scan stops
SWEEP_PAN_SPEED = 0.15     # Pan speed 0-1 (0.15 = slow)


class PTZState(Enum):
    SWEEPING = "sweeping"
    PAUSED = "paused"
    ENROLLING = "enrolling"
    IDLE = "idle"


class ONVIFCamera:
    def __init__(self, camera_id: str, config: dict):
        self.camera_id = camera_id
        self.config = config
        self.cam = None
        self.ptz = None
        self.media = None
        self.profile_token = None
        self.state = PTZState.IDLE
        self.running = False
        self._lock = threading.Lock()
        self._sweep_thread = None
        self._connect()

    def _connect(self):
        try:
            from onvif import ONVIFCamera as OC
            self.cam = OC(
                self.config["ip"],
                self.config["port"],
                self.config["user"],
                self.config["password"]
            )
            self.ptz = self.cam.create_ptz_service()
            self.media = self.cam.create_media_service()
            profiles = self.media.GetProfiles()
            self.profile_token = profiles[0].token
            logger.info(f"✓ ONVIF PTZ connected: {self.camera_id}")
        except ImportError:
            logger.warning("onvif-zeep not installed — run: pip install onvif-zeep")
            self.cam = None
        except Exception as e:
            logger.warning(f"ONVIF connect failed for {self.camera_id}: {e}")
            self.cam = None

    def _move(self, pan_speed: float, tilt_speed: float = 0.0):
        if self.ptz is None:
            return
        try:
            req = self.ptz.create_type("ContinuousMove")
            req.ProfileToken = self.profile_token
            req.Velocity = {
                "PanTilt": {"x": pan_speed, "y": tilt_speed},
                "Zoom": {"x": 0}
            }
            self.ptz.ContinuousMove(req)
        except Exception as e:
            logger.debug(f"PTZ move error: {e}")

    def _stop(self):
        if self.ptz is None:
            return
        try:
            req = self.ptz.create_type("Stop")
            req.ProfileToken = self.profile_token
            req.PanTilt = True
            req.Zoom = False
            self.ptz.Stop(req)
        except Exception as e:
            logger.debug(f"PTZ stop error: {e}")

    def start_sweep(self):
        with self._lock:
            if self.state == PTZState.SWEEPING:
                return
            self.state = PTZState.SWEEPING
            self.running = True
        self._sweep_thread = threading.Thread(target=self._sweep_loop, daemon=True)
        self._sweep_thread.start()
        logger.info(f"PTZ sweep started: {self.camera_id}")

    def _sweep_loop(self):
        direction = 1
        steps_per_side = 6

        # Start at left edge
        self._move(-SWEEP_PAN_SPEED)
        time.sleep(SWEEP_MOVE_TIME * steps_per_side)
        self._stop()
        time.sleep(SWEEP_STEP_DELAY * 2)

        step = 0

        while self.running:
            with self._lock:
                current_state = self.state

            if current_state != PTZState.SWEEPING:
                self._stop()
                time.sleep(0.3)
                continue

            # Move
            self._move(SWEEP_PAN_SPEED * direction)
            elapsed = 0
            while elapsed < SWEEP_MOVE_TIME and self.running:
                with self._lock:
                    if self.state != PTZState.SWEEPING:
                        self._stop()
                        break
                time.sleep(0.1)
                elapsed += 0.1

            # Stop and scan
            self._stop()
            elapsed = 0
            pause = SWEEP_STEP_DELAY * 2 if step >= steps_per_side - 1 else SWEEP_STEP_DELAY
            while elapsed < pause and self.running:
                with self._lock:
                    if self.state != PTZState.SWEEPING:
                        break
                time.sleep(0.1)
                elapsed += 0.1

            step += 1
            if step >= steps_per_side:
                step = 0
                direction *= -1

    def pause_sweep(self):
        with self._lock:
            self.state = PTZState.PAUSED
        self._stop()

    def resume_sweep(self):
        with self._lock:
            self.state = PTZState.SWEEPING

    def start_enrollment_pause(self):
        with self._lock:
            self.state = PTZState.ENROLLING
        self._stop()

    def end_enrollment_pause(self):
        with self._lock:
            self.state = PTZState.SWEEPING

    def stop(self):
        self.running = False
        with self._lock:
            self.state = PTZState.IDLE
        self._stop()

    def get_status(self) -> dict:
        return {
            "camera_id": self.camera_id,
            "name": self.config["name"],
            "state": self.state.value,
            "connected": self.cam is not None
        }


class PTZManager:
    def __init__(self):
        self.cameras = {
            cam_id: ONVIFCamera(cam_id, config)
            for cam_id, config in CAMERAS.items()
        }

    def start_all_sweeps(self):
        for cam in self.cameras.values():
            cam.start_sweep()

    def pause_for_enrollment(self, camera_id: str):
        if camera_id in self.cameras:
            self.cameras[camera_id].start_enrollment_pause()

    def resume_after_enrollment(self, camera_id: str):
        if camera_id in self.cameras:
            self.cameras[camera_id].end_enrollment_pause()

    def get_all_status(self) -> list:
        return [cam.get_status() for cam in self.cameras.values()]


ptz_manager = PTZManager()


def _load_ptz_config():
    """Load PTZ config from config.json if it exists, else use defaults above."""
    import json
    from pathlib import Path
    config_path = Path(__file__).parent / "config.json"
    if config_path.exists():
        try:
            with open(config_path) as f:
                cfg = json.load(f)
            if cfg.get("ptz"):
                return cfg["ptz"]
        except Exception:
            pass
    return CAMERAS


# Override with config.json if present
CAMERAS = _load_ptz_config()
