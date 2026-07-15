"""
app.py
Main FastAPI backend — recognition, enrollment, streaming, headcount
"""

import cv2
import asyncio
import json
import numpy as np
import threading
import time
import logging
import base64
from pathlib import Path
from datetime import datetime
from collections import deque
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

from face_engine import engine
from face_database import db
from camera_manager import camera_manager
from ptz_controller import ptz_manager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("app")

# ============================================================================
# APP STATE
# ============================================================================

class AppState:
    def __init__(self):
        self.recognition_results = {}   # {camera_id: [DetectionResult]}
        self.headcount = {}             # {camera_id: int}
        self.total_headcount = 0
        self.enrollment_state = None    # None or EnrollmentSession
        self.active_camera = "cam1"
        self.websocket_clients = set()
        self.lock = threading.Lock()
        self.running = True
        self.settings = {
            "recognition_threshold": 0.95,
            "detection_threshold": 0.6
        }

    def update_results(self, camera_id: str, detections: list):
        with self.lock:
            self.recognition_results[camera_id] = detections
            self.headcount[camera_id] = len(detections)
            self.total_headcount = sum(self.headcount.values())


class EnrollmentSession:
    def __init__(self, name: str, camera_id: str, target_samples: int = 20):
        self.name = name
        self.camera_id = camera_id
        self.target_samples = target_samples
        self.captured = 0
        self.countdown = 5     # Countdown seconds before capture starts
        self.countdown_start = time.time()
        self.capturing = False
        self.complete = False
        self.last_capture_time = 0


state = AppState()

# ============================================================================
# RECOGNITION LOOP
# ============================================================================

def recognition_loop():
    """
    Background thread: continuously process all camera frames.
    """
    logger.info("Recognition loop started")
    loop_count = 0
    last_db_reload = 0

    while state.running:
        # Reload database every 2 seconds to pick up new enrollments
        now = time.time()
        if now - last_db_reload > 2.0:
            db.reload()
            last_db_reload = now

    while state.running:
        for cam_id in list(camera_manager.streams.keys()):
            if (state.enrollment_state and
                state.enrollment_state.camera_id == cam_id and
                not state.enrollment_state.capturing):
                continue

            frame = camera_manager.get_frame(cam_id)
            if frame is None or frame.sum() == 0:
                continue

            detections_raw = engine.detect_faces(frame)

            results = []
            for det in detections_raw:
                face_crop = engine.crop_face(frame, det)
                if face_crop is None:
                    continue

                embedding = engine.encode_face(face_crop)
                name, similarity = db.match(embedding, threshold=state.settings["recognition_threshold"])

                # Handle enrollment capture
                if (state.enrollment_state and
                    state.enrollment_state.camera_id == cam_id and
                    state.enrollment_state.capturing and
                    state.enrollment_state.captured < state.enrollment_state.target_samples):

                    now = time.time()
                    if now - state.enrollment_state.last_capture_time >= 0.3:
                        db.add_embedding(state.enrollment_state.name, embedding)
                        state.enrollment_state.captured += 1
                        state.enrollment_state.last_capture_time = now

                        if state.enrollment_state.captured >= state.enrollment_state.target_samples:
                            state.enrollment_state.complete = True
                            state.enrollment_state.capturing = False
                            ptz_manager.resume_after_enrollment(cam_id)
                            db.reload()  # Instantly pick up new face
                            logger.info(f"Enrollment complete: {state.enrollment_state.name}")

                results.append({
                    "x": det["x"],
                    "y": det["y"],
                    "w": det["w"],
                    "h": det["h"],
                    "confidence": round(det["confidence"], 3),
                    "name": name or "Unknown",
                    "similarity": round(similarity, 3),
                    "recognized": name is not None
                })

            state.update_results(cam_id, results)

            # Log every 100 loops so we can see it's working
            loop_count += 1
            if loop_count % 100 == 0:
                logger.info(f"Recognition loop alive — cam:{cam_id} faces:{len(results)} db_people:{len(db.data)}")

        time.sleep(0.05)


# ============================================================================
# FASTAPI APP
# ============================================================================

app = FastAPI(title="FaceWatch")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)


@app.on_event("startup")
async def startup():
    ptz_manager.start_all_sweeps()

    # Start recognition loop as proper daemon thread
    recog_thread = threading.Thread(target=recognition_loop, daemon=True)
    recog_thread.start()

    logger.info("FaceWatch started")


@app.on_event("shutdown")
async def shutdown():
    state.running = False
    camera_manager.stop_all()


# ============================================================================
# VIDEO STREAMING
# ============================================================================

def generate_mjpeg(camera_id: str):
    """Generate annotated MJPEG stream for a camera."""
    while True:
        frame = camera_manager.get_frame(camera_id)

        # Draw detections
        with state.lock:
            detections = state.recognition_results.get(camera_id, [])

        for det in detections:
            x, y, w, h = det["x"], det["y"], det["w"], det["h"]
            name = det["name"]
            recognized = det["recognized"]
            sim = det["similarity"]

            # Color: green=recognized, white=unknown
            color = (0, 255, 120) if recognized else (200, 200, 200)
            border = 2 if recognized else 1

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, border)

            label = f"{name} {sim:.0%}" if recognized else "Unknown"
            cv2.putText(frame, label, (x, y - 8),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)

        # Enrollment countdown overlay
        if (state.enrollment_state and
            state.enrollment_state.camera_id == camera_id and
            not state.enrollment_state.complete):

            elapsed = time.time() - state.enrollment_state.countdown_start
            remaining = max(0, state.enrollment_state.countdown - elapsed)

            if not state.enrollment_state.capturing:
                # Countdown overlay
                overlay = frame.copy()
                cv2.rectangle(overlay, (0, 0), (frame.shape[1], frame.shape[0]),
                             (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

                count_text = str(int(remaining) + 1) if remaining > 0 else "GO"
                cv2.putText(frame, count_text,
                           (frame.shape[1] // 2 - 60, frame.shape[0] // 2 + 40),
                           cv2.FONT_HERSHEY_SIMPLEX, 6, (255, 255, 255), 8, cv2.LINE_AA)

                name_text = f"Enrolling: {state.enrollment_state.name}"
                cv2.putText(frame, name_text,
                           (20, frame.shape[0] - 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

                if remaining <= 0:
                    state.enrollment_state.capturing = True
            else:
                # Capture progress bar
                progress = state.enrollment_state.captured / state.enrollment_state.target_samples
                bar_w = int(frame.shape[1] * progress)
                cv2.rectangle(frame, (0, frame.shape[0] - 8),
                             (bar_w, frame.shape[0]), (0, 255, 120), -1)

                prog_text = f"Capturing {state.enrollment_state.captured}/{state.enrollment_state.target_samples}"
                cv2.putText(frame, prog_text, (20, frame.shape[0] - 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

        # Encode MJPEG
        ret, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
        if not ret:
            continue

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" +
            jpeg.tobytes() +
            b"\r\n"
        )

        time.sleep(1.0 / 25)


@app.get("/stream/{camera_id}")
async def video_stream(camera_id: str):
    return StreamingResponse(
        generate_mjpeg(camera_id),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/api/status")
async def get_status():
    return {
        "timestamp": datetime.now().isoformat(),
        "headcount": state.headcount,
        "total_headcount": state.total_headcount,
        "cameras": camera_manager.get_all_status(),
        "ptz": ptz_manager.get_all_status(),
        "model": engine.get_model_info(),
        "people": db.list_people(),
        "enrollment": {
            "active": state.enrollment_state is not None and not (state.enrollment_state.complete if state.enrollment_state else True),
            "name": state.enrollment_state.name if state.enrollment_state else None,
            "captured": state.enrollment_state.captured if state.enrollment_state else 0,
            "target": state.enrollment_state.target_samples if state.enrollment_state else 0,
            "complete": state.enrollment_state.complete if state.enrollment_state else False,
            "capturing": state.enrollment_state.capturing if state.enrollment_state else False,
        } if state.enrollment_state else {"active": False}
    }


@app.get("/api/detections/{camera_id}")
async def get_detections(camera_id: str):
    with state.lock:
        return {
            "camera_id": camera_id,
            "detections": state.recognition_results.get(camera_id, []),
            "headcount": state.headcount.get(camera_id, 0)
        }


class EnrollRequest(BaseModel):
    name: str
    camera_id: str
    samples: int = 20


@app.post("/api/enroll/start")
async def start_enrollment(req: EnrollRequest):
    if state.enrollment_state and not state.enrollment_state.complete:
        raise HTTPException(400, "Enrollment already in progress")

    if not req.name.strip():
        raise HTTPException(400, "Name cannot be empty")

    # Pause PTZ on selected camera
    ptz_manager.pause_for_enrollment(req.camera_id)

    # Start enrollment session
    state.enrollment_state = EnrollmentSession(
        name=req.name.strip(),
        camera_id=req.camera_id,
        target_samples=req.samples
    )

    db.add_person(req.name.strip())

    return {"status": "countdown_started", "name": req.name, "countdown": 5}


@app.post("/api/enroll/cancel")
async def cancel_enrollment():
    if state.enrollment_state:
        name = state.enrollment_state.name
        cam_id = state.enrollment_state.camera_id
        ptz_manager.resume_after_enrollment(cam_id)
        # Remove if no samples captured
        if state.enrollment_state.captured == 0:
            db.delete_person(name)
        state.enrollment_state = None
    return {"status": "cancelled"}


class EnrollFrameRequest(BaseModel):
    image_b64: str  # base64 encoded JPEG from browser webcam
    name: str


@app.post("/api/enroll/frame")
async def enroll_frame(req: EnrollFrameRequest):
    """
    Accept a single frame from the browser webcam.
    Detect face, encode it, add to database.
    Returns detection result.
    """
    import base64

    if not req.name.strip():
        raise HTTPException(400, "Name required")

    # Decode base64 image
    try:
        img_data = base64.b64decode(req.image_b64.split(",")[-1])
        np_arr = np.frombuffer(img_data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    except Exception as e:
        raise HTTPException(400, f"Invalid image: {e}")

    if frame is None:
        raise HTTPException(400, "Could not decode image")

    # Detect faces
    detections = engine.detect_faces(frame)
    if not detections:
        return {"status": "no_face", "captured": 0}

    # Use largest face
    largest = max(detections, key=lambda d: d["w"] * d["h"])
    face_crop = engine.crop_face(frame, largest)
    if face_crop is None:
        return {"status": "crop_failed", "captured": 0}

    # Encode and store
    embedding = engine.encode_face(face_crop)
    if req.name not in db.data:
        db.add_person(req.name)
    db.add_embedding(req.name, embedding)
    db.reload()  # Instantly available for recognition

    count = db.data[req.name]["sample_count"]
    return {"status": "captured", "captured": count, "confidence": round(float(largest["confidence"]), 3)}


@app.delete("/api/person/{name:path}")
async def delete_person(name: str):
    from urllib.parse import unquote
    name = unquote(name)
    if not db.delete_person(name):
        raise HTTPException(404, "Person not found")
    return {"status": "deleted", "name": name}


@app.get("/api/people")
async def get_people():
    return {"people": db.list_people(), "stats": db.get_stats()}


class SettingsRequest(BaseModel):
    recognition_threshold: Optional[float] = None
    detection_threshold: Optional[float] = None


@app.get("/api/settings")
async def get_settings():
    return {
        "recognition_threshold": state.settings.get("recognition_threshold", 0.95),
        "detection_threshold": state.settings.get("detection_threshold", 0.6)
    }


@app.post("/api/settings")
async def update_settings(req: SettingsRequest):
    if req.recognition_threshold is not None:
        val = round(max(0.1, min(1.0, req.recognition_threshold)), 2)
        state.settings["recognition_threshold"] = val
    if req.detection_threshold is not None:
        val = round(max(0.1, min(1.0, req.detection_threshold)), 2)
        state.settings["detection_threshold"] = val
        if engine.detector:
            try:
                engine.detector.setScoreThreshold(val)
            except Exception:
                pass
    return await get_settings()


@app.post("/api/model/reload")
async def reload_model():
    """Hot-swap trained model without restarting server."""
    engine.reload_encoder()
    return {"status": "reloaded", "model_info": engine.get_model_info()}


@app.get("/api/cameras")
async def get_cameras():
    return {"cameras": camera_manager.list_cameras()}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    state.websocket_clients.add(websocket)
    try:
        while True:
            with state.lock:
                data = {
                    "total_headcount": state.total_headcount,
                    "headcount": state.headcount,
                    "detections": state.recognition_results,
                    "enrollment": {
                        "active": bool(state.enrollment_state and not getattr(state.enrollment_state, 'complete', True)),
                        "captured": getattr(state.enrollment_state, 'captured', 0),
                        "target": getattr(state.enrollment_state, 'target_samples', 0),
                        "capturing": getattr(state.enrollment_state, 'capturing', False),
                        "name": getattr(state.enrollment_state, 'name', None),
                    }
                }
            await websocket.send_json(data)
            await asyncio.sleep(0.5)
    except WebSocketDisconnect:
        state.websocket_clients.discard(websocket)


@app.get("/")
async def root():
    html_path = Path("./dashboard.html")
    if html_path.exists():
        with open(html_path, "r") as f:
            return HTMLResponse(f.read())
    return HTMLResponse("<h1>FaceWatch — dashboard.html not found</h1>")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")


# ============================================================================
# SETUP PAGE ENDPOINTS
# ============================================================================

CONFIG_PATH = Path(__file__).parent / "config.json"


@app.get("/setup", response_class=HTMLResponse)
async def setup_page():
    setup_path = Path(__file__).parent / "setup_page.html"
    if setup_path.exists():
        with open(setup_path, "r") as f:
            return HTMLResponse(f.read())
    return HTMLResponse("<h1>setup_page.html not found</h1>")


@app.get("/api/setup/config")
async def get_config():
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, "r") as f:
            return json.load(f)
    raise HTTPException(404, "No config yet")


@app.post("/api/setup/save")
async def save_config(config: dict):
    with open(CONFIG_PATH, "w") as f:
        json.dump(config, f, indent=2)

    # Apply settings immediately
    if "settings" in config:
        s = config["settings"]
        if "recognition_threshold" in s:
            state.settings["recognition_threshold"] = float(s["recognition_threshold"])
        if "detection_threshold" in s:
            state.settings["detection_threshold"] = float(s["detection_threshold"])
            if engine.detector:
                try:
                    engine.detector.setScoreThreshold(float(s["detection_threshold"]))
                except Exception:
                    pass

    return {"status": "saved"}


@app.post("/api/setup/test-rtsp")
async def test_rtsp(body: dict):
    url = body.get("url", "")
    if not url:
        return {"ok": False, "error": "No URL provided"}
    try:
        import cv2 as _cv2
        cap = _cv2.VideoCapture(url, _cv2.CAP_FFMPEG)
        cap.set(_cv2.CAP_PROP_BUFFERSIZE, 1)
        ret, frame = cap.read()
        cap.release()
        if ret and frame is not None:
            h, w = frame.shape[:2]
            return {"ok": True, "resolution": f"{w}x{h}"}
        return {"ok": False, "error": "Stream opened but no frames"}
    except Exception as e:
        return {"ok": False, "error": str(e)}


@app.post("/api/setup/test-ptz")
async def test_ptz_connection(body: dict):
    try:
        from onvif import ONVIFCamera as OC
        cam = OC(body["ip"], body["port"], body["user"], body["password"])
        cam.create_ptz_service()
        return {"ok": True}
    except ImportError:
        return {"ok": False, "error": "onvif-zeep not installed — run: pip install onvif-zeep"}
    except Exception as e:
        return {"ok": False, "error": str(e)}
