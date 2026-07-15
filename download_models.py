#!/usr/bin/env python3
"""
download_models.py
Downloads YuNet detector model.
Place your trained face_encoder.pth in ./models/ when ready.
"""

import urllib.request
import sys
from pathlib import Path

MODELS_DIR = Path("./models")
MODELS_DIR.mkdir(exist_ok=True)

YUNET_URL = (
    "https://github.com/opencv/opencv_zoo/raw/master/"
    "models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
)
YUNET_PATH = MODELS_DIR / "face_detection_yunet_2023mar.onnx"


def download(url, path, name):
    if path.exists():
        print(f"✓ {name} already exists ({path.stat().st_size // 1024}KB)")
        return True

    print(f"Downloading {name}...")

    def progress(count, block_size, total_size):
        pct = int(count * block_size * 100 / total_size)
        sys.stdout.write(f"\r  {pct}%")
        sys.stdout.flush()

    try:
        urllib.request.urlretrieve(url, path, reporthook=progress)
        print(f"\n✓ {name} downloaded ({path.stat().st_size // 1024}KB)")
        return True
    except Exception as e:
        print(f"\n✗ Failed: {e}")
        return False


if __name__ == "__main__":
    print("=== FaceWatch Model Downloader ===\n")

    download(YUNET_URL, YUNET_PATH, "YuNet face detector")

    encoder_path = MODELS_DIR / "face_encoder.pth"
    if encoder_path.exists():
        print(f"✓ face_encoder.pth found ({encoder_path.stat().st_size // 1024}KB)")
    else:
        print(f"\n⚠  face_encoder.pth not found in ./models/")
        print(f"   The app will run with a fallback encoder until your trained model is ready.")
        print(f"   When training is done, copy face_encoder.pth to ./models/")
        print(f"   Then click 'Reload model' in the dashboard header.")

    print("\n✓ Done! Run: python3 app.py")
