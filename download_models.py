#!/usr/bin/env python3
"""
download_models.py
------------------
Downloads the pre-trained banana ripeness model weights from the
ParanKafleUTS/AI_Matrix_banana_ripness GitHub repository.

Usage:
    python download_models.py                  # download all three models
    python download_models.py --model efficientnet
    python download_models.py --model mobilenet
    python download_models.py --model resnet
"""

import argparse
import os
import sys
import requests

# ---------------------------------------------------------------------------
# Raw download URLs (GitHub LFS / raw content)
# ---------------------------------------------------------------------------
BASE_URL = (
    "https://raw.githubusercontent.com/"
    "ParanKafleUTS/AI_Matrix_banana_ripness/main/"
)

MODELS = {
    "efficientnet": {
        "filename": "EfficientNetB0_banana_ripeness.pth",
        "url": BASE_URL + "EfficientNetB0_banana_ripeness.pth",
        "size_mb": 15.6,
    },
    "mobilenet": {
        "filename": "MobileNetV3_banana_ripeness.pth",
        "url": BASE_URL + "MobileNetV3_banana_ripeness.pth",
        "size_mb": 16.3,
    },
    "resnet": {
        "filename": "ResNet50_banana_ripeness.pth",
        "url": BASE_URL + "ResNet50_banana_ripeness.pth",
        "size_mb": 90.1,
    },
}

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")


def download_file(url: str, dest_path: str, size_mb: float) -> None:
    """Stream-download *url* to *dest_path* with a simple progress indicator."""
    print(f"  Downloading → {dest_path}  (~{size_mb:.1f} MB)")
    response = requests.get(url, stream=True, timeout=120)
    response.raise_for_status()

    total = int(response.headers.get("content-length", 0))
    downloaded = 0
    chunk_size = 1024 * 256  # 256 KB

    with open(dest_path, "wb") as fh:
        for chunk in response.iter_content(chunk_size=chunk_size):
            if chunk:
                fh.write(chunk)
                downloaded += len(chunk)
                if total:
                    pct = downloaded / total * 100
                    print(f"\r  Progress: {pct:.1f}%", end="", flush=True)

    print()  # newline after progress


def download_models(names: list) -> None:
    os.makedirs(MODELS_DIR, exist_ok=True)

    for name in names:
        info = MODELS[name]
        dest = os.path.join(MODELS_DIR, info["filename"])

        if os.path.exists(dest):
            print(f"[SKIP] {info['filename']} already exists.")
            continue

        print(f"[DOWNLOADING] {info['filename']}")
        try:
            download_file(info["url"], dest, info["size_mb"])
            print(f"[OK]  Saved to {dest}\n")
        except requests.HTTPError as exc:
            print(f"[ERROR] HTTP error for {name}: {exc}", file=sys.stderr)
            sys.exit(1)
        except (requests.RequestException, OSError) as exc:
            print(f"[ERROR] {exc}", file=sys.stderr)
            sys.exit(1)

    print("All requested models are ready in:", MODELS_DIR)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download banana-ripeness model weights."
    )
    parser.add_argument(
        "--model",
        choices=list(MODELS.keys()),
        default=None,
        help="Which model to download (default: all).",
    )
    args = parser.parse_args()

    names = [args.model] if args.model else list(MODELS.keys())
    download_models(names)


if __name__ == "__main__":
    main()
