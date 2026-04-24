# 🍌 Banana Ripeness Classification API

A production-ready REST API that classifies banana ripeness into **6 categories** using fine-tuned deep-learning models built with PyTorch and served with FastAPI.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Ripeness Classes](#2-ripeness-classes)
3. [Model Origin](#3-model-origin)
4. [Architecture & Performance](#4-architecture--performance)
5. [Repository Structure](#5-repository-structure)
6. [Prerequisites](#6-prerequisites)
7. [Installation](#7-installation)
8. [Downloading Model Weights](#8-downloading-model-weights)
9. [Running the API Server](#9-running-the-api-server)
10. [API Reference](#10-api-reference)
11. [Testing the API](#11-testing-the-api)
12. [How Inference Works](#12-how-inference-works)
13. [Troubleshooting](#13-troubleshooting)

---

## 1. Project Overview

This repository (**BANNF**) provides a lightweight, production-ready web service that wraps three pre-trained PyTorch image-classification models and exposes them via a single HTTP endpoint (`POST /predict`).

A client sends a banana image; the API returns:

- the **predicted ripeness class** (e.g. `ripe`, `overripe`),
- the **confidence** score for that class, and
- the full **probability distribution** across all six classes.

The API is built with **[FastAPI](https://fastapi.tiangolo.com/)** and served by **[Uvicorn](https://www.uvicorn.org/)**, making it easy to integrate into web applications, mobile back-ends, or agricultural IoT pipelines.

---

## 2. Ripeness Classes

The models were trained to distinguish the following six stages:

| Class | Description |
|---|---|
| `freshripe` | Fully ripe and freshly harvested |
| `freshunripe` | Freshly harvested but not yet ripe |
| `overripe` | Past peak ripeness; still edible but over-matured |
| `ripe` | At ideal ripeness for consumption |
| `rotten` | Spoiled; not suitable for consumption |
| `unripe` | Green banana, not yet ripe |

---

## 3. Model Origin

All three model weight files originate from the companion research repository:

> **[ParanKafleUTS/AI_Matrix_banana_ripness](https://github.com/ParanKafleUTS/AI_Matrix_banana_ripness)**

The models were trained in an AWS SageMaker environment using **transfer learning** (ImageNet pre-trained weights) with extensive data augmentation:

- Random resized crop (80–100 % zoom)
- Horizontal and vertical flips
- 90° / 180° rotations
- Random rotation ±15°
- Colour jitter (brightness, contrast, saturation, hue ±10 %)
- Gaussian blur (up to 1 px)

The dataset is the *banana-ripeness-dataset-original* split into **train / valid / test** folds.

---

## 4. Architecture & Performance

| Model Key | Architecture | Weights File | Test Accuracy | File Size |
|---|---|---|---|---|
| `efficientnet` | EfficientNet-B0 | `EfficientNetB0_banana_ripeness.pth` | **96.98 %** | ~15.6 MB |
| `mobilenet` | MobileNet-V3-Large | `MobileNetV3_banana_ripeness.pth` | 96.09 % | ~16.3 MB |
| `resnet` | ResNet-50 | `ResNet50_banana_ripeness.pth` | 96.26 % | ~90.1 MB |

All models replace the final classification head with a `Linear(in_features, 6)` layer.  
Training used **Adam** (lr = 0.001) for **10 epochs** with **CrossEntropyLoss**.

---

## 5. Repository Structure

```
BANNF/
├── main.py              # FastAPI application (the API server)
├── download_models.py   # Script to download .pth weights
├── requirements.txt     # Python package dependencies
├── models/              # Folder created by download_models.py
│   ├── EfficientNetB0_banana_ripeness.pth
│   ├── MobileNetV3_banana_ripeness.pth
│   └── ResNet50_banana_ripeness.pth
└── README.md            # This file
```

---

## 6. Prerequisites

| Requirement | Minimum Version |
|---|---|
| Python | 3.10+ |
| pip | 23+ |
| (Optional) CUDA | 11.8+ for GPU inference |

> **Note:** The API works perfectly on CPU. GPU inference is faster but not required.

---

## 7. Installation

```bash
# 1. Clone this repository
git clone https://github.com/ParanKafleUTS/BANNF.git
cd BANNF

# 2. Create and activate a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install all dependencies
pip install -r requirements.txt
```

---

## 8. Downloading Model Weights

The `.pth` files are stored in the source repository
[ParanKafleUTS/AI_Matrix_banana_ripness](https://github.com/ParanKafleUTS/AI_Matrix_banana_ripness).
Use the included helper script to fetch them:

### Download all three models (recommended)

```bash
python download_models.py
```

This creates a `models/` sub-directory and downloads all three weight files (~120 MB total).

### Download a single model

```bash
# Only EfficientNetB0 (smallest + best accuracy)
python download_models.py --model efficientnet

# Only MobileNetV3
python download_models.py --model mobilenet

# Only ResNet50 (largest file)
python download_models.py --model resnet
```

### Manual download (alternative)

If the script cannot reach GitHub directly, download the files manually and place them in `models/`:

```
models/EfficientNetB0_banana_ripeness.pth
models/MobileNetV3_banana_ripeness.pth
models/ResNet50_banana_ripeness.pth
```

Direct raw URLs:

```
https://raw.githubusercontent.com/ParanKafleUTS/AI_Matrix_banana_ripness/main/EfficientNetB0_banana_ripeness.pth
https://raw.githubusercontent.com/ParanKafleUTS/AI_Matrix_banana_ripness/main/MobileNetV3_banana_ripeness.pth
https://raw.githubusercontent.com/ParanKafleUTS/AI_Matrix_banana_ripness/main/ResNet50_banana_ripeness.pth
```

---

## 9. Running the API Server

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

The server starts at **http://localhost:8000**.

### Development mode (auto-reload on code changes)

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### Interactive API documentation

Once the server is running, open your browser:

| URL | Description |
|---|---|
| http://localhost:8000/docs | Swagger UI (interactive) |
| http://localhost:8000/redoc | ReDoc documentation |

---

## 10. API Reference

### `GET /`  —  Health Check

Returns the service status and the compute device in use.

**Response**

```json
{
  "status": "ok",
  "device": "cpu"
}
```

---

### `GET /models`  —  List Available Models

Returns all supported model keys, the default model, and the output classes.

**Response**

```json
{
  "available_models": ["efficientnet", "mobilenet", "resnet"],
  "default_model": "efficientnet",
  "classes": ["freshripe", "freshunripe", "overripe", "ripe", "rotten", "unripe"]
}
```

---

### `POST /predict`  —  Classify a Banana Image

**Form parameters**

| Parameter | Type | Required | Default | Description |
|---|---|---|---|---|
| `file` | `UploadFile` | ✅ | — | Image file (JPEG, PNG, WebP, BMP, …) |
| `model` | `string` (query) | ❌ | `efficientnet` | Model key to use |

**Success response (200)**

```json
{
  "model_used": "efficientnet",
  "predicted_class": "ripe",
  "confidence": 0.983412,
  "class_probabilities": {
    "freshripe":   0.002341,
    "freshunripe": 0.000123,
    "overripe":    0.005210,
    "ripe":        0.983412,
    "rotten":      0.007894,
    "unripe":      0.001020
  }
}
```

**Error responses**

| Code | Meaning |
|---|---|
| 400 | Bad request (invalid model key, empty file, or unreadable image) |
| 503 | Model weights file not found on disk |

---

## 11. Testing the API

### Using `curl`

```bash
# Default model (EfficientNetB0)
curl -X POST "http://localhost:8000/predict" \
     -F "file=@/path/to/banana.jpg"

# Specify a different model
curl -X POST "http://localhost:8000/predict?model=mobilenet" \
     -F "file=@/path/to/banana.jpg"

# Health check
curl http://localhost:8000/
```

### Using Python (`requests`)

```python
import requests

url = "http://localhost:8000/predict"

with open("banana.jpg", "rb") as f:
    response = requests.post(
        url,
        files={"file": ("banana.jpg", f, "image/jpeg")},
        params={"model": "efficientnet"},
    )

result = response.json()
print("Predicted class :", result["predicted_class"])
print("Confidence      :", f"{result['confidence']*100:.2f}%")
print("All probabilities:")
for cls, prob in result["class_probabilities"].items():
    print(f"  {cls:<15} {prob:.4f}")
```

### Using the Swagger UI

1. Open http://localhost:8000/docs in your browser.
2. Click **POST /predict** → **Try it out**.
3. Upload a banana image and click **Execute**.

### Comparing all three models at once

```python
import requests

image_path = "banana.jpg"
base_url = "http://localhost:8000/predict"

for model_key in ["efficientnet", "mobilenet", "resnet"]:
    with open(image_path, "rb") as f:
        r = requests.post(
            base_url,
            files={"file": f},
            params={"model": model_key},
        )
    data = r.json()
    print(f"[{model_key:12s}] {data['predicted_class']}  ({data['confidence']*100:.2f}%)")
```

---

## 12. How Inference Works

When `/predict` receives a request, the API:

1. **Decodes** the uploaded bytes into a PIL `RGB` image.
2. **Transforms** the image using the same pipeline that was used during validation/testing:
   - Resize to 224 × 224 pixels
   - Convert to a float tensor (values in [0, 1])
   - Normalise with ImageNet mean `[0.485, 0.456, 0.406]` and std `[0.229, 0.224, 0.225]`
3. **Runs** the tensor through the selected neural network in `eval()` mode with `torch.no_grad()`.
4. **Applies** `softmax` to the raw logits to obtain class probabilities.
5. **Returns** the class with the highest probability, its confidence, and the full distribution.

Models are loaded **lazily on first use** and then **cached in memory** for all subsequent requests, so the overhead of reading the `.pth` file is paid only once per process lifetime.

---

## 13. Troubleshooting

| Problem | Solution |
|---|---|
| `503 Service Unavailable` — weights not found | Run `python download_models.py` to fetch the `.pth` files |
| `pip install` fails on `torch` | Install PyTorch manually from https://pytorch.org/get-started/locally/ |
| Slow inference | Expected on CPU for ResNet50. Use EfficientNetB0 or MobileNetV3 for faster response |
| Image decoding error (400) | Ensure the uploaded file is a valid image (JPEG, PNG, WebP, BMP) |
| Port already in use | Change the port: `uvicorn main:app --port 8001` |

---

## License

This repository is provided for educational and research purposes.  
The model weights are derived from the training work in
[ParanKafleUTS/AI_Matrix_banana_ripness](https://github.com/ParanKafleUTS/AI_Matrix_banana_ripness).