"""
main.py
-------
FastAPI REST API for banana ripeness classification.

Supported models:
    - EfficientNetB0  (default, highest accuracy ~96.98 %)
    - MobileNetV3Large
    - ResNet50

Run:
    uvicorn main:app --host 0.0.0.0 --port 8000 --reload

Endpoints:
    GET  /            – health check
    GET  /models      – list available models
    POST /predict     – classify an uploaded image

ClearML experiment tracking is enabled automatically when ClearML credentials
are configured (see CLEARML_TRACKING.md).  The API degrades gracefully when
ClearML is unavailable or not configured.
"""

import io
import itertools
import os
from contextlib import asynccontextmanager
from typing import Optional

import torch
import torch.nn as nn
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from PIL import Image
from pydantic import BaseModel
from torchvision import models, transforms

# ---------------------------------------------------------------------------
# ClearML (optional – degrades gracefully when not configured)
# ---------------------------------------------------------------------------

try:
    from clearml import Task as ClearMLTask

    _CLEARML_AVAILABLE = True
except ImportError:
    _CLEARML_AVAILABLE = False

_clearml_task = None
_prediction_counter = itertools.count(start=1)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CLASS_NAMES = ["freshripe", "freshunripe", "overripe", "ripe", "rotten", "unripe"]
NUM_CLASSES = len(CLASS_NAMES)
INPUT_SIZE = 224
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")

# Map of friendly model keys → filename
MODEL_REGISTRY = {
    "efficientnet": "EfficientNetB0_banana_ripeness.pth",
    "mobilenet": "MobileNetV3_banana_ripeness.pth",
    "resnet": "ResNet50_banana_ripeness.pth",
}

# Inference transforms (no augmentation; same as val/test in training)
INFERENCE_TRANSFORMS = transforms.Compose(
    [
        transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)

# ---------------------------------------------------------------------------
# Model builders
# ---------------------------------------------------------------------------


def _build_efficientnet(num_classes: int) -> nn.Module:
    m = models.efficientnet_b0(weights=None)
    in_features = m.classifier[1].in_features
    m.classifier[1] = nn.Linear(in_features, num_classes)
    return m


def _build_mobilenet(num_classes: int) -> nn.Module:
    m = models.mobilenet_v3_large(weights=None)
    in_features = m.classifier[3].in_features
    m.classifier[3] = nn.Linear(in_features, num_classes)
    return m


def _build_resnet(num_classes: int) -> nn.Module:
    m = models.resnet50(weights=None)
    in_features = m.fc.in_features
    m.fc = nn.Linear(in_features, num_classes)
    return m


_BUILDERS = {
    "efficientnet": _build_efficientnet,
    "mobilenet": _build_mobilenet,
    "resnet": _build_resnet,
}

# ---------------------------------------------------------------------------
# Model loading (cached so each model is loaded only once per process)
# ---------------------------------------------------------------------------

_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(model_key: str) -> nn.Module:
    """Load and cache a model from disk."""
    if model_key not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model key: {model_key!r}")

    pth_path = os.path.join(MODELS_DIR, MODEL_REGISTRY[model_key])
    if not os.path.exists(pth_path):
        raise FileNotFoundError(
            f"Model weights not found at {pth_path}. "
            "Run `python download_models.py` to fetch them."
        )

    net = _BUILDERS[model_key](NUM_CLASSES)
    state_dict = torch.load(pth_path, map_location=_DEVICE, weights_only=True)
    net.load_state_dict(state_dict)
    net.to(_DEVICE)
    net.eval()
    return net


# Use a module-level cache dict instead of lru_cache so it can be populated
# inside the lifespan handler.
_model_cache: dict[str, nn.Module] = {}


def get_model(model_key: str) -> nn.Module:
    if model_key not in _model_cache:
        _model_cache[model_key] = load_model(model_key)
    return _model_cache[model_key]


# ---------------------------------------------------------------------------
# Application lifespan – ClearML init / teardown
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):  # noqa: ARG001
    """Start ClearML Task on startup; close it on shutdown."""
    global _clearml_task  # noqa: PLW0603

    if _CLEARML_AVAILABLE:
        try:
            _clearml_task = ClearMLTask.init(
                project_name="Banana Ripeness API",
                task_name="Inference Session",
                task_type=ClearMLTask.TaskTypes.inference,
                reuse_last_task_id=False,
                auto_connect_frameworks=False,
            )
            # Log static configuration as hyperparameters
            _clearml_task.set_parameters_as_dict(
                {
                    "model_registry": list(MODEL_REGISTRY.keys()),
                    "num_classes": NUM_CLASSES,
                    "class_names": CLASS_NAMES,
                    "input_size": INPUT_SIZE,
                    "device": str(_DEVICE),
                }
            )
        except Exception as exc:  # noqa: BLE001
            # ClearML misconfiguration must not break the API
            print(f"[ClearML] Could not initialise task: {exc}")
            _clearml_task = None

    yield  # application runs here

    if _clearml_task is not None:
        _clearml_task.close()


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Banana Ripeness Classification API",
    description=(
        "REST API for classifying banana ripeness using PyTorch models "
        "trained on a 6-class dataset. Models originate from "
        "ParanKafleUTS/AI_Matrix_banana_ripness."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Pydantic response schemas
# ---------------------------------------------------------------------------


class PredictionResponse(BaseModel):
    model_used: str
    predicted_class: str
    confidence: float
    class_probabilities: dict[str, float]


class HealthResponse(BaseModel):
    status: str
    device: str
    clearml_tracking: bool


class ModelsResponse(BaseModel):
    available_models: list[str]
    default_model: str
    classes: list[str]


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _predict(image_bytes: bytes, model_key: str) -> PredictionResponse:
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Cannot decode image: {exc}") from exc

    try:
        net = get_model(model_key)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    tensor = INFERENCE_TRANSFORMS(image).unsqueeze(0).to(_DEVICE)

    with torch.no_grad():
        logits = net(tensor)
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu().tolist()

    pred_idx = probs.index(max(probs))
    class_probs = {name: round(p, 6) for name, p in zip(CLASS_NAMES, probs)}

    result = PredictionResponse(
        model_used=model_key,
        predicted_class=CLASS_NAMES[pred_idx],
        confidence=round(probs[pred_idx], 6),
        class_probabilities=class_probs,
    )

    # ---- ClearML logging ------------------------------------------------
    if _clearml_task is not None:
        iteration = next(_prediction_counter)
        logger = _clearml_task.get_logger()

        # Confidence of the winning class
        logger.report_scalar(
            title="Inference / Confidence",
            series=model_key,
            value=result.confidence,
            iteration=iteration,
        )

        # Full probability distribution
        for cls_name, prob in class_probs.items():
            logger.report_scalar(
                title="Inference / Class Probabilities",
                series=cls_name,
                value=prob,
                iteration=iteration,
            )

        # Predicted class encoded as its index (for numeric plotting)
        logger.report_scalar(
            title="Inference / Predicted Class Index",
            series=model_key,
            value=pred_idx,
            iteration=iteration,
        )

        # Human-readable log line
        logger.report_text(
            f"[{iteration}] model={model_key} | "
            f"class={result.predicted_class} | "
            f"confidence={result.confidence:.4f}"
        )
    # ---------------------------------------------------------------------

    return result


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/", response_model=HealthResponse, tags=["Health"])
def health_check():
    """Return service health and compute device."""
    return HealthResponse(
        status="ok",
        device=str(_DEVICE),
        clearml_tracking=_clearml_task is not None,
    )


@app.get("/models", response_model=ModelsResponse, tags=["Models"])
def list_models():
    """List all supported models and the output classes."""
    return ModelsResponse(
        available_models=list(MODEL_REGISTRY.keys()),
        default_model="efficientnet",
        classes=CLASS_NAMES,
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(
    file: UploadFile = File(..., description="Banana image (JPEG / PNG / WebP …)"),
    model: Optional[str] = Query(
        default="efficientnet",
        description="Model key to use for inference: efficientnet | mobilenet | resnet",
    ),
):
    """
    Classify the ripeness of a banana from an uploaded image.

    - **file**: image file (JPEG, PNG, WebP, BMP, …)
    - **model**: one of `efficientnet` (default), `mobilenet`, `resnet`

    Returns the predicted class, confidence, and full probability distribution.
    Each call is logged to ClearML when tracking is active.
    """
    if model not in MODEL_REGISTRY:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown model {model!r}. Choose from: {list(MODEL_REGISTRY.keys())}",
        )

    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    return _predict(contents, model)


