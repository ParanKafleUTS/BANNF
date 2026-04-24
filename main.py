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
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
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

# Recipe suggestions keyed by ripeness class
RECIPES: dict[str, dict] = {
    "freshripe": {
        "status": "Perfect for eating right now!",
        "edible": True,
        "recipes": [
            {"name": "Banana Smoothie",    "desc": "Blend with milk, honey and a pinch of cinnamon."},
            {"name": "Banana on Toast",    "desc": "Slice over sourdough toast with peanut butter."},
            {"name": "Banana Yoghurt Bowl","desc": "Top Greek yoghurt with banana slices and granola."},
            {"name": "Banana Oatmeal",     "desc": "Stir sliced banana into warm oats with maple syrup."},
        ],
    },
    "freshunripe": {
        "status": "Let it ripen 1-2 more days, or use in savoury dishes.",
        "edible": True,
        "recipes": [
            {"name": "Green Banana Curry",   "desc": "Simmer in coconut milk with curry leaves and spices."},
            {"name": "Banana Chips",         "desc": "Slice thinly, bake or air-fry with salt and turmeric."},
            {"name": "Raw Banana Stir-Fry",  "desc": "Sauté with mustard seeds, chilli and grated coconut."},
            {"name": "Green Banana Porridge","desc": "Boil and mash with coconut milk for a savoury porridge."},
        ],
    },
    "ripe": {
        "status": "Great flavour — ideal for most recipes.",
        "edible": True,
        "recipes": [
            {"name": "Banana Smoothie",    "desc": "Blend with frozen berries and almond milk."},
            {"name": "Banana Split",       "desc": "Halve and top with ice cream and hot fudge sauce."},
            {"name": "Banana Pudding",     "desc": "Layer with vanilla custard and crushed biscuits."},
            {"name": "Caramelised Banana", "desc": "Pan-fry in butter and brown sugar; serve with ice cream."},
        ],
    },
    "overripe": {
        "status": "Very sweet — perfect for baking and frozen desserts.",
        "edible": True,
        "recipes": [
            {"name": "Banana Bread",    "desc": "Classic moist loaf with walnuts baked at 180 °C."},
            {"name": "Banana Muffins",  "desc": "Quick muffins with chocolate chips ready in 25 minutes."},
            {"name": "Banana Pancakes", "desc": "Mashed banana, eggs and oats — fluffy 3-ingredient pancakes."},
            {"name": "Banana Ice Cream","desc": "Blend frozen chunks for a one-ingredient nice-cream."},
        ],
    },
    "rotten": {
        "status": "Not safe to eat. Please discard or compost.",
        "edible": False,
        "recipes": [],
    },
    "unripe": {
        "status": "Too firm to eat raw — allow to ripen 2-3 more days.",
        "edible": True,
        "recipes": [
            {"name": "Banana Chips",       "desc": "Thinly slice, toss in oil, bake at 160 °C until golden."},
            {"name": "Raw Banana Sabzi",   "desc": "Indian stir-fry with cumin, turmeric and coriander."},
            {"name": "Plantain-style Fry", "desc": "Shallow-fry rounds until golden; season with chilli salt."},
            {"name": "Banana Stew",        "desc": "Simmer in a light coconut and tomato stew."},
        ],
    },
}

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(STATIC_DIR, exist_ok=True)

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

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


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


class RecipeItem(BaseModel):
    name: str
    desc: str


class RecipesResponse(BaseModel):
    ripeness_class: str
    status: str
    edible: bool
    recipes: list[RecipeItem]


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


@app.get("/ui", tags=["UI"], include_in_schema=False)
def web_ui():
    """Serve the web interface for banana ripeness identification."""
    html_path = os.path.join(STATIC_DIR, "index.html")
    if not os.path.exists(html_path):
        raise HTTPException(status_code=404, detail="Web UI not found.")
    return FileResponse(html_path, media_type="text/html")


@app.get("/recipes/{class_name}", response_model=RecipesResponse, tags=["Recipes"])
def get_recipes(class_name: str):
    """
    Return recipe suggestions for a given banana ripeness class.

    **class_name** must be one of: freshripe, freshunripe, ripe, overripe, rotten, unripe
    """
    if class_name not in RECIPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown class {class_name!r}. Choose from: {list(RECIPES.keys())}",
        )
    data = RECIPES[class_name]
    return RecipesResponse(
        ripeness_class=class_name,
        status=data["status"],
        edible=data["edible"],
        recipes=[RecipeItem(**r) for r in data["recipes"]],
    )


