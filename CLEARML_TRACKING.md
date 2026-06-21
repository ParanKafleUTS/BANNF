# ClearML Experiment Tracking

This guide explains how to set up and use **ClearML** to track every inference
request made to the Banana Ripeness Classification API.

---

## Table of Contents

1. [What Gets Tracked](#1-what-gets-tracked)
2. [Prerequisites](#2-prerequisites)
3. [Create a ClearML Account](#3-create-a-clearml-account)
4. [Configure Credentials](#4-configure-credentials)
5. [Install Dependencies](#5-install-dependencies)
6. [Run the API with Tracking Enabled](#6-run-the-api-with-tracking-enabled)
7. [Viewing Results in the ClearML Dashboard](#7-viewing-results-in-the-clearml-dashboard)
8. [Tracked Metrics Reference](#8-tracked-metrics-reference)
9. [Using the Self-Hosted ClearML Server](#9-using-the-self-hosted-clearml-server)
10. [Disabling Tracking](#10-disabling-tracking)
11. [Troubleshooting](#11-troubleshooting)

---

## 1. What Gets Tracked

Every time a client calls `POST /predict`, the API automatically logs the
following information to a ClearML **Inference Session** task:

| Data | ClearML Location | Description |
|---|---|---|
| Winning class confidence | Scalars → *Inference / Confidence* | The probability score of the predicted class |
| All 6 class probabilities | Scalars → *Inference / Class Probabilities* | Per-class softmax probabilities |
| Predicted class index | Scalars → *Inference / Predicted Class Index* | Integer index (0–5) of the winning class |
| Human-readable summary | Console log | `[N] model=… class=… confidence=…` |
| Static configuration | Hyperparameters | Class names, model registry, device, input size |

Each prediction call uses an auto-incrementing **iteration** counter so you can
plot how predictions change over time.

---

## 2. Prerequisites

- Python 3.10+
- A ClearML account (free tier available at [app.clear.ml](https://app.clear.ml))
- The API dependencies already installed (`pip install -r requirements.txt`)

---

## 3. Create a ClearML Account

1. Go to **[https://app.clear.ml](https://app.clear.ml)** and sign up for free.
2. After logging in, open **Settings → Workspace** (top-right avatar menu).
3. Click **Create new credentials**.  
   A dialog appears with your personal API credentials.

---

## 4. Configure Credentials

### Option A — Interactive wizard (recommended for first time)

```bash
clearml-init
```

Follow the prompts. Paste the **API host**, **access key**, and **secret key**
shown in the ClearML workspace settings.  
This writes your credentials to `~/clearml.conf`.

### Option B — Edit `~/clearml.conf` manually

Create or edit `~/.clearml/clearml.conf` (Linux/macOS) or
`%USERPROFILE%\.clearml\clearml.conf` (Windows):

```ini
api {
    web_server: https://app.clear.ml
    api_server: https://api.clear.ml
    files_server: https://files.clear.ml
    credentials {
        access_key: "YOUR_ACCESS_KEY"
        secret_key:  "YOUR_SECRET_KEY"
    }
}
```

### Option C — Environment variables

Set these before starting the server (useful in Docker / CI):

```bash
export CLEARML_WEB_HOST="https://app.clear.ml"
export CLEARML_API_HOST="https://api.clear.ml"
export CLEARML_FILES_HOST="https://files.clear.ml"
export CLEARML_API_ACCESS_KEY="YOUR_ACCESS_KEY"
export CLEARML_API_SECRET_KEY="YOUR_SECRET_KEY"
```

---

## 5. Install Dependencies

```bash
pip install -r requirements.txt
```

`clearml>=1.14.0` is already listed in `requirements.txt`.  
You can verify the installation:

```bash
python -c "import clearml; print(clearml.__version__)"
```

---

## 6. Run the API with Tracking Enabled

Once credentials are configured, simply start the server as normal:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

On startup you will see a log line similar to:

```
ClearML Task: created new task id=<task-id>
...
```

This confirms a new **Inference Session** task has been registered in ClearML.

Confirm tracking is active via the health endpoint:

```bash
curl http://localhost:8000/
```

```json
{
  "status": "ok",
  "device": "cpu",
  "clearml_tracking": true
}
```

If `"clearml_tracking": false`, check the [Troubleshooting](#11-troubleshooting) section.

---

## 7. Viewing Results in the ClearML Dashboard

1. Open **[https://app.clear.ml](https://app.clear.ml)** and log in.
2. In the left sidebar select **Projects → Banana Ripeness API**.
3. Click on the **Inference Session** task for the current server run.
4. Navigate to:

   | Tab | What you'll find |
   |---|---|
   | **Scalars** | Line charts for confidence, class probabilities, predicted class index over time |
   | **Console** | Human-readable log of every prediction (`[N] model=… class=… confidence=…`) |
   | **Configuration → Hyperparameters** | Static model config (class names, device, etc.) |

### Example scalar charts

After sending a few requests you will see plots like:

```
Inference / Confidence
  efficientnet  ────────────────────
  mobilenet     - - - - - - - - - -
  resnet        ················

Inference / Class Probabilities
  freshripe     ____
  ripe          ████████████████████
  overripe      ▁▁▁▁
  ...
```

---

## 8. Tracked Metrics Reference

### Scalars

| Title | Series | Value |
|---|---|---|
| `Inference / Confidence` | `<model_key>` | Confidence score (0–1) of predicted class |
| `Inference / Class Probabilities` | `<class_name>` | Softmax probability (0–1) for each of the 6 classes |
| `Inference / Predicted Class Index` | `<model_key>` | Integer 0–5 mapping to the CLASS_NAMES list |

### Class index mapping

| Index | Class |
|---|---|
| 0 | freshripe |
| 1 | freshunripe |
| 2 | overripe |
| 3 | ripe |
| 4 | rotten |
| 5 | unripe |

### Hyperparameters (logged once at startup)

| Key | Value |
|---|---|
| `model_registry` | `["efficientnet", "mobilenet", "resnet"]` |
| `num_classes` | `6` |
| `class_names` | `["freshripe", …]` |
| `input_size` | `224` |
| `device` | `cpu` or `cuda` |

---

## 9. Using the Self-Hosted ClearML Server

If you prefer to keep data on your own infrastructure, run the open-source
ClearML server locally with Docker Compose:

```bash
# Clone the ClearML server repo
git clone https://github.com/allegroai/clearml-server.git
cd clearml-server/docker

# Start all services (API, web UI, file server)
docker-compose up -d
```

The default ports are:
- Web UI: `http://localhost:8080`
- API server: `http://localhost:8008`
- File server: `http://localhost:8081`

Then update your `~/clearml.conf` to point to localhost:

```ini
api {
    web_server: http://localhost:8080
    api_server: http://localhost:8008
    files_server: http://localhost:8081
    credentials {
        access_key: "YOUR_KEY"
        secret_key:  "YOUR_SECRET"
    }
}
```

---

## 10. Disabling Tracking

ClearML tracking is **optional** and the API works perfectly without it.

To disable tracking entirely, simply do **not** configure `~/clearml.conf` or
the environment variables.  When ClearML initialisation fails or the package is
missing, the server prints a warning and continues running — all prediction
endpoints remain fully functional.

You can confirm tracking is off via the health endpoint:

```json
{
  "status": "ok",
  "device": "cpu",
  "clearml_tracking": false
}
```

---

## 11. Troubleshooting

| Problem | Solution |
|---|---|
| `"clearml_tracking": false` in health check | Run `clearml-init` to configure credentials, or check environment variables |
| `[ClearML] Could not initialise task: …` in server logs | Credentials are wrong or the ClearML server is unreachable |
| `ModuleNotFoundError: No module named 'clearml'` | Run `pip install clearml>=1.14.0` |
| Task does not appear in the dashboard | Ensure `project_name="Banana Ripeness API"` in `main.py` matches what you are searching for |
| Old tasks clutter the dashboard | Each server restart creates a new task (`reuse_last_task_id=False`); archive old ones in the ClearML UI |
| Metrics stop updating | The ClearML logger flushes on a background thread; wait a few seconds after requests |

---

## Quick-start checklist

```bash
# 1. Install deps
pip install -r requirements.txt

# 2. Configure credentials (one time)
clearml-init

# 3. Start the API
uvicorn main:app --port 8000

# 4. Send a test prediction
curl -X POST http://localhost:8000/predict -F "file=@banana.jpg"

# 5. Open https://app.clear.ml  →  Banana Ripeness API  →  Inference Session
```
