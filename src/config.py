"""
Central configuration for the Mask-Aware Liveness / Anti-Spoofing module.

Keeping all shared constants in one place makes it easy for a teammate
(e.g. the partner integrating ArcFace attendance) to see every tunable
value without digging through each script.
"""

import os
import torch

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
CELEBA_SPOOF_DIR = os.path.join(DATA_DIR, "celeba_spoof")
CASIA_FASD_DIR = os.path.join(DATA_DIR, "casia_fasd")
SAMPLES_DIR = os.path.join(DATA_DIR, "samples")

MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
BEST_MODEL_PATH = os.path.join(MODELS_DIR, "best_liveness.pth")
LAST_MODEL_PATH = os.path.join(MODELS_DIR, "last_liveness.pth")
ONNX_MODEL_PATH = os.path.join(MODELS_DIR, "liveness.onnx")

OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")
PLOTS_DIR = os.path.join(OUTPUTS_DIR, "plots")
PREDICTIONS_DIR = os.path.join(OUTPUTS_DIR, "predictions")

LOGS_DIR = os.path.join(PROJECT_ROOT, "logs")
ATTENDANCE_LOG_PATH = os.path.join(LOGS_DIR, "attendance_log.csv")

# ---------------------------------------------------------------------------
# Class labels
# ---------------------------------------------------------------------------
# CelebA-Spoof convention: 0 = live, 1 = spoof (we follow the same mapping
# so the raw dataset folders can be used with minimal renaming).
LABEL_LIVE = 0
LABEL_SPOOF = 1
CLASS_NAMES = {LABEL_LIVE: "LIVE", LABEL_SPOOF: "SPOOF"}

# ---------------------------------------------------------------------------
# Model / training defaults
# ---------------------------------------------------------------------------
IMAGE_SIZE = 224                 # face crop input size (also supports 128)
BACKBONE = "mobilenet_v3_small"  # or "efficientnet_b0"
BATCH_SIZE = 32
NUM_EPOCHS = 15
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 4
RANDOM_SEED = 42

# Decision threshold on P(spoof). Anything >= threshold is classified SPOOF.
# Tunable at inference time without retraining.
DECISION_THRESHOLD = 0.5

DEVICE = torch.device("cuda" if torch.cuda.is_available() else
                       ("mps" if torch.backends.mps.is_available() else "cpu"))

# ---------------------------------------------------------------------------
# Face detection (MediaPipe)
# ---------------------------------------------------------------------------
FACE_DETECTION_CONFIDENCE = 0.5
FACE_CROP_MARGIN = 0.25   # fraction of bbox size added as padding around the face
