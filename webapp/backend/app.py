"""
app.py -- backend for the Liveness Gate dashboard.

Serves real project data (training history, test metrics, confusion
matrix plot, decision log) and runs real live detection by calling the
project's own, already-tested src/face_utils.py and src/liveness_infer.py
directly -- no reimplementation, no mock data, no invented numbers.

Run from webapp/backend/:
    python app.py
Then open http://localhost:5050
"""

import csv
import io
import json
import os
import sys
from collections import deque
from datetime import datetime

from flask import Flask, jsonify, request, send_file, send_from_directory

# --- wire up the real project source, one level up from this webapp folder ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
sys.path.insert(0, SRC_DIR)

import config  # noqa: E402
from face_utils import FaceDetector, crop_face  # noqa: E402
from liveness_infer import LivenessPredictor, classify_three_way, load_gray_zone_band  # noqa: E402
from realtime_pipeline import AttendanceLogger  # noqa: E402

import cv2  # noqa: E402
import numpy as np  # noqa: E402

FRONTEND_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "frontend")

app = Flask(__name__, static_folder=FRONTEND_DIR, static_url_path="")

# --- real model components, loaded once at startup ---
# config.ACTIVE_MODEL_PATH defaults to v2 (face-crop-trained, see TASK G);
# set the LIVENESS_MODEL_VERSION=v1 env var before starting this process to
# run the original raw-frame-trained model instead, for comparison.
print(f"Loading real face detector and liveness model (version={config.MODEL_VERSION})...")
detector = FaceDetector()
predictor = LivenessPredictor(checkpoint_path=config.ACTIVE_MODEL_PATH)
logger = AttendanceLogger()
print(f"Model loaded: backbone={predictor.backbone}, threshold={predictor.threshold}")

# Three-way LIVE/SPOOF/UNCERTAIN band, loaded from evaluate_gray_zone.py's
# real results JSON for the active model version -- never hardcoded, so
# re-running that script and restarting this process is all it takes to
# update the live band. If the results JSON doesn't exist yet, the
# three-way decision is disabled (every frame is a plain LIVE/SPOOF
# decision) rather than crashing the server.
try:
    GRAY_ZONE_BAND = load_gray_zone_band()
    print(f"Gray-zone band loaded for {config.MODEL_VERSION}: {GRAY_ZONE_BAND}")
except FileNotFoundError as e:
    GRAY_ZONE_BAND = None
    print(f"WARNING: {e} -- three-way UNCERTAIN decision disabled, falling back to plain LIVE/SPOOF.")

# in-memory copy of the real test-set predictions, used to compute real
# APCER/BPCER/HTER at any threshold the user picks, without re-running the
# model -- this is a genuine recomputation from stored real probabilities,
# not an interpolation or invented curve.
_PREDICTIONS_CSV = os.path.join(PROJECT_ROOT, "outputs", "predictions", "predictions_test.csv")
_true_labels = []
_pred_probs = []
if os.path.isfile(_PREDICTIONS_CSV):
    with open(_PREDICTIONS_CSV) as f:
        for row in csv.DictReader(f):
            _true_labels.append(int(row["true_label"]))
            _pred_probs.append(float(row["prob_spoof"]))
    _true_labels = np.array(_true_labels)
    _pred_probs = np.array(_pred_probs)
    print(f"Loaded {len(_true_labels)} real test predictions for live threshold recomputation.")
else:
    print(f"WARNING: {_PREDICTIONS_CSV} not found -- run evaluate.py first. Threshold tuner will be empty.")
    _true_labels = np.array([])
    _pred_probs = np.array([])


def _compute_metrics_at_threshold(threshold: float) -> dict:
    """Recomputes real accuracy/APCER/BPCER/HTER from the stored real test
    predictions at an arbitrary threshold -- same logic as evaluate.py's
    compute_anti_spoof_metrics, applied live."""
    if len(_true_labels) == 0:
        return {"error": "no predictions available"}

    y_pred = (_pred_probs >= threshold).astype(int)
    y_true = _true_labels

    spoof_mask = y_true == config.LABEL_SPOOF
    live_mask = y_true == config.LABEL_LIVE

    apcer = float(np.mean(y_pred[spoof_mask] == config.LABEL_LIVE)) if spoof_mask.sum() else float("nan")
    bpcer = float(np.mean(y_pred[live_mask] == config.LABEL_SPOOF)) if live_mask.sum() else float("nan")
    hter = (apcer + bpcer) / 2
    accuracy = float(np.mean(y_true == y_pred))

    return {
        "threshold": threshold,
        "accuracy": accuracy,
        "apcer": apcer,
        "bpcer": bpcer,
        "hter": hter,
        "n_live": int(live_mask.sum()),
        "n_spoof": int(spoof_mask.sum()),
    }


# ---------------------------------------------------------------------------
# Static frontend
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    return send_from_directory(FRONTEND_DIR, "index.html")


# ---------------------------------------------------------------------------
# Real data endpoints
# ---------------------------------------------------------------------------
@app.route("/api/metrics")
def api_metrics():
    """Real final test-set metrics from evaluate.py's actual output."""
    path = os.path.join(PROJECT_ROOT, "outputs", "predictions", "metrics_test.json")
    if not os.path.isfile(path):
        return jsonify({"error": "metrics_test.json not found -- run evaluate.py first"}), 404
    with open(path) as f:
        return jsonify(json.load(f))


@app.route("/api/training-history")
def api_training_history():
    """Real per-epoch train/val loss and accuracy from train.py's actual run."""
    path = os.path.join(PROJECT_ROOT, "outputs", "plots", "training_history.json")
    if not os.path.isfile(path):
        return jsonify({"error": "training_history.json not found -- run train.py first"}), 404
    with open(path) as f:
        return jsonify(json.load(f))


@app.route("/api/plots/training-curves.png")
def api_plot_training_curves():
    path = os.path.join(PROJECT_ROOT, "outputs", "plots", "training_curves.png")
    if not os.path.isfile(path):
        return jsonify({"error": "not found"}), 404
    return send_file(path, mimetype="image/png")


@app.route("/api/plots/confusion-matrix.png")
def api_plot_confusion_matrix():
    path = os.path.join(PROJECT_ROOT, "outputs", "plots", "confusion_matrix_test.png")
    if not os.path.isfile(path):
        return jsonify({"error": "not found"}), 404
    return send_file(path, mimetype="image/png")


@app.route("/api/threshold-metrics")
def api_threshold_metrics():
    """Real APCER/BPCER/HTER/accuracy recomputed live at any threshold,
    from the real stored test-set prediction probabilities."""
    try:
        threshold = float(request.args.get("threshold", config.DECISION_THRESHOLD))
    except ValueError:
        return jsonify({"error": "threshold must be a number"}), 400
    threshold = max(0.0, min(1.0, threshold))
    return jsonify(_compute_metrics_at_threshold(threshold))


@app.route("/api/threshold", methods=["GET", "POST"])
def api_threshold():
    """GET returns the live predictor's current decision threshold.
    POST actually changes it -- affects every /api/detect call after."""
    if request.method == "GET":
        return jsonify({"threshold": predictor.threshold})

    data = request.get_json(force=True, silent=True) or {}
    try:
        new_threshold = float(data.get("threshold"))
    except (TypeError, ValueError):
        return jsonify({"error": "threshold must be a number"}), 400
    new_threshold = max(0.0, min(1.0, new_threshold))
    predictor.set_threshold(new_threshold)
    return jsonify({"threshold": predictor.threshold})


@app.route("/api/decision-log")
def api_decision_log():
    """Real decisions from the real CSV log written by the live pipeline."""
    path = config.ATTENDANCE_LOG_PATH
    if not os.path.isfile(path):
        return jsonify({"entries": [], "total": 0})

    with open(path) as f:
        rows = list(csv.DictReader(f))

    rows = list(reversed(rows))  # most recent first
    limit = int(request.args.get("limit", 100))
    return jsonify({"entries": rows[:limit], "total": len(rows)})


@app.route("/api/dataset-info")
def api_dataset_info():
    """Real dataset split counts, computed live by listing the actual
    data/celeba_spoof/{split}/{live,spoof} folders -- not hardcoded."""
    counts = {}
    for split in ("train", "val", "test"):
        for cls in ("live", "spoof"):
            d = os.path.join(config.CELEBA_SPOOF_DIR, split, cls)
            counts[f"{split}_{cls}"] = len(os.listdir(d)) if os.path.isdir(d) else 0
    counts["total"] = sum(counts.values())
    return jsonify(counts)


@app.route("/api/model-info")
def api_model_info():
    """Real, currently-loaded model facts -- no invented hardware claims."""
    return jsonify({
        "backbone": predictor.backbone,
        "image_size": predictor.image_size,
        "device": str(predictor.device),
        "detector": "MediaPipe BlazeFace (short-range)",
        "current_threshold": predictor.threshold,
        "model_version": config.MODEL_VERSION,
        "gray_zone_band": GRAY_ZONE_BAND,  # null if not loaded (three-way decision disabled)
    })


# ---------------------------------------------------------------------------
# Temporal smoothing + event-based logging state.
#
# A single raw per-frame P(spoof) is noisy: small pose/lighting changes
# between frames can swing it by 30-40 points even for a genuinely still
# live face, which showed up as confidence jumping between ~58% and ~95%
# within one second on the live page. We smooth by averaging the last N
# raw probabilities and basing the LIVE/SPOOF decision on that average
# instead of the instantaneous value.
#
# N=5 was chosen from the web pipeline's own measured round-trip rate
# (~15-20 FPS through the browser -> HTTP -> Flask path, slower than the
# 26.7 FPS measured for the bare Python pipeline without that overhead --
# see /api/model-info and the Live Detection page for the real web FPS).
# 5 frames at ~15-20 FPS is roughly 250-330ms of averaging: long enough to
# cancel single-frame jitter from pose/lighting micro-changes, short
# enough that holding up an actual spoof attempt still flips the decision
# within about a third of a second rather than lagging noticeably behind.
# ---------------------------------------------------------------------------
SMOOTHING_WINDOW = 5
_recent_probs = deque(maxlen=SMOOTHING_WINDOW)
_last_logged_decision = None  # None, "LIVE_ACCEPTED", "SPOOF_REJECTED", or "UNCERTAIN"
_face_was_present = False


# ---------------------------------------------------------------------------
# Debug: save real received frames + the exact crop sent to the classifier,
# so bounding-box / crop correctness can be visually verified against what
# the backend actually receives (not what the mirrored CSS preview shows).
# ---------------------------------------------------------------------------
DEBUG_SAVE_FRAMES = os.environ.get("LIVENESS_DEBUG_FRAMES", "0") == "1"
DEBUG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "debug_frames")
_debug_counter = {"n": 0}


def _save_debug_frame(frame, detections):
    os.makedirs(DEBUG_DIR, exist_ok=True)
    annotated = frame.copy()
    for det in detections:
        cv2.rectangle(annotated, (det.x, det.y), (det.x2, det.y2), (0, 255, 0), 2)
    n = _debug_counter["n"]
    cv2.imwrite(os.path.join(DEBUG_DIR, f"{n:03d}_full_frame_raw_bbox.jpg"), annotated)


def _save_debug_crop(crop):
    n = _debug_counter["n"]
    cv2.imwrite(os.path.join(DEBUG_DIR, f"{n:03d}_crop_sent_to_classifier.jpg"), crop)
    _debug_counter["n"] += 1


# ---------------------------------------------------------------------------
# Live detection: the real pipeline, one frame at a time
# ---------------------------------------------------------------------------
@app.route("/api/detect", methods=["POST"])
def api_detect():
    """
    Accepts one JPEG frame (multipart form field 'frame'), runs the real
    face_utils.FaceDetector + liveness_infer.LivenessPredictor on it, logs
    the decision to the real CSV log (same AttendanceLogger the actual
    realtime_pipeline.py script uses), and returns real results.
    """
    if "frame" not in request.files:
        return jsonify({"error": "no frame uploaded"}), 400

    file_bytes = np.frombuffer(request.files["frame"].read(), dtype=np.uint8)
    frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if frame is None:
        return jsonify({"error": "could not decode image"}), 400

    global _last_logged_decision, _face_was_present

    detections = detector.detect(frame)

    if DEBUG_SAVE_FRAMES:
        _save_debug_frame(frame, detections)

    if not detections:
        # Face left the frame: reset smoothing so a freshly-reappearing
        # face doesn't get averaged in with stale, unrelated probabilities.
        _recent_probs.clear()
        _face_was_present = False
        return jsonify({"face_detected": False})

    best = max(detections, key=lambda d: d.width * d.height)
    crop = crop_face(frame, best)
    if crop is None or crop.size == 0:
        return jsonify({"face_detected": False})

    if DEBUG_SAVE_FRAMES:
        _save_debug_crop(crop)

    # Raw, single-frame probability (unsmoothed) -- shown to the user
    # alongside the smoothed decision so the smoothing is visible, not
    # hidden.
    raw_prob_spoof = predictor.predict_prob(crop)
    _recent_probs.append(raw_prob_spoof)
    smoothed_prob_spoof = sum(_recent_probs) / len(_recent_probs)

    # Three-way LIVE/SPOOF/UNCERTAIN, applied to the SMOOTHED probability
    # (not the raw single-frame one) -- the gray-zone band was selected
    # against a model that sees one still image at a time, and smoothing
    # already reduces frame-to-frame noise the same way averaging multiple
    # independent exposures would, so the band is applied after smoothing
    # for a fair, stable UNCERTAIN decision rather than one that flickers
    # with single-frame jitter.
    if GRAY_ZONE_BAND is not None:
        label = classify_three_way(smoothed_prob_spoof, *GRAY_ZONE_BAND, threshold=predictor.threshold)
    else:
        label = "SPOOF" if smoothed_prob_spoof >= predictor.threshold else "LIVE"

    if label == "SPOOF":
        smoothed_confidence = smoothed_prob_spoof
    elif label == "LIVE":
        smoothed_confidence = 1.0 - smoothed_prob_spoof
    else:  # UNCERTAIN: no class to be "confident" in -- show P(spoof) itself,
           # the number that actually put this frame in the gray zone.
        smoothed_confidence = smoothed_prob_spoof

    raw_label = "SPOOF" if raw_prob_spoof >= predictor.threshold else "LIVE"
    raw_confidence = raw_prob_spoof if raw_label == "SPOOF" else 1.0 - raw_prob_spoof

    timestamp = datetime.now().isoformat(timespec="seconds")
    decision = {"LIVE": "LIVE_ACCEPTED", "SPOOF": "SPOOF_REJECTED", "UNCERTAIN": "UNCERTAIN"}[label]
    message = "Hold still / improve lighting" if label == "UNCERTAIN" else None

    # Log once per event (face newly appearing, or the smoothed decision
    # changing) instead of once per frame, so the decision log reflects
    # real events rather than flooding with ~15-20 near-duplicate rows
    # per second. UNCERTAIN events are logged the same way as LIVE/SPOOF
    # ones, not dropped.
    decision_changed = decision != _last_logged_decision
    face_newly_appeared = not _face_was_present
    if decision_changed or face_newly_appeared:
        logger.log(decision, smoothed_confidence, best)
        _last_logged_decision = decision
    _face_was_present = True

    return jsonify({
        "face_detected": True,
        "label": label,
        "confidence": smoothed_confidence,
        "message": message,
        "raw_label": raw_label,
        "raw_confidence": raw_confidence,
        "smoothing_window": len(_recent_probs),
        "bbox": {"x": best.x, "y": best.y, "width": best.width, "height": best.height},
        "timestamp": timestamp,
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5050, debug=False)
