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
import re
import sys
import time
from collections import deque
from datetime import datetime

from flask import Flask, jsonify, request, send_file, send_from_directory

# --- wire up the real project source, one level up from this webapp folder ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
sys.path.insert(0, SRC_DIR)

import base64  # noqa: E402

import config  # noqa: E402
from face_utils import FaceDetector, bgr_crop_to_pil, crop_face  # noqa: E402
from gradcam import GradCAM, overlay_heatmap  # noqa: E402
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

def _active_model_path(v2_relpath, v1_relpath):
    """Resolves to the ACTIVE model's (config.MODEL_VERSION) own result
    file, falling back to the original v1-era path when running v1 for
    comparison. Centralizes the v2/v1 branch used by every endpoint below
    so the backend never serves a v1-era result while v2 is active (TASK N)."""
    relpath = v2_relpath if config.MODEL_VERSION == "v2" else v1_relpath
    return os.path.join(PROJECT_ROOT, *relpath)


# in-memory copy of the real test-set predictions, used to compute real
# APCER/BPCER/HTER at any threshold the user picks, without re-running the
# model -- this is a genuine recomputation from stored real probabilities,
# not an interpolation or invented curve. Points at the ACTIVE model's own
# predictions (see _active_model_path).
_PREDICTIONS_CSV = _active_model_path(
    ("outputs", "predictions", "v2_cropped", "predictions_v2_cropped_test.csv"),
    ("outputs", "predictions", "predictions_test.csv"),
)
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
    """Real final test-set metrics for the ACTIVE model (v2 by default)
    from evaluate.py's actual output. NOT the v1-era
    outputs/predictions/metrics_test.json, which this backend never reads
    while v2 is active (TASK N)."""
    path = _active_model_path(
        ("outputs", "predictions", "v2_cropped", "metrics_v2_cropped_test.json"),
        ("outputs", "predictions", "metrics_test.json"),
    )
    if not os.path.isfile(path):
        return jsonify({"error": f"{os.path.basename(path)} not found -- run evaluate.py first"}), 404
    with open(path) as f:
        return jsonify(json.load(f))


@app.route("/api/training-history")
def api_training_history():
    """Real per-epoch train/val loss and accuracy for the ACTIVE model's
    actual training run (TASK N: not the stale v1-era path)."""
    path = _active_model_path(
        ("outputs", "plots", "v2_cropped", "training_history.json"),
        ("outputs", "plots", "training_history.json"),
    )
    if not os.path.isfile(path):
        return jsonify({"error": "training_history.json not found -- run train.py first"}), 404
    with open(path) as f:
        return jsonify(json.load(f))


@app.route("/api/plots/training-curves.png")
def api_plot_training_curves():
    path = _active_model_path(
        ("outputs", "plots", "v2_cropped", "training_curves.png"),
        ("outputs", "plots", "training_curves.png"),
    )
    if not os.path.isfile(path):
        return jsonify({"error": "not found"}), 404
    return send_file(path, mimetype="image/png")


@app.route("/api/plots/confusion-matrix.png")
def api_plot_confusion_matrix():
    path = _active_model_path(
        ("outputs", "plots", "v2_cropped", "confusion_matrix_v2_cropped_test.png"),
        ("outputs", "plots", "confusion_matrix_test.png"),
    )
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
# TASK N: data API for the rebuilt frontend -- every number/image the new
# UI needs, served from the real result files these scripts already
# produce (summary.json, the v2 Grad-CAM gallery, masked-eval examples,
# and the remaining result plot PNGs), so the UI itself can be pure
# frontend work with no new backend logic needed per page.
# ---------------------------------------------------------------------------
@app.route("/api/summary")
def api_summary():
    """Serves results/summary.json -- the single source of truth built by
    build_summary.py (TASK K). Every number the dashboard shows should
    come from here, not be recomputed ad hoc in the frontend."""
    path = os.path.join(PROJECT_ROOT, "results", "summary.json")
    if not os.path.isfile(path):
        return jsonify({"error": "summary.json not found -- run build_summary.py first"}), 404
    with open(path) as f:
        return jsonify(json.load(f))


_GRADCAM_DIR = os.path.join(PROJECT_ROOT, "results", "v2_gradcam")


@app.route("/api/gradcam/gallery")
def api_gradcam_gallery():
    """Index of the v2 Grad-CAM gallery (generate_gradcam_gallery.py) --
    file, true/predicted label, spoof type, probability, category per
    tile. Images themselves are served by /api/gradcam/image/<file>."""
    path = os.path.join(_GRADCAM_DIR, "gallery.json")
    if not os.path.isfile(path):
        return jsonify({"error": "v2 gradcam gallery.json not found -- run generate_gradcam_gallery.py for v2 first"}), 404
    with open(path) as f:
        return jsonify(json.load(f))


@app.route("/api/gradcam/image/<path:filename>")
def api_gradcam_image(filename):
    """Serves one gallery tile by filename (as listed in /api/gradcam/gallery)."""
    full = os.path.realpath(os.path.join(_GRADCAM_DIR, filename))
    if not full.startswith(os.path.realpath(_GRADCAM_DIR) + os.sep) or not os.path.isfile(full):
        return jsonify({"error": "not found"}), 404
    return send_file(full, mimetype="image/png")


# v1's Grad-CAM gallery (generated in TASK D, before the v2 retrain) --
# served alongside v2's for Page 3's "before/after" PC-replay comparison
# (TASK P). Already includes 16 real PC-replay tiles, so nothing new
# needed to be generated for this.
_GRADCAM_V1_DIR = os.path.join(PROJECT_ROOT, "results", "gradcam")


@app.route("/api/gradcam/v1/gallery")
def api_gradcam_v1_gallery():
    path = os.path.join(_GRADCAM_V1_DIR, "gallery.json")
    if not os.path.isfile(path):
        return jsonify({"error": "v1 gradcam gallery.json not found -- run generate_gradcam_gallery.py for v1 first"}), 404
    with open(path) as f:
        return jsonify(json.load(f))


@app.route("/api/gradcam/v1/image/<path:filename>")
def api_gradcam_v1_image(filename):
    full = os.path.realpath(os.path.join(_GRADCAM_V1_DIR, filename))
    if not full.startswith(os.path.realpath(_GRADCAM_V1_DIR) + os.sep) or not os.path.isfile(full):
        return jsonify({"error": "not found"}), 404
    return send_file(full, mimetype="image/png")


_MASKED_EXAMPLES_DIR = os.path.join(PROJECT_ROOT, "results", "masked", "examples")
_MASKED_FILENAME_RE = re.compile(r"^\d+_(white|light_blue|black)_(live|spoof)_(.+)$")


@app.route("/api/masked/examples")
def api_masked_examples():
    """Lists the synthetic masked-face examples (evaluate_masked.py,
    TASK I) -- color and true label are parsed back out of each filename
    (there's no separate index JSON for these, unlike the Grad-CAM
    gallery). Images themselves are served by /api/masked/image/<file>."""
    if not os.path.isdir(_MASKED_EXAMPLES_DIR):
        return jsonify({"examples": []})

    examples = []
    for fname in sorted(os.listdir(_MASKED_EXAMPLES_DIR)):
        if fname == ".gitkeep" or not fname.lower().endswith((".png", ".jpg", ".jpeg")):
            continue
        m = _MASKED_FILENAME_RE.match(fname)
        examples.append({
            "file": fname,
            "color": m.group(1) if m else None,
            "true_label": m.group(2) if m else None,
            "source_file": m.group(3) if m else None,
        })
    return jsonify({"examples": examples})


@app.route("/api/masked/image/<path:filename>")
def api_masked_image(filename):
    """Serves one masked-face example image by filename."""
    full = os.path.realpath(os.path.join(_MASKED_EXAMPLES_DIR, filename))
    if not full.startswith(os.path.realpath(_MASKED_EXAMPLES_DIR) + os.sep) or not os.path.isfile(full):
        return jsonify({"error": "not found"}), 404
    return send_file(full, mimetype="image/png")


@app.route("/api/plots/breakdown-spoof-type.png")
def api_plot_breakdown_spoof_type():
    path = os.path.join(PROJECT_ROOT, "results", "plots", "v2", "breakdown_spoof_type.png")
    if not os.path.isfile(path):
        return jsonify({"error": "not found"}), 404
    return send_file(path, mimetype="image/png")


@app.route("/api/plots/breakdown-illumination.png")
def api_plot_breakdown_illumination():
    path = os.path.join(PROJECT_ROOT, "results", "plots", "v2", "breakdown_illumination.png")
    if not os.path.isfile(path):
        return jsonify({"error": "not found"}), 404
    return send_file(path, mimetype="image/png")


@app.route("/api/plots/breakdown-environment.png")
def api_plot_breakdown_environment():
    path = os.path.join(PROJECT_ROOT, "results", "plots", "v2", "breakdown_environment.png")
    if not os.path.isfile(path):
        return jsonify({"error": "not found"}), 404
    return send_file(path, mimetype="image/png")


@app.route("/api/plots/gray-zone-curve.png")
def api_plot_gray_zone_curve():
    path = os.path.join(PROJECT_ROOT, "results", "plots", "v2", "gray_zone_coverage_vs_error.png")
    if not os.path.isfile(path):
        return jsonify({"error": "not found"}), 404
    return send_file(path, mimetype="image/png")


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
def _run_pipeline_core(frame, decode_ms: float, use_smoothing: bool):
    """
    Runs the real Detect -> Crop -> Classify -> Decision stages on one
    already-decoded BGR frame, with real per-stage timing (the pipeline
    strip on the Live page, TASK P). Returns a plain dict, not a Flask
    response, so both /api/detect (smoothing + decision-log side effects)
    and the sample-image demo endpoints (no smoothing, no logging --
    each call is an independent static image, not a video stream frame)
    can share this without duplicating the detection/decision logic.
    """
    timings = {"frame_ms": decode_ms}

    t0 = time.perf_counter()
    detections = detector.detect(frame)
    timings["detect_ms"] = (time.perf_counter() - t0) * 1000

    if DEBUG_SAVE_FRAMES:
        _save_debug_frame(frame, detections)

    if not detections:
        return {"face_detected": False, "timings": timings}

    best = max(detections, key=lambda d: d.width * d.height)

    t0 = time.perf_counter()
    crop = crop_face(frame, best)
    timings["crop_ms"] = (time.perf_counter() - t0) * 1000
    if crop is None or crop.size == 0:
        return {"face_detected": False, "timings": timings}

    if DEBUG_SAVE_FRAMES:
        _save_debug_crop(crop)

    t0 = time.perf_counter()
    raw_prob_spoof = predictor.predict_prob(crop)
    timings["classify_ms"] = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    if use_smoothing:
        _recent_probs.append(raw_prob_spoof)
        smoothed_prob_spoof = sum(_recent_probs) / len(_recent_probs)
        smoothing_window = len(_recent_probs)
    else:
        # A sample/demo image is a single independent exposure, not one
        # frame of a continuing video stream -- averaging it with whatever
        # happens to be in the live camera's smoothing window would be
        # meaningless (and would corrupt that window for the camera tab).
        smoothed_prob_spoof = raw_prob_spoof
        smoothing_window = 1

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
    timings["decision_ms"] = (time.perf_counter() - t0) * 1000

    raw_label = "SPOOF" if raw_prob_spoof >= predictor.threshold else "LIVE"
    raw_confidence = raw_prob_spoof if raw_label == "SPOOF" else 1.0 - raw_prob_spoof
    message = "Hold still / improve lighting" if label == "UNCERTAIN" else None

    return {
        "face_detected": True,
        "label": label,
        "confidence": smoothed_confidence,
        "message": message,
        "raw_label": raw_label,
        "raw_confidence": raw_confidence,
        "smoothing_window": smoothing_window,
        "bbox": {"x": best.x, "y": best.y, "width": best.width, "height": best.height},
        "timings": timings,
        "_crop": crop,  # internal use only (e.g. by callers building an Explain overlay too); stripped before jsonify
        "_best": best,
    }


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

    t0 = time.perf_counter()
    file_bytes = np.frombuffer(request.files["frame"].read(), dtype=np.uint8)
    frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    decode_ms = (time.perf_counter() - t0) * 1000
    if frame is None:
        return jsonify({"error": "could not decode image"}), 400

    global _last_logged_decision, _face_was_present

    result = _run_pipeline_core(frame, decode_ms, use_smoothing=True)

    if not result["face_detected"]:
        # Face left the frame: reset smoothing so a freshly-reappearing
        # face doesn't get averaged in with stale, unrelated probabilities.
        _recent_probs.clear()
        _face_was_present = False
        return jsonify({"face_detected": False, "timings": result["timings"]})

    best = result.pop("_best")
    result.pop("_crop", None)

    timestamp = datetime.now().isoformat(timespec="seconds")
    decision = {"LIVE": "LIVE_ACCEPTED", "SPOOF": "SPOOF_REJECTED", "UNCERTAIN": "UNCERTAIN"}[result["label"]]

    # Log once per event (face newly appearing, or the smoothed decision
    # changing) instead of once per frame, so the decision log reflects
    # real events rather than flooding with ~15-20 near-duplicate rows
    # per second. UNCERTAIN events are logged the same way as LIVE/SPOOF
    # ones, not dropped.
    decision_changed = decision != _last_logged_decision
    face_newly_appeared = not _face_was_present
    if decision_changed or face_newly_appeared:
        logger.log(decision, result["confidence"], best)
        _last_logged_decision = decision
    _face_was_present = True

    result["timestamp"] = timestamp
    return jsonify(result)


# ---------------------------------------------------------------------------
# Grad-CAM "Explain this frame" (TASK M) -- on-demand only, never on every
# frame (Grad-CAM is ~20-90ms depending on backbone, measured in
# benchmark_backbones.py / the earlier Grad-CAM investigation -- too slow to
# run per-frame in the live loop, but fine for a single user-triggered call).
# ---------------------------------------------------------------------------
# Last-conv-block getter per backbone, same choice already used by
# generate_gradcam_gallery.py (mobilenet_v3_small) and benchmark_backbones.py
# (all three) -- verified there by checking the real activation shape before
# trusting it.
_GRADCAM_LAYER_GETTER = {
    "mobilenet_v3_small": lambda m: m.feature_extractor.features[-1],
    "resnet18": lambda m: m.feature_extractor.layer4[-1],
    "efficientnet_b0": lambda m: m.feature_extractor.features[-1],
}


@app.route("/api/explain", methods=["POST"])
def api_explain():
    """
    Accepts one JPEG frame (multipart form field 'frame', same as
    /api/detect), runs the real deployment crop (BlazeFace + crop_face)
    and a single on-demand Grad-CAM pass on the ACTIVE model, and returns
    the heatmap overlay as a base64 JPEG plus the prediction/probability
    for that single frame.

    Deliberately independent of /api/detect's global smoothing state
    (_recent_probs / _last_logged_decision / _face_was_present) and of the
    decision log -- this is a one-off "explain this frame" action, not a
    live-loop event, and must never shift or reset the ongoing smoothing
    window or get logged as a LIVE/SPOOF/UNCERTAIN decision. GradCAM's own
    hook lifecycle (register per call, remove in __exit__) already
    guarantees predictor.model is left exactly as it was found -- same
    model instance /api/detect uses next, unmodified -- verified by
    tests/test_gradcam.py's test_model_parameters_unchanged_after_call.
    """
    if "frame" not in request.files:
        return jsonify({"error": "no frame uploaded"}), 400

    file_bytes = np.frombuffer(request.files["frame"].read(), dtype=np.uint8)
    frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if frame is None:
        return jsonify({"error": "could not decode image"}), 400

    return jsonify(_explain_frame(frame))


def _explain_frame(frame) -> dict:
    """Shared core of /api/explain, also used by the sample-image demo
    endpoints (TASK P) so both go through the exact same real pipeline."""
    detections = detector.detect(frame)
    if not detections:
        return {"face_detected": False}

    best = max(detections, key=lambda d: d.width * d.height)
    crop = crop_face(frame, best)
    if crop is None or crop.size == 0:
        return {"face_detected": False}

    getter = _GRADCAM_LAYER_GETTER.get(predictor.backbone)
    if getter is None:
        return {"error": f"Grad-CAM not supported for backbone {predictor.backbone!r}"}

    pil_image = bgr_crop_to_pil(crop)
    tensor = predictor.transform(pil_image).unsqueeze(0).to(predictor.device)

    target_layer = getter(predictor.model)
    with GradCAM(predictor.model, target_layer) as cam:
        heatmap, prob_spoof = cam(tensor, target="predicted")

    crop_resized = cv2.resize(crop, (predictor.image_size, predictor.image_size))
    overlay_bgr = overlay_heatmap(crop_resized, heatmap)
    ok, buf = cv2.imencode(".jpg", overlay_bgr)
    if not ok:
        return {"error": "failed to encode overlay image"}
    overlay_b64 = base64.b64encode(buf.tobytes()).decode("ascii")

    label = "SPOOF" if prob_spoof >= predictor.threshold else "LIVE"
    confidence = prob_spoof if label == "SPOOF" else 1.0 - prob_spoof

    return {
        "face_detected": True,
        "label": label,
        "confidence": confidence,
        "prob_spoof": prob_spoof,
        "backbone": predictor.backbone,
        "overlay_image_base64": f"data:image/jpeg;base64,{overlay_b64}",
        "bbox": {"x": best.x, "y": best.y, "width": best.width, "height": best.height},
    }


# ---------------------------------------------------------------------------
# TASK P: "Camera / Sample images" demo mode. Each sample is a REAL image
# from the actual test set (or the masked-eval examples), picked to cover
# one condition each; running one through detect/explain hits the exact
# same code paths as a live webcam frame -- never a cached/precomputed
# result -- so the numbers shown are always freshly computed.
# ---------------------------------------------------------------------------
_SAMPLE_IMAGES = {
    "live": {
        "path": os.path.join(PROJECT_ROOT, "data", "celeba_spoof", "test", "live", "test_10059_001096.png"),
        "label": "Live", "description": "Genuine, unmasked",
    },
    "masked_live": {
        "path": os.path.join(PROJECT_ROOT, "results", "masked", "examples", "01_white_live_test_7036_000466.png.png"),
        "label": "Masked Live", "description": "Genuine, synthetic surgical mask overlay",
    },
    "photo": {
        "path": os.path.join(PROJECT_ROOT, "data", "celeba_spoof", "test", "spoof", "test_5023_000033.png"),
        "label": "Printed Photo", "description": "Printed-photo attack",
    },
    "phone": {
        "path": os.path.join(PROJECT_ROOT, "data", "celeba_spoof", "test", "spoof", "test_9149_001134.png"),
        "label": "Phone Replay", "description": "Photo displayed on a phone screen",
    },
    "pc_replay": {
        "path": os.path.join(PROJECT_ROOT, "data", "celeba_spoof", "test", "spoof", "test_5223_001725.png"),
        "label": "PC Replay", "description": "Photo displayed on a PC screen -- the weakest attack type measured",
    },
    "poster": {
        "path": os.path.join(PROJECT_ROOT, "data", "celeba_spoof", "test", "spoof", "test_5520_000625.png"),
        "label": "Poster", "description": "Poster attack",
    },
    "mask_3d": {
        "path": os.path.join(PROJECT_ROOT, "data", "celeba_spoof", "test", "spoof", "test_10059_001051.png"),
        "label": "3D Mask", "description": "Rigid 3D mask attack",
    },
    "known_failure": {
        "path": os.path.join(PROJECT_ROOT, "data", "celeba_spoof", "test", "spoof", "test_5966_000293.png"),
        "label": "Known Failure (PC replay)", "description": "A real PC-replay false accept from the v2 test run",
    },
}


@app.route("/api/sample-images")
def api_sample_images():
    """Lists the curated demo-mode images (id, label, description only --
    no file paths exposed)."""
    return jsonify({
        "images": [{"id": k, "label": v["label"], "description": v["description"]} for k, v in _SAMPLE_IMAGES.items()]
    })


def _load_sample_frame(sample_id):
    sample = _SAMPLE_IMAGES.get(sample_id)
    if sample is None:
        return None
    return cv2.imread(sample["path"])


@app.route("/api/sample-images/<sample_id>/image")
def api_sample_image_file(sample_id):
    """Serves the sample's own raw source image, so the UI can show what
    was actually analyzed instead of a blank camera panel in sample mode."""
    sample = _SAMPLE_IMAGES.get(sample_id)
    if sample is None or not os.path.isfile(sample["path"]):
        return jsonify({"error": "not found"}), 404
    return send_file(sample["path"], mimetype="image/png")


@app.route("/api/sample-images/<sample_id>/detect", methods=["POST"])
def api_sample_image_detect(sample_id):
    """Runs the exact same real Detect->Crop->Classify->Decision pipeline
    as /api/detect against a curated sample image instead of an uploaded
    webcam frame. Deliberately NOT smoothed (use_smoothing=False) and
    never logged -- a static demo image is one independent exposure, not
    a frame in an ongoing video stream."""
    if sample_id not in _SAMPLE_IMAGES:
        return jsonify({"error": f"unknown sample id {sample_id!r}"}), 404

    t0 = time.perf_counter()
    frame = _load_sample_frame(sample_id)
    decode_ms = (time.perf_counter() - t0) * 1000
    if frame is None:
        return jsonify({"error": "sample image file not found on disk"}), 500

    result = _run_pipeline_core(frame, decode_ms, use_smoothing=False)
    result.pop("_crop", None)
    result.pop("_best", None)
    result["timestamp"] = datetime.now().isoformat(timespec="seconds")
    return jsonify(result)


@app.route("/api/sample-images/<sample_id>/explain", methods=["POST"])
def api_sample_image_explain(sample_id):
    """Runs the exact same real Grad-CAM explain pipeline as /api/explain
    against a curated sample image."""
    if sample_id not in _SAMPLE_IMAGES:
        return jsonify({"error": f"unknown sample id {sample_id!r}"}), 404

    frame = _load_sample_frame(sample_id)
    if frame is None:
        return jsonify({"error": "sample image file not found on disk"}), 500

    return jsonify(_explain_frame(frame))


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5050, debug=False)
