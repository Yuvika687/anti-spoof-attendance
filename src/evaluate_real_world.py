"""
evaluate_real_world.py
------------------------
Runs the REAL, full detect -> crop -> classify pipeline (the same
face_utils.FaceDetector + liveness_infer.LivenessPredictor used by
realtime_pipeline.py and the webapp) against your own recorded real-world
test videos, and reports real per-attack-type APCER/BPCER/HTER.

This exists because the CelebA-Spoof test set (99.15% accuracy) only
proves the model works on CelebA-Spoof's own capture distribution. It does
NOT prove the model works on your actual webcam, your actual room, your
actual phone/printer. This script measures that, honestly, whatever the
numbers turn out to be.

Expected folder layout -- one subfolder per attack type, containing short
video recordings (.mp4/.mov/.avi) or still photos (.jpg/.png) of that
condition:

    real_world_test/
        live_no_mask/        <- you, unmasked, genuine
        live_with_mask/      <- you, wearing a mask, genuine
        printed_photo/       <- a printed photo of a face, held up to camera
        printed_photo_masked/<- a printed photo of a masked face
        phone_photo_replay/  <- a photo displayed on a phone screen
        phone_video_replay/  <- a video played on a phone screen
        screen_replay/       <- a laptop/monitor screen showing a face

Folders are matched by name; "live" in the folder name = genuine (ground
truth LIVE), anything else = attack (ground truth SPOOF). Vary lighting
and distance across your recordings within each folder for a more
representative sample.

Usage:
    python evaluate_real_world.py --data_root ../real_world_test \\
        --checkpoint ../models/best_liveness.pth
"""

import argparse
import csv
import os
from collections import defaultdict

import cv2
import numpy as np

import config
from face_utils import FaceDetector, crop_face
from liveness_infer import LivenessPredictor

VIDEO_EXTENSIONS = (".mp4", ".mov", ".avi", ".m4v")
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")


def is_genuine_folder(folder_name: str) -> bool:
    """Ground truth: a folder is LIVE only if its name starts with 'live'."""
    return folder_name.lower().startswith("live")


def iter_frames_from_video(path: str, sample_every: int = 5):
    """Yields every Nth frame from a video file to avoid over-sampling
    near-identical consecutive frames."""
    cap = cv2.VideoCapture(path)
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % sample_every == 0:
            yield frame
        idx += 1
    cap.release()


def collect_samples(folder_path: str):
    """Yields (frame, source_filename) for every video/image in a folder."""
    for fname in sorted(os.listdir(folder_path)):
        full = os.path.join(folder_path, fname)
        ext = os.path.splitext(fname)[1].lower()
        if ext in VIDEO_EXTENSIONS:
            for frame in iter_frames_from_video(full):
                yield frame, fname
        elif ext in IMAGE_EXTENSIONS:
            frame = cv2.imread(full)
            if frame is not None:
                yield frame, fname


def main():
    parser = argparse.ArgumentParser(description="Evaluate the real pipeline on real-world recordings")
    parser.add_argument("--data_root", required=True,
                         help="Folder containing one subfolder per attack type (see module docstring)")
    parser.add_argument("--checkpoint", default=config.BEST_MODEL_PATH)
    parser.add_argument("--threshold", type=float, default=config.DECISION_THRESHOLD)
    parser.add_argument("--sample_every", type=int, default=5,
                         help="Sample every Nth frame from videos (avoids near-duplicate consecutive frames)")
    parser.add_argument("--out_csv", default=os.path.join(config.PREDICTIONS_DIR, "real_world_predictions.csv"))
    args = parser.parse_args()

    if not os.path.isdir(args.data_root):
        raise SystemExit(f"--data_root not found: {args.data_root}")

    detector = FaceDetector()
    predictor = LivenessPredictor(checkpoint_path=args.checkpoint, threshold=args.threshold)

    # per-attack-type: list of (predicted_label, true_label)
    results_by_type = defaultdict(list)
    rows = []
    n_no_face = 0

    attack_folders = sorted([
        d for d in os.listdir(args.data_root)
        if os.path.isdir(os.path.join(args.data_root, d))
    ])
    if not attack_folders:
        raise SystemExit(f"No subfolders found under {args.data_root}. See the module docstring for expected layout.")

    print(f"Found {len(attack_folders)} attack-type folders: {attack_folders}\n")

    for folder_name in attack_folders:
        folder_path = os.path.join(args.data_root, folder_name)
        true_label = config.LABEL_LIVE if is_genuine_folder(folder_name) else config.LABEL_SPOOF
        true_label_name = "LIVE" if true_label == config.LABEL_LIVE else "SPOOF"

        n_processed = 0
        for frame, source_file in collect_samples(folder_path):
            detections = detector.detect(frame)
            if not detections:
                n_no_face += 1
                continue

            best = max(detections, key=lambda d: d.width * d.height)
            crop = crop_face(frame, best)
            if crop is None or crop.size == 0:
                n_no_face += 1
                continue

            label, confidence = predictor.predict(crop)
            pred_label = config.LABEL_LIVE if label == "LIVE" else config.LABEL_SPOOF

            results_by_type[folder_name].append((pred_label, true_label))
            rows.append({
                "attack_type": folder_name,
                "source_file": source_file,
                "true_label": true_label_name,
                "pred_label": label,
                "confidence": f"{confidence:.4f}",
            })
            n_processed += 1

        print(f"[{folder_name}] ground truth = {true_label_name}, {n_processed} frames with a detected face processed")

    detector.close()

    if n_no_face:
        print(f"\n{n_no_face} frames had no detectable face and were skipped (not counted in metrics).")

    # Write per-sample predictions CSV.
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["attack_type", "source_file", "true_label", "pred_label", "confidence"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nPer-sample predictions saved to: {args.out_csv}")

    # Per-attack-type metrics.
    print("\n" + "=" * 70)
    print("REAL-WORLD EVALUATION -- BY ATTACK TYPE")
    print("=" * 70)
    print(f"{'Attack type':<24}{'N':<6}{'True label':<12}{'Accuracy':<12}{'Error rate'}")
    print("-" * 70)

    all_preds, all_trues = [], []
    for folder_name in attack_folders:
        pairs = results_by_type.get(folder_name, [])
        if not pairs:
            print(f"{folder_name:<24}{'0':<6}{'--':<12}{'no samples':<12}")
            continue
        preds = np.array([p for p, t in pairs])
        trues = np.array([t for p, t in pairs])
        all_preds.extend(preds.tolist())
        all_trues.extend(trues.tolist())

        is_genuine = is_genuine_folder(folder_name)
        acc = float(np.mean(preds == trues))
        if is_genuine:
            # error rate for a genuine-sample folder = BPCER contribution (live wrongly rejected)
            error_rate = float(np.mean(preds == config.LABEL_SPOOF))
            error_name = "BPCER"
        else:
            # error rate for an attack folder = APCER contribution (spoof wrongly accepted)
            error_rate = float(np.mean(preds == config.LABEL_LIVE))
            error_name = "APCER"
        print(f"{folder_name:<24}{len(pairs):<6}{'LIVE' if is_genuine else 'SPOOF':<12}{acc*100:>6.2f}%     {error_name}={error_rate*100:.2f}%")

    print("-" * 70)

    if all_preds:
        all_preds = np.array(all_preds)
        all_trues = np.array(all_trues)
        spoof_mask = all_trues == config.LABEL_SPOOF
        live_mask = all_trues == config.LABEL_LIVE

        overall_apcer = float(np.mean(all_preds[spoof_mask] == config.LABEL_LIVE)) if spoof_mask.sum() else float("nan")
        overall_bpcer = float(np.mean(all_preds[live_mask] == config.LABEL_SPOOF)) if live_mask.sum() else float("nan")
        overall_hter = (overall_apcer + overall_bpcer) / 2
        overall_acc = float(np.mean(all_preds == all_trues))

        print(f"\nOVERALL (pooled across all real-world folders, N={len(all_preds)}):")
        print(f"  Accuracy : {overall_acc*100:.2f}%")
        print(f"  APCER    : {overall_apcer*100:.2f}%  (pooled across all attack-type folders)")
        print(f"  BPCER    : {overall_bpcer*100:.2f}%  (pooled across all live folders)")
        print(f"  HTER     : {overall_hter*100:.2f}%")
        print("\nCompare this honestly against the CelebA-Spoof test-set numbers")
        print("(99.15% accuracy, 0.85% HTER) -- a real gap here is expected and is")
        print("exactly what this script exists to measure, not hide.")
    print("=" * 70)


if __name__ == "__main__":
    main()
