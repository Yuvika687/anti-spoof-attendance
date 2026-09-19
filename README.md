# Mask-Aware Liveness / Anti-Spoofing Access Gate

**CSE471: Deep Learning for Computer Vision — Course Assignment**

A real-time face liveness detection module that blocks printed-photo, screen-replay,
and video-replay spoofing attacks on a face-based attendance system, while still
working correctly when the person is wearing a face mask. This repo implements
**Stage 1 (mask-aware face detection)** and **Stage 2 (CNN liveness classification)**
of the pipeline, and exposes a clean integration point for a teammate's existing
**ArcFace recognition + attendance logging** code.

---

## 1. Project Summary

Face-attendance systems that only check "is there a face" can be fooled by holding
up a printed photo, or replaying a video/photo on a phone or laptop screen. This
project adds a **liveness gate** in front of face recognition:

1. A face is detected in the webcam frame — using MediaPipe's BlazeFace detector,
   which keeps working even when the lower half of the face is covered by a mask,
   because it localizes faces mainly from the eyes/eyebrows/nose-bridge region.
2. The detected face is cropped (with margin) and fed to a **CNN binary classifier**
   (MobileNetV3-Small or EfficientNet-B0 backbone, fine-tuned) that predicts
   **LIVE** or **SPOOF**.
3. Only faces classified as **LIVE** are passed further down the pipeline to
   ArcFace recognition and attendance marking. **SPOOF** faces are rejected
   immediately and never reach the recognition/attendance stage.

The model is trained on a curated ~24–30k image subset of **CelebA-Spoof** and
additionally evaluated (never trained) on **CASIA-FASD** to check generalization
to a different camera/attack distribution.

---

## 2. Pipeline Diagram (text)

```
                 ┌───────────────────┐
                 │   Webcam Frame     │
                 └─────────┬──────────┘
                            │
                            ▼
      ┌──────────────────────────────────────────┐
      │ STAGE 1 — Mask-Aware Face Detection        │
      │ (MediaPipe BlazeFace, face_utils.py)       │
      │  -> works even with mask covering mouth    │
      └─────────────────────┬──────────────────────┘
                            │  bounding box
                            ▼
                 ┌───────────────────┐
                 │   Crop + Margin    │
                 │  (face_utils.py)   │
                 └─────────┬──────────┘
                            │  face crop (BGR)
                            ▼
      ┌──────────────────────────────────────────┐
      │ STAGE 2 — CNN Liveness Classifier          │
      │ (MobileNetV3 / EfficientNet, model.py)     │
      │  -> outputs P(spoof) via liveness_infer.py │
      └─────────────────────┬──────────────────────┘
                            │
              ┌─────────────┴─────────────┐
              ▼                            ▼
     P(spoof) >= threshold          P(spoof) < threshold
              │                            │
              ▼                            ▼
     ┌─────────────────┐          ┌──────────────────┐
     │  SPOOF DETECTED  │          │       LIVE        │
     │  red box, reject │          │  green box, pass   │
     │  logged, no      │          │  to ArcFace        │
     │  attendance      │          │  mark_attendance() │
     └─────────────────┘          └──────────────────┘
                                            │
                                            ▼
                                ┌───────────────────────┐
                                │ ArcFace recognition +  │
                                │ attendance logging      │
                                │ (partner's module,      │
                                │  plugged into           │
                                │  mark_attendance())     │
                                └───────────────────────┘
```

---

## 3. Repository Structure

```
anti_spoof_attendance/
├── data/
│   ├── celeba_spoof/        # train/val/test subset (live/spoof folders)
│   ├── casia_fasd/          # generalization test set only
│   └── samples/masked/      # a few masked-face photos for Stage 1 check
├── models/
│   └── best_liveness.pth    # produced by train.py
├── src/
│   ├── config.py             # all shared paths / hyperparameters
│   ├── dataset.py             # LivenessDataset + transforms
│   ├── model.py                # LivenessCNN (MobileNetV3 / EfficientNet-B0)
│   ├── train.py                # training loop, checkpointing, plots
│   ├── evaluate.py             # accuracy, APCER, BPCER, HTER, confusion matrix
│   ├── face_utils.py           # MediaPipe mask-aware detection + cropping
│   ├── liveness_infer.py       # single-crop / batch inference wrapper
│   ├── realtime_pipeline.py    # webcam demo + mark_attendance() placeholder
│   ├── prepare_celeba_subset.py# builds identity-disjoint CelebA-Spoof subset
│   ├── prepare_casia_fasd.py   # extracts face crops from CASIA-FASD videos
│   ├── test_masked_samples.py  # Stage 1 sanity check on masked photos
│   ├── baseline_opencv.py      # classical blur-based baseline (optional)
│   └── export_onnx.py          # ONNX export for faster inference (optional)
├── notebooks/
│   └── demo_analysis.ipynb     # loads plots/metrics for the report
├── outputs/
│   ├── plots/                  # training curves, confusion matrices
│   └── predictions/            # metrics JSON, per-sample CSVs, sample crops
├── logs/
│   └── attendance_log.csv      # every accepted/rejected frame decision
├── requirements.txt
├── README.md
└── report_notes.md
```

---

## 4. Setup

```bash
cd anti_spoof_attendance
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Requires Python 3.9–3.11 (MediaPipe does not yet support the very latest
Python versions on every platform — if `pip install mediapipe` fails, use
Python 3.10 or 3.11).

---

## 5. Dataset Setup

### 5.1 CelebA-Spoof (primary training data)

1. Official source: [ZhangYuanhan-AI/CelebA-Spoof on GitHub](https://github.com/ZhangYuanhan-AI/CelebA-Spoof)
   (625,537 images, 10,177 identities, non-commercial research use). Its
   README links to:
   - Google Drive folder: 50 split archive parts named
     `CelebA_Spoof.zip.001` … `CelebA_Spoof.zip.050` (~1 GB each, ~50 GB total)
   - Baidu Netdisk (password `61fd`), same content
2. Download all 50 parts, then reassemble and extract:
   ```bash
   cat CelebA_Spoof.zip.* > CelebA_Spoof_full.zip
   unzip CelebA_Spoof_full.zip -d CelebA_Spoof
   ```
   (Downloading 50 files by hand from Drive is tedious — the tool `gdown`
   can pull an entire shared Drive folder in one command:
   `pip install gdown && gdown --folder <the_drive_folder_url>`.)
3. The dataset's own label ground-truth (`metas/*.json`) uses index `43` of
   a 43-attribute-per-image vector as the live(0)/spoof(1) label, and ships
   a bounding-box `.txt` file alongside every image. In practice, once
   extracted, images are organized per-identity as
   `Data/train/<identity_id>/{live,spoof}/*.png` and
   `Data/test/<identity_id>/{live,spoof}/*.png` — which is exactly what
   `prepare_celeba_subset.py` expects. **After extracting, run
   `find CelebA_Spoof -maxdepth 4 | head -50` and check it matches this
   pattern before running the subset builder** — tell me if it looks
   different and I'll adjust the script to match.
4. Build a manageable, identity-disjoint subset (do **not** use all ~625k images):

```bash
cd src
python prepare_celeba_subset.py \
    --raw_root /path/to/CelebA_Spoof \
    --out_root ../data/celeba_spoof \
    --live_total 13000 --spoof_total 13000 \
    --train_frac 0.70 --val_frac 0.15 --test_frac 0.15 \
    --copy_mode symlink
```

This produces `data/celeba_spoof/{train,val,test}/{live,spoof}/*.jpg`, with
**no identity appearing in more than one split**.

### 5.2 CASIA-FASD (extra generalization test set only)

CASIA-FASD has 20 training subjects + 30 testing subjects; each subject has
12 videos (3 genuine + 9 attack: warped-photo, cut-photo, video-replay),
600 videos total. It has no simple public download link like CelebA-Spoof —
you typically need to request access from the original CBSR/CASIA
authors, though third-party mirrors occasionally surface (search "CASIA
Face Anti-Spoofing Database download"). This dataset is optional/lower
priority — the project works and can be reported on using CelebA-Spoof
alone; CASIA-FASD only adds a generalization data point.

1. Once you have it, sort the `.avi` clips into two flat folders using the
   genuine/attack split documented with your copy of the dataset:

   ```
   casia_fasd_sorted/
       live/   *.avi
       spoof/  *.avi
   ```

2. Extract face-crop frames:

```bash
python prepare_casia_fasd.py \
    --raw_root /path/to/casia_fasd_sorted \
    --out_root ../data/casia_fasd/test \
    --frames_per_video 5
```

CASIA-FASD is used **only** with `evaluate.py`, never with `train.py`.

### 5.3 Masked face samples (Stage 1 check only)

Drop 5–10 photos of masked faces (phone photos are fine) into
`data/samples/masked/`, then run:

```bash
python test_masked_samples.py --images_dir ../data/samples/masked
```

---

## 6. How to Train

```bash
cd src
python train.py \
    --data_root ../data/celeba_spoof \
    --backbone mobilenet_v3_small \
    --epochs 15 --batch_size 32 --lr 3e-4 \
    --image_size 224 --freeze_epochs 2
```

- Trains a head-only warm-up for `--freeze_epochs` epochs, then unfreezes the
  full backbone for fine-tuning at a lower learning rate.
- Saves `models/last_liveness.pth` every epoch and `models/best_liveness.pth`
  whenever validation accuracy improves.
- Saves `outputs/plots/training_curves.png` (loss + accuracy curves) and
  `outputs/plots/training_history.json`.

To use EfficientNet-B0 instead:

```bash
python train.py --data_root ../data/celeba_spoof --backbone efficientnet_b0 --epochs 15
```

---

## 7. How to Evaluate

```bash
cd src

# Held-out CelebA-Spoof test split
python evaluate.py --data_root ../data/celeba_spoof/test \
    --checkpoint ../models/best_liveness.pth --tag test

# Generalization check on CASIA-FASD (never trained on)
python evaluate.py --data_root ../data/casia_fasd/test \
    --checkpoint ../models/best_liveness.pth --tag casia_fasd

# Optional classical baseline for comparison
python baseline_opencv.py --data_root ../data/celeba_spoof/test
```

`evaluate.py` prints Accuracy, APCER, BPCER, HTER, per-class precision/recall,
and the confusion matrix, and saves:
- `outputs/plots/confusion_matrix_<tag>.png`
- `outputs/predictions/metrics_<tag>.json`
- `outputs/predictions/predictions_<tag>.csv` (per-image predictions)

---

## 8. How to Run the Real-Time Webcam Demo

```bash
cd src
python realtime_pipeline.py --checkpoint ../models/best_liveness.pth
```

Optional flags:

```bash
# stricter spoof detection (raise threshold on P(spoof))
python realtime_pipeline.py --threshold 0.6

# use a different camera device
python realtime_pipeline.py --camera 1

# also save periodic accepted/rejected face crops for the report
python realtime_pipeline.py --save_samples --sample_every 15
```

Behavior:
- Green box + "LIVE (xx.x%)" for real faces -> `mark_attendance()` is called.
- Red box + "SPOOF - REJECTED (xx.x%)" for attacks -> attendance is skipped.
- FPS shown top-left; "No face detected" shown when nothing is found.
- Every decision (LIVE/SPOOF, confidence, bbox) is appended to
  `logs/attendance_log.csv`.
- Press **q** to quit.

### Plugging in ArcFace

`src/realtime_pipeline.py` defines:

```python
def mark_attendance(face_crop_bgr, confidence, timestamp):
    ...
```

This is called **only** for faces already confirmed LIVE, and receives the
exact BGR face crop (same format OpenCV/ArcFace expect). Your partner should
replace the placeholder body with their ArcFace embedding + matching +
attendance-DB logging code — no other changes to the pipeline are needed.

---

## 9. Optional: ONNX Export

```bash
cd src
python export_onnx.py --checkpoint ../models/best_liveness.pth --out ../models/liveness.onnx
```

---

## 10. Understanding APCER and BPCER

- **APCER (Attack Presentation Classification Error Rate)** — the percentage
  of *spoof* samples that were wrongly classified as *live*. This is the
  dangerous error for an access-control system: it means an attacker with a
  printed photo or a screen replay got through.
- **BPCER (Bona Fide Presentation Classification Error Rate)** — the
  percentage of *live* (real) samples that were wrongly classified as
  *spoof*. This is the annoying error: a genuine person is rejected and has
  to try again.
- **HTER (Half Total Error Rate)** = `(APCER + BPCER) / 2` — a single summary
  number balancing both error types, standard in anti-spoofing literature
  (equivalent in spirit to averaging false-accept and false-reject rates).

A good liveness system keeps **APCER low** (security) without making
**BPCER** so high that real users get rejected constantly (usability). The
`--threshold` flag lets you trade one off against the other without
retraining: raising the threshold lowers APCER but raises BPCER, and vice versa.

---

## 11. Suggested Results Table (for the report)

| Metric | CelebA-Spoof (test) | CASIA-FASD (generalization) |
|---|---|---|
| Accuracy | | |
| APCER | | |
| BPCER | | |
| HTER | | |
| Precision (LIVE) | | |
| Recall (LIVE) | | |
| Precision (SPOOF) | | |
| Recall (SPOOF) | | |

| Model | Backbone | Params | Accuracy | HTER |
|---|---|---|---|---|
| Classical baseline | Laplacian-variance heuristic | — | | |
| Ours | MobileNetV3-Small | ~1.5M (head) | | |
| Ours (optional 2nd run) | EfficientNet-B0 | ~4.0M (head) | | |

Fill these in from `outputs/predictions/metrics_test.json` and
`outputs/predictions/metrics_casia_fasd.json` after running `evaluate.py`.

---

## 12. Notes / Known Limitations

- CelebA-Spoof labels follow the convention `0 = live, 1 = spoof`; this is
  preserved throughout the codebase (`config.LABEL_LIVE`, `config.LABEL_SPOOF`).
- The identity-disjoint split (`prepare_celeba_subset.py`) prevents the model
  from memorizing specific faces instead of learning general liveness cues.
- CASIA-FASD is intentionally never used for training — it exists purely to
  measure how well the model generalizes beyond CelebA-Spoof's capture setup.
- The real-time pipeline processes one webcam stream on CPU; on a typical
  laptop with MobileNetV3-Small expect roughly 10–20 FPS depending on
  hardware and the number of faces in frame.
