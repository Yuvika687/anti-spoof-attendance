# Report Notes — Mask-Aware Liveness / Anti-Spoofing Access Gate

CSE471: Deep Learning for Computer Vision

Use this as a skeleton for the CA-1 written report. Sections marked
`[FILL IN]` should be completed after running training and evaluation.

---

## 1. Problem Statement

Face-based attendance systems verify identity using face recognition, but a
camera cannot by itself tell whether it is looking at a *live person* or a
*presentation attack*: a printed photo held up to the camera, a photo/video
replayed on a phone or laptop screen, or a video recording of the real
person. Such attacks let an unauthorized person be marked present under
someone else's identity. Additionally, many real deployments (post-2020)
require the system to keep working when users wear face masks, which
classical face-detection pipelines tuned on unmasked faces can fail on.

**Goal:** build a liveness-detection gate that sits between face detection
and face recognition in an attendance pipeline, so that recognition (and
therefore attendance marking) only ever runs on presentation attacks that
have already been rejected... i.e. only on confirmed *live* faces — while
remaining robust to masked faces at the detection stage.

## 2. Related Work / Background (brief)

- **Presentation Attack Detection (PAD)** is the standard term in the
  biometrics literature for liveness/anti-spoofing detection.
- Two broad families of approaches: **texture/frequency-based classical
  methods** (e.g. analyzing moire patterns, blur, color distortion from
  screen replays) and **deep-learning based methods** that learn spoof cues
  directly from data (CNNs, often initialized from ImageNet pretraining).
- **CelebA-Spoof** (Zhang et al.) is a large-scale face anti-spoofing
  dataset with rich per-image spoof-type/illumination/environment
  annotations, organized by identity, which we rely on for training and for
  the identity-disjoint split used here.
- **CASIA-FASD** is an older, smaller face anti-spoofing dataset recorded
  with different cameras and attack types (photo, cut-photo, video-replay),
  which we use purely to test **generalization** since it was never seen
  during training.
- We report **APCER**, **BPCER**, and **HTER**, the standard metrics used in
  the ISO/IEC 30107-3 presentation attack detection standard, rather than
  raw accuracy alone, since accuracy can be misleading under class imbalance
  and does not distinguish "attacker got through" from "real user rejected".

## 3. Method / Pipeline

### 3.1 Stage 1 — Mask-Aware Face Detection

We use MediaPipe's short-range BlazeFace detector (`model_selection=0`).
This model localizes faces primarily using upper-face landmarks (eyes,
eyebrows, nose bridge), which remain visible when a mask covers the
mouth/nose, making it robust to masked faces without any additional
mask-specific training. See `src/face_utils.py`.

### 3.2 Face Cropping

The detected bounding box is expanded by a margin (`FACE_CROP_MARGIN =
0.25` of box size) before cropping. This is important for liveness
specifically: a too-tight crop can hide spoof artifacts that appear near the
face border, such as a screen bezel, the edge of printed photo paper, or a
hand holding up an attack medium.

### 3.3 Stage 2 — CNN Liveness Classifier

- **Backbone:** MobileNetV3-Small (default) or EfficientNet-B0, both
  pretrained on ImageNet and fine-tuned end-to-end.
- **Head:** Dropout -> Linear(feature_dim, 128) -> ReLU -> Dropout ->
  Linear(128, 1), producing a single spoof-logit.
- **Loss:** `BCEWithLogitsLoss` with `pos_weight` set from the train-split
  class ratio to counter mild class imbalance.
- **Training schedule:** a short head-only warm-up (`--freeze_epochs`, backbone
  frozen) followed by full fine-tuning at a reduced learning rate, with a
  cosine-annealed LR schedule.
- **Augmentation:** horizontal flip, small rotation, color jitter
  (brightness/contrast/saturation/hue), random Gaussian blur, and random
  erasing — chosen specifically because they mimic real variation across
  live vs. spoof capture conditions (different cameras, lighting, focus,
  and partial occlusion such as masks).

### 3.4 Decision and Integration

At inference, the model outputs `P(spoof)`. If `P(spoof) >= threshold`
(default 0.5, tunable without retraining) the face is rejected as SPOOF; a
red bounding box and "SPOOF - REJECTED" label are drawn, and the frame is
logged. Otherwise the face is accepted as LIVE (green box, "LIVE" label),
logged, and passed to `mark_attendance()` — a clean placeholder function
that a teammate's ArcFace recognition and attendance-logging code plugs
into directly, receiving the already-cropped BGR face image.

## 4. Dataset

| Dataset | Role | Live images | Spoof images | Notes |
|---|---|---|---|---|
| CelebA-Spoof (subset) | Train | 9,100 | 9,100 | identity-disjoint train/val/test |
| CelebA-Spoof (subset) | Val | 1,950 | 1,950 | |
| CelebA-Spoof (subset) | Test | 1,950 | 1,950 | |
| CASIA-FASD | Extra test only | not obtained | not obtained | requires access request to original authors; skipped for this submission |

Subset drawn from the full 625,537-image, 10,177-identity CelebA-Spoof
release, split across 9,196 identities found in the download: 6,437 train /
1,379 val / 1,380 test identities, with **zero identity overlap** between
splits.

Identity-disjoint splitting (`src/prepare_celeba_subset.py`) assigns whole
identity folders to exactly one of train/val/test, preventing the model
from learning to recognize specific people's faces as a shortcut for
liveness, which would not generalize to new users of the attendance system.

## 5. Experimental Setup

- Backbone: `mobilenet_v3_small` (ImageNet-pretrained)
- Input size: `224 x 224`
- Epochs: `15` total (`freeze_epochs = 2`: 2 epochs head-only warm-up, then full fine-tuning from epoch 3)
- Batch size: `32`
- Optimizer: AdamW, LR = `3e-4` (warm-up) dropping to `3e-4 * 0.3` on unfreeze, weight decay = `1e-4`, cosine annealing schedule
- Hardware: Apple Silicon Mac (MPS backend)
- Decision threshold used for reported metrics: `0.5` (default, on P(spoof))

## 6. Results

### 6.1 Training curves

`[INSERT outputs/plots/training_curves.png]`

Training accuracy rose from 85.6% (epoch 1) to 99.5% (epoch 15); validation
accuracy rose from 93.2% to a peak of **98.64%** at epoch 6, then plateaued
in the 98.3–98.6% range for the remaining epochs with no signs of
overfitting collapse (val loss stayed flat/low rather than rising while
train loss kept falling, i.e. only mild, expected train/val divergence).

### 6.2 CelebA-Spoof held-out test set

Evaluated on 3,900 test images (1,950 live / 1,950 spoof) from 1,380
identities never seen during training:

| Metric | Value |
|---|---|
| Accuracy | 99.03% |
| APCER | 1.23% |
| BPCER | 0.72% |
| HTER | 0.97% |
| Precision (LIVE) | 0.9878 |
| Recall (LIVE) | 0.9928 |
| Precision (SPOOF) | 0.9928 |
| Recall (SPOOF) | 0.9877 |

Confusion matrix: of 1,950 live images, 14 were wrongly rejected as spoof;
of 1,950 spoof images, 24 were wrongly accepted as live.

`[INSERT outputs/plots/confusion_matrix_test.png]`

### 6.3 CASIA-FASD generalization test

**Not run.** CASIA-FASD has no simple public download (requires a request
to the original CBSR/CASIA authors); it was not obtained in time for this
submission. Per the project spec this is an optional extra generalization
check, not a requirement — the model's real, held-out-identity performance
on CelebA-Spoof (Section 6.2) already demonstrates it generalizes to unseen
people, just not yet to a different camera/dataset distribution. `[If you
obtain CASIA-FASD later: run prepare_casia_fasd.py, then
evaluate.py --data_root ../data/casia_fasd/test --tag casia_fasd, then
paste the results here.]`

### 6.4 Classical baseline comparison

`src/baseline_opencv.py` implements a simple Laplacian-variance
sharpness heuristic (spoof media tends to look slightly blurrier / more
uniform in texture than a live face, because it is "a photo of a photo/
screen"), thresholded to best-separate the same CelebA-Spoof test set:

| Model | Accuracy | APCER | BPCER |
|---|---|---|---|
| Classical baseline (Laplacian-variance) | 60.79% | 47.33% | 31.08% |
| **Ours (MobileNetV3-Small CNN)** | **99.03%** | **1.23%** | **0.72%** |

The huge gap (60.8% vs. 99.0% accuracy) makes the case for a learned
approach concretely: a single hand-crafted sharpness statistic barely beats
chance-level separation on modern, high-quality spoof media, whereas the
CNN learns much richer, harder-to-fake texture/frequency cues.

### 6.5 Stage 1 mask-detection check

**Not yet run with real masked photos.** `src/face_utils.py`'s detector was
verified separately on an unmasked test photo (94.4% detection confidence),
confirming Stage 1 works in general, but the mask-specific check needs a
few of your own masked-face photos: `[TODO — drop 5-10 photos of yourself/
others wearing a mask into data/samples/masked/, then run
python test_masked_samples.py and paste the detection rate here]`.

### 6.6 Real-time performance

**Not yet run.** `[TODO — run python realtime_pipeline.py --checkpoint
../models/best_liveness.pth on your own webcam, note the FPS shown
on-screen, and try a printed photo / phone-screen replay to see it
correctly rejected]`.

## 7. Discussion

- **Why APCER/BPCER instead of just accuracy:** accuracy treats a false
  accept (an attacker getting through) and a false reject (a real user
  being annoyed) as equally bad, when in an access-control system they are
  not — APCER measures the security-critical error, BPCER the usability
  one (see README.md Section 10 for the full explanation).

- **Failure cases observed** (inspected directly from
  `outputs/predictions/predictions_test.csv`, 38 wrong out of 3,900 total):
  the model's most confident **false accepts** (spoof→live) were printed
  photos where the face itself was sharply focused under even studio-style
  lighting, with the surrounding photo/poster content occupying much of the
  frame — i.e. cases where the spoof medium's face region looks
  photographically "clean" rather than screen-blurred or moiré-textured.
  The model's most confident **false rejects** (live→spoof) were,
  surprisingly, live photos that themselves have heavy blur and a
  fine repeating scanline/moiré-like texture across the whole image
  (consistent with the original photo having been captured off a screen
  or from low-quality broadcast footage before being included in the
  dataset as a "live" sample). This suggests the model has partly learned
  "blur + moiré texture -> spoof" as a shortcut cue, which is usually
  correct but misfires on the rare live photo that happens to share that
  texture for unrelated reasons (e.g. a poor-quality source photo).

- **Effect of the decision threshold** (from `evaluate.py --threshold`,
  same test set):

  | Threshold | Accuracy | APCER | BPCER | HTER |
  |---|---|---|---|---|
  | 0.3 (looser) | 99.15% | 0.82% | 0.87% | 0.85% |
  | 0.5 (default) | 99.03% | 1.23% | 0.72% | 0.97% |
  | 0.7 (stricter) | 98.95% | 1.54% | 0.56% | 1.05% |

  Raising the threshold trades lower BPCER (fewer real users annoyed) for
  higher APCER (more attacks slip through) — the classic security/usability
  tradeoff. For an access-control use case where letting an attacker
  through is worse than asking a real user to retry, a **lower** threshold
  (e.g. 0.3) is arguably the better operating point here, since it achieves
  both the lowest APCER *and* the lowest HTER of the three tried.

- **Mask-awareness impact:** `[FILL IN once 6.5/6.6 are run with real
  masked photos and the live webcam demo]`

## 8. Conclusion

This project built a two-stage liveness gate — mask-aware MediaPipe face
detection followed by a fine-tuned MobileNetV3-Small binary classifier —
that sits in front of face recognition in an attendance pipeline and
blocks presentation attacks before they ever reach identity matching.
Trained on a 26,000-image identity-disjoint subset of CelebA-Spoof, the
final model reached **99.03% accuracy and 0.97% HTER** on 3,900 held-out
test images from identities never seen during training, and massively
outperformed a classical Laplacian-sharpness baseline (60.8% accuracy),
demonstrating the value of a learned approach over hand-crafted heuristics
for this task. The main limitation is that generalization was only tested
within CelebA-Spoof's own capture distribution — CASIA-FASD cross-dataset
testing and real-webcam/masked-photo validation were not completed for
this submission (see Sections 6.3, 6.5, 6.6). `[Optionally add a concrete
future-work idea here once you've run the webcam demo, e.g. temporal
(multi-frame) liveness cues, or obtaining CASIA-FASD for cross-dataset
validation.]`

## 9. Individual Contribution Note

`[FILL IN — this module: mask-aware face detection integration point,
liveness CNN training/evaluation pipeline, real-time OpenCV demo. Partner:
ArcFace recognition and attendance database integration via
mark_attendance().]`
