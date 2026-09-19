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
| CelebA-Spoof (subset) | Train | [FILL IN] | [FILL IN] | identity-disjoint train/val/test |
| CelebA-Spoof (subset) | Val | [FILL IN] | [FILL IN] | |
| CelebA-Spoof (subset) | Test | [FILL IN] | [FILL IN] | |
| CASIA-FASD | Extra test only | [FILL IN] | [FILL IN] | never used for training |

Identity-disjoint splitting (`src/prepare_celeba_subset.py`) assigns whole
identity folders to exactly one of train/val/test, preventing the model
from learning to recognize specific people's faces as a shortcut for
liveness, which would not generalize to new users of the attendance system.

## 5. Experimental Setup

- Backbone: `[FILL IN: mobilenet_v3_small / efficientnet_b0]`
- Input size: `[FILL IN: 128 / 224]`
- Epochs: `[FILL IN]`  (freeze_epochs = `[FILL IN]`)
- Batch size: `[FILL IN]`
- Optimizer: AdamW, LR = `[FILL IN]`, weight decay = `[FILL IN]`
- Hardware: `[FILL IN: e.g. Apple M1 CPU / NVIDIA GTX 1650]`
- Decision threshold used for reported metrics: `[FILL IN]`

## 6. Results

### 6.1 Training curves

`[INSERT outputs/plots/training_curves.png]`

### 6.2 CelebA-Spoof held-out test set

| Metric | Value |
|---|---|
| Accuracy | [FILL IN] |
| APCER | [FILL IN] |
| BPCER | [FILL IN] |
| HTER | [FILL IN] |
| Precision (LIVE) | [FILL IN] |
| Recall (LIVE) | [FILL IN] |
| Precision (SPOOF) | [FILL IN] |
| Recall (SPOOF) | [FILL IN] |

`[INSERT outputs/plots/confusion_matrix_test.png]`

### 6.3 CASIA-FASD generalization test

| Metric | Value |
|---|---|
| Accuracy | [FILL IN] |
| APCER | [FILL IN] |
| BPCER | [FILL IN] |
| HTER | [FILL IN] |

Discuss: does HTER increase noticeably on CASIA-FASD compared to the
CelebA-Spoof test set? If so, this indicates the model has partially
overfit to CelebA-Spoof-specific capture characteristics (camera sensor,
compression, typical attack presentation style) rather than learning fully
camera-agnostic liveness cues. `[FILL IN discussion]`

### 6.4 Classical baseline comparison

`src/baseline_opencv.py` implements a simple Laplacian-variance
sharpness heuristic (spoof media tends to look slightly blurrier / more
uniform in texture than a live face, because it is "a photo of a photo/
screen"). Compare its accuracy/APCER/BPCER against the CNN to show the
CNN's improvement over a non-deep-learning approach. `[FILL IN table]`

### 6.5 Stage 1 mask-detection check

Using `src/test_masked_samples.py` on a small local set of masked-face
photos: `[FILL IN] / [FILL IN]` faces detected (`[FILL IN]%` detection
rate), confirming Stage 1 remains functional with masks on.

### 6.6 Real-time performance

Observed FPS on `[FILL IN hardware]`: `[FILL IN]` FPS with one face in
frame. `[FILL IN any qualitative notes: e.g. behavior under different
lighting, phone-replay attempts, printed-photo attempts]`

## 7. Discussion

- **Why APCER/BPCER instead of just accuracy:** `[FILL IN — reference the
  APCER/BPCER explanation in README.md section 10]`
- **Failure cases observed:** `[FILL IN — e.g. glossy printed photos under
  bright light sometimes misclassified as live; very dim lighting increases
  BPCER]`
- **Effect of the decision threshold:** `[FILL IN — try 2-3 threshold values
  with evaluate.py --threshold and report how APCER/BPCER trade off]`
- **Mask-awareness impact:** `[FILL IN — any qualitative observations from
  the realtime demo/masked sample check]`

## 8. Conclusion

`[FILL IN — 3-5 sentences summarizing what was built, key results, and one
concrete future-work idea, e.g. adding depth/IR sensing, temporal
(multi-frame) liveness cues, or a learned attack-type classifier instead of
binary live/spoof]`

## 9. Individual Contribution Note

`[FILL IN — this module: mask-aware face detection integration point,
liveness CNN training/evaluation pipeline, real-time OpenCV demo. Partner:
ArcFace recognition and attendance database integration via
mark_attendance().]`
