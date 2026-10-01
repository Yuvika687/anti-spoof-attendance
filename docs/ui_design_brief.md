# UI Design Brief — Mask-Aware Liveness / Anti-Spoofing Access Gate

CSE471 — Deep Learning for Computer Vision

This is a design reference for building a UI around this project (e.g. in
Google Stitch or any other design tool). It lists the real pipeline, real
results, and the screens worth designing, with real data instead of
placeholders — pull numbers from here rather than inventing new ones.

---

## Project overview

**Name:** Mask-Aware Liveness / Anti-Spoofing Access Gate

**Problem it solves:** Face-attendance systems can be fooled by a printed
photo or a phone/screen replay held up to the camera. This system adds a
"liveness gate" in front of face recognition — it checks whether the
camera sees a real, live person (even wearing a mask) before letting
attendance be marked. Spoofed faces are rejected before they ever reach
identity recognition.

**Pipeline (4 stages — the core visual metaphor for any UI):**
1. **Webcam Frame** — raw 1920×1080 camera input
2. **Detect + Crop** — MediaPipe BlazeFace finds the face (works even with
   a mask covering nose/mouth, since it uses eye/eyebrow landmarks)
3. **Classify** — a fine-tuned MobileNetV3-Small CNN scores the cropped
   face as LIVE or SPOOF
4. **Decision** — LIVE → green box, passed to attendance; SPOOF → red box,
   rejected, logged

**Color convention** (already baked into the real app — keep this in any
UI design): **green = LIVE/accepted**, **red = SPOOF/rejected**.

---

## Real results (use these exact numbers)

| Metric | Value |
|---|---|
| Accuracy | 99.15% |
| APCER (spoof wrongly accepted as live) | 1.13% |
| BPCER (live wrongly rejected as spoof) | 0.56% |
| HTER | 0.85% |
| Live-webcam FPS (full pipeline) | 26.7 |
| Training images | 26,000 (identity-disjoint: 18,200 train / 3,900 val / 3,900 test) |
| Training epochs | 15 |
| Classical baseline comparison | 61.2% accuracy (vs. 99.15% for this model) |

**Confusion matrix (3,900 test images):**
- 1,939 live correctly accepted, 11 live wrongly rejected
- 1,928 spoof correctly rejected, 22 spoof wrongly accepted

**Threshold trade-off (for a settings/tuner screen):**

| Threshold | APCER | BPCER |
|---|---|---|
| 0.3 (looser) | 0.82% | 0.82% |
| 0.5 (default) | 1.13% | 0.56% |
| 0.7 (stricter) | 1.59% | 0.51% |

---

## Screens worth designing

### 1. Landing / Overview
- Project name, one-line pitch
- The 4-stage pipeline as a visual flow (webcam → detect → classify → decision)
- Headline stats: 99.15% accuracy, 0.85% HTER, 26.7 FPS
- "How it works" in plain language

### 2. Live Detection View (the actual working app)
- Webcam feed, mirrored
- Bounding box around the detected face: **green** + "LIVE (98.8%)" or
  **red** + "SPOOF — REJECTED (91.2%)"
- FPS counter, top corner
- "No face detected" state (orange text) when nobody's in frame
- A small toast/line when attendance is marked: "✓ Attendance marked"
  (only fires on LIVE)

### 3. Results / Analytics Dashboard
- Big-number tiles: Accuracy, APCER, BPCER, HTER
- Training curve chart (loss + accuracy over 15 epochs — train line vs.
  val line, val climbs from 93.7% → 99.21%)
- Confusion matrix (2×2 grid: Live/Spoof actual vs. predicted)
- Bar comparison: classical baseline (61.2%) vs. this model (99.15%)

### 4. Decision Log / History
A table/list of real-time decisions, each row: timestamp, decision
(colored LIVE/SPOOF tag), confidence %, bounding box coords (optional).

Example real-format rows (from the actual CSV log schema):
```
2026-09-30 19:32:04   LIVE    98.80%
2026-09-30 19:32:05   LIVE    99.30%
2026-09-30 19:32:06   SPOOF   91.18%
```

### 5. Settings / Threshold Tuner (optional but nice)
- A slider for the decision threshold (0.0–1.0, default 0.5)
- Live readout of how APCER/BPCER trade off as you move it (see table
  above)
- Model info: backbone (MobileNetV3-Small), input size (224×224), device
  (Apple MPS)

---

## Tone / feel

Given the subject (security checkpoint, biometric verification), directions
that fit well: a security/access-control console feel, a lab-instrument
feel, or a clean modern SaaS-dashboard feel — whichever matches what you
want to present. The one thing worth keeping regardless of visual style:
**green/red as the LIVE/SPOOF semantic color**, since that's literally how
the running app behaves.
