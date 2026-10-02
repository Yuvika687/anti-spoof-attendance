"""
liveness_infer.py
------------------
Thin, reusable inference wrapper around a trained LivenessCNN checkpoint.
Used by realtime_pipeline.py, and can also be imported directly by a
teammate's ArcFace/attendance script if they want a single function call:

    from liveness_infer import LivenessPredictor
    predictor = LivenessPredictor("../models/best_liveness.pth")
    label, confidence = predictor.predict(face_crop_bgr)
"""

from typing import Tuple

import numpy as np
import torch

import config
from dataset import get_eval_transform
from face_utils import bgr_crop_to_pil
from model import build_model


class LivenessPredictor:
    """Loads a trained checkpoint once and exposes fast single-crop inference."""

    def __init__(self, checkpoint_path: str = config.BEST_MODEL_PATH,
                 device: torch.device = config.DEVICE,
                 threshold: float = config.DECISION_THRESHOLD):
        self.device = device
        self.threshold = threshold

        checkpoint = torch.load(checkpoint_path, map_location=device)
        self.backbone = checkpoint.get("backbone", config.BACKBONE)
        self.image_size = checkpoint.get("image_size", config.IMAGE_SIZE)

        self.model = build_model(backbone=self.backbone, pretrained=False).to(device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        self.transform = get_eval_transform(self.image_size)

    @torch.no_grad()
    def predict_prob(self, face_crop_bgr: np.ndarray) -> float:
        """Returns the raw, unsmoothed P(spoof) for a single crop, with no
        thresholding applied. Useful for callers that want to do their own
        temporal smoothing across frames (see webapp/backend/app.py)."""
        pil_image = bgr_crop_to_pil(face_crop_bgr)
        tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
        logit = self.model(tensor)
        return torch.sigmoid(logit).item()

    def predict(self, face_crop_bgr: np.ndarray) -> Tuple[str, float]:
        """
        Args:
            face_crop_bgr: cropped face image, BGR (as cropped by face_utils.crop_face).

        Returns:
            (label, confidence) where label is "LIVE" or "SPOOF" and
            confidence is the model's confidence in that label, in [0, 1].
        """
        prob_spoof = self.predict_prob(face_crop_bgr)

        if prob_spoof >= self.threshold:
            return config.CLASS_NAMES[config.LABEL_SPOOF], prob_spoof
        else:
            return config.CLASS_NAMES[config.LABEL_LIVE], 1.0 - prob_spoof

    @torch.no_grad()
    def predict_batch(self, face_crops_bgr: list) -> list:
        """Batched version for efficiency when multiple faces appear in one frame."""
        if not face_crops_bgr:
            return []

        tensors = torch.stack([
            self.transform(bgr_crop_to_pil(crop)) for crop in face_crops_bgr
        ]).to(self.device)

        logits = self.model(tensors)
        probs_spoof = torch.sigmoid(logits).cpu().numpy()

        results = []
        for p in probs_spoof:
            if p >= self.threshold:
                results.append((config.CLASS_NAMES[config.LABEL_SPOOF], float(p)))
            else:
                results.append((config.CLASS_NAMES[config.LABEL_LIVE], float(1.0 - p)))
        return results

    def set_threshold(self, threshold: float):
        """Allows the real-time pipeline / a UI slider to tune sensitivity live."""
        self.threshold = threshold


if __name__ == "__main__":
    import sys
    import cv2

    if len(sys.argv) < 2:
        print("Usage: python liveness_infer.py <path_to_face_image>")
        sys.exit(1)

    image_path = sys.argv[1]
    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not read image: {image_path}")
        sys.exit(1)

    predictor = LivenessPredictor()
    label, confidence = predictor.predict(image)
    print(f"Prediction: {label}  (confidence={confidence:.4f})")
