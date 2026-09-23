"""Node H - water and mud detection.

Our RELLIS-3D benchmark measured 82.3 % of labelled water as false-safe: mud and
puddles were read as trail or grass and would have been driven into. Water is
non-traversable for good reason - a UGV that enters a puddle of unknown depth
may not come out.

Polarisation is the strongest cue in the literature (Nguyen et al., 2017) and
needs a filter we do not have. The ECCV 2018 reflection-attention network has no
public implementation. So this detector encodes the three monocular cues
directly as physics rather than learning them:

  * **Reflection** - water is a mirror, so a patch of water correlates with the
    scene vertically above it. This is the reflection-attention idea of
    Han et al. (2018) computed explicitly instead of learned.
  * **Smoothness** - still water carries almost no texture, so local intensity
    variance collapses relative to surrounding ground.
  * **Sky chroma** - a puddle reflects the sky, so it shifts blue and loses
    saturation compared with nearby soil.

Each cue alone is weak and produces false positives on shadow or smooth rock.
Combined, and restricted to the ground region, they are usable.

STATUS: EXPERIMENTAL - DOES NOT WORK ON REAL DATA YET
-----------------------------------------------------
Measured against RELLIS-3D water labels (52 frames, 38 containing water) by
``scripts/benchmark_water.py``:

    threshold 0.45 (best)     recall  5.5 %   precision 11.9 %
    threshold 0.55 (default)  recall  0.6 %   precision 20.2 %
    semantic-only baseline    recall 17.7 %

**It is roughly three times worse than the baseline it was written to beat.**
It is therefore NOT wired into the pipeline or the fusion layer, and must not be
until it clears 17.7 % recall.

Why it fails, and why the unit tests did not catch it: the synthetic puddle in
the tests is smooth, blue and sky-reflecting, which is what a puddle on tarmac
looks like. RELLIS "withwater" is dominated by **mud** - brown, textured, in
shallow depressions, reflecting nothing. The sky-chroma cue is not merely weak
there, it points the wrong way, and the smoothness cue is weak because wet soil
keeps its texture.

Fixing this needs mud-specific cues (darkness relative to surrounding soil,
saturation increase from wetness, position in terrain depressions) rather than
tuning these three. The tests were validating an assumption, not reality.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

__all__ = ["WaterResult", "WaterDetector"]


@dataclass
class WaterResult:
    mask: np.ndarray        # (H, W) bool, pixels believed to be water or mud
    score: np.ndarray       # (H, W) float32 in [0, 1], combined confidence
    reflection: np.ndarray  # (H, W) float32, mirror-correlation cue
    smoothness: np.ndarray  # (H, W) float32, texture-collapse cue
    chroma: np.ndarray      # (H, W) float32, sky-colour cue


class WaterDetector:
    """Monocular water and mud detection from reflection, texture and colour.

    Args:
        threshold: combined score above which a pixel is called water.
        w_reflection, w_smoothness, w_chroma: cue weights; they are normalised,
            so only their ratio matters.
        patch: half-height in pixels of the vertical window used to test the
            mirror hypothesis.
        min_area: connected components smaller than this are discarded, since a
            puddle is never a handful of scattered pixels.
    """

    def __init__(self, threshold: float = 0.55, w_reflection: float = 1.0,
                 w_smoothness: float = 1.0, w_chroma: float = 0.8,
                 patch: int = 12, min_area: int = 150) -> None:
        self.threshold = float(threshold)
        total = float(w_reflection + w_smoothness + w_chroma)
        self.w = (w_reflection / total, w_smoothness / total, w_chroma / total)
        self.patch = int(patch)
        self.min_area = int(min_area)

    # ------------------------------------------------------------------ cues
    @staticmethod
    def _gray(rgb: np.ndarray) -> np.ndarray:
        rgb = np.asarray(rgb)
        if rgb.ndim == 3:
            return cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)
        return rgb.astype(np.float32)

    def reflection_cue(self, rgb: np.ndarray) -> np.ndarray:
        """How well does each row resemble the scene mirrored above it?

        A puddle at row ``r`` shows an inverted copy of whatever stands behind
        it. Comparing a downward window against the upward window flipped is a
        direct test of that hypothesis.
        """
        g = self._gray(rgb)
        h, w = g.shape
        p = self.patch
        blur = cv2.GaussianBlur(g, (0, 0), 1.5)
        out = np.zeros((h, w), np.float32)

        for r in range(p, h - p):
            below = blur[r:r + p, :]                 # the candidate surface
            above = blur[r - p:r, :][::-1, :]        # the scene, mirrored
            b = below - below.mean(axis=0, keepdims=True)
            a = above - above.mean(axis=0, keepdims=True)
            denom = (np.linalg.norm(b, axis=0) * np.linalg.norm(a, axis=0)) + 1e-6
            out[r, :] = np.clip((b * a).sum(axis=0) / denom, 0.0, 1.0)

        return cv2.GaussianBlur(out, (0, 0), 3.0)

    def smoothness_cue(self, rgb: np.ndarray) -> np.ndarray:
        """Texture collapse: still water is far smoother than soil or grass."""
        g = self._gray(rgb)
        mean = cv2.blur(g, (9, 9))
        var = cv2.blur(g * g, (9, 9)) - mean * mean
        sd = np.sqrt(np.maximum(var, 0.0))
        hi = float(np.percentile(sd, 85)) + 1e-6
        return np.clip(1.0 - sd / hi, 0.0, 1.0).astype(np.float32)

    def chroma_cue(self, rgb: np.ndarray) -> np.ndarray:
        """Sky reflection: puddles shift blue and lose saturation versus soil."""
        rgb = np.asarray(rgb)
        if rgb.ndim != 3:
            return np.zeros(rgb.shape[:2], np.float32)
        hsv = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
        sat = hsv[..., 1] / 255.0
        f = rgb.astype(np.float32) + 1e-6
        blueness = f[..., 2] / (f.sum(axis=-1) / 3.0)      # B relative to mean
        desaturated = np.clip(1.0 - sat / 0.45, 0.0, 1.0)
        blue = np.clip((blueness - 0.95) / 0.35, 0.0, 1.0)
        return cv2.GaussianBlur((0.5 * desaturated + 0.5 * blue).astype(np.float32),
                                (0, 0), 3.0)

    # --------------------------------------------------------------- combine
    def process(self, rgb: np.ndarray, valid: np.ndarray | None = None,
                height: np.ndarray | None = None,
                flat_tol: float = 0.12) -> WaterResult:
        """Args:
            rgb: (H, W, 3) uint8 image.
            valid: optional bool mask of analysable ground; sky must be excluded
                or its own smoothness and blueness score as water.
            height: optional height-above-ground field. Water lies flat, so
                anything standing proud of the plane is rejected.
            flat_tol: metres above the ground plane still considered flat.

        Returns:
            WaterResult.
        """
        rgb = np.asarray(rgb)
        h, w = rgb.shape[:2]
        valid = np.ones((h, w), bool) if valid is None else np.asarray(valid, bool)

        refl = self.reflection_cue(rgb)
        smooth = self.smoothness_cue(rgb)
        chroma = self.chroma_cue(rgb)

        wr, ws, wc = self.w
        score = (wr * refl + ws * smooth + wc * chroma).astype(np.float32)

        if height is not None:                   # water cannot stand up
            score = score * (np.asarray(height, np.float32) < flat_tol)
        score = score * valid

        mask = score >= self.threshold
        mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN,
                                np.ones((5, 5), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                                np.ones((11, 11), np.uint8)).astype(bool)
        mask = self._despeckle(mask)
        return WaterResult(mask=mask, score=score, reflection=refl,
                           smoothness=smooth, chroma=chroma)

    def _despeckle(self, mask: np.ndarray) -> np.ndarray:
        if self.min_area <= 1 or not mask.any():
            return mask
        n, lab, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
        keep = np.zeros_like(mask)
        for i in range(1, n):
            if stats[i, cv2.CC_STAT_AREA] >= self.min_area:
                keep |= lab == i
        return keep
