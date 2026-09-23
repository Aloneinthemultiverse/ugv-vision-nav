"""Node B+ - self-supervised traversability adaptation.

Our RELLIS-3D benchmark measured the dominant error precisely: SegFormer labels
35 % of true obstacle pixels as "trail", because it is trained on ADE20k where
off-road bush looks like *earth* or *field*. Fine-tuning would fix it, and needs
annotated off-road data we do not have.

Wild Visual Navigation (Frey et al., RSS 2023) supplies the insight that makes
labels free: **terrain the vehicle has already driven over is, by definition,
traversable.** The robot's own experience is the annotation.

We use the prototype formulation (arXiv 2504.12109) rather than WVN's online
gradient training, because prototypes need no backward pass and stay within the
CPU-only budget. Two running prototypes are maintained in a small hand-built
feature space; new patches are classified by relative distance, and the result
is blended with the pre-trained semantic prediction according to how much
evidence has actually been gathered.

The adapter is never trusted before it has seen enough terrain - it reports its
own confidence, and with no experience it defers to the pre-trained model
entirely.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

__all__ = ["patch_features", "Prototype", "PrototypeSet",
           "SelfSupervisedTraversability"]

#: Length of the per-patch feature vector produced by :func:`patch_features`.
FEATURE_DIM = 7


def patch_features(rgb: np.ndarray, patch: int = 16,
                   height: np.ndarray | None = None) -> np.ndarray:
    """Describe every patch of an image with a compact appearance vector.

    Deliberately hand-built and cheap - colour statistics, texture energy and
    optionally geometry - rather than a ViT embedding, so this runs on CPU at
    frame rate.

    Args:
        rgb: (H, W, 3) uint8 image.
        patch: patch size in pixels.
        height: optional (H, W) height-above-ground field, averaged per patch.

    Returns:
        (rows, cols, FEATURE_DIM) float32 features:
        mean R/G/B, saturation, value, texture energy, mean height.
    """
    rgb = np.asarray(rgb, np.uint8)
    h, w = rgb.shape[:2]
    rows, cols = h // patch, w // patch
    if rows == 0 or cols == 0:
        return np.zeros((0, 0, FEATURE_DIM), np.float32)

    img = rgb[:rows * patch, :cols * patch].astype(np.float32) / 255.0
    hsv = cv2.cvtColor(rgb[:rows * patch, :cols * patch], cv2.COLOR_RGB2HSV)
    gray = cv2.cvtColor(rgb[:rows * patch, :cols * patch], cv2.COLOR_RGB2GRAY)
    tex = np.abs(cv2.Laplacian(gray.astype(np.float32), cv2.CV_32F)) / 255.0

    def pool(a):
        return a.reshape(rows, patch, cols, patch).mean(axis=(1, 3))

    feats = [pool(img[..., 0]), pool(img[..., 1]), pool(img[..., 2]),
             pool(hsv[..., 1].astype(np.float32) / 255.0),
             pool(hsv[..., 2].astype(np.float32) / 255.0),
             pool(tex)]

    if height is not None:
        hh = np.asarray(height, np.float32)[:rows * patch, :cols * patch]
        feats.append(np.clip(pool(hh), -2.0, 2.0) / 2.0)
    else:
        feats.append(np.zeros((rows, cols), np.float32))

    return np.stack(feats, axis=-1).astype(np.float32)


@dataclass
class Prototype:
    """Running mean of the features of one class, with an evidence count."""
    mean: np.ndarray = field(default_factory=lambda: np.zeros(FEATURE_DIM, np.float32))
    count: float = 0.0

    def update(self, features: np.ndarray, weight: float = 1.0) -> None:
        f = np.asarray(features, np.float32).reshape(-1, FEATURE_DIM)
        if f.size == 0:
            return
        n = f.shape[0] * weight
        total = self.count + n
        self.mean = ((self.mean * self.count + f.sum(axis=0) * weight) / total
                     ).astype(np.float32)
        self.count = float(total)

    def distance(self, features: np.ndarray) -> np.ndarray:
        f = np.asarray(features, np.float32)
        return np.linalg.norm(f - self.mean, axis=-1)


class PrototypeSet:
    """A multi-modal class model: several centroids rather than one mean.

    A single mean cannot describe a class whose members form separate clusters.
    Off-road "traversable" is exactly that - dirt, grass and gravel differ
    sharply in colour and texture, and their average resembles none of them.

    Following the online-clustering formulation of arXiv 2504.12109, a sample
    close to an existing centroid updates it; a sample far from all of them
    creates a new centroid, so the model grows to fit terrain it has never seen
    without any offline clustering pass.

    Args:
        max_prototypes: cap on centroids, so an outdoor run cannot grow without
            bound.
        novelty: distance beyond which a sample starts a new centroid rather
            than updating the nearest one.
        momentum: update rate for an existing centroid.
    """

    def __init__(self, max_prototypes: int = 24, novelty: float = 0.22,
                 momentum: float = 0.08) -> None:
        self.max_prototypes = int(max_prototypes)
        self.novelty = float(novelty)
        self.momentum = float(momentum)
        self.centroids: list[np.ndarray] = []
        self.weights: list[float] = []
        self.count = 0.0

    def __len__(self) -> int:
        return len(self.centroids)

    @property
    def mean(self) -> np.ndarray:
        """Weighted mean of all centroids, for comparison and diagnostics."""
        if not self.centroids:
            return np.zeros(FEATURE_DIM, np.float32)
        w = np.asarray(self.weights, np.float32)[:, None]
        return (np.stack(self.centroids) * w).sum(axis=0) / max(w.sum(), 1e-6)

    def update(self, features: np.ndarray, weight: float = 1.0) -> None:
        """Assign each sample to its nearest centroid, or start a new one."""
        f = np.asarray(features, np.float32).reshape(-1, FEATURE_DIM)
        if f.size == 0:
            return
        for v in f:
            if not self.centroids:
                self.centroids.append(v.copy()); self.weights.append(weight)
                continue
            d = np.linalg.norm(np.stack(self.centroids) - v, axis=1)
            i = int(np.argmin(d))
            if d[i] > self.novelty and len(self.centroids) < self.max_prototypes:
                self.centroids.append(v.copy()); self.weights.append(weight)
            else:
                m = self.momentum
                self.centroids[i] = ((1.0 - m) * self.centroids[i] + m * v).astype(np.float32)
                self.weights[i] += weight
        self.count += float(f.shape[0] * weight)

    def distance(self, features: np.ndarray) -> np.ndarray:
        """Distance to the NEAREST centroid - the multi-modal match."""
        f = np.asarray(features, np.float32)
        if not self.centroids:
            return np.full(f.shape[:-1], np.inf, np.float32)
        c = np.stack(self.centroids)                       # (K, D)
        d = np.linalg.norm(f[..., None, :] - c, axis=-1)   # (..., K)
        return d.min(axis=-1).astype(np.float32)


class SelfSupervisedTraversability:
    """Online traversability adapter trained by driving, not by annotation.

    Args:
        patch: patch size in pixels, matching :func:`patch_features`.
        min_evidence: patches required in *both* prototypes before the adapter
            is trusted at all.
        full_evidence: evidence at which the adapter is trusted fully.
        margin: distance ratio below which a patch is called traversable.
        max_prototypes: centroids retained per class.
        novelty: distance beyond which a sample starts a new centroid.
    """

    def __init__(self, patch: int = 16, min_evidence: float = 40.0,
                 full_evidence: float = 400.0, margin: float = 1.0,
                 max_prototypes: int = 24, novelty: float = 0.22) -> None:
        self.patch = int(patch)
        self.min_evidence = float(min_evidence)
        self.full_evidence = float(full_evidence)
        self.margin = float(margin)
        # Multi-modal by default: a single mean cannot represent a class whose
        # members form separate clusters, and measurement says recall is our
        # weak axis. See PrototypeSet.
        self.traversable = PrototypeSet(max_prototypes, novelty)
        self.obstacle = PrototypeSet(max_prototypes, novelty)

    # ------------------------------------------------------------- learning
    @property
    def evidence(self) -> float:
        """Experience gathered so far, limited by the weaker of the two classes."""
        return min(self.traversable.count, self.obstacle.count)

    @property
    def confidence(self) -> float:
        """How far to trust this adapter over the pre-trained model, in [0, 1]."""
        if self.evidence < self.min_evidence:
            return 0.0
        span = max(self.full_evidence - self.min_evidence, 1e-6)
        return float(np.clip((self.evidence - self.min_evidence) / span, 0.0, 1.0))

    def learn(self, rgb: np.ndarray, driven: np.ndarray | None = None,
              blocked: np.ndarray | None = None,
              height: np.ndarray | None = None) -> tuple[int, int]:
        """Harvest labels from experience.

        Args:
            rgb: the frame.
            driven: (H, W) bool - ground the vehicle actually traversed without
                incident. These are free positive labels.
            blocked: (H, W) bool - ground confirmed non-traversable, from a
                collision or a lethal costmap cell.
            height: optional height field, used as a feature.

        Returns:
            (positive_patches, negative_patches) added.
        """
        feats = patch_features(rgb, self.patch, height)
        if feats.size == 0:
            return (0, 0)
        added = [0, 0]
        for proto, mask, idx in ((self.traversable, driven, 0),
                                 (self.obstacle, blocked, 1)):
            if mask is None:
                continue
            m = self._pool_mask(np.asarray(mask, bool), feats.shape[:2])
            if m.any():
                proto.update(feats[m])
                added[idx] = int(m.sum())
        return tuple(added)                               # type: ignore[return-value]

    def _pool_mask(self, mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
        rows, cols = shape
        p = self.patch
        m = mask[:rows * p, :cols * p].astype(np.float32)
        return m.reshape(rows, p, cols, p).mean(axis=(1, 3)) > 0.6

    # ------------------------------------------------------------ inference
    def predict(self, rgb: np.ndarray,
                height: np.ndarray | None = None) -> np.ndarray:
        """Per-pixel traversability belief in [0, 1]. 0.5 means "no opinion"."""
        h, w = rgb.shape[:2]
        if self.confidence <= 0.0:
            return np.full((h, w), 0.5, np.float32)

        feats = patch_features(rgb, self.patch, height)
        if feats.size == 0:
            return np.full((h, w), 0.5, np.float32)

        d_trav = self.traversable.distance(feats)
        d_obst = self.obstacle.distance(feats)
        # Relative distance: near the traversable prototype -> near 1.
        belief = d_obst / (d_trav + d_obst + 1e-6)
        belief = np.clip((belief - 0.5) * self.margin + 0.5, 0.0, 1.0)
        return cv2.resize(belief.astype(np.float32), (w, h),
                          interpolation=cv2.INTER_LINEAR)

    def refine(self, semantic_traversable: np.ndarray, rgb: np.ndarray,
               height: np.ndarray | None = None,
               accept: float = 0.6, reject: float = 0.4) -> np.ndarray:
        """Correct a pre-trained semantic mask using gathered experience.

        The adapter only overrides the pre-trained model in proportion to its
        own confidence, so an untrained adapter changes nothing.

        Args:
            semantic_traversable: (H, W) bool from Node B.
            accept: belief above which the adapter may mark a pixel traversable.
            reject: belief below which it may mark a pixel non-traversable.

        Returns:
            (H, W) bool corrected traversability mask.
        """
        base = np.asarray(semantic_traversable, bool)
        c = self.confidence
        if c <= 0.0:
            return base

        belief = self.predict(rgb, height)
        out = base.copy()
        # Only flip where the adapter is decisive, scaled by its confidence.
        strong_yes = belief > (accept + (1.0 - c) * (1.0 - accept))
        strong_no = belief < reject * c
        out[strong_yes] = True
        out[strong_no] = False
        return out
