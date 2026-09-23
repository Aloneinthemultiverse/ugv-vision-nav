"""Node B - semantic segmentation.

Maps pixels to a small traversability vocabulary. The real backend is SegFormer
(Apache-2.0, ADE20k). As with depth, the network sits behind an interface so
the fusion logic can be tested without downloading weights.
"""
from __future__ import annotations

from typing import Protocol

import numpy as np

__all__ = ["TERRAIN_CLASSES", "TRAVERSABLE", "Segmenter",
           "SemanticSegmenter", "StubSegmenter", "traversable_mask"]

MODEL_ID = "nvidia/segformer-b0-finetuned-ade-512-512"

#: Compact vocabulary the navigation stack reasons about.
TERRAIN_CLASSES = {
    0: "unknown",
    1: "trail",      # dirt, path, road surface
    2: "grass",      # low vegetation, drivable with caution
    3: "vegetation", # bush, tree, dense scrub
    4: "obstacle",   # rock, structure, vehicle, person
    5: "sky",
    6: "water",
}

#: Classes the planner may drive across, and their base cost multiplier.
TRAVERSABLE = {1: 0.0, 2: 0.35}

# ADE20k label id -> our compact class id.
_ADE_MAP = {
    3: 1, 6: 1, 11: 1, 13: 1, 29: 1, 52: 1,     # road / path / earth / field
    9: 2, 46: 2,                                 # grass
    4: 3, 17: 3, 66: 3, 72: 3,                   # tree / plant / bush
    2: 5,                                        # sky
    21: 6, 26: 6, 60: 6, 109: 6, 128: 6,         # water / sea / river / lake
    12: 4, 20: 4, 76: 4, 80: 4, 83: 4, 127: 4,   # person / car / rock / structure
}


class Segmenter(Protocol):
    def infer(self, rgb: np.ndarray) -> np.ndarray:  # pragma: no cover - protocol
        ...


class SemanticSegmenter:
    """SegFormer wrapper returning our compact class ids.

    The published checkpoint ships no ``preprocessor_config.json``, so the
    image processor is constructed explicitly rather than auto-loaded. Working
    from the model's own ``id2label`` table also means the ADE20k vocabulary is
    read from the checkpoint instead of being hard-coded here.
    """

    def __init__(self, model_id: str = MODEL_ID, device: str = "cpu") -> None:
        import torch
        from transformers import (SegformerForSemanticSegmentation,
                                  SegformerImageProcessor)

        self._torch = torch
        self._proc = SegformerImageProcessor(do_resize=True, size={"height": 512,
                                                                  "width": 512})
        self._model = SegformerForSemanticSegmentation.from_pretrained(model_id)
        self._model.eval().to(device)
        self._device = device
        self._lut = {int(i): _name_to_class(str(n))
                     for i, n in self._model.config.id2label.items()}

    def infer(self, rgb: np.ndarray) -> np.ndarray:
        import cv2

        rgb = np.asarray(rgb, dtype=np.uint8)
        inputs = self._proc(images=rgb, return_tensors="pt").to(self._device)
        with self._torch.no_grad():
            logits = self._model(**inputs).logits           # (1, C, h, w)
        ade = logits.argmax(dim=1)[0].cpu().numpy().astype(np.int32)
        ade = cv2.resize(ade, (rgb.shape[1], rgb.shape[0]),
                         interpolation=cv2.INTER_NEAREST)

        out = np.zeros(ade.shape, dtype=np.uint8)
        for ade_id, compact in self._lut.items():
            if compact:
                out[ade == ade_id] = compact
        return out


def _name_to_class(name: str) -> int:
    """Map an ADE20k label *name* to our compact vocabulary."""
    n = name.lower()
    if any(k in n for k in ("sky",)):
        return 5
    if any(k in n for k in ("water", "sea", "river", "lake", "pool")):
        return 6
    if any(k in n for k in ("road", "path", "earth", "dirt", "sand", "land", "field", "floor")):
        return 1
    if "grass" in n:
        return 2
    if any(k in n for k in ("tree", "plant", "bush", "flower", "palm")):
        return 3
    if any(k in n for k in ("person", "car", "truck", "rock", "stone", "building",
                            "wall", "fence", "pole", "house", "van", "animal")):
        return 4
    return 0


class StubSegmenter:
    """Deterministic stand-in used by the test suite."""

    def __init__(self, labels: np.ndarray) -> None:
        self._labels = np.asarray(labels, dtype=np.uint8)

    def infer(self, rgb: np.ndarray) -> np.ndarray:
        return self._labels.copy()


def traversable_mask(labels: np.ndarray) -> np.ndarray:
    """Boolean mask of pixels semantics believes are drivable."""
    labels = np.asarray(labels)
    out = np.zeros(labels.shape, dtype=bool)
    for cid in TRAVERSABLE:
        out |= labels == cid
    return out
