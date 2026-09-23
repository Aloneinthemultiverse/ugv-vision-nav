import cv2
import numpy as np
import pytest

from ugvnav.layer1.selfsup import (FEATURE_DIM, Prototype,
                                   SelfSupervisedTraversability, patch_features)
from ugvnav.layer2.water import WaterDetector


# ============================================================ water detection
def _terrain(h=180, w=240, seed=0):
    """Textured brown ground - the thing water must be distinguished from."""
    rng = np.random.default_rng(seed)
    base = np.zeros((h, w, 3), np.uint8)
    base[..., 0] = 120; base[..., 1] = 95; base[..., 2] = 65
    noise = rng.normal(0, 26, (h, w, 1))
    return np.clip(base + noise, 0, 255).astype(np.uint8)


def _add_puddle(img, y0, y1, x0, x1, seed=1):
    """A smooth, blue-shifted, near-textureless patch: a sky-reflecting puddle."""
    out = img.copy()
    rng = np.random.default_rng(seed)
    patch = np.zeros((y1 - y0, x1 - x0, 3), np.float32)
    patch[..., 0] = 128; patch[..., 1] = 146; patch[..., 2] = 178
    patch += rng.normal(0, 2.0, patch.shape)          # almost no texture
    out[y0:y1, x0:x1] = np.clip(patch, 0, 255).astype(np.uint8)
    return out


def test_smoothness_cue_is_high_on_flat_water():
    d = WaterDetector()
    img = _add_puddle(_terrain(), 110, 160, 60, 180)
    s = d.smoothness_cue(img)
    assert s[120:150, 80:160].mean() > s[20:60, 20:200].mean() + 0.2


def test_chroma_cue_prefers_blue_desaturated_regions():
    d = WaterDetector()
    img = _add_puddle(_terrain(), 110, 160, 60, 180)
    c = d.chroma_cue(img)
    assert c[120:150, 80:160].mean() > c[20:60, 20:200].mean()


def test_chroma_cue_is_zero_for_grayscale_input():
    assert np.all(WaterDetector().chroma_cue(np.zeros((30, 30), np.uint8)) == 0)


def test_reflection_cue_is_bounded():
    r = WaterDetector().reflection_cue(_terrain())
    assert r.shape == (180, 240)
    assert r.min() >= 0.0 and r.max() <= 1.0


def test_detects_a_puddle_and_not_the_surrounding_ground():
    d = WaterDetector(threshold=0.5, min_area=200)
    img = _add_puddle(_terrain(), 110, 160, 60, 180)
    res = d.process(img)

    inside = res.mask[118:155, 70:170].mean()
    outside = res.mask[10:70, 10:230].mean()
    assert inside > 0.5, "puddle not detected"
    assert outside < 0.15, "ground falsely flagged as water"


def test_clean_ground_produces_few_detections():
    res = WaterDetector(threshold=0.5, min_area=200).process(_terrain(seed=4))
    assert res.mask.mean() < 0.10


def test_height_rejects_things_that_stand_up():
    """Water lies flat. A smooth blue boulder is still a boulder."""
    d = WaterDetector(threshold=0.5, min_area=100)
    img = _add_puddle(_terrain(), 110, 160, 60, 180)

    flat = np.zeros(img.shape[:2], np.float32)
    raised = np.zeros(img.shape[:2], np.float32)
    raised[110:160, 60:180] = 0.8                    # 80 cm proud of the plane

    assert d.process(img, height=flat).mask.sum() > 0
    assert d.process(img, height=raised).mask.sum() == 0


def test_valid_mask_excludes_the_sky():
    """Sky is smooth and blue - without masking it scores as a lake."""
    img = _terrain()
    img[:60] = np.array([150, 175, 205], np.uint8)   # sky band
    d = WaterDetector(threshold=0.5, min_area=100)

    valid = np.ones(img.shape[:2], bool)
    valid[:60] = False
    assert d.process(img, valid=valid).mask[:60].sum() == 0


def test_result_fields_are_populated():
    res = WaterDetector().process(_terrain())
    for f in (res.score, res.reflection, res.smoothness, res.chroma):
        assert f.shape == (180, 240)
        assert np.isfinite(f).all()


def test_despeckle_removes_specks():
    d = WaterDetector(threshold=0.5, min_area=100000)   # nothing is big enough
    img = _add_puddle(_terrain(), 110, 160, 60, 180)
    assert d.process(img).mask.sum() == 0


# ====================================================== self-supervised adapter
def _two_terrains(h=128, w=128, seed=0):
    """Left half green 'grass', right half grey 'rock', both textured."""
    rng = np.random.default_rng(seed)
    img = np.zeros((h, w, 3), np.uint8)
    img[:, :w // 2] = [70, 140, 60]
    img[:, w // 2:] = [140, 140, 145]
    return np.clip(img + rng.normal(0, 10, img.shape), 0, 255).astype(np.uint8)


def test_patch_features_shape_and_range():
    f = patch_features(_two_terrains(), patch=16)
    assert f.shape == (8, 8, FEATURE_DIM)
    assert np.isfinite(f).all()


def test_patch_features_separate_different_terrain():
    f = patch_features(_two_terrains(), patch=16)
    left = f[:, :4].reshape(-1, FEATURE_DIM).mean(axis=0)
    right = f[:, 4:].reshape(-1, FEATURE_DIM).mean(axis=0)
    assert np.linalg.norm(left - right) > 0.1


def test_patch_features_handles_tiny_images():
    assert patch_features(np.zeros((4, 4, 3), np.uint8), patch=16).size == 0


def test_prototype_tracks_a_running_mean():
    p = Prototype()
    p.update(np.ones((10, FEATURE_DIM), np.float32))
    assert p.count == 10
    assert np.allclose(p.mean, 1.0)
    p.update(np.zeros((10, FEATURE_DIM), np.float32))
    assert np.allclose(p.mean, 0.5, atol=1e-6)


def test_prototype_ignores_empty_updates():
    p = Prototype()
    p.update(np.zeros((0, FEATURE_DIM), np.float32))
    assert p.count == 0


def test_adapter_has_no_opinion_before_it_has_driven():
    """The critical safety property: an untrained adapter must defer entirely."""
    a = SelfSupervisedTraversability()
    img = _two_terrains()
    assert a.confidence == 0.0
    assert np.allclose(a.predict(img), 0.5)

    base = np.zeros(img.shape[:2], bool)
    base[:, :64] = True
    assert np.array_equal(a.refine(base, img), base)     # unchanged


def test_confidence_grows_with_experience():
    a = SelfSupervisedTraversability(min_evidence=10, full_evidence=100)
    img = _two_terrains()
    driven = np.zeros(img.shape[:2], bool); driven[:, :64] = True
    blocked = np.zeros(img.shape[:2], bool); blocked[:, 64:] = True

    assert a.confidence == 0.0
    for _ in range(3):
        a.learn(img, driven=driven, blocked=blocked)
    grown = a.confidence
    assert grown > 0.0
    for _ in range(10):
        a.learn(img, driven=driven, blocked=blocked)
    assert a.confidence > grown
    assert a.confidence <= 1.0


def test_learn_reports_what_it_harvested():
    a = SelfSupervisedTraversability()
    img = _two_terrains()
    driven = np.zeros(img.shape[:2], bool); driven[:, :64] = True
    pos, neg = a.learn(img, driven=driven)
    assert pos > 0 and neg == 0


def test_adapter_learns_which_terrain_was_driven():
    """The WVN insight, asserted: terrain we drove over is traversable."""
    a = SelfSupervisedTraversability(min_evidence=10, full_evidence=60)
    img = _two_terrains()
    driven = np.zeros(img.shape[:2], bool); driven[:, :56] = True
    blocked = np.zeros(img.shape[:2], bool); blocked[:, 72:] = True

    for _ in range(12):
        a.learn(img, driven=driven, blocked=blocked)

    belief = a.predict(img)
    assert belief[:, :56].mean() > belief[:, 72:].mean() + 0.1


def test_refine_can_correct_a_mislabelled_obstacle():
    """The measured failure: the pre-trained model calls rock traversable.
    After driving, experience must be able to overrule it."""
    a = SelfSupervisedTraversability(min_evidence=5, full_evidence=40)
    img = _two_terrains()
    driven = np.zeros(img.shape[:2], bool); driven[:, :56] = True
    blocked = np.zeros(img.shape[:2], bool); blocked[:, 72:] = True
    for _ in range(20):
        a.learn(img, driven=driven, blocked=blocked)

    wrong = np.ones(img.shape[:2], bool)        # semantics: "all drivable"
    fixed = a.refine(wrong, img)
    assert fixed[:, 80:].mean() < wrong[:, 80:].mean()


def test_refine_is_conservative_at_low_confidence():
    a = SelfSupervisedTraversability(min_evidence=10, full_evidence=10000)
    img = _two_terrains()
    driven = np.zeros(img.shape[:2], bool); driven[:, :64] = True
    blocked = np.zeros(img.shape[:2], bool); blocked[:, 64:] = True
    for _ in range(3):
        a.learn(img, driven=driven, blocked=blocked)

    base = np.ones(img.shape[:2], bool)
    changed = (a.refine(base, img) != base).mean()
    assert a.confidence < 0.05
    assert changed < 0.35, "barely-trained adapter overrode too much"
