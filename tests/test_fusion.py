import numpy as np
import pytest

from ugvnav.fusion import (SAFE, CAUTION, LETHAL, UNKNOWN, COST,
                           fuse, geometry_override)


@pytest.fixture
def flat_grass():
    """A 20x20 scene that semantics calls grass and geometry calls flat."""
    labels = np.full((20, 20), 2, dtype=np.uint8)     # grass = traversable
    height = np.zeros((20, 20), dtype=np.float32)
    return labels, height


# ------------------------------------------------------- geometry override
def test_geometry_override_catches_a_buried_rock(flat_grass):
    """The headline claim: appearance says grass, shape says rock. Shape wins."""
    labels, height = flat_grass
    height[5:10, 5:10] = 0.4                          # a lump sticking up

    override = geometry_override(labels, height, spike_m=0.22)

    assert override[5:10, 5:10].all()
    assert not override[12:, 12:].any()


def test_geometry_override_ignores_bumps_below_threshold(flat_grass):
    labels, height = flat_grass
    height[5:10, 5:10] = 0.1                          # just a tussock
    assert not geometry_override(labels, height, spike_m=0.22).any()


def test_geometry_override_only_applies_to_traversable_classes():
    """A spike on an already-lethal class is not an *override*, just an obstacle."""
    labels = np.full((10, 10), 4, dtype=np.uint8)     # obstacle class
    height = np.full((10, 10), 0.9, dtype=np.float32)
    assert not geometry_override(labels, height).any()


def test_geometry_override_makes_the_cell_lethal(flat_grass):
    labels, height = flat_grass
    height[5:10, 5:10] = 0.5

    res = fuse(labels, height)

    assert (res.grid[5:10, 5:10] == LETHAL).all()
    assert (res.grid[15:, 15:] == SAFE).all()
    assert res.reasons["geometry_override"] == 25


# ------------------------------------------------------------------ fusion
def test_untouched_traversable_ground_is_safe(flat_grass):
    labels, height = flat_grass
    assert (fuse(labels, height).grid == SAFE).all()


def test_non_traversable_semantics_is_lethal(flat_grass):
    labels, height = flat_grass
    labels[0:5, :] = 3                                 # vegetation
    res = fuse(labels, height)
    assert (res.grid[0:5, :] == LETHAL).all()


def test_negative_obstacle_marks_lethal(flat_grass):
    labels, height = flat_grass
    neg = np.zeros((20, 20), bool)
    neg[10:14, 2:6] = True
    res = fuse(labels, height, negative=neg)
    assert (res.grid[10:14, 2:6] == LETHAL).all()
    assert res.reasons["negative_obstacle"] == 16


def test_dynamic_and_overhead_mark_lethal(flat_grass):
    labels, height = flat_grass
    dyn = np.zeros((20, 20), bool); dyn[0:3, 0:3] = True
    over = np.zeros((20, 20), bool); over[17:, 17:] = True
    res = fuse(labels, height, dynamic=dyn, overhead=over)
    assert (res.grid[0:3, 0:3] == LETHAL).all()
    assert (res.grid[17:, 17:] == LETHAL).all()


def test_uncertainty_only_downgrades_to_caution(flat_grass):
    labels, height = flat_grass
    unc = np.zeros((20, 20), np.float32)
    unc[4:8, 4:8] = 0.9
    res = fuse(labels, height, uncertainty=unc, uncertainty_thresh=0.6)
    assert (res.grid[4:8, 4:8] == CAUTION).all()


def test_uncertainty_never_downgrades_a_lethal_cell(flat_grass):
    """The most pessimistic claim must survive - CAUTION cannot erase LETHAL."""
    labels, height = flat_grass
    neg = np.zeros((20, 20), bool); neg[4:8, 4:8] = True
    unc = np.ones((20, 20), np.float32)

    res = fuse(labels, height, negative=neg, uncertainty=unc)

    assert (res.grid[4:8, 4:8] == LETHAL).all()
    assert (res.grid[12:, 12:] == CAUTION).all()


def test_valid_mask_produces_unknown(flat_grass):
    labels, height = flat_grass
    valid = np.zeros((20, 20), bool); valid[10:, :] = True
    res = fuse(labels, height, valid=valid)
    assert (res.grid[:10, :] == UNKNOWN).all()
    assert (res.grid[10:, :] == SAFE).all()


def test_costmap_uses_nav2_values(flat_grass):
    labels, height = flat_grass
    neg = np.zeros((20, 20), bool); neg[0:2, 0:2] = True
    unc = np.zeros((20, 20), np.float32); unc[5:7, 5:7] = 0.95

    cm = fuse(labels, height, negative=neg, uncertainty=unc).costmap

    assert cm[0, 0] == 254
    assert cm[5, 5] == 128
    assert cm[15, 15] == 0
    assert COST[LETHAL] == 254 and COST[CAUTION] == 128 and COST[SAFE] == 0


def test_reasons_are_reported_without_double_counting(flat_grass):
    """A cell claimed by two rules is attributed to the first, strongest one."""
    labels, height = flat_grass
    height[3:6, 3:6] = 0.6
    neg = np.zeros((20, 20), bool); neg[3:6, 3:6] = True

    res = fuse(labels, height, negative=neg)

    assert res.reasons["geometry_override"] == 9
    assert res.reasons["negative_obstacle"] == 0     # already lethal
    assert (res.grid == LETHAL).sum() == 9
