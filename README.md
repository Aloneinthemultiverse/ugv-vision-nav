# ugvnav — vision-based autonomous navigation for UGVs

Monocular perception and hazard resolution for an unmanned ground vehicle
driving in unstructured outdoor terrain, **without GPS and without LiDAR**.

Reference implementation for **SIH26126** — *Vision Based Autonomous Navigation
for Unmanned Ground Vehicle for Outdoor Environment* (Bharat Electronics
Limited, theme *Smart Automation*, category *Software*).

```
68 tests · pure NumPy/OpenCV core · CPU only · no ROS required to run or test
```

---

## Why this exists

A camera returns colour. It does not return danger. Five situations defeat
appearance-based perception outdoors, and each one is capable of destroying a
vehicle:

| # | Situation | Why appearance fails |
|---|-----------|----------------------|
| 1 | Ditches and drop-offs | In a photograph a hole and a shadow look identical |
| 2 | Moving hazards | One frame cannot say where a rolling rock will be in two seconds |
| 3 | Rocks buried in dirt | The obstacle is wearing the terrain, so segmentation says "soil" |
| 4 | Low branches | The ground is clear; the cargo deck is not |
| 5 | Glare and deep shadow | Networks do not go quiet when confused — they stay confident |

`ugvnav` answers all five with **geometry**, and fuses the answers into a single
safety grid the planner consumes.

---

## The central idea: geometry outranks appearance

```python
from ugvnav import fuse, geometry_override

# semantics is confident this is drivable grass
labels = np.full((20, 20), 2, np.uint8)

# but the height field says something is sticking 40 cm out of it
height = np.zeros((20, 20), np.float32)
height[5:10, 5:10] = 0.4

result = fuse(labels, height)
result.grid[7, 7]        # LETHAL — the rock was caught
result.reasons           # {'geometry_override': 25, ...}
result.costmap[7, 7]     # 254, the Nav2 lethal value
```

A half-buried boulder is labelled *grass* because that is what its surface looks
like. Its **shape** gives it away. Where appearance and height disagree,
height wins. This is the `geometry_override`, and it is the core contribution.

---

## Architecture

Information flows one way. Each layer consumes only the layer above it, so any
component can be disabled and its contribution measured in isolation.

```
INPUT      camera · IMU · wheel odometry            (simulated or real ROS topics)
                          ↓
LAYER 1    A Elevation      B Semantics             perception
           C Uncertainty    D Visual odometry
                          ↓
LAYER 2    E Negative obstacles                     hazard resolution
           F Dynamic obstacles
           G Voxel / overhead clearance
                          ↓
LAYER 3    multi-layer costmap fusion               ← the contribution
           SAFE 0 · CAUTION 128 · LETHAL 254
                          ↓
LAYER 4    global planner · local planner · control
```

### Layer 1 — perception

| Node | Module | Responsibility |
|------|--------|----------------|
| **A** | `layer1.elevation` | RANSAC ground-plane fit → signed height field, rescaled to the known camera mounting height to resolve monocular scale |
| **B** | `layer1.semantics` | Pixel classes reduced to a 7-word traversability vocabulary |
| **C** | `layer1.uncertainty` | Augmentation-consistency: run the depth net under perturbations, measure disagreement |
| **D** | `layer1.odometry` | ORB + essential matrix visual odometry — **pose without GPS** |
| — | `layer1.depth` | Monocular depth behind an interface (Depth Anything V2) |

### Layer 2 — hazard resolution

| Node | Module | Responsibility |
|------|--------|----------------|
| **E** | `layer2.negative_obstacle` | Vertical depth derivative + below-plane deviation → drop-offs, emitted as a per-column *virtual laser scan* |
| **F** | `layer2.dynamic` | Predicts the flow the vehicle's own motion causes, subtracts it; what still moves is genuinely moving, and gets a velocity vector |
| **G** | `layer2.voxel` | Sparse 3D occupancy → "is anything solid between the wheels and the roof of this vehicle?" |

### Layer 3 — fusion

`fusion.fuse()` implements the decision logic in plain NumPy so it runs and is
testable without ROS. The full system ships the same rules as Nav2
`costmap_2d` plugins.

**The most pessimistic claim wins.** A cell is SAFE only when nothing objected.
Uncertainty can downgrade SAFE to CAUTION but can never erase a LETHAL —
there is a regression test for exactly that.

---

## Capabilities today

✅ **Implemented and tested**

- Pinhole camera model, depth back-projection, RANSAC ground-plane fitting
- Metric scale recovery from known camera mounting height
- Height-above-ground field and top-down elevation rasterisation
- Negative obstacle detection (two independent cues) + virtual scan output
- Ego-motion compensated dynamic obstacle detection with velocity tracks
- Sparse voxel mapping and overhead clearance queries
- Augmentation-consistency uncertainty estimation
- Monocular visual odometry with scale injection
- Full costmap fusion with the geometry override and Nav2 cost values

⚠️ **Runs, not yet tuned on field data**

- Semantic segmentation (SegFormer/ADE20k label mapping is approximate for
  off-road classes; RUGD or RELLIS-3D fine-tuning is the next step)
- Depth scale is plausible rather than surveyed

❌ **Not implemented**

- Layer 4: global planner, local planner, controller — the design specifies
  Nav2 Smac + MPPI, and none of it is written here
- ROS 2 node wrappers and the `costmap_2d` plugin builds
- Loop closure / full SLAM back-end (Node D is odometry, not SLAM)
- Simulator integration and end-to-end navigation runs

Nothing in this repository has driven a vehicle, simulated or real. It is a
perception and hazard-resolution library with a verified core.

---

## Quick start

```bash
pip install -r requirements.txt
pytest -q                    # 68 tests, ~1 s, no model download needed
```

The geometric core has no ML dependency at all. Model-backed nodes sit behind
interfaces (`StubDepth`, `StubSegmenter`), which is why the whole suite runs in
about a second on CPU.

```python
import numpy as np
from ugvnav import Camera
from ugvnav.layer1 import ElevationNetwork, MonocularDepth, to_metric
from ugvnav.layer2 import NegativeObstacleDetector

cam   = Camera.from_fov(640, 480, hfov_deg=70, height_m=0.8)
depth = to_metric(MonocularDepth().infer(rgb))       # downloads weights once
elev  = ElevationNetwork(cam).process(depth)
holes = NegativeObstacleDetector().process(depth, height=elev.height)

holes.ranges      # per-column distance to the nearest drop-off
```

---

## Testing philosophy

Every geometric component is checked against **exactly-known ground truth**,
not against its own previous output:

- Back-projection then projection must be the identity, to 1e-6
- A synthetic plane at `y = 1.5` must be recovered exactly, and still be
  recovered with 20 % outliers present
- Visual odometry must recover a **known** 5° rotation and a known translation
  direction from synthetically projected 3D points
- Ego-motion compensation must report **zero** moving objects when the entire
  scene is translated, and must find the object when one patch moves further
  than the background
- A branch at 1.2 m must block a 1.5 m vehicle and clear a 1.0 m one

### Bugs these tests caught

The suite earned its keep immediately — four real defects, all found by tests
rather than by inspection:

1. **Voxel quantisation.** `1.2 / 0.2` evaluates to `5.999…` in binary floating
   point, so a branch at exactly 1.2 m was filed one voxel too low and a 1.0 m
   vehicle was reported as blocked. Fixed with an epsilon in the floor.
2. **Zero-gradient threshold collapse.** On perfectly flat depth the percentile
   threshold became `0`, and `magnitude >= 0` flagged *every pixel in the image*
   as a drop-off edge.
3. The same collapse defeated the small-blob filter.
4. **`clearance_map` point sampling.** It probed one point per cell, so an
   occupied voxel sitting between sample points was silently missed. It now
   tests every voxel column overlapping each cell.

There is also an honest known limitation, discovered on real imagery: a naive
depth gradient fires hardest on the **horizon**, because the sky/ground boundary
is the strongest depth discontinuity in any outdoor frame. The `valid` mask
parameter exists for that reason — restrict analysis to the ground region.

---

## Roadmap

| Stage | Work | Status |
|-------|------|--------|
| 1 | Layer 1 + Layer 2 core, fully tested | **done** |
| 2 | Fine-tune segmentation on RUGD / RELLIS-3D off-road classes | next |
| 3 | ROS 2 Humble node wrappers, real topic I/O | planned |
| 4 | Nav2 `costmap_2d` plugins replacing `fusion.fuse` | planned |
| 5 | Gazebo worlds with authored ditches, overhangs, moving hazards | planned |
| 6 | Layer 4 planning: Smac global + MPPI local | planned |
| 7 | Ablation study — disable one layer, measure collision rate | planned |

---

## Licensing

This project is **Apache-2.0**. Dependencies were selected so the result stays
permissively licensed and can be handed over without copyleft obligations —
which matters for a defence-sector recipient.

| Component | Licence | Note |
|-----------|---------|------|
| Depth Anything V2 | Apache-2.0 | chosen over monodepth2, which is research-only |
| SegFormer | Apache-2.0 | chosen over YOLOv8, which is AGPL-3.0 |
| OpenCV, NumPy | BSD / Apache | |
| *Planned:* RTAB-Map | BSD-3 | chosen over ORB-SLAM3, which is **GPL-3.0** |
| *Planned:* Nav2 | Apache-2.0 | |

No GPL or AGPL component is used anywhere in this stack.

---

## Repository layout

```
ugvnav/
  camera.py              pinhole model, back-projection, RANSAC plane fitting
  fusion.py              Layer 3 — geometry override and costmap fusion
  layer1/
    depth.py             monocular depth interface + Depth Anything V2
    elevation.py         Node A — ground plane and height field
    semantics.py         Node B — traversability vocabulary
    uncertainty.py       Node C — augmentation-consistency uncertainty
    odometry.py          Node D — visual odometry, pose without GPS
  layer2/
    negative_obstacle.py Node E — ditches and drop-offs
    dynamic.py           Node F — ego-motion compensated tracking
    voxel.py             Node G — 3D occupancy and overhead clearance
tests/                   68 tests against synthetic ground truth
```
