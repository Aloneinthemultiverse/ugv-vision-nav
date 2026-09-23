# Related work — who covers what

A survey of open-source off-road navigation projects, checked against the
capability set this project targets. Repository facts (stars, licence, last
push) were verified via the GitHub API in September 2026; capability claims come
from each project's own README or paper.

**Headline: no single open-source project covers all of it.** The closest,
`LARIAD/Offroad-Nav`, is ahead of us on localization and field maturity and
absent on every one of the five hazard cases. We are the inverse.

---

## Coverage matrix

Legend: ● full · ◐ partial · ○ absent

| | **ugvnav** (this) | Offroad-Nav | Autoware | WVN | elevation_​mapping_cupy | Nav2 | RTAB-Map |
|---|---|---|---|---|---|---|---|
| Monocular-only capable | ● | ● | ○ | ● | ◐ | n/a | ◐ |
| Works without LiDAR | ● | ● | ○ | ● | ◐ | n/a | ● |
| Monocular depth | ● | ● | ○ | ○ | ○ | ○ | ○ |
| Semantic segmentation | ● | ○ | ● | ◐ | ○ | ○ | ○ |
| Elevation / height field | ● | ● | ● | ○ | ● | ○ | ◐ |
| **Localization / SLAM** | **◐ VO only** | **● VINS-Mono** | ● | ○ | ○ | ○ | ● |
| Uncertainty estimation | ● | ○ | ○ | ● | ◐ | ○ | ○ |
| **Negative obstacles** | **●** | ○ | ◐ | ○ | ◐ | ○ | ○ |
| **Dynamic obstacles** | **●** | ○ | ● | ○ | ○ | ◐ | ○ |
| **Overhead clearance** | **●** | ○ | ◐ | ○ | ○ | ○ | ○ |
| **Geometry-over-semantics override** | **●** | ○ | ○ | ○ | ○ | ○ | ○ |
| Multi-layer costmap | ● | ◐ binary | ● | ○ | ◐ | ● | ○ |
| Global planner | ● A\* | ● A\* | ● | ○ | ○ | ● | ○ |
| Local planner | ● MPPI | ● TEB | ● | ○ | ○ | ● | ○ |
| **ROS 2** | ○ | ○ ROS 1 | ● | ● ROS 1 | ● | ● | ● |
| **Field-tested on a vehicle** | **○** | **●** | ● | ● | ● | ● | ● |
| Permissive licence | ● Apache-2.0 | ● MIT | ● Apache-2.0 | ● MIT | ● MIT | ● Apache-2.0 | ◐ mixed |

---

## The projects

### LARIAD/Offroad-Nav — the closest comparable
*MIT · 68★ · last push 2026-04 ·
[repo](https://github.com/LARIAD/Offroad-Nav) ·
[paper](https://arxiv.org/abs/2604.03096)*

"An Open-Source LiDAR and Monocular Off-Road Autonomous Navigation Stack."
Training-free, foundation-model based, and it uses **Depth Anything V2 — the
same depth backbone we chose independently**.

Pipeline: LiDAR *or* monocular depth → VINS-Mono for metric rescaling → EKF
fusion of GNSS/IMU/SLAM → Cloth Simulation Filter ground segmentation →
robot-centric 2.5D elevation → **binary** costmap at a 30 cm height threshold →
A\* global → TEB local.

Reported: 100 % success rate in both Isaac Sim and real-world trials, 59 % mean
SPL monocular vs 69 % LiDAR — monocular matches LiDAR with a modest efficiency
loss.

**Where they are ahead of us:** real metric localization (VINS-Mono + EKF),
demonstrated closed-loop driving on an actual vehicle, and published SR/SPL
numbers. We have none of those.

**Where we differ:** their costmap is a single binary height threshold. There is
no semantic layer, no negative-obstacle reasoning, no dynamic obstacle
prediction, no overhead clearance. A 30 cm threshold cannot distinguish a rock
from tall grass, and cannot see a hole at all — a ditch is *below* the plane, so
a height threshold never fires.

### autowarefoundation/autoware
*Apache-2.0 · 12 084★ · actively developed*

The most complete open AV stack in existence, and the benchmark for engineering
maturity. But it is built for structured roads: lane graphs, HD maps, and a
LiDAR-first sensor model. The off-road variants exist but inherit the road
assumptions. Not monocular-viable.

### leggedrobotics/wild_visual_navigation
*MIT · 316★ · last push 2026-05*

The most interesting idea in the field: **self-supervised traversability** —
the robot drives, and whatever it successfully traversed becomes a positive
label, so the model adapts online to terrain nobody annotated. Directly attacks
the domain-gap problem our RELLIS benchmark exposed.

Vision-only and elegant, but it estimates *traversability appearance* and does
not reason about geometry: no negative obstacles, no overhead clearance, no
planner of its own.

### leggedrobotics/elevation_mapping_cupy · ANYbotics/elevation_mapping
*MIT · 1 105★ · and BSD-3 · 1 872★*

The reference implementations for robot-centric 2.5D elevation mapping, GPU
accelerated, with a traversability layer. Mature and widely deployed on legged
robots. They are **mapping libraries, not navigation stacks** — they need pose
supplied from outside and hand off to a separate planner.

### ros-navigation/navigation2
*4 739★ · actively developed*

Not a competitor — the substrate. Nav2 defines the `costmap_2d` plugin
architecture and the cost values (0 / 128 / 253 / 254) this project mirrors
exactly, so that `layer3.costmap` can be swapped for real plugins later.

### introlab/rtabmap
*4 006★ · actively developed*

Best-in-class BSD-licensed visual SLAM, and our planned replacement for the
GPL-3.0 ORB-SLAM3. Localization and mapping only; no traversability reasoning.

### konyul/STONE
*ICRA 2026 · 59★ · no licence file*

A scalable multi-modal surround-view off-road **dataset**, not a stack. Worth
tracking as evaluation data. The absent licence is a blocker for reuse.

---

## Honest read

**What is genuinely ours.** No surveyed project implements the
geometry-over-semantics override, and none covers negative obstacles, dynamic
obstacle prediction and overhead clearance together. Offroad-Nav's single
binary height threshold is the state of the practice for monocular off-road
costmaps, and our RELLIS benchmark shows permissive/single-cue fusion is
measurably the weakest strategy tested (method C, 58.5 % false-safe).

**Where we are behind, and it matters.**

1. **Localization.** Offroad-Nav has VINS-Mono plus an EKF over GNSS/IMU/SLAM.
   We have frame-to-frame visual odometry with no loop closure and no filter.
   For a problem statement that explicitly requires *"estimating position and
   orientation without GPS"*, this is our weakest area.
2. **Field validation.** They drove a real vehicle and published SR/SPL. We have
   segmentation IoU on 52 still frames and no closed-loop result of any kind.
   Numbers from the two projects are not comparable, and ours is the weaker
   evidence.
3. **ROS 2.** We claim Nav2 compatibility by mirroring its cost values; we have
   not built the plugins. Offroad-Nav at least ships a working ROS 1 system.
4. **The domain gap.** WVN's self-supervised approach is a better answer to the
   35 % obstacle-mislabel rate our benchmark measured than fine-tuning would be,
   because it needs no off-road annotations at all.

**Conclusion.** The contribution is the multi-hazard fusion policy, and the
benchmark supports it. The gap is everything around it: localization, a real
robot, and ROS 2 integration. Claiming a complete navigation system would be
false; claiming a better traversability fusion layer than the surveyed
alternatives is supported by measurement.
