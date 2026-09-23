"""End-to-end run: one photograph in, one steering command out.

    python scripts/run_pipeline.py data/offroad/scene03.jpg assets/scene03.jpg

Produces a six-panel figure showing every stage of what the vehicle sees:
input, depth, height above ground, hazards, costmap, and the planned route.
"""
from __future__ import annotations

import argparse
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ugvnav import Camera                                          # noqa: E402
from ugvnav.fusion import CAUTION as F_CAUTION, LETHAL as F_LETHAL, SAFE, UNKNOWN, fuse  # noqa: E402
from ugvnav.layer1 import ElevationNetwork, MonocularDepth, to_metric  # noqa: E402
from ugvnav.layer1.semantics import SemanticSegmenter               # noqa: E402
from ugvnav.layer2 import NegativeObstacleDetector                  # noqa: E402
from ugvnav.layer3 import Costmap, GridSpec, CAUTION, INSCRIBED, LETHAL  # noqa: E402
from ugvnav.layer4 import AStarPlanner, MPPIPlanner, PurePursuit    # noqa: E402

PANEL_W = 420
GREEN, AMBER, RED, GREY = (70, 190, 120), (232, 176, 46), (206, 60, 48), (86, 94, 106)


def panel(img, title, sub=""):
    h = int(img.shape[0] * PANEL_W / img.shape[1])
    out = cv2.resize(img, (PANEL_W, h), interpolation=cv2.INTER_AREA)
    bar = np.full((34, PANEL_W, 3), (22, 28, 38), np.uint8)
    cv2.putText(bar, title, (10, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (150, 230, 190), 1, cv2.LINE_AA)
    if sub:
        cv2.putText(bar, sub, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (150, 160, 175), 1, cv2.LINE_AA)
    return np.vstack([bar, out])


def heat(x, cmap=cv2.COLORMAP_INFERNO):
    x = np.asarray(x, np.float32)
    lo, hi = float(np.nanmin(x)), float(np.nanmax(x))
    n = (x - lo) / (hi - lo + 1e-6)
    return cv2.applyColorMap((n * 255).astype(np.uint8), cmap)[:, :, ::-1]


def tint(rgb, grid):
    out = rgb.copy().astype(np.float32)
    for state, col in ((F_CAUTION, AMBER), (F_LETHAL, RED), (UNKNOWN, GREY)):
        m = grid == state
        if m.any():
            out[m] = out[m] * 0.45 + np.array(col, np.float32) * 0.55
    return out.astype(np.uint8)


def render_costmap(cm, path=None, traj=None, size=360):
    g = cm.grid
    img = np.zeros((*g.shape, 3), np.uint8)
    img[g == 0] = (34, 78, 52)
    band = (g > 0) & (g < INSCRIBED)
    if band.any():
        f = (g[band].astype(np.float32) / CAUTION).clip(0, 1)[:, None]
        img[band] = (np.array([34, 78, 52]) * (1 - f) + np.array(AMBER) * f).astype(np.uint8)
    img[g >= INSCRIBED] = (150, 60, 50)
    img[g >= LETHAL] = RED

    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_NEAREST)
    sx, sy = size / g.shape[1], size / g.shape[0]

    def draw(pts, col, thick):
        if pts is None or len(pts) < 2:
            return
        c, r = cm.spec.world_to_cell(pts[:, 0], pts[:, 1])
        xy = np.stack([c * sx, r * sy], -1).astype(np.int32)
        cv2.polylines(img, [xy], False, col, thick, cv2.LINE_AA)

    draw(traj, (255, 210, 90), 2)
    draw(path, (120, 205, 255), 3)

    c, r = cm.spec.world_to_cell(0.0, 0.0)
    cv2.circle(img, (int(c * sx), int(r * sy) - 4), 6, (240, 245, 250), -1)
    return img


def pick_goal(cm, min_y=1.5):
    """Furthest reachable point along the corridor, nearest to straight ahead.

    Outdoor navigation rarely has a surveyed goal pose. The useful default is
    "keep going as far as the ground allows", so we take the free cell with the
    greatest forward distance and break ties toward the centre line.
    """
    free = np.argwhere(cm.grid < INSCRIBED)
    if free.size == 0:
        return (0.0, min_y)
    x, y = cm.spec.cell_to_world(free[:, 1], free[:, 0])
    ok = y >= min_y
    if not ok.any():
        return (0.0, min_y)
    x, y = x[ok], y[ok]
    best_y = y.max()
    band = y >= best_y - 0.3
    i = int(np.argmin(np.abs(x[band])))
    return (float(x[band][i]), float(y[band][i]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("out")
    ap.add_argument("--goal", type=float, nargs=2, default=None,
                    help="world goal in metres; omitted = drive as far along "
                         "the traversable corridor as possible")
    args = ap.parse_args()

    rgb = cv2.cvtColor(cv2.imread(args.image), cv2.COLOR_BGR2RGB)
    if max(rgb.shape[:2]) > 720:
        s = 720 / max(rgb.shape[:2])
        rgb = cv2.resize(rgb, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    h, w = rgb.shape[:2]
    cam = Camera.from_fov(w, h, hfov_deg=70.0, height_m=0.8)

    # ---- Layer 1 -----------------------------------------------------------
    depth = to_metric(MonocularDepth().infer(rgb), near_m=1.0, far_m=25.0)
    elev = ElevationNetwork(cam).process(depth)
    labels = SemanticSegmenter().infer(rgb)
    ground = (labels != 5) & (depth < 22.0)          # drop sky and far field

    # ---- Layer 2 -----------------------------------------------------------
    neg = NegativeObstacleDetector(grad_percentile=99.2, min_area=60).process(
        depth, height=elev.height, valid=ground)

    # ---- Layer 3 (image-space preview + metric costmap) --------------------
    f = fuse(labels, elev.height, negative=neg.mask, valid=ground)

    cm = Costmap(GridSpec(90, 90, 0.1, -4.5, 0.0), robot_radius=0.35,
                 inflation_radius=0.9)
    ROBOT_H = 1.4
    pts = elev.points.reshape(-1, 3)
    hgt = elev.height.reshape(-1)
    bad = ((f.grid == F_LETHAL) & ground).reshape(-1)

    # Only obstruction *within the vehicle's own height band* blocks the ground.
    # A tree canopy 4 m up sits directly above the path; projecting it straight
    # down would wall off a corridor the vehicle can actually drive through.
    # Anything higher is overhead clearance - Node G's job, not the ground layer.
    at_body_height = bad & (hgt <= ROBOT_H)
    overhead = bad & (hgt > ROBOT_H)
    if at_body_height.any():
        q = pts[at_body_height]
        keep = np.isfinite(q).all(1) & (q[:, 2] < 12.0)
        cm.mark(q[keep, 0], q[keep, 2])              # camera Z -> world forward
    cm.inflate()
    n_over = int(overhead.sum())

    # ---- Layer 4 -----------------------------------------------------------
    start = (0.0, 0.3)
    goal = tuple(args.goal) if args.goal else pick_goal(cm)
    if cm.cost_at(*goal) >= INSCRIBED:
        goal = pick_goal(cm)                 # requested goal is not reachable
    path = AStarPlanner(cm).plan(start, goal)
    traj = np.empty((0, 3))
    cmd = None
    if path.size:
        cmd, best = MPPIPlanner(cm, samples=500, seed=2).plan((*start, 0.0), path)
        traj = best[:, :2] if best.size else traj
        cmd = PurePursuit(lookahead=1.2).step((*start, 0.0), path, costmap=cm)

    # ---- render ------------------------------------------------------------
    hs = np.clip(elev.height, -1.0, 1.0)
    row1 = [panel(rgb, "1  INPUT", os.path.basename(args.image)),
            panel(heat(depth), "2  DEPTH  (Node A1)", "Depth Anything V2"),
            panel(heat(hs, cv2.COLORMAP_VIRIDIS), "3  HEIGHT ABOVE GROUND  (Node A)",
                  f"RANSAC plane, scale {elev.scale:.2f}")]
    n_ov = int(f.reasons.get("geometry_override", 0))
    row2 = [panel(tint(rgb, f.grid), "4  HAZARDS  (Node E + fusion)",
                  f"override {n_ov} px | drop-off {int(neg.mask.sum())} px | canopy {n_over} px"),
            panel(render_costmap(cm), "5  COSTMAP  (Layer 3)", "inflated, vehicle frame, 9m x 9m"),
            panel(render_costmap(cm, path, traj), "6  PLAN  (Layer 4)",
                  (f"cmd v={cmd.linear:.2f} m/s  w={cmd.angular:+.2f} rad/s"
                   if cmd else "no route found"))]

    def stack(row):
        hh = max(p.shape[0] for p in row)
        row = [np.vstack([p, np.full((hh - p.shape[0], p.shape[1], 3), (18, 22, 30), np.uint8)])
               for p in row]
        return np.hstack(row)

    a, b = stack(row1), stack(row2)
    wmax = max(a.shape[1], b.shape[1])
    canvas = np.full((a.shape[0] + b.shape[0] + 8, wmax, 3), (18, 22, 30), np.uint8)
    canvas[:a.shape[0], :a.shape[1]] = a
    canvas[a.shape[0] + 8:, :b.shape[1]] = b

    cv2.imwrite(args.out, canvas[:, :, ::-1])
    print(f"{os.path.basename(args.image)}  ->  {args.out}")
    print(f"  reasons={f.reasons}  route={'yes' if path.size else 'NO'}"
          f"  cmd={'v=%.2f w=%+.2f' % (cmd.linear, cmd.angular) if cmd else '-'}")


if __name__ == "__main__":
    main()
