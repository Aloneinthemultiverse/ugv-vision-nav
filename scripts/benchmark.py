"""Compare traversability methods on identical input.

Four configurations, one shared set of perception outputs, so any difference in
the result is attributable to the fusion policy alone and not to a different
depth network or a different image.

    A  semantic-only     what most appearance-based off-road stacks do
    B  geometry-only     height threshold above the fitted ground plane
    C  semantic AND geometry   the permissive intersection
    D  ours              semantics + geometry override + drop-offs + uncertainty

There is no pixel-wise ground truth for these photographs, so this is not an
accuracy benchmark. It measures something arguably more useful for a navigation
stack: **each method plans a route, and every route is then scored against every
method's hazard map.** A path that method A calls safe but method D calls lethal
is a concrete, countable failure of A.

    python scripts/benchmark.py
"""
from __future__ import annotations

import csv
import json
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ugvnav import Camera                                              # noqa: E402
from ugvnav.fusion import LETHAL as F_LETHAL, fuse                     # noqa: E402
from ugvnav.layer1 import ElevationNetwork, MonocularDepth, to_metric  # noqa: E402
from ugvnav.layer1.semantics import SemanticSegmenter, traversable_mask  # noqa: E402
from ugvnav.layer2 import NegativeObstacleDetector                     # noqa: E402
from ugvnav.layer3 import Costmap, GridSpec, INSCRIBED, LETHAL         # noqa: E402
from ugvnav.layer4 import AStarPlanner, PurePursuit                    # noqa: E402

ROBOT_H = 1.4
SPIKE_M = 0.22
METHODS = ["A semantic-only", "B geometry-only", "C semantic AND geometry", "D ours"]


def hazard_masks(labels, height, neg_mask, unc, ground):
    """Image-space lethal masks for each configuration."""
    drivable = traversable_mask(labels)
    bumpy = height > SPIKE_M

    A = ~drivable
    B = bumpy
    C = ~drivable & bumpy                       # only when both agree
    D = fuse(labels, height, negative=neg_mask, uncertainty=unc,
             valid=ground).grid == F_LETHAL
    return {
        METHODS[0]: A & ground,
        METHODS[1]: B & ground,
        METHODS[2]: C & ground,
        METHODS[3]: D & ground,
    }


def build_costmap(points, height, lethal_px):
    cm = Costmap(GridSpec(90, 90, 0.1, -4.5, 0.0), robot_radius=0.35,
                 inflation_radius=0.9)
    sel = lethal_px.reshape(-1) & (height.reshape(-1) <= ROBOT_H)
    if sel.any():
        q = points.reshape(-1, 3)[sel]
        keep = np.isfinite(q).all(1) & (q[:, 2] < 12.0)
        if keep.any():
            cm.mark(q[keep, 0], q[keep, 2])
    cm.inflate()
    return cm


def pick_goal(cm, min_y=1.5):
    free = np.argwhere(cm.grid < INSCRIBED)
    if free.size == 0:
        return (0.0, min_y)
    x, y = cm.spec.cell_to_world(free[:, 1], free[:, 0])
    ok = y >= min_y
    if not ok.any():
        return (0.0, min_y)
    x, y = x[ok], y[ok]
    band = y >= y.max() - 0.3
    i = int(np.argmin(np.abs(x[band])))
    return (float(x[band][i]), float(y[band][i]))


def score_path(path, cm):
    """How many steps of this path land in cells *this* costmap calls lethal."""
    if path.size == 0:
        return 0, 0.0
    hits = sum(1 for x, y in path if cm.cost_at(float(x), float(y)) >= LETHAL)
    length = float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum())
    return hits, length


def compare_figure(fn: str, out: str) -> None:
    """Side-by-side costmap + planned route for all four methods on one scene."""
    from ugvnav.layer3 import CAUTION as L3_CAUTION

    rgb = cv2.cvtColor(cv2.imread(f"data/offroad/{fn}"), cv2.COLOR_BGR2RGB)
    s_ = 720 / max(rgb.shape[:2])
    if s_ < 1:
        rgb = cv2.resize(rgb, None, fx=s_, fy=s_, interpolation=cv2.INTER_AREA)
    h, w = rgb.shape[:2]
    cam = Camera.from_fov(w, h, 70.0, 0.8)

    depth = to_metric(MonocularDepth().infer(rgb), 1.0, 25.0)
    elev = ElevationNetwork(cam).process(depth)
    labels = SemanticSegmenter().infer(rgb)
    ground = (labels != 5) & (depth < 22.0)
    neg = NegativeObstacleDetector(grad_percentile=99.2, min_area=60).process(
        depth, height=elev.height, valid=ground)
    unc = np.clip(np.abs(cv2.Laplacian(depth, cv2.CV_32F)) / 8.0, 0, 1)

    masks = hazard_masks(labels, elev.height, neg.mask, unc, ground)
    ours_cm = build_costmap(elev.points, elev.height, masks[METHODS[3]])

    S = 300
    tiles = []
    for m in METHODS:
        cm = build_costmap(elev.points, elev.height, masks[m])
        path = AStarPlanner(cm).plan((0.0, 0.3), pick_goal(cm))
        bad, _ = score_path(path, ours_cm)

        g = cm.grid
        img = np.zeros((*g.shape, 3), np.uint8)
        img[g == 0] = (34, 78, 52)
        band = (g > 0) & (g < INSCRIBED)
        if band.any():
            f = (g[band].astype(np.float32) / L3_CAUTION).clip(0, 1)[:, None]
            img[band] = (np.array([34, 78, 52]) * (1 - f)
                         + np.array([232, 176, 46]) * f).astype(np.uint8)
        img[g >= INSCRIBED] = (150, 60, 50)
        img[g >= LETHAL] = (206, 60, 48)
        img = cv2.resize(img, (S, S), interpolation=cv2.INTER_NEAREST)

        if path.size > 1:
            c, r = cm.spec.world_to_cell(path[:, 0], path[:, 1])
            xy = np.stack([c * S / g.shape[1], r * S / g.shape[0]], -1).astype(np.int32)
            cv2.polylines(img, [xy], False, (120, 205, 255), 3, cv2.LINE_AA)
            # mark the steps our hazard map considers lethal
            for x, y in path:
                if ours_cm.cost_at(float(x), float(y)) >= LETHAL:
                    cc, rr = cm.spec.world_to_cell(x, y)
                    cv2.circle(img, (int(cc * S / g.shape[1]), int(rr * S / g.shape[0])),
                               3, (255, 70, 60), -1)
        cv2.circle(img, (S // 2, S - 8), 6, (240, 245, 250), -1)

        bar = np.full((38, S, 3), (22, 28, 38), np.uint8)
        cv2.putText(bar, m, (8, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.40,
                    (150, 230, 190), 1, cv2.LINE_AA)
        col = (120, 230, 150) if bad == 0 else (255, 110, 95)
        cv2.putText(bar, f"route {'yes' if path.size else 'NO'} | "
                         f"{bad} unsafe step(s) by our map", (8, 31),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.33, col, 1, cv2.LINE_AA)
        tiles.append(np.vstack([bar, img]))

    ph = int(rgb.shape[0] * S / rgb.shape[1])
    photo = cv2.resize(rgb, (S, ph), interpolation=cv2.INTER_AREA)
    pbar = np.full((38, S, 3), (22, 28, 38), np.uint8)
    cv2.putText(pbar, "INPUT", (8, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.40,
                (150, 230, 190), 1, cv2.LINE_AA)
    cv2.putText(pbar, fn, (8, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.33,
                (150, 160, 175), 1, cv2.LINE_AA)
    photo = np.vstack([pbar, photo])

    hh = max([t.shape[0] for t in tiles] + [photo.shape[0]])
    def pad(t):
        return np.vstack([t, np.full((hh - t.shape[0], t.shape[1], 3), (18, 22, 30), np.uint8)])
    cv2.imwrite(out, np.hstack([pad(photo)] + [pad(t) for t in tiles])[:, :, ::-1])
    print("wrote", out)


def main() -> None:
    if len(sys.argv) > 2 and sys.argv[1] == "--figure":
        compare_figure(sys.argv[2], sys.argv[3] if len(sys.argv) > 3
                       else "assets/comparison.jpg")
        return
    files = sorted(f for f in os.listdir("data/offroad") if f.endswith(".jpg"))
    depth_net, seg_net = MonocularDepth(), SemanticSegmenter()
    rows, confusion = [], {m: {n: 0 for n in METHODS} for m in METHODS}
    t0 = time.time()

    for fn in files:
        rgb = cv2.cvtColor(cv2.imread(f"data/offroad/{fn}"), cv2.COLOR_BGR2RGB)
        s = 720 / max(rgb.shape[:2])
        if s < 1:
            rgb = cv2.resize(rgb, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        h, w = rgb.shape[:2]
        cam = Camera.from_fov(w, h, 70.0, 0.8)

        # ---- shared perception: computed once, used by every method ----
        depth = to_metric(depth_net.infer(rgb), 1.0, 25.0)
        elev = ElevationNetwork(cam).process(depth)
        labels = seg_net.infer(rgb)
        ground = (labels != 5) & (depth < 22.0)
        neg = NegativeObstacleDetector(grad_percentile=99.2, min_area=60).process(
            depth, height=elev.height, valid=ground)
        unc = np.clip(np.abs(cv2.Laplacian(depth, cv2.CV_32F)) / 8.0, 0, 1)

        masks = hazard_masks(labels, elev.height, neg.mask, unc, ground)
        cms = {m: build_costmap(elev.points, elev.height, msk)
               for m, msk in masks.items()}
        paths = {}
        for m, cm in cms.items():
            paths[m] = AStarPlanner(cm).plan((0.0, 0.3), pick_goal(cm))

        for m in METHODS:
            cm, path = cms[m], paths[m]
            cmd = PurePursuit(lookahead=1.2).step((0.0, 0.3, 0.0), path, cm) \
                if path.size else None
            # score this method's path against every method's hazard map
            for judge in METHODS:
                confusion[m][judge] += score_path(path, cms[judge])[0]
            hits_ours, length = score_path(path, cms[METHODS[3]])
            rows.append({
                "scene": fn, "method": m,
                "lethal_px": int(masks[m].sum()),
                "lethal_cells": int((cm.grid >= LETHAL).sum()),
                "route": int(path.size > 0),
                "path_m": round(length, 2),
                "unsafe_steps_vs_ours": hits_ours,
                "v": round(cmd.linear, 2) if cmd else 0.0,
                "w": round(cmd.angular, 2) if cmd else 0.0,
            })
        print(f"{fn}: " + "  ".join(
            f"{m.split()[0]}={'ok' if paths[m].size else 'NO'}" for m in METHODS))

    os.makedirs("assets", exist_ok=True)
    with open("assets/benchmark.csv", "w", newline="") as f:
        wri = csv.DictWriter(f, fieldnames=list(rows[0]))
        wri.writeheader()
        wri.writerows(rows)

    # ---- aggregate ----
    print(f"\n{len(files)} scenes, {time.time() - t0:.0f}s\n")
    summary = {}
    hdr = f"{'method':<26}{'routes':>8}{'mean path m':>13}{'unsafe steps':>14}{'mean lethal %':>15}"
    print(hdr); print("-" * len(hdr))
    for m in METHODS:
        r = [x for x in rows if x["method"] == m]
        routes = sum(x["route"] for x in r)
        pth = np.mean([x["path_m"] for x in r if x["route"]]) if routes else 0.0
        unsafe = sum(x["unsafe_steps_vs_ours"] for x in r)
        lethal = np.mean([x["lethal_cells"] for x in r]) / 8100 * 100
        summary[m] = dict(routes=routes, of=len(files), mean_path_m=round(float(pth), 2),
                          unsafe_steps=int(unsafe), mean_lethal_pct=round(float(lethal), 1))
        print(f"{m:<26}{routes:>4}/{len(files):<3}{pth:>13.2f}{unsafe:>14}{lethal:>14.1f}%")

    print("\nrows = whose path, columns = whose hazard map judged it")
    print(f"{'':<26}" + "".join(f"{m.split()[0]:>8}" for m in METHODS))
    for m in METHODS:
        print(f"{m:<26}" + "".join(f"{confusion[m][j]:>8}" for j in METHODS))

    json.dump({"summary": summary, "confusion": confusion},
              open("assets/benchmark.json", "w"), indent=2)
    print("\nwrote assets/benchmark.csv and assets/benchmark.json")


if __name__ == "__main__":
    main()
