"""Measure the water detector against RELLIS-3D water labels.

The detector was written because our traversability benchmark measured 82.3 % of
labelled water as false-safe. It was then never evaluated on that label, which
is exactly the mistake the detector was meant to fix. This closes that loop.

RELLIS-3D 5-label ground truth: class 2 is "withwater" (mud, puddle, water).

Reported per configuration:

    recall      fraction of labelled water actually found
    precision   fraction of detections that really are water
    FP on trav  fraction of labelled drivable ground wrongly called water,
                which is the cost of over-detecting

    python scripts/benchmark_water.py
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ugvnav import Camera                                              # noqa: E402
from ugvnav.layer1 import ElevationNetwork, MonocularDepth, to_metric   # noqa: E402
from ugvnav.layer1.semantics import SemanticSegmenter                   # noqa: E402
from ugvnav.layer2.water import WaterDetector                           # noqa: E402

GT_IGNORE, GT_OBSTACLE, GT_WATER, GT_UNSTABLE, GT_STABLE = 0, 1, 2, 3, 4


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--width", type=int, default=640)
    args = ap.parse_args()

    imgs = sorted(glob.glob("data/rellis3d/images/*.jpg"))
    if args.limit:
        imgs = imgs[:args.limit]
    if not imgs:
        sys.exit("no frames - run scripts/fetch_rellis.py first")

    configs = {
        "threshold 0.45": dict(threshold=0.45, min_area=150),
        "threshold 0.55 (default)": dict(threshold=0.55, min_area=150),
        "threshold 0.65": dict(threshold=0.65, min_area=150),
        "0.45, no height gate": dict(threshold=0.45, min_area=150),
    }
    totals = {k: dict(tp=0, fp=0, fn=0, fp_trav=0, gt_water=0, gt_trav=0)
              for k in configs}

    depth_net, seg_net = MonocularDepth(), SemanticSegmenter()
    t0 = time.time()
    frames_with_water = 0

    for k, ip in enumerate(imgs, 1):
        stem = os.path.splitext(os.path.basename(ip))[0]
        gt_path = f"data/rellis3d/masks/{stem}.png"
        if not os.path.exists(gt_path):
            continue
        rgb = cv2.cvtColor(cv2.imread(ip), cv2.COLOR_BGR2RGB)
        gt = cv2.imread(gt_path, cv2.IMREAD_UNCHANGED)
        s = args.width / rgb.shape[1]
        rgb = cv2.resize(rgb, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        gt = cv2.resize(gt, (rgb.shape[1], rgb.shape[0]),
                        interpolation=cv2.INTER_NEAREST)

        gt_water = gt == GT_WATER
        gt_trav = np.isin(gt, (GT_UNSTABLE, GT_STABLE))
        if gt_water.any():
            frames_with_water += 1

        h, w = rgb.shape[:2]
        cam = Camera.from_fov(w, h, 70.0, 0.8)
        depth = to_metric(depth_net.infer(rgb), 1.0, 25.0)
        elev = ElevationNetwork(cam).process(depth)
        labels = seg_net.infer(rgb)
        valid = (gt != GT_IGNORE) & (labels != 5)

        for name, kw in configs.items():
            det = WaterDetector(**kw)
            height = None if "no height" in name else elev.height
            mask = det.process(rgb, valid=valid, height=height).mask
            t = totals[name]
            t["tp"] += int((mask & gt_water).sum())
            t["fp"] += int((mask & ~gt_water & valid).sum())
            t["fn"] += int((~mask & gt_water).sum())
            t["fp_trav"] += int((mask & gt_trav).sum())
            t["gt_water"] += int(gt_water.sum())
            t["gt_trav"] += int(gt_trav.sum())

        if k % 10 == 0:
            print(f"  {k}/{len(imgs)}  ({time.time() - t0:.0f}s)")

    print(f"\n{len(imgs)} frames, {frames_with_water} contain labelled water, "
          f"{time.time() - t0:.0f}s\n")
    hdr = f"{'config':<26}{'recall':>9}{'precision':>11}{'FP on trav':>12}"
    print(hdr); print("-" * len(hdr))
    out = {}
    for name, t in totals.items():
        rec = t["tp"] / t["gt_water"] if t["gt_water"] else float("nan")
        prec = t["tp"] / (t["tp"] + t["fp"]) if (t["tp"] + t["fp"]) else float("nan")
        fpt = t["fp_trav"] / t["gt_trav"] if t["gt_trav"] else float("nan")
        out[name] = dict(recall=round(rec, 4), precision=round(prec, 4),
                         fp_on_traversable=round(fpt, 4), **t)
        print(f"{name:<26}{rec * 100:>8.1f}%{prec * 100:>10.1f}%{fpt * 100:>11.1f}%")

    print("\nBaseline: the semantic-only stack missed 82.3% of this class,")
    print("i.e. a recall of 17.7%. Anything below that is not an improvement.")

    os.makedirs("assets", exist_ok=True)
    json.dump(out, open("assets/benchmark_water.json", "w"), indent=2)
    print("\nwrote assets/benchmark_water.json")


if __name__ == "__main__":
    main()
