"""Traversability benchmark against RELLIS-3D ground truth.

Unlike ``benchmark.py``, which could only measure methods against each other,
this scores every method against **human pixel labels**. No method judges its
own output, so the numbers are not circular.

RELLIS-3D 5-label ground truth (from the official relabelling script):

    0  background  void, sky                                   ignored
    1  obstacle    tree, bush, person, fence, vehicle, log...  must be blocked
    2  water       mud, puddle, water                          must be blocked
    3  unstable    grass, dirt                                 traversable
    4  stable      concrete                                    traversable

Metrics reported per method:

    IoU trav / IoU block / mIoU   standard segmentation scores
    FALSE-SAFE                    GT obstacle predicted drivable  <- dangerous
    FALSE-BLOCK                   GT drivable predicted lethal    <- costs capability

FALSE-SAFE is the number that matters for a vehicle. A missed obstacle is a
collision; an over-blocked patch of grass is only a detour.

    python scripts/benchmark_gt.py --limit 30
"""
from __future__ import annotations

import argparse
import csv
import glob
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

GT_IGNORE, GT_OBSTACLE, GT_WATER, GT_UNSTABLE, GT_STABLE = 0, 1, 2, 3, 4
SPIKE_M = 0.22
METHODS = ["A semantic-only", "B geometry-only", "C semantic AND geometry", "D ours"]


def predictions(labels, height, neg_mask, unc, valid):
    """Lethal masks for each method. True = this method blocks the pixel."""
    drivable = traversable_mask(labels)
    bumpy = height > SPIKE_M
    return {
        METHODS[0]: (~drivable) & valid,
        METHODS[1]: bumpy & valid,
        METHODS[2]: (~drivable & bumpy) & valid,
        METHODS[3]: (fuse(labels, height, negative=neg_mask, uncertainty=unc,
                          valid=valid).grid == F_LETHAL) & valid,
    }


def score(pred_block, gt):
    """Confusion counts against ground truth, ignoring void/sky."""
    keep = gt != GT_IGNORE
    gt_block = np.isin(gt, (GT_OBSTACLE, GT_WATER)) & keep
    gt_trav = np.isin(gt, (GT_UNSTABLE, GT_STABLE)) & keep
    pb = pred_block & keep
    pt = (~pred_block) & keep

    tp_b = int((pb & gt_block).sum())      # correctly blocked
    fp_b = int((pb & gt_trav).sum())       # false block - lost capability
    fn_b = int((pt & gt_block).sum())      # FALSE SAFE - dangerous
    tp_t = int((pt & gt_trav).sum())       # correctly drivable
    return dict(tp_block=tp_b, false_block=fp_b, false_safe=fn_b, tp_trav=tp_t,
                gt_block=int(gt_block.sum()), gt_trav=int(gt_trav.sum()))


def iou(tp, fp, fn):
    d = tp + fp + fn
    return tp / d if d else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="0 = all available")
    ap.add_argument("--width", type=int, default=640)
    args = ap.parse_args()

    imgs = sorted(glob.glob("data/rellis3d/images/*.jpg"))
    if args.limit:
        imgs = imgs[:args.limit]
    if not imgs:
        sys.exit("no frames - run scripts/fetch_rellis.py first")
    print(f"RELLIS-3D test frames: {len(imgs)}\n")

    depth_net, seg_net = MonocularDepth(), SemanticSegmenter()
    totals = {m: dict(tp_block=0, false_block=0, false_safe=0, tp_trav=0,
                      gt_block=0, gt_trav=0) for m in METHODS}
    rows = []
    t0 = time.time()

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

        h, w = rgb.shape[:2]
        cam = Camera.from_fov(w, h, 70.0, 0.8)
        depth = to_metric(depth_net.infer(rgb), 1.0, 25.0)
        elev = ElevationNetwork(cam).process(depth)
        labels = seg_net.infer(rgb)
        valid = gt != GT_IGNORE
        neg = NegativeObstacleDetector(grad_percentile=99.2, min_area=60).process(
            depth, height=elev.height, valid=valid)
        unc = np.clip(np.abs(cv2.Laplacian(depth, cv2.CV_32F)) / 8.0, 0, 1)

        preds = predictions(labels, elev.height, neg.mask, unc, valid)
        for m, p in preds.items():
            sc = score(p, gt)
            for key in totals[m]:
                totals[m][key] += sc[key]
            rows.append(dict(frame=stem, method=m, **sc))

        if k % 5 == 0 or k == len(imgs):
            print(f"  {k}/{len(imgs)}  ({time.time() - t0:.0f}s)")

    os.makedirs("assets", exist_ok=True)
    with open("assets/benchmark_gt.csv", "w", newline="") as f:
        wri = csv.DictWriter(f, fieldnames=list(rows[0]))
        wri.writeheader(); wri.writerows(rows)

    print(f"\n{len(imgs)} frames, {time.time() - t0:.0f}s")
    print("scored against RELLIS-3D human labels (void/sky excluded)\n")
    hdr = (f"{'method':<26}{'IoU trav':>10}{'IoU block':>11}{'mIoU':>8}"
           f"{'FALSE-SAFE':>12}{'FALSE-BLOCK':>13}")
    print(hdr); print("-" * len(hdr))

    summary = {}
    for m in METHODS:
        t = totals[m]
        iou_t = iou(t["tp_trav"], t["false_safe"], t["false_block"])
        iou_b = iou(t["tp_block"], t["false_block"], t["false_safe"])
        fs = t["false_safe"] / t["gt_block"] if t["gt_block"] else float("nan")
        fb = t["false_block"] / t["gt_trav"] if t["gt_trav"] else float("nan")
        summary[m] = dict(iou_trav=round(iou_t, 4), iou_block=round(iou_b, 4),
                          miou=round((iou_t + iou_b) / 2, 4),
                          false_safe_rate=round(fs, 4), false_block_rate=round(fb, 4),
                          **t)
        print(f"{m:<26}{iou_t:>10.3f}{iou_b:>11.3f}{(iou_t + iou_b) / 2:>8.3f}"
              f"{fs * 100:>11.1f}%{fb * 100:>12.1f}%")

    print("\nFALSE-SAFE  = GT obstacle the method would have driven into (lower is better)")
    print("FALSE-BLOCK = GT drivable ground the method refused (lower is better)")

    json.dump(summary, open("assets/benchmark_gt.json", "w"), indent=2)
    print("\nwrote assets/benchmark_gt.csv and assets/benchmark_gt.json")


if __name__ == "__main__":
    main()
