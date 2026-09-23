"""Fetch a subset of the RELLIS-3D test split (5-label version).

The dataset is published as a single 5.2 GB archive. Rather than download all
of it, this reads the ZIP central directory over HTTP range requests and pulls
only the frames we need.

    python scripts/fetch_rellis.py --n 120

LICENCE: RELLIS-3D is CC BY-NC-SA 3.0 - non-commercial, attribution required,
share-alike. The downloaded frames are therefore NOT committed to this
repository (see .gitignore); run this script to obtain them locally.

    Jiang et al., "RELLIS-3D Dataset: Data, Benchmarks and Analysis", 2020.
    https://github.com/unmannedlab/RELLIS-3D
"""
from __future__ import annotations

import argparse
import json
import os
import time
import zipfile

REPO = "GAIA-URJC/Rellis3D-5Labels"
ZIP_IN_REPO = f"datasets/{REPO}/rellis3D_5.zip"
OUT = os.path.join("data", "rellis3d")


def retry(fn, n=5, wait=3, label=""):
    for i in range(n):
        try:
            return fn()
        except Exception as exc:                      # noqa: BLE001
            print(f"  {label} attempt {i + 1}/{n}: {type(exc).__name__}: {str(exc)[:70]}")
            time.sleep(wait)
    raise RuntimeError(f"giving up on {label}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=120, help="number of frames")
    ap.add_argument("--stride", type=int, default=12,
                    help="take every Nth frame, so the subset spans whole sequences")
    args = ap.parse_args()

    from huggingface_hub import HfFileSystem

    os.makedirs(f"{OUT}/images", exist_ok=True)
    os.makedirs(f"{OUT}/masks", exist_ok=True)

    def open_zip():
        return zipfile.ZipFile(HfFileSystem().open(ZIP_IN_REPO, "rb"))

    t0 = time.time()
    z = retry(open_zip, label="open zip")
    names = z.namelist()
    print(f"remote index: {len(names)} entries in {time.time() - t0:.1f}s")

    imgs = sorted(n for n in names if "/test/" in n and n.endswith(".jpg"))
    masks = {os.path.basename(n).rsplit("_5", 1)[0]: n
             for n in names if "/test/" in n and n.endswith(".png")}
    print(f"test split: {len(imgs)} images, {len(masks)} masks")

    picked, manifest = 0, []
    for path in imgs[::args.stride]:
        if picked >= args.n:
            break
        stem = os.path.splitext(os.path.basename(path))[0]
        mpath = masks.get(stem)
        if mpath is None:
            continue
        try:
            img = retry(lambda p=path: z.read(p), n=3, label=stem)
            msk = retry(lambda p=mpath: z.read(p), n=3, label=stem)
        except RuntimeError:
            continue
        open(f"{OUT}/images/{stem}.jpg", "wb").write(img)
        open(f"{OUT}/masks/{stem}.png", "wb").write(msk)
        manifest.append(stem)
        picked += 1
        if picked % 10 == 0:
            print(f"  {picked}/{args.n}  ({time.time() - t0:.0f}s)")

    json.dump({
        "dataset": "RELLIS-3D (5-label relabelling)",
        "source": f"https://huggingface.co/datasets/{REPO}",
        "upstream": "https://github.com/unmannedlab/RELLIS-3D",
        "citation": "Jiang et al., RELLIS-3D Dataset: Data, Benchmarks and Analysis, 2020",
        "license": "CC BY-NC-SA 3.0",
        "split": "test",
        "frames": manifest,
    }, open(f"{OUT}/DATASET.json", "w"), indent=2)
    print(f"\n{picked} frames -> {OUT}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
