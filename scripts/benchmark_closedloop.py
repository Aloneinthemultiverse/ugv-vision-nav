"""Closed-loop navigation benchmark.

Still-frame IoU measures perception. This measures whether the vehicle
*arrives*, driving the real Layer 3 and Layer 4 code through randomised worlds
with a limited, occluded, noisy sensor.

Metrics match the off-road navigation literature so the numbers are comparable
in kind (not in setting) to published stacks:

    SR   success rate
    SPL  success weighted by inverse path length

Ablations run by default, because a single success rate says nothing about
which component earned it:

    memory on/off        does remembering what left the field of view help?
    pursuit vs mppi      does the sampling local planner beat pure pursuit?

    python scripts/benchmark_closedloop.py --episodes 25
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ugvnav.sim import random_world, run_episode          # noqa: E402

DIFFICULTIES = ("easy", "medium", "hard")


def evaluate(episodes: int, difficulty: str, **kw) -> dict:
    runs = [run_episode(random_world(s, difficulty=difficulty), seed=s, **kw)
            for s in range(episodes)]
    n = len(runs)
    return {
        "episodes": n,
        "sr": sum(r.success for r in runs) / n,
        "spl": sum(r.spl for r in runs) / n,
        "collisions": sum(r.collided for r in runs),
        "timeouts": sum(r.timeout for r in runs),
        "mean_path_m": round(sum(r.path_length for r in runs) / n, 2),
    }


def table(title: str, rows: dict) -> None:
    print(f"\n{title}")
    hdr = f"{'':<10}{'SR':>9}{'SPL':>8}{'collide':>9}{'timeout':>9}{'path m':>9}"
    print(hdr); print("-" * len(hdr))
    for name, m in rows.items():
        print(f"{name:<10}{m['sr'] * 100:>8.1f}%{m['spl']:>8.3f}"
              f"{m['collisions']:>9}{m['timeouts']:>9}{m['mean_path_m']:>9}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=25)
    args = ap.parse_args()
    t0 = time.time()
    out: dict = {"episodes_per_cell": args.episodes}

    print(f"closed-loop benchmark: {args.episodes} episodes per difficulty")

    full = {d: evaluate(args.episodes, d, controller="pursuit", remember=True)
            for d in DIFFICULTIES}
    table("FULL STACK  (map memory on, pure pursuit)", full)
    out["full"] = full

    no_mem = {d: evaluate(args.episodes, d, controller="pursuit", remember=False)
              for d in DIFFICULTIES}
    table("ABLATION: map memory OFF  (costmap rebuilt from each frame)", no_mem)
    out["no_memory"] = no_mem

    mppi = {d: evaluate(args.episodes, d, controller="mppi", remember=True)
            for d in DIFFICULTIES}
    table("ABLATION: MPPI local planner instead of pure pursuit", mppi)
    out["mppi"] = mppi

    print("\ndelta from removing map memory (SR percentage points):")
    for d in DIFFICULTIES:
        print(f"  {d:<8}{(full[d]['sr'] - no_mem[d]['sr']) * 100:+6.1f}")

    os.makedirs("assets", exist_ok=True)
    json.dump(out, open("assets/benchmark_closedloop.json", "w"), indent=2)
    print(f"\n{time.time() - t0:.0f}s -> assets/benchmark_closedloop.json")


if __name__ == "__main__":
    main()
