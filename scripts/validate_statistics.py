"""Check the Model Arena statistics on synthetic data with known answers.

Runs Monte-Carlo simulations of the exact functions the Arena uses and prints:

* coverage of the 95 % bootstrap interval (simple and stratified resampling) – should be close to 0.95;
* type-I error of the paired difference test when two models are truly equal – should be ≤ 0.05;
* false-equivalence rate of the TOST test when the true difference equals the margin – should be ≤ 0.05.

Usage (from the repository root, about 2–4 minutes):
    backend\\.venv\\Scripts\\python scripts\\validate_statistics.py            (Windows)
    backend/.venv/bin/python scripts/validate_statistics.py --sims 300      (quicker, less precise)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from app.arena.stats import Resampling, bootstrap_cis, bootstrap_distributions, bootstrap_p_values  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sims", type=int, default=1000, help="simulations per check (default 1000)")
    ap.add_argument("--resamples", type=int, default=1000, help="bootstrap resamples per simulation")
    ap.add_argument("--seed", type=int, default=100)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    sims, boot = args.sims, args.resamples
    se = lambda p: (p * (1 - p) / sims) ** 0.5  # noqa: E731 - Monte-Carlo standard error

    print(f"{sims} simulations each, {boot} bootstrap resamples\n")
    for method in ("simple", "stratified"):
        covered = 0
        for s in range(sims):
            y = rng.integers(0, 2, 150)
            pred = np.where(rng.random(150) < 0.8, y, 1 - y)  # true accuracy 0.80, n = 150
            rs = Resampling(method, boot, s, strata=y if method == "stratified" else None)
            ci = bootstrap_cis("classification", y, pred, 2, rs)["accuracy_ci"]
            covered += ci[0] <= 0.8 <= ci[1]
        c = covered / sims
        print(f"95% interval coverage ({method:10s}): {c:.3f} ± {se(c):.3f}   (target 0.95)")

    rejected = 0
    for s in range(sims):
        y = rng.integers(0, 2, 200)
        a = np.where(rng.random(200) < 0.8, y, 1 - y)  # both models truly 0.80
        b = np.where(rng.random(200) < 0.8, y, 1 - y)
        da, db = bootstrap_distributions("classification", y, [a, b], 2, Resampling("simple", boot, s), ["accuracy"])
        p, _ = bootstrap_p_values(da["accuracy"] - db["accuracy"], 0.02)
        rejected += p < 0.05
    t1 = rejected / sims
    print(f"Type-I error of the difference test:  {t1:.3f} ± {se(t1):.3f}   (target ≤ 0.05)")

    false_eq = 0
    for s in range(sims):
        y = rng.integers(0, 2, 1000)
        a = np.where(rng.random(1000) < 0.85, y, 1 - y)  # true difference 0.05 = the margin
        b = np.where(rng.random(1000) < 0.80, y, 1 - y)
        da, db = bootstrap_distributions("classification", y, [a, b], 2, Resampling("simple", boot, s), ["accuracy"])
        _, p_tost = bootstrap_p_values(da["accuracy"] - db["accuracy"], 0.05)
        false_eq += p_tost < 0.05
    fe = false_eq / sims
    print(f"False equivalence at Δ = margin:      {fe:.3f} ± {se(fe):.3f}   (target ≤ 0.05)")
    print("\nThese checks cover the resampling and test functions on independent rows. They do not cover "
          "re-training variance, dependence between rows that the chosen split ignores, or shifts in the data.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
