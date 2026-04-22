#!/usr/bin/env python3
"""
main_kepler.py  --  Kepler Orbital Period Discovery
Physics: T ∝ a^1.5  (Kepler's Third Law)

Fixes applied via shared modules:
  C1  -- PhysicsAuditor built with {"a": 1.5}
  C5  -- KAN grid range set dynamically from actual data.
         Original [-2, 6] excluded planets with a < 0.14 AU (log(a) < -2).
         TrueKAN.fit() now calls _update_grids() before training.
  C6  -- 3-value return unpacked
  F2  -- Train/test split
  F8  -- Exponent error is primary
  M2  -- lambda 10 → 40
  M7  -- output_name="T"
"""

from utils.data_utils import load_kepler, DOMAIN_REGISTRY
from experiments.comprehensive_comparison import (
    run_comprehensive_experiment,
    print_comparison_table,
)


def main():
    print("\n" + "=" * 64)
    print("AGENTIC PHYSICS DISCOVERY  --  KEPLER (EXOPLANETS)")
    print("=" * 64)

    cfg = DOMAIN_REGISTRY["kepler"]

    X_log, y_log, raw_vars = load_kepler(n_samples=500)

    df = run_comprehensive_experiment(
        X                = X_log,
        y                = y_log,
        raw_vars         = raw_vars,
        physics_targets  = cfg["physics"],       # {"a": 1.5}
        output_name      = cfg["output_name"],   # "T"
        mlp_lambda_start = cfg["lambda_start"],
        mlp_lambda_end   = cfg["lambda_end"],
        seed             = 42,
        epochs_kan       = cfg["epochs_kan"],
        epochs_mlp       = cfg["epochs_mlp"],
        tol              = cfg["tol"],
        test_size        = 0.2,
    )

    print_comparison_table(df)
    df.to_csv("results_kepler.csv", index=False)
    print("\n[OK] Saved to results_kepler.csv")


if __name__ == "__main__":
    main()
