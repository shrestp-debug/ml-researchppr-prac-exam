#!/usr/bin/env python3
"""
main_wind.py  --  Wind Turbine Power Curve Discovery
Physics: P ∝ v³  (Betz kinetic energy law: P = ½ρAv³)

Fixes applied via shared modules:
  C1  -- PhysicsAuditor built with {"v": 3.0}
  C5  -- KAN grid range set dynamically in TrueKAN.fit()
  C6  -- controller.run() returns 3 values, all unpacked
  C7  -- N/A (single-regime dataset)
  C8  -- load_wind() filters v >= 3.5 m/s, P >= 10 kW (below cut-in invalid)
  F2  -- 80/20 train/test split; MSE on test only
  F8  -- Exponent error is primary metric
  M1  -- MLPLearner uses actual training data for gradient estimation
  M2  -- lambda_phys = 10 → 50 (tuned for this dataset)
  M3  -- improvement % shown only for exponent error
  M5  -- Seeded
  M7  -- output_name="P"

FIX: Use full dataset (no n_samples cap). A previous version passed
n_samples=500 which reduced 38k samples to 500, making results
non-reproducible from the output log and inconsistent with other mains.
"""

from utils.data_utils import load_wind, DOMAIN_REGISTRY
from experiments.comprehensive_comparison import (
    run_comprehensive_experiment,
    print_comparison_table,
)


def main():
    print("\n" + "=" * 64)
    print("AGENTIC PHYSICS DISCOVERY  --  WIND TURBINE")
    print("=" * 64)

    cfg = DOMAIN_REGISTRY["wind"]

    # FIX: No n_samples argument — use the full dataset (~38k after cut-in filter).
    # Passing n_samples=500 was a bug introduced in the previous revision that
    # made this output non-reproducible from the logged results.
    X_log, y_log, raw_vars = load_wind()

    df = run_comprehensive_experiment(
        X                = X_log,
        y                = y_log,
        raw_vars         = raw_vars,
        physics_targets  = cfg["physics"],       # {"v": 3.0}
        output_name      = cfg["output_name"],   # "P"
        mlp_lambda_start = cfg["lambda_start"],  # 10.0
        mlp_lambda_end   = cfg["lambda_end"],    # 50.0
        seed             = 42,
        epochs_kan       = cfg["epochs_kan"],    # 1000
        epochs_mlp       = cfg["epochs_mlp"],    # 500
        tol              = cfg["tol"],
        test_size        = 0.2,
    )

    print_comparison_table(df)
    df.to_csv("results_wind.csv", index=False)
    print("\n[OK] Saved to results_wind.csv")


if __name__ == "__main__":
    main()
