#!/usr/bin/env python3
"""
main_pendulum.py  --  Simple Pendulum Period Discovery
Physics: T ∝ l^0.5  (T = 2π√(l/g))

Fixes applied via shared modules:
  C1  -- PhysicsAuditor built with {"l": 0.5}
  C5  -- KAN grid range set dynamically
  C6  -- 3-value return unpacked
  F2  -- Train/test split
  F8  -- Exponent error is primary
  M2  -- lambda 5 → 20 (gentle -- baseline already close to correct)
  M7  -- output_name="T"
"""

from utils.data_utils import load_pendulum, DOMAIN_REGISTRY
from experiments.comprehensive_comparison import (
    run_comprehensive_experiment,
    print_comparison_table,
)


def main():
    print("\n" + "=" * 64)
    print("AGENTIC PHYSICS DISCOVERY  --  SIMPLE PENDULUM")
    print("=" * 64)

    cfg = DOMAIN_REGISTRY["pendulum"]

    X_log, y_log, raw_vars = load_pendulum()

    df = run_comprehensive_experiment(
        X                = X_log,
        y                = y_log,
        raw_vars         = raw_vars,
        physics_targets  = cfg["physics"],       # {"l": 0.5}
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
    df.to_csv("results_pendulum.csv", index=False)
    print("\n[OK] Saved to results_pendulum.csv")


if __name__ == "__main__":
    main()
