#!/usr/bin/env python3
"""
main_stars.py  —  Stellar Mass-Luminosity Discovery
Physics: L ∝ M^3.9  (Main Sequence only, M > 0.5 M_sun)

Key fix for this dataset:
  FIX C7: Filter to main-sequence stars only (M > 0.5 M_sun).

  Red dwarfs (M < 0.5 M_sun) are FULLY CONVECTIVE — no radiative core.
  They follow L ∝ M^2.3, not M^3.9. Mixing both populations makes a single
  power law physically meaningless.

  Without this filter:
    - Both models fit a compromise exponent ~3.2–3.6 (wrong for either)
    - MLP has very low MSE because it learned a piecewise function
      (which is physically MORE correct, but not what we want to discover)
    - "Agent improvement" just forces a single power law onto mixed data

  With main-sequence filter only:
    - Single regime: L ∝ M^3.9 is physically well-defined
    - MSE and exponent error become meaningful metrics
    - KAN's symbolic bias directly helps here

Other fixes:
  C1  — PhysicsAuditor built with {"Mass_Msun": 3.9}
         Old version had {"a": 1.5} — Kepler physics, not stellar!
  C5  — KAN grid range dynamically set
  C6  — 3-value return unpacked
  F2  — Train/test split
  F8  — Exponent error is primary
  M2  — lambda 5 → 25 (small dataset, needs gentle nudge)
  M7  — output_name="L"
"""

from utils.data_utils import load_stars, DOMAIN_REGISTRY
from experiments.comprehensive_comparison import (
    run_comprehensive_experiment,
    print_comparison_table,
)


def main():
    print("\n" + "=" * 64)
    print("AGENTIC PHYSICS DISCOVERY  --  STELLAR MASS-LUMINOSITY")
    print("=" * 64)
    print("Note: Main sequence only (M > 0.5 M_sun). Red dwarfs excluded.")
    print("      Red dwarfs follow L~M^2.3 -- different physics regime.")

    cfg = DOMAIN_REGISTRY["stars"]

    X_log, y_log, raw_vars = load_stars()  # min_mass=0.5 is default

    df = run_comprehensive_experiment(
        X                = X_log,
        y                = y_log,
        raw_vars         = raw_vars,
        physics_targets  = cfg["physics"],       # {"Mass_Msun": 3.9}
        output_name      = cfg["output_name"],   # "L"
        mlp_lambda_start = cfg["lambda_start"],
        mlp_lambda_end   = cfg["lambda_end"],
        seed             = 42,
        epochs_kan       = cfg["epochs_kan"],
        epochs_mlp       = cfg["epochs_mlp"],
        tol              = cfg["tol"],           # 0.15 (slightly relaxed)
        test_size        = 0.2,
    )

    print_comparison_table(df)
    df.to_csv("results_stars.csv", index=False)
    print("\n[OK] Saved to results_stars.csv")


if __name__ == "__main__":
    main()
