#!/usr/bin/env python3
"""
main_argon.py  --  Argon Thermodynamics Discovery
Physics: P ∝ ρ^1.0  (Ideal Gas Law -- GAS PHASE ONLY)

Key fix for this dataset:
  Filtering to gas phase (ρ < 5 mol/L) is critical.
  The full NIST dataset at 150K spans the gas→liquid phase transition.
  Above ~5 mol/L argon is liquid and van der Waals dominates --
  the physics is no longer P ∝ ρ¹.

  Without this filter:
    - MLP learns a piecewise function (MSE=0.0025, exponent≈2.05)
    - KAN forces a single power law (MSE=0.13, exponent≈0.97)
    - "KAN wins on physics" only looks true because we're comparing
      the wrong model (MLP learned the CORRECT richer physics).

  With gas-phase filter only:
    - Both models should converge to exponent≈1.0
    - MSE values become comparable and meaningful

Other fixes:
  C1  -- PhysicsAuditor built with {"Density_mol_l": 1.0}
         Old version had {"a": 1.5} -- wrong physics for this domain!
  C5  -- KAN grid range dynamically set
  C6  -- 3-value return unpacked
  F2  -- Train/test split
  F8  -- Exponent error is primary
  M2  -- lambda 5 → 20 (gas phase is clean; gentle correction sufficient)
  M7  -- output_name="P"
"""

from utils.data_utils import load_argon, DOMAIN_REGISTRY
from experiments.comprehensive_comparison import (
    run_comprehensive_experiment,
    print_comparison_table,
)


def main():
    print("\n" + "=" * 64)
    print("AGENTIC PHYSICS DISCOVERY  --  ARGON THERMODYNAMICS")
    print("=" * 64)
    print("Note: Gas phase only (rho < 5 mol/L). Liquid phase excluded.")
    print("      Full dataset violates single power-law assumption.")

    cfg = DOMAIN_REGISTRY["argon"]

    # gas_phase_cutoff=5.0 set in loader (DOMAIN_REGISTRY default)
    X_log, y_log, raw_vars = load_argon(n_samples=500)

    df = run_comprehensive_experiment(
        X                = X_log,
        y                = y_log,
        raw_vars         = raw_vars,
        physics_targets  = cfg["physics"],       # {"Density_mol_l": 1.0}
        output_name      = cfg["output_name"],   # "P"
        mlp_lambda_start = cfg["lambda_start"],
        mlp_lambda_end   = cfg["lambda_end"],
        seed             = 42,
        epochs_kan       = cfg["epochs_kan"],
        epochs_mlp       = cfg["epochs_mlp"],
        tol              = cfg["tol"],
        test_size        = 0.2,
    )

    print_comparison_table(df)
    df.to_csv("results_argon.csv", index=False)
    print("\n[OK] Saved to results_argon.csv")


if __name__ == "__main__":
    main()
