#!/usr/bin/env python3
"""
comprehensive_robustness_test.py

Three robustness experiments across all 5 domains:
  1. Seed robustness      — 5 random seeds, baseline models only
  2. Sample efficiency    — varying N, baseline models
  3. Agent sample efficiency — varying N with physics feedback

FIXES applied:
  C1  — PhysicsAuditor always receives physics_targets explicitly
  C6  — controller.run() 3-value unpack everywhere
  C7  — Stars: main-sequence only (M > 0.5)
  C8  — Wind: cut-in filter (v >= 3.5, P >= 10)
  F2  — Test MSE on held-out set in all experiments
  F8  — Exponent error reported as primary metric
  M1  — MLPLearner uses training data for gradient estimation
  M2  — Per-domain lambda_phys schedule
  M5  — Seeds explicitly set before every model instantiation

FIX epoch consistency — _baseline_pair() previously hardcoded epochs_kan=1000
  and epochs_mlp=2000 as defaults, and experiment_agent_efficiency() hardcoded
  epochs=1000/2000 directly. Neither read from DOMAIN_REGISTRY, so robustness
  experiments ran at different epoch counts than the main experiments (2000).
  All epoch values now come from DOMAIN_REGISTRY to ensure consistency.
"""

import numpy as np
import pandas as pd
import torch

from agents.controller           import DiscoveryController
from auditor.physics_auditor     import PhysicsAuditor
from kan.true_kan                 import TrueKAN
from models.mlp_learner          import MLPLearner
from utils.data_utils            import (
    load_wind, load_pendulum, load_kepler, load_argon, load_stars,
    train_test_split, DOMAIN_REGISTRY,
)


# ──────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────────────

def _load_domain(domain: str, n_samples: int = None, seed: int = 42):
    """Load a domain's data, applying all domain-specific filters."""
    loaders = {
        "wind":     lambda: load_wind(n_samples=n_samples, seed=seed),
        "pendulum": lambda: load_pendulum(n_samples=n_samples, seed=seed),
        "kepler":   lambda: load_kepler(n_samples=n_samples, seed=seed),
        "argon":    lambda: load_argon(n_samples=n_samples, seed=seed),
        "stars":    lambda: load_stars(n_samples=n_samples, seed=seed),
    }
    return loaders[domain]()


def _baseline_pair(
    X_tr, y_tr, X_te, y_te,
    var, target, lambda_start, lambda_end,
    seed,
    epochs_kan,   # FIX: always passed from DOMAIN_REGISTRY, no default
    epochs_mlp,   # FIX: always passed from DOMAIN_REGISTRY, no default
):
    """
    Train KAN and MLP baselines. Return (exp, test_mse, exp_error) for each.
    FIX M5: seeded. FIX F2: MSE on test. FIX F8: exp_error primary.
    FIX epochs: values come from DOMAIN_REGISTRY, not hardcoded defaults.
    """
    # KAN
    np.random.seed(seed)
    torch.manual_seed(seed)
    kan = TrueKAN(input_dim=1, epochs=epochs_kan)
    kan.fit(X_tr, y_tr, feedback=None, raw_vars=[var])
    kan_eq  = kan.discover_equation([var])
    kan_exp = kan_eq.exponents[var]
    kan_err = abs(kan_exp - target)
    kan_mse = float(np.mean((y_te - kan.predict(X_te)) ** 2))

    # MLP
    np.random.seed(seed + 1)
    torch.manual_seed(seed + 1)
    mlp = MLPLearner(input_dim=1, epochs=epochs_mlp,
                     lambda_phys_start=lambda_start,
                     lambda_phys_end=lambda_end)
    mlp.fit(X_tr, y_tr, feedback=None, raw_vars=[var])
    mlp_eq  = mlp.discover_equation([var])
    mlp_exp = mlp_eq.exponents[var]
    mlp_err = abs(mlp_exp - target)
    mlp_mse = float(np.mean((y_te - mlp.predict(X_te)) ** 2))

    return (kan_exp, kan_mse, kan_err,
            mlp_exp, mlp_mse, mlp_err)


# ──────────────────────────────────────────────────────────────────────────────
# EXPERIMENT 1: Seed Robustness
# ──────────────────────────────────────────────────────────────────────────────

def experiment_seed_robustness(domain: str, n_samples: int = 300, seeds=None):
    """Run baselines with 5 different random seeds."""
    if seeds is None:
        seeds = [42, 123, 456, 789, 999]

    cfg    = DOMAIN_REGISTRY[domain]
    var    = list(cfg["physics"].keys())[0]
    target = cfg["physics"][var]
    lam    = (cfg["lambda_start"], cfg["lambda_end"])
    # FIX: read epochs from registry
    epochs_kan = cfg["epochs_kan"]
    epochs_mlp = cfg["epochs_mlp"]

    print(f"\n{'='*72}")
    print(f"EXPERIMENT 1: SEED ROBUSTNESS  --  {domain.upper()}")
    print(f"Target: {var}^{target}  |  epochs_kan={epochs_kan}  epochs_mlp={epochs_mlp}")
    print(f"{'='*72}")

    # Load data once (same data for all seeds)
    X, y, _ = _load_domain(domain, n_samples, seed=42)
    rows = []

    for seed in seeds:
        X_tr, y_tr, X_te, y_te = train_test_split(X, y, seed=seed)
        kan_exp, kan_mse, kan_err, mlp_exp, mlp_mse, mlp_err = _baseline_pair(
            X_tr, y_tr, X_te, y_te,
            var, target, *lam, seed,
            epochs_kan=epochs_kan,
            epochs_mlp=epochs_mlp,
        )
        print(f"  seed={seed}:  KAN {var}^{kan_exp:.3f} err={kan_err:.3f} mse={kan_mse:.4f}"
              f"  |  MLP {var}^{mlp_exp:.3f} err={mlp_err:.3f} mse={mlp_mse:.4f}")
        rows.append(dict(seed=seed, domain=domain,
                         kan_exp=kan_exp, kan_mse=kan_mse, kan_err=kan_err,
                         mlp_exp=mlp_exp, mlp_mse=mlp_mse, mlp_err=mlp_err))

    df = pd.DataFrame(rows)
    print(f"\n  Summary -- exponent std across seeds:")
    print(f"    KAN: mean={df.kan_exp.mean():.3f}  std={df.kan_exp.std():.3f}"
          f"  mean_err={df.kan_err.mean():.3f}")
    print(f"    MLP: mean={df.mlp_exp.mean():.3f}  std={df.mlp_exp.std():.3f}"
          f"  mean_err={df.mlp_err.mean():.3f}")

    tol = cfg["tol"]
    for model, col, err_col in [("KAN", "kan_exp", "kan_err"), ("MLP", "mlp_exp", "mlp_err")]:
        std      = df[col].std()
        mean_err = df[err_col].mean()
        if std > 0.3:
            flag = "[!] HIGH variance"
        elif mean_err > tol:
            flag = "[!] BIASED (low variance but mean_err > tol)"
        else:
            flag = "[OK] stable"
        print(f"    {model}: {flag}  (std={std:.3f}, mean_err={mean_err:.3f}, tol={tol})")

    return df


# ──────────────────────────────────────────────────────────────────────────────
# EXPERIMENT 2: Sample Efficiency (baseline)
# ──────────────────────────────────────────────────────────────────────────────

def experiment_sample_efficiency(domain: str, sample_sizes=None, seed: int = 42):
    """Baseline models at varying training sizes."""
    if sample_sizes is None:
        sample_sizes = [50, 100, 200, 400]

    cfg    = DOMAIN_REGISTRY[domain]
    var    = list(cfg["physics"].keys())[0]
    target = cfg["physics"][var]
    lam    = (cfg["lambda_start"], cfg["lambda_end"])
    # FIX: read epochs from registry
    epochs_kan = cfg["epochs_kan"]
    epochs_mlp = cfg["epochs_mlp"]

    print(f"\n{'='*72}")
    print(f"EXPERIMENT 2: SAMPLE EFFICIENCY (baseline)  --  {domain.upper()}")
    print(f"Target: {var}^{target}  |  epochs_kan={epochs_kan}  epochs_mlp={epochs_mlp}")
    print(f"{'='*72}")

    rows   = []
    prev_n = None
    for n in sample_sizes:
        X, y, _ = _load_domain(domain, n, seed=seed)
        actual_n = len(X)
        if actual_n < 10:
            print(f"  N={n}: too few samples after filter, skipping")
            continue
        if prev_n is not None and actual_n == prev_n:
            print(f"  N={n:4d}: [SATURATED] dataset has only {actual_n} samples after filtering -- identical to previous row, skipping")
            continue
        prev_n = actual_n
        X_tr, y_tr, X_te, y_te = train_test_split(X, y, seed=seed)
        kan_exp, kan_mse, kan_err, mlp_exp, mlp_mse, mlp_err = _baseline_pair(
            X_tr, y_tr, X_te, y_te,
            var, target, *lam, seed,
            epochs_kan=epochs_kan,
            epochs_mlp=epochs_mlp,
        )
        winner = "KAN" if kan_err < mlp_err else "MLP"
        print(f"  N={n:4d}:  "
              f"KAN {var}^{kan_exp:.3f} err={kan_err:.3f} mse={kan_mse:.4f}"
              f"  |  MLP {var}^{mlp_exp:.3f} err={mlp_err:.3f} mse={mlp_mse:.4f}"
              f"  Winner(err)={winner}")
        rows.append(dict(n=n, domain=domain,
                         kan_exp=kan_exp, kan_mse=kan_mse, kan_err=kan_err,
                         mlp_exp=mlp_exp, mlp_mse=mlp_mse, mlp_err=mlp_err))

    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────────────────────
# EXPERIMENT 3: Agent Sample Efficiency
# ──────────────────────────────────────────────────────────────────────────────

def experiment_agent_efficiency(domain: str, sample_sizes=None, seed: int = 42):
    """Models with physics feedback at varying training sizes."""
    if sample_sizes is None:
        sample_sizes = [50, 100, 200, 400]

    cfg             = DOMAIN_REGISTRY[domain]
    var             = list(cfg["physics"].keys())[0]
    target          = cfg["physics"][var]
    physics_targets = cfg["physics"]
    lam             = (cfg["lambda_start"], cfg["lambda_end"])
    # FIX: read epochs from registry
    epochs_kan = cfg["epochs_kan"]
    epochs_mlp = cfg["epochs_mlp"]

    # FIX C1: build auditor once with correct targets
    auditor = PhysicsAuditor(expected=physics_targets, tol=cfg["tol"])

    print(f"\n{'='*72}")
    print(f"EXPERIMENT 3: AGENT SAMPLE EFFICIENCY  --  {domain.upper()}")
    print(f"Target: {var}^{target}  |  epochs_kan={epochs_kan}  epochs_mlp={epochs_mlp}")
    print(f"{'='*72}")

    rows   = []
    prev_n = None

    for n in sample_sizes:
        X, y, raw_vars = _load_domain(domain, n, seed=seed)
        actual_n = len(X)
        if actual_n < 10:
            continue
        if prev_n is not None and actual_n == prev_n:
            print(f"  N={n:4d}: [SATURATED] dataset has only {actual_n} samples -- skipping")
            continue
        prev_n = actual_n
        X_tr, y_tr, X_te, y_te = train_test_split(X, y, seed=seed)

        for model_name in ["KAN", "MLP"]:
            # FIX M5: seed before each learner
            np.random.seed(seed)
            torch.manual_seed(seed)

            # FIX epochs: use registry values, not hardcoded 1000/2000
            if model_name == "KAN":
                learner = TrueKAN(input_dim=1, epochs=epochs_kan)
            else:
                learner = MLPLearner(
                    input_dim=1,
                    epochs=epochs_mlp,
                    lambda_phys_start=lam[0],
                    lambda_phys_end=lam[1],
                )

            # Controller delegates to LLM-powered agent in agentic mode
            ctrl = DiscoveryController(
                learner=learner,
                auditor=auditor,
                physics_targets=physics_targets,
                max_iters=5,
                enable_audit=True,
                enable_correction=True,
                initial_lambda=lam[0],
            )

            # FIX C6: 3-value unpack
            eq, converged, iters = ctrl.run(X_tr, y_tr, raw_vars)

            exp = eq.exponents[var]
            err = abs(exp - target)
            mse = float(np.mean((y_te - learner.predict(X_te)) ** 2))

            print(f"  N={n:4d} {model_name}:  {var}^{exp:.3f}  err={err:.3f}  "
                  f"converged={converged}  iters={iters}  test_mse={mse:.4f}")
            rows.append(dict(n=n, domain=domain, model=model_name,
                             exp=exp, err=err, converged=converged,
                             iters=iters, test_mse=mse))

    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Configure per-domain sample sizes
    SAMPLE_SIZES = {
        "wind":     [100, 200, 300, 396],    # full dataset ~396 after cut-in filter
        "pendulum": [20, 40, 60, 90],         # full dataset = 90
        "kepler":   [50, 100, 200, 400],      # full dataset > 500
        "argon":    [15, 25, 35, 46],          # full dataset = 46 after gas-phase filter
        "stars":    [10, 15, 20, 31],          # full dataset = 31 after main-seq filter
    }

    all_results = {}

    for domain in DOMAIN_REGISTRY:
        print(f"\n\n{'#'*72}")
        print(f"# DOMAIN: {domain.upper()}")
        print(f"{'#'*72}")

        sizes = SAMPLE_SIZES[domain]
        max_n = max(sizes)

        r1 = experiment_seed_robustness(domain=domain, n_samples=max_n)
        r2 = experiment_sample_efficiency(domain=domain, sample_sizes=sizes)
        r3 = experiment_agent_efficiency(domain=domain, sample_sizes=sizes[:3])

        r1.to_csv(f"results_{domain}_seed_robustness.csv",   index=False)
        r2.to_csv(f"results_{domain}_sample_efficiency.csv", index=False)
        r3.to_csv(f"results_{domain}_agent_efficiency.csv",  index=False)
        all_results[domain] = (r1, r2, r3)

    print("\n\n" + "=" * 72)
    print("ALL ROBUSTNESS EXPERIMENTS COMPLETE")
    print("=" * 72)
    print("Output CSVs: results_{domain}_{experiment}.csv for each domain")
