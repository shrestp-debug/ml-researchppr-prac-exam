"""
experiments/ablation.py

Single-configuration ablation runner.

FIXES:
  C6 — controller.run() returns 3 values; old code unpacked only 2
       causing ValueError crash on every ablation run.
  C1 — PhysicsAuditor and controller both receive physics_targets
       as an argument — never hardcoded.
  F2 — All MSE on held-out test set.
  M5 — Seeded before model instantiation.
"""

import numpy as np
import torch

from agents.controller       import DiscoveryController
from auditor.physics_auditor import PhysicsAuditor
from kan.true_kan             import TrueKAN
from kan.linear_log_kan      import LinearLogKAN
from models.mlp_learner      import MLPLearner


def run_ablation(
    name:               str,
    X_train:            np.ndarray,
    y_train:            np.ndarray,
    X_test:             np.ndarray,
    y_test:             np.ndarray,
    raw_vars:           list,
    physics_targets:    dict,           # FIX C1
    model_type:         str   = "KAN",  # "KAN", "LinearKAN", or "MLP"
    enable_audit:       bool  = True,
    enable_correction:  bool  = True,
    use_llm:            bool  = False,
    epochs:             int   = 2000,
    lambda_phys_start:  float = 10.0,   # FIX M2
    lambda_phys_end:    float = 50.0,
    seed:               int   = 42,     # FIX M5
) -> dict:
    """
    Run a single ablation configuration and return metrics.

    Args:
        name:             human-readable experiment label
        X_train/y_train:  training data (log-space)
        X_test/y_test:    held-out test data (FIX F2)
        raw_vars:         variable names list
        physics_targets:  dict {var: target_exponent}  (FIX C1)
        model_type:       architecture to use
        enable_audit:     whether to audit physics after each iteration
        enable_correction:whether to pass feedback to learner
        use_llm:          whether to attach ConstraintFeedback
        epochs:           training epochs
        lambda_phys_*:    physics loss schedule (FIX M2)
        seed:             random seed (FIX M5)

    Returns:
        dict with: equation, converged, iterations, total_exp_error,
                   test_mse, history
    """
    # FIX M5: seed
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Build learner
    input_dim = X_train.shape[1]
    if model_type == "LinearKAN":
        learner = LinearLogKAN(input_dim=input_dim, epochs=epochs)
    elif model_type == "MLP":
        learner = MLPLearner(
            input_dim=input_dim,
            epochs=epochs,
            lambda_phys_start=lambda_phys_start,
            lambda_phys_end=lambda_phys_end,
        )
    else:  # "KAN" (default)
        learner = TrueKAN(input_dim=input_dim, epochs=epochs)

    # FIX C1: auditor receives physics_targets explicitly
    auditor = PhysicsAuditor(expected=physics_targets, tol=0.1) if enable_audit else None

    controller = DiscoveryController(
        learner=learner,
        auditor=auditor,
        physics_targets=physics_targets,
        max_iters=5,
        enable_audit=enable_audit,
        enable_correction=enable_correction,
        initial_lambda=lambda_phys_start,
    )

    # FIX C6: unpack all 3 values — old code did `eq, converged = ...` → crash
    equation, converged, iterations = controller.run(X_train, y_train, raw_vars)

    # FIX F2: evaluate on test set
    y_pred_test = learner.predict(X_test)
    test_mse    = float(np.mean((y_test.flatten() - y_pred_test) ** 2))

    total_exp_error = sum(
        abs(equation.exponents.get(var, 0.0) - physics_targets.get(var, 1.0))
        for var in raw_vars
        if var in physics_targets
    )

    print(f"\n[Ablation] {name}")
    print(f"  Equation:         {equation}")
    print(f"  Converged:        {converged}")
    print(f"  Iterations:       {iterations}")
    print(f"  Exponent error:   {total_exp_error:.4f}  [PRIMARY]")
    print(f"  Test MSE:         {test_mse:.4f}          [secondary]")

    return {
        "name":             name,
        "equation":         equation,
        "converged":        converged,
        "iterations":       iterations,
        "total_exp_error":  total_exp_error,
        "test_mse":         test_mse,
        "history":          [],
    }
