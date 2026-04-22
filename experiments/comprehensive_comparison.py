"""
experiments/comprehensive_comparison.py

Runs all 4 configs (KAN baseline, KAN agentic, MLP baseline, MLP agentic)
and produces a comparison DataFrame.

Agentic mode now uses a real LLM-powered PhysicsDiscoveryAgent that
autonomously decides training strategy, hyperparameters, and stopping.
"""

import numpy as np
import pandas as pd
import torch

from agents.controller       import DiscoveryController
from auditor.physics_auditor import PhysicsAuditor
from kan.true_kan             import TrueKAN
from models.mlp_learner      import MLPLearner
from utils.data_utils        import train_test_split


def run_comprehensive_experiment(
    X:                  np.ndarray,
    y:                  np.ndarray,
    raw_vars:           list,
    physics_targets:    dict,
    output_name:        str   = "y",
    mlp_lambda_start:   float = 10.0,
    mlp_lambda_end:     float = 50.0,
    seed:               int   = 42,
    epochs_kan:         int   = 2000,
    epochs_mlp:         int   = 2000,
    tol:                float = 0.1,
    test_size:          float = 0.2,
    max_iters:          int   = 5,
) -> pd.DataFrame:

    if not physics_targets:
        raise ValueError("physics_targets must be a non-empty dict.")

    X_train, y_train, X_test, y_test = train_test_split(X, y, test_size, seed)
    print(f"\n  Data split: train={len(X_train)}, test={len(X_test)}")

    physics_auditor = PhysicsAuditor(expected=physics_targets, tol=tol)

    # ── Resolve LLM client once (shared by both KAN and MLP agentic runs) ──
    llm_client = None
    try:
        from agents.llm_client import get_llm_client
        llm_client = get_llm_client()
        print(f"  LLM backend: {llm_client}")
    except RuntimeError as e:
        print(f"\n  [WARNING] {e}")
        print("  Agentic runs will use heuristic fallback (no LLM).\n")

    configs = [
        ("KAN", False),
        ("KAN", True),
        ("MLP", False),
        ("MLP", True),
    ]

    results          = []
    baseline_results = {}

    for model_name, enable_correction in configs:
        tag = f"{model_name}_{'agentic' if enable_correction else 'baseline'}"
        print(f"\n{'='*60}\n  Config: {tag}\n{'='*60}")

        # Short-circuit: if baseline already passed, agent adds nothing
        if enable_correction:
            baseline = baseline_results.get(model_name)
            if baseline and baseline["converged"]:
                print(f"  Baseline {model_name} already satisfies physics -- skipping agent.")
                row = dict(baseline)
                row.update({"config": tag, "agentic": True, "iterations": 0})
                results.append(row)
                continue

        np.random.seed(seed)
        torch.manual_seed(seed)

        if model_name == "KAN":
            learner = TrueKAN(input_dim=X_train.shape[1], epochs=epochs_kan)
        else:
            learner = MLPLearner(
                input_dim=X_train.shape[1],
                epochs=epochs_mlp,
                lambda_phys_start=mlp_lambda_start,
                lambda_phys_end=mlp_lambda_end,
            )

        controller = DiscoveryController(
            learner=learner,
            auditor=physics_auditor,
            physics_targets=physics_targets,
            max_iters=max_iters,
            enable_audit=True,
            enable_correction=enable_correction,
            output_name=output_name,
            # Agent-specific params (ignored in baseline mode)
            llm_client=llm_client,
            initial_lambda=mlp_lambda_start,
        )

        equation, converged, iterations = controller.run(X_train, y_train, raw_vars)

        total_exp_error = sum(
            abs(equation.exponents.get(var, 0.0) - physics_targets[var])
            for var in raw_vars if var in physics_targets
        )

        y_pred_test = learner.predict(X_test)
        test_mse    = float(np.mean((y_test.flatten() - y_pred_test) ** 2))

        row = {
            "config":          tag,
            "model":           model_name,
            "agentic":         enable_correction,
            "converged":       converged,
            "iterations":      iterations,
            "total_exp_error": total_exp_error,
            "test_mse":        test_mse,
            "equation":        str(equation),
        }
        for var in raw_vars:
            row[f"exp_{var}"]       = equation.exponents.get(var, float("nan"))
            row[f"exp_error_{var}"] = abs(
                equation.exponents.get(var, 0.0) - physics_targets.get(var, 1.0)
            )

        if not enable_correction:
            baseline_results[model_name] = row

        results.append(row)

        print(f"\n  Result: {tag}")
        print(f"    Equation:       {equation}")
        print(f"    Converged:      {converged}")
        print(f"    Iterations:     {iterations}")
        print(f"    Exponent error: {total_exp_error:.4f}  <-- PRIMARY METRIC")
        print(f"    Test MSE:       {test_mse:.4f}         <-- secondary (held-out)")

    return pd.DataFrame(results)


def print_comparison_table(df: pd.DataFrame):
    sep = "=" * 72

    print(f"\n{sep}")
    print("RESULTS -- EXPONENT ERROR IS PRIMARY METRIC")
    print(f"{sep}\n")

    for model in ["KAN", "MLP"]:
        sub  = df[df["model"] == model]
        base = sub[~sub["agentic"]].iloc[0]
        agt  = sub[sub["agentic"]].iloc[0]

        print(f"  {model}:")
        print(f"    {'':30s} {'Baseline':>12}  {'Agentic':>12}  {'Delta':>10}")
        print(f"    {'-'*68}")

        be, ae = base["total_exp_error"], agt["total_exp_error"]
        delta  = ((be - ae) / be * 100) if be > 0 else 0.0
        flag   = "BETTER" if delta > 0 else ("WORSE" if delta < 0 else "--")
        print(f"    {'Exponent error  [PRIMARY]':<30} {be:>12.4f}  {ae:>12.4f}  "
              f"{delta:>+8.1f}%  {flag}")

        bm, am = base["test_mse"], agt["test_mse"]
        if bm < 0.001:
            print(f"    {'Test MSE        [secondary]':<30} {bm:>12.4f}  {am:>12.4f}       --  (baseline near zero)")
        else:
            dm = ((bm - am) / bm * 100)
            print(f"    {'Test MSE        [secondary]':<30} {bm:>12.4f}  {am:>12.4f}  {dm:>+8.1f}%")

        print(f"    {'Iterations':<30} {base['iterations']:>12}  {agt['iterations']:>12}")
        print(f"    {'Converged':<30} {str(base['converged']):>12}  {str(agt['converged']):>12}")
        print(f"    Baseline eq: {base['equation']}")
        print(f"    Agentic  eq: {agt['equation']}")
        print()

    print(f"\n{sep}")
    print("KAN vs MLP  --  BASELINE COMPARISON")
    print(f"{sep}")
    _head_to_head(df, agentic=False)

    print(f"\n{sep}")
    print("KAN vs MLP  --  AGENTIC COMPARISON")
    print(f"{sep}")

    kan_agt = df[(df["model"] == "KAN") & (df["agentic"] == True)].iloc[0]
    mlp_agt = df[(df["model"] == "MLP") & (df["agentic"] == True)].iloc[0]

    if kan_agt["iterations"] == 0 and mlp_agt["iterations"] == 0:
        print("\n  [Not shown] Both models satisfied physics at baseline.")
        print("  Agentic comparison is identical to baseline -- omitted to avoid redundancy.")
    else:
        _head_to_head(df, agentic=True)


def _head_to_head(df: pd.DataFrame, agentic: bool):
    kan = df[(df["model"] == "KAN") & (df["agentic"] == agentic)].iloc[0]
    mlp = df[(df["model"] == "MLP") & (df["agentic"] == agentic)].iloc[0]

    print(f"\n  {'Metric':30s} {'KAN':>12}  {'MLP':>12}  {'Winner'}")
    print(f"  {'-'*60}")

    ke, me = kan["total_exp_error"], mlp["total_exp_error"]
    ki, mi = kan["iterations"], mlp["iterations"]

    # Suppress winner when agentic and one model was never corrected
    suppress_winner = agentic and (ki == 0 or mi == 0)

    if suppress_winner:
        print(f"  {'Exponent error [PRIMARY]':<30} {ke:>12.4f}  {me:>12.4f}  --")
    else:
        print(f"  {'Exponent error [PRIMARY]':<30} {ke:>12.4f}  {me:>12.4f}  "
              f"{'KAN' if ke < me else 'MLP'}")

    km, mm = kan["test_mse"], mlp["test_mse"]
    if suppress_winner:
        print(f"  {'Test MSE [secondary]':<30} {km:>12.4f}  {mm:>12.4f}  --")
    else:
        print(f"  {'Test MSE [secondary]':<30} {km:>12.4f}  {mm:>12.4f}  "
              f"{'KAN' if km < mm else 'MLP'}")

    print(f"  {'Iterations':<30} {ki:>12}  {mi:>12}  "
          f"{'KAN' if ki < mi else ('MLP' if mi < ki else 'Tie')}")

    print(f"  {'Converged':<30} {str(kan['converged']):>12}  {str(mlp['converged']):>12}")
    print(f"\n  KAN eq: {kan['equation']}")
    print(f"  MLP eq: {mlp['equation']}")

    if suppress_winner:
        if ki == 0 and mi == 0:
            print("\n  * Winner omitted: both models satisfied physics at baseline.")
            print("    No agent correction was applied to either model.")
            print("    This comparison is identical to the baseline table above.")
        else:
            print("\n  * Winner omitted: one model satisfied physics at baseline")
            print("    and was not corrected by the agent. Direct comparison is not meaningful.")
