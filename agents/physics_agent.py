"""
agents/physics_agent.py

A proper LLM-powered ReAct agent for physics discovery.

HOW THIS WORKS:
  1. Step 0: Train the model with NO physics feedback (pure data fit).
     This is identical to baseline — the model learns from data only.
  2. Evaluate: Extract the equation, audit physics.  If it passes, done.
  3. If it FAILS: Ask the LLM to analyze WHY the exponent is wrong
     and DECIDE what hyperparameters to use for the correction.
  4. The LLM chooses: lambda_phys, learning_rate, epochs, and whether
     to reset the model.  These are NOT hardcoded.
  5. Train with the LLM's chosen parameters.  Evaluate again.
  6. Repeat until physics passes or max_steps reached.

The LLM is called on EVERY correction step.  It is NOT a fallback.
It IS the decision-maker.
"""

import json
import numpy as np
import torch
import torch.nn as nn

from agents.llm_client import LLMClient
from agents.memory import DiscoveryMemory
from auditor.physics_auditor import PhysicsAuditor, PowerLawEquation


# ---------------------------------------------------------------------------
# System prompt — the LLM's identity and instructions
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """\
You are a **Physics Discovery Agent** controlling a machine learning model.

SITUATION:
A model (KAN or MLP) has been trained on scientific data to discover a
power-law relationship.  The model works in log-space:
  y = c * x^a  becomes  log(y) = log(c) + a*log(x)

The model was trained on DATA ONLY (no physics constraints).  Its discovered
exponent is WRONG.  Your job is to guide the model toward the correct
physics by choosing training hyperparameters for a physics-regularized
retraining step.

AVAILABLE TOOLS (pick exactly ONE per step):

1. "train" — Retrain the model WITH physics regularization.
   You must choose these hyperparameters:
   - lambda_phys  (float, 1-100): Strength of the physics penalty.
     Higher = the model tries harder to match the target exponent,
     but too high = ignores the data and overfits to the constraint.
   - learning_rate (float, 1e-4 to 0.05): Step size for gradient descent.
     Lower = more stable but slower.  Reduce if the exponent oscillates.
   - epochs (int, 200-5000): How many training iterations.
     More epochs = more compute but potentially better convergence.
   params: {"lambda_phys": float, "learning_rate": float, "epochs": int}

2. "reset_and_train" — Wipe the model weights and retrain from scratch
   with physics regularization.  Use this when the model is stuck in
   a bad local minimum (e.g., exponent barely changes between steps).
   params: same as "train"

3. "finish" — Stop and accept the best result found so far.
   Use when: the exponent is close enough, or you've tried everything.
   params: {"reason": "string explaining why you're stopping"}

REASONING PROCESS — think through these questions:
  1. How far is the current exponent from the target?
  2. Is the error improving, stagnant, or getting worse across steps?
  3. If improving: keep the strategy but maybe fine-tune (lower lr).
  4. If stagnant: try a bigger lambda or reset the model entirely.
  5. If overshooting: reduce lambda or learning rate.

RESPOND WITH ONLY A JSON OBJECT (no markdown fences, no extra text):
{
  "thought": "Your detailed reasoning about the current state and what to do",
  "tool": "train" | "reset_and_train" | "finish",
  "params": { ... }
}
"""


class PhysicsDiscoveryAgent:
    """
    LLM-powered agent that controls the physics discovery process.

    The agent does NOT use hardcoded hyperparameters.  Every correction
    step is decided by the LLM based on the observed results.

    Flow:
      Step 0:  Train WITHOUT physics feedback (pure baseline).
      Step 1+: LLM analyzes failure → chooses hyperparams → retrain.
    """

    def __init__(
        self,
        learner,
        auditor: PhysicsAuditor,
        physics_targets: dict,
        llm: LLMClient,
        max_steps: int = 8,
        output_name: str = "y",
    ):
        self.learner = learner
        self.auditor = auditor
        self.physics_targets = physics_targets
        self.llm = llm
        self.max_steps = max_steps
        self.output_name = output_name

        # Tracking
        self.history = []
        self.best_equation = None
        self.best_error = float("inf")

    # ------------------------------------------------------------------
    #  Main entry point
    # ------------------------------------------------------------------
    def run(self, X, y, raw_vars: list):
        """
        Returns (equation, converged, total_steps).

        Step 0: pure data training (no physics).
        Steps 1+: LLM-driven correction loop.
        """
        print(f"\n  [Agent] Starting physics discovery (max {self.max_steps} correction steps)")
        print(f"  [Agent] Target physics: {self.physics_targets}")
        print(f"  [Agent] LLM backend: {self.llm}")

        # ── Step 0: Train WITHOUT physics feedback ─────────────────────
        # This is identical to baseline.  The model learns from data only.
        print(f"\n  === Step 0: Baseline Training (NO physics feedback) ===")
        self.learner.fit(X, y, feedback=None, raw_vars=raw_vars)
        equation, passed = self._evaluate(raw_vars)

        if passed:
            print("  [Agent] Baseline already satisfies physics — no correction needed.")
            return equation, True, 1

        print(f"  [Agent] Baseline FAILED physics.  Starting LLM-driven correction.\n")

        # ── Steps 1+: LLM decides every correction ────────────────────
        for step in range(1, self.max_steps + 1):
            print(f"  === Agent Step {step} / {self.max_steps} ===")

            # 1. Ask the LLM what to do
            action = self._ask_llm(raw_vars, step)

            tool   = action["tool"]
            params = action.get("params", {})

            # 2. Show the LLM's full reasoning
            print(f"  [LLM Thought]  {action.get('thought', '(no reasoning provided)')}")
            print(f"  [LLM Decision] tool={tool}, params={json.dumps(params)}")

            # 3. Execute the chosen tool
            if tool == "finish":
                print(f"  [Agent] LLM decided to stop: {params.get('reason', 'no reason')}")
                break

            elif tool == "reset_and_train":
                self._reset_learner()
                self._train_with_physics(
                    X, y, raw_vars,
                    epochs=int(params.get("epochs", 2000)),
                    lr=float(params.get("learning_rate", 0.005)),
                    lambda_phys=float(params.get("lambda_phys", 25.0)),
                )

            elif tool == "train":
                self._train_with_physics(
                    X, y, raw_vars,
                    epochs=int(params.get("epochs", 2000)),
                    lr=float(params.get("learning_rate", 0.005)),
                    lambda_phys=float(params.get("lambda_phys", 25.0)),
                )

            else:
                print(f"  [Agent] Unknown tool '{tool}' — treating as 'train'")
                self._train_with_physics(
                    X, y, raw_vars,
                    epochs=int(params.get("epochs", 2000)),
                    lr=float(params.get("learning_rate", 0.005)),
                    lambda_phys=float(params.get("lambda_phys", 25.0)),
                )

            # 4. Evaluate
            equation, passed = self._evaluate(raw_vars)
            if passed:
                print(f"  [Agent] Physics PASSED at step {step}!")
                return equation, True, step + 1

            print()  # blank line between steps

        # Return best result found
        if self.best_equation is None:
            self.best_equation = PowerLawEquation(
                1.0, {v: 0.0 for v in raw_vars}, output_name=self.output_name
            )

        total = min(step + 1, self.max_steps + 1)
        print(f"\n  [Agent] Max steps reached.  Best error: {self.best_error:.4f}")
        return self.best_equation, False, total

    # ------------------------------------------------------------------
    #  LLM interaction — called on EVERY correction step
    # ------------------------------------------------------------------
    def _ask_llm(self, raw_vars, step):
        """Build the observation and ask the LLM for a decision."""
        observation = self._build_observation(raw_vars, step)

        try:
            raw_response = self.llm.generate(observation, system=SYSTEM_PROMPT)
            print(f"  [LLM Raw]      {raw_response[:200]}...")  # show first 200 chars
            action = self._parse_json(raw_response)
            if action and "tool" in action:
                return action
            print("  [Agent] LLM response was not valid JSON.")
        except Exception as e:
            print(f"  [Agent] LLM call failed: {e}")

        # Fallback ONLY if LLM is broken (not as normal operation)
        print("  [Agent] WARNING: Using emergency fallback (LLM unavailable)")
        return self._emergency_fallback(step)

    def _build_observation(self, raw_vars, step):
        """Build a detailed observation for the LLM to reason about."""
        lines = [
            "=" * 50,
            "CURRENT STATE — You must decide what to do next.",
            "=" * 50,
            "",
            f"Model type       : {type(self.learner).__name__}",
            f"Step             : {step} / {self.max_steps}",
            f"Variables        : {raw_vars}",
            "",
            "TARGET PHYSICS:",
        ]
        for var, target in self.physics_targets.items():
            lines.append(f"  {var} exponent should be {target}")

        lines.append("")
        lines.append("HISTORY OF ALL ATTEMPTS:")
        lines.append("-" * 50)

        for h in self.history:
            exps_str = ", ".join(
                f"{v}={h['exponents'].get(v, '?'):.4f}" for v in raw_vars
                if v in h["exponents"]
            )
            lines.append(
                f"  Step {h['step']:>2} | "
                f"exponents: [{exps_str}] | "
                f"total_error: {h['total_error']:.4f} | "
                f"passed: {h['passed']}"
            )
            if h.get("had_physics_feedback"):
                lines.append(
                    f"           | lr={h['lr']:.6f}, lambda={h['lambda']:.1f}, "
                    f"epochs={h['epochs']}"
                )
            else:
                lines.append("           | (no physics feedback — pure data fit)")

        lines.append("-" * 50)

        # Error trend
        if len(self.history) >= 2:
            errors = [h["total_error"] for h in self.history]
            prev, curr = errors[-2], errors[-1]
            delta = curr - prev
            if delta < -0.05:
                trend = f"IMPROVING (error dropped by {abs(delta):.4f})"
            elif delta > 0.05:
                trend = f"WORSENING (error increased by {delta:.4f})"
            else:
                trend = f"STAGNANT (error changed by only {delta:+.4f})"
            lines.append(f"\nTREND: {trend}")
            lines.append(f"Error history: {[f'{e:.4f}' for e in errors]}")

        lines.append(f"\nBest error so far: {self.best_error:.4f}")
        lines.append("")
        lines.append(
            "Based on this information, decide what tool to use and "
            "what hyperparameters to set.  Respond with ONLY a JSON object."
        )

        return "\n".join(lines)

    # ------------------------------------------------------------------
    #  JSON parsing
    # ------------------------------------------------------------------
    @staticmethod
    def _parse_json(text: str):
        """Robustly extract a JSON object from LLM output."""
        text = text.strip()

        # Strip markdown fences if present
        if "```" in text:
            parts = text.split("```")
            for part in parts:
                part = part.strip()
                if part.startswith("json"):
                    part = part[4:].strip()
                if part.startswith("{"):
                    try:
                        return json.loads(part)
                    except json.JSONDecodeError:
                        continue

        # Try direct extraction
        brace_start = text.find("{")
        brace_end = text.rfind("}")
        if brace_start != -1 and brace_end > brace_start:
            try:
                return json.loads(text[brace_start: brace_end + 1])
            except json.JSONDecodeError:
                pass

        return None

    # ------------------------------------------------------------------
    #  Emergency fallback (only if LLM is completely broken)
    # ------------------------------------------------------------------
    def _emergency_fallback(self, step):
        """
        Used ONLY when the LLM fails to respond.
        This is NOT the normal path — the LLM should always be called.
        """
        return {
            "thought": "EMERGENCY: LLM unavailable.  Using default physics penalty.",
            "tool": "train",
            "params": {"lambda_phys": 25.0, "learning_rate": 0.005, "epochs": 2000},
        }

    # ------------------------------------------------------------------
    #  Training execution
    # ------------------------------------------------------------------
    def _train_with_physics(self, X, y, raw_vars, epochs, lr, lambda_phys):
        """
        Train the model with physics regularization using the LLM's
        chosen hyperparameters.  NOT hardcoded.
        """
        print(f"  [Train] lr={lr:.6f}, lambda_phys={lambda_phys:.1f}, epochs={epochs}")

        # Save for logging
        self._last_lr = lr
        self._last_lambda = lambda_phys
        self._last_epochs = epochs

        # Update optimizer learning rate (MLP)
        if hasattr(self.learner, "opt"):
            for pg in self.learner.opt.param_groups:
                pg["lr"] = lr

        # Update learning rate (KAN)
        if hasattr(self.learner, "lr"):
            self.learner.lr = lr

        # Set lambda on MLP learner (start == end = constant)
        if hasattr(self.learner, "lambda_phys_start"):
            self.learner.lambda_phys_start = lambda_phys
            self.learner.lambda_phys_end = lambda_phys

        # Build physics feedback dict
        feedback = {
            var: {"target": self.physics_targets[var], "current": None, "weight": 1.0}
            for var in raw_vars if var in self.physics_targets
        }
        feedback["_iteration"] = len(self.history)
        feedback["_max_iters"] = self.max_steps
        feedback["_lambda_override"] = lambda_phys  # Direct control for KAN

        self.learner.fit(X, y, feedback=feedback, raw_vars=raw_vars, epochs=epochs)

    # ------------------------------------------------------------------
    #  Evaluation
    # ------------------------------------------------------------------
    def _evaluate(self, raw_vars):
        """Extract equation, audit physics, update tracking."""
        equation = self.learner.discover_equation(raw_vars, output_name=self.output_name)
        passed, audit_feedback = self.auditor.audit(equation)

        total_error = sum(
            abs(equation.exponents.get(var, 0.0) - self.physics_targets[var])
            for var in raw_vars if var in self.physics_targets
        )

        # Did this step use physics feedback?
        had_physics = len(self.history) > 0 or False  # step 0 never has physics
        step_idx = len(self.history)

        # Track best
        if total_error < self.best_error:
            self.best_error = total_error
            self.best_equation = equation

        self.history.append({
            "step": step_idx,
            "equation": str(equation),
            "exponents": dict(equation.exponents),
            "total_error": total_error,
            "passed": passed,
            "feedback": audit_feedback,
            "had_physics_feedback": step_idx > 0,
            "lr": getattr(self, "_last_lr", 0),
            "lambda": getattr(self, "_last_lambda", 0),
            "epochs": getattr(self, "_last_epochs", 0),
        })

        status = "PASSED" if passed else "FAILED"
        exp_str = ", ".join(f"{v}={equation.exponents.get(v, 0):.4f}" for v in raw_vars)
        print(f"  [Eval] Equation     : {equation}")
        print(f"  [Eval] Exponents    : {exp_str}")
        print(f"  [Eval] Total error  : {total_error:.4f}")
        print(f"  [Eval] Physics audit: {status}")

        return equation, passed

    # ------------------------------------------------------------------
    #  Model reset
    # ------------------------------------------------------------------
    def _reset_learner(self):
        """Re-initialise model weights from scratch."""
        print("  [Agent] Resetting model weights from scratch...")

        if hasattr(self.learner, "model"):
            # MLP — reinitialise all linear layers
            for m in self.learner.model.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_normal_(m.weight, gain=0.8)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
            import torch.optim as optim
            self.learner.opt = optim.Adam(
                self.learner.model.parameters(), lr=0.001
            )

        elif hasattr(self.learner, "functions"):
            # KAN — reinitialise spline weights
            for layer in self.learner.functions:
                if hasattr(layer, "_reset_parameters"):
                    layer._reset_parameters()
            self.learner.trained = False
