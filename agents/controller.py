"""
agents/controller.py

Orchestrates physics discovery for a single model.

Two modes:
  Baseline (enable_correction=False):
      Train once -> audit once -> report -> STOP.
      No feedback, no re-training, no LLM calls.

  Agentic (enable_correction=True):
      Delegates entirely to PhysicsDiscoveryAgent, which uses an LLM
      to autonomously decide training strategy, hyperparameters,
      and when to stop.
"""

from auditor.physics_auditor import PowerLawEquation


class DiscoveryController:
    """
    Thin wrapper that routes to the correct execution mode.

    Baseline:  deterministic single-pass training (no agent).
    Agentic:   hands off to PhysicsDiscoveryAgent for LLM-driven control.
    """

    def __init__(
        self,
        learner,
        auditor=None,
        physics_targets: dict = None,
        max_iters: int = 5,
        enable_audit: bool = True,
        enable_correction: bool = True,
        output_name: str = "y",
        # Agent-specific (only used when enable_correction=True)
        llm_client=None,
        initial_lambda: float = 10.0,
    ):
        self.learner = learner
        self.auditor = auditor
        self.physics_targets = physics_targets or {}
        self.max_iters = max_iters
        self.enable_audit = enable_audit
        self.enable_correction = enable_correction
        self.output_name = output_name
        self.llm_client = llm_client
        self.initial_lambda = initial_lambda

    def run(self, X, y, raw_vars: list = None):
        """
        Returns (equation, converged, iterations).

        Baseline mode (enable_correction=False):
            Trains exactly once.  Audits physics and reports pass/fail.

        Agentic mode (enable_correction=True):
            Creates a PhysicsDiscoveryAgent and lets the LLM drive.
        """
        if raw_vars is None:
            raw_vars = list(self.physics_targets.keys()) or ["x0"]

        if self.enable_correction:
            return self._run_agentic(X, y, raw_vars)
        else:
            return self._run_baseline(X, y, raw_vars)

    # ------------------------------------------------------------------
    #  Baseline mode — unchanged from the original
    # ------------------------------------------------------------------
    def _run_baseline(self, X, y, raw_vars):
        """Single training pass, no feedback, no iteration."""
        print(f"\n  --- Baseline (single training run) ---")
        print(f"  Training WITHOUT physics feedback")

        self.learner.fit(X, y, feedback=None, raw_vars=raw_vars)

        # Extract equation
        if hasattr(self.learner, "discover_equation"):
            equation = self.learner.discover_equation(
                raw_vars, output_name=self.output_name
            )
        else:
            equation = PowerLawEquation(
                1.0, {v: 0.0 for v in raw_vars}, output_name=self.output_name
            )

        print(f"  Discovered: {equation}")

        # Audit
        converged = False
        if self.enable_audit and self.auditor:
            passed, audit_feedback = self.auditor.audit(equation)
            status = "PASSED" if passed else "FAILED"
            print(f"  Physics audit: {status}")
            converged = passed
        else:
            converged = True

        return equation, converged, 1

    # ------------------------------------------------------------------
    #  Agentic mode — delegates to PhysicsDiscoveryAgent
    # ------------------------------------------------------------------
    def _run_agentic(self, X, y, raw_vars):
        """LLM-driven agent loop."""
        from agents.physics_agent import PhysicsDiscoveryAgent

        if self.llm_client is None:
            # Try to auto-detect an available LLM
            try:
                from agents.llm_client import get_llm_client
                self.llm_client = get_llm_client()
            except RuntimeError as e:
                print(f"\n  [WARNING] {e}")
                print("  The agent REQUIRES an LLM to make decisions.")
                print("  Running with emergency fallback (fixed defaults).\n")
                self.llm_client = _HeuristicOnlyLLM()

        agent = PhysicsDiscoveryAgent(
            learner=self.learner,
            auditor=self.auditor,
            physics_targets=self.physics_targets,
            llm=self.llm_client,
            max_steps=self.max_iters,
            output_name=self.output_name,
        )

        return agent.run(X, y, raw_vars)


class _HeuristicOnlyLLM:
    """
    Emergency stub -- used ONLY when no API key is set.
    The agent will use fixed emergency defaults instead of LLM reasoning.
    This is NOT the intended mode of operation.
    """

    def generate(self, prompt: str, system: str = None) -> str:
        return "NO_LLM_AVAILABLE"

    def __repr__(self):
        return "EMERGENCY_FALLBACK (no LLM key -- set GROQ_API_KEY in .env)"
