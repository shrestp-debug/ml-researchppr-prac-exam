"""
auditor/constraint_feedback.py

LLM-based reasoning layer. Explains physics violations and suggests
corrective constraints. Does NOT decide pass/fail — that is PhysicsAuditor's job.

No structural bugs in the original. This version is cleaned up and made
robust when feedback contains partial or malformed entries.
"""


class ConstraintFeedback:
    """
    Generates natural-language explanations of physics audit failures and
    proposes corrective constraints for the next training iteration.

    When llm=None (default), returns a structured mock response that still
    drives the agentic correction loop correctly.
    """

    def __init__(self, llm=None):
        self.llm = llm

    def analyze(self, equation, feedback: dict, history: list) -> dict:
        """
        Args:
            equation:  PowerLawEquation from the learner
            feedback:  dict from PhysicsAuditor.audit() — variables that failed
            history:   list of past memory entries from DiscoveryMemory

        Returns:
            {
                "explanation":           str — human-readable problem description
                "suggested_constraints": str — recommended fix for next iteration
            }
        """
        if self.llm is not None:
            return self._call_llm(equation, feedback, history)

        return self._mock_response(equation, feedback, history)

    def _mock_response(self, equation, feedback: dict, history: list = None) -> dict:
        explanations = []
        suggestions = []

        for var, details in feedback.items():
            if var == "error":
                explanations.append(str(details))
                continue

            if not isinstance(details, dict):
                continue

            current = details.get("current")
            target = details.get("target")

            if current is None or target is None:
                continue

            diff = float(current) - float(target)
            direction = "too high" if diff > 0 else "too low"

            explanations.append(
                f"'{var}' exponent is {current:.3f} but should be {target:.3f} "
                f"(off by {diff:+.3f}, {direction})"
            )
            suggestions.append(
                f"Constrain '{var}' exponent to "
                f"[{target - 0.1:.2f}, {target + 0.1:.2f}]"
            )

        if not explanations:
            explanations = ["Equation violates physics constraints (see feedback)"]
            suggestions = [str(feedback)]

        n_attempts = len(history) if history else 0
        header = (
            f"After {n_attempts} attempt(s): " if n_attempts > 0 else ""
        )

        return {
            "explanation": header + "; ".join(explanations),
            "suggested_constraints": "; ".join(suggestions),
        }

    def _call_llm(self, equation, feedback, history) -> dict:
        """Real LLM call path — only used when self.llm is set."""
        prompt = f"""You are a physics-aware scientific auditor.

A machine learning model proposed this equation:
{equation}

The physics auditor flagged these violations:
{feedback}

Previous attempts in this session:
{history}

Tasks:
1. Explain clearly why this equation violates physics.
2. Suggest concrete corrective constraints (exponent ranges).
3. Do NOT decide acceptance/rejection yourself.

Respond in plain text."""

        response = self.llm.invoke(prompt)
        return {
            "explanation": str(response),
            "suggested_constraints": str(feedback),
        }
