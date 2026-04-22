"""
agents/memory.py

DiscoveryMemory — logs each iteration's equation, audit result, and feedback.
Provides iteration history for ConstraintFeedback context and convergence analysis.

No bugs in the original. This version adds helper methods used by the
robustness tests (get_exponent_history, iterations_to_converge).
"""


class DiscoveryMemory:
    def __init__(self, experiment_name: str = "unnamed"):
        self.experiment_name = experiment_name
        self.history = []

    def log(self, iteration: int, equation, accepted: bool, feedback: dict):
        """Record one iteration's outcome."""
        self.history.append({
            "experiment":  self.experiment_name,
            "iteration":   iteration,
            "equation_str": str(equation),
            "exponents":   (
                dict(equation.exponents)
                if hasattr(equation, "exponents") else None
            ),
            "accepted":    accepted,
            "feedback":    feedback,
        })

    def summary(self) -> list:
        """Return full history (passed to ConstraintFeedback for context)."""
        return self.history

    def iterations_to_converge(self):
        """Return 1-indexed iteration count when first accepted, or None."""
        for entry in self.history:
            if entry["accepted"]:
                return entry["iteration"] + 1
        return None

    def get_exponent_history(self, var: str) -> list:
        """Return list of discovered exponent values for a given variable."""
        return [
            entry["exponents"][var]
            for entry in self.history
            if entry["exponents"] and var in entry["exponents"]
        ]

    def __len__(self):
        return len(self.history)
