"""
auditor/physics_auditor.py

FIX C1 — PhysicsAuditor no longer hardcodes any domain-specific expected
          exponents. They are ALWAYS injected at construction time.
          Old versions had self.expected = {"a": 1.5} hardcoded even in the
          thermodynamics and stellar datasets, making audits against Kepler
          physics regardless of which dataset was running.

FIX M7 — PowerLawEquation accepts an output_name so printed equations say
          "T = ..." for pendulum, "L = ..." for stars, etc., not always "P =".
"""


class PowerLawEquation:
    """
    Represents a power-law equation:
        output = constant * x0^e0 * x1^e1 * ...
    in log-space this becomes:
        log(output) = log(constant) + e0*log(x0) + e1*log(x1) + ...
    """

    def __init__(self, constant: float, exponents: dict, output_name: str = "y"):
        """
        Args:
            constant:    multiplicative constant (in original units)
            exponents:   dict mapping variable name -> exponent value
                         e.g. {"v": 2.97} or {"l": 0.50, "g": -0.50}
            output_name: label for the left-hand side of the equation string
                         e.g. "P" for power, "T" for period, "L" for luminosity
        """
        self.constant = float(constant)
        self.exponents = {k: float(v) for k, v in exponents.items()}
        self.output_name = output_name   # FIX M7

    def __str__(self):
        terms = " * ".join(
            f"{var}^{exp:.3f}" for var, exp in self.exponents.items()
        )
        return f"{self.output_name} = {self.constant:.4f} * {terms}"

    def __repr__(self):
        return f"PowerLawEquation({self.__str__()})"


class PhysicsAuditor:
    """
    Audits a PowerLawEquation against known physics targets.

    FIX C1: 'expected' is ALWAYS passed in — never hardcoded in this class.
    Every dataset must explicitly provide its own physics targets:

        # Wind turbine
        auditor = PhysicsAuditor(expected={"v": 3.0}, tol=0.1)

        # Pendulum
        auditor = PhysicsAuditor(expected={"l": 0.5}, tol=0.1)

        # Kepler
        auditor = PhysicsAuditor(expected={"a": 1.5}, tol=0.1)

        # Thermodynamics (gas phase)
        auditor = PhysicsAuditor(expected={"Density_mol_l": 1.0}, tol=0.1)

        # Stellar (main sequence only)
        auditor = PhysicsAuditor(expected={"Mass_Msun": 3.9}, tol=0.15)
    """

    def __init__(self, expected: dict, tol: float = 0.1):
        """
        Args:
            expected: dict {variable_name: target_exponent}
                      Must be non-empty.
            tol:      absolute tolerance for exponent acceptance.
                      |discovered - target| <= tol  →  PASS
        """
        if not expected:
            raise ValueError(
                "PhysicsAuditor requires a non-empty 'expected' dict. "
                "Example: PhysicsAuditor(expected={'v': 3.0})"
            )
        self.expected = expected
        self.tol = tol

    def audit(self, equation) -> tuple:
        """
        Audit a PowerLawEquation against the expected physics.

        Args:
            equation: PowerLawEquation instance

        Returns:
            (passed: bool, feedback: dict)

            feedback dict has one entry per variable that FAILED:
                {var: {"target": float, "current": float, "weight": 1.0}}
            If all variables pass, feedback is empty and passed is True.
        """
        if not hasattr(equation, "exponents"):
            # Non-symbolic model output — cannot audit
            return False, {"error": "Equation has no 'exponents' attribute"}

        feedback = {}
        all_passed = True

        for var, target in self.expected.items():
            if var not in equation.exponents:
                # Variable not discovered — soft fail with neutral feedback
                feedback[var] = {
                    "target": target,
                    "current": 0.0,
                    "weight": 1.0,
                }
                all_passed = False
                continue

            current = float(equation.exponents[var])
            error = abs(current - target)

            if error > self.tol:
                all_passed = False
                feedback[var] = {
                    "target": target,
                    "current": current,
                    "weight": 1.0,
                }

        return all_passed, feedback
