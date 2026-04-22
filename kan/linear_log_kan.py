"""
kan/linear_log_kan.py

LinearLogKAN — simplified KAN variant for ablation studies.

FIX minor — original hardcoded variable names ["v", "rho", "R"] making
this class broken for any non-wind dataset (pendulum, Kepler, etc.).
discover_equation() now accepts raw_vars from the caller.

FIX physics index — physics loss previously used enumerate(var_names) index
which reflects feedback dict iteration order, NOT the column position of the
variable in X. For single-variable datasets this is harmless, but for
multi-variable datasets it silently applies the wrong penalty to the wrong
weight. Fixed to match TrueKAN: use self.raw_vars.index(var) for column index.
"""

import numpy as np
import torch
import torch.nn as nn

from auditor.physics_auditor import PowerLawEquation


class LinearLogKAN(nn.Module):
    """
    Simplified KAN: linear function in log-space.
        log(y) = sum_i  w_i * log(x_i)  +  bias
    This is equivalent to fitting a power law directly.
    Useful as an interpretable ablation baseline.
    """

    is_symbolic = True

    def __init__(self, input_dim: int = 1, epochs: int = 2000, lr: float = 0.01):
        super().__init__()
        self.input_dim = input_dim
        self.epochs    = epochs
        self.lr        = lr

        self.weights = nn.Parameter(torch.ones(input_dim))   # initial exponent guess = 1
        self.bias    = nn.Parameter(torch.zeros(1))

        self.raw_vars = None
        self.trained  = False

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        # X shape: (N, D) — already in log-space
        return (X * self.weights).sum(dim=1, keepdim=True) + self.bias

    def fit(self, X, y, feedback: dict = None, raw_vars: list = None, **kwargs):
        if raw_vars is not None:
            self.raw_vars = raw_vars

        X_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.float32).view(-1, 1)

        opt = torch.optim.Adam(self.parameters(), lr=self.lr)

        for epoch in range(self.epochs):
            opt.zero_grad()
            pred = self.forward(X_t)
            loss = nn.functional.mse_loss(pred, y_t)

            if feedback is not None:
                iteration   = feedback.get("_iteration", 0)
                max_iters   = feedback.get("_max_iters", 5)
                lambda_phys = 5.0 + (iteration / max(max_iters - 1, 1)) * 20.0

                var_names = [k for k in feedback if not k.startswith("_")]
                phys_loss = torch.tensor(0.0)
                for var in var_names:
                    # FIX: use raw_vars to get the correct column index.
                    # Do NOT use enumerate(var_names) — feedback key order is not
                    # guaranteed to match the column order of X.
                    # This matches the same fix already applied in TrueKAN.
                    if not (self.raw_vars and var in self.raw_vars):
                        continue
                    i = self.raw_vars.index(var)
                    if i >= self.input_dim:
                        continue
                    target    = feedback[var]["target"]
                    phys_loss = phys_loss + (self.weights[i] - target) ** 2
                loss = loss + lambda_phys * phys_loss

            loss.backward()
            opt.step()

        self.trained = True

    def discover_equation(self, raw_vars: list = None, output_name: str = "y") -> PowerLawEquation:
        """
        raw_vars must be passed in — not hardcoded.
        """
        if not self.trained:
            raise RuntimeError("Must call fit() before discover_equation()")

        if raw_vars is None:
            raw_vars = self.raw_vars
        if raw_vars is None:
            raw_vars = [f"x{i}" for i in range(self.input_dim)]

        exponents = {
            var: float(self.weights[i].item())
            for i, var in enumerate(raw_vars)
            if i < self.input_dim
        }
        constant = float(torch.exp(self.bias).item())
        return PowerLawEquation(constant, exponents, output_name=output_name)

    def predict(self, X) -> np.ndarray:
        self.eval()
        X_t = torch.tensor(X, dtype=torch.float32)
        with torch.no_grad():
            return self.forward(X_t).numpy().flatten()
