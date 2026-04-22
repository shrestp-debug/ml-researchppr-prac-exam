"""
kan/true_kan.py
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from auditor.physics_auditor import PowerLawEquation


class KANLayer(nn.Module):
    def __init__(
        self,
        in_features: int = 1,
        out_features: int = 1,
        grid_size: int = 5,
        spline_order: int = 3,
        scale_base: float = 1.0,
        scale_spline: float = 1.0,
        grid_range: tuple = (-2.0, 6.0),
    ):
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features
        self.grid_size    = grid_size
        self.spline_order = spline_order
        self._build_grid(grid_range[0], grid_range[1])
        self.base_weight = nn.Parameter(torch.empty(out_features, in_features))
        self.spline_weight = nn.Parameter(
            torch.empty(out_features, in_features * (grid_size + spline_order))
        )
        self.scale_base   = scale_base
        self.scale_spline = scale_spline
        self._reset_parameters()

    def _build_grid(self, lo: float, hi: float):
        h = (hi - lo) / self.grid_size
        grid = (
            torch.arange(
                -self.spline_order,
                self.grid_size + self.spline_order + 1,
                dtype=torch.float32
            ) * h + lo
        ).expand(self.in_features, -1).contiguous()
        if hasattr(self, "grid"):
            self.grid = grid
        else:
            self.register_buffer("grid", grid)

    def update_grid(self, lo: float, hi: float):
        self._build_grid(lo, hi)

    def _reset_parameters(self):
        nn.init.kaiming_uniform_(self.base_weight, a=np.sqrt(5) * self.scale_base)
        nn.init.xavier_normal_(self.spline_weight, gain=0.1 * self.scale_spline)

    def b_splines(self, x: torch.Tensor) -> torch.Tensor:
        assert x.dim() == 2 and x.size(1) == self.in_features
        grid = self.grid
        x = x.unsqueeze(-1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).float()
        for k in range(1, self.spline_order + 1):
            left  = grid[:, : -(k + 1)]
            right = grid[:, k + 1:]
            denom_left  = grid[:, k:-1] - left  + 1e-8
            denom_right = right - grid[:, 1:-k] + 1e-8
            bases = (
                (x - left) / denom_left * bases[:, :, :-1]
                + (right - x) / denom_right * bases[:, :, 1:]
            )
        return bases.contiguous()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out     = F.linear(x, self.base_weight)
        spline_basis = self.b_splines(x).view(x.size(0), -1)
        spline_out   = F.linear(spline_basis, self.spline_weight)
        return base_out + spline_out


class TrueKAN(nn.Module):
    is_symbolic = True

    def __init__(
        self,
        input_dim:    int   = 1,
        epochs:       int   = 2000,
        lr:           float = 0.01,
        grid_size:    int   = 5,
        spline_order: int   = 3,
        grid_margin:  float = 0.5,
    ):
        super().__init__()
        self.input_dim    = input_dim
        self.epochs       = epochs
        self.lr           = lr
        self.grid_size    = grid_size
        self.spline_order = spline_order
        self.grid_margin  = grid_margin
        self.functions = nn.ModuleList([
            KANLayer(in_features=1, out_features=1,
                     grid_size=grid_size, spline_order=spline_order)
            for _ in range(input_dim)
        ])
        self.bias = nn.Parameter(torch.zeros(1))
        self._data_ranges = None
        self.raw_vars = None
        self.trained  = False

    def _update_grids(self, X_t: torch.Tensor):
        self._data_ranges = []
        margin = self.grid_margin
        for i, layer in enumerate(self.functions):
            col = X_t[:, i]
            lo  = float(col.min().item()) - margin
            hi  = float(col.max().item()) + margin
            layer.update_grid(lo, hi)
            self._data_ranges.append((lo, hi))

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        out = torch.zeros(X.shape[0], 1, device=X.device, dtype=X.dtype)
        for i, layer in enumerate(self.functions):
            out = out + layer(X[:, i : i + 1])
        return out + self.bias

    def fit(self, X, y, feedback: dict = None, raw_vars: list = None, lamb_l1: float = 0.001, epochs: int = None):
        if raw_vars is not None:
            self.raw_vars = raw_vars
        if epochs is not None:
            self.epochs = epochs

        X_t = torch.tensor(X, dtype=torch.float32) if not torch.is_tensor(X) else X.float()
        y_t = (torch.tensor(y, dtype=torch.float32) if not torch.is_tensor(y) else y.float()).view(-1, 1)

        # FIX: only update grids on first fit — subsequent calls are warm-starts.
        # Rebuilding the grid shifts knot positions under already-learned weights,
        # corrupting the warm-start. Data range is fixed after the first call.
        if not self.trained:
            self._update_grids(X_t)

        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        loss_fn   = torch.nn.MSELoss()

        for epoch in range(self.epochs):
            optimizer.zero_grad()
            pred      = self.forward(X_t)
            data_loss = loss_fn(pred, y_t)
            reg_loss  = sum(layer.spline_weight.abs().mean() for layer in self.functions)

            phys_loss = torch.tensor(0.0)
            if feedback is not None:
                # Agent can override lambda directly via feedback dict
                if "_lambda_override" in feedback:
                    lambda_phys = float(feedback["_lambda_override"])
                else:
                    iteration   = feedback.get("_iteration", 0)
                    max_iters   = feedback.get("_max_iters", 5)
                    lambda_phys = 5.0 + (iteration / max(max_iters - 1, 1)) * 20.0

                var_names = [k for k in feedback if not k.startswith("_")]
                for var in var_names:
                    # FIX CRITICAL: use raw_vars to get the correct column index.
                    # Do NOT use enumerate(var_names) — feedback key order is not
                    # guaranteed to match the column order of X.
                    if not (self.raw_vars and var in self.raw_vars):
                        continue
                    i = self.raw_vars.index(var)
                    if i >= self.input_dim:
                        continue

                    target  = feedback[var]["target"]
                    lo, hi  = self._data_ranges[i] if self._data_ranges else (-2.0, 6.0)
                    x_probe = torch.linspace(lo + 0.1, hi - 0.1, 100).view(-1, 1)
                    x_probe.requires_grad_(True)
                    y_probe = self.functions[i](x_probe)
                    grads   = torch.autograd.grad(
                        y_probe, x_probe,
                        grad_outputs=torch.ones_like(y_probe),
                        create_graph=True,
                    )[0]
                    phys_loss = phys_loss + (grads.mean() - target) ** 2

                total_loss = data_loss + lamb_l1 * reg_loss + lambda_phys * phys_loss
            else:
                total_loss = data_loss + lamb_l1 * reg_loss

            total_loss.backward()
            optimizer.step()

            log_interval = max(1, self.epochs // 4)
            if (epoch + 1) % log_interval == 0 or (epoch + 1) == self.epochs:
                if feedback is not None:
                    print(f"    Epoch {epoch+1}/{self.epochs}: data={data_loss.item():.4f}  "
                          f"phys={phys_loss.item():.4f}  reg={reg_loss.item():.4f}")
                else:
                    print(f"    Epoch {epoch+1}/{self.epochs}: loss={data_loss.item():.4f}")

        self.trained = True

    def discover_equation(self, raw_vars: list = None, output_name: str = "y") -> PowerLawEquation:
        if not self.trained:
            raise RuntimeError("KAN must be fitted before calling discover_equation()")
        if raw_vars is None:
            raw_vars = self.raw_vars or [f"x{i}" for i in range(self.input_dim)]

        exponents = {}
        for i, var in enumerate(raw_vars):
            if i >= self.input_dim:
                break
            lo, hi  = self._data_ranges[i] if self._data_ranges else (-2.0, 6.0)
            x_probe = torch.linspace(lo + 0.05, hi - 0.05, 500).view(-1, 1)
            x_probe.requires_grad_(True)
            y_probe = self.functions[i](x_probe)
            grads   = torch.autograd.grad(y_probe, x_probe, grad_outputs=torch.ones_like(y_probe))[0]
            exponents[var] = float(grads.mean().item())

        constant = float(torch.exp(self.bias).item())
        return PowerLawEquation(constant, exponents, output_name=output_name)

    def predict(self, X) -> np.ndarray:
        self.eval()
        X_t = torch.tensor(X, dtype=torch.float32)
        with torch.no_grad():
            return self.forward(X_t).numpy().flatten()
