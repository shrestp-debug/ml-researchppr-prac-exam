"""
models/mlp_learner.py
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from auditor.physics_auditor import PowerLawEquation


class _MLP(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64), nn.Tanh(),
            nn.Linear(64,        64), nn.Tanh(),
            nn.Linear(64,        32), nn.Tanh(),
            nn.Linear(32,         1),
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=0.8)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)


class MLPLearner:
    is_symbolic = False

    def __init__(
        self,
        input_dim:         int   = 1,
        lr:                float = 1e-3,
        epochs:            int   = 2000,
        lambda_phys_start: float = 10.0,
        lambda_phys_end:   float = 50.0,
    ):
        self.model   = _MLP(input_dim)
        self.opt     = optim.Adam(self.model.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
        self.epochs  = epochs
        self.lambda_phys_start = lambda_phys_start
        self.lambda_phys_end   = lambda_phys_end
        self.input_dim = input_dim
        self._X_train: torch.Tensor = None
        self.raw_vars: list         = None

    def _estimate_exponents(
        self, raw_vars: list, X_eval: torch.Tensor, create_graph: bool = False
    ) -> dict:
        self.model.eval()
        X_eval = X_eval.detach().requires_grad_(True)
        pred   = self.model(X_eval)

        exponents    = {}
        grad_outputs = torch.ones_like(pred)
        var_list     = list(enumerate(raw_vars))

        for idx, (i, var) in enumerate(var_list):
            if i >= self.input_dim:
                break
            retain = (idx < len(var_list) - 1) or create_graph
            grads  = torch.autograd.grad(
                outputs=pred, inputs=X_eval,
                grad_outputs=grad_outputs,
                retain_graph=retain,
                create_graph=create_graph,
            )[0]
            exponents[var] = grads[:, i].mean()

        return exponents

    def fit(self, X, y, feedback: dict = None, raw_vars: list = None, epochs: int = None):
        if epochs is None:
            epochs = self.epochs
        if raw_vars is not None:
            self.raw_vars = raw_vars
        if self.raw_vars is None:
            self.raw_vars = [f"x{i}" for i in range(self.input_dim)]

        X_t = torch.from_numpy(np.asarray(X, dtype=np.float32))
        y_t = torch.from_numpy(np.asarray(y, dtype=np.float32).reshape(-1, 1))
        self._X_train = X_t

        for epoch in range(epochs):
            self.model.train()
            self.opt.zero_grad()

            pred      = self.model(X_t)
            data_loss = self.loss_fn(pred, y_t)
            total_loss    = data_loss
            phys_loss_val = 0.0

            if feedback is not None:
                iteration   = feedback.get("_iteration", 0)
                max_iters   = feedback.get("_max_iters", 5)
                t           = iteration / max(max_iters - 1, 1)
                lambda_phys = self.lambda_phys_start + t * (
                    self.lambda_phys_end - self.lambda_phys_start
                )

                var_names    = [k for k in feedback if not k.startswith("_")]

                # Subsample for physics gradient estimation:
                # create_graph=True (second-order grads) is very expensive.
                # Using a 512-point subset gives the same gradient signal
                # but avoids O(N) cost on large datasets (e.g. 38K wind).
                _PHYS_SUBSAMPLE = 512
                if X_t.shape[0] > _PHYS_SUBSAMPLE:
                    g = torch.Generator().manual_seed(epoch)
                    idx = torch.randperm(X_t.shape[0], generator=g)[:_PHYS_SUBSAMPLE]
                    X_phys = X_t[idx]
                else:
                    X_phys = X_t
                current_exps = self._estimate_exponents(var_names, X_phys, create_graph=True)

                # Restore train mode — _estimate_exponents() sets model.eval()
                self.model.train()

                phys_loss = torch.zeros(1, device=X_t.device)
                for var in var_names:
                    if var in feedback and var in current_exps:
                        target    = feedback[var]["target"]
                        current   = current_exps[var]
                        phys_loss = phys_loss + (current - target) ** 2

                phys_loss_val = phys_loss.item()
                total_loss    = data_loss + lambda_phys * phys_loss

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.opt.step()

            log_interval = max(1, epochs // 4)
            if (epoch + 1) % log_interval == 0 or (epoch + 1) == epochs:
                if feedback is not None:
                    iteration   = feedback.get("_iteration", 0)
                    max_iters   = feedback.get("_max_iters", 5)
                    t  = iteration / max(max_iters - 1, 1)
                    lp = self.lambda_phys_start + t * (
                        self.lambda_phys_end - self.lambda_phys_start
                    )
                    print(f"    Epoch {epoch+1}/{epochs}: data={data_loss.item():.4f}  "
                          f"phys={phys_loss_val:.4f}  lambda={lp:.1f}  "
                          f"total={total_loss.item():.4f}")
                else:
                    print(f"    Epoch {epoch+1}/{epochs}: loss={data_loss.item():.4f}")

    def discover_equation(self, raw_vars: list = None, output_name: str = "y") -> PowerLawEquation:
        if self._X_train is None:
            raise RuntimeError("Call fit() before discover_equation()")
        if raw_vars is None:
            raw_vars = self.raw_vars or [f"x{i}" for i in range(self.input_dim)]

        exps_tensor = self._estimate_exponents(raw_vars, self._X_train, create_graph=False)
        exponents   = {
            var: float(val.item() if torch.is_tensor(val) else val)
            for var, val in exps_tensor.items()
        }

        # CALCULATE THE EXACT CONSTANT
        with torch.no_grad():
            self.model.eval()
            log_pred   = self.model(self._X_train).flatten()
            log_linear = torch.zeros(len(self._X_train), device=self._X_train.device)
            for i, var in enumerate(raw_vars):
                if var in exponents and i < self._X_train.shape[1]:
                    log_linear = log_linear + exponents[var] * self._X_train[:, i]
            constant = float(torch.exp((log_pred - log_linear).mean()).item())

        return PowerLawEquation(constant, exponents, output_name=output_name)

    def predict(self, X) -> np.ndarray:
        self.model.eval()
        X_t = torch.tensor(np.asarray(X, dtype=np.float32))
        with torch.no_grad():
            return self.model(X_t).numpy().flatten()
