"""
utils/data_utils.py
"""

import numpy as np
import pandas as pd
import os


def train_test_split(X: np.ndarray, y: np.ndarray, test_frac: float = 0.2, seed: int = 42):
    rng    = np.random.RandomState(seed)
    n      = len(X)
    idx    = rng.permutation(n)
    n_test = max(1, int(n * test_frac))
    y = y.flatten()
    return X[idx[n_test:]], y[idx[n_test:]], X[idx[:n_test]], y[idx[:n_test]]


def load_wind(filepath: str = "data/dataset_A_frozen.csv", n_samples: int = None, seed: int = 42):
    """P ∝ v³ — filter to valid turbine operating range (cut-in speed)."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Wind data not found: {filepath}")
    df    = pd.read_csv(filepath)
    v     = df["wind_speed"].values.astype(float)
    power = df["LV ActivePower (kW)"].values.astype(float)
    valid = (v >= 3.5) & (power >= 10.0)
    v, power = v[valid], power[valid]
    if n_samples and n_samples < len(v):
        idx = np.random.RandomState(seed).choice(len(v), n_samples, replace=False)
        v, power = v[idx], power[idx]
    print(f"[Wind]     N={len(v):4d}  v:[{v.min():.1f},{v.max():.1f}] m/s  (cut-in filter applied)")
    return np.log(v).reshape(-1, 1), np.log(power).flatten(), ["v"]


def load_pendulum(filepath: str = "data/Simple pendulum data.csv", n_samples: int = None, seed: int = 42):
    """T ∝ l^0.5"""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Pendulum data not found: {filepath}")
    df    = pd.read_csv(filepath)
    l, t  = df["length(l)"].values.astype(float), df["time(t)"].values.astype(float)
    valid = (l > 0) & (t > 0)
    l, t  = l[valid], t[valid]
    if n_samples and n_samples < len(l):
        idx = np.random.RandomState(seed).choice(len(l), n_samples, replace=False)
        l, t = l[idx], t[idx]
    print(f"[Pendulum] N={len(l):4d}  l:[{l.min():.3f},{l.max():.2f}] m")
    return np.log(l).reshape(-1, 1), np.log(t).flatten(), ["l"]


def load_kepler(filepath: str = "data/kepler_data_clean.csv", n_samples: int = 500, seed: int = 42):
    """T ∝ a^1.5 — KAN grid set dynamically in TrueKAN.fit() to cover full log(a) range."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Kepler data not found: {filepath}")
    df    = pd.read_csv(filepath)
    a, T  = df["pl_orbsmax"].values.astype(float), df["pl_orbper"].values.astype(float)
    valid = (a > 0) & (T > 0)
    a, T  = a[valid], T[valid]
    if n_samples and n_samples < len(a):
        idx = np.random.RandomState(seed).choice(len(a), n_samples, replace=False)
        a, T = a[idx], T[idx]
    X_log = np.log(a).reshape(-1, 1)
    print(f"[Kepler]   N={len(a):4d}  a:[{a.min():.4f},{a.max():.1f}] AU  log(a):[{X_log.min():.2f},{X_log.max():.2f}]")
    return X_log, np.log(T).flatten(), ["a"]


def load_argon(filepath: str = "data/argon_150k.csv", n_samples: int = 500,
               seed: int = 42, gas_phase_cutoff: float = 5.0):
    """P ∝ rho^1.0 — gas phase only. Liquid phase violates single power-law assumption."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Argon data not found: {filepath}")
    df       = pd.read_csv(filepath)
    rho, P   = df["Density_mol_l"].values.astype(float), df["Pressure_MPa"].values.astype(float)
    valid    = (rho > 0) & (P > 0) & (rho < gas_phase_cutoff)
    rho, P   = rho[valid], P[valid]
    if n_samples and n_samples < len(rho):
        idx = np.random.RandomState(seed).choice(len(rho), n_samples, replace=False)
        rho, P = rho[idx], P[idx]
    print(f"[Argon]    N={len(rho):4d}  rho:[{rho.min():.4f},{rho.max():.2f}] mol/L  (gas phase only)")
    return np.log(rho).reshape(-1, 1), np.log(P).flatten(), ["Density_mol_l"]


def load_stars(filepath: str = "data/mass_luminosity.csv", min_mass: float = 0.5,
               n_samples: int = None, seed: int = 42):
    """L ∝ M^3.9 — main sequence only (M > 0.5). Red dwarfs follow L∝M^2.3, different regime."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Stellar data not found: {filepath}")
    df    = pd.read_csv(filepath)
    M, L  = df["Mass_Msun"].values.astype(float), df["Luminosity_Lsun"].values.astype(float)
    valid = (M > min_mass) & (L > 0)
    M, L  = M[valid], L[valid]
    sort_idx = np.argsort(M)
    M, L  = M[sort_idx], L[sort_idx]
    if n_samples and n_samples < len(M):
        idx = np.sort(np.random.RandomState(seed).choice(len(M), n_samples, replace=False))
        M, L = M[idx], L[idx]
    print(f"[Stars]    N={len(M):4d}  M:[{M.min():.2f},{M.max():.1f}] M_sun  (main sequence only)")
    return np.log(M).reshape(-1, 1), np.log(L).flatten(), ["Mass_Msun"]


# epochs_kan=1000, epochs_mlp=500 for all domains.
# KAN needs more epochs (symbolic spline fitting); MLP converges faster.
DOMAIN_REGISTRY = {
    "wind":     {"loader": load_wind,     "physics": {"v": 3.0},             "output_name": "P", "tol": 0.10, "lambda_start": 10.0, "lambda_end": 50.0, "epochs_kan": 1000, "epochs_mlp": 500},
    "pendulum": {"loader": load_pendulum, "physics": {"l": 0.5},             "output_name": "T", "tol": 0.10, "lambda_start":  5.0, "lambda_end": 20.0, "epochs_kan": 1000, "epochs_mlp": 500},
    "kepler":   {"loader": load_kepler,   "physics": {"a": 1.5},             "output_name": "T", "tol": 0.10, "lambda_start": 10.0, "lambda_end": 40.0, "epochs_kan": 1000, "epochs_mlp": 500},
    "argon":    {"loader": load_argon,    "physics": {"Density_mol_l": 1.0}, "output_name": "P", "tol": 0.10, "lambda_start":  5.0, "lambda_end": 20.0, "epochs_kan": 1000, "epochs_mlp": 500},
    "stars":    {"loader": load_stars,    "physics": {"Mass_Msun": 3.9},     "output_name": "L", "tol": 0.15, "lambda_start":  5.0, "lambda_end": 25.0, "epochs_kan": 1000, "epochs_mlp": 500},
}
