"""
ORBIT — Order-aware Regression with Basis, Interactions & Trends.

Implements the core estimator (`ORBITRegressor`) and a simple
bagging wrapper (`ORBITBaggedRegressor`) along with utilities for
feature-bank construction, screening + whitening, estimation, and
lightweight evaluation helpers.

Notes
-----
Design choices and algorithmic background are summarized in the project
README (feature bank, two-stage screening with whitening, ElasticNet +
bank dropout, optional Huber-ridge refit, ridge stacker, and split
conformal PIs).
"""


import sys, time, warnings, math
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.feature_selection import mutual_info_regression
from sklearn.linear_model import Ridge, ElasticNetCV, HuberRegressor
from sklearn.datasets import load_diabetes, fetch_california_housing, make_friedman1, make_regression

from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor,
    AdaBoostRegressor, HistGradientBoostingRegressor
)
from sklearn.neural_network import MLPRegressor
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.linear_model import ElasticNetCV, Ridge

from dataclasses import dataclass
from typing import Optional, List, Dict, Any
from sklearn.inspection import permutation_importance


# Optional wavelets
try:
    import pywt
    HAS_PYWT = True
except Exception:
    HAS_PYWT = False

# ---------- utils ----------
def rmse(y_true, y_pred):
    """
    Root mean squared error.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground-truth target.
    y_pred : array-like of shape (n_samples,)
        Predicted target.

    Returns
    -------
    float
        RMSE over finite entries only.
    """
    yt = np.asarray(y_true).ravel(); yp = np.asarray(y_pred).ravel()
    mask = np.isfinite(yt) & np.isfinite(yp)
    if mask.sum() == 0: return np.inf
    yt = yt[mask]; yp = yp[mask]
    return float(np.sqrt(np.mean((yt - yp) ** 2)))

def nrmse(y_true, y_pred):
    """
    Normalized RMSE (divided by std of `y_true`).

    Parameters
    ----------
    y_true, y_pred : array-like
        True and predicted targets.

    Returns
    -------
    float
        RMSE / std(y_true[finite]).
    """
    yt = np.asarray(y_true).ravel()
    s = float(np.std(yt[np.isfinite(yt)]) + 1e-12)
    v = rmse(y_true, y_pred) / s
    return float(v)

def safe_nan_to_num(A):
    """
    Replace non-finite values with zeros (float).

    Parameters
    ----------
    A : array-like
        Input array.

    Returns
    -------
    ndarray
        Same shape as `A`, cast to float with NaN/±inf → 0.0.
    """
    A = np.asarray(A, float)
    A = np.where(np.isfinite(A), A, 0.0)
    return A

def temporal_split(X, y, test_size=0.25, min_train=50):
    """
    Chronological train/validation split.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Features in time order (row t precedes t+1).
    y : array-like of shape (n_samples,)
        Target in time order.
    test_size : float, default=0.25
        Fraction of the tail used for validation.
    min_train : int, default=50
        Minimum number of training rows to keep.

    Returns
    -------
    (X_tr, X_val, y_tr, y_val) : tuple of ndarrays
        Contiguous head = train, tail = validation.
    """
    n = len(y)
    n_test = max(1, int(round(test_size * n)))
    n_train = max(min_train, n - n_test)
    n_train = min(n_train, n - 1)
    return X[:n_train], X[n_train:], y[:n_train], y[n_train:]

def safe_metrics(ytrue, ypred, PI=None):
    """
    Convenience metrics (MAE, RMSE, NRMSE, R²) computed safely.

    Returns
    -------
    dict
        Keys: 'mae', 'rmse', 'nrmse', 'r2'.
    """
    y = np.asarray(ytrue).ravel()
    yhat = np.asarray(ypred).ravel()
    if PI is not None:
        PI = np.asarray(PI, float)
    mask = np.isfinite(y) & np.isfinite(yhat)
    if PI is not None:
        mask = mask & np.isfinite(PI).all(axis=1)
    if mask.sum() == 0:
        raise ValueError("No finite rows remain after masking.")
    y = y[mask]; yhat = yhat[mask]
    if PI is not None: PI = PI[mask]
    rm = rmse(y, yhat)
    ma = mean_absolute_error(y, yhat)
    try:
        r2 = r2_score(y, yhat)
    except Exception:
        r2 = np.nan
    cov = np.nan
    if PI is not None:
        lo, hi = PI[:,0], PI[:,1]
        cov = float(np.mean((y >= lo) & (y <= hi)))
    return rm, ma, r2, cov, y, yhat, PI

# ---------- wavelet denoise ----------
def wavelet_denoise_1d(x, wavelet="db2", level=None, thr_mult=0.8):
    """
    Soft-threshold wavelet denoising (db2).

    Parameters
    ----------
    x : array-like of shape (n_samples,)
        1D series.

    Returns
    -------
    ndarray
        Denoised vector. If `pywt` unavailable or n<8, returns `x`.
    Notes
    -----
    Used as an optional prefilter before univariate basis functions. 
    """
    x = np.asarray(x, float)
    if not HAS_PYWT or x.size < 8: return x
    try:
        w = pywt.Wavelet(wavelet)
        max_level = pywt.dwt_max_level(len(x), w.dec_len)
        level = min(level or max_level, max_level)
        if level <= 0: return x
        coeffs = pywt.wavedec(x, wavelet, level=level)
        sigma = np.median(np.abs(coeffs[-1])) / 0.6745 + 1e-12
        thr = thr_mult * sigma * np.sqrt(2 * np.log(len(x)))
        coeffs_d = [coeffs[0]] + [pywt.threshold(c, thr, mode='soft') for c in coeffs[1:]]
        x_rec = pywt.waverec(coeffs_d, wavelet)
        if x_rec.size != x.size: x_rec = x_rec[:x.size]
        return safe_nan_to_num(x_rec)
    except Exception:
        return x

# ---------- feature banks ----------
def gaussian_rbf_1d(x, centers, width):
    """
    Gaussian radial basis bank on a single standardized feature.

    Parameters
    ----------
    x : ndarray of shape (n_samples,)
    centers : ndarray of shape (n_centers,)
        Quantile-based centers.
    width : float
        Shared bandwidth.

    Returns
    -------
    ndarray of shape (n_samples, n_centers)
        Exp(- (x - c_k)^2 / (2*width^2)).
    """
    x = x[:, None]
    return np.exp(-((x - centers[None, :]) ** 2) / (2.0 * (width ** 2) + 1e-12))

def hinge_bank_1d(x, knots):
    """
    Two-sided hinge spline features on a single feature.

    Returns
    -------
    ndarray, shape (n_samples, 2 * n_knots)
        [max(0, x - κ_m), max(0, κ_m - x)] for each knot.
    """
    x = x[:, None]; K = knots[None, :]
    return np.hstack([np.maximum(0.0, x - K), np.maximum(0.0, K - x)])

def make_univariate_bank(X, active_dims, n_knots=6, use_wavelet=True, include_splines=True):
    """
    Build per-dimension basis functions (RBFs, sin/cos, hinge splines).

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Standardized features.
    active_dims : sequence of int
        Indices of features to expand.
    n_knots : int, default=6
        Number of spline knots per feature.
    use_wavelet : bool, default=True
        Whether to pre-denoise each column.
    include_splines : bool, default=True
        Include hinge-spline atoms.

    Returns
    -------
    (Phi, names) : (ndarray, list[str])
        Feature bank block and corresponding names. 
    """
    n, p = X.shape; feats = []
    for j in active_dims:
        xj = X[:, j]
        xj_d = wavelet_denoise_1d(xj) if use_wavelet else xj
        if np.std(xj_d) < 1e-12:
            continue
        qs = np.linspace(0.05, 0.95, n_knots)
        centers = np.quantile(xj_d, qs)
        if len(centers) < 2: continue
        width = (np.quantile(xj_d, 0.9) - np.quantile(xj_d, 0.1)) / (2.0 * n_knots) + 1e-12
        rbf = gaussian_rbf_1d(xj_d, centers, width)
        base = xj_d / (np.std(xj_d) + 1e-12)
        trig = np.column_stack([np.sin(base * f) for f in (0.5, 1.0, 2.0, 3.0)] +
                               [np.cos(base * f) for f in (0.5, 1.0, 2.0, 3.0)])
        feats.append(rbf); feats.append(trig)
        if include_splines:
            knots = np.quantile(xj_d, np.linspace(0.1, 0.9, max(4, n_knots - 2)))
            spl = hinge_bank_1d(xj_d, knots)
            feats.append(spl)
    return np.hstack(feats) if feats else np.zeros((n, 0))

def stability_select_pairs(X, y, top_k=48, rounds=6, subsample=0.7, rng=0):
    """
    Stability selection for pairwise interaction candidates.

    Generates multiple subsampled rankings by absolute correlation
    with `y` and keeps the most frequently selected pairs.

    Parameters
    ----------
    X : ndarray, shape (n_samples, n_features)
        Standardized features.
    y : ndarray, shape (n_samples,)
        Target (centered).
    top_k : int, default=48
        Number of pairs to retain.
    rounds : int, default=6
        Subsampled ranking repeats.
    subsample : float, default=0.7
        Row/feature subsample fraction per round.
    rng : int, default=0
        Random seed.

    Returns
    -------
    list[tuple[int, int]]
        Chosen (i, j) feature index pairs. 
    """
    rng = np.random.RandomState(rng); n, p = X.shape; counts = {}
    for _ in range(rounds):
        m = max(16, int(subsample * n))
        if m <= 2: break
        idx = rng.choice(n, size=m, replace=False)
        Xi, yi = X[idx], y[idx]
        D = min(p, max(8, int(0.6 * p)))
        dims = np.sort(rng.choice(p, size=D, replace=False))
        yy = yi - yi.mean()
        denom_yy = float(np.sqrt(np.dot(yy, yy)) + 1e-12)
        cand = []
        for a in range(len(dims)):
            i = dims[a]; xi = Xi[:, i]
            for b in range(a+1, len(dims)):
                j = dims[b]; xj = Xi[:, j]
                z = xi * xj; zc = z - z.mean()
                num = float(np.dot(zc, yy))
                den = float(np.sqrt(np.dot(zc, zc)) * denom_yy + 1e-12)
                cand.append((abs(num/den), i, j))
        cand.sort(reverse=True)
        for _, i, j in cand[:max(8, top_k // 4)]:
            counts[(i, j)] = counts.get((i, j), 0) + 1
    items = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0][0], kv[0][1]))
    return [(i, j) for (i, j), _ in items[:top_k]]

def make_pair_bank(X, pairs):
    """
    Construct pairwise interaction features for given pairs.

    Returns
    -------
    (Phi, names) : (ndarray, list[str])
        Includes products, absolute differences, and simple trig couplings.
    """
    if not pairs: return np.zeros((X.shape[0], 0))
    cols = []
    for (i, j) in pairs:
        xi = X[:, i]; xj = X[:, j]
        cols.append(xi * xj)
        cols.append(np.abs(xi - xj))
        for f in (0.5, 1.0, 2.0):
            cols.append(np.sin(f*(xi + xj)))
            cols.append(np.cos(f*(xi - xj)))
    return np.column_stack(cols) if cols else np.zeros((X.shape[0], 0))

def select_top_triples(X, y, pairs, max_triples=6):
    """
    Heuristic selection of triple interactions from indices seen in `pairs`.

    Parameters
    ----------
    max_triples : int, default=6
        Cap on triple terms to keep.

    Returns
    -------
    list[tuple[int, int, int]]
        Selected triples by correlation with `y`.
    """
    if max_triples <= 0 or not pairs: return []
    uniq = sorted(set([i for (i, j) in pairs] + [j for (i, j) in pairs]))
    cand = []
    yy = y - y.mean()
    denom_yy = float(np.sqrt(np.dot(yy, yy)) + 1e-12)
    for a in range(len(uniq)):
        for b in range(a+1, len(uniq)):
            for c in range(b+1, len(uniq)):
                i, j, k = uniq[a], uniq[b], uniq[c]
                z = X[:, i] * X[:, j] * X[:, k]; zc = z - z.mean()
                num = float(np.dot(zc, yy))
                den = float(np.sqrt(np.dot(zc, zc)) * denom_yy + 1e-12)
                cand.append((abs(num/den), (i, j, k)))
                if len(cand) > 1500: break
            if len(cand) > 1500: break
        if len(cand) > 1500: break
    cand.sort(reverse=True)
    return [tpl for _, tpl in cand[:max_triples]]

def build_triple_cols(X, triples):
    """
    Build multiplicative triple interaction columns given triplets.

    Returns
    -------
    (Phi, names) : (ndarray, list[str])
        Column block for x_i * x_j * x_k.
    """
    if not triples: return np.zeros((X.shape[0], 0))
    return np.column_stack([X[:, i] * X[:, j] * X[:, k] for (i, j, k) in triples])

def median_gamma(Xs, sample=512):
    """
    Median-heuristic gamma for RBF/RFF using a subsample of rows.

    Parameters
    ----------
    Xs : ndarray, shape (n_samples, n_features)
        Standardized features.
    sample : int, default=512
        Max rows sampled for pairwise distances.

    Returns
    -------
    float
        Gamma (>0). Used for RFF draw scale.
    """
    n = Xs.shape[0]
    if n < 2: return 1.0
    idx = np.random.RandomState(0).choice(n, size=min(sample, n), replace=False)
    Xs = Xs[idx]
    d2 = np.sum((Xs[:, None, :] - Xs[None, :, :])**2, axis=2)
    tri = d2[np.triu_indices_from(d2, k=1)]
    med = np.median(tri) if tri.size else np.mean(d2)
    med = float(med) if np.isfinite(med) and med > 1e-12 else 1.0
    return 1.0 / (med + 1e-9)

# ---------- safe linear algebra ----------
def safe_cholesky_psd(M, init_jitter=1e-6, max_tries=6):
    """
    Cholesky factorization with adaptive jitter for near-PSD matrices.

    Parameters
    ----------
    M : ndarray of shape (d, d)
        Symmetric matrix.
    init_jitter : float, default=1e-6
        Initial diagonal jitter.
    max_tries : int, default=6
        Doubling attempts on jitter.

    Returns
    -------
    L : ndarray of shape (d, d)
        Lower triangular factor s.t. (M + jitter*I) = L L^T.
    """
    I = np.eye(M.shape[0])
    jitter = init_jitter
    for _ in range(max_tries):
        try:
            return np.linalg.cholesky(M + jitter * I), jitter
        except np.linalg.LinAlgError:
            jitter *= 10.0
    # eigen repair
    w, V = np.linalg.eigh((M + M.T) * 0.5)
    w_clipped = np.clip(w, 1e-8, None)
    Msym = (V * w_clipped) @ V.T
    L = np.linalg.cholesky(Msym + 1e-8 * I)
    return L, jitter

# ---------- calibration ----------
def robust_affine(yhat_val, y_val):
    """
    Robust linear calibration `y ≈ a + b * yhat` on validation data.

    Parameters
    ----------
    yhat_val, y_val : array-like
        Validation predictions and targets.

    Returns
    -------
    (a, b) : tuple[float, float]
        Intercept and slope estimated (Huber or robust alternative).
    Notes
    -----
    Applied at predict-time to mitigate scale bias on held-out windows.
    """
    yhat_val = np.asarray(yhat_val).ravel()
    y_val = np.asarray(y_val).ravel()
    # Huber on [1, yhat] -> y
    X = np.column_stack([np.ones_like(yhat_val), yhat_val])
    try:
        hub = HuberRegressor(alpha=0.0, fit_intercept=False)
        hub.fit(X, y_val)
        a, b = float(hub.coef_[0]), float(hub.coef_[1])
    except Exception:
        # fallback least squares
        A = X.T @ X + 1e-9*np.eye(X.shape[1])
        a, b = np.linalg.solve(A, X.T @ y_val)
        a, b = float(a), float(b)
    return a, b


def _ensure_feature_names(X, feature_names):
    """
    Return a list of feature names matching columns of `X`.

    If `feature_names` is None or length-mismatched, synthesizes names.
    """
    if feature_names is not None:
        return list(feature_names)
    if hasattr(X, "columns"):
        return list(X.columns)
    return [f"x{j}" for j in range(X.shape[1])]

@dataclass
class _FIResult:
    """
    Container for feature importance results.

    Attributes
    ----------
    names : list of str
        Feature names (aligned with importances).
    importances : ndarray
        Scores for each feature.
    details : dict
        Optional method-specific extras (e.g., per-repeat drops).
    """
    names: List[str]
    importances: np.ndarray
    details: Dict[str, Any]


class ORBITRegressor(BaseEstimator, RegressorMixin):
    """
    Order-aware Regression with Basis, Interactions & Trends (ORBIT).

    Combines a linear head on standardized raw features with a screened,
    whitened nonlinear feature bank (univariate bases, pair/triple
    interactions, random Fourier features, and local radial "atom"
    features). The bank is estimated with ElasticNet + bank dropout and
    optionally refit by Huber-ridge. A ridge stacker blends heads on a
    forward validation split; split-conformal residuals provide PIs.

    Parameters
    ----------
    n_keep_dims : int, default=20
        Active dimensions for univariate basis expansion.
    n_knots : int, default=7
        Spline knots per active dimension.
    use_wavelet : bool, default=True
        Apply db2 soft-threshold denoising per feature (if available).
    include_splines : bool, default=True
        Include hinge-spline atoms in the univariate bank.
    top_pairs : int, default=64
        Number of pairwise interactions kept via stability selection.
    triple_max : int, default=6
        Max triple interactions kept by correlation.
    rff_dims : int, default=240
        Number of random Fourier features (RBF kernel approx).
    rff_gamma : {"auto", float}, default="auto"
        RBF scale; "auto" uses median heuristic. 
    n_atoms : int, default=64
        Number of local radial atoms.
    ridge_alpha : float, default=1.0
        Ridge regularization for robust refit.
    pre_screen_frac : float, default=0.25
        Fraction of bank kept by cosine screening before whitening.
    pre_screen_cap : int, default=900
        Max cols kept in stage-1 screening.
    screen_frac : float, default=0.60
        Fraction kept after whitening (stage-2 screening).
    screen_cap : int, default=1100
        Max cols kept in stage-2 screening.
    bank_dropout : float, default=0.10
        Probability of dropping each bank column per ENet fit.
    huber_refit : bool, default=True
        Enable Huber–ridge refit on the active set.
    huber_iters : int, default=3
        IRLS iterations for robust refit.
    stack_mode : {"ridge", "blend"}, default="ridge"
        Stacker choice; "blend" does convex linear/nonlinear mix.
    stack_ridge_alpha : float, default=1e-3
        Ridge penalty for stacker. 
    val_size : float, default=0.20
        Validation fraction (tail if temporal).
    val_mode : {"temporal", "random"}, default="temporal"
        Split protocol. Temporal preserves order. 
    use_target_transform : bool, default=False
        Learn a target transform on train (Yeo-Johnson) and invert at predict.
    target_transform : {"identity","yeo-johnson"}, default="identity"
        Transform to apply to targets. 
    random_state : int, default=0
        Seed controlling stochastic parts (pairs selection, dropout, RFF).

    Attributes
    ----------
    scaler_ : StandardScaler
        Fitted on training features.
    bank_params_ : dict
        Frozen parameters describing the constructed feature bank.
    W_whiten_ : ndarray of shape (k0, k0)
        Whitening transform from stage-1 to stage-2 screening.
    bank_keep_idx_ : ndarray of int
        Column indices retained after stage-2 screening.
    coef_lin_ : ndarray
        ElasticNet coefficients for linear head.
    bank_coef_ : ndarray
        Coefficients for nonlinear bank head (post-refit if enabled).
    stack_theta_ : ndarray
        Ridge stacker weights.
    q90_ : float
        Conformal PI half-width for 90% intervals (split residuals).
    """

    def __init__(self,
                 n_keep_dims=20, n_knots=7, use_wavelet=True, include_splines=True,
                 top_pairs=64, triple_max=6,
                 rff_dims=240, rff_gamma='auto',
                 n_atoms=64, ridge_alpha=1.0,
                 pre_screen_frac=0.25, pre_screen_cap=900,
                 screen_frac=0.6, screen_cap=1100,
                 bank_dropout=0.10, huber_refit=True, huber_iters=3,
                 stack_mode="ridge", stack_ridge_alpha=1e-3,
                 val_size=0.20, val_mode="temporal",
                 use_target_transform=False, target_transform="identity",
                 random_state=0):
        # feature-bank & screening
        self.n_keep_dims = int(n_keep_dims)
        self.n_knots = int(n_knots)
        self.use_wavelet = bool(use_wavelet)
        self.include_splines = bool(include_splines)
        self.top_pairs = int(top_pairs)
        self.triple_max = int(triple_max)
        self.rff_dims = int(rff_dims)
        self.rff_gamma = rff_gamma
        self.n_atoms = int(n_atoms)
        self.ridge_alpha = float(ridge_alpha)
        self.pre_screen_frac = float(pre_screen_frac)
        self.pre_screen_cap = int(pre_screen_cap)
        self.screen_frac = float(screen_frac)
        self.screen_cap = int(screen_cap)
        # estimation
        self.bank_dropout = float(bank_dropout)
        self.huber_refit = bool(huber_refit)
        self.huber_iters = int(huber_iters)
        self.stack_mode = str(stack_mode)
        self.stack_ridge_alpha = float(stack_ridge_alpha)
        # validation
        self.val_size = float(val_size)
        self.val_mode = str(val_mode).lower()  # "temporal" or "random"
        # target transform
        self.use_target_transform = bool(use_target_transform)
        self.target_transform = str(target_transform)
        # other
        self.random_state = int(random_state)

    # ---------- helpers ----------
    def _dynamic_sizes(self, n, p):
        """
        Compute size caps that depend on effective train size.

        Returns
        -------
        dict
            Derived sizes for pairs/triples/RFF/atoms and screening caps.
        """
        pairs = min(self.top_pairs, int(max(12, 2.1 * np.sqrt(max(n, 1)))))
        triples = min(self.triple_max, int(max(0, np.sqrt(max(n, 1)) / 4)))
        rff = min(self.rff_dims, max(120, int(min(320, 0.5 * n))))
        atoms = min(self.n_atoms, max(40, int(min(100, 0.24 * n))))
        return pairs, triples, rff, atoms

    def _init_bank_params(self, Xs, ytr):
        """
        Draw and cache all fixed elements required to build the bank.

        Notes
        -----
        Includes RFF projection parameters, atom centers/bandwidth,
        and bookkeeping for names and shapes.
        """
        n, p = Xs.shape
        # active dims by MI or cosine corr (fallback)
        try:
            from sklearn.feature_selection import mutual_info_regression
            mi = mutual_info_regression(Xs, ytr, random_state=self.random_state)
        except Exception:
            yc = ytr - ytr.mean()
            Zc = Xs - Xs.mean(axis=0)
            num = np.abs(Zc.T @ yc)
            den = np.sqrt(np.sum(Zc**2, axis=0)) * np.sqrt(np.sum(yc**2) + 1e-12) + 1e-12
            mi = num / den
        order = np.argsort(mi)[::-1]
        self.active_dims_ = order[:min(self.n_keep_dims, p)]

        pairs_k, triples_k, rff_k, atoms_k = self._dynamic_sizes(n, p)
        self.pairs_k_, self.triples_k_, self.rff_k_, self.atoms_k_ = pairs_k, triples_k, rff_k, atoms_k

        # interactions
        self.pairs_ = stability_select_pairs(Xs, ytr, top_k=pairs_k, rounds=6, subsample=0.7, rng=self.random_state)
        self.triples_ = select_top_triples(Xs, ytr, self.pairs_, max_triples=triples_k)

        # random Fourier features
        gamma = median_gamma(Xs) if (isinstance(self.rff_gamma, str) and self.rff_gamma == 'auto') else float(self.rff_gamma)
        rng = np.random.RandomState(self.random_state + 7)
        self.rff_W_ = rng.normal(0, np.sqrt(max(gamma, 1e-6)), size=(p, rff_k))
        self.rff_b_ = rng.uniform(0, 2*np.pi, size=(rff_k,))

        # atoms
        rngc = np.random.RandomState(self.random_state + 13)
        m = min(n, atoms_k)
        idx = rngc.choice(n, size=m, replace=False)
        self.atom_centers_ = Xs[idx]
        subn = min(512, n); sub = Xs[rngc.choice(n, size=subn, replace=False)]
        d = np.linalg.norm(sub[:, None, :] - sub[None, :, :], axis=2)
        med = np.median(d) + 1e-9
        self.atom_s2_ = (0.5 * med) ** 2

    def _make_bank_given_params(self, Xs):
        """
        Materialize the full (unstandardized) feature bank given cached params.

        Returns
        -------
        (Phi, names) : (ndarray, list[str])
            Concatenated columns: [1, Xs, uni, pair, triple, RFF, atoms].
        """
        n, p = Xs.shape
        U  = make_univariate_bank(Xs, self.active_dims_, n_knots=self.n_knots,
                                  use_wavelet=self.use_wavelet, include_splines=self.include_splines)
        PB = make_pair_bank(Xs, self.pairs_)
        TB = build_triple_cols(Xs, self.triples_)
        RFF = np.sqrt(2.0 / max(1, self.rff_k_)) * np.cos(Xs @ self.rff_W_ + self.rff_b_) if self.rff_k_ > 0 else np.zeros((n, 0))
        if getattr(self, "atom_centers_", None) is not None and self.atom_centers_.size > 0:
            D = np.linalg.norm(Xs[:, None, :] - self.atom_centers_[None, :, :], axis=2)
            AT = np.exp(-(D ** 2) / (2 * self.atom_s2_ + 1e-12))
        else:
            AT = np.zeros((n, 0))
        Phi = np.hstack([np.ones((n,1)), Xs, U, PB, TB, RFF, AT]) if n > 0 else np.zeros((0,1))
        return safe_nan_to_num(Phi)

    def _standardize_design(self, Phi):
        """
        Column-standardize bank (mean/std on train); leave intercept unscaled.

        Returns
        -------
        (Zc, mu, sigma) : (ndarray, ndarray, ndarray)
            Standardized matrix and per-column moments.
        """
        if Phi.size == 0:
            return Phi, np.array([0.0]), np.array([1.0])
        mu = Phi.mean(axis=0); sd = Phi.std(axis=0) + 1e-12
        mu[0] = 0.0; sd[0] = 1.0
        Z = (Phi - mu) / sd; Z[:,0] = 1.0
        return Z, mu, sd

    def _whiten(self, Z):
        """
        Compute whitening transformation after stage-1 screening.

        Returns
        -------
        (Z_w, W) : (ndarray, ndarray)
            Whitened design and whitener `W` (Cholesky-based).
        """
        C = (Z.T @ Z) / max(1, Z.shape[0])
        L, _ = safe_cholesky_psd(C, init_jitter=1e-6, max_tries=6)
        W = np.linalg.solve(L, np.eye(L.shape[0]))
        return W, Z @ W

    def _target_transform_fit(self, y):
        """
        Optionally fit a Yeo-Johnson transform on training targets.

        Returns
        -------
        ndarray
            Transformed targets y^(t). Inverse via `_target_inverse`.
        """
        if self.use_target_transform and self.target_transform.lower() != "identity":
            self.pt_ = PowerTransformer(method='yeo-johnson', standardize=False)
            return self.pt_.fit_transform(y.reshape(-1,1)).ravel()
        # identity
        self.pt_ = None
        return y.copy()

    def _target_inverse(self, yt):
        """Invert the fitted target transform (no-op if identity)."""
        if hasattr(self, "pt_") and self.pt_ is not None:
            return self.pt_.inverse_transform(np.asarray(yt).reshape(-1,1)).ravel()
        return np.asarray(yt).ravel()

    # ---------- fit/predict ----------
    def fit(self, X, y):
        """
        Fit ORBIT on (X, y).

        Steps
        -----
        1) Coerce/standardize X; optionally transform y.
        2) Build bank → stage-1 cosine screening → whitening → stage-2 screening.
        3) Fit linear head (ElasticNet) on raw standardized features.
        4) Fit nonlinear head with bank dropout; optional Huber-ridge refit.
        5) Temporal or random split for stacker; fit ridge/blend stacker.
        6) Calibrate conformal PI half-width on validation residuals.

        Returns
        -------
        self : ORBITRegressor
            Fitted estimator. 
        """
        X = np.asarray(X, float); y = np.asarray(y, float).ravel()
        X = safe_nan_to_num(X); y = safe_nan_to_num(y)
        n = len(y)

        # scaler on full train block
        self.scaler_ = StandardScaler().fit(X)
        Xs_all = self.scaler_.transform(X)

        # temporal vs random validation split (within *training data* only)
        n_val = max(1, int(self.val_size * n))
        if self.val_mode == "temporal":
            tr_idx = np.arange(0, n - n_val)
            val_idx = np.arange(n - n_val, n)
        else:
            rs = np.random.RandomState(self.random_state)
            idx = rs.permutation(n); val_idx, tr_idx = idx[:n_val], idx[n_val:]

        Xtr, ytr = Xs_all[tr_idx], y[tr_idx]
        Xval, yval = Xs_all[val_idx], y[val_idx]

        # target transform
        ytr_t  = self._target_transform_fit(ytr)
        yval_t = yval if self.pt_ is None else self.pt_.transform(yval.reshape(-1,1)).ravel()

        # ---- Stage A: Linear head on raw Xs ----
        try:
            elin = ElasticNetCV(
                l1_ratio=[0.1, 0.3, 0.6, 0.9],
                alphas=np.logspace(-3, 0.7, 12),
                cv=3, fit_intercept=False, random_state=self.random_state, max_iter=4000
            )
            elin.fit(Xtr, ytr_t)
            self.lin_coef_ = elin.coef_.astype(float)
        except Exception:
            # ridge fallback
            A = Xtr.T @ Xtr + 1.0*np.eye(Xtr.shape[1])
            b = Xtr.T @ ytr_t
            self.lin_coef_ = np.linalg.solve(A, b)

        yhat_lin_tr  = Xtr  @ self.lin_coef_
        yhat_lin_val = Xval @ self.lin_coef_

        # ---- Stage B: Nonlinear bank ----
        self._init_bank_params(Xtr, ytr_t)
        Phi_tr = self._make_bank_given_params(Xtr)
        Z_tr, mu, sd = self._standardize_design(Phi_tr)
        self.mu_bank_, self.sd_bank_ = mu, sd

        # Stage 1 screening
        yc = ytr_t - ytr_t.mean()
        Zc = Z_tr - Z_tr.mean(axis=0)
        if Z_tr.shape[1] == 0:
            pre_keep = np.array([0], dtype=int)
            Z_pre = np.ones((Z_tr.shape[0], 1))
        else:
            num = np.abs(np.dot(yc, Zc))
            den = (np.sqrt(np.sum(Zc**2, axis=0)) + 1e-12) * np.sqrt(np.sum(yc**2) + 1e-12)
            scor = num / den
            pre_nom = int(self.pre_screen_frac * Z_tr.shape[1])
            pre_cap = min(self.pre_screen_cap, int(0.9 * max(1, Z_tr.shape[0])))
            k0 = max(140, min(pre_nom, pre_cap))
            pre_keep = np.unique(np.concatenate([[0], np.argsort(scor)[::-1][:k0]]))
            Z_pre = Z_tr[:, pre_keep]

        # Whitening
        self.W_whiten_, Z_tr_w = self._whiten(Z_pre)

        # Stage 2 screening
        Zc2 = Z_tr_w - Z_tr_w.mean(axis=0)
        if Z_tr_w.shape[1] == 0:
            keep_idx = np.array([0], dtype=int); Zs = np.ones((Z_tr_w.shape[0], 1))
        else:
            num2 = np.abs(np.dot(yc, Zc2))
            den2 = (np.sqrt(np.sum(Zc2**2, axis=0)) + 1e-12) * np.sqrt(np.sum(yc**2) + 1e-12)
            scor2 = num2 / den2
            k_nom = int(self.screen_frac * Z_tr_w.shape[1])
            cap   = min(self.screen_cap, int(0.8 * max(1, Z_tr_w.shape[0])))
            k     = max(90, min(k_nom, cap))
            idx_sorted = np.argsort(scor2)[::-1][:k]
            keep_idx = np.unique(np.concatenate([[0], idx_sorted]))
            Zs = Z_tr_w[:, keep_idx]

        # ElasticNet + bank dropout (averaged)
        rng = np.random.RandomState(self.random_state + 111)

        def _fit_enet(Zin, y_t):
            enet = ElasticNetCV(
                l1_ratio=[0.1, 0.3, 0.6],
                alphas=np.logspace(-3, 0.7, 10),
                cv=3, fit_intercept=False, random_state=self.random_state, max_iter=3000
            )
            enet.fit(Zin, y_t)
            return enet.coef_.astype(float)

        coef_accum = np.zeros(Zs.shape[1])
        draws = 2 if self.bank_dropout > 0 else 1
        for _ in range(draws):
            if self.bank_dropout > 0 and Zs.shape[1] > 1:
                mask = np.ones(Zs.shape[1], dtype=bool)
                mask[0] = True
                drop = rng.rand(Zs.shape[1]) < self.bank_dropout
                mask &= ~drop
                if mask.sum() < max(12, int(0.2*Zs.shape[1])):
                    mask[:] = True; mask[0] = True
                w = _fit_enet(Zs[:, mask], ytr_t)
                tmp = np.zeros_like(coef_accum); tmp[mask] = w
                coef_accum += tmp
            else:
                coef_accum += _fit_enet(Zs, ytr_t)
        w_s = coef_accum / draws

        # robust refit (Huber–ridge) on active set
        act = np.flatnonzero(np.abs(w_s) > 1e-8)
        if act.size == 0: act = np.array([0], int)
        Za = Zs[:, act]
        if self.huber_refit and Za.shape[1] > 0:
            w_act = w_s[act].copy()
            r = ytr_t - Za @ w_act
            delta = 1.345 * (np.median(np.abs(r - np.median(r))) / 0.6745 + 1e-12)
            for _ in range(self.huber_iters):
                r = ytr_t - Za @ w_act
                wts = np.where(np.abs(r) <= delta, 1.0, delta / (np.abs(r) + 1e-12))
                W12 = np.sqrt(wts)[:, None]
                A = (Za * W12).T @ (Za * W12) + (self.ridge_alpha + 1e-6) * np.eye(Za.shape[1])
                b = (Za * W12).T @ (ytr_t * np.sqrt(wts))
                w_act = np.linalg.solve(A, b)
            w_s = np.zeros_like(w_s); w_s[act] = w_act

        # stash bank params
        self.pre_keep_      = pre_keep
        self.bank_keep_idx_ = keep_idx
        self.bank_coef_     = w_s

        # Validation preds for stacking
        Phi_val_full = self._make_bank_given_params(Xval)
        Z_val0 = (Phi_val_full - self.mu_bank_) / self.sd_bank_; Z_val0[:, 0] = 1.0

        # 1) select the same pre-screened columns as train
        Z_val_pre = Z_val0[:, self.pre_keep_]

        # 2) apply the whitener learned on train (no transpose)
        #    train did: Z_tr_w = Z_pre @ W
        #    so here:  Z_val_w = Z_val_pre @ W
        if Z_val_pre.shape[1] != self.W_whiten_.shape[0]:
            raise ValueError(
                f"Whitening shape mismatch: Z_val_pre has {Z_val_pre.shape[1]} cols, "
                f"W_whiten_ expects {self.W_whiten_.shape[0]}."
            )
        Z_val_w = Z_val_pre @ self.W_whiten_

        # 3) keep the same screened columns after whitening
        Zv = Z_val_w[:, self.bank_keep_idx_]

        yhat_nl_val = Zv @ self.bank_coef_

        # ================= Anti-flatline + calibration =================
        tiny = 1e-8
        std_lin = float(np.std(yhat_lin_val))
        std_nl  = float(np.std(yhat_nl_val))

        self.w_stack_ = None
        self.alpha_   = None

        # 1) If both heads near-constant, force intercept = mean(yval_t)
        if std_lin < 1e-6 and std_nl < 1e-6:
            self.w_stack_ = np.array([float(np.mean(yval_t)), 0.0, 0.0, 0.0], dtype=float)
        elif self.stack_mode.lower() == "ridge":
            S = np.column_stack([np.ones_like(yhat_lin_val),
                                 yhat_lin_val,
                                 yhat_nl_val,
                                 yhat_lin_val * yhat_nl_val])
            try:
                rrg = Ridge(alpha=self.stack_ridge_alpha, fit_intercept=False, random_state=self.random_state)
                rrg.fit(S, yval_t)
                self.w_stack_ = rrg.coef_.astype(float)
            except Exception:
                self.w_stack_ = None

        # 2) Fallback convex blend if ridge failed
        if self.w_stack_ is None:
            alphas = np.linspace(0.0, 1.0, 21)
            best_alpha, best_rmse = 0.5, 1e18
            for a in alphas:
                yv = a*yhat_lin_val + (1.0 - a)*yhat_nl_val
                r  = float(np.sqrt(np.mean((yv - yval_t)**2)))
                if r < best_rmse: best_rmse, best_alpha = r, float(a)
            self.alpha_ = best_alpha

        # 3) Build validation preds in transformed space
        if self.w_stack_ is not None:
            yval_hat_t = (self.w_stack_[0]
                          + self.w_stack_[1]*yhat_lin_val
                          + self.w_stack_[2]*yhat_nl_val
                          + self.w_stack_[3]*(yhat_lin_val*yhat_nl_val))
        else:
            yval_hat_t = self.alpha_*yhat_lin_val + (1.0 - self.alpha_)*yhat_nl_val

        # 4) Calibration y_val_t ≈ a + b*yval_hat_t
        if float(np.std(yval_hat_t)) < 1e-8:
            self.cal_a_, self.cal_b_ = float(np.mean(yval_t)), 0.0
        else:
            Xcal = np.column_stack([np.ones_like(yval_hat_t), yval_hat_t])
            coef, *_ = np.linalg.lstsq(Xcal, yval_t, rcond=None)
            self.cal_a_, self.cal_b_ = float(coef[0]), float(coef[1])

        # 5) Conformal on *calibrated* validation preds (inverse after cal)
        yval_hat_t_cal = self.cal_a_ + self.cal_b_ * yval_hat_t
        yval_hat = self._target_inverse(yval_hat_t_cal)
        resid = np.abs(yval - yval_hat)
        self.q90_ = float(np.quantile(resid, 0.90)) if resid.size else 0.0

        return self

    def predict(self, X, return_interval=False):
        """
        Predict on new data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input features.
        return_interval : bool, default=False
            If True, return split-conformal 90% PIs based on stored half-width.

        Returns
        -------
        y : ndarray of shape (n_samples,)
            Point predictions on original target scale.
        (optional) PI : ndarray of shape (n_samples, 2)
            Lower/upper bounds if `return_interval=True`. 
        """
        X = np.asarray(X, float); X = safe_nan_to_num(X)
        Xs = self.scaler_.transform(X)

        # linear head
        y_lin_t = Xs @ self.lin_coef_

        # nonlinear bank
        Phi = self._make_bank_given_params(Xs)
        Z0  = (Phi - self.mu_bank_) / self.sd_bank_; Z0[:,0] = 1.0
        Z_pre = Z0[:, self.pre_keep_]
        Z_w   = Z_pre @ self.W_whiten_
        Zk    = Z_w[:, self.bank_keep_idx_]
        y_nl_t = Zk @ self.bank_coef_

        # stack/blend
        if getattr(self, "w_stack_", None) is not None:
            yhat_t = (self.w_stack_[0]
                      + self.w_stack_[1]*y_lin_t
                      + self.w_stack_[2]*y_nl_t
                      + self.w_stack_[3]*(y_lin_t*y_nl_t))
        else:
            yhat_t = self.alpha_*y_lin_t + (1.0 - self.alpha_)*y_nl_t

        # apply calibration (a + b*yhat_t)
        a = getattr(self, "cal_a_", 0.0)
        b = getattr(self, "cal_b_", 1.0)
        yhat_t = a + b * yhat_t

        # inverse transform + safety
        yhat = self._target_inverse(yhat_t)
        yhat = safe_nan_to_num(yhat)

        if not return_interval:
            return yhat
        q = float(getattr(self, "q90_", 0.0))
        return yhat, np.column_stack([yhat - q, yhat + q])

    def linear_feature_importance(self, feature_names=None, normalize=True):
        """
        Absolute-coefficient importances for the linear head.

        Parameters
        ----------
        normalize : bool, default=True
            Normalize to sum to 1 if positive.

        Returns
        -------
        pandas.DataFrame
            Columns: feature, importance_lin.
        """
        import pandas as pd

        if not hasattr(self, "lin_coef_"):
            raise RuntimeError("Model not yet fit; lin_coef_ missing.")

        names = _ensure_feature_names(
            np.zeros((1, len(self.lin_coef_))), feature_names
        )
        imp = np.abs(np.asarray(self.lin_coef_).ravel().astype(float))
        if normalize and imp.sum() > 0:
            imp = imp / imp.sum()

        df = pd.DataFrame({"feature": names, "importance_linear": imp})
        df.sort_values("importance_linear", ascending=False, inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df

    # =============== Sensitivity importance (model-agnostic) ===============
    def sensitivity_feature_importance(
        self,
        X_ref,
        feature_names=None,
        eps_std: float = 0.25,
        sample: int = 1024,
        random_state: int = 0,
        normalize: bool = True,
        return_details: bool = False,
    ):
        """
        Sensitivity-based feature importance via small random perturbations.

        Parameters
        ----------
        X : ndarray
            Baseline inputs (standardized internally).
        frac : float, default=0.05
            Relative perturbation scale per feature.
        repeats : int, default=8
            Monte Carlo repeats.
        random_state : int, default=0
            Seed.
        normalize : bool, default=True
            Normalize to sum to 1.

        Returns
        -------
        pandas.DataFrame
            Columns: feature, importance_sens_mean, importance_sens_std.
        """
        import pandas as pd

        X_ref = np.asarray(X_ref, float)
        n, p = X_ref.shape
        names = _ensure_feature_names(X_ref, feature_names)

        # subsample rows for speed
        rng = np.random.RandomState(random_state)
        if sample is not None and sample < n:
            idx = rng.choice(n, size=sample, replace=False)
            Xb = X_ref[idx].copy()
        else:
            Xb = X_ref.copy()

        # per-feature finite-difference slope magnitude
        scales = getattr(self, "scaler_", None)
        if scales is None or not hasattr(scales, "scale_"):
            # fallback: empirical std on X_ref
            scale_vec = np.std(X_ref, axis=0) + 1e-12
            mu_vec = np.mean(X_ref, axis=0)
        else:
            scale_vec = np.asarray(self.scaler_.scale_).ravel() + 1e-12
            mu_vec = np.asarray(self.scaler_.mean_).ravel()

        delta_orig = eps_std * scale_vec  # perturb size in ORIGINAL units
        base_pred = np.asarray(self.predict(Xb)).ravel()

        imp = np.zeros(p, float)
        for j in range(p):
            if delta_orig[j] == 0.0:
                imp[j] = 0.0
                continue
            Xp = Xb.copy(); Xm = Xb.copy()
            Xp[:, j] += delta_orig[j]
            Xm[:, j] -= delta_orig[j]
            yp = np.asarray(self.predict(Xp)).ravel()
            ym = np.asarray(self.predict(Xm)).ravel()
            # central difference derivative
            g = (yp - ym) / (2.0 * delta_orig[j])
            imp[j] = float(np.mean(np.abs(g)))

        if normalize and imp.sum() > 0:
            imp = imp / imp.sum()

        df = pd.DataFrame({"feature": names, "importance_sensitivity": imp})
        df.sort_values("importance_sensitivity", ascending=False, inplace=True)
        df.reset_index(drop=True, inplace=True)

        if return_details:
            return df, _FIResult(names, imp, dict(
                eps_std=eps_std, sample=len(Xb), method="sensitivity"))
        return df

    # =============== Permutation importance (model-agnostic) ===============
    def permutation_feature_importance(
        self,
        X,
        y,
        feature_names=None,
        n_repeats: int = 10,
        random_state: int = 0,
        scoring: str = "neg_mean_squared_error",
        sample: Optional[int] = None,
        block_size: Optional[int] = None,
        normalize: bool = True,
        return_details: bool = False,
    ):
        """
        Scikit-learn permutation importance wrapper for the full model.

        Parameters
        ----------
        X, y : ndarray
            Data to evaluate on (no re-fitting).
        n_repeats : int, default=10
            Number of permutations per feature.
        random_state : int, default=0
            Seed.
        normalize : bool, default=True
            Normalize mean drops to sum to 1 if positive.
        return_details : bool, default=False
            If True, also return the raw `permutation_importance` result.

        Returns
        -------
        pandas.DataFrame
            Feature-wise mean/std drop in score.
        (optional) details : sklearn.inspection._PermutationImportance
            Full result if requested.
        """
        import pandas as pd

        X = np.asarray(X, float); y = np.asarray(y, float).ravel()
        names = _ensure_feature_names(X, feature_names)
        rng = np.random.RandomState(random_state)

        # Optional row subsample (speeds up on long series)
        if sample is not None and sample < X.shape[0]:
            idx = rng.choice(X.shape[0], size=sample, replace=False)
            X_eval, y_eval = X[idx], y[idx]
        else:
            X_eval, y_eval = X, y

        if block_size is None:
            # vanilla sklearn permutation importance
            pi = permutation_importance(
                self, X_eval, y_eval,
                n_repeats=n_repeats,
                random_state=random_state,
                scoring=scoring
            )
            mean_imp = np.maximum(0.0, pi.importances_mean)
            std_imp  = pi.importances_std
        else:
            # block permutation (custom)
            n, p = X_eval.shape
            K = int(np.ceil(n / block_size))
            base_pred = self.predict(X_eval)
            if scoring == "neg_mean_squared_error":
                def scorer(y_true, y_pred):
                    return -np.mean((y_true - y_pred)**2)
            else:
                # default to RMSE drop (negative)
                def scorer(y_true, y_pred):
                    return -np.sqrt(np.mean((y_true - y_pred)**2))

            base_score = scorer(y_eval, base_pred)
            reps = n_repeats
            drops = np.zeros((reps, p), float)

            for r in range(reps):
                for j in range(p):
                    Xp = X_eval.copy()
                    # permute column j by shuffling blocks
                    order = np.arange(K)
                    rng.shuffle(order)
                    col = Xp[:, j].copy()
                    blocks = [col[k*block_size:(k+1)*block_size] for k in range(K)]
                    perm_col = np.concatenate([blocks[k] for k in order])[:n]
                    Xp[:, j] = perm_col
                    score_j = scorer(y_eval, self.predict(Xp))
                    drops[r, j] = base_score - score_j  # higher is more important

            mean_imp = np.maximum(0.0, drops.mean(axis=0))
            std_imp  = drops.std(axis=0)

        if normalize and mean_imp.sum() > 0:
            norm = mean_imp.sum()
            mean_imp = mean_imp / norm
            std_imp = std_imp / norm

        df = pd.DataFrame({
            "feature": names,
            "importance_perm_mean": mean_imp,
            "importance_perm_std": std_imp
        }).sort_values("importance_perm_mean", ascending=False).reset_index(drop=True)

        if return_details:
            return df, _FIResult(names, mean_imp, dict(
                n_repeats=n_repeats, scoring=scoring, sample=getattr(X_eval, "shape", [0])[0],
                block_size=block_size, method="permutation"))
        return df

    # =============== Unified convenience: sets feature_importances_ ===============
    def compute_feature_importances(
        self,
        X_ref,
        y_ref=None,
        feature_names=None,
        method: str = "sensitivity",
        **kwargs
    ):
        """
        Unified feature-importance entrypoint.

        Parameters
        ----------
        mode : {"linear", "sensitivity", "permutation"}, default="permutation"
            Which importance to compute.
        X, y : optional
            Required for "sensitivity" and "permutation".

        Returns
        -------
        pandas.DataFrame or tuple
            Importance table (and details if requested).
        """
        if method == "linear":
            df = self.linear_feature_importance(feature_names=feature_names)
            self.feature_importances_ = df["importance_linear"].values
            self.feature_names_in_ = df["feature"].tolist()
            meta = {"method": "linear"}
            return df, meta

        if method == "sensitivity":
            df = self.sensitivity_feature_importance(
                X_ref, feature_names=feature_names, **kwargs
            )
            self.feature_importances_ = df["importance_sensitivity"].values
            self.feature_names_in_ = df["feature"].tolist()
            meta = {"method": "sensitivity", **kwargs}
            return df, meta

        if method == "permutation":
            if y_ref is None:
                raise ValueError("y_ref is required for permutation importance.")
            df = self.permutation_feature_importance(
                X_ref, y_ref, feature_names=feature_names, **kwargs
            )
            self.feature_importances_ = df["importance_perm_mean"].values
            self.feature_names_in_ = df["feature"].tolist()
            meta = {"method": "permutation", **kwargs}
            return df, meta

        raise ValueError(f"Unknown method '{method}'.")

    def predict_components(self, X, on_target_scale=True, include_stack=False, return_interval=False):
        """
        Decompose predictions into linear and nonlinear components.

        Parameters
        ----------
        X : array-like
        on_target_scale : bool, default=True
            If True, return on original target scale (inverse-transform + calibration).
        include_stack : bool, default=False
            If True, also return final stacked prediction.
        return_interval : bool, default=False
            If True and `include_stack`, include 90% PI for the stacked output.

        Returns
        -------
        dict
            Keys: "y_lin", "y_nl" and optionally "y_stack" and "PI".
        """
        import numpy as np

        X = np.asarray(X, float)
        X = np.where(np.isfinite(X), X, 0.0)
        Xs = self.scaler_.transform(X)

        # linear head (target-transformed space)
        y_lin_t = Xs @ self.lin_coef_

        # nonlinear head (via bank)
        Phi = self._make_bank_given_params(Xs)
        Z0  = (Phi - self.mu_bank_) / self.sd_bank_
        Z0[:, 0] = 1.0
        Z_pre = Z0[:, self.pre_keep_]
        Z_w   = Z_pre @ self.W_whiten_           # whiten
        Zk    = Z_w[:, self.bank_keep_idx_]
        y_nl_t = Zk @ self.bank_coef_

        # stack (still in transformed space)
        if getattr(self, "w_stack_", None) is not None:
            y_stack_t = (self.w_stack_[0]
                        + self.w_stack_[1]*y_lin_t
                        + self.w_stack_[2]*y_nl_t
                        + self.w_stack_[3]*(y_lin_t*y_nl_t))
        else:
            y_stack_t = self.alpha_*y_lin_t + (1.0 - self.alpha_)*y_nl_t

        # back to original target scale
        if on_target_scale:
            y_lin = self._target_inverse(y_lin_t)
            y_nl  = self._target_inverse(y_nl_t)
            y_out = self._target_inverse(y_stack_t)
        else:
            y_lin, y_nl, y_out = y_lin_t, y_nl_t, y_stack_t

        out = {"y_lin": y_lin, "y_nl": y_nl}

        if include_stack:
            out["y_stack"] = y_out
            if return_interval:
                q = getattr(self, "q90_", 0.0)
                out["PI"] = np.column_stack([y_out - q, y_out + q])

        return out


class ORBITBaggedRegressor(BaseEstimator, RegressorMixin):
    """
    Bagging wrapper around `ORBITRegressor`.

    Trains `n_estimators` independent base models with different seeds
    and averages predictions. PI half-widths are aggregated by RMS.

    Parameters
    ----------
    n_estimators : int, default=3
        Number of base learners.
    base_kwargs : dict or None, default=None
        kwargs forwarded to `ORBITRegressor` (exclude `random_state`).
    random_state : int, default=0
        Seed to diversify base learners.
    """
    def __init__(self, n_estimators=3, base_kwargs=None, random_state=0):
        self.n_estimators = int(n_estimators)
        self.base_kwargs  = dict(base_kwargs or {})
        self.random_state = int(random_state)

    def fit(self, X, y):
        """
        Fit all base estimators independently and cache per-model artifacts.

        Returns
        -------
        self : ORBITBaggedRegressor
        """
        self.models_ = []
        for k in range(self.n_estimators):
            seed = self.random_state + 177 * k
            # inject seed; do NOT pass any other random_state from base_kwargs
            kwargs = dict(self.base_kwargs)
            kwargs['random_state'] = seed
            m = ORBITRegressor(**kwargs)
            m.fit(X, y)
            self.models_.append(m)
        # aggregate PI half-widths by RMS
        self.q90_ = float(np.sqrt(np.mean([getattr(m, "q90_", 0.0)**2 for m in self.models_])))
        return self

    def predict(self, X, return_interval=False):
        """
        Average base predictions; optionally return aggregated 90% PIs.

        Returns
        -------
        y : ndarray of shape (n_samples,)
        (optional) PI : ndarray of shape (n_samples, 2)
            If `return_interval=True`.
        """
        preds = [m.predict(X) for m in self.models_]
        yhat = np.mean(preds, axis=0)
        if not return_interval:
            return yhat
        q = float(getattr(self, "q90_", 0.0))
        return yhat, np.column_stack([yhat - q, yhat + q])

    def compute_feature_importances(
        self, X_ref, y_ref=None, feature_names=None, method="sensitivity", **kwargs
    ):
        """
        Compute importances by averaging results from base estimators.

        Notes
        -----
        For permutation or sensitivity modes, the same (X, y) are reused
        across members for fair comparison.
        """
        import pandas as pd
        if not hasattr(self, "models_") or len(self.models_) == 0:
            raise RuntimeError("Bagged model not yet fit.")

        dfs = []
        for m in self.models_:
            df, meta = m.compute_feature_importances(
                X_ref, y_ref=y_ref, feature_names=feature_names,
                method=method, **kwargs
            )
            dfs.append(df.set_index("feature"))

        # union of features, fill missing with 0
        all_idx = dfs[0].index
        for d in dfs[1:]:
            all_idx = all_idx.union(d.index)

        # pick the right column based on method
        col = ("importance_sensitivity" if method == "sensitivity"
               else "importance_perm_mean" if method == "permutation"
               else "importance_linear")

        stacked = []
        for d in dfs:
            stacked.append(d.reindex(all_idx).fillna(0.0)[col].values)

        arr = np.vstack(stacked)
        mean_imp = arr.mean(axis=0)

        out = pd.DataFrame({
            "feature": list(all_idx),
            "importance_mean": mean_imp
        }).sort_values("importance_mean", ascending=False).reset_index(drop=True)

        # set sklearn-friendly attributes on the bagged wrapper
        self.feature_importances_ = out["importance_mean"].values
        self.feature_names_in_ = out["feature"].tolist()
        meta["bags"] = len(self.models_)
        meta["method"] = method
        return out, meta

    def predict_components(self, X, on_target_scale=True, include_stack=False, return_interval=False):
        """
        Average component-wise outputs from members ("y_lin", "y_nl"),
        and optionally "y_stack" with an ensemble PI.
        """
        import numpy as np

        if not hasattr(self, "models_") or len(self.models_) == 0:
            raise RuntimeError("ORBITBaggedRegressor is not fitted yet.")

        # collect per-model component outputs
        outs = [
            m.predict_components(
                X,
                on_target_scale=on_target_scale,
                include_stack=include_stack,
                return_interval=False,          # we'll handle PI at the bag level
            )
            for m in self.models_
        ]

        y_lin = np.mean([o["y_lin"] for o in outs], axis=0)
        y_nl  = np.mean([o["y_nl"]  for o in outs], axis=0)

        result = {"y_lin": y_lin, "y_nl": y_nl}

        if include_stack:
            y_stack = np.mean([o["y_stack"] for o in outs], axis=0)
            result["y_stack"] = y_stack
            if return_interval:
                # use bag's stored q90_ (RMS aggregation done at fit)
                q = getattr(self, "q90_", 0.0)
                result["PI"] = np.column_stack([y_stack - q, y_stack + q])

        return result
