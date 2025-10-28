# ORBIT: Order-Aware Regression with Basis, Interactions & Trends

ORBIT is a structured regression framework designed for tabular and time-dependent data leveraging sparse linear modeling, interpretable nonlinear feature banks, robust estimation, and conformal uncertainty calibration. In principle the solution can be described by three components: structured feature banks, order aware training and robust uncertainty. 

ORBIt aims to provide:
1. Interpretability: Sparse linear structure and interpretable basis features
2. Numerical stability: Two-stage correlation screening with whitening
3. Temporal awareness: Forward-validation with stacking and conformal calibration
4. Robustness: Huber IRLS refit and optional bagging
5. Efficiency: Fixed feature banks with no gradient training enabled fast inference compared to it's neural network competitor

## Key features 

- Linear head: Sparse ElasticNet on standardized raw features
- Nonlinear head: ElasticNet over a screened, whitened feature bank
- Feature bank: Combines wavelet-denoised univariate bases, pair/triple interactions, RFFs, and local atom features
- Screening + Whitening: Two-stage cosine correlation screening with Cholesky whitening for stability
- Huber–ridge refit: Robust IRLS refinement on the active set
- Ridge stacker: Blends linear and nonlinear heads using forward-validation stacking
- Conformal PIs: Split-conformal residual calibration for 90% prediction intervals
- Bagging: Optional ensemble for stability and improved coverage



## Quick start example

``` python
from orbit.regression import ORBITRegressor, ORBITBaggedRegressor
from sklearn.datasets import make_friedman1
from sklearn.metrics import mean_absolute_error, r2_score, PredictionErrorDisplay
import numpy as np, matplotlib.pyplot as plt, shap

# 1) Prepare data
X_df, y_df = make_friedman1(n_samples=4000, n_features=15, noise=1.0, random_state=42)
n_test = 150
Xtr, Xte = X_df[:-n_test], X_df[-n_test:]
ytr, yte = y_df[:-n_test], y_df[-n_test:]

# 2) Initialize bagged ORBIT model
orbit = ORBITBaggedRegressor(
    n_estimators=3,
    random_state=42,
    base_kwargs=dict(
        n_keep_dims=22, n_knots=7, use_wavelet=True, include_splines=True,
        top_pairs=48, triple_max=4,
        rff_dims=200, rff_gamma='auto',
        n_atoms=64, ridge_alpha=1.0,
        pre_screen_frac=0.25, pre_screen_cap=800,
        screen_frac=0.6, screen_cap=900,
        bank_dropout=0.10, huber_refit=True, huber_iters=3,
        stack_mode="ridge", stack_ridge_alpha=1e-3,
        val_size=0.20,
    )
)

# 3) Fit model and predict
orbit.fit(Xtr, ytr)
yhat, PI = orbit.predict(Xte, return_interval=True)

# 4) Evaluate
rmse = np.sqrt(np.mean((yte - yhat)**2))
mae = mean_absolute_error(yte, yhat)
r2 = r2_score(yte, yhat)
coverage = np.mean((yte >= PI[:,0]) & (yte <= PI[:,1]))

print(f"RMSE={rmse:.4f}, MAE={mae:.4f}, R2={r2:.4f}, 90% PI coverage={coverage:.3f}")

# 5) Plot forecasts and prediction intervals
plt.figure(figsize=(10,4.5))
plt.plot(yte, lw=1.5, label="Actual")
plt.plot(yhat, lw=1.5, label="ORBIT Forecast")
plt.fill_between(np.arange(len(yte)), PI[:,0], PI[:,1], alpha=0.2, label="90% PI")
plt.legend(); plt.tight_layout(); plt.show()

# 6) Diagnostic plots
fig, axs = plt.subplots(1, 2, figsize=(12, 5))
PredictionErrorDisplay.from_estimator(orbit, Xte, yte, kind="actual_vs_predicted", ax=axs[0])
PredictionErrorDisplay.from_estimator(orbit, Xte, yte, kind="residual_vs_predicted", ax=axs[1])
plt.show()

# 7) SHAP-based explanations
from orbit.explain import shap_callables, beeswarm_one
f_full, f_lin, f_nl = shap_callables(orbit)
mask = shap.maskers.Independent(shap.sample(Xtr, 1000))
expl_full = shap.Explainer(f_full, mask, feature_names=[f"x{i}" for i in range(X_df.shape[1])])
s_full = expl_full(Xte)
beeswarm_one(s_full, "Full Model")
```

### Explainability

orbit.explain provides optional utilities to understand ORBIT’s predictions:
- shap_callables(model): builds SHAP compatible functions for the full model, linear head, and nonlinear bank.
- beeswarm_one(sv, title): plots SHAP beeswarm charts.
- perm_importance_callable(f, X, y): computes permutation feature importance for any callable predictor.

## Very high level theory

The estimator can be seen as:

$\hat{y}(x) = \theta_0 + \theta_\text{lin}^T X^{(s)} + \theta_\text{nl}^T \Phi(X^{(s)}) + \text{stacked correction}$

where the nonlinear bank $\Phi(\cdot)$ spans RBFs, trigonometric bases, hinge splines, interactions, and random Fourier features. Whitening and ElasticNet produce sparse, stable coefficients. Conformal prediction intervals provide data-driven uncertainty.

## When to use ORBIT?

- Tabular regression with moderate feature counts (10–200)
- Time series forecasting with structured lags or exogenous variables
- Cases needing interpretable nonlinear effects and uncertainty
- Scenarios where model stability and calibration matter

ORBITBaggedRegressor provides an ensemble averaging that can produce smoother and more reliable uncertainty however ORBITRegressor is much faster and works well most of the time.
