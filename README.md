# ORBIT: Order-Aware Regression with Basis, Interactions & Trends

ORBIT is a structured regression framework designed for tabular and time-dependent data leveraging sparse linear modeling, interpretable nonlinear feature banks, robust estimation, and conformal uncertainty calibration. In principle the solution can be described by three components: structured feature banks, order aware training and robust uncertainty. 

ORBIt aims to provide:
1. Interpretability: Sparse linear structure and interpretable basis features
2. Numerical stability: Two-stage correlation screening with whitening
3. Temporal awareness: Forward-validation with stacking and conformal calibration
4. Robustness: Huber IRLS refit and optional bagging
5. Efficiency: Fixed feature banks with no gradient training enabled fast inference compared to it's neural network competitor

## Key features 

- Linear head: Sparse ElasticNet on standardised raw features
- Nonlinear head: ElasticNet over a screened, whitened feature bank
- Feature bank: Combines wavelet-denoised univariate bases, pair/triple interactions, RFFs, and local atom features
- Screening + Whitening: Two-stage cosine correlation screening with Cholesky whitening for stability
- Huber–ridge refit: Robust IRLS refinement on the active set
- Ridge stacker: Blends linear and nonlinear heads using forward-validation stacking
- Conformal PIs: Split-conformal residual calibration for 90% prediction intervals
- Bagging: Optional ensemble for stability and improved coverage



## Quick start example

``` python
from orbit.model.regression import ORBITRegressor, ORBITBaggedRegressor
from sklearn.datasets import make_friedman1
from sklearn.metrics import mean_absolute_error, r2_score, PredictionErrorDisplay
import numpy as np, matplotlib.pyplot as plt, shap

# 1) Prepare data
X_df, y_df = make_friedman1(n_samples=4000, n_features=15, noise=1.0, random_state=42)
n_test = 150
Xtr, Xte = X_df[:-n_test], X_df[-n_test:]
ytr, yte = y_df[:-n_test], y_df[-n_test:]

# 2) Initialise bagged ORBIT model
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
from orbit.model.explain import shap_callables, beeswarm_one
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

``` python
import shap
from model.explain import shap_callables, beeswarm_one, perm_importance_callable

f_full, f_lin, f_nl = shap_callables(orbit)

Xtr_df, Xte_df = X_df[:-n_test], X_df[-n_test:]
X_s = shap.sample(Xtr_df, 1000) 
mask  = shap.maskers.Independent(X_s)

expl_full = shap.Explainer(f_full, mask, feature_names=[f'x{i}' for i in range(X_df.shape[0])])
expl_lin  = shap.Explainer(f_lin,  mask, feature_names=[f'x{i}' for i in range(X_df.shape[0])])
expl_nl   = shap.Explainer(f_nl,   mask, feature_names=[f'x{i}' for i in range(X_df.shape[0])])

s_full = expl_full(Xte_df)
s_lin  = expl_lin(Xte_df)
s_nl   = expl_nl(Xte_df)

beeswarm_one(s_full, "Full model")
beeswarm_one(s_lin,  "Linear head")
beeswarm_one(s_nl,   "Nonlinear head")
```

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


# Mathematical breakdown of ORBIT

## Notation and assumptions

- Observations $\{(x_t, y_t)\}_{t=1}^n$ with $x_t\in\mathbb{R}^p, y_t\in\mathbb{R}$.
- In time series, index t is chronological; training/validation/test must respect order.
- $X\in\mathbb{R}^{n\times p}$: feature matrix; $y\in\mathbb{R}^n$: target vector.
- $\mathbb{1}$: vector of ones; $\|v\|_2$ Euclidean norm; $\odot$ Hadamard product.

There is a general assumption that all numerical features are finite and any non-nuermical columns have been appropriately encoded before passed to ORBIT.

## Preprocessing

**1. Finite coercion**

Non-finite values are replaced by zero:

$\tilde{X}{ij} \leftarrow
\begin{cases} X{ij}, & \text{finite} \\ 0, & \text{otherwise}\end{cases},\quad
\tilde{y}_i \leftarrow
\begin{cases} y_i, & \text{finite} \\ 0, & \text{otherwise}\end{cases}$

**2. Standardisation**

Fit mean $\mu_X$, std $\sigma_X>0$ on train and compute
$X^{(s)} = (X-\mu_X)\oslash \sigma_X$

**3. Target Transform**

The default is identity: $y^{(t)}=y$ however ORBIT offers the opption to apply Yeo–Johnson $T_\lambda(y)$ (without standardisation) that is learnt on train and then inverted at prediction time.

## Feature Bank Construction

The feature bank $\Phi(\cdot)$ is in essence a concatenation of interpretable, fixed basis blocks applied to the scaled features $x^{(s)}$.

**1. Wavelet denoising**

For a feature $x$, the solution applies a Daubechies (db2) wavelet decomposition to level $L$ with soft-threshold $\tau\propto \hat{\sigma}\sqrt{2\log n}$. Reconstruct to obtain $x^{(d)}$ if the module `pywt` is unavailable or $n<8$, bypass.

**2. Active-dimension selection**

ORBIT computes a relevance score $s_j$ per feature using either:
- Mutual information $s_j=\mathrm{MI}(X_{\cdot j},y)$, or
- Absolute (cosine) correlation:
$s_j = \dfrac{|\langle Z_j, y_c\rangle|}{\|Z_j\|_2\|y_c\|_2}$
where $Z_j$ is the centered/standardized column and $y_c=y-\bar{y}$.

Keeping the top $K_1$ for dimensions $\mathcal{D}$.

**3. Univariate basis for each $j\in\mathcal{D}$**

Let $x_j^{(d)}$ be the denoised column; $\tilde{x}_j = x_j^{(d)}/\mathrm{sd}(x_j^{(d)}$).
	
1.	Gaussian RBFs with quantile centers $c_{jk}$ and width $w_j$:
$\phi_{jk}^{\text{rbf}}(x) = \exp\!\left(-\frac{(x_j^{(d)}-c_{jk})^2}{2w_j^2+\varepsilon}\right)$.
2.	Trig atoms at frequencies $f\in\{0.5,1,2,3\}:
\sin(f\,\tilde{x}_j),\quad \cos(f\,\tilde{x}_j)$.
3.	Hinge splines at knots $\kappa_{jm}:
\max(0, x_j^{(d)}-\kappa_{jm}),\quad \max(0,\kappa_{jm}-x_j^{(d)})$.

**4. Pairwise interactions (stability selection)**

Generate candidates $z_{ij}=x_i^{(s)}x_j^{(s)}, |x_i^{(s)}-x_j^{(s)}|, \sin(f(x_i+x_j)), \cos(f(x_i-x_j))$ with $f\in\{0.5,1,2\}$. Use multiple subsamples of rows/features; ranked by absolute correlation with $y$; keeping pairs with highest selection frequency (top $K_2$).

**5. Triple interactions**

From indices appearing in the kept pairs, ORBIT enumerate limited triples and keeps the top $K_3$ by absolute correlation:
$\theta_{ijk}=x_i^{(s)}x_j^{(s)}x_k^{(s)}$.

**6. Random Fourier Features (RFF)**

Approximating the RBF kernel using D features, ORBIt estimates $W_{\cdot d}\sim \mathcal{N}(0,\gamma I), b_d\sim \mathrm{Unif}(0,2\pi)$; with set $\gamma$ via the median heuristic on pairwise distances of a subsample. This can be define and represented by:
$\varphi_d(x)=\sqrt{\frac{2}{D}}\cos(W_{\cdot d}^\top x^{(s)}+b_d),\quad d=1,\dots,D$.

**7. Atom (radial) features**

Selects m centers $\{c_a\}$ from rows of $X^{(s)}$. Set $\sigma^2=(0.5\cdot \mathrm{median\ distance})^2$, as defined by:
$\alpha_a(x)=\exp\!\left(-\frac{\|x^{(s)}-c_a\|_2^2}{2\sigma^2+\varepsilon}\right)$.

**8. Concatenation and standardisation**

Form $\Phi(x) = [\,1,\ x^{(s)},\ \phi^{\text{uni}},\ \psi^{\text{pair}},\ \theta^{\text{tri}},\ \varphi^{\text{RFF}},\ \alpha^{\text{atom}}\,]$, the solution standardises columns to obtain $Z$ whilst ensuring the intercept remains unscaled.

## Two-Stage Screening with Whitening

For this section, let $Z\in\mathbb{R}^{n\times D_\Phi}$ be the standardised feature bank for the train dataset.

**1. Stage 1 screening (pre-whiten)**

Computes the centered versions $Z_c$, $y_c$, scoring each column by cosine similarity
$\mathrm{score}$ $k = \frac{|\langle Z{c,\cdot k}, y_c\rangle|}{\|Z_{c,\cdot k}\|_2\|y_c\|_2}$.
Keeping top $k_0=\min(\text{cap}0,\ \lfloor \rho_0 D\Phi\rfloor)$.

**2. Whitening**

With $Z^{(0)}$ as the stage-1 matrix, compute the covariance $C=\tfrac{1}{n} Z^{(0)\top}Z^{(0)}$. With jitter $\epsilon I$, take $C+\epsilon I = LL^\top$ and define whitener $W=L^{-1}$. The whitened design is
$Z^{(w)} = Z^{(0)} W$.

Whitening increased the reliablity of correlation-based screening.

**3. Stage 2 screening (post-whiten)**

Recompute cosine scores on $Z^{(w)}$ vs. $y_c$, keeping $k=\min(\text{cap},\ \lfloor \rho\,\mathrm{cols}(Z^{(w)})\rfloor)$. We denote final bank as $\tilde{Z}\in\mathbb{R}^{n\times k}$.

## Estimation

**1. Linear head**

Fit ElasticNet on $X^{(s)}$:
$\hat{\beta}{\text{lin}}=\arg\min{\beta}\ \tfrac{1}{2n}\|y^{(t)}-X^{(s)}\beta\|_2^2 + \lambda\big(\alpha\|\beta\|1+\tfrac{1-\alpha}{2}\|\beta\|2^2\big)$.

CV is applied on the train folds $(\lambda,\alpha)$, to evaluate the final set of Linear predictions: $\hat{y}^{(t)}{\text{lin}}=X^{(s)}\hat{\beta}{\text{lin}}$.

**2. Nonlinear head**

ElasticNet + bank dropout. 

Repeating R times: the solution randomly drops each column (keeping prob $1-\pi$, note the intercept is never dropped), and fit ElasticNet on the masked $\tilde{Z}$, and average coefficients:

$\bar{w}{\text{EN}} = \frac{1}{R}\sum{r=1}^R w^{(r)}$.

ORBIT includes an optional Huber–ridge refit, to improve robustness of the fit, that restricts to the active set $\mathcal{A} = \{j:|\bar{w}{\text{EN},j}|>\tau\}$ with $\tau$ small (e.g., $10^{-8}$). 

Solving by IRLS:

$w^{(t+1)} = \arg\min_w \ \| \Omega^{1/2}(y^{(t)} - \tilde{Z}\mathcal{A}w)\|_2^2 + \lambda_r\|w\|_2^2$,
where $\Omega=\mathrm{diag}(\omega_i), \omega_i=\min\{1,\ \delta/|r_i|\}, r_i=y^{(t)}i - (\tilde{Z}\mathcal{A}w^{(t)})i$, and $\delta$ is the Huber threshold.

the solution finds $w{\text{nl}}$ that in turn produces nonlinear predictions: $\hat{y}^{(t)}{\text{nl}}=\tilde{Z} w{\text{nl}}$.


**3. Stacking (Hydra approach)**

Here the solution considers combining linear and non-linear heads to produce more reliable predictions. For a given forward validation split, form
$S = \big[\mathbb{1},\ \hat{y}^{(t)}{\text{lin}},\ \hat{y}^{(t)}{\text{nl}},\ \hat{y}^{(t)}{\text{lin}}\odot \hat{y}^{(t)}{\text{nl}}\big]$.

Fit ridge:
$\hat{\theta}=\arg\min_\theta \ \|y^{(t)}{\text{val}}-S{\text{val}}\theta\|_2^2+\lambda_s\|\theta\|2^2$, 
stacking predictions on any set: $\hat{y}^{(t)}{\text{stack}}=S\hat{\theta}$.

The soluution offers a fallback if ridge unavailable, offering a convex blend $\alpha\in[0,1]$ whilst also minimising validation RMSE.

## Conformal Prediction Intervals

Absolute residuals
$e_i = |y_{\text{val},i}-\hat{y}{\text{val},i}|$ are cmputed on the validaiton set (never on train).
For a given miscoverage $\epsilon$ (e.g., $\epsilon=0.1$ for 90%), let $q{1-\epsilon}$ be the empirical $(1-\epsilon)$-quantile of $\{e_i\}$, and for any test $x$,
$\mathrm{PI}{1-\epsilon}(x) = \big[\hat{y}(x)-q{1-\epsilon},\ \hat{y}(x)+q_{1-\epsilon}\big]$.

It is worth noting that for time-series application the recommended implementation would be to use recent validation windows only (e.g., the last 15–25% of the training horizon) for calibration. Potential future extensions to the solutuon might consider weighted or rolling conformal (useful if drifting trends are present in the data).

## Bagging

The currenty implentation offers an optional bagging wrapper to the ORBIT regresssor. Given independent ORBIT instances with different seeds $\{s_b\}$. Average predictions:
$\hat{y}{\text{bag}}=\frac{1}{B}\sum{b=1}^B \hat{y}^{(b)}$.
Aggregate half-widths $q^{(b)}$ by RMS:

$\bar{q}=\sqrt{\frac{1}{B}\sum_{b=1}^B \big(q^{(b)}\big)^2},\quad
\mathrm{PI}=[\hat{y}{\text{bag}}-\bar{q},\ \hat{y}{\text{bag}}+\bar{q}]$.

## Complexity

To summarise the mathematic representation of ORBIT it is important to summarise the areas that lead to high-computational demand. The below will be important to consider before hyperparameter tuning.

Given:
- $n$: train size, $p$: base features, $D_\Phi$: raw bank columns,
- $k_0$: stage-1 kept, $k$: final kept columns,
- $D$: RFF dims, $m$: atom centers.

The dominant costs are:
- Univariate bank: $O(n\,|\mathcal{D}|\,c)$ with small constant $c$ (e.g., 20–60 cols per dim).
- Stability-selection for pairs: $O(R\cdot n\cdot D’^2)$ on subsampled dims $D’\ll p$.
- RFF & atoms: $O(n p D)+O(n m p)$ to compute distances/projections (distances vectorized).
- Whitening: Cholesky on $k_0\times k_0: O(k_0^3)$.
- ElasticNet (CD): $O(n k \cdot \text{passes})$.
- Huber IRLS: ~5–10 iterations, each $O(k^3)$ solve on small systems.

The current imeplmentation attemtps to ensure runtime scales near-linearly in $n$ with moderate constants.

Recommended scaling given a train size of $n$:
- Active dims $K_1=\min(24,\ p)$.
- Pairs $K_2 \approx \min(64,\ \lfloor 2.1\sqrt{n}\rfloor)$.
- Triples $K_3 \approx \min(6,\ \lfloor \sqrt{n}/4\rfloor)$.
- RFF dims $D = \min(320,\ \max(120,\ \lfloor 0.5n\rfloor))$.
- Atoms $m=\min(100,\ \max(40,\ \lfloor 0.24n\rfloor))$.
- Screening caps: $k_0=\min(900,\ \lfloor 0.25 D_\Phi\rfloor), k=\min(1100,\ \lfloor 0.6 k_0\rfloor)$.
- Dropout $\pi\in[0.10,0.20]$; Huber $\delta=1.345\cdot \mathrm{MAD}/0.6745$; ridge $\lambda_r\approx 1$.
- Stacker ridge $\lambda_s\in[10^{-4},10^{-2}]$.
- Validation fraction of train: 15–25% (ensure $≥500$ points for noisy high-frequency data).

---

**Example Hyperparameter Table**

|Component|Parameter|Default / Rule|
|---------|---------|--------------|
|Active dims|K_1|24|
|Pairs|K_2|$\min(64, \lfloor 2.1\sqrt{n}\rfloor)$|
|Triples|K_3|$\min(6, \lfloor \sqrt{n}/4\rfloor)$|
|RFF|D|$\min(320,\max(120,\lfloor 0.5n\rfloor))$|
|Atoms|m|$\min(100,\max(40,\lfloor 0.24n\rfloor))$|
|Screen 1|k_0|$\min(900,\lfloor 0.25D_\Phi\rfloor)$|
|Screen 2|k|$\min(1100,\lfloor 0.6k_0\rfloor)$|
|Dropout|$\pi$|0.10|
|Huber|$\delta$|$1.345\cdot \mathrm{MAD}/0.6745$|
|Ridge refit|$\lambda_r$|1.0|
|Stacker ridge|$\lambda_s$|$10^{-3}$|
|Validation frac||0.15–0.25 of train