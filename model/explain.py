import numpy as np
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance
import shap

def shap_callables(orbit):
    """
    Build callables for SHAP explanations of an ORBIT model.

    Parameters
    ----------
    orbit : ORBITRegressor or ORBITBaggedRegressor
        Fitted model exposing `.predict` and `.predict_components`.

    Returns
    -------
    (f_full, f_lin, f_nl) : tuple[callable, callable, callable]
        Functions mapping X → predictions for the full model,
        the linear head, and the nonlinear bank head (on target scale).
    """
    
    f_full = lambda X: orbit.predict(np.asarray(X, float))
    f_lin  = lambda X: orbit.predict_components(np.asarray(X, float),
                                                on_target_scale=True,
                                                include_stack=False)["y_lin"]
    f_nl   = lambda X: orbit.predict_components(np.asarray(X, float),
                                                on_target_scale=True,
                                                include_stack=False)["y_nl"]
    return f_full, f_lin, f_nl

def beeswarm_one(sv, title, width=6, height=4):
    """
    Convenience wrapper to draw a SHAP beeswarm plot.

    Parameters
    ----------
    sv : shap.Explanation
        SHAP values produced by a SHAP explainer.
    title : str
        Plot title.
    width, height : float, default=(6, 4)
        Figure size in inches.

    Returns
    -------
    None
        Displays the plot with tight layout.
    """
    shap.plots.beeswarm(sv, max_display=20, show=False, plot_size=None)
    plt.gcf().set_size_inches(width, height)
    plt.title(title)
    plt.tight_layout()
    plt.show()


def perm_importance_callable(f, X, y, n_repeats=10, random_state=42):
    """
    Compute permutation importance for any prediction callable.

    Parameters
    ----------
    f : callable
        Function mapping X → ŷ.
    X : ndarray of shape (n_samples, n_features)
        Inputs to evaluate on.
    y : ndarray of shape (n_samples,)
        Targets for scoring.
    n_repeats : int, default=10
        Number of shuffles per feature.
    random_state : int, default=42
        Seed.

    Returns
    -------
    sklearn.inspection._PermutationImportance
        Result object with importances and per-feature statistics.
    """
    class _F:
        def __init__(self, f): self.f = f
        def fit(self, X, y): return self
        def predict(self, X): return self.f(X)
    est = _F(f).fit(None, None)
    r = permutation_importance(est, X, y, n_repeats=n_repeats,
                               random_state=random_state, n_jobs=-1)
    return r