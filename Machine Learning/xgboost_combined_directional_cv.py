#!/usr/bin/env python3
"""
XGBoost pipeline for brain-age prediction with a custom loss function.

Key design choices:
  - Stratified cross-validation (outer + inner) so every fold preserves
    the Young / Middle-aged / Old distribution.
  - A custom XGBoost objective that combines TWO weighting schemes:
      1) Directional bias weighting — penalises the model more when it
         over-predicts age for Young subjects or under-predicts for Old
         subjects, pushing predictions toward the true age.
      2) Inverse-frequency weighting — gives more importance to
         under-represented age groups so the model does not ignore them.

Default directional weights:
  alpha_high = 1.5   (heavy penalty for the "wrong" direction)
  alpha_low  = 0.5   (light penalty for the "acceptable" direction)
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable as IterableABC
from pathlib import Path
from typing import Dict, Iterable, List, Literal

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

# ---------------------------------------------------------------------------
# Global constants
# ---------------------------------------------------------------------------
AGE_LABELS = ["Young", "Middle aged", "Old"]  # Expected age-group labels
# Weight applied when prediction error goes in the WRONG direction
ALPHA_HIGH = 1.5
# Weight applied when prediction error goes in the ACCEPTABLE direction
ALPHA_LOW = 0.5
INVERSE_WEIGHT_MODES = ("group", "age")
DEFAULT_XGBOOST_REPORTS_DIR = (
    "/Users/alpmac/CodeWorks/Trento/Dortmund_Vital_Alp_Akova_Clean/"
    "Machine Learning/Test Results/Xgboost Reports"
)


# ===========================================================================
# 1. BUILD A LOOKUP TABLE:  exact age value  -->  age-group label
# ===========================================================================
def _build_age_to_group_map(
    y: pd.Series, groups: pd.Series, target_col: str, group_col: str
) -> Dict[float, str]:
    """
    Create a dictionary that maps each unique age value to its age-group
    label (e.g. 25.0 -> "Young").

    This is needed inside the custom objective so that, given only a
    y_true value, we can look up whether the subject is Young / Middle /
    Old and apply the correct directional weight.

    Raises ValueError if a single age value is associated with more than
    one group (data inconsistency).
    """
    # Pair every age value with its group label
    pairs = pd.DataFrame({target_col: y.values, group_col: groups.values})

    # Safety check: each age value must belong to exactly one group
    nunique = pairs.groupby(target_col)[group_col].nunique()
    inconsistent = nunique[nunique > 1]
    if not inconsistent.empty:
        bad = list(inconsistent.index[:10])
        raise ValueError(
            f"Found ages mapped to multiple groups for ages: {bad}"
        )

    # Return a clean {age_value: group_label} dictionary
    return (
        pairs.drop_duplicates(subset=[target_col, group_col])
        .set_index(target_col)[group_col]
        .to_dict()
    )


def _validate_inverse_weight_mode(mode: str) -> str:
    """Validate and normalise inverse weighting mode."""
    mode_norm = str(mode).strip().lower()
    if mode_norm not in INVERSE_WEIGHT_MODES:
        allowed = ", ".join(INVERSE_WEIGHT_MODES)
        raise ValueError(
            f"inverse_weight_mode must be one of: {allowed}; got '{mode}'"
        )
    return mode_norm


def _normalise_inverse_weight_modes(
    modes: Iterable[str] | None, default_modes: Iterable[str]
) -> List[str]:
    """Convert iterable/None mode input into a validated deduplicated list."""
    if modes is None:
        raw = list(default_modes)
    elif isinstance(modes, (str, bytes)):
        raw = [modes]
    else:
        raw = list(modes)

    if not raw:
        raise ValueError("inverse_weight_modes must contain at least one mode")

    result: List[str] = []
    for mode in raw:
        mode_norm = _validate_inverse_weight_mode(mode)
        if mode_norm not in result:
            result.append(mode_norm)
    return result


def _compute_balanced_inverse_weights(keys: np.ndarray) -> np.ndarray:
    """
    Compute per-sample balanced inverse-frequency weights.

    Formula:
        n_samples / (n_unique_keys * count_of_this_key)
    """
    keys_arr = np.asarray(keys)
    if keys_arr.ndim != 1:
        keys_arr = keys_arr.reshape(-1)
    if keys_arr.size == 0:
        raise ValueError("Cannot compute inverse weights on an empty array")
    if pd.isna(keys_arr).any():
        raise ValueError("Cannot compute inverse weights with missing keys")

    _, inverse_idx, counts = np.unique(
        keys_arr, return_inverse=True, return_counts=True
    )
    n_samples = float(keys_arr.size)
    n_unique = float(len(counts))
    per_key_weight = n_samples / (n_unique * counts.astype(float))
    return per_key_weight[inverse_idx].astype(float, copy=False)


def _path_with_mode_suffix(path: str, mode: str) -> str:
    """Append an inverse-weight mode suffix before a file extension."""
    p = Path(path)
    return str(p.with_name(f"{p.stem}_{mode}{p.suffix}"))


def _ensure_parent_dir(path: str) -> None:
    """Create parent directory for an output path if it does not exist."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def _safe_metric_column_prefix(label: str) -> str:
    """Convert labels like 'Middle aged' to stable column prefixes."""
    return label.strip().lower().replace(" ", "_")


# ===========================================================================
# 2. CUSTOM XGBOOST OBJECTIVE (the heart of the weighting logic)
# ===========================================================================
def make_combined_objective(
    age_to_group: Dict[float, str],
    alpha_high: float = ALPHA_HIGH,
    alpha_low: float = ALPHA_LOW,
    inverse_weight_mode: Literal["group", "age"] = "group",
    young_label: str = "Young",
    old_label: str = "Old",
):
    """
    Return a custom objective function compatible with XGBRegressor.

    The objective computes per-sample gradient and hessian for a
    WEIGHTED mean-squared-error loss, where each sample's weight is:

        weight = directional_weight  *  inverse_frequency_weight

    Directional weight rules (intuition: penalise age-inflation for
    young and age-deflation for old):
      - Young subjects:
            prediction > true age  (over-predicting) -> alpha_high  (punish)
            prediction <= true age (under-predicting) -> alpha_low  (tolerate)
      - Old subjects:
            prediction < true age  (under-predicting) -> alpha_high (punish)
            prediction >= true age (over-predicting)  -> alpha_low  (tolerate)
      - Middle-aged / other: weight = 1.0 (no directional preference)

    Inverse-frequency weight:
        total_samples / (num_groups * count_of_this_group)
      This is the standard sklearn "balanced" class-weight formula applied
      per sample. Keys are selected by inverse_weight_mode:
        - "group": use age_group label frequencies
        - "age":   use exact age value frequencies
    """
    if alpha_high <= 0 or alpha_low <= 0:
        raise ValueError("alpha_high and alpha_low must be positive")
    inverse_weight_mode = _validate_inverse_weight_mode(inverse_weight_mode)

    def objective(y_true, y_pred, sample_weight=None):
        y_true_arr = np.asarray(y_true)
        y_pred_arr = np.asarray(y_pred)

        # residual = prediction - truth  (positive means over-prediction)
        residual = y_pred_arr - y_true_arr

        # Look up each sample's age-group using the precomputed map
        groups = pd.Series(y_true_arr).map(age_to_group).to_numpy()

        # Make sure every sample could be mapped (catches new unseen ages)
        missing = pd.isna(groups)
        if missing.any():
            missing_ages = np.unique(y_true_arr[missing])
            raise ValueError(
                f"Could not map {len(missing_ages)} ages to groups: {missing_ages[:10]}"
            )

        # --- Directional weighting ---
        # Start with neutral weight = 1.0 for all samples
        dir_weights = np.ones_like(residual, dtype=float)

        young_mask = groups == young_label
        old_mask = groups == old_label

        # Young: punish over-prediction (residual > 0), tolerate under-prediction
        dir_weights[young_mask & (residual > 0)] = alpha_high
        dir_weights[young_mask & (residual <= 0)] = alpha_low

        # Old: punish under-prediction (residual < 0), tolerate over-prediction
        dir_weights[old_mask & (residual < 0)] = alpha_high
        dir_weights[old_mask & (residual >= 0)] = alpha_low

        # --- Inverse-frequency weighting ---
        # Frequency keys are fold-local because they are computed from y_true
        # passed by XGBoost for the current training subset.
        inverse_keys = groups if inverse_weight_mode == "group" else y_true_arr
        freq_weights = _compute_balanced_inverse_weights(inverse_keys)

        # --- Combine both weighting schemes ---
        combined = dir_weights * freq_weights

        # If XGBoost (or a manual caller) also passes sample_weight, fold it in
        if sample_weight is not None:
            combined = combined * np.asarray(sample_weight, dtype=float)

        # Gradient and Hessian for weighted MSE:
        #   Loss_i  = 0.5 * w_i * (y_pred - y_true)^2
        #   grad_i  = w_i * (y_pred - y_true)          = combined * residual
        #   hess_i  = w_i                                = combined
        grad = combined * residual
        hess = combined
        return grad, hess

    return objective


# ===========================================================================
# 3. HYPERPARAMETER SEARCH SPACE
# ===========================================================================
def _default_param_dist():
    """
    Define the randomised search space for XGBoost hyperparameters.
    Each entry is a scipy distribution that RandomizedSearchCV samples from.
    """
    return {
        # number of boosting rounds
        "n_estimators": stats.randint(50, 401),
        "learning_rate": stats.loguniform(1e-3, 2e-1),   # step size shrinkage
        # tree depth (controls complexity)
        "max_depth": stats.randint(1, 9),
        # minimum sum of instance weight in a child
        "min_child_weight": stats.loguniform(0.5, 20.0),
        # minimum loss reduction for a split
        "gamma": stats.uniform(0.0, 3.0),
        # row sampling ratio  (range 0.5–1.0)
        "subsample": stats.uniform(0.5, 0.5),
        # column sampling ratio (range 0.5–1.0)
        "colsample_bytree": stats.uniform(0.5, 0.5),
        "reg_alpha": stats.loguniform(1e-4, 10.0),        # L1 regularisation
        "reg_lambda": stats.loguniform(1e-1, 20.0),       # L2 regularisation
    }


# ===========================================================================
# 3b. SHAP-style additive contribution summaries (from pred_contribs)
# ===========================================================================
def _scope_importance_df(
    scope: str, contrib_matrix: np.ndarray, feature_cols: List[str]
) -> pd.DataFrame:
    """Create ranked feature-importance rows for one scope."""
    mean_abs = np.mean(np.abs(contrib_matrix), axis=0)
    mean_signed = np.mean(contrib_matrix, axis=0)

    scope_df = pd.DataFrame(
        {
            "scope": scope,
            "feature": feature_cols,
            "mean_abs_shap": mean_abs,
            "mean_signed_shap": mean_signed,
        }
    )
    scope_df = scope_df.sort_values(
        "mean_abs_shap", ascending=False, kind="mergesort"
    ).reset_index(drop=True)
    scope_df["rank"] = np.arange(1, len(scope_df) + 1, dtype=int)
    return scope_df[
        ["scope", "feature", "mean_abs_shap", "mean_signed_shap", "rank"]
    ]


def _build_shap_importance_table(
    oof_contribs: np.ndarray, feature_cols: List[str], groups_arr: np.ndarray
) -> pd.DataFrame:
    """
    Build long-format SHAP-style importance table:
      - global scope
      - one scope per age group present in AGE_LABELS
    """
    tables = [_scope_importance_df("global", oof_contribs, feature_cols)]
    for group_name in AGE_LABELS:
        mask = groups_arr == group_name
        if mask.any():
            tables.append(
                _scope_importance_df(
                    group_name, oof_contribs[mask], feature_cols)
            )
    return pd.concat(tables, axis=0, ignore_index=True)


def _normalise_alpha_values(
    values: float | Iterable[float] | None, default_value: float, name: str
) -> List[float]:
    """Convert scalar/list alpha inputs to a validated non-empty float list."""
    def _flatten_numeric_items(items):
        for item in items:
            if isinstance(item, (str, bytes)) or np.isscalar(item):
                yield item
            elif isinstance(item, IterableABC):
                yield from _flatten_numeric_items(item)
            else:
                raise TypeError(
                    f"Unsupported item type {type(item).__name__} in {name}"
                )

    if values is None:
        raw_values = [default_value]
    elif isinstance(values, (str, bytes)) or np.isscalar(values):
        raw_values = [float(values)]
    else:
        try:
            raw_values = [float(v) for v in _flatten_numeric_items(values)]
        except TypeError as exc:
            raise ValueError(
                f"{name} must be a float or an iterable of floats "
                "(nested lists are allowed)"
            ) from exc
        except ValueError as exc:
            raise ValueError(
                f"{name} must contain only numeric values "
                "(or nested iterables of numeric values)"
            ) from exc

    if not raw_values:
        raise ValueError(f"{name} must contain at least one value")

    for v in raw_values:
        if not np.isfinite(v):
            raise ValueError(f"{name} contains non-finite value: {v}")
        if v <= 0:
            raise ValueError(
                f"{name} must contain only positive values; found {v}")

    return raw_values


def _run_single_alpha_config(
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    participant_ids: pd.Series,
    feature_cols: List[str],
    age_to_group: Dict[float, str],
    alpha_high: float,
    alpha_low: float,
    inverse_weight_mode: str,
    n_splits: int,
    n_iter: int,
    random_state: int,
    n_jobs: int,
    shap_check_tol: float,
    shap_check_rtol: float,
):
    """Run the full nested CV pipeline for one (alpha_high, alpha_low) pair."""
    objective_fn = make_combined_objective(
        age_to_group,
        alpha_high=alpha_high,
        alpha_low=alpha_low,
        inverse_weight_mode=inverse_weight_mode,
    )
    param_dist = _default_param_dist()

    print(
        f"Running combined objective CV | samples={len(X)} | "
        f"features={len(feature_cols)} | alpha_high={alpha_high}, alpha_low={alpha_low} | "
        f"inverse_weight_mode={inverse_weight_mode}"
    )
    for g in AGE_LABELS:
        c = int((groups == g).sum())
        print(f"  {g}: {c} ({100.0 * c / len(groups):.1f}%)")

    # Array to collect out-of-fold predictions (one per sample)
    oof_preds = np.zeros(len(X), dtype=float)
    oof_contribs = np.zeros((len(X), len(feature_cols)), dtype=float)
    oof_bias = np.zeros(len(X), dtype=float)
    shap_filled = np.zeros(len(X), dtype=bool)

    fold_metrics = []       # Per-fold MAE, RMSE, R2
    fold_best_params = []   # Best hyperparameters chosen in each fold

    # StratifiedKFold ensures each fold has the same Young/Middle/Old ratio
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True,
                          random_state=random_state)

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, groups), start=1):
        print(f"\n===== FOLD {fold}/{n_splits} =====")

        # Split data into training and validation sets for this fold
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        g_train, g_val = groups.iloc[train_idx], groups.iloc[val_idx]

        # Verify inner CV is feasible (need at least 5 samples per group)
        inner_min_count = g_train.value_counts().min()
        if inner_min_count < 5:
            raise ValueError(
                "Inner CV requires at least 5 samples per class; "
                f"found {inner_min_count}"
            )

        # Inner CV: tune hyperparameters on the training fold only
        inner_seed = random_state + fold  # Different seed per fold for variety
        inner_cv = StratifiedKFold(
            n_splits=5, shuffle=True, random_state=inner_seed)

        # Create an XGBRegressor that uses our custom weighted objective
        estimator = xgb.XGBRegressor(
            objective=objective_fn,
            random_state=random_state,
            n_jobs=1,        # Each worker uses 1 core; parallelism is at the search level
            verbosity=0,     # Suppress XGBoost's own logs
        )

        # Randomised search over the hyperparameter space
        search = RandomizedSearchCV(
            estimator=estimator,
            param_distributions=param_dist,
            n_iter=n_iter,                                     # Number of random combos to try
            scoring="neg_mean_absolute_error",                 # Optimise for lowest MAE
            # Stratified inner folds
            cv=list(inner_cv.split(X_train, g_train)),
            random_state=inner_seed,
            n_jobs=n_jobs,                                     # Parallelise across combos
            verbose=1,
        )
        search.fit(X_train, y_train)

        # Predict on the held-out validation fold using the best model
        fold_best_params.append(search.best_params_)
        best_model = search.best_estimator_
        y_pred = best_model.predict(X_val)
        oof_preds[val_idx] = y_pred  # Store predictions in the OOF array

        # OOF SHAP-style contributions from XGBoost native pred_contribs
        dval = xgb.DMatrix(X_val, feature_names=feature_cols)
        booster = best_model.get_booster()
        contrib = booster.predict(dval, pred_contribs=True)
        expected_shape = (len(X_val), len(feature_cols) + 1)
        if contrib.shape != expected_shape:
            raise ValueError(
                f"Unexpected pred_contribs shape {contrib.shape}; "
                f"expected {expected_shape}"
            )

        fold_contrib = contrib[:, :-1]
        fold_bias = contrib[:, -1]
        if shap_filled[val_idx].any():
            raise ValueError(
                "OOF SHAP assignment overlap detected for validation rows")
        oof_contribs[val_idx, :] = fold_contrib
        oof_bias[val_idx] = fold_bias
        shap_filled[val_idx] = True

        margin = booster.predict(dval, output_margin=True).astype(
            np.float64, copy=False)
        recon = (fold_contrib.sum(axis=1) +
                 fold_bias).astype(np.float64, copy=False)
        delta = margin - recon
        max_abs_diff = float(np.max(np.abs(delta)))
        denom = np.maximum(np.abs(margin), 1e-12)
        max_rel_diff = float(np.max(np.abs(delta) / denom))
        additive_ok = np.allclose(
            margin, recon, atol=shap_check_tol, rtol=shap_check_rtol)
        if not additive_ok:
            raise ValueError(
                f"Fold {fold} SHAP additivity check failed: "
                f"max_abs_diff={max_abs_diff:.3e} (atol={shap_check_tol:.3e}), "
                f"max_rel_diff={max_rel_diff:.3e} (rtol={shap_check_rtol:.3e})"
            )
        print(
            f"Fold {fold}: SHAP additivity "
            f"max_abs_diff={max_abs_diff:.3e}, max_rel_diff={max_rel_diff:.3e}"
        )

        # Compute and print per-fold metrics (overall + per age-group)
        fold_mae = mean_absolute_error(y_val, y_pred)
        fold_rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        fold_r2 = r2_score(y_val, y_pred)
        fold_metrics.append(
            {"mae": fold_mae, "rmse": fold_rmse, "r2": fold_r2})
        print(
            f"Fold {fold}: MAE={fold_mae:.2f}, RMSE={fold_rmse:.2f}, R2={fold_r2:.4f}")

        # Per age-group breakdown within this fold
        for g in AGE_LABELS:
            m = g_val.values == g
            if m.sum() > 0:
                g_mae = mean_absolute_error(y_val.values[m], y_pred[m])
                g_bias = float(np.mean(y_pred[m] - y_val.values[m]))
                print(
                    f"  {g}: MAE={g_mae:.2f}, mean_bias={g_bias:+.2f} (n={m.sum()})")

    y_true = y.to_numpy()
    groups_arr = groups.to_numpy()
    if not shap_filled.all():
        missing = int((~shap_filled).sum())
        raise ValueError(f"OOF SHAP matrix has {missing} unfilled rows")

    oof_rmse = np.sqrt(mean_squared_error(y_true, oof_preds))
    oof_mae = mean_absolute_error(y_true, oof_preds)
    oof_r2 = r2_score(y_true, oof_preds)
    oof_mean_bias = float(np.mean(oof_preds - y_true))

    print("\n===== OOF SUMMARY =====")
    print(
        f"RMSE={oof_rmse:.2f}, MAE={oof_mae:.2f}, R2={oof_r2:.4f}, mean_bias={oof_mean_bias:+.2f}"
    )

    # Per age-group OOF breakdown
    oof_group_metrics = {}
    for g in AGE_LABELS:
        m = groups_arr == g
        if m.sum() > 0:
            g_mae = mean_absolute_error(y_true[m], oof_preds[m])
            g_bias = float(np.mean(oof_preds[m] - y_true[m]))
            oof_group_metrics[g] = {
                "mae": float(g_mae),
                "mean_bias": g_bias,
                "n": int(m.sum()),
            }
            print(f"  {g}: MAE={g_mae:.2f}, mean_bias={g_bias:+.2f} (n={m.sum()})")

    # Per-participant error export table (not written to disk yet)
    participant_errors_df = pd.DataFrame(
        {
            "participant_id": participant_ids.to_numpy(),
            "actual_age": y_true,
            "predicted_age": oof_preds,
        }
    )
    participant_errors_df["error"] = (
        participant_errors_df["predicted_age"] -
        participant_errors_df["actual_age"]
    )
    participant_errors_df["absolute_error"] = participant_errors_df["error"].abs(
    )

    # Per-participant SHAP-style contribution table (not written to disk yet)
    shap_feature_cols = [f"shap__{feat}" for feat in feature_cols]
    shap_values_df = pd.DataFrame(oof_contribs, columns=shap_feature_cols)
    shap_values_df.insert(0, "shap_bias_term", oof_bias)
    shap_values_df.insert(0, "age_group", groups_arr)
    shap_values_df.insert(0, "predicted_age", oof_preds)
    shap_values_df.insert(0, "actual_age", y_true)
    shap_values_df.insert(0, "participant_id", participant_ids.to_numpy())

    shap_importance_df = _build_shap_importance_table(
        oof_contribs=oof_contribs,
        feature_cols=feature_cols,
        groups_arr=groups_arr,
    )

    return {
        "oof_preds": oof_preds,               # Out-of-fold predicted ages
        # True ages (same order as oof_preds)
        "y_true": y_true,
        "age_groups": groups_arr,             # Age-group labels (same order)
        "fold_metrics": fold_metrics,         # List of dicts with mae/rmse/r2 per fold
        "fold_best_params": fold_best_params,  # Best hyperparams per fold
        "oof_rmse": oof_rmse,
        "oof_mae": oof_mae,
        "oof_r2": oof_r2,
        "oof_mean_bias": oof_mean_bias,
        "oof_group_metrics": oof_group_metrics,
        "oof_shap_values": oof_contribs,
        "oof_shap_bias": oof_bias,
        "alpha_high": alpha_high,
        "alpha_low": alpha_low,
        "inverse_weight_mode": inverse_weight_mode,
        "participant_errors_df": participant_errors_df,
        "shap_values_df": shap_values_df,
        "shap_importance_df": shap_importance_df,
    }


# ===========================================================================
# 4. MAIN CROSS-VALIDATION PIPELINE
# ===========================================================================
def run_xgboost_combined_cv(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str = "age",
    age_group_col: str = "age_group",
    id_col: str = "participant_id",
    errors_csv: str = "partiicpant_prediciton_errors_xgboost.csv",
    save_shap: bool = True,
    shap_values_csv: str = "xgboost_oof_shap_values.csv",
    shap_importance_csv: str = "xgboost_oof_shap_importance.csv",
    shap_check_tol: float = 1e-6,
    shap_check_rtol: float = 1e-5,
    n_splits: int = 5,
    n_iter: int = 200,
    random_state: int = 42,
    n_jobs: int = -1,
    alpha_high_values: float | Iterable[float] | None = None,
    alpha_low_values: float | Iterable[float] | None = None,
    inverse_weight_mode: str = "group",
):
    """
    Run a nested, age-group-stratified cross-validation pipeline.

    For each (alpha_high, alpha_low) combination:
      - Outer loop  (n_splits folds): produces out-of-fold (OOF) predictions.
      - Inner loop  (5 folds each):   tunes hyperparameters via RandomizedSearchCV.

    At the end, the best alpha combination is chosen by lowest OOF MAE
    (with RMSE then R2 as tie-breakers), and outputs are saved for that
    best run only.
    """

    # -----------------------------------------------------------------------
    # 4a. Input validation
    # -----------------------------------------------------------------------
    if target_col not in df.columns:
        raise ValueError(f"target_col '{target_col}' not found")
    if age_group_col not in df.columns:
        raise ValueError(f"age_group_col '{age_group_col}' not found")
    if id_col not in df.columns:
        raise ValueError(f"id_col '{id_col}' not found")

    missing_features = [c for c in feature_cols if c not in df.columns]
    if missing_features:
        raise ValueError(f"Missing feature columns: {missing_features[:10]}")
    if shap_check_tol <= 0:
        raise ValueError("shap_check_tol must be > 0")
    if shap_check_rtol < 0:
        raise ValueError("shap_check_rtol must be >= 0")
    inverse_weight_mode = _validate_inverse_weight_mode(inverse_weight_mode)

    # -----------------------------------------------------------------------
    # 4b. Prepare feature matrix X, target vector y, and group labels
    # -----------------------------------------------------------------------
    X = df[feature_cols].copy()
    y = df[target_col].copy()
    groups = df[age_group_col].astype(str).copy()
    participant_ids = df[id_col].copy()

    # Drop rows with any missing values in features, target, or group
    mask = X.notna().all(axis=1) & y.notna() & groups.notna()
    X = X.loc[mask]
    y = y.loc[mask]
    groups = groups.loc[mask]
    participant_ids = participant_ids.loc[mask]

    if len(X) == 0:
        raise ValueError("No rows left after NaN filtering")

    # Make sure every group has enough samples for stratified splitting
    group_counts = groups.value_counts()
    if group_counts.min() < n_splits:
        raise ValueError(
            f"Smallest group has {group_counts.min()} samples, below n_splits={n_splits}"
        )

    # -----------------------------------------------------------------------
    # 4c. Build mappings and alpha grids
    # -----------------------------------------------------------------------
    age_to_group = _build_age_to_group_map(
        y, groups, target_col, age_group_col)
    alpha_high_list = _normalise_alpha_values(
        alpha_high_values, ALPHA_HIGH, "alpha_high_values"
    )
    alpha_low_list = _normalise_alpha_values(
        alpha_low_values, ALPHA_LOW, "alpha_low_values"
    )
    alpha_pairs = [
        (alpha_high, alpha_low)
        for alpha_high in alpha_high_list
        for alpha_low in alpha_low_list
    ]

    # -----------------------------------------------------------------------
    # 4d. Print dataset summary before starting
    # -----------------------------------------------------------------------
    print(
        f"Prepared CV dataset | samples={len(X)} | features={len(feature_cols)} | "
        f"alpha_high_values={alpha_high_list} | alpha_low_values={alpha_low_list} | "
        f"combinations={len(alpha_pairs)} | inverse_weight_mode={inverse_weight_mode}"
    )
    for g in AGE_LABELS:
        c = int((groups == g).sum())
        print(f"  {g}: {c} ({100.0 * c / len(groups):.1f}%)")

    alpha_results = []
    best_result = None
    best_key = None

    # -----------------------------------------------------------------------
    # 4e. Evaluate every alpha combination
    # -----------------------------------------------------------------------
    for combo_idx, (alpha_high, alpha_low) in enumerate(alpha_pairs, start=1):
        print(
            f"\n===== ALPHA COMBINATION {combo_idx}/{len(alpha_pairs)} | "
            f"alpha_high={alpha_high}, alpha_low={alpha_low} ====="
        )

        result = _run_single_alpha_config(
            X=X,
            y=y,
            groups=groups,
            participant_ids=participant_ids,
            feature_cols=feature_cols,
            age_to_group=age_to_group,
            alpha_high=alpha_high,
            alpha_low=alpha_low,
            inverse_weight_mode=inverse_weight_mode,
            n_splits=n_splits,
            n_iter=n_iter,
            random_state=random_state,
            n_jobs=n_jobs,
            shap_check_tol=shap_check_tol,
            shap_check_rtol=shap_check_rtol,
        )

        alpha_results.append(
            {
                "alpha_high": alpha_high,
                "alpha_low": alpha_low,
                "inverse_weight_mode": inverse_weight_mode,
                "oof_mae": result["oof_mae"],
                "oof_rmse": result["oof_rmse"],
                "oof_r2": result["oof_r2"],
                "oof_mean_bias": result["oof_mean_bias"],
            }
        )

        candidate_key = (
            float(result["oof_mae"]),      # lower is better
            float(result["oof_rmse"]),     # lower is better
            -float(result["oof_r2"]),      # higher is better
        )
        if best_key is None or candidate_key < best_key:
            best_key = candidate_key
            best_result = result

    if best_result is None:
        raise RuntimeError("No alpha combination was evaluated")

    alpha_results_df = pd.DataFrame(alpha_results).sort_values(
        by=["oof_mae", "oof_rmse", "oof_r2"],
        ascending=[True, True, False],
        kind="mergesort",
    ).reset_index(drop=True)

    print("\n===== ALPHA SEARCH SUMMARY (sorted by OOF MAE) =====")
    for _, row in alpha_results_df.iterrows():
        print(
            f"alpha_high={row['alpha_high']}, alpha_low={row['alpha_low']} -> "
            f"MAE={row['oof_mae']:.2f}, RMSE={row['oof_rmse']:.2f}, "
            f"R2={row['oof_r2']:.4f}, mean_bias={row['oof_mean_bias']:+.2f}"
        )

    print("\n===== BEST ALPHA COMBINATION =====")
    print(
        f"mode={best_result['inverse_weight_mode']} | "
        f"alpha_high={best_result['alpha_high']}, alpha_low={best_result['alpha_low']} | "
        f"OOF MAE={best_result['oof_mae']:.2f}, RMSE={best_result['oof_rmse']:.2f}, "
        f"R2={best_result['oof_r2']:.4f}, mean_bias={best_result['oof_mean_bias']:+.2f}"
    )

    # Save outputs from the best alpha run only.
    participant_errors_df = best_result["participant_errors_df"]
    shap_values_df = best_result["shap_values_df"]
    shap_importance_df = best_result["shap_importance_df"]

    _ensure_parent_dir(errors_csv)
    participant_errors_df.to_csv(errors_csv, index=False)
    print(f"\nSaved participant errors to: {errors_csv}")

    if save_shap:
        _ensure_parent_dir(shap_values_csv)
        _ensure_parent_dir(shap_importance_csv)
        shap_values_df.to_csv(shap_values_csv, index=False)
        shap_importance_df.to_csv(shap_importance_csv, index=False)
        print(f"Saved OOF SHAP-style values to: {shap_values_csv}")
        print(f"Saved OOF SHAP-style importance to: {shap_importance_csv}")

    # -----------------------------------------------------------------------
    # 4f. Return best run + alpha-search summary
    # -----------------------------------------------------------------------
    payload = dict(best_result)
    payload["participant_errors_csv"] = errors_csv
    payload["shap_values_csv"] = shap_values_csv if save_shap else None
    payload["shap_importance_csv"] = shap_importance_csv if save_shap else None
    payload["alpha_high_values"] = alpha_high_list
    payload["alpha_low_values"] = alpha_low_list
    payload["alpha_search_results"] = alpha_results_df
    payload["best_alpha_high"] = float(best_result["alpha_high"])
    payload["best_alpha_low"] = float(best_result["alpha_low"])
    payload["inverse_weight_mode"] = inverse_weight_mode
    return payload


def run_inverse_weight_mode_comparison(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str = "age",
    age_group_col: str = "age_group",
    id_col: str = "participant_id",
    errors_csv: str = "partiicpant_prediciton_errors_xgboost.csv",
    save_shap: bool = True,
    shap_values_csv: str = "xgboost_oof_shap_values.csv",
    shap_importance_csv: str = "xgboost_oof_shap_importance.csv",
    comparison_csv: str = "xgboost_inverse_weight_mode_comparison.csv",
    inverse_weight_modes: Iterable[str] | None = None,
    shap_check_tol: float = 1e-6,
    shap_check_rtol: float = 1e-5,
    n_splits: int = 5,
    n_iter: int = 200,
    random_state: int = 42,
    n_jobs: int = -1,
    alpha_high_values: float | Iterable[float] | None = None,
    alpha_low_values: float | Iterable[float] | None = None,
):
    """
    Compare inverse-frequency weighting modes under identical CV settings.

    Ranking rule:
      1) Lower OOF MAE
      2) Lower OOF RMSE
      3) Higher OOF R2
    """
    modes = _normalise_inverse_weight_modes(
        inverse_weight_modes, default_modes=INVERSE_WEIGHT_MODES
    )
    mode_payloads = {}
    comparison_rows = []

    print(
        "\n===== INVERSE WEIGHT MODE COMPARISON =====\n"
        f"Modes: {modes}"
    )

    for mode in modes:
        print(f"\n===== RUNNING MODE: {mode} =====")
        mode_errors_csv = _path_with_mode_suffix(errors_csv, mode)
        mode_shap_values_csv = _path_with_mode_suffix(shap_values_csv, mode)
        mode_shap_importance_csv = _path_with_mode_suffix(shap_importance_csv, mode)

        mode_result = run_xgboost_combined_cv(
            df=df,
            feature_cols=feature_cols,
            target_col=target_col,
            age_group_col=age_group_col,
            id_col=id_col,
            errors_csv=mode_errors_csv,
            save_shap=save_shap,
            shap_values_csv=mode_shap_values_csv,
            shap_importance_csv=mode_shap_importance_csv,
            shap_check_tol=shap_check_tol,
            shap_check_rtol=shap_check_rtol,
            n_splits=n_splits,
            n_iter=n_iter,
            random_state=random_state,
            n_jobs=n_jobs,
            alpha_high_values=alpha_high_values,
            alpha_low_values=alpha_low_values,
            inverse_weight_mode=mode,
        )
        mode_payloads[mode] = mode_result

        row = {
            "inverse_weight_mode": mode,
            "oof_mae": mode_result["oof_mae"],
            "oof_rmse": mode_result["oof_rmse"],
            "oof_r2": mode_result["oof_r2"],
            "oof_mean_bias": mode_result["oof_mean_bias"],
            "best_alpha_high": mode_result["best_alpha_high"],
            "best_alpha_low": mode_result["best_alpha_low"],
            "participant_errors_csv": mode_result["participant_errors_csv"],
            "shap_values_csv": mode_result["shap_values_csv"],
            "shap_importance_csv": mode_result["shap_importance_csv"],
        }
        for group_name in AGE_LABELS:
            group_metrics = mode_result["oof_group_metrics"].get(group_name)
            prefix = _safe_metric_column_prefix(group_name)
            if group_metrics is not None:
                row[f"{prefix}_mae"] = group_metrics["mae"]
                row[f"{prefix}_mean_bias"] = group_metrics["mean_bias"]
                row[f"{prefix}_n"] = group_metrics["n"]
            else:
                row[f"{prefix}_mae"] = np.nan
                row[f"{prefix}_mean_bias"] = np.nan
                row[f"{prefix}_n"] = 0
        comparison_rows.append(row)

    comparison_results_df = pd.DataFrame(comparison_rows).sort_values(
        by=["oof_mae", "oof_rmse", "oof_r2"],
        ascending=[True, True, False],
        kind="mergesort",
    ).reset_index(drop=True)
    _ensure_parent_dir(comparison_csv)
    comparison_results_df.to_csv(comparison_csv, index=False)

    winner_mode = str(comparison_results_df.loc[0, "inverse_weight_mode"])
    winner_result = mode_payloads[winner_mode]

    print("\n===== INVERSE MODE COMPARISON SUMMARY (sorted by OOF MAE) =====")
    for _, row in comparison_results_df.iterrows():
        print(
            f"mode={row['inverse_weight_mode']} -> "
            f"MAE={row['oof_mae']:.2f}, RMSE={row['oof_rmse']:.2f}, "
            f"R2={row['oof_r2']:.4f}, mean_bias={row['oof_mean_bias']:+.2f}"
        )
    print(f"\nWinner mode: {winner_mode}")
    print(f"Saved inverse-weight comparison to: {comparison_csv}")

    return {
        "comparison_results_df": comparison_results_df,
        "comparison_csv": comparison_csv,
        "winner_mode": winner_mode,
        "mode_payloads": mode_payloads,
        "best_mode_result": winner_result,
    }


# ===========================================================================
# 5. COMMAND-LINE INTERFACE
# ===========================================================================
def parse_args():
    """Parse command-line arguments for standalone execution."""
    parser = argparse.ArgumentParser(
        description=(
            "Age-group stratified XGBoost CV with combined directional + "
            "inverse-frequency objective (supports group vs age inverse weighting comparison)."
        )
    )
    parser.add_argument(
        "--data-csv",
        default="/Users/alpmac/CodeWorks/Trento/Dortmund_Vital_Alp_Akova_Clean/Data/merged_df_connectivity_power.csv",
        help="Path to the main dataset CSV (must contain features, age, and age_group columns)",
    )
    parser.add_argument(
        "--features-csv",
        default="/Users/alpmac/CodeWorks/Trento/Dortmund_Vital_Alp_Akova_Clean/Machine Learning/Test Results/globally_stable_features.csv",
        help="CSV with a single column named 'feature' listing which features to use",
    )
    parser.add_argument(
        "--use-all-features",
        type=int,
        choices=[0, 1],
        default=0,
        help=(
            "Feature mode switch. 0 = use --features-csv (default), "
            "1 = ignore --features-csv and use all numeric columns from --data-csv "
            "except target/age_group/id columns."
        ),
    )
    parser.add_argument("--target-col", default="age")
    parser.add_argument("--age-group-col", default="age_group")
    parser.add_argument("--id-col", default="participant_id")
    parser.add_argument(
        "--errors-csv",
        default=f"{DEFAULT_XGBOOST_REPORTS_DIR}/partiicpant_prediciton_errors_xgboost_global_features.csv",
        help="Path for per-participant OOF error CSV output",
    )
    parser.add_argument(
        "--save-shap",
        type=int,
        choices=[0, 1],
        default=1,
        help="Whether to save SHAP-style outputs from pred_contribs (1=yes, 0=no)",
    )
    parser.add_argument(
        "--shap-values-csv",
        default=f"{DEFAULT_XGBOOST_REPORTS_DIR}/xgboost_oof_shap_values.csv",
        help="Path for OOF per-sample SHAP-style additive contributions CSV",
    )
    parser.add_argument(
        "--shap-importance-csv",
        default=f"{DEFAULT_XGBOOST_REPORTS_DIR}/xgboost_oof_shap_importance.csv",
        help="Path for global + by-group mean(|SHAP|) importance CSV",
    )
    parser.add_argument(
        "--shap-check-tol",
        type=float,
        default=1e-6,
        help="Maximum tolerated additivity error for pred_contribs reconstruction",
    )
    parser.add_argument(
        "--shap-check-rtol",
        type=float,
        default=1e-5,
        help="Relative tolerance used with --shap-check-tol in additivity checks",
    )
    parser.add_argument(
        "--alpha-high-values",
        type=float,
        nargs="+",
        default=[ALPHA_HIGH],
        help=(
            "One or more alpha_high values to evaluate. Example: "
            "--alpha-high-values 1.5 2.0 2.5"
        ),
    )
    parser.add_argument(
        "--alpha-low-values",
        type=float,
        nargs="+",
        default=[ALPHA_LOW],
        help=(
            "One or more alpha_low values to evaluate. Example: "
            "--alpha-low-values 0.3 0.5 0.7"
        ),
    )
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--n-iter", type=int, default=200)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument(
        "--inverse-weight-modes",
        nargs="+",
        choices=list(INVERSE_WEIGHT_MODES),
        default=None,
        help=(
            "Inverse-frequency weighting modes to run. "
            "Supported: group age. "
            "If omitted: legacy single run defaults to 'group'; "
            "comparison mode defaults to 'group age'."
        ),
    )
    parser.add_argument(
        "--run-comparison",
        type=int,
        choices=[0, 1],
        default=0,
        help=(
            "Run a cross-mode comparison (1) or a single-mode run (0). "
            "In comparison mode, results are ranked by OOF MAE then RMSE then R2."
        ),
    )
    parser.add_argument(
        "--comparison-csv",
        default=f"{DEFAULT_XGBOOST_REPORTS_DIR}/xgboost_inverse_weight_mode_comparison.csv",
        help="Path for inverse-weight mode comparison summary CSV",
    )
    return parser.parse_args()


def main():
    """Entry point: load data, choose feature mode, run the CV pipeline."""
    args = parse_args()

    # Load the full dataset
    df = pd.read_csv(args.data_csv)

    if bool(args.use_all_features):
        excluded = {args.target_col, args.age_group_col, args.id_col}
        feature_cols = [
            c for c in df.columns
            if c not in excluded and pd.api.types.is_numeric_dtype(df[c])
        ]
        skipped_non_numeric = [
            c for c in df.columns
            if c not in excluded and not pd.api.types.is_numeric_dtype(df[c])
        ]
        if not feature_cols:
            raise ValueError(
                "No usable numeric feature columns found in --data-csv after excluding "
                f"{sorted(excluded)}"
            )
        print(
            f"Feature mode: all features from dataset | selected {len(feature_cols)} "
            "numeric columns"
        )
        if skipped_non_numeric:
            preview = skipped_non_numeric[:10]
            suffix = "..." if len(skipped_non_numeric) > 10 else ""
            print(
                f"Skipped {len(skipped_non_numeric)} non-numeric columns: "
                f"{preview}{suffix}"
            )
    else:
        # Load the list of selected features to use
        feature_df = pd.read_csv(args.features_csv)
        if "feature" not in feature_df.columns:
            raise ValueError("features CSV must contain a 'feature' column")
        feature_cols = feature_df["feature"].tolist()
        print(
            f"Feature mode: pre-selected features CSV | selected {len(feature_cols)} columns"
        )

    if bool(args.run_comparison):
        comparison_modes = _normalise_inverse_weight_modes(
            args.inverse_weight_modes, default_modes=INVERSE_WEIGHT_MODES
        )
        run_inverse_weight_mode_comparison(
            df=df,
            feature_cols=feature_cols,
            target_col=args.target_col,
            age_group_col=args.age_group_col,
            id_col=args.id_col,
            errors_csv=args.errors_csv,
            save_shap=bool(args.save_shap),
            shap_values_csv=args.shap_values_csv,
            shap_importance_csv=args.shap_importance_csv,
            comparison_csv=args.comparison_csv,
            inverse_weight_modes=comparison_modes,
            shap_check_tol=args.shap_check_tol,
            shap_check_rtol=args.shap_check_rtol,
            n_splits=args.n_splits,
            n_iter=args.n_iter,
            random_state=args.random_state,
            n_jobs=args.n_jobs,
            alpha_high_values=args.alpha_high_values,
            alpha_low_values=args.alpha_low_values,
        )
    else:
        single_modes = _normalise_inverse_weight_modes(
            args.inverse_weight_modes, default_modes=["group"]
        )
        if len(single_modes) != 1:
            raise ValueError(
                "Single-run mode expects exactly one inverse weighting mode. "
                "Use --run-comparison 1 to evaluate multiple modes."
            )
        run_xgboost_combined_cv(
            df=df,
            feature_cols=feature_cols,
            target_col=args.target_col,
            age_group_col=args.age_group_col,
            id_col=args.id_col,
            errors_csv=args.errors_csv,
            save_shap=bool(args.save_shap),
            shap_values_csv=args.shap_values_csv,
            shap_importance_csv=args.shap_importance_csv,
            shap_check_tol=args.shap_check_tol,
            shap_check_rtol=args.shap_check_rtol,
            n_splits=args.n_splits,
            n_iter=args.n_iter,
            random_state=args.random_state,
            n_jobs=args.n_jobs,
            alpha_high_values=args.alpha_high_values,
            alpha_low_values=args.alpha_low_values,
            inverse_weight_mode=single_modes[0],
        )


if __name__ == "__main__":
    main()
