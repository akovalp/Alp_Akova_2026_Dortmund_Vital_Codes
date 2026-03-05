# XGBoost Age Prediction with Custom Directional Objective: Methodological Details

## 1. Overview

This document describes the methodology implemented in `xgboost_combined_directional_cv_sensor.py` for predicting chronological age from EEG-derived sensor-space features using XGBoost with a custom loss function. The pipeline is positioned as the **second stage** of a two-stage analysis: it receives the stable feature subset identified by the Sparse Group LASSO (SGL) pipeline (documented in `methods_ml_age.md`) and trains a nonlinear boosted-tree regressor on that curated feature set. By accepting pre-selected features rather than the full feature matrix, the XGBoost model benefits from the SGL's theoretically grounded feature selection while adding the representational power of gradient-boosted trees.

### 1.1 Position in the Analysis Pipeline

```
Stage 1: SGL Feature Selection (ROI_Based_Feature_Selection_ML_age.py)
    │
    │  Output: globally_stable_features_sensor.csv
    │          (features stable across ≥80% of folds and multiple seeds)
    │
    ▼
Stage 2: XGBoost Age Prediction (xgboost_combined_directional_cv_sensor.py)
    │
    │  Input:  merged_df_connectivity_power_sensor.csv  (full feature matrix)
    │          globally_stable_features_sensor.csv       (SGL-selected feature names)
    │
    │  Output: participant prediction errors, SHAP importance tables
```

The SGL pipeline (Stage 1) performs embedded feature selection on a combined source-space power and PLI connectivity feature matrix, identifying features that are consistently selected across repeated nested cross-validation runs and multiple random seeds. The XGBoost pipeline (Stage 2) then loads only those stable features by name, constructs a nonlinear predictive model, and provides interpretable SHAP-style contribution estimates for each feature.

### 1.2 Primary Objectives

- Predict chronological age from SGL-selected EEG features using a nonlinear gradient-boosted tree model
- Reduce regression-to-mean bias for extreme age groups (Young, Old) through directional loss weighting
- Address class imbalance across age groups through inverse-frequency sample weighting
- Provide out-of-fold (OOF) SHAP-style feature contribution estimates for interpretability
- Enable data-driven comparison of directional weight configurations and inverse-weighting strategies

### 1.3 Motivation for XGBoost After SGL

The SGL produces a sparse linear model whose coefficients are globally interpretable but whose expressiveness is inherently limited to linear relationships. EEG-derived features — particularly connectivity measures and power ratios — may exhibit nonlinear relationships with age (e.g., alpha power peaking in early adulthood and declining thereafter, or frontal theta showing non-monotonic trajectory across the lifespan). XGBoost captures these interactions automatically via tree-based splitting, while the SGL's pre-selection ensures the feature space is small enough to avoid overfitting despite the added model complexity.

---

## 2. Input Data and Feature Loading

### 2.1 Primary Dataset

**Default path**:
```
merged_df_connectivity_power_sensor.csv
```

**Required columns**:

| Column | Type | Description |
|---|---|---|
| `participant_id` | string | BIDS-compatible participant identifier |
| `age` | numeric | Chronological age in years (prediction target) |
| `age_group` | categorical | `Young`, `Middle aged`, or `Old` |
| Feature columns | numeric | EEG-derived connectivity and power features |

### 2.2 Feature Selection Mode

The pipeline supports two feature-loading modes, controlled via `--use-all-features`:

#### Mode 0 (Default): SGL-Derived Stable Features

```bash
--use-all-features 0 --features-csv globally_stable_features_sensor.csv
```

The features CSV contains a single column named `feature` listing all feature names that survived the SGL stability selection procedure. These are loaded by name and matched against the columns of the primary dataset.

**Expected source**: `globally_stable_features_sensor.csv` is the output of the SGL multi-seed stability analysis, containing features with a global selection frequency ≥ the configured stability threshold (default: 80%). Only features that were consistently selected across multiple random seeds and repeated cross-validation folds appear in this file.

**Rationale**: By restricting XGBoost to SGL-stable features, the pipeline benefits from:
1. Reduced dimensionality, lowering the risk of overfitting
2. Neurophysiologically interpretable features with established age-predictiveness
3. Avoidance of redundant or noisy features that the SGL pruned

#### Mode 1: All Numeric Features

```bash
--use-all-features 1
```

All numeric columns in the primary dataset are used, excluding `age`, `age_group`, and `participant_id`. Non-numeric columns are skipped with a warning. This mode is intended for exploratory analyses or ablation studies comparing SGL-selected versus full-feature performance.

### 2.3 Missing Data Handling

Before model fitting, rows containing any NaN values in features, the target column, or the age-group column are **dropped**:

```python
mask = X.notna().all(axis=1) & y.notna() & groups.notna()
```

If no rows remain after filtering, a `ValueError` is raised. A minimum of `n_splits` participants per age group is required for stratified splitting.

### 2.4 Age-to-Group Mapping

`_build_age_to_group_map()` constructs a dictionary `{age_value: group_label}` from the training data. This lookup table is used inside the custom objective function to determine each sample's age group from its age value alone — necessary because XGBoost's custom objective receives only `y_true` and `y_pred` arrays, without access to additional metadata columns.

**Validation**: If any age value maps to more than one group label (data inconsistency), a `ValueError` is raised before any model fitting begins.

---

## 3. Custom Weighted Objective Function

### 3.1 Design Rationale

Standard MSE loss treats all prediction errors symmetrically: a 10-year overprediction for a young participant is penalized identically to a 10-year underprediction for the same participant. In the context of brain-age research, however, the direction of error carries distinct scientific meaning:

- **Young participants** predicted to be older than they are: the model inflates their apparent brain age, which may lead to incorrect conclusions about accelerated aging
- **Old participants** predicted to be younger than they are: the model deflates their apparent brain age, masking genuine aging effects

Furthermore, if age groups are unequally represented (as is common in lifespan studies), a standard model tends to ignore minority groups and regresses predictions toward the majority group's mean.

The custom objective addresses both problems simultaneously by combining **directional bias weighting** with **inverse-frequency weighting**.

### 3.2 Mathematical Formulation

The objective implements a **weighted MSE** loss where each sample's contribution is scaled by a combined weight:

```
Loss_i = 0.5 · w_i · (ŷ_i − y_i)²

grad_i = w_i · (ŷ_i − y_i)      [gradient]
hess_i = w_i                      [hessian]
```

Where the combined weight is:

```
w_i = directional_weight_i × inverse_frequency_weight_i
```

### 3.3 Directional Weighting

The directional weight `w_dir` penalizes errors that go in the "wrong" direction based on the participant's age group:

| Age Group | Error Direction | Weight |
|---|---|---|
| Young | Overprediction (ŷ > y, residual > 0) | `alpha_high` (heavy penalty) |
| Young | Underprediction (ŷ ≤ y, residual ≤ 0) | `alpha_low` (light penalty) |
| Old | Underprediction (ŷ < y, residual < 0) | `alpha_high` (heavy penalty) |
| Old | Overprediction (ŷ ≥ y, residual ≥ 0) | `alpha_low` (light penalty) |
| Middle aged | Any direction | 1.0 (neutral) |

**Default values**:
- `alpha_high = 1.5` — predictions that go in the "wrong" direction receive 1.5× the standard gradient
- `alpha_low = 0.5` — predictions that go in the "acceptable" direction receive only 0.5× the gradient

**Intuition**: For young participants, the model is more harshly penalized when it over-predicts (making them appear older than they are) than when it under-predicts. This asymmetric gradient pushes the optimizer to avoid overestimating young participants' ages. The reverse applies for old participants. Middle-aged participants are treated symmetrically because there is no a priori directional bias concern.

**Implementation**:
```python
dir_weights = np.ones_like(residual)

young_mask = groups == "Young"
old_mask   = groups == "Old"

dir_weights[young_mask & (residual >  0)] = alpha_high
dir_weights[young_mask & (residual <= 0)] = alpha_low
dir_weights[old_mask   & (residual <  0)] = alpha_high
dir_weights[old_mask   & (residual >= 0)] = alpha_low
```

### 3.4 Inverse-Frequency Weighting

Inverse-frequency weights compensate for age-group imbalance by assigning higher gradient magnitudes to under-represented groups:

```
freq_weight_i = n_samples / (n_unique_keys × count(key_i))
```

This is the same formula as scikit-learn's `class_weight="balanced"`, applied per sample. Weights are normalized such that the sum of weights equals `n_samples`.

**Two keying modes** control which frequency is inverted:

| Mode | Key per Sample | Effect |
|---|---|---|
| `group` | Age-group label (`Young`, `Middle aged`, `Old`) | Balances at the group level |
| `age` | Exact chronological age value | Balances at the individual age level; rarer ages get higher weight |

**Rationale for `group` mode (default)**: Groups with few participants (e.g., if "Old" participants are underrepresented) are upweighted as a block, which is the most direct correction for group-level imbalance.

**Rationale for `age` mode**: If the age distribution within a group is uneven (e.g., many 25-year-olds but few 22-year-olds within the "Young" group), exact-age weighting provides finer-grained correction. This can reduce bias for rare ages but may increase variance.

### 3.5 Combined Weight and Gradient

The two weighting schemes are multiplicatively combined:

```python
combined = dir_weights × freq_weights
grad = combined × residual
hess = combined
```

If XGBoost or an external caller also provides `sample_weight`, it is folded in multiplicatively:
```python
combined = combined × sample_weight
```

**Note**: The hessian equals the combined weight (constant per sample), which is the correct Hessian for a quadratic loss. This ensures that XGBoost's tree-building algorithm receives accurate second-order information for split selection.

### 3.6 Objective Factory Pattern

The objective is constructed via `make_combined_objective()`, which returns a closure capturing the `age_to_group` map and weight parameters. This factory pattern is necessary because XGBoost's custom objective API accepts only `(y_true, y_pred)` arguments — the closure provides access to the group lookup table without relying on global state.

---

## 4. Hyperparameter Search Space

XGBoost hyperparameters are tuned via `RandomizedSearchCV` over the following distributions:

| Hyperparameter | Distribution | Range | Description |
|---|---|---|---|
| `n_estimators` | `randint` | [50, 400] | Number of boosting rounds |
| `learning_rate` | `loguniform` | [1×10⁻³, 0.2] | Shrinkage applied to each tree's contribution |
| `max_depth` | `randint` | [1, 8] | Maximum tree depth; controls model complexity |
| `min_child_weight` | `loguniform` | [0.5, 20.0] | Minimum sum of instance weight in a leaf node |
| `gamma` | `uniform` | [0, 3] | Minimum loss reduction required for a split |
| `subsample` | `uniform` | [0.5, 1.0] | Fraction of samples drawn per tree |
| `colsample_bytree` | `uniform` | [0.5, 1.0] | Fraction of features sampled per tree |
| `reg_alpha` | `loguniform` | [1×10⁻⁴, 10] | L1 regularization on leaf weights |
| `reg_lambda` | `loguniform` | [0.1, 20] | L2 regularization on leaf weights |

**Sampling strategy**: `RandomizedSearchCV` draws `n_iter = 200` random combinations from the Cartesian product of these distributions. Log-uniform distributions are used for scale-sensitive parameters (`learning_rate`, `min_child_weight`, `reg_alpha`, `reg_lambda`) so that small and large values are sampled with equal probability in log-space.

**Scoring**: `neg_mean_absolute_error` — hyperparameter combinations are ranked by MAE, consistent with the SGL pipeline.

**Parallelization**: `n_jobs=-1` within `RandomizedSearchCV` spreads the 200 candidate evaluations across all available CPU cores. Each individual XGBoost fit uses `n_jobs=1` to avoid nested parallelism conflicts.

---

## 5. Nested Cross-Validation Framework

### 5.1 Design Overview

The pipeline implements a **two-level nested cross-validation** framework:

```
Outer CV (n_splits = 5, StratifiedKFold)
│
├── Fold 1: Train on 80% → Inner CV → Best params → Predict on 20%
├── Fold 2: Train on 80% → Inner CV → Best params → Predict on 20%
├── Fold 3: Train on 80% → Inner CV → Best params → Predict on 20%
├── Fold 4: Train on 80% → Inner CV → Best params → Predict on 20%
└── Fold 5: Train on 80% → Inner CV → Best params → Predict on 20%
         └── Inner CV (5 folds, StratifiedKFold, 200 random configs)
```

### 5.2 Outer Cross-Validation

**Method**: `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`

- Stratification variable: `age_group` (Young / Middle aged / Old)
- Ensures that every outer fold contains a representative proportion of each age group
- Each test fold contains exactly 20% of the dataset (80/20 split)
- `random_state=42` ensures reproducible fold assignments

**Minimum sample requirement**: Before starting, the pipeline verifies that the smallest age group contains at least `n_splits` participants. If not, a `ValueError` is raised.

### 5.3 Inner Cross-Validation

For each outer fold, `RandomizedSearchCV` tunes hyperparameters using a 5-fold stratified inner CV on the outer training set:

```python
inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state + fold)
```

**Seed variation**: The inner CV uses `random_state + fold` as its seed, so each outer fold has a distinct inner CV partitioning. This reduces the risk of systematic bias from any particular inner fold configuration.

**Minimum inner sample requirement**: The pipeline checks that the smallest age group in the training set has at least 5 participants before fitting the inner CV, raising a `ValueError` if not.

### 5.4 Out-of-Fold Prediction Assembly

Each participant appears in exactly one outer test fold. The XGBoost model trained on the corresponding outer training set predicts that participant's age, and the prediction is stored in a pre-allocated array:

```python
oof_preds = np.zeros(len(X), dtype=float)
oof_preds[val_idx] = best_model.predict(X_val)
```

After all folds complete, `oof_preds` contains one prediction per participant, assembled without any data leakage (each prediction was made by a model that never saw that participant during training or hyperparameter tuning).

### 5.5 Per-Fold Diagnostics

After each outer fold, the pipeline reports:
- **Overall fold metrics**: MAE, RMSE, R²
- **Per age-group breakdown**: MAE and mean signed bias (predicted − actual) for each of Young, Middle aged, Old within that fold's test set

The mean signed bias reveals whether the directional weighting is correcting the expected regression-to-mean pattern (Young: negative or near-zero bias; Old: positive or near-zero bias after correction).

### 5.6 OOF Summary Metrics

After all folds complete, the following metrics are computed on the full OOF prediction array:

| Metric | Formula | Interpretation |
|---|---|---|
| OOF MAE | `mean_absolute_error(y, oof_preds)` | Primary performance metric; years |
| OOF RMSE | `sqrt(mean_squared_error(y, oof_preds))` | Penalizes large errors more than MAE |
| OOF R² | `r2_score(y, oof_preds)` | Proportion of age variance explained |
| Mean OOF bias | `mean(oof_preds − y)` | Global systematic over/underprediction |

Per age-group OOF metrics (MAE, mean bias, n) are additionally reported, enabling direct evaluation of whether the directional weighting reduced group-specific biases.

---

## 6. SHAP-Style Feature Importance via Prediction Contributions

### 6.1 XGBoost Native Prediction Contributions

The pipeline uses XGBoost's native `pred_contribs=True` mechanism to obtain tree-based additive feature contributions. These are computed via the path-dependent SHAP approximation implemented internally in XGBoost's booster:

```python
dval = xgb.DMatrix(X_val, feature_names=feature_cols)
contrib = booster.predict(dval, pred_contribs=True)
```

**Output shape**: `(n_samples, n_features + 1)`
- Columns 0 to n_features−1: per-feature additive contributions for each sample
- Column n_features: bias term (expected prediction value)

**Additivity property**:
```
prediction_i = bias + Σ_j contrib_{i,j}
```

This property guarantees that the contribution values sum to the model's predicted output, making them interpretable as the individual feature's additive impact on each prediction.

### 6.2 Additivity Verification

After each fold, the pipeline rigorously verifies that the SHAP additivity property holds numerically:

```python
margin = booster.predict(dval, output_margin=True)
recon  = fold_contrib.sum(axis=1) + fold_bias
delta  = margin - recon

additive_ok = np.allclose(margin, recon, atol=shap_check_tol, rtol=shap_check_rtol)
```

**Default tolerances**:
- `shap_check_tol = 1e-6` (absolute tolerance)
- `shap_check_rtol = 1e-5` (relative tolerance)

If additivity fails, a `ValueError` is raised with the maximum absolute and relative discrepancy values. This check prevents silently incorrect SHAP values from being written to output files.

**Reported diagnostics per fold**: `max_abs_diff` and `max_rel_diff` are printed regardless of pass/fail status, enabling monitoring of numerical precision across folds.

### 6.3 OOF SHAP Matrix Assembly

Contributions from each fold's test set are assembled into a global OOF matrix:

```python
oof_contribs = np.zeros((len(X), len(feature_cols)), dtype=float)
oof_bias     = np.zeros(len(X), dtype=float)
oof_contribs[val_idx, :] = fold_contrib
oof_bias[val_idx]        = fold_bias
```

An overlap guard (`shap_filled` boolean mask) raises a `ValueError` if any participant's contribution slot is written twice — a diagnostic that catches fold indexing bugs.

### 6.4 SHAP Importance Aggregation

`_build_shap_importance_table()` computes ranked feature importance from the OOF contribution matrix at two scopes:

**Global scope**: Computed from all participants
```
mean_abs_shap[j] = (1/n) * Σ_i |contrib_{i,j}|
mean_signed_shap[j] = (1/n) * Σ_i contrib_{i,j}
```

**Per age-group scopes**: Computed separately for `Young`, `Middle aged`, and `Old` participants

| Output Column | Description |
|---|---|
| `scope` | `global`, `Young`, `Middle aged`, or `Old` |
| `feature` | Feature name |
| `mean_abs_shap` | Mean absolute contribution (unsigned importance) |
| `mean_signed_shap` | Mean signed contribution (direction of effect) |
| `rank` | Rank within scope by descending `mean_abs_shap` |

**Interpretation of `mean_signed_shap`**: A positive value indicates that higher feature values push predictions upward (toward older age), while a negative value indicates the opposite. This directional information is unavailable from standard gain-based XGBoost importance.

**Per age-group importance**: Features that are important globally but unimportant for Young participants — or vice versa — can be identified by comparing `mean_abs_shap` across scopes. This decomposition is particularly valuable for understanding which neurophysiological signals are age-sensitive specifically in the young, old, or middle-aged subpopulation.

### 6.5 SHAP Output Files

| File | Contents |
|---|---|
| `xgboost_oof_shap_values.csv` | Per-participant SHAP contributions (columns: `participant_id`, `actual_age`, `predicted_age`, `age_group`, `shap_bias_term`, `shap__{feature_name}` for each feature) |
| `xgboost_oof_shap_importance.csv` | Long-format importance table (global + per age-group rankings) |

---

## 7. Directional Weight Grid Search

### 7.1 Alpha Pair Evaluation

The pipeline supports evaluating **multiple combinations** of `(alpha_high, alpha_low)` in a single run. All combinations from the Cartesian product of `alpha_high_values` and `alpha_low_values` are evaluated sequentially:

```python
alpha_pairs = [
    (alpha_high, alpha_low)
    for alpha_high in alpha_high_list
    for alpha_low  in alpha_low_list
]
```

Each combination runs a complete nested CV independently, producing its own OOF predictions, metrics, and SHAP values.

### 7.2 Best Combination Selection

Combinations are ranked by a composite key:

```python
candidate_key = (
    float(result["oof_mae"]),    # lower is better (primary)
    float(result["oof_rmse"]),   # lower is better (tie-breaker 1)
    -float(result["oof_r2"]),    # higher is better (tie-breaker 2)
)
```

The combination with the lexicographically smallest key is selected as the best. Output files (participant errors, SHAP values, SHAP importance) are saved **only for the best combination**, while the alpha search summary table is always saved.

### 7.3 Default Configuration

| Parameter | Default | Description |
|---|---|---|
| `alpha_high` | 1.5 | Penalty multiplier for errors in the "wrong" direction |
| `alpha_low` | 0.5 | Penalty multiplier for errors in the "acceptable" direction |

These defaults reflect a moderate asymmetry: wrong-direction errors receive 3× the gradient magnitude of acceptable-direction errors (1.5 / 0.5 = 3). This asymmetry is substantial enough to steer predictions away from the worst-direction errors without overwhelming the signal from other samples.

**Command-line grid search example**:
```bash
python xgboost_combined_directional_cv_sensor.py \
    --alpha-high-values 1.0 1.5 2.0 2.5 \
    --alpha-low-values  0.3 0.5 0.7
```
This evaluates all 12 combinations and selects the best.

---

## 8. Inverse-Frequency Weighting Mode Comparison

### 8.1 Purpose

`run_inverse_weight_mode_comparison()` runs the full nested CV pipeline independently under each specified inverse-weighting mode and identifies which mode produces better OOF generalization.

**Available modes**:

| Mode | Key | Effect |
|---|---|---|
| `group` | Age-group label | Balances at the group level (default) |
| `age` | Exact chronological age | Balances at the individual age level |

### 8.2 Comparison Protocol

For each mode:
1. A complete `run_xgboost_combined_cv()` is executed under the same `random_state`, `n_splits`, `n_iter`, and alpha grid
2. Output files are saved with a mode-specific suffix (e.g., `errors_group.csv`, `errors_age.csv`)
3. OOF metrics are collected

Modes are ranked by the same composite key as the alpha search (MAE → RMSE → R²). The winning mode and its full result payload are returned.

**Output**: `xgboost_inverse_weight_mode_comparison.csv` — one row per mode with all OOF metrics and per-group breakdowns.

**Rationale for empirical comparison**: The relative benefit of group-level versus age-level inverse weighting depends on the actual sample size distribution. In a perfectly balanced dataset, both modes are equivalent. In an imbalanced lifespan sample, `group` mode is more robust because individual-age counts are very small (often n=1 per exact age), which can make exact-age inverse weights extremely large for rare ages. The comparison quantifies this trade-off empirically on the actual dataset.

---

## 9. Output Files

### 9.1 Per-Participant Predictions

**File**: `participant_prediction_errors_xgboost.csv` (best alpha combination only)

| Column | Description |
|---|---|
| `participant_id` | Participant identifier |
| `actual_age` | Chronological age in years |
| `predicted_age` | Out-of-fold XGBoost prediction |
| `error` | `predicted_age − actual_age` (signed brain-PAD) |
| `absolute_error` | `|predicted_age − actual_age|` |

### 9.2 SHAP Feature Contributions

**File**: `xgboost_oof_shap_values.csv`

Long table with one row per participant. Columns include identifiers (`participant_id`, `actual_age`, `predicted_age`, `age_group`), the bias term (`shap_bias_term`), and one `shap__{feature_name}` column per input feature.

### 9.3 SHAP Feature Importance

**File**: `xgboost_oof_shap_importance.csv`

Long-format table with one row per (scope, feature) combination. Columns: `scope`, `feature`, `mean_abs_shap`, `mean_signed_shap`, `rank`.

### 9.4 Alpha Search Summary

**File** (if multiple alpha combinations are evaluated): included in the returned `alpha_search_results` DataFrame, which can be saved by the caller.

### 9.5 Inverse-Weight Mode Comparison

**File**: `xgboost_inverse_weight_mode_comparison.csv`

One row per weighting mode, with columns for OOF MAE, RMSE, R², mean bias, per-group MAE and bias, and the best alpha configuration selected for that mode.

---

## 10. Command-Line Interface

The script can be executed as a standalone program. Key arguments:

### 10.1 Data Arguments

| Argument | Default | Description |
|---|---|---|
| `--data-csv` | (see script) | Path to the primary feature CSV |
| `--features-csv` | `globally_stable_features_sensor.csv` | Path to the SGL-derived stable feature list |
| `--use-all-features` | `0` | `0` = use features CSV; `1` = use all numeric columns |
| `--target-col` | `age` | Name of the age column |
| `--age-group-col` | `age_group` | Name of the age-group column |
| `--id-col` | `participant_id` | Name of the participant ID column |

### 10.2 Output Arguments

| Argument | Default | Description |
|---|---|---|
| `--errors-csv` | (see script) | Path for OOF participant prediction error CSV |
| `--save-shap` | `1` | `1` = save SHAP outputs, `0` = skip |
| `--shap-values-csv` | (see script) | Path for per-participant SHAP contributions CSV |
| `--shap-importance-csv` | (see script) | Path for SHAP importance ranking CSV |

### 10.3 Model Arguments

| Argument | Default | Description |
|---|---|---|
| `--n-splits` | `5` | Number of outer CV folds |
| `--n-iter` | `200` | Number of random hyperparameter configurations to evaluate |
| `--random-state` | `42` | Random seed for CV and hyperparameter search |
| `--n-jobs` | `-1` | Number of parallel jobs for `RandomizedSearchCV` |

### 10.4 Objective Arguments

| Argument | Default | Description |
|---|---|---|
| `--alpha-high-values` | `[1.5]` | One or more alpha_high values (space-separated) |
| `--alpha-low-values` | `[0.5]` | One or more alpha_low values (space-separated) |
| `--inverse-weight-modes` | `group` | Weighting mode(s) for inverse-frequency component |
| `--shap-check-tol` | `1e-6` | Absolute tolerance for SHAP additivity check |
| `--shap-check-rtol` | `1e-5` | Relative tolerance for SHAP additivity check |

### 10.5 Comparison Mode

| Argument | Default | Description |
|---|---|---|
| `--run-comparison` | `0` | `0` = single mode run; `1` = compare all specified modes |
| `--comparison-csv` | (see script) | Path for inverse-weight mode comparison CSV |

**Example: Single run with default settings (SGL-selected features)**:
```bash
python xgboost_combined_directional_cv_sensor.py
```

**Example: Grid search over alpha values**:
```bash
python xgboost_combined_directional_cv_sensor.py \
    --alpha-high-values 1.0 1.5 2.0 \
    --alpha-low-values  0.3 0.5
```

**Example: Compare group vs age inverse weighting**:
```bash
python xgboost_combined_directional_cv_sensor.py \
    --run-comparison 1 \
    --inverse-weight-modes group age
```

**Example: Run on all features (no SGL pre-selection, ablation study)**:
```bash
python xgboost_combined_directional_cv_sensor.py --use-all-features 1
```

---

## 11. Integration with the SGL Feature Selection Pipeline

### 11.1 Feature Handoff

The SGL pipeline (Stage 1) produces `globally_stable_features_sensor.csv` through its multi-seed stability analysis. This file lists the names of features that were selected in ≥80% of outer CV folds across all seeds, ranked by global stability and reliability score.

The XGBoost pipeline reads this file and restricts its feature matrix to exactly those columns:
```python
feature_cols = pd.read_csv(args.features_csv)["feature"].tolist()
X = df[feature_cols].copy()
```

This ensures that:
1. The XGBoost model sees only features whose age-predictiveness was established independently in Stage 1
2. No hyperparameter information from Stage 2 leaks back to influence Stage 1's feature selection
3. The two-stage pipeline is modular: either stage can be modified or re-run independently

### 11.2 Complementary Roles

| Aspect | SGL (Stage 1) | XGBoost (Stage 2) |
|---|---|---|
| Model class | Sparse linear regression | Gradient-boosted trees |
| Primary purpose | Feature selection | Prediction and interpretation |
| Feature interactions | None (linear model) | Captured via tree splits |
| Interpretability | Sparse coefficients | SHAP contributions per sample |
| Regularization | Group sparsity + L1 | Depth, regularization, subsampling |
| Stability analysis | Multi-seed, CPSS, Nogueira | Not applicable (single run) |
| OOF predictions | Averaged across seeds | Single-seed OOF |

### 11.3 Consistency of CV Framework

Both pipelines use 5-fold stratified cross-validation on `age_group`, ensuring that performance metrics are comparable. The XGBoost pipeline's OOF MAE and R² can be directly compared against the SGL's nested CV results to quantify the gain from nonlinear modeling on the same feature subset.

---

## 12. Methodological Justification Summary

| Design Choice | Rationale |
|---|---|
| SGL-derived feature subset | Reduces dimensionality; ensures neurophysiological relevance; avoids overfitting |
| XGBoost | Captures nonlinear age–feature relationships missed by the SGL linear model |
| Custom combined objective | Corrects directional bias for extreme age groups; corrects group-frequency imbalance |
| `alpha_high = 1.5, alpha_low = 0.5` | Moderate 3:1 asymmetry; aggressive enough to steer predictions without dominating normal-age samples |
| Stratified outer CV | Preserves age-group distribution in every test fold; enables meaningful per-group metrics |
| Stratified inner CV | Prevents hyperparameter overfitting to a skewed training-fold composition |
| `RandomizedSearchCV (n_iter=200)` | Broad exploration of hyperparameter space; more efficient than grid search for high-dimensional grids |
| `neg_mean_absolute_error` scoring | Consistent with SGL; robust to outliers; directly interpretable in years |
| OOF predictions (no re-fit) | Honest performance estimates; avoids test-set contamination |
| SHAP via `pred_contribs` | Native XGBoost implementation; exact additivity to model output; per-sample and per-group decomposition |
| Additivity check (atol=1e-6) | Guards against numerical SHAP errors that could corrupt importance rankings |
| Inverse-frequency weighting | Compensates for unequal age-group sample sizes; ensures minority groups influence the objective |
| Mode comparison (group vs age) | Empirically determines whether group-level or individual-age-level frequency correction is better |

---

**Document Version**: 1.0
**Last Updated**: 2026-03-05
**Corresponding Script**: `xgboost_combined_directional_cv_sensor.py`
**Precedes in pipeline**: `ROI_Based_Feature_Selection_ML_age.py` (Stage 1 SGL feature selection)
**Compatible with**: XGBoost ≥ 1.7, scikit-learn ≥ 1.0, Python ≥ 3.10
