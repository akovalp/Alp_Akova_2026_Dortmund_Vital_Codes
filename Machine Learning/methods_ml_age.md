# ROI-Based Feature Selection and Age Prediction: Methodological Details

## 1. Overview

This document describes the comprehensive methodology implemented in `ROI_Based_Feature_Selection_ML_age.py` for predicting chronological age from multimodal EEG-derived features using Sparse Group LASSO (SGL) regression. The pipeline integrates source-space power features and connectivity features extracted from four anatomically defined brain regions (frontal, parietal, temporal, occipital), submitting them to a hierarchical regularised regression framework with extensive validation through nested cross-validation, stability selection, and complementary pairs subsampling.

### 1.1 Primary Objectives

- Identify EEG-derived neuromarkers of chronological age through embedded feature selection
- Exploit neurophysiological structure of the feature space via group-aware regularization
- Provide robust, reproducible feature selection estimates insulated from overfitting
- Quantify the reliability of selected features across different randomization seeds and subsampling schemes
- Diagnose prediction performance stratified by age group and detect regression-to-mean artifacts

### 1.2 Scientific Rationale

Standard LASSO penalizes features individually and disregards the structure of the feature space. In multimodal EEG studies, however, features naturally cluster by neurophysiological meaning — within-region connectivity, between-region connectivity, relative power, spectral flatness, peak alpha frequency (PAF), and band ratios — each of which in turn clusters by brain region or region pair. Sparse Group LASSO simultaneously enforces sparsity within groups and sparsity at the group level, enabling the model to select interpretable neurophysiological modules rather than isolated features. This structural prior directly mirrors the ROI architecture used in the companion source-power and PLI connectivity extraction pipelines.

---

## 2. Input Data Structure and Requirements

### 2.1 Data Path and Format

**Default Input File**:
```
/Dortmund_Vital_Alp_Akova_Clean/Data/merged_df_connectivity_power.csv
```

This CSV contains one row per participant, with the following mandatory columns:

| Column | Type | Description |
|--------|------|-------------|
| `participant_id` | string | BIDS-compatible participant identifier |
| `age` | numeric | Chronological age in years (prediction target) |
| `age_group` | categorical | Categorical age stratum: `Young`, `Middle aged`, or `Old` |

All remaining columns are treated as candidate features.

### 2.2 Session Filtering

**Parameter**: `FEATURE_SESSION`

| Value | Behavior |
|-------|----------|
| `"pre"` | Retain only features containing `_pre_` in their column name |
| `"post"` | Retain only features containing `_post_` in their column name |
| `"both"` | Use all available features (default) |

**Rationale**: Restricting to a single session enables evaluation of whether baseline (pre) or post-intervention (post) features are more predictive of age, and prevents data leakage when one session has incomplete coverage.

### 2.3 Required Column Validation

Before any analysis, the pipeline verifies:
- All required columns (`participant_id`, `age`, `age_group`) are present
- `age_group` contains no null values
- `age_group` values belong exclusively to `{Young, Middle aged, Old}`

Violations raise a `ValueError` with a descriptive message, preventing silent downstream failures.

---

## 3. Feature Space Definition

### 3.1 Regions and Bands

**Anatomical Regions**:
```
frontal, parietal, temporal, occipital
```

**Frequency Bands**:
```
theta (4–8 Hz), alpha (8–14 Hz), beta (14–30 Hz), broadband
```

**Canonical Region Pairs** (6 bilateral pairs):
```
frontal_parietal, frontal_temporal, frontal_occipital,
parietal_temporal, parietal_occipital, temporal_occipital
```

### 3.2 Feature Types

The pipeline recognizes seven distinct neurophysiological feature categories:

| Feature Type | Identifier | Description |
|---|---|---|
| Within-region connectivity | `within_wpli` | wPLI computed between electrodes within the same ROI |
| Between-region connectivity | `between_wpli` | wPLI computed between pairs of distinct ROIs |
| Relative power | `relative_power` | Band power normalized to total power (band × region) |
| Region relative power | `region_relative_power` | Regional power collapsed across bands |
| Spectral flatness | `spectral_flatness` | Geometric/arithmetic mean ratio within a band |
| Peak alpha frequency | `paf` | Interpolated peak in the 8–14 Hz range |
| Band ratios | `band_ratio` | Derived ratios: alpha/theta, beta/theta, beta/alpha |

### 3.3 Band-Ratio Feature Engineering

Before analysis, the pipeline automatically **derives missing band-ratio features** from existing relative power columns using `ensure_band_ratio_features()`.

**Derived Ratios** (per condition × session × region):

| Ratio Name | Numerator | Denominator |
|---|---|---|
| `alpha_theta_ratio` | alpha relative power | theta relative power |
| `beta_theta_ratio` | beta relative power | theta relative power |
| `beta_alpha_ratio` | beta relative power | alpha relative power |

**Column naming convention**: `{condition}_{session}_{region}_{ratio_suffix}`

**Denominator Guard**: A small epsilon (ε = 1×10⁻⁸) prevents division by zero. Ratios where the denominator is too small are set to NaN and subsequently filled with 0.0.

**Rationale**: Band ratios capture the balance between frequency bands at a given region. For example, the theta/alpha ratio is a well-established neuromarker of cognitive aging. Deriving these features ensures they participate in feature selection even if they were not computed during upstream processing.

### 3.4 Feature Name Parsing

All feature metadata (type, region, region pair, band) is parsed programmatically from column names using `_parse_feature_metadata()`. The parser supports:

1. **Short connectivity notation**: Regular expressions matching the compact format `{EC|EO}_{PRE|POST}_{BAND}_{REGION}_{WTH|BTW}` using `SHORT_WTH_PATTERN` and `SHORT_BTW_PATTERN`
2. **Long descriptive notation**: Substring matching against known region names, band names, and type keywords (e.g., `_within_pli`, `_between_pli`, `_flatness`, `_paf`, `_rel`, `_region_rel`, `_ratio`)

**Code mapping tables**:
- Regions: `FR` → `frontal`, `PA` → `parietal`, `TE` → `temporal`, `OC` → `occipital`
- Bands: `T` → `theta`, `A` → `alpha`, `B` → `beta`, `BB` → `broadband`

---

## 4. Feature Grouping for Sparse Group LASSO

### 4.1 Purpose of Grouping

The Sparse Group LASSO requires that features be partitioned into groups prior to fitting. The penalty simultaneously shrinks entire groups to zero (group-level sparsity) and shrinks individual features within retained groups (within-group sparsity). The grouping scheme therefore encodes the researcher's prior belief about which features form a natural neurophysiological module.

### 4.2 Grouping Tiers

**Parameter**: `GROUPING_TIER ∈ {tier1, tier2}`

#### Tier 1: Feature-Type Grouping

Groups features exclusively by their neurophysiological category:

| Group Name | Members |
|---|---|
| `type_within_wpli` | All within-region wPLI features |
| `type_between_wpli` | All between-region wPLI features |
| `type_relative_power` | All band×region relative power features |
| `type_region_relative_power` | All region-only relative power features |
| `type_spectral_flatness` | All spectral flatness features |
| `type_paf` | All peak alpha frequency features |
| `type_band_ratio` | All band-ratio features |

**Advantage**: Creates large groups (hundreds of features per group), maximizing the power of group-level shrinkage.
**Limitation**: Ignores spatial structure; the model cannot selectively activate one region's connectivity over another within the same feature type.

#### Tier 2: Feature-Type × Spatial Unit Grouping (Default)

Groups features by the joint combination of feature type and anatomical spatial unit:

| Feature Type | Spatial Unit | Example Group |
|---|---|---|
| `within_wpli` | Region | `within_wpli__frontal` |
| `between_wpli` | Region pair | `between_wpli__frontal_parietal` |
| `relative_power` | Region | `relative_power__occipital` |
| `region_relative_power` | Region | `region_relative_power__temporal` |
| `spectral_flatness` | Region | `spectral_flatness__parietal` |
| `paf` | Region | `paf__frontal` |
| `band_ratio` | Region | `band_ratio__occipital` |

**Advantage**: Enables region-specific feature selection, allowing the model to identify, for example, that occipital alpha connectivity ages differently from frontal theta power.
**Limitation**: Some groups (e.g., `paf__frontal`) may contain only 4 features (2 conditions × 2 sessions), reducing the effectiveness of pure group-level shrinkage for those categories.

**Note**: Band-ratio features are always kept separate from relative-power features regardless of tier, preventing the model from conflating normalized power levels with power ratios.

### 4.3 Group Assignment Algorithm

```
create_feature_groups(feature_names, grouping_tier, strict_assignment=True)
```

1. For each feature, call `_parse_feature_metadata()` to obtain its type, region, region pair, and band
2. Compute the group key via `_group_key_from_metadata()` according to the active tier
3. Append the feature index to the corresponding group's index array
4. Sort groups by feature type rank (preserving a canonical display order)
5. Print a full group assignment report (see below)
6. If `strict_assignment=True` and any features are unassigned, raise a `ValueError`

**Group Assignment Report** printed to stdout includes:
- Total features assigned per feature type
- Number of groups and group size statistics (min, max, median)
- List of any unassigned features with the reason for non-assignment

---

## 5. Data Quality Assessment

Prior to modelling, `check_data_quality()` audits the feature matrix `X` for:

| Check | Criterion | Action |
|---|---|---|
| Missing values | `np.isnan(X).any()` | Warns; reports count of affected features |
| Constant features | `variance < 1e-10` | Warns; lists up to 3 examples |
| Near-zero variance | `variance < 0.01` | Informational note |
| Extreme outliers | `|z-score| > 5` | Reports count of cells and affected participants |

No automatic imputation or removal is performed; the researcher is responsible for resolving flagged issues before re-running.

---

## 6. Model Architecture: Sparse Group LASSO

### 6.1 Mathematical Formulation

The SGL objective function minimizes:

```
L(β) = (1/2n) ||y - Xβ||² + α · [l1_ratio · ||β||₁ + (1 - l1_ratio) · Σ_g √p_g ||β_g||₂]
```

Where:
- **α** (alpha): Global regularization strength; larger α → more sparsity
- **l1_ratio**: Mix between individual (LASSO) and group (Group LASSO) penalties
  - `l1_ratio = 1.0` → pure LASSO (no group structure)
  - `l1_ratio = 0.0` → pure Group LASSO (all-or-nothing group selection)
  - `0 < l1_ratio < 1` → hybrid (simultaneous group and feature sparsity)
- **p_g**: Number of features in group g
- **β_g**: Coefficient sub-vector for group g

**Implementation**: `groupyr.SGL` and `groupyr.SGLCV` from the `groupyr` library, patched for scikit-learn ≥ 1.0 API compatibility (see Section 6.2).

### 6.2 Compatibility Patching

The `groupyr` library predates the `__sklearn_tags__` API introduced in scikit-learn ≥ 1.6. Three monkey-patch functions are applied at import time:

```python
_patch_groupyr_tags()  # Applied immediately at module load
```

This sets `estimator_type = "regressor"` and attaches `RegressorTags` and `TransformerTags` to `SGL`, `SGLCV`, and `SGLBaseEstimator`, preventing deprecation warnings and ensuring correct pipeline behavior.

`PatchedSGLCV` and `StratifiedPatchedSGLCV` are subclasses that additionally re-apply the patch on every `fit()` call to guard against tag resets by the sklearn Pipeline machinery.

### 6.3 Stratified Inner Cross-Validation

`StratifiedPatchedSGLCV.fit()` accepts an optional `age_group` array. When provided:
- The `cv` parameter (integer) is replaced by explicit `StratifiedKFold` splits on `age_group`
- This ensures each inner fold maintains the Young / Middle aged / Old proportion
- The `inner_random_state` parameter allows each outer fold to use a different inner seed, reducing systematic bias

**Rationale**: Without stratification, inner folds in a small sample could accidentally contain only one age group, causing the model to optimize hyperparameters on a non-representative subset.

### 6.4 Hyperparameter Grid

| Hyperparameter | Values | Description |
|---|---|---|
| `alpha` | 100-point log-spaced grid | Global regularization strength |
| `l1_ratio` | `[0.0, 0.1, 0.25, 0.5, 0.7, 0.8, 0.9, 0.95, 1.0]` | Individual/group penalty mix |
| `eps` | 1×10⁻³ | Ratio of minimum to maximum alpha in grid |

The alpha grid is constructed as `α_max × [eps, 1]` with `n_alphas = 100` log-spaced points, following the `LassoCV` convention.

**Scoring**: `neg_mean_absolute_error` — appropriate for continuous age prediction where outlier-robustness is preferred over minimizing squared errors.

---

## 7. Feature Scaling

**Method**: `RobustScaler(quantile_range=(25, 75))`

**Rationale**: EEG-derived power and connectivity features often exhibit heavy-tailed distributions and participant-level outliers. The Robust Scaler centers features on the median and scales by the interquartile range, making it insensitive to extreme values that would distort a `StandardScaler`.

**Alternative**: `StandardScaler` (zero-mean, unit-variance) is available via `SCALER_METHOD = "standard"`.

**Pipeline placement**: Scaling is the first step in the `sklearn.pipeline.Pipeline`, ensuring that the scaler is fit only on training data within each fold, preventing data leakage.

---

## 8. Nested Cross-Validation Framework

### 8.1 Design Rationale

Nested CV separates two concerns:
1. **Outer loop**: Estimates generalisation error (what MAE/R² the model achieves on unseen data)
2. **Inner loop**: Selects optimal hyperparameters (alpha, l1_ratio) via `SGLCV`

Using a single-layer CV for both would produce optimistically biased performance estimates, because hyperparameters are selected using the same data that evaluates their performance. Nesting ensures the outer test set is never seen during hyperparameter tuning.

### 8.2 Outer Cross-Validation

**Method**: Repeated Stratified K-Fold (`make_repeated_stratified_splits`)

| Parameter | Value |
|---|---|
| `SGL_OUTER_FOLDS` | 5 folds |
| `SGL_OUTER_REPEATS` | 5 repeats |
| **Total outer fits** | 25 (5 folds × 5 repeats) |
| **Stratification** | By `age_group` (Young / Middle aged / Old) |

Each repeat uses a distinct random seed (`base_seed + repeat_index`) to ensure different fold partitionings across repeats.

**Fold Composition Reporting**: `summarize_fold_composition()` prints the min, mean, and max proportion of each age group across test folds, verifying that stratification succeeded.

### 8.3 Inner Cross-Validation

**Method**: Stratified K-Fold within `StratifiedPatchedSGLCV`

| Parameter | Value |
|---|---|
| `SGL_INNER_FOLDS` | 5 folds |
| `Scoring` | `neg_mean_absolute_error` |

The inner CV is executed once per outer training set to select the best `(alpha, l1_ratio)` combination.

### 8.4 Parallelization

```python
with parallel_backend("loky", n_jobs=outer_n_jobs):
    results_list = Parallel(n_jobs=outer_n_jobs, verbose=10)(
        delayed(_run_outer_fold)(...) for split_idx, train_idx, test_idx in split_tasks
    )
```

**Strategy**: Outer folds are parallelized across CPU cores using `joblib.Parallel` with the `loky` backend (fork-safe process pool). Inner SGLCV runs are kept sequential per outer fold (`inner_n_jobs=1`) to avoid over-subscription on machines with limited memory.

**Platform handling**: On non-Windows systems, `JOBLIB_START_METHOD = "fork"` is set as a default to maximize startup speed.

### 8.5 Performance Metrics

For each outer fold, the pipeline computes:

| Metric | Formula | Interpretation |
|---|---|---|
| Test MAE | `mean_absolute_error(y_test, ŷ_test)` | Average prediction error in years |
| Test R² | `r2_score(y_test, ŷ_test)` | Proportion of age variance explained |
| Train MAE | `mean_absolute_error(y_train, ŷ_train)` | In-sample fit quality |
| Train R² | `r2_score(y_train, ŷ_train)` | In-sample variance explained |

**Aggregated across folds**:
- Mean ± SD of test MAE and R²
- 95% confidence intervals via Student's t-distribution
- Per-repeat averages (5 folds each) to assess stability
- Train–Test R² gap (gap > 0.15 triggers an overfitting warning)

### 8.6 Out-of-Fold Predictions

For each participant, predictions from all outer folds (across all repeats) where they appeared in the test set are **averaged**:

```
ŷ_oof[i] = (1/k_i) * Σ ŷ_test[i, repeat]
```

where `k_i` is the number of times participant `i` appeared in a test fold. These averaged out-of-fold predictions are used for all downstream diagnostic plots and error analyses.

---

## 9. Fold-Level Feature Selection Stability

### 9.1 Selection Tracking

`analyze_fold_stability()` collects per-fold selection information from all `n_outer_folds × n_repeats` estimators:

- **Feature selection matrix**: Binary matrix of shape `(n_folds, n_features)` — 1 if a feature's coefficient is nonzero, 0 otherwise
- **Group selection matrix**: Binary matrix of shape `(n_folds, n_groups)` — 1 if any feature in the group is nonzero
- **Coefficient matrix**: Raw coefficient values of shape `(n_folds, n_features)` for sign and magnitude analysis

### 9.2 Stability Computation

**Feature stability**: Proportion of folds in which each feature is selected:
```
stability[j] = (1/n_folds) * Σ_k I(β̂_jk ≠ 0)
```

**Group stability**: Proportion of folds in which the group is active (at least one nonzero feature):
```
group_stability[g] = (1/n_folds) * Σ_k I(∃ j ∈ g: β̂_jk ≠ 0)
```

**Stability Threshold**: Features with `stability ≥ 0.80` (selected in ≥80% of outer folds) are classified as highly stable.

### 9.3 Stable Feature Export

`export_stable_features_cv()` writes a CSV with columns:

| Column | Description |
|---|---|
| `feature` | Feature name |
| `group` | Assigned SGL group |
| `stability` | Fold-level selection frequency |
| `mean_coef` | Mean coefficient across folds |
| `std_coef` | Standard deviation of coefficient |
| `abs_mean_coef` | Absolute mean coefficient (for ranking) |

**Output**: `selected_features_stable_cv.csv`

---

## 10. LASSO Baseline

`run_lasso_baseline()` provides a structurally equivalent comparison model:
- Replaces `SGLCV` with `LassoCV` (no group structure)
- Uses identical outer-fold splits, stratification scheme, and scaler
- Inner CV alpha grid matches SGL parameters (`n_alphas`, `eps`)

**Purpose**: Quantifies the added value of group structure. If SGL-tier2 achieves lower MAE than the LASSO baseline, group structure is beneficial. If they are equivalent, individual feature penalties alone suffice.

---

## 11. Grouping Tier Benchmark

`benchmark_grouping_tiers_vs_lasso()` runs, in sequence:
1. Full nested CV with Tier 1 grouping
2. Full nested CV with Tier 2 grouping
3. LASSO baseline

Results are sorted by ascending test MAE and saved to `grouping_tier_vs_lasso_benchmark.csv`.

**Control parameter**: `GROUPING_BENCHMARK_REPEATS = 1` (single repeat for the benchmark to reduce computation; the main analysis uses 5 repeats).

---

## 12. Multi-Seed Stability Analysis

### 12.1 Purpose

A single set of cross-validation splits is subject to sampling variability: different random seeds produce different fold partitionings, which can substantially alter which features are selected. Multi-seed analysis quantifies whether the feature selection pattern is reproducible across random initializations, distinguishing genuinely age-predictive features from fold-specific artefacts.

### 12.2 Protocol

`run_multi_seed_stability_analysis()` repeats the full nested CV pipeline `n_seeds` times, each time using a different `random_state`:

| Parameter | Value |
|---|---|
| `MULTI_SEED_N_SEEDS` | 5 |
| Base seeds | `[42, 123, 456, 789, 1011]` |

All seed runs share the same `groups`, `group_names`, and hyperparameter grid. Results are aggregated across all `n_seeds × n_repeats × n_outer_folds` fitted estimators.

### 12.3 Aggregated Metrics

| Metric | Description |
|---|---|
| `global_feature_stability` | Selection frequency across all folds and all seeds |
| `global_group_stability` | Group activation frequency across all folds and all seeds |
| `mean_coef_global` | Grand mean coefficient across all runs |
| `std_coef_global` | Grand standard deviation of coefficients |
| `coef_cv` | Coefficient of variation: `std / |mean|`; measures coefficient reliability |
| `per_seed_stability` | Per-seed selection frequency matrix `(n_seeds, n_features)` |
| `seed_stability_std` | Standard deviation of selection rate across seeds; flags inconsistency |

### 12.4 Consistency Classification

| Condition | Classification |
|---|---|
| `seed_stability_std < 0.15` | Consistent — feature is reliably selected or reliably not selected |
| `seed_stability_std ≥ 0.15` | Inconsistent — seed-dependent selection |
| `global_stability > 0.30 AND seed_stability_std > 0.20` | Potentially unstable — selected frequently but inconsistently |

### 12.5 Reliability Score

```python
reliability = global_stability * (1 - clip(cross_seed_std / 0.3, 0, 1))
```

This composite score rewards features that are both frequently selected (high global stability) and consistently selected across seeds (low cross-seed standard deviation). Features are exported sorted by descending reliability to `legacy_stability_summary.csv`.

---

## 13. Complementary Pairs Stability Selection (CPSS)

### 13.1 Theoretical Background

Complementary Pairs Subsampling Selection (CPSS), following Meinshausen & Bühlmann (2010) and Shah & Samworth (2013), provides finite-sample control over the expected number of falsely selected variables. The method repeatedly fits the model on random half-samples and reports the proportion of runs in which each feature is selected (selection probability π_j). Features with π_j exceeding a data-driven threshold π_thr are declared stable.

**Key advantage over standard stability selection**: Complementary pairs reduce variance in the selection probability estimator by ensuring both halves of every split are used, making the analysis more efficient.

### 13.2 Operating Point Selection

Before running CPSS, the pipeline identifies a single `(alpha, l1_ratio)` operating point from the nested-CV fold parameters:

```python
select_cpss_operating_point_from_nested_cv(fold_params, alpha_grid, l1_grid)
```

**Procedure**:
1. Collect `alpha` and `l1_ratio` from all outer fold estimators
2. Compute `l1_ratio_median`; snap to nearest grid value (preferring higher values)
3. Compute `log(alpha)` median; exponentiate; snap to nearest alpha grid value

**Rationale**: The median of the fold-selected hyperparameters reflects the model complexity that generalizes best on this dataset. Using the grid-snapped value ensures the operating point exists in the SGLCV alpha path.

### 13.3 Stratified Complementary Pair Generation

`generate_stratified_complementary_pairs()` creates `n_pairs` independent complementary half-sample splits:

| Parameter | Value |
|---|---|
| `CPSS_N_PAIRS` | 50 (= 100 half-sample runs total) |
| `CPSS_MIN_COUNT_PER_GROUP` | 2 per age group per half-sample |
| `CPSS_MAX_PAIR_RETRIES` | 5000 |

**Algorithm**:
1. Use `StratifiedShuffleSplit(test_size=0.5)` to create balanced halves
2. Validate both halves contain at least `min_count_per_group` participants from each age stratum
3. Check for duplicate splits using byte-level mask fingerprints
4. Retry if validation fails; raise `RuntimeError` if `max_retries` is exhausted

**Complementarity**: For each split, half_a and half_b are complementary (half_a ∪ half_b = full dataset, half_a ∩ half_b = ∅), satisfying the formal requirement for variance reduction.

### 13.4 Fixed-Point CPSS Execution

`run_cpss_fixed_point()` fits the SGL at the fixed operating point on each of the 2×`n_pairs` half-samples:

```
coefs = Parallel(n_jobs=-1)(
    delayed(_run_fixed_sgl_on_subset)(X, y, subset_idx, groups, alpha, l1_ratio)
    for subset_idx in all_subset_indices
)
```

From the resulting coefficient matrices, selection probabilities are estimated:

```
π_j = (1/n_runs) * Σ_k I(β̂_jk ≠ 0)         # Feature selection probability
π_g = (1/n_runs) * Σ_k I(∃ j ∈ g: β̂_jk ≠ 0)  # Group selection probability
```

### 13.5 Selection Threshold Computation

The per-study selection threshold is derived from the expected number of selected features:

```
q̂ = mean(|{j : β̂_j ≠ 0}|)                    # Average selected feature count per run
π_thr_feature = 0.5 * (1 + q̂² / (p * EV))      # EV = expected false positives bound (= 1.0)
```

**Clamping for reporting**: The threshold is clamped to `[0.60, 0.99]` for display purposes, as raw thresholds outside this range are rarely interpretable.

**EV = 1.0**: The expected number of falsely selected variables under this threshold is bounded by 1.0, per the Shah–Samworth theoretical guarantee.

### 13.6 Alpha Fallback Mechanism

`run_cpss_with_alpha_fallback()` handles cases where the initial operating-point alpha is too weak (producing `pi_thr_feature_raw > 1.0`, which is infeasible):

1. Start at `alpha_star` from the operating point
2. Fit CPSS and evaluate feasibility (`pi_thr_feature_raw ≤ 1.0`)
3. If infeasible, move to a stronger alpha using exponential stepping (`step *= 2`)
4. Repeat until feasibility is achieved or the alpha grid is exhausted

**Shared pairs**: All alpha attempts use the same set of complementary pairs (generated once), ensuring comparability across attempts.

### 13.7 CPSS Output Files

| File | Contents |
|---|---|
| `cpss_feature_selection_probabilities.csv` | π_j for all features, sorted descending |
| `cpss_stable_features_ev1.csv` | Features with π_j ≥ π_thr_feature_raw |
| `cpss_group_selection_probabilities.csv` | π_g for all groups |
| `cpss_stable_groups_ev1.csv` | Groups with π_g ≥ π_thr_group_raw |
| `cpss_operating_point.json` | Full record of selected operating point and fallback history |
| `cpss_final_model_concordance.csv` | Feature-level concordance between CPSS stable set and final model |

---

## 14. Nogueira Stability Index

### 14.1 Definition

The Nogueira stability index Φ (Nogueira et al., 2018) quantifies the stability of a feature selection algorithm across `m` runs, accounting for the expected variance under random selection:

```
Φ = 1 - (1/(m·(m-1))) * Σ_{i≠j} dH(Si, Sj)² / [k̄/p * (1 - k̄/p)]
```

Where:
- `Si` and `Sj` are selection vectors from runs i and j
- `k̄` = mean number of selected features per run
- `p` = total number of features
- `dH` = Hamming distance between selection vectors

**Simplified implementation** (`_nogueira_phi`):
```python
kbar = selection_matrix.sum(axis=1).mean()
denom = (kbar / p) * (1 - kbar / p)
var_j = selection_matrix.var(axis=0, ddof=1)
phi = 1 - (var_j.mean() / denom)
phi = clip(phi, -1, 1)
```

### 14.2 Interpretation

| Φ range | Qualitative label | Interpretation |
|---|---|---|
| Φ > 0.75 | `excellent` | Feature selection is highly stable across runs |
| 0.40 ≤ Φ ≤ 0.75 | `intermediate_to_good` | Moderate stability; some variability across runs |
| Φ < 0.40 | `poor` | Low stability; results are highly seed-dependent |

### 14.3 Bootstrap Confidence Intervals

`compute_nogueira_index()` bootstraps the Φ estimate (`n_bootstraps = 1000`, `random_state = 42`):
- Resamples rows of the selection matrix with replacement
- Computes Φ on each bootstrap sample
- Reports 2.5th and 97.5th percentiles as the 95% confidence interval
- Computes a one-sided p-value: `P(boot_phi ≤ 0)`, indicating whether stability exceeds chance

### 14.4 Convergence Analysis

`compute_nogueira_convergence()` evaluates Φ at incremental run counts:

**Checkpoints**: `[10, 20, 30, 40, 50, 60, 70, 80, 90, 100]`

This diagnostic verifies whether 100 CPSS runs are sufficient to produce a stable Φ estimate. A Φ that has not plateaued by the final checkpoint indicates insufficient runs.

### 14.5 Primary and Supplementary Sources

Nogueira stability is computed from two independent sources:

| Source | Selection Matrix | Purpose |
|---|---|---|
| `primary_cpss` | CPSS half-sample runs (100 runs) | Primary stability estimate (fixed operating point) |
| `supplementary_cv` | Outer fold estimators (25 runs) | Cross-check from nested CV |

These are kept strictly separate to avoid conflating different sources of variability. The CPSS source is primary because it uses a fixed regularization point; the CV source is supplementary because different folds may select different optimal hyperparameters.

**Output files**: `nogueira_primary_summary.csv`, `nogueira_convergence.csv`

---

## 15. Final Model Training

`train_final_model()` fits a final `StratifiedPatchedSGLCV` on the **complete dataset** (all participants), using the same hyperparameter grid and stratified inner CV as the nested CV:

```python
Pipeline([
    ("scaler", RobustScaler(...)),
    ("sgl", StratifiedPatchedSGLCV(
        groups=groups,
        l1_ratio=SGL_L1_RATIOS,
        cv=5,
        scoring="neg_mean_absolute_error",
        n_jobs=-1,
    ))
])
```

**Important distinction**: The final model is trained for **feature interpretation** only, not for performance estimation. Performance estimates come exclusively from nested CV. The final model's selected features and coefficient magnitudes are used to characterize the neural architecture of age.

**Output**: `selected_features.csv` — all nonzero-coefficient features sorted by `|coefficient|`

---

## 16. CPSS–Final Model Concordance Check

`check_cpss_final_model_concordance()` compares the CPSS-stable feature set (EV = 1.0) against the final full-data model's selected features:

| Concordance Metric | Description |
|---|---|
| Recall (CPSS in final) | Fraction of CPSS-stable features retained by final model |
| Precision (final vs CPSS) | Fraction of final-model features that are CPSS-stable |
| Overlap count | |CPSS_stable ∩ final_selected| |
| Final-not-CPSS | Features in final model but not CPSS-stable (potential overfitting signals) |
| CPSS-missing-in-final | CPSS-stable features excluded by final model (may warrant investigation) |

**Warning trigger**: If `final_not_cpss` is non-empty, a concordance warning is printed, indicating that the final model selected features that were not robustly replicated across half-samples.

**Output**: `cpss_final_model_concordance.csv`

---

## 17. Prediction Diagnostics

### 17.1 Age-Group Error Analysis

`analyze_errors_by_age_group()` stratifies prediction errors (out-of-fold) by `age_group`:

| Reported Statistic | Description |
|---|---|
| Mean true age | Confirms group composition |
| Mean predicted age | Reveals systematic over/underprediction |
| Mean signed error | Direction of bias (positive = overprediction) |
| Std of signed error | Within-group variability |
| Mean absolute error | Magnitude of error per group |

### 17.2 Regression-to-Mean Diagnostics

A well-known artefact in age prediction models is **regression to the mean**: young participants are systematically overpredicted (predicted older than they are) while old participants are systematically underpredicted (predicted younger than they are). This occurs because the model's prediction is pulled toward the sample mean age.

`plot_regression_to_mean_diagnostics()` provides four diagnostic plots:
1. **Actual vs predicted scatter** with perfect-prediction line and regression line, slope annotated
2. **Signed error vs true age** scatter with linear trend overlay
3. **Error boxplot** by age group
4. **Brain-PAD histogram** per age group (Brain-PAD = predicted age − actual age)

`quantify_regression_to_mean()` reports:

| Metric | Ideal value | Interpretation |
|---|---|---|
| Prediction slope | 1.0 | `1 − slope` = fractional regression to mean |
| Error–age correlation | 0.0 | Negative correlation confirms regression to mean |

**Example**: A slope of 0.70 means that for every 10 years of true age difference, the model predicts only 7 years — 30% regression to the mean.

### 17.3 Participant-Level Error Export

`export_participant_prediction_errors()` generates a per-participant table with columns: `participant_id`, `actual_age`, `predicted_age`, `error`, `absolute_error`.

**Output**: `participant_prediction_errors.csv`

---

## 18. Statistical Validation: Permutation Test

`run_permutation_test()` assesses whether the model's performance exceeds chance by permuting the target variable:

```python
permutation_test_score(
    pipeline, X, y,
    scoring='neg_mean_absolute_error',
    cv=5,
    n_permutations=1000,
    n_jobs=-1,
    random_state=42
)
```

**Output**:
- True MAE (observed)
- Permuted MAE: mean ± SD across 1000 permutations
- p-value: proportion of permuted scores at least as good as the true score

**Significance thresholds**: p < 0.001 (highly significant), p < 0.01, p < 0.05, or p ≥ 0.05 (not significant).

**Plot**: `permutation_test.png` — histogram of permuted MAE scores with true score marked.

---

## 19. Visualization Suite

| Plot File | Function | Contents |
|---|---|---|
| `sgl_results_summary.png` | `plot_results()` | Top features by |coef|; group importance; region importance; band importance; group stability |
| `age_prediction_scatter.png` | `plot_predictions()` | Actual vs predicted age scatter with perfect-prediction and regression lines |
| `regression_to_mean_diagnostics.png` | `plot_regression_to_mean_diagnostics()` | 4-panel regression-to-mean diagnostic |
| `multi_seed_stability_analysis.png` | `plot_multi_seed_stability()` | 6-panel: MAE per seed; global stability histogram; cross-seed consistency scatter; per-seed heatmap; group stability bars; coefficient bars |
| `cpss_stability_paths.png` | `plot_cpss_stability_paths()` | Feature and group selection probability sorted paths with thresholds |
| `nogueira_convergence.png` | `plot_nogueira_convergence()` | Nogueira Φ vs. number of CPSS runs |
| `group_vs_feature_stability.png` | `plot_group_vs_feature_stability()` | Bar chart comparing feature vs. group Φ from CPSS and CV sources |
| `permutation_test.png` | `run_permutation_test()` | Permuted score null distribution with true score marker |

**Plot saving mode**: When `SAVE_PLOTS_ONLY = True`, all figures are immediately closed after saving (no GUI display). Set to `False` in interactive sessions to view figures inline.

---

## 20. Output Files Summary

| File | Contents | Primary Use |
|---|---|---|
| `selected_features_stable_cv.csv` | Features with ≥80% fold stability | Primary CV-based feature list |
| `selected_features.csv` | Final full-data model nonzero features | Feature interpretation |
| `participant_prediction_errors.csv` | Per-participant OOF predictions | Individual brain-PAD analysis |
| `grouping_tier_vs_lasso_benchmark.csv` | Tier1 vs Tier2 vs LASSO performance | Justify grouping choice |
| `cpss_feature_selection_probabilities.csv` | All-feature CPSS π values | CPSS feature ranking |
| `cpss_stable_features_ev1.csv` | CPSS-stable features at EV = 1.0 | Primary stable feature list |
| `cpss_group_selection_probabilities.csv` | All-group CPSS π values | CPSS group ranking |
| `cpss_stable_groups_ev1.csv` | CPSS-stable groups at EV = 1.0 | Primary stable group list |
| `cpss_operating_point.json` | Selected alpha, l1_ratio, fallback history | Reproducibility record |
| `cpss_final_model_concordance.csv` | Per-feature CPSS vs final model comparison | Concordance audit |
| `nogueira_primary_summary.csv` | Φ, CI, p-value from CPSS and CV | Stability reporting |
| `nogueira_convergence.csv` | Φ at incremental run counts | Convergence verification |
| `legacy_stability_summary.csv` | Multi-seed reliability scores | Extended stability analysis |

---

## 21. Configuration Parameter Reference

### 21.1 Session and Grouping

| Parameter | Default | Description |
|---|---|---|
| `FEATURE_SESSION` | `"both"` | Which session's features to use |
| `GROUPING_TIER` | `"tier2"` | Feature grouping scheme (tier1 or tier2) |
| `RUN_GROUPING_BENCHMARK` | `True` | Run Tier1 vs Tier2 vs LASSO comparison |
| `GROUPING_BENCHMARK_REPEATS` | 1 | Outer repeats for benchmark (fewer for speed) |

### 21.2 Model and Cross-Validation

| Parameter | Default | Description |
|---|---|---|
| `SCALER_METHOD` | `"robust"` | Feature scaler (`robust` or `standard`) |
| `SGL_EPS` | 1×10⁻³ | Alpha grid ratio (min/max) |
| `SGL_N_ALPHAS` | 100 | Number of alpha values in CV grid |
| `SGL_L1_RATIOS` | 9 values [0, 0.1, …, 1.0] | l1/group penalty mix values |
| `SGL_OUTER_FOLDS` | 5 | Outer CV folds |
| `SGL_INNER_FOLDS` | 5 | Inner CV folds |
| `SGL_OUTER_REPEATS` | 5 | Outer CV repeats |
| `SGL_RANDOM_STATE` | 42 | Primary random seed |

### 21.3 Multi-Seed Stability

| Parameter | Default | Description |
|---|---|---|
| `RUN_MULTI_SEED_STABILITY` | `True` | Enable multi-seed analysis |
| `MULTI_SEED_N_SEEDS` | 5 | Number of independent random seeds |
| `STABILITY_THRESHOLD` | 0.80 | Minimum selection frequency for stable features |

### 21.4 CPSS

| Parameter | Default | Description |
|---|---|---|
| `RUN_CPSS` | `True` | Enable CPSS analysis |
| `CPSS_N_PAIRS` | 50 | Number of complementary half-sample pairs |
| `CPSS_MIN_COUNT_PER_GROUP` | 2 | Minimum per-group count per half-sample |
| `CPSS_MAX_PAIR_RETRIES` | 5000 | Maximum attempts to find valid pairs |
| `CPSS_EV_TARGET` | 1.0 | Expected number of false positives bound |
| `CPSS_PI_REPORT_CLAMP` | (0.60, 0.99) | Threshold clamp for display |
| `CPSS_RANDOM_STATE` | 42 | CPSS random seed |
| `CPSS_N_JOBS` | -1 | Parallel jobs for CPSS runs |

### 21.5 Nogueira Index

| Parameter | Default | Description |
|---|---|---|
| `NOGUEIRA_BOOTSTRAPS` | 1000 | Bootstrap samples for CI estimation |
| `NOGUEIRA_RANDOM_STATE` | 42 | Bootstrap random seed |
| `NOGUEIRA_CONVERGENCE_CHECKPOINTS` | [10, 20, …, 100] | Run counts for convergence plot |

---

## 22. Execution Pipeline Summary

The `__main__` block executes the following stages in order:

```
1.  Load CSV data
2.  Derive band-ratio features
3.  Validate required columns and age_group values
4.  Extract feature matrix X and target y
5.  Data quality check (missing values, outliers, constant features)
6.  [Optional] Benchmark Tier1 vs Tier2 vs LASSO
7.  Create feature groups (default: Tier2)
8.  Run nested CV with SGL (5 folds × 5 repeats)
9.  Export CV-based stable feature list
10. Select CPSS operating point from nested CV
11. [Optional] Run CPSS with alpha fallback
12. Compute Nogueira index (CPSS primary + CV supplementary)
13. Plot Nogueira convergence and group-vs-feature stability
14. Train final model on full data
15. Create results tables (feature, group, region, band)
16. Plot summary results
17. Export out-of-fold predictions and per-participant errors
18. Plot actual vs. predicted age and regression-to-mean diagnostics
19. Extract and save final selected features
20. [Optional] CPSS–final model concordance check
21. [Optional] Multi-seed stability analysis
22. Export reliability summary
```

---

## 23. Methodological Justification Summary

| Design Choice | Rationale |
|---|---|
| Sparse Group LASSO | Encodes neurophysiological feature structure; simultaneous group- and feature-level sparsity |
| Tier 2 grouping | Enables region-specific feature selection within each feature type |
| Nested CV (5×5×5) | Unbiased performance estimation; separation of model selection from evaluation |
| Stratified folds | Preserves age-group distribution across all fold types |
| RobustScaler | Robust to heavy-tailed EEG distributions and participant outliers |
| Repeated CV (5 repeats) | Reduces fold-sampling variance in performance and stability estimates |
| CPSS (50 pairs) | Provides theoretically grounded false-discovery control at EV = 1.0 |
| Nogueira index | Quantifies stability relative to chance, with bootstrap CI |
| Multi-seed analysis (5 seeds) | Distinguishes genuine neurophysiological signal from randomisation artefacts |
| Out-of-fold predictions | Honest performance assessment; no test-set contamination |
| Regression-to-mean diagnostics | Characterizes systematic age-group-specific bias in predictions |

---

**Document Version**: 1.0
**Last Updated**: 2026-03-05
**Corresponding Script**: `ROI_Based_Feature_Selection_ML_age.py`
**Compatible with**: scikit-learn ≥ 1.6, groupyr ≥ 0.3, MNE-Python ≥ 1.0, Python ≥ 3.10
