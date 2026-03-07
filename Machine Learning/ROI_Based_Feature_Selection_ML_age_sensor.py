from sklearn.utils._tags import RegressorTags, TransformerTags
from sklearn.base import BaseEstimator, clone
from groupyr._base import SGLBaseEstimator
from groupyr import SGL, SGLCV
import warnings
import re
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import (
    StratifiedKFold,
    StratifiedShuffleSplit,
    permutation_test_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.linear_model import LassoCV
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from joblib import Parallel, delayed, parallel_backend
import os
import sys
import json
if sys.platform != "win32":
    os.environ.setdefault("JOBLIB_START_METHOD", "fork")
extra_warnings = (
    "ignore:.*_get_tags.*:FutureWarning,"
    "ignore:.*_validate_data.*:FutureWarning"
)
if "PYTHONWARNINGS" in os.environ and os.environ["PYTHONWARNINGS"]:
    os.environ["PYTHONWARNINGS"] = (
        os.environ["PYTHONWARNINGS"].rstrip(",") + "," + extra_warnings
    )
else:
    os.environ["PYTHONWARNINGS"] = extra_warnings



# Which session's features to use: "pre", "post", or "both"

FEATURE_SESSION = "both"
# "tier1" (feature-type), "tier2" (feature-type × region/pair)
GROUPING_TIER = "tier2"
RUN_GROUPING_BENCHMARK = True  # optional: compare tier1, tier2, and plain LASSO
GROUPING_BENCHMARK_REPEATS = 1
GROUPING_BENCHMARK_CSV = "grouping_tier_vs_lasso_benchmar_sensor.csv"
RUN_MULTI_SEED_STABILITY = True
MULTI_SEED_N_SEEDS = 5
SAVE_PLOTS_ONLY = True
STABILITY_THRESHOLD = 0.8
STABLE_FEATURES_CSV = "selected_features_stable_cv_sensor.csv"
PARTICIPANT_PREDICTION_ERRORS_CSV = "participant_prediction_errors_sensor_sgl.csv"
RUN_CPSS = True

DEFAULT_DATA_PATH = (
    "/Users/alpmac/CodeWorks/Trento/Dortmund_Vital_Alp_Akova_Clean/Data/merged_df_connectivity_power_sensor.csv"
)
REQUIRED_COLUMNS = {"participant_id", "age", "age_group"}
VALID_AGE_GROUPS = ["Young", "Middle aged", "Old"]
SCALER_METHOD = "robust"
SGL_EPS = 1e-3
SGL_N_ALPHAS = 100
SGL_L1_RATIOS = [0.0, 0.1, 0.25, 0.5, 0.7, 0.8, 0.9, 0.95, 1.0]
SGL_OUTER_FOLDS = 5
SGL_INNER_FOLDS = 5
SGL_OUTER_REPEATS = 5
SGL_RANDOM_STATE = 42

CPSS_N_PAIRS = 50  # 100 half-samples total
CPSS_MIN_COUNT_PER_GROUP = 2
CPSS_MAX_PAIR_RETRIES = 5000
CPSS_EV_TARGET = 1.0
CPSS_PI_REPORT_CLAMP = (0.6, 0.99)
CPSS_RANDOM_STATE = 42
CPSS_N_JOBS = -1
CPSS_FINAL_CONCORDANCE_CSV = "cpss_final_model_concordance.csv"

NOGUEIRA_BOOTSTRAPS = 1000
NOGUEIRA_RANDOM_STATE = 42
NOGUEIRA_CONVERGENCE_CHECKPOINTS = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

warnings.filterwarnings("ignore")
warnings.filterwarnings(
    "ignore", category=FutureWarning, message=".*PatchedSGLCV.*_get_tags.*"
)
warnings.filterwarnings(
    "ignore", category=FutureWarning, message=".*StratifiedPatchedSGLCV.*_get_tags.*"
)
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=".*BaseEstimator._validate_data.*deprecated.*",
)
warnings.filterwarnings(
    "ignore", category=FutureWarning, module="sklearn\\.utils\\._tags"
)
warnings.filterwarnings(
    "ignore", category=FutureWarning, module="sklearn\\.base"
)

# Monkey-patch groupyr estimators to have correct sklearn tags


def _finalize_plot(fig):
    """Save-only mode helper to avoid displaying figures."""
    if SAVE_PLOTS_ONLY:
        plt.close(fig)
    else:
        plt.show()


def _sglcv__sklearn_tags__(self):
    tags = BaseEstimator.__sklearn_tags__(self)
    tags.estimator_type = "regressor"
    tags.regressor_tags = RegressorTags()
    tags.transformer_tags = TransformerTags()
    tags.target_tags.required = True
    return tags


# Monkey-patch groupyr estimators to have correct sklearn tags


def _sgl__sklearn_tags__(self):
    tags = BaseEstimator.__sklearn_tags__(self)
    tags.estimator_type = "regressor"
    tags.regressor_tags = RegressorTags()
    tags.transformer_tags = TransformerTags()
    tags.target_tags.required = True
    return tags


def _sglbase__sklearn_tags__(self):
    tags = BaseEstimator.__sklearn_tags__(self)
    tags.estimator_type = "regressor"
    tags.regressor_tags = RegressorTags()
    tags.transformer_tags = TransformerTags()
    tags.target_tags.required = True
    return tags


def _patch_groupyr_tags():
    SGL.__sklearn_tags__ = _sgl__sklearn_tags__
    SGLBaseEstimator.__sklearn_tags__ = _sglbase__sklearn_tags__
    SGLCV.__sklearn_tags__ = _sglcv__sklearn_tags__


_patch_groupyr_tags()


class PatchedSGLCV(SGLCV):
    def __sklearn_tags__(self):
        return _sglcv__sklearn_tags__(self)

    def fit(self, X, y, **fit_params):
        _patch_groupyr_tags()
        return super().fit(X, y, **fit_params)


class StratifiedPatchedSGLCV(PatchedSGLCV):
    def __sklearn_tags__(self):
        return _sglcv__sklearn_tags__(self)

    def fit(self, X, y, age_group=None, inner_random_state=None, **fit_params):
        warnings.filterwarnings(
            "ignore", category=FutureWarning, message=".*_get_tags.*"
        )
        warnings.filterwarnings(
            "ignore",
            category=FutureWarning,
            message=".*BaseEstimator._validate_data.*deprecated.*",
        )
        _patch_groupyr_tags()
        if age_group is not None and isinstance(self.cv, int):
            cv = StratifiedKFold(
                n_splits=self.cv, shuffle=True, random_state=inner_random_state
            )
            self.cv = list(cv.split(X, age_group))
        return super().fit(X, y, **fit_params)


def make_repeated_stratified_splits(age_group, n_splits, n_repeats, random_state=42):
    """Create repeated StratifiedKFold splits based on age_group."""
    age_group = np.asarray(age_group)
    splits = []
    for repeat in range(n_repeats):
        seed = None if random_state is None else int(random_state) + repeat
        cv = StratifiedKFold(
            n_splits=n_splits, shuffle=True, random_state=seed)
        splits.extend(list(cv.split(np.zeros(len(age_group)), age_group)))
    return splits


def summarize_fold_composition(age_group, splits, group_order=None):
    """Print min/mean/max age_group proportions across folds."""
    age_group = np.asarray(age_group)
    if group_order is None:
        group_order = ["Young", "Middle aged", "Old"]
    proportions = []
    for _, test_idx in splits:
        fold_counts = (
            pd.Series(age_group[test_idx]).value_counts(
                normalize=True).reindex(group_order).fillna(0)
        )
        proportions.append(fold_counts)
    prop_df = pd.DataFrame(proportions)

    print("\n" + "-" * 40)
    print("FOLD AGE-GROUP COMPOSITION (proportion)")
    print("-" * 40)
    for group in group_order:
        if group in prop_df:
            min_v = prop_df[group].min()
            mean_v = prop_df[group].mean()
            max_v = prop_df[group].max()
            print(
                f"  {group}: min {min_v:.2%} | mean {mean_v:.2%} | max {max_v:.2%}")

    return prop_df


REGIONS = ["frontal", "parietal", "temporal", "occipital"]
BANDS = ["theta", "alpha", "beta", "broadband"]
REGION_CODE_TO_NAME = {
    "FR": "frontal",
    "PA": "parietal",
    "TE": "temporal",
    "OC": "occipital",
}
BAND_CODE_TO_NAME = {"T": "theta", "A": "alpha",
                     "B": "beta", "BB": "broadband"}
FEATURE_TYPES = [
    "within_wpli",
    "between_wpli",
    "relative_power",
    "region_relative_power",
    "spectral_flatness",
    "paf",
    "band_ratio",
]
FEATURE_TYPE_LABELS = {
    "within_wpli": "Within-region connectivity (wPLI)",
    "between_wpli": "Between-region connectivity (wPLI)",
    "relative_power": "Relative power",
    "region_relative_power": "Region-level relative power",
    "spectral_flatness": "Spectral flatness",
    "paf": "Peak alpha frequency (PAF)",
    "band_ratio": "Band ratios",
}
GROUPING_TIERS = {"tier1", "tier2"}
SHORT_WTH_PATTERN = re.compile(
    r"^(EC|EO)_(PRE|POST)_([A-Z]+)_([A-Z]{2})_WTH$"
)
SHORT_BTW_PATTERN = re.compile(
    r"^(EC|EO)_(PRE|POST)_([A-Z]+)_([A-Z]{2})_([A-Z]{2})_BTW$"
)


def _canonical_region_pair(region_a, region_b):
    if region_a is None or region_b is None:
        return None
    ordered = sorted([region_a, region_b], key=lambda r: REGIONS.index(r))
    return f"{ordered[0]}_{ordered[1]}"


def _all_region_pairs():
    pairs = []
    for i, region_a in enumerate(REGIONS):
        for region_b in REGIONS[i + 1:]:
            pairs.append(f"{region_a}_{region_b}")
    return pairs


REGION_PAIRS = _all_region_pairs()


def _group_type_from_group_name(group_name):
    if group_name.startswith("type_"):
        return group_name.replace("type_", "", 1)
    return group_name.split("__", 1)[0]


def _group_sort_key(group_name):
    group_type = _group_type_from_group_name(group_name)
    group_rank = FEATURE_TYPES.index(
        group_type) if group_type in FEATURE_TYPES else len(FEATURE_TYPES)
    return (group_rank, group_name)


def _is_relative_power_feature(feat_lower):
    if "_region_rel" in feat_lower:
        return False
    if "_ratio" in feat_lower:
        return False
    return "_rel" in feat_lower


def _identify_single_region(feat_lower, feat_original=None):
    for region in REGIONS:
        if (
            f"_{region}_" in feat_lower
            or feat_lower.endswith(f"_{region}")
            or feat_lower.startswith(f"{region}_")
        ):
            return region

    if feat_original is None:
        return None

    feat_upper = feat_original.upper()
    for code, region in REGION_CODE_TO_NAME.items():
        if f"_{code}_" in feat_upper or feat_upper.endswith(f"_{code}"):
            return region
    return None


def _identify_region_pair(feat_lower, feat_original=None):
    for pair in REGION_PAIRS:
        if f"_{pair}_" in feat_lower or feat_lower.endswith(f"_{pair}"):
            return pair

    if feat_original is None:
        return None

    feat_upper = feat_original.upper()
    region_hits = []
    for code, region in REGION_CODE_TO_NAME.items():
        if f"_{code}_" in feat_upper or feat_upper.endswith(f"_{code}"):
            region_hits.append(region)
    region_hits = list(dict.fromkeys(region_hits))
    if len(region_hits) >= 2:
        return _canonical_region_pair(region_hits[0], region_hits[1])
    return None


def _identify_band(feat_lower, feat_original=None):
    if "_broadband_" in feat_lower:
        return "broadband"
    for band in BANDS:
        if band == "broadband":
            continue
        if f"_{band}_" in feat_lower:
            return band

    if feat_original is None:
        return None

    feat_upper = feat_original.upper()
    for code, band in BAND_CODE_TO_NAME.items():
        if f"_{code}_" in feat_upper:
            return band
    return None


def _parse_short_connectivity_feature(feat):
    feat_upper = feat.upper()
    match_within = SHORT_WTH_PATTERN.match(feat_upper)
    if match_within:
        _, _, band_code, region_code = match_within.groups()
        region = REGION_CODE_TO_NAME.get(region_code)
        band = BAND_CODE_TO_NAME.get(band_code)
        if region is None:
            return None
        return {
            "feature_type": "within_wpli",
            "region": region,
            "region_pair": None,
            "band": band,
        }

    match_between = SHORT_BTW_PATTERN.match(feat_upper)
    if match_between:
        _, _, band_code, region_code_1, region_code_2 = match_between.groups()
        region_1 = REGION_CODE_TO_NAME.get(region_code_1)
        region_2 = REGION_CODE_TO_NAME.get(region_code_2)
        region_pair = _canonical_region_pair(region_1, region_2)
        band = BAND_CODE_TO_NAME.get(band_code)
        if region_pair is None:
            return None
        return {
            "feature_type": "between_wpli",
            "region": None,
            "region_pair": region_pair,
            "band": band,
        }

    return None


def _parse_feature_metadata(feature_name):
    feat = str(feature_name)
    feat_lower = feat.lower()

    short_conn = _parse_short_connectivity_feature(feat)
    if short_conn is not None:
        return short_conn

    if "_between_pli" in feat_lower:
        region_pair = _identify_region_pair(feat_lower, feat)
        band = _identify_band(feat_lower, feat)
        if region_pair is not None:
            return {
                "feature_type": "between_wpli",
                "region": None,
                "region_pair": region_pair,
                "band": band,
            }
        return None

    if "_within_pli" in feat_lower:
        region = _identify_single_region(feat_lower, feat)
        band = _identify_band(feat_lower, feat)
        if region is not None:
            return {
                "feature_type": "within_wpli",
                "region": region,
                "region_pair": None,
                "band": band,
            }
        return None

    region = _identify_single_region(feat_lower, feat)
    band = _identify_band(feat_lower, feat)

    if "_ratio" in feat_lower:
        return {
            "feature_type": "band_ratio",
            "region": region,
            "region_pair": None,
            "band": None,
        }
    if "_region_rel" in feat_lower:
        return {
            "feature_type": "region_relative_power",
            "region": region,
            "region_pair": None,
            "band": None,
        }
    if "_flatness" in feat_lower:
        return {
            "feature_type": "spectral_flatness",
            "region": region,
            "region_pair": None,
            "band": band,
        }
    if "_paf" in feat_lower:
        return {
            "feature_type": "paf",
            "region": region,
            "region_pair": None,
            "band": None,
        }
    if _is_relative_power_feature(feat_lower):
        return {
            "feature_type": "relative_power",
            "region": region,
            "region_pair": None,
            "band": band,
        }

    return None


def _group_key_from_metadata(metadata, grouping_tier):
    feature_type = metadata["feature_type"]

    if grouping_tier == "tier1":
        return f"type_{feature_type}"

    if feature_type == "within_wpli":
        region = metadata.get("region") or "unknown_region"
        return f"within_wpli__{region}"

    if feature_type == "between_wpli":
        region_pair = metadata.get("region_pair") or "unknown_pair"
        return f"between_wpli__{region_pair}"

    region = metadata.get("region")
    if region is None:
        return f"{feature_type}__global"
    return f"{feature_type}__{region}"


def create_feature_groups(
    feature_names,
    grouping_tier="tier2",
    strict_assignment=True,
):
    """
    Assign each feature to a sparse-group-lasso group.

    Tier 1:
      - Group only by feature type:
        within_wpli, between_wpli, relative_power, region_relative_power,
        spectral_flatness, paf, band_ratio.

    Tier 2:
      - Feature type × spatial unit:
        within_wpli by region, between_wpli by region-pair,
        and other feature types by region.
      - In the current EO/EC × pre/post design, region_relative_power and
        paf groups are small (4 features per region), which weakens pure
        group-level shrinkage for those categories.

    Band-ratio features are always separated from relative-power features.
    """
    tier = str(grouping_tier).strip().lower()
    if tier not in GROUPING_TIERS:
        raise ValueError(
            f"grouping_tier must be one of {sorted(GROUPING_TIERS)}; got '{grouping_tier}'."
        )

    groups = {}
    feature_to_group = {}
    unassigned = []

    for idx, feature_name in enumerate(feature_names):
        metadata = _parse_feature_metadata(feature_name)
        if metadata is None:
            unassigned.append(
                (idx, feature_name, "feature type not recognized"))
            continue

        key = _group_key_from_metadata(metadata, tier)
        groups.setdefault(key, []).append(idx)
        feature_to_group[idx] = key

    group_names = sorted(
        [name for name, idxs in groups.items() if len(idxs) > 0],
        key=_group_sort_key,
    )
    group_indices = [np.array(groups[name], dtype=int) for name in group_names]

    _print_group_report(
        feature_names=feature_names,
        group_names=group_names,
        group_indices=group_indices,
        unassigned=unassigned,
        grouping_tier=tier,
    )

    if strict_assignment and unassigned:
        preview = ", ".join([x[1] for x in unassigned[:3]])
        raise ValueError(
            f"{len(unassigned)} features were unassigned by create_feature_groups. "
            f"Examples: {preview}"
        )

    return group_indices, group_names, feature_to_group


def _print_group_report(
    feature_names, group_names, group_indices, unassigned, grouping_tier
):
    print("=" * 72)
    print("GROUP ASSIGNMENT SUMMARY")
    print("=" * 72)
    print(f"Grouping tier: {grouping_tier}")

    total = 0
    for feature_type in FEATURE_TYPES:
        type_groups = [
            (name, idx)
            for name, idx in zip(group_names, group_indices)
            if _group_type_from_group_name(name) == feature_type
        ]
        if not type_groups:
            continue

        label = FEATURE_TYPE_LABELS.get(feature_type, feature_type)
        type_total = sum(len(idx) for _, idx in type_groups)
        total += type_total
        sizes = [len(idx) for _, idx in type_groups]

        print(
            f"\n  -- {label} ({len(type_groups)} groups, {type_total} features) --"
        )
        for name, indices in sorted(type_groups, key=lambda x: -len(x[1])):
            print(f"    {name:>40s}: {len(indices):3d} features")

        print(
            f"    {'Group sizes':>40s}: min={min(sizes)}, max={max(sizes)}, "
            f"median={sorted(sizes)[len(sizes) // 2]}"
        )

    print(f"\n  Total assigned: {total} / {len(feature_names)}")

    if unassigned:
        print(f"\n  Unassigned features ({len(unassigned)}):")
        for _, feature_name, reason in unassigned[:10]:
            print(f"      {feature_name} ({reason})")
        if len(unassigned) > 10:
            print(f"      ... and {len(unassigned) - 10} more")

    print("=" * 72)


def print_group_details(feature_names, groups, group_names):
    """Print detailed breakdown of each group's features."""
    print("\n" + "=" * 60)
    print("DETAILED GROUP CONTENTS")
    print("=" * 60)

    for name, indices in zip(group_names, groups):
        if len(indices) > 0:
            print(f"\n{name} ({len(indices)} features):")
            for idx in indices[:10]:  # Show first 10
                print(f"    - {feature_names[idx]}")
            if len(indices) > 10:
                print(f"    ... and {len(indices) - 10} more")


def make_scaler(method="robust"):
    """Factory for feature scaling."""
    method = str(method).lower().strip()
    if method == "robust":
        return RobustScaler(quantile_range=(25, 75))
    if method == "standard":
        return StandardScaler()
    raise ValueError("scaler_method must be one of {'robust', 'standard'}")


REL_POWER_COLUMN_PATTERN = re.compile(
    r"^(Eyes(?:Closed|Open)_(?:pre|post))_"
    r"(theta|alpha|beta)_"
    r"(frontal|parietal|temporal|occipital)_rel$",
    flags=re.IGNORECASE,
)
RATIO_SPECS = [
    ("alpha_theta_ratio", "alpha", "theta"),
    ("beta_theta_ratio", "beta", "theta"),
    ("beta_alpha_ratio", "beta", "alpha"),
]


def ensure_band_ratio_features(df, denominator_eps=1e-8):
    """
    Add missing band-ratio features derived from relative power columns.

    Ratios are created per (condition × session × region):
      alpha/theta, beta/theta, beta/alpha.
    """
    df = df.copy()
    rel_lookup = {}

    for col in df.columns:
        match = REL_POWER_COLUMN_PATTERN.match(str(col))
        if match is None:
            continue
        prefix = match.group(1)
        band = match.group(2).lower()
        region = match.group(3).lower()
        rel_lookup.setdefault((prefix, region), {})[band] = col

    created_cols = []
    for (prefix, region), band_cols in sorted(rel_lookup.items()):
        required = {"theta", "alpha", "beta"}
        if not required.issubset(band_cols):
            continue

        for ratio_suffix, num_band, den_band in RATIO_SPECS:
            ratio_col = f"{prefix}_{region}_{ratio_suffix}"
            if ratio_col in df.columns:
                continue

            numerator = pd.to_numeric(df[band_cols[num_band]], errors="coerce")
            denominator = pd.to_numeric(
                df[band_cols[den_band]], errors="coerce")
            valid_denominator = denominator.where(
                np.abs(denominator) > float(denominator_eps), np.nan
            )
            ratio_values = numerator / valid_denominator
            ratio_values = ratio_values.replace(
                [np.inf, -np.inf], np.nan).fillna(0.0)
            df[ratio_col] = ratio_values
            created_cols.append(ratio_col)

    if created_cols:
        print(
            f"Added {len(created_cols)} derived ratio columns "
            "(alpha/theta, beta/theta, beta/alpha)."
        )
        print("  Examples:", ", ".join(created_cols[:6]))
    else:
        print("Band-ratio columns already present or insufficient rel-power columns.")

    return df, created_cols


def load_and_prepare_data(df):
    df, _ = ensure_band_ratio_features(df)

    missing_required = REQUIRED_COLUMNS.difference(df.columns)
    if missing_required:
        missing_txt = ", ".join(sorted(missing_required))
        raise ValueError(f"Missing required columns: {missing_txt}")

    if df["age_group"].isna().any():
        raise ValueError(
            "Column 'age_group' contains null values; cannot stratify.")

    unknown_groups = sorted(
        set(df["age_group"].unique()) - set(VALID_AGE_GROUPS))
    if unknown_groups:
        raise ValueError(
            "Column 'age_group' contains unexpected values: "
            + ", ".join(map(str, unknown_groups))
        )

    exclude_cols = ["participant_id", "age", "age_group", "Participant"]
    feature_cols = [col for col in df.columns if col not in exclude_cols]

    if FEATURE_SESSION in ("pre", "post"):
        feature_cols = [c for c in feature_cols if f"_{FEATURE_SESSION}_" in c]
        print(f"[FEATURE_SESSION = '{FEATURE_SESSION}'] "
              f"Kept {len(feature_cols)} features.")
    else:
        print(f"[FEATURE_SESSION = '{FEATURE_SESSION}'] Using all features.")

    X = df[feature_cols].values
    y = df["age"].values
    feature_names = feature_cols
    print(f"Data loaded: {X.shape[0]} samples, {X.shape[1]} features.")
    print(f"Target variable 'age' has range: {y.min()} to {y.max()} years.")
    print(
        f"Number of participants in age groups:\n{df['age_group'].value_counts()}")
    return X, y, feature_names, df


def check_data_quality(X, feature_names):
    """Check for missing values, outliers, and constant features."""
    print("\n" + "=" * 60)
    print("DATA QUALITY CHECK")
    print("=" * 60)

    issues = []

    # Missing values
    missing = np.isnan(X).sum(axis=0)
    if missing.sum() > 0:
        n_missing = (missing > 0).sum()
        issues.append(f"Missing values in {n_missing} features")
        print(f"⚠️  Missing values in {n_missing} features")
    else:
        print("✓  No missing values")

    # Constant features
    variance = X.var(axis=0)
    constant = variance < 1e-10
    if constant.sum() > 0:
        issues.append(f"{constant.sum()} constant features")
        print(f"⚠️  {constant.sum()} constant features (will cause issues)")
        for idx in np.where(constant)[0][:3]:
            print(f"    - {feature_names[idx]}")
    else:
        print("✓  No constant features")

    # Near-zero variance
    low_var = (variance > 1e-10) & (variance < 0.01)
    if low_var.sum() > 0:
        print(f"ℹ️  {low_var.sum()} features with very low variance (< 0.01)")

    # Outliers (values > 5 SD from mean)
    z_scores = np.abs((X - X.mean(axis=0)) / (X.std(axis=0) + 1e-10))
    outlier_cells = (z_scores > 5).sum()
    if outlier_cells > 0:
        outlier_participants = (z_scores > 5).any(axis=1).sum()
        print(
            f"ℹ️  {outlier_cells} outlier cells (> 5 SD) in {outlier_participants} participants"
        )
    else:
        print("✓  No extreme outliers")

    return issues


def _run_outer_fold(
    split_idx,
    train_idx,
    test_idx,
    X,
    y,
    age_group,
    pipeline,
    n_outer_folds,
    n_repeats,
    random_state,
):
    repeat = split_idx // n_outer_folds + 1
    fold = split_idx % n_outer_folds + 1
    inner_seed = None if random_state is None else int(
        random_state) + (repeat - 1)

    estimator = clone(pipeline)
    estimator.fit(
        X[train_idx],
        y[train_idx],
        sgl__age_group=age_group[train_idx],
        sgl__inner_random_state=inner_seed,
    )

    y_test_pred = estimator.predict(X[test_idx])
    y_train_pred = estimator.predict(X[train_idx])

    return {
        "split_idx": split_idx,
        "repeat": repeat,
        "fold": fold,
        "test_idx": test_idx,
        "y_test_pred": y_test_pred,
        "y_train_pred": y_train_pred,
        "test_mae": mean_absolute_error(y[test_idx], y_test_pred),
        "test_r2": r2_score(y[test_idx], y_test_pred),
        "train_mae": mean_absolute_error(y[train_idx], y_train_pred),
        "train_r2": r2_score(y[train_idx], y_train_pred),
        "estimator": estimator,
    }


def run_sgl_analysis(
    X,
    y,
    groups,
    group_names,
    feature_names,
    age_group=None,
    l1_ratios=None,
    n_outer_folds=5,
    n_inner_folds=5,
    n_repeats=5,
    outer_n_jobs=-1,
    inner_n_jobs=1,
    random_state=42,
    scaler_method="robust",
    n_alphas=100,
    eps=1e-3,
):
    """
    Run full Sparse Group LASSO analysis with nested CV.

    Args:
        X: Feature matrix
        y: Target (age)
        groups: List of arrays with feature indices per group
        group_names: List of group names
        feature_names: List of feature names
        age_group: Array-like age group labels for stratification
        l1_ratios: List of l1_ratio values to test
        n_outer_folds: Number of outer CV folds
        n_inner_folds: Number of inner CV folds
        n_repeats: Number of repeated outer CV runs
        outer_n_jobs: Number of parallel outer folds (joblib)
        inner_n_jobs: Number of parallel jobs in inner CV (SGLCV)
        random_state: Random seed

    Returns:
        results: Dictionary with all results
    """

    if l1_ratios is None:
        l1_ratios = list(SGL_L1_RATIOS)

    # Create pipeline
    pipeline = Pipeline(
        [
            ("scaler", make_scaler(scaler_method)),
            (
                "sgl",
                StratifiedPatchedSGLCV(
                    groups=groups,
                    l1_ratio=l1_ratios,
                    eps=eps,
                    n_alphas=n_alphas,
                    cv=n_inner_folds,
                    scoring="neg_mean_absolute_error",
                    n_jobs=inner_n_jobs,
                    verbose=0,
                ),
            ),
        ]
    )

    # Outer CV
    if age_group is None:
        raise ValueError("age_group is required for stratified outer CV.")
    outer_splits = make_repeated_stratified_splits(
        age_group, n_outer_folds, n_repeats, random_state=random_state
    )

    print("\n" + "=" * 60)
    print("RUNNING NESTED CROSS-VALIDATION")
    print("=" * 60)
    print(f"Outer folds: {n_outer_folds}")
    print(f"Inner folds: {n_inner_folds}")
    print(f"Outer repeats: {n_repeats}")
    print(f"Scaler: {scaler_method}")
    print(f"eps: {eps}")
    print(f"n_alphas: {n_alphas}")
    print(f"l1_ratios tested: {l1_ratios}")
    print(f"Outer parallel jobs: {outer_n_jobs}")
    print(f"Inner CV jobs per fold: {inner_n_jobs}")
    print("This may take a few minutes...")
    print("Progress: joblib will report completed folds below.")

    summarize_fold_composition(age_group, outer_splits)

    # Run nested CV in parallel with correct age_group subsetting per fold
    n_total = len(outer_splits)
    split_tasks = [
        (split_idx, train_idx, test_idx)
        for split_idx, (train_idx, test_idx) in enumerate(outer_splits)
    ]

    with parallel_backend("loky", n_jobs=outer_n_jobs):
        results_list = Parallel(n_jobs=outer_n_jobs, verbose=10)(
            delayed(_run_outer_fold)(
                split_idx,
                train_idx,
                test_idx,
                X,
                y,
                age_group,
                pipeline,
                n_outer_folds,
                n_repeats,
                random_state,
            )
            for split_idx, train_idx, test_idx in split_tasks
        )

    results_list = sorted(results_list, key=lambda r: r["split_idx"])
    test_mae = np.array([r["test_mae"] for r in results_list])
    test_r2 = np.array([r["test_r2"] for r in results_list])
    train_mae = np.array([r["train_mae"] for r in results_list])
    train_r2 = np.array([r["train_r2"] for r in results_list])
    estimators = [r["estimator"] for r in results_list]

    y_pred_oof = np.zeros_like(y, dtype=float)
    y_pred_counts = np.zeros_like(y, dtype=float)
    for r in results_list:
        y_pred_oof[r["test_idx"]] += r["y_test_pred"]
        y_pred_counts[r["test_idx"]] += 1

    y_pred_oof /= y_pred_counts  # Average across repeats

    print("\n" + "=" * 60)
    print("NESTED CV RESULTS")
    print("=" * 60)
    print(f"Test MAE:  {test_mae.mean():.2f} ± {test_mae.std():.2f} years")
    print(f"Test R²:   {test_r2.mean():.3f} ± {test_r2.std():.3f}")
    print(f"Train MAE: {train_mae.mean():.2f} ± {train_mae.std():.2f} years")
    print(f"Train R²:  {train_r2.mean():.3f} ± {train_r2.std():.3f}")

    # Confidence intervals (optional)
    try:
        from scipy import stats

        mae_ci = stats.t.interval(
            0.95, len(test_mae) - 1, loc=test_mae.mean(), scale=stats.sem(test_mae)
        )
        r2_ci = stats.t.interval(
            0.95, len(test_r2) - 1, loc=test_r2.mean(), scale=stats.sem(test_r2)
        )
        print(f"95% CI (Test MAE): {mae_ci[0]:.2f} to {mae_ci[1]:.2f} years")
        print(f"95% CI (Test R²):  {r2_ci[0]:.3f} to {r2_ci[1]:.3f}")
    except Exception as exc:
        print(f"CI not computed (scipy unavailable): {exc}")

    # Repeat-level stability
    print("\n" + "-" * 40)
    print("REPEAT-LEVEL STABILITY (mean per repeat)")
    print("-" * 40)
    for repeat in range(n_repeats):
        start = repeat * n_outer_folds
        end = (repeat + 1) * n_outer_folds
        rep_mae = test_mae[start:end].mean()
        rep_r2 = test_r2[start:end].mean()
        print(f"  Repeat {repeat + 1}: MAE {rep_mae:.2f} | R² {rep_r2:.3f}")

    gap = train_r2.mean() - test_r2.mean()
    print(f"\nTrain-Test R² Gap: {gap:.3f}")
    if gap > 0.15:
        print("⚠️  Warning: Possible overfitting (gap > 0.15)")
    else:
        print("✓  Gap acceptable")

    # Analyze stability across folds
    stability = analyze_fold_stability(
        estimators, feature_names, groups, group_names
    )

    alpha_grid = None
    if estimators:
        sgl0 = estimators[0].named_steps["sgl"]
        alpha_grid = getattr(sgl0, "alphas_", None)
        if alpha_grid is not None:
            alpha_grid = np.array(alpha_grid, dtype=float)
        else:
            alpha_grid = np.array(
                sorted({fp["alpha"]
                       for fp in stability["fold_params"] if fp["alpha"] > 0}),
                dtype=float,
            )

    return {
        "test_mae": test_mae,
        "test_r2": test_r2,
        "train_mae": train_mae,
        "train_r2": train_r2,
        "stability": stability,
        "estimators": estimators,
        "outer_splits": outer_splits,
        "y_pred_oof": y_pred_oof,
        "pipeline": pipeline,
        "alpha_grid": alpha_grid,
        "l1_ratio_grid": np.array(l1_ratios, dtype=float),
    }


def _run_lasso_outer_fold(
    split_idx,
    train_idx,
    test_idx,
    X,
    y,
    age_group,
    n_outer_folds,
    random_state,
    scaler_method,
    n_inner_folds,
    n_alphas,
    eps,
):
    repeat = split_idx // n_outer_folds + 1
    inner_seed = None if random_state is None else int(
        random_state) + (repeat - 1)
    inner_cv = StratifiedKFold(
        n_splits=n_inner_folds,
        shuffle=True,
        random_state=inner_seed,
    )
    inner_splits = list(
        inner_cv.split(
            np.zeros(len(train_idx)),
            age_group[train_idx],
        )
    )

    pipeline = Pipeline(
        [
            ("scaler", make_scaler(scaler_method)),
            (
                "lasso",
                LassoCV(
                    eps=eps,
                    n_alphas=n_alphas,
                    cv=inner_splits,
                    random_state=inner_seed,
                    n_jobs=1,
                    max_iter=10000,
                ),
            ),
        ]
    )

    pipeline.fit(X[train_idx], y[train_idx])
    y_test_pred = pipeline.predict(X[test_idx])
    y_train_pred = pipeline.predict(X[train_idx])
    return {
        "split_idx": split_idx,
        "test_idx": test_idx,
        "y_test_pred": y_test_pred,
        "test_mae": mean_absolute_error(y[test_idx], y_test_pred),
        "test_r2": r2_score(y[test_idx], y_test_pred),
        "train_mae": mean_absolute_error(y[train_idx], y_train_pred),
        "train_r2": r2_score(y[train_idx], y_train_pred),
    }


def run_lasso_baseline(
    X,
    y,
    age_group,
    n_outer_folds=5,
    n_inner_folds=5,
    n_repeats=1,
    random_state=42,
    scaler_method="robust",
    n_alphas=100,
    eps=1e-3,
    outer_n_jobs=-1,
):
    """Nested-CV LASSO baseline with the same stratified outer folds as SGL."""
    if age_group is None:
        raise ValueError(
            "age_group is required for stratified LASSO baseline CV.")

    outer_splits = make_repeated_stratified_splits(
        age_group, n_outer_folds, n_repeats, random_state=random_state
    )
    split_tasks = [
        (split_idx, train_idx, test_idx)
        for split_idx, (train_idx, test_idx) in enumerate(outer_splits)
    ]

    print("\n" + "=" * 60)
    print("RUNNING LASSO BASELINE (NESTED CV)")
    print("=" * 60)
    print(f"Outer folds: {n_outer_folds}, repeats: {n_repeats}")
    print(f"Inner folds: {n_inner_folds}")
    print(f"n_alphas: {n_alphas}, eps: {eps}")

    with parallel_backend("loky", n_jobs=outer_n_jobs):
        results_list = Parallel(n_jobs=outer_n_jobs, verbose=10)(
            delayed(_run_lasso_outer_fold)(
                split_idx=split_idx,
                train_idx=train_idx,
                test_idx=test_idx,
                X=X,
                y=y,
                age_group=age_group,
                n_outer_folds=n_outer_folds,
                random_state=random_state,
                scaler_method=scaler_method,
                n_inner_folds=n_inner_folds,
                n_alphas=n_alphas,
                eps=eps,
            )
            for split_idx, train_idx, test_idx in split_tasks
        )
    results_list = sorted(results_list, key=lambda r: r["split_idx"])

    test_mae = np.array([r["test_mae"] for r in results_list], dtype=float)
    test_r2 = np.array([r["test_r2"] for r in results_list], dtype=float)
    train_mae = np.array([r["train_mae"] for r in results_list], dtype=float)
    train_r2 = np.array([r["train_r2"] for r in results_list], dtype=float)

    y_pred_oof = np.zeros_like(y, dtype=float)
    y_pred_counts = np.zeros_like(y, dtype=float)
    for r in results_list:
        y_pred_oof[r["test_idx"]] += r["y_test_pred"]
        y_pred_counts[r["test_idx"]] += 1
    y_pred_oof /= y_pred_counts

    print(
        f"LASSO Test MAE: {test_mae.mean():.2f} ± {test_mae.std():.2f} years")
    print(f"LASSO Test R²:  {test_r2.mean():.3f} ± {test_r2.std():.3f}")

    return {
        "test_mae": test_mae,
        "test_r2": test_r2,
        "train_mae": train_mae,
        "train_r2": train_r2,
        "y_pred_oof": y_pred_oof,
    }


def benchmark_grouping_tiers_vs_lasso(
    X,
    y,
    feature_names,
    age_group,
    n_outer_folds,
    n_inner_folds,
    n_repeats,
    random_state,
    scaler_method,
    n_alphas,
    eps,
):
    """
    Empirically compare Tier 1 vs Tier 2 SGL groupings and plain LASSO baseline.
    """
    benchmark_rows = []

    for tier in ("tier1", "tier2"):
        groups_tier, group_names_tier, _ = create_feature_groups(
            feature_names=feature_names,
            grouping_tier=tier,
            strict_assignment=True,
        )
        tier_results = run_sgl_analysis(
            X=X,
            y=y,
            groups=groups_tier,
            group_names=group_names_tier,
            feature_names=feature_names,
            age_group=age_group,
            l1_ratios=SGL_L1_RATIOS,
            n_outer_folds=n_outer_folds,
            n_inner_folds=n_inner_folds,
            n_repeats=n_repeats,
            random_state=random_state,
            scaler_method=scaler_method,
            n_alphas=n_alphas,
            eps=eps,
        )
        benchmark_rows.append(
            {
                "model": f"SGL_{tier}",
                "n_groups": len(groups_tier),
                "test_mae_mean": float(tier_results["test_mae"].mean()),
                "test_mae_std": float(tier_results["test_mae"].std()),
                "test_r2_mean": float(tier_results["test_r2"].mean()),
                "test_r2_std": float(tier_results["test_r2"].std()),
            }
        )

    lasso_results = run_lasso_baseline(
        X=X,
        y=y,
        age_group=age_group,
        n_outer_folds=n_outer_folds,
        n_inner_folds=n_inner_folds,
        n_repeats=n_repeats,
        random_state=random_state,
        scaler_method=scaler_method,
        n_alphas=n_alphas,
        eps=eps,
    )
    benchmark_rows.append(
        {
            "model": "LASSO_baseline",
            "n_groups": 0,
            "test_mae_mean": float(lasso_results["test_mae"].mean()),
            "test_mae_std": float(lasso_results["test_mae"].std()),
            "test_r2_mean": float(lasso_results["test_r2"].mean()),
            "test_r2_std": float(lasso_results["test_r2"].std()),
        }
    )

    benchmark_df = (
        pd.DataFrame(benchmark_rows)
        .sort_values(["test_mae_mean", "test_r2_mean"], ascending=[True, False])
        .reset_index(drop=True)
    )
    benchmark_df.to_csv(GROUPING_BENCHMARK_CSV, index=False)

    print("\n" + "=" * 60)
    print("GROUPING TIER + LASSO BENCHMARK")
    print("=" * 60)
    print(benchmark_df.to_string(index=False))
    print(f"Saved benchmark summary to: {GROUPING_BENCHMARK_CSV}")

    return benchmark_df


def analyze_fold_stability(estimators, feature_names, groups, group_names):
    """Analyze feature/group selection stability across CV folds."""
    n_folds = len(estimators)
    n_features = len(feature_names)
    n_groups = len(groups)

    # Track selections
    feature_selections = np.zeros((n_folds, n_features))
    group_selections = np.zeros((n_folds, n_groups))
    coefficients = np.zeros((n_folds, n_features))

    fold_params = []

    for fold_idx, estimator in enumerate(estimators):
        sgl = estimator.named_steps["sgl"]
        coef = sgl.coef_

        coefficients[fold_idx] = coef
        feature_selections[fold_idx] = (coef != 0).astype(int)

        for g_idx, g_indices in enumerate(groups):
            if len(g_indices) > 0 and np.any(coef[g_indices] != 0):
                group_selections[fold_idx, g_idx] = 1

        fold_params.append(
            {
                "l1_ratio": sgl.l1_ratio_,
                "alpha": sgl.alpha_,
                "n_selected": (coef != 0).sum(),
            }
        )

    # Compute stability metrics
    feature_stability = feature_selections.mean(axis=0)
    group_stability = group_selections.mean(axis=0)

    # Hyperparameter consistency
    l1_ratios = [p["l1_ratio"] for p in fold_params]
    n_selected = [p["n_selected"] for p in fold_params]
    print(f"\nSelected l1_ratios across folds: {l1_ratios}")
    print(f"Features selected per fold: {n_selected}")

    # Group stability
    print("\n" + "-" * 40)
    print("GROUP STABILITY (selected in X% of folds)")
    print("-" * 40)
    for name, stab in sorted(zip(group_names, group_stability), key=lambda x: -x[1]):
        if stab > 0:
            status = "✓" if stab >= 0.8 else ("~" if stab >= 0.5 else "○")
            print(f"  {status} {name}: {stab:.0%}")

    # Most stable features
    stable_mask = feature_stability >= 0.8
    print(f"\nFeatures selected in ≥80% of folds: {stable_mask.sum()}")

    return {
        "feature_stability": feature_stability,
        "group_stability": group_stability,
        "feature_selections": feature_selections,
        "group_selections": group_selections,
        "coefficients": coefficients,
        "fold_params": fold_params,
    }


def _nogueira_phi(selection_matrix):
    selection_matrix = np.asarray(selection_matrix, dtype=float)
    if selection_matrix.ndim != 2:
        raise ValueError("selection_matrix must be a 2D array.")
    m, p = selection_matrix.shape
    if m < 2 or p == 0:
        return np.nan

    kbar = selection_matrix.sum(axis=1).mean()
    denom = (kbar / p) * (1 - (kbar / p))
    var_j = selection_matrix.var(axis=0, ddof=1)
    var_mean = var_j.mean()

    if denom <= 0:
        return 1.0 if var_mean == 0 else np.nan

    phi = 1 - (var_mean / denom)
    return float(np.clip(phi, -1.0, 1.0))


def _nogueira_label(phi):
    if np.isnan(phi):
        return "undefined"
    if phi > 0.75:
        return "excellent"
    if phi >= 0.40:
        return "intermediate_to_good"
    return "poor"


def compute_nogueira_index(selection_matrix, n_bootstraps=1000, random_state=42):
    """Compute Nogueira stability index with bootstrap CI and one-sided p-value."""
    selection_matrix = np.asarray(selection_matrix, dtype=int)
    m, p = selection_matrix.shape
    phi = _nogueira_phi(selection_matrix)

    rng = np.random.RandomState(random_state)
    boot_vals = []
    if m >= 2:
        for _ in range(int(n_bootstraps)):
            row_idx = rng.choice(m, size=m, replace=True)
            boot_phi = _nogueira_phi(selection_matrix[row_idx])
            if not np.isnan(boot_phi):
                boot_vals.append(boot_phi)
    boot_vals = np.array(boot_vals, dtype=float)

    if boot_vals.size > 0:
        ci_low, ci_high = np.percentile(boot_vals, [2.5, 97.5])
        p_value = (np.sum(boot_vals <= 0) + 1) / (len(boot_vals) + 1)
    else:
        ci_low, ci_high, p_value = np.nan, np.nan, np.nan

    return {
        "phi": phi,
        "ci_low": float(ci_low) if not np.isnan(ci_low) else np.nan,
        "ci_high": float(ci_high) if not np.isnan(ci_high) else np.nan,
        "p_value": float(p_value) if not np.isnan(p_value) else np.nan,
        "mean_subset_size": float(selection_matrix.sum(axis=1).mean()) if m > 0 else np.nan,
        "n_runs": int(m),
        "n_features": int(p),
        "interpretation": _nogueira_label(phi),
    }


def compute_nogueira_convergence(selection_matrix, checkpoints):
    """Compute Nogueira index at incremental run counts."""
    selection_matrix = np.asarray(selection_matrix, dtype=int)
    n_runs = selection_matrix.shape[0]
    rows = []
    for m in sorted(set(checkpoints)):
        if m > n_runs or m < 2:
            continue
        phi_m = _nogueira_phi(selection_matrix[:m])
        rows.append({"runs": int(m), "phi": phi_m,
                    "label": _nogueira_label(phi_m)})
    return pd.DataFrame(rows)


def _coerce_numeric_grid(grid):
    """Flatten nested numeric structures into a sorted unique float array."""
    raw = np.asarray(grid, dtype=object).ravel()
    flat = []
    for item in raw:
        if isinstance(item, (list, tuple, np.ndarray, pd.Series)):
            flat.extend(np.asarray(item, dtype=float).ravel().tolist())
        else:
            flat.append(float(item))
    out = np.array(flat, dtype=float)
    out = out[np.isfinite(out)]
    return np.array(sorted(set(out.tolist())), dtype=float)


def _snap_to_grid(value, grid, prefer_higher=True):
    grid = _coerce_numeric_grid(grid)
    if grid.size == 0:
        raise ValueError("Grid cannot be empty.")
    dist = np.abs(grid - float(value))
    min_d = dist.min()
    candidates = grid[np.isclose(dist, min_d)]
    if candidates.size == 1:
        return float(candidates[0])
    return float(candidates.max() if prefer_higher else candidates.min())


def select_cpss_operating_point_from_nested_cv(fold_params, alpha_grid, l1_grid):
    """Select fixed CPSS operating point from nested-CV fold parameters."""
    if not fold_params:
        raise ValueError(
            "fold_params is empty; cannot select CPSS operating point.")

    l1_vals = np.array([fp["l1_ratio"] for fp in fold_params], dtype=float)
    alpha_vals = np.array([fp["alpha"] for fp in fold_params], dtype=float)
    alpha_vals = alpha_vals[alpha_vals > 0]
    if alpha_vals.size == 0:
        raise ValueError("No positive alpha values found in fold_params.")

    l1_median = float(np.median(l1_vals))
    l1_star = _snap_to_grid(l1_median, l1_grid, prefer_higher=True)

    log_alpha_median = float(np.median(np.log(alpha_vals)))
    alpha_median = float(np.exp(log_alpha_median))
    alpha_star = _snap_to_grid(alpha_median, alpha_grid, prefer_higher=True)

    return {
        "l1_ratio_median": l1_median,
        "alpha_log_median": log_alpha_median,
        "alpha_median": alpha_median,
        "l1_ratio_star": l1_star,
        "alpha_star": alpha_star,
    }


def _group_selection_from_coef(coef, groups):
    sel = np.zeros(len(groups), dtype=int)
    for g_idx, g_indices in enumerate(groups):
        if len(g_indices) > 0 and np.any(coef[g_indices] != 0):
            sel[g_idx] = 1
    return sel


def _validate_half_sample(age_group, idx, min_count_per_group=2):
    group_counts = (
        pd.Series(age_group[idx])
        .value_counts()
        .reindex(VALID_AGE_GROUPS)
        .fillna(0)
        .astype(int)
    )
    if (group_counts < min_count_per_group).any():
        return False, group_counts
    return True, group_counts


def generate_stratified_complementary_pairs(
    age_group,
    n_pairs,
    min_count_per_group=2,
    random_state=42,
    max_retries=5000,
    verbose=True,
):
    """Generate complementary half-sample pairs with stratification checks."""
    age_group = np.asarray(age_group)
    n_samples = len(age_group)
    if n_samples < 2:
        raise ValueError("Need at least 2 samples for CPSS pairing.")

    pairs = []
    seen = set()
    rng = np.random.RandomState(random_state)
    retries = 0
    attempts = 0

    while len(pairs) < n_pairs and attempts < max_retries:
        attempts += 1
        split_seed = int(rng.randint(0, 2**31 - 1))
        sss = StratifiedShuffleSplit(
            n_splits=1, test_size=0.5, random_state=split_seed)
        half_a, half_b = next(sss.split(np.zeros(n_samples), age_group))

        valid_a, _ = _validate_half_sample(
            age_group, half_a, min_count_per_group=min_count_per_group
        )
        valid_b, _ = _validate_half_sample(
            age_group, half_b, min_count_per_group=min_count_per_group
        )
        if not (valid_a and valid_b):
            retries += 1
            continue

        mask = np.zeros(n_samples, dtype=bool)
        mask[half_a] = True
        key = mask.tobytes()
        comp_key = (~mask).tobytes()
        if key in seen or comp_key in seen:
            retries += 1
            continue

        seen.add(key)
        pairs.append((half_a, half_b))

    if len(pairs) < n_pairs:
        raise RuntimeError(
            f"Unable to generate {n_pairs} valid complementary pairs. "
            f"Generated {len(pairs)} after {attempts} attempts."
        )

    if verbose:
        print(
            f"Generated {len(pairs)} complementary stratified pairs "
            f"(retries: {retries}, attempts: {attempts})."
        )
    return pairs


def _run_fixed_sgl_on_subset(
    X,
    y,
    subset_idx,
    groups,
    alpha,
    l1_ratio,
    scaler_method="robust",
):
    """Fit fixed-point SGL on a subset and return coefficients."""
    pipeline = Pipeline(
        [
            ("scaler", make_scaler(scaler_method)),
            ("sgl", SGL(groups=groups, alpha=float(alpha), l1_ratio=float(l1_ratio))),
        ]
    )
    pipeline.fit(X[subset_idx], y[subset_idx])
    coef = pipeline.named_steps["sgl"].coef_
    return coef


def run_cpss_fixed_point(
    X,
    y,
    age_group,
    groups,
    group_names,
    feature_names,
    alpha,
    l1_ratio,
    scaler_method="robust",
    n_pairs=50,
    min_count_per_group=2,
    max_pair_retries=5000,
    random_state=42,
    ev_target=1.0,
    n_jobs=-1,
    pairs=None,
    log_pair_generation=True,
):
    """Run fixed-point CPSS (Option A) at a single (alpha, l1_ratio)."""
    if pairs is None:
        pairs = generate_stratified_complementary_pairs(
            age_group,
            n_pairs=n_pairs,
            min_count_per_group=min_count_per_group,
            random_state=random_state,
            max_retries=max_pair_retries,
            verbose=log_pair_generation,
        )
    else:
        pairs = list(pairs)
        if len(pairs) == 0:
            raise ValueError("pairs must be non-empty when provided.")

    n_features = len(feature_names)
    n_groups = len(groups)
    n_runs = 2 * len(pairs)
    subset_indices = [idx for half_a,
                      half_b in pairs for idx in (half_a, half_b)]

    coefs = Parallel(n_jobs=n_jobs)(
        delayed(_run_fixed_sgl_on_subset)(
            X,
            y,
            subset_idx,
            groups=groups,
            alpha=alpha,
            l1_ratio=l1_ratio,
            scaler_method=scaler_method,
        )
        for subset_idx in subset_indices
    )

    feature_selections = np.zeros((n_runs, n_features), dtype=int)
    group_selections = np.zeros((n_runs, n_groups), dtype=int)
    selected_feature_counts = np.zeros(n_runs, dtype=int)
    selected_group_counts = np.zeros(n_runs, dtype=int)

    for run_idx, coef in enumerate(coefs):
        feat_sel = (coef != 0).astype(int)
        grp_sel = _group_selection_from_coef(coef, groups)

        feature_selections[run_idx] = feat_sel
        group_selections[run_idx] = grp_sel
        selected_feature_counts[run_idx] = int(feat_sel.sum())
        selected_group_counts[run_idx] = int(grp_sel.sum())

    pi_feature = feature_selections.mean(axis=0)
    pi_group = group_selections.mean(axis=0)
    q_hat = float(np.mean(selected_feature_counts))
    qg_hat = float(np.mean(selected_group_counts))

    p = float(n_features)
    g = float(n_groups)
    pi_thr_feature_raw = 0.5 * (1.0 + ((q_hat ** 2) / (p * ev_target)))
    pi_thr_group_raw = 0.5 * (1.0 + ((qg_hat ** 2) / (g * ev_target)))

    lo, hi = CPSS_PI_REPORT_CLAMP
    pi_thr_feature_report = float(np.clip(pi_thr_feature_raw, lo, hi))
    pi_thr_group_report = float(np.clip(pi_thr_group_raw, lo, hi))

    return {
        "alpha": float(alpha),
        "l1_ratio": float(l1_ratio),
        "feature_selections": feature_selections,
        "group_selections": group_selections,
        "pi_feature": pi_feature,
        "pi_group": pi_group,
        "q_hat": q_hat,
        "qg_hat": qg_hat,
        "pi_thr_feature_raw": float(pi_thr_feature_raw),
        "pi_thr_group_raw": float(pi_thr_group_raw),
        "pi_thr_feature_report": pi_thr_feature_report,
        "pi_thr_group_report": pi_thr_group_report,
        "n_pairs": int(len(pairs)),
        "n_runs": int(n_runs),
        "pairs": pairs,
    }


def run_cpss_with_alpha_fallback(
    X,
    y,
    age_group,
    groups,
    group_names,
    feature_names,
    alpha_grid,
    l1_ratio,
    alpha_start,
    scaler_method="robust",
    n_pairs=50,
    min_count_per_group=2,
    max_pair_retries=5000,
    random_state=42,
    ev_target=1.0,
    n_jobs=-1,
):
    """Run fixed-point CPSS and increase alpha until feature-level feasibility is met."""
    alpha_grid = _coerce_numeric_grid(alpha_grid)
    alpha_current = _snap_to_grid(alpha_start, alpha_grid, prefer_higher=True)
    start_idx = int(np.argmin(np.abs(alpha_grid - alpha_current)))
    history = []
    attempt = 0
    step = 1
    tried_idx = set()

    # Generate complementary pairs once and reuse for all alpha attempts.
    shared_pairs = generate_stratified_complementary_pairs(
        age_group,
        n_pairs=n_pairs,
        min_count_per_group=min_count_per_group,
        random_state=random_state,
        max_retries=max_pair_retries,
        verbose=True,
    )

    idx = start_idx

    while True:
        if idx in tried_idx:
            raise RuntimeError(
                "CPSS alpha fallback revisited same index; aborting.")
        tried_idx.add(idx)
        alpha_current = float(alpha_grid[idx])
        attempt += 1

        cpss = run_cpss_fixed_point(
            X,
            y,
            age_group=age_group,
            groups=groups,
            group_names=group_names,
            feature_names=feature_names,
            alpha=alpha_current,
            l1_ratio=l1_ratio,
            scaler_method=scaler_method,
            n_pairs=n_pairs,
            min_count_per_group=min_count_per_group,
            max_pair_retries=max_pair_retries,
            random_state=random_state,
            ev_target=ev_target,
            n_jobs=n_jobs,
            pairs=shared_pairs,
            log_pair_generation=False,
        )
        print(
            f"CPSS attempt {attempt}: alpha={alpha_current:.6g}, "
            f"q_hat={cpss['q_hat']:.2f}, "
            f"pi_thr_feature_raw={cpss['pi_thr_feature_raw']:.3f}"
        )
        history.append(
            {
                "alpha": cpss["alpha"],
                "pi_thr_feature_raw": cpss["pi_thr_feature_raw"],
                "q_hat": cpss["q_hat"],
            }
        )
        if cpss["pi_thr_feature_raw"] <= 1.0:
            cpss["alpha_fallback_history"] = history
            return cpss

        if idx >= len(alpha_grid) - 1:
            raise RuntimeError(
                "CPSS infeasible at all available alpha values "
                f"(feature pi_thr_raw last={cpss['pi_thr_feature_raw']:.3f})."
            )

        # Exponential stepping reaches strong alpha values quickly.
        next_idx = min(idx + step, len(alpha_grid) - 1)
        if next_idx == idx:
            next_idx = min(idx + 1, len(alpha_grid) - 1)
        idx = next_idx
        step = max(1, step * 2)


def plot_cpss_stability_paths(cpss_results, feature_names, group_names):
    """Plot fixed-point CPSS selection probabilities for features and groups."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax1 = axes[0]
    sorted_idx_feat = np.argsort(cpss_results["pi_feature"])[::-1]
    sorted_pi_feat = cpss_results["pi_feature"][sorted_idx_feat]
    ax1.plot(sorted_pi_feat, lw=1.8, color="#1f77b4")
    ax1.axhline(
        cpss_results["pi_thr_feature_report"],
        color="red",
        linestyle="--",
        label=f"Threshold (report)={cpss_results['pi_thr_feature_report']:.2f}",
    )
    ax1.set_title("CPSS Feature Selection Probabilities")
    ax1.set_xlabel("Features (sorted)")
    ax1.set_ylabel("Selection Probability")
    ax1.set_ylim(0, 1)
    ax1.legend()

    ax2 = axes[1]
    sorted_idx_group = np.argsort(cpss_results["pi_group"])[::-1]
    sorted_pi_group = cpss_results["pi_group"][sorted_idx_group]
    sorted_group_names = [group_names[i] for i in sorted_idx_group]
    ax2.bar(range(len(sorted_pi_group)), sorted_pi_group, color="#2ca02c")
    ax2.axhline(
        cpss_results["pi_thr_group_report"],
        color="red",
        linestyle="--",
        label=f"Threshold (report)={cpss_results['pi_thr_group_report']:.2f}",
    )
    ax2.set_title("CPSS Group Selection Probabilities")
    ax2.set_xlabel("Groups (sorted)")
    ax2.set_ylabel("Selection Probability")
    ax2.set_ylim(0, 1)
    if len(sorted_group_names) <= 25:
        ax2.set_xticks(range(len(sorted_group_names)))
        ax2.set_xticklabels(sorted_group_names, rotation=90, fontsize=7)
    else:
        ax2.set_xticks([])
    ax2.legend()

    plt.tight_layout()
    plt.savefig("cpss_stability_paths.png", dpi=150, bbox_inches="tight")
    _finalize_plot(fig)
    return fig


def plot_nogueira_convergence(convergence_df):
    """Plot Nogueira index convergence over CPSS runs."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for source, sub in convergence_df.groupby("source"):
        ax.plot(sub["runs"], sub["phi"], marker="o", label=source)
    ax.axhline(0.75, linestyle="--", color="green", alpha=0.4)
    ax.axhline(0.40, linestyle="--", color="orange", alpha=0.4)
    ax.set_xlabel("Number of Runs")
    ax.set_ylabel("Nogueira Phi")
    ax.set_title("Nogueira Convergence")
    ax.set_ylim(-0.2, 1.0)
    ax.legend()
    plt.tight_layout()
    plt.savefig("nogueira_convergence.png", dpi=150, bbox_inches="tight")
    _finalize_plot(fig)
    return fig


def plot_group_vs_feature_stability(summary_df):
    """Plot feature-vs-group Nogueira values for primary and supplementary sources."""
    fig, ax = plt.subplots(figsize=(8, 5))
    pivot = summary_df.pivot(index="source", columns="level", values="phi")
    pivot = pivot.reindex(
        ["primary_cpss", "supplementary_cv"]).dropna(how="all")
    pivot.plot(kind="bar", ax=ax, color=["#1f77b4", "#2ca02c"])
    ax.axhline(0.75, linestyle="--", color="green", alpha=0.4)
    ax.axhline(0.40, linestyle="--", color="orange", alpha=0.4)
    ax.set_ylabel("Nogueira Phi")
    ax.set_title("Group vs Feature Stability")
    ax.set_xlabel("Source")
    ax.set_ylim(-0.2, 1.0)
    plt.tight_layout()
    plt.savefig("group_vs_feature_stability.png", dpi=150, bbox_inches="tight")
    _finalize_plot(fig)
    return fig


def compute_reliability_score(global_stab, cross_seed_std):
    """Higher = more reliable feature (stable and consistent across seeds)."""
    consistency = 1 - np.clip(cross_seed_std / 0.3, 0, 1)
    return global_stab * consistency


def run_multi_seed_stability_analysis(
    X,
    y,
    groups,
    group_names,
    feature_names,
    age_group,
    n_seeds=10,
    base_seeds=None,
    n_outer_folds=5,
    n_inner_folds=5,
    n_repeats=5,
    l1_ratios=None,
    outer_n_jobs=-1,
    inner_n_jobs=1,
    scaler_method="robust",
    n_alphas=100,
    eps=1e-3,
):
    """
    Run the full SGL analysis multiple times with different random seeds
    to assess stability of feature selection across random initializations.

    Returns aggregated stability metrics across all seeds.
    """
    if l1_ratios is None:
        l1_ratios = list(SGL_L1_RATIOS)

    if base_seeds is None:
        base_seeds = [
            42,
            123,
            456,
            789,
            1011,
            1213,
            1415,
            1617,
            1819,
            2021,
        ][:n_seeds]
    elif n_seeds > len(base_seeds):
        raise ValueError("n_seeds exceeds length of base_seeds")
    else:
        base_seeds = base_seeds[:n_seeds]

    # Storage for results across seeds
    all_coefficients = []
    all_feature_selections = []
    all_group_selections = []
    all_test_mae = []
    all_test_r2 = []
    seed_results = []

    print("=" * 70)
    print(f"MULTI-SEED STABILITY ANALYSIS ({n_seeds} independent seeds)")
    print("=" * 70)

    for seed_idx, seed in enumerate(base_seeds):
        print(f"\n{'=' * 60}")
        print(f"SEED {seed_idx + 1}/{n_seeds}: random_state = {seed}")
        print(f"{'=' * 60}")

        results = run_sgl_analysis(
            X,
            y,
            groups,
            group_names,
            feature_names,
            age_group=age_group,
            l1_ratios=l1_ratios,
            n_outer_folds=n_outer_folds,
            n_inner_folds=n_inner_folds,
            n_repeats=n_repeats,
            outer_n_jobs=outer_n_jobs,
            inner_n_jobs=inner_n_jobs,
            random_state=seed,
            scaler_method=scaler_method,
            n_alphas=n_alphas,
            eps=eps,
        )

        stability = results["stability"]
        all_coefficients.append(stability["coefficients"])
        all_feature_selections.append(stability["feature_selections"])
        all_group_selections.append(stability["group_selections"])
        all_test_mae.append(results["test_mae"])
        all_test_r2.append(results["test_r2"])

        seed_results.append(
            {
                "seed": seed,
                "mean_test_mae": results["test_mae"].mean(),
                "std_test_mae": results["test_mae"].std(),
                "mean_test_r2": results["test_r2"].mean(),
                "std_test_r2": results["test_r2"].std(),
                "feature_stability": stability["feature_stability"],
                "group_stability": stability["group_stability"],
            }
        )

    print("\n" + "=" * 70)
    print("AGGREGATING RESULTS ACROSS ALL SEEDS")
    print("=" * 70)

    all_coef_stacked = np.vstack(all_coefficients)
    all_selections_stacked = np.vstack(all_feature_selections)
    all_group_sel_stacked = np.vstack(all_group_selections)

    global_feature_stability = all_selections_stacked.mean(axis=0)
    global_group_stability = all_group_sel_stacked.mean(axis=0)

    mean_coef_global = all_coef_stacked.mean(axis=0)
    std_coef_global = all_coef_stacked.std(axis=0)

    with np.errstate(divide="ignore", invalid="ignore"):
        coef_cv = np.where(
            np.abs(mean_coef_global) > 1e-10,
            std_coef_global / np.abs(mean_coef_global),
            np.inf,
        )

    all_mae_flat = np.concatenate(all_test_mae)
    all_r2_flat = np.concatenate(all_test_r2)

    print(f"\nTotal folds analyzed: {len(all_coef_stacked)}")
    print(f"  ({n_seeds} seeds × {n_repeats} repeats × {n_outer_folds} folds)")

    print("\n" + "-" * 50)
    print("PERFORMANCE ACROSS ALL SEEDS")
    print("-" * 50)
    print(f"MAE:  {all_mae_flat.mean():.2f} ± {all_mae_flat.std():.2f} years")
    print(f"R²:   {all_r2_flat.mean():.3f} ± {all_r2_flat.std():.3f}")

    print("\n" + "-" * 50)
    print("PER-SEED SUMMARY")
    print("-" * 50)
    for sr in seed_results:
        print(
            f"  Seed {sr['seed']:5d}: MAE = {sr['mean_test_mae']:.2f} ± "
            f"{sr['std_test_mae']:.2f}, R² = {sr['mean_test_r2']:.3f} ± "
            f"{sr['std_test_r2']:.3f}"
        )

    print("\n" + "-" * 50)
    print("GLOBAL FEATURE STABILITY (across all seeds)")
    print("-" * 50)

    stability_thresholds = [0.9, 0.8, 0.7, 0.5]
    for thresh in stability_thresholds:
        n_stable = (global_feature_stability >= thresh).sum()
        print(f"  Features selected in ≥{thresh:.0%} of all folds: {n_stable}")

    highly_stable_mask = global_feature_stability >= 0.8
    highly_stable_idx = np.where(highly_stable_mask)[0]

    print(f"\n{'=' * 50}")
    print("HIGHLY STABLE FEATURES (≥80% selection rate globally)")
    print(f"{'=' * 50}")

    stable_df = pd.DataFrame()
    if len(highly_stable_idx) > 0:
        stable_features = []
        for idx in highly_stable_idx:
            stable_features.append(
                {
                    "feature": feature_names[idx],
                    "global_stability": global_feature_stability[idx],
                    "mean_coef": mean_coef_global[idx],
                    "std_coef": std_coef_global[idx],
                    "coef_cv": coef_cv[idx],
                }
            )

        stable_df = pd.DataFrame(stable_features).sort_values(
            "global_stability", ascending=False
        )

        print(f"\n{len(stable_df)} features are highly stable:\n")
        for _, row in stable_df.iterrows():
            direction = "+" if row["mean_coef"] > 0 else "-"
            print(
                f"  {row['global_stability']:.1%} | "
                f"{direction}{abs(row['mean_coef']):.4f} ± {row['std_coef']:.4f} | "
                f"{row['feature']}"
            )
    else:
        print("  No features reached 80% stability threshold.")

    print(f"\n{'=' * 50}")
    print("GLOBAL GROUP STABILITY")
    print(f"{'=' * 50}")
    for name, stab in sorted(
        zip(group_names, global_group_stability), key=lambda x: -x[1]
    ):
        if stab > 0:
            status = (
                "✓✓"
                if stab >= 0.9
                else ("✓" if stab >= 0.8 else ("~" if stab >= 0.5 else "○"))
            )
            print(f"  {status} {name}: {stab:.1%}")

    print(f"\n{'=' * 50}")
    print("CROSS-SEED CONSISTENCY CHECK")
    print(f"{'=' * 50}")

    per_seed_stability = np.array(
        [sr["feature_stability"] for sr in seed_results]
    )

    seed_stability_std = per_seed_stability.std(axis=0)
    consistent_features = seed_stability_std < 0.15

    print(
        "Features with consistent selection rate across seeds "
        f"(std < 0.15): {consistent_features.sum()}"
    )
    print(
        "Features with inconsistent selection rate "
        f"(std ≥ 0.15): {(~consistent_features).sum()}"
    )

    problematic = (global_feature_stability > 0.3) & (seed_stability_std > 0.2)
    if problematic.sum() > 0:
        print("\n⚠️  Potentially unstable features (selected often but inconsistently):")
        for idx in np.where(problematic)[0][:10]:
            print(
                f"    {feature_names[idx]}: "
                f"mean={global_feature_stability[idx]:.1%}, "
                f"std={seed_stability_std[idx]:.2f}"
            )

    return {
        "seed_results": seed_results,
        "global_feature_stability": global_feature_stability,
        "global_group_stability": global_group_stability,
        "mean_coef_global": mean_coef_global,
        "std_coef_global": std_coef_global,
        "coef_cv": coef_cv,
        "all_coefficients": all_coef_stacked,
        "all_feature_selections": all_selections_stacked,
        "all_group_selections": all_group_sel_stacked,
        "per_seed_stability": per_seed_stability,
        "seed_stability_std": seed_stability_std,
        "stable_features_df": stable_df,
        "all_test_mae": all_mae_flat,
        "all_test_r2": all_r2_flat,
    }


def plot_multi_seed_stability(multi_seed_results, feature_names, group_names):
    """Visualize stability across multiple seeds."""

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # 1. Performance distribution across all seeds
    ax1 = axes[0, 0]
    seed_results = multi_seed_results["seed_results"]
    seeds = [sr["seed"] for sr in seed_results]
    maes = [sr["mean_test_mae"] for sr in seed_results]
    mae_stds = [sr["std_test_mae"] for sr in seed_results]

    ax1.errorbar(range(len(seeds)), maes, yerr=mae_stds, fmt="o-", capsize=5)
    ax1.axhline(
        np.mean(maes), color="red", linestyle="--", label=f"Mean = {np.mean(maes):.2f}"
    )
    ax1.fill_between(
        range(len(seeds)),
        np.mean(maes) - np.std(maes),
        np.mean(maes) + np.std(maes),
        alpha=0.2,
        color="red",
    )
    ax1.set_xticks(range(len(seeds)))
    ax1.set_xticklabels([f"{s}" for s in seeds], rotation=45)
    ax1.set_xlabel("Random Seed")
    ax1.set_ylabel("MAE (years)")
    ax1.set_title("Performance Stability Across Seeds")
    ax1.legend()

    # 2. Global feature stability histogram
    ax2 = axes[0, 1]
    global_stability = multi_seed_results["global_feature_stability"]
    ax2.hist(global_stability, bins=20, edgecolor="black", alpha=0.7)
    ax2.axvline(0.8, color="green", linestyle="--", label="80% threshold")
    ax2.axvline(0.5, color="orange", linestyle="--", label="50% threshold")
    ax2.set_xlabel("Selection Frequency (across all folds & seeds)")
    ax2.set_ylabel("Number of Features")
    ax2.set_title("Global Feature Stability Distribution")
    ax2.legend()

    # 3. Cross-seed consistency vs mean stability
    ax3 = axes[0, 2]
    seed_stability_std = multi_seed_results["seed_stability_std"]
    colors = np.where(
        seed_stability_std < 0.15,
        "green",
        np.where(seed_stability_std < 0.25, "orange", "red"),
    )
    ax3.scatter(
        global_stability,
        seed_stability_std,
        c=colors,
        alpha=0.5,
        edgecolors="none",
    )
    ax3.axhline(0.15, color="green", linestyle="--", alpha=0.5)
    ax3.axhline(0.25, color="orange", linestyle="--", alpha=0.5)
    ax3.set_xlabel("Mean Selection Rate")
    ax3.set_ylabel("Std of Selection Rate Across Seeds")
    ax3.set_title("Cross-Seed Consistency\n(Lower std = more consistent)")

    # 4. Per-seed stability heatmap for top features
    ax4 = axes[1, 0]
    per_seed_stability = multi_seed_results["per_seed_stability"]

    top_idx = np.argsort(global_stability)[::-1][:20]
    top_stability = per_seed_stability[:, top_idx]
    top_names = [feature_names[i][-25:] for i in top_idx]

    im = ax4.imshow(top_stability.T, aspect="auto",
                    cmap="YlGnBu", vmin=0, vmax=1)
    ax4.set_xticks(range(len(seed_results)))
    ax4.set_xticklabels(
        [f"S{sr['seed']}" for sr in seed_results], rotation=45, fontsize=8
    )
    ax4.set_yticks(range(len(top_names)))
    ax4.set_yticklabels(top_names, fontsize=8)
    ax4.set_xlabel("Random Seed")
    ax4.set_title("Per-Seed Stability (Top 20 Features)")
    plt.colorbar(im, ax=ax4, label="Selection Rate")

    # 5. Group stability across seeds
    ax5 = axes[1, 1]
    global_group_stability = multi_seed_results["global_group_stability"]

    sorted_idx = np.argsort(global_group_stability)[::-1]
    sorted_names = [
        group_names[i] for i in sorted_idx if global_group_stability[i] > 0
    ]
    sorted_stab = [
        global_group_stability[i]
        for i in sorted_idx
        if global_group_stability[i] > 0
    ]

    colors = [
        "#2ecc71" if s >= 0.8 else "#f39c12" if s >= 0.5 else "#e74c3c"
        for s in sorted_stab
    ]
    ax5.barh(range(len(sorted_names)), sorted_stab, color=colors)
    ax5.set_yticks(range(len(sorted_names)))
    ax5.set_yticklabels(sorted_names, fontsize=9)
    ax5.invert_yaxis()
    ax5.axvline(0.8, color="green", linestyle="--", alpha=0.5)
    ax5.axvline(0.5, color="orange", linestyle="--", alpha=0.5)
    ax5.set_xlabel("Global Selection Rate")
    ax5.set_title("Group Stability Across All Seeds")
    ax5.set_xlim(0, 1)

    # 6. Coefficient stability for selected features
    ax6 = axes[1, 2]
    mean_coef = multi_seed_results["mean_coef_global"]
    std_coef = multi_seed_results["std_coef_global"]

    nonzero_mask = np.abs(mean_coef) > 1e-6
    if nonzero_mask.sum() > 0:
        nonzero_idx = np.where(nonzero_mask)[0]
        sorted_by_coef = nonzero_idx[
            np.argsort(np.abs(mean_coef[nonzero_idx]))[::-1]
        ][:15]

        coefs = mean_coef[sorted_by_coef]
        stds = std_coef[sorted_by_coef]
        names = [feature_names[i][-25:] for i in sorted_by_coef]

        colors = ["#2ecc71" if c > 0 else "#e74c3c" for c in coefs]
        y_pos = range(len(names))
        ax6.barh(y_pos, coefs, xerr=stds, color=colors, capsize=3, alpha=0.7)
        ax6.set_yticks(y_pos)
        ax6.set_yticklabels(names, fontsize=8)
        ax6.invert_yaxis()
        ax6.axvline(0, color="black", linestyle="-", linewidth=0.5)
        ax6.set_xlabel("Mean Coefficient ± Std")
        ax6.set_title("Coefficient Stability (Top 15 Features)")

    plt.tight_layout()
    plt.savefig("multi_seed_stability_analysis.png",
                dpi=150, bbox_inches="tight")
    _finalize_plot(fig)

    return fig


def train_final_model(
    X,
    y,
    groups,
    age_group=None,
    l1_ratios=None,
    scaler_method="robust",
    n_alphas=100,
    eps=1e-3,
):
    """Train final model on full data."""

    print("\n" + "=" * 60)
    print("TRAINING FINAL MODEL ON FULL DATA")
    print("=" * 60)
    print("Fitting final model (inner CV with stratification)...")

    if l1_ratios is None:
        l1_ratios = list(SGL_L1_RATIOS)

    pipeline = Pipeline(
        [
            ("scaler", make_scaler(scaler_method)),
            (
                "sgl",
                StratifiedPatchedSGLCV(
                    groups=groups,
                    l1_ratio=l1_ratios,
                    eps=eps,
                    n_alphas=n_alphas,
                    cv=5,
                    scoring="neg_mean_absolute_error",
                    verbose=1,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    with parallel_backend("multiprocessing", n_jobs=-1):
        pipeline.fit(
            X, y, sgl__age_group=age_group, sgl__inner_random_state=42
        )

    sgl = pipeline.named_steps["sgl"]
    coef = sgl.coef_

    print(f"Final l1_ratio: {sgl.l1_ratio_}")
    print(f"Final alpha: {sgl.alpha_:.4f}")
    print(f"Non-zero coefficients: {(coef != 0).sum()} / {len(coef)}")

    return pipeline, coef


def _compute_region_and_band_importance(feature_names, coef):
    """Aggregate absolute coefficient importance by region and by band."""
    abs_coef = np.abs(np.asarray(coef, dtype=float))
    region_importance = {region: 0.0 for region in REGIONS}
    band_importance = {band: 0.0 for band in BANDS}

    for feature_name, coef_value in zip(feature_names, abs_coef):
        if coef_value == 0:
            continue
        metadata = _parse_feature_metadata(feature_name)
        if metadata is None:
            continue

        region = metadata.get("region")
        if region in region_importance:
            region_importance[region] += float(coef_value)

        region_pair = metadata.get("region_pair")
        if region_pair:
            pair_regions = [r for r in region_pair.split(
                "_") if r in region_importance]
            if pair_regions:
                share = float(coef_value) / float(len(pair_regions))
                for region_name in pair_regions:
                    region_importance[region_name] += share

        band = metadata.get("band")
        if band in band_importance:
            band_importance[band] += float(coef_value)

    region_df = pd.DataFrame(
        [{"region": region, "importance": value}
         for region, value in region_importance.items()]
    ).sort_values("importance", ascending=False)

    band_order = ["theta", "alpha", "beta", "broadband"]
    band_df = pd.DataFrame(
        [{"band": band, "importance": band_importance.get(band, 0.0)}
         for band in band_order]
    ).sort_values("importance", ascending=False)

    return region_df, band_df


def create_results_tables(coef, feature_names, groups, group_names, stability):
    """Create summary DataFrames for reporting."""

    # Feature-level summary
    feature_to_group = {}
    for g_idx, (name, indices) in enumerate(zip(group_names, groups)):
        for idx in indices:
            feature_to_group[idx] = name

    feature_df = pd.DataFrame(
        {
            "feature": feature_names,
            "coefficient": coef,
            "abs_coefficient": np.abs(coef),
            "selected": coef != 0,
            "stability": stability["feature_stability"],
            "group": [
                feature_to_group.get(i, "unknown") for i in range(len(feature_names))
            ],
        }
    )
    feature_df = feature_df.sort_values("abs_coefficient", ascending=False)

    # Group-level summary
    group_rows = []
    for name, g_indices, g_stability in zip(group_names, groups, stability["group_stability"]):
        n_features = len(g_indices)
        n_selected = int(np.sum(coef[g_indices] != 0)) if n_features > 0 else 0
        sum_abs_coef = float(
            np.sum(np.abs(coef[g_indices]))) if n_features > 0 else 0.0
        mean_stability = (
            float(stability["feature_stability"][g_indices].mean())
            if n_features > 0
            else 0.0
        )
        group_rows.append(
            {
                "group": name,
                "n_features": n_features,
                "n_selected": n_selected,
                "sum_abs_coef": sum_abs_coef,
                "mean_stability": mean_stability,
                "group_stability": float(g_stability),
            }
        )
    group_df = pd.DataFrame(group_rows)
    group_df["selection_rate"] = group_df["n_selected"] / group_df[
        "n_features"
    ].replace(0, 1)
    group_df = group_df.sort_values("sum_abs_coef", ascending=False)

    # Region/band aggregation from feature metadata (works for both tier1 and tier2).
    region_df, band_df = _compute_region_and_band_importance(
        feature_names, coef)

    return feature_df, group_df, region_df, band_df


def plot_results(coef, feature_names, groups, group_names, stability, y, y_pred=None):
    """Create comprehensive visualization of results."""

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # 1. Top features by coefficient
    ax1 = axes[0, 0]
    sorted_idx = np.argsort(np.abs(coef))[::-1]
    top_k = min(15, (coef != 0).sum())
    if top_k > 0:
        top_features = [
            feature_names[i][-30:] for i in sorted_idx[:top_k]
        ]  # Truncate names
        top_coefs = coef[sorted_idx[:top_k]]
        colors = ["#e74c3c" if c < 0 else "#2ecc71" for c in top_coefs]
        ax1.barh(range(top_k), np.abs(top_coefs), color=colors)
        ax1.set_yticks(range(top_k))
        ax1.set_yticklabels(top_features, fontsize=8)
        ax1.invert_yaxis()
    ax1.set_xlabel("|Coefficient|")
    ax1.set_title("Top Features\n(Green=positive, Red=negative)")

    # 2. Group importance
    ax2 = axes[0, 1]
    group_importance = []
    for g_idx, (name, indices) in enumerate(zip(group_names, groups)):
        if len(indices) > 0:
            importance = np.sum(np.abs(coef[indices]))
            group_importance.append((name, importance))

    group_importance.sort(key=lambda x: -x[1])
    g_names, g_vals = zip(*group_importance) if group_importance else ([], [])

    colors = []
    for name in g_names:
        group_type = _group_type_from_group_name(name)
        if group_type == "within_wpli":
            colors.append("#1f77b4")
        elif group_type == "between_wpli":
            colors.append("#17becf")
        elif group_type == "relative_power":
            colors.append("#2ca02c")
        elif group_type == "region_relative_power":
            colors.append("#98df8a")
        elif group_type == "spectral_flatness":
            colors.append("#ff7f0e")
        elif group_type == "paf":
            colors.append("#9467bd")
        elif group_type == "band_ratio":
            colors.append("#d62728")
        else:
            colors.append("#95a5a6")

    ax2.barh(range(len(g_names)), g_vals, color=colors)
    ax2.set_yticks(range(len(g_names)))
    ax2.set_yticklabels(g_names, fontsize=9)
    ax2.invert_yaxis()
    ax2.set_xlabel("Sum of |Coefficients|")
    ax2.set_title("Group Importance")

    region_df, band_df = _compute_region_and_band_importance(
        feature_names, coef)

    # 3. Region importance
    ax3 = axes[0, 2]
    region_df_plot = region_df[region_df["importance"] > 0].copy()
    if region_df_plot.empty:
        region_df_plot = region_df.copy()
    ax3.bar(region_df_plot["region"],
            region_df_plot["importance"], color="coral")
    ax3.set_ylabel("Sum of |Coefficients|")
    ax3.set_title("Importance by Brain Region")
    ax3.tick_params(axis="x", rotation=45)

    # 4. Band importance
    ax4 = axes[1, 0]
    band_color_map = {
        "theta": "#2ecc71",
        "alpha": "#3498db",
        "beta": "#e74c3c",
        "broadband": "#95a5a6",
    }
    band_df_plot = band_df.sort_values(
        "band", key=lambda s: s.map({"theta": 0, "alpha": 1, "beta": 2, "broadband": 3})
    )
    ax4.bar(
        band_df_plot["band"],
        band_df_plot["importance"],
        color=[band_color_map.get(band, "#95a5a6")
               for band in band_df_plot["band"]],
    )
    ax4.set_ylabel("Sum of |Coefficients|")
    ax4.set_title("Importance by Frequency Band")

    # 6. Group stability
    ax6 = axes[1, 2]
    g_stab = [
        (name, stab)
        for name, stab in zip(group_names, stability["group_stability"])
        if stab > 0
    ]
    g_stab.sort(key=lambda x: -x[1])
    if g_stab:
        names, stabs = zip(*g_stab)
        colors = [
            "#2ecc71" if s >= 0.8 else "#f39c12" if s >= 0.5 else "#e74c3c"
            for s in stabs
        ]
        ax6.barh(range(len(names)), stabs, color=colors)
        ax6.set_yticks(range(len(names)))
        ax6.set_yticklabels(names, fontsize=9)
        ax6.invert_yaxis()
        ax6.axvline(0.8, color="green", linestyle="--", alpha=0.5)
        ax6.axvline(0.5, color="orange", linestyle="--", alpha=0.5)
    ax6.set_xlabel("Selection Frequency")
    ax6.set_title("Group Selection Stability")
    ax6.set_xlim(0, 1)

    plt.tight_layout()
    plt.savefig("sgl_results_summary.png", dpi=150, bbox_inches="tight")
    _finalize_plot(fig)

    return fig


def plot_predictions(y_true, y_pred, title="Actual vs Predicted Age"):
    """Plot actual vs predicted age scatter."""
    fig, ax = plt.subplots(figsize=(8, 8))

    ax.scatter(y_true, y_pred, alpha=0.5, edgecolors='none')

    # Perfect prediction line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val],
            'r--', label='Perfect prediction')

    # Regression line
    z = np.polyfit(y_true, y_pred, 1)
    p = np.poly1d(z)
    ax.plot([min_val, max_val], [p(min_val), p(max_val)],
            'b-', alpha=0.7, label='Regression line')

    # Metrics
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    ax.set_xlabel('Chronological Age (years)', fontsize=12)
    ax.set_ylabel('Predicted Age (years)', fontsize=12)
    ax.set_title(f'{title}\nMAE = {mae:.2f} years, R² = {r2:.3f}', fontsize=14)
    ax.legend()
    ax.set_aspect('equal')

    plt.tight_layout()
    plt.savefig('age_prediction_scatter.png', dpi=150, bbox_inches='tight')
    _finalize_plot(fig)

    return fig


# =============================================================================
# SECTION 6: ERROR DIAGNOSTICS
# =============================================================================


def analyze_errors_by_age_group(y_true, y_pred, age_groups):
    """Analyze prediction errors stratified by age group."""
    results = pd.DataFrame(
        {
            "true_age": y_true,
            "predicted_age": y_pred,
            "error": y_pred - y_true,
            "abs_error": np.abs(y_pred - y_true),
            "age_group": age_groups,
        }
    )

    print("=" * 60)
    print("ERROR ANALYSIS BY AGE GROUP")
    print("=" * 60)

    summary = (
        results.groupby("age_group")
        .agg(
            {
                "true_age": ["mean", "min", "max", "count"],
                "predicted_age": "mean",
                "error": ["mean", "std"],
                "abs_error": ["mean", "std"],
            }
        )
        .round(2)
    )

    print(summary)

    print("\n" + "-" * 40)
    print("REGRESSION TO MEAN DIAGNOSTIC")
    print("-" * 40)
    for group in ["Young", "Middle aged", "Old"]:
        group_data = results[results["age_group"] == group]
        if group_data.empty:
            continue
        mean_error = group_data["error"].mean()
        direction = "OVERPREDICTED" if mean_error > 0 else "UNDERPREDICTED"
        print(
            f"  {group}: {direction} by {abs(mean_error):.2f} years "
            f"(mean error = {mean_error:+.2f})"
        )

    return results


def export_participant_prediction_errors(
    participant_ids,
    y_true,
    y_pred,
    output_path="participant_prediction_errors.csv",
):
    """Export participant-level predictions and errors."""
    participant_ids = np.asarray(participant_ids)
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    if not (len(participant_ids) == len(y_true) == len(y_pred)):
        raise ValueError(
            "Length mismatch in export_participant_prediction_errors: "
            f"participant_ids={len(participant_ids)}, "
            f"y_true={len(y_true)}, y_pred={len(y_pred)}"
        )

    participant_df = pd.DataFrame(
        {
            "participant_id": participant_ids,
            "actual_age": y_true,
            "predicted_age": y_pred,
        }
    )
    participant_df["error"] = participant_df["predicted_age"] - \
        participant_df["actual_age"]
    participant_df["absolute_error"] = participant_df["error"].abs()
    participant_df.to_csv(output_path, index=False)

    print("\n" + "=" * 60)
    print("PARTICIPANT-LEVEL PREDICTION ERRORS")
    print("=" * 60)
    print(f"Saved participant error table to: {output_path}")

    return participant_df


def check_cpss_final_model_concordance(
    final_coef,
    cpss_results,
    feature_names,
    output_path="cpss_final_model_concordance.csv",
):
    """
    Compare CPSS EV=1 stable features with final full-data model selections.

    Returns summary counts and writes per-feature concordance table.
    """
    final_coef = np.asarray(final_coef)
    stable_feature_mask = cpss_results["pi_feature"] >= cpss_results["pi_thr_feature_raw"]

    cpss_stable_set = set(np.flatnonzero(stable_feature_mask))
    final_selected_set = set(np.flatnonzero(final_coef != 0))
    overlap = cpss_stable_set & final_selected_set
    final_not_cpss = final_selected_set - cpss_stable_set
    cpss_missing_in_final = cpss_stable_set - final_selected_set

    cpss_denom = max(len(cpss_stable_set), 1)
    final_denom = max(len(final_selected_set), 1)

    print("\n" + "=" * 60)
    print("CPSS vs FINAL MODEL CONCORDANCE")
    print("=" * 60)
    print(
        f"CPSS-stable features in final model: {len(overlap)}/{len(cpss_stable_set)} "
        f"({100.0 * len(overlap) / cpss_denom:.1f}%)"
    )
    print(f"Final model features NOT CPSS-stable: {len(final_not_cpss)}")
    print(
        f"CPSS-stable features missing from final model: {len(cpss_missing_in_final)}"
    )
    if final_not_cpss:
        print("⚠️  Concordance warning: final model includes features not CPSS-stable.")

    all_idx = np.arange(len(feature_names))
    concordance_df = pd.DataFrame(
        {
            "feature_index": all_idx,
            "feature": feature_names,
            "final_selected": final_coef != 0,
            "cpss_selection_probability": cpss_results["pi_feature"],
            "cpss_stable_ev1": stable_feature_mask,
        }
    )
    concordance_df["overlap_selected_and_stable"] = (
        concordance_df["final_selected"] & concordance_df["cpss_stable_ev1"]
    )
    concordance_df["final_selected_not_cpss_stable"] = (
        concordance_df["final_selected"] & ~concordance_df["cpss_stable_ev1"]
    )
    concordance_df["cpss_stable_not_selected_by_final"] = (
        concordance_df["cpss_stable_ev1"] & ~concordance_df["final_selected"]
    )
    concordance_df.to_csv(output_path, index=False)

    print(f"Saved CPSS/final concordance table to: {output_path}")

    return {
        "n_cpss_stable": int(len(cpss_stable_set)),
        "n_final_selected": int(len(final_selected_set)),
        "n_overlap": int(len(overlap)),
        "n_final_not_cpss_stable": int(len(final_not_cpss)),
        "n_cpss_stable_not_final": int(len(cpss_missing_in_final)),
        "recall_cpss_in_final": float(len(overlap) / cpss_denom),
        "precision_final_vs_cpss": float(len(overlap) / final_denom),
    }


def plot_regression_to_mean_diagnostics(y_true, y_pred, age_groups):
    """Visualize regression to mean and age-dependent bias."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    ax1 = axes[0, 0]
    colors = {"Young": "#2ecc71", "Middle aged": "#3498db", "Old": "#e74c3c"}
    for group in ["Young", "Middle aged", "Old"]:
        mask = age_groups == group
        if not np.any(mask):
            continue
        ax1.scatter(
            y_true[mask],
            y_pred[mask],
            c=colors[group],
            label=group,
            alpha=0.6,
            edgecolors="none",
        )

    ax1.plot(
        [y_true.min(), y_true.max()],
        [y_true.min(), y_true.max()],
        "k--",
        label="Perfect prediction",
    )

    slope, intercept = np.polyfit(y_true, y_pred, 1)
    ax1.plot(
        [y_true.min(), y_true.max()],
        [slope * y_true.min() + intercept, slope * y_true.max() + intercept],
        "r-",
        linewidth=2,
        label=f"Fit (slope={slope:.2f})",
    )

    ax1.set_xlabel("Chronological Age")
    ax1.set_ylabel("Predicted Age")
    ax1.set_title(
        "Predictions by Age Group\n"
        f"Slope = {slope:.2f} (1.0 = no regression to mean)"
    )
    ax1.legend()
    ax1.set_aspect("equal")

    ax2 = axes[0, 1]
    errors = y_pred - y_true
    for group in ["Young", "Middle aged", "Old"]:
        mask = age_groups == group
        if not np.any(mask):
            continue
        ax2.scatter(
            y_true[mask],
            errors[mask],
            c=colors[group],
            label=group,
            alpha=0.6,
            edgecolors="none",
        )

    ax2.axhline(0, color="black", linestyle="--")
    ax2.set_xlabel("Chronological Age")
    ax2.set_ylabel("Prediction Error (Predicted - Actual)")
    ax2.set_title("Signed Error vs Age\n(Positive = overprediction)")

    z = np.polyfit(y_true, errors, 1)
    p = np.poly1d(z)
    ax2.plot(
        [y_true.min(), y_true.max()],
        [p(y_true.min()), p(y_true.max())],
        "r-",
        linewidth=2,
    )
    ax2.legend()

    ax3 = axes[1, 0]
    error_df = pd.DataFrame({"error": errors, "age_group": age_groups})
    error_df["age_group"] = pd.Categorical(
        error_df["age_group"], categories=["Young", "Middle aged", "Old"]
    )
    error_df.boxplot(column="error", by="age_group", ax=ax3)
    ax3.axhline(0, color="red", linestyle="--")
    ax3.set_xlabel("Age Group")
    ax3.set_ylabel("Prediction Error")
    ax3.set_title("Error Distribution by Age Group")
    plt.suptitle("")

    ax4 = axes[1, 1]
    for group in ["Young", "Middle aged", "Old"]:
        mask = age_groups == group
        if not np.any(mask):
            continue
        ax4.hist(
            errors[mask], bins=20, alpha=0.5, label=group, color=colors[group]
        )
    ax4.axvline(0, color="black", linestyle="--")
    ax4.set_xlabel("Brain-PAD (Predicted - Actual Age)")
    ax4.set_ylabel("Count")
    ax4.set_title("Brain-PAD Distribution by Age Group")
    ax4.legend()

    plt.tight_layout()
    plt.savefig("regression_to_mean_diagnostics.png",
                dpi=150, bbox_inches="tight")
    _finalize_plot(fig)

    print(f"\nPrediction slope: {slope:.3f}")
    print(
        f"  → A slope of {slope:.2f} means for every 10 years of true age difference,"
    )
    print(f"    the model only predicts {slope * 10:.1f} years difference.")
    print(
        f"  → This represents {(1 - slope) * 100:.1f}% regression to the mean.")

    return slope


def quantify_regression_to_mean(y_true, y_pred):
    """Calculate key metrics for regression to mean."""
    slope, intercept = np.polyfit(y_true, y_pred, 1)
    error_age_corr = np.corrcoef(y_true, y_pred - y_true)[0, 1]
    mean_age = y_true.mean()

    print("=" * 60)
    print("REGRESSION TO MEAN QUANTIFICATION")
    print("=" * 60)
    print(f"Sample mean age: {mean_age:.1f} years")
    print(f"Prediction slope: {slope:.3f}")
    print("  → Perfect model: slope = 1.0")
    print(f"  → Your model: {(1 - slope) * 100:.1f}% shrinkage toward mean")
    print(f"\nCorrelation(true_age, error): {error_age_corr:.3f}")
    print("  → Perfect model: r = 0.0")
    print("  → Negative r means young overpredicted, old underpredicted")

    return {
        "slope": slope,
        "intercept": intercept,
        "error_age_corr": error_age_corr,
    }


def export_stable_features_cv(
    stability,
    feature_names,
    feature_to_group,
    threshold=0.8,
    output_path="selected_features_stable_cv.csv",
):
    """
    Export features that are stably selected across outer CV folds.

    This list is intended as the primary evidence of age-specific, robust features,
    independent of any full-data fit.
    """
    feature_stability = stability["feature_stability"]
    coefficients = stability["coefficients"]

    mean_coef = coefficients.mean(axis=0)
    std_coef = coefficients.std(axis=0)

    stable_mask = feature_stability >= threshold
    stable_idx = np.where(stable_mask)[0]

    stable_df = (
        pd.DataFrame(
            {
                "feature": [feature_names[i] for i in stable_idx],
                "group": [feature_to_group.get(i, "unknown") for i in stable_idx],
                "stability": feature_stability[stable_idx],
                "mean_coef": mean_coef[stable_idx],
                "std_coef": std_coef[stable_idx],
                "abs_mean_coef": np.abs(mean_coef[stable_idx]),
            }
        )
        .sort_values(["stability", "abs_mean_coef"], ascending=False)
        .reset_index(drop=True)
    )

    stable_df.to_csv(output_path, index=False)

    print("\n" + "=" * 60)
    print("STABILITY-BASED FEATURE LIST (CV ONLY)")
    print("=" * 60)
    print(
        f"Threshold: >={threshold:.0%} of outer folds | "
        f"Selected: {len(stable_df)} / {len(feature_names)}"
    )
    print(f"Saved to: {output_path}")

    return stable_df


def extract_selected_features(pipeline, feature_names, groups, group_names):
    """Extract and organize the features selected by SGL."""
    sgl = pipeline.named_steps["sgl"]
    coef = sgl.coef_

    feature_to_group = {}
    for g_idx, (name, indices) in enumerate(zip(group_names, groups)):
        for idx in indices:
            feature_to_group[idx] = name

    selected_mask = coef != 0
    selected_idx = np.where(selected_mask)[0]

    selected_df = (
        pd.DataFrame(
            {
                "feature_index": selected_idx,
                "feature_name": [feature_names[i] for i in selected_idx],
                "coefficient": coef[selected_idx],
                "abs_coefficient": np.abs(coef[selected_idx]),
                "group": [feature_to_group.get(i, "unknown") for i in selected_idx],
            }
        )
        .sort_values("abs_coefficient", ascending=False)
        .reset_index(drop=True)
    )

    print("=" * 60)
    print(f"SELECTED FEATURES: {len(selected_idx)} / {len(feature_names)}")
    print("=" * 60)

    print("\nSelected features per group:")
    group_summary = (
        selected_df.groupby("group")
        .agg(
            n_selected=("feature_name", "count"),
            total_importance=("abs_coefficient", "sum"),
        )
        .sort_values("total_importance", ascending=False)
    )
    print(group_summary)

    print("\n" + "-" * 40)
    print("TOP 20 FEATURES BY |COEFFICIENT|")
    print("-" * 40)
    for _, row in selected_df.head(20).iterrows():
        direction = "+" if row["coefficient"] > 0 else "-"
        print(
            f"  {direction} {row['feature_name']}: "
            f"{row['coefficient']:.4f} ({row['group']})"
        )

    return selected_df, coef


# =============================================================================
# SECTION 6: STATISTICAL VALIDATION
# =============================================================================

def run_permutation_test(pipeline, X, y, n_permutations=1000, n_jobs=-1):
    """Run permutation test to validate model significance."""
    print("\n" + "=" * 60)
    print("RUNNING PERMUTATION TEST")
    print("=" * 60)
    print(f"Number of permutations: {n_permutations}")
    print("This may take several minutes...")

    score, perm_scores, p_value = permutation_test_score(
        pipeline, X, y,
        scoring='neg_mean_absolute_error',
        cv=5,
        n_permutations=n_permutations,
        n_jobs=n_jobs,
        random_state=42
    )

    print(f"\nTrue MAE: {-score:.2f} years")
    print(
        f"Permuted MAE (mean ± SD): {-perm_scores.mean():.2f} ± {perm_scores.std():.2f} years")
    print(f"p-value: {p_value:.4f}")

    if p_value < 0.001:
        print("✓ Highly significant (p < 0.001)")
    elif p_value < 0.01:
        print("✓ Significant (p < 0.01)")
    elif p_value < 0.05:
        print("✓ Significant (p < 0.05)")
    else:
        print("⚠️ Not significant (p ≥ 0.05)")

    # Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(-perm_scores, bins=30, alpha=0.7, label='Permuted')
    ax.axvline(-score, color='red', linestyle='--', linewidth=2,
               label=f'True MAE = {-score:.2f}')
    ax.set_xlabel('MAE (years)')
    ax.set_ylabel('Count')
    ax.set_title(f'Permutation Test (p = {p_value:.4f})')
    ax.legend()
    plt.tight_layout()
    plt.savefig('permutation_test.png', dpi=150, bbox_inches='tight')
    _finalize_plot(fig)

    return score, perm_scores, p_value


if __name__ == '__main__':
    df = pd.read_csv(DEFAULT_DATA_PATH)
    X, y, feature_names, df = load_and_prepare_data(df)
    check_data_quality(X, feature_names)
    age_groups = df["age_group"].values

    if RUN_GROUPING_BENCHMARK:
        benchmark_grouping_tiers_vs_lasso(
            X=X,
            y=y,
            feature_names=feature_names,
            age_group=age_groups,
            n_outer_folds=SGL_OUTER_FOLDS,
            n_inner_folds=SGL_INNER_FOLDS,
            n_repeats=GROUPING_BENCHMARK_REPEATS,
            random_state=SGL_RANDOM_STATE,
            scaler_method=SCALER_METHOD,
            n_alphas=SGL_N_ALPHAS,
            eps=SGL_EPS,
        )

    print(f"\nUsing grouping tier: {GROUPING_TIER}")
    groups, group_names, feature_to_group = create_feature_groups(
        feature_names,
        grouping_tier=GROUPING_TIER,
        strict_assignment=True,
    )
    results = run_sgl_analysis(
        X,
        y,
        groups,
        group_names,
        feature_names,
        age_group=age_groups,
        l1_ratios=SGL_L1_RATIOS,
        n_outer_folds=SGL_OUTER_FOLDS,
        n_inner_folds=SGL_INNER_FOLDS,
        n_repeats=SGL_OUTER_REPEATS,
        random_state=SGL_RANDOM_STATE,
        scaler_method=SCALER_METHOD,
        n_alphas=SGL_N_ALPHAS,
        eps=SGL_EPS,
    )
    stable_cv_df = export_stable_features_cv(
        results["stability"],
        feature_names,
        feature_to_group,
        threshold=STABILITY_THRESHOLD,
        output_path=STABLE_FEATURES_CSV,
    )

    # Primary stability workflow: fixed-point CPSS + Nogueira
    alpha_grid = results.get("alpha_grid")
    if alpha_grid is None or len(alpha_grid) == 0:
        alpha_grid = np.array(
            sorted(
                {
                    fp["alpha"]
                    for fp in results["stability"]["fold_params"]
                    if fp["alpha"] > 0
                }
            ),
            dtype=float,
        )
    if alpha_grid.size == 0:
        raise ValueError("Could not determine alpha grid for CPSS fallback.")

    operating = select_cpss_operating_point_from_nested_cv(
        results["stability"]["fold_params"],
        alpha_grid=alpha_grid,
        l1_grid=SGL_L1_RATIOS,
    )

    cpss_results = None
    nogueira_feature_primary = None
    nogueira_group_primary = None
    convergence_df = pd.DataFrame()

    if RUN_CPSS:
        cpss_results = run_cpss_with_alpha_fallback(
            X,
            y,
            age_group=age_groups,
            groups=groups,
            group_names=group_names,
            feature_names=feature_names,
            alpha_grid=alpha_grid,
            l1_ratio=operating["l1_ratio_star"],
            alpha_start=operating["alpha_star"],
            scaler_method=SCALER_METHOD,
            n_pairs=CPSS_N_PAIRS,
            min_count_per_group=CPSS_MIN_COUNT_PER_GROUP,
            max_pair_retries=CPSS_MAX_PAIR_RETRIES,
            random_state=CPSS_RANDOM_STATE,
            ev_target=CPSS_EV_TARGET,
            n_jobs=CPSS_N_JOBS,
        )

        stable_feature_mask = cpss_results["pi_feature"] >= cpss_results["pi_thr_feature_raw"]
        stable_group_mask = cpss_results["pi_group"] >= cpss_results["pi_thr_group_raw"]

        feature_prob_df = pd.DataFrame(
            {
                "feature": feature_names,
                "group": [feature_to_group.get(i, "unknown") for i in range(len(feature_names))],
                "selection_probability": cpss_results["pi_feature"],
                "stable_ev1": stable_feature_mask,
            }
        ).sort_values("selection_probability", ascending=False)
        feature_prob_df.to_csv(
            "cpss_feature_selection_probabilities.csv", index=False)
        feature_prob_df[feature_prob_df["stable_ev1"]].to_csv(
            "cpss_stable_features_ev1.csv", index=False
        )

        group_prob_df = pd.DataFrame(
            {
                "group": group_names,
                "selection_probability": cpss_results["pi_group"],
                "stable_ev1": stable_group_mask,
            }
        ).sort_values("selection_probability", ascending=False)
        group_prob_df.to_csv(
            "cpss_group_selection_probabilities.csv", index=False)
        group_prob_df[group_prob_df["stable_ev1"]].to_csv(
            "cpss_stable_groups_ev1.csv", index=False
        )

        operating_payload = {
            "nested_cv_selected": {
                "alpha_star": float(operating["alpha_star"]),
                "l1_ratio_star": float(operating["l1_ratio_star"]),
                "alpha_median": float(operating["alpha_median"]),
                "l1_ratio_median": float(operating["l1_ratio_median"]),
            },
            "final_cpss_point": {
                "alpha": float(cpss_results["alpha"]),
                "l1_ratio": float(cpss_results["l1_ratio"]),
                "q_hat": float(cpss_results["q_hat"]),
                "qg_hat": float(cpss_results["qg_hat"]),
                "pi_thr_feature_raw": float(cpss_results["pi_thr_feature_raw"]),
                "pi_thr_group_raw": float(cpss_results["pi_thr_group_raw"]),
                "pi_thr_feature_report": float(cpss_results["pi_thr_feature_report"]),
                "pi_thr_group_report": float(cpss_results["pi_thr_group_report"]),
            },
            "alpha_fallback_history": cpss_results.get("alpha_fallback_history", []),
        }
        with open("cpss_operating_point.json", "w", encoding="utf-8") as f:
            json.dump(operating_payload, f, indent=2)

        nogueira_feature_primary = compute_nogueira_index(
            cpss_results["feature_selections"],
            n_bootstraps=NOGUEIRA_BOOTSTRAPS,
            random_state=NOGUEIRA_RANDOM_STATE,
        )
        nogueira_group_primary = compute_nogueira_index(
            cpss_results["group_selections"],
            n_bootstraps=NOGUEIRA_BOOTSTRAPS,
            random_state=NOGUEIRA_RANDOM_STATE,
        )

        convergence_feature = compute_nogueira_convergence(
            cpss_results["feature_selections"], NOGUEIRA_CONVERGENCE_CHECKPOINTS
        )
        convergence_group = compute_nogueira_convergence(
            cpss_results["group_selections"], NOGUEIRA_CONVERGENCE_CHECKPOINTS
        )
        convergence_feature["source"] = "feature_primary_cpss"
        convergence_group["source"] = "group_primary_cpss"
        convergence_df = pd.concat(
            [convergence_feature, convergence_group], ignore_index=True
        )
        convergence_df.to_csv("nogueira_convergence.csv", index=False)

        plot_cpss_stability_paths(cpss_results, feature_names, group_names)
        plot_nogueira_convergence(convergence_df)

    # Supplementary Nogueira from nested-CV folds (kept separate from CPSS)
    nogueira_feature_cv = compute_nogueira_index(
        results["stability"]["feature_selections"],
        n_bootstraps=NOGUEIRA_BOOTSTRAPS,
        random_state=NOGUEIRA_RANDOM_STATE,
    )
    nogueira_group_cv = compute_nogueira_index(
        results["stability"]["group_selections"],
        n_bootstraps=NOGUEIRA_BOOTSTRAPS,
        random_state=NOGUEIRA_RANDOM_STATE,
    )

    nogueira_rows = []
    if nogueira_feature_primary is not None and nogueira_group_primary is not None:
        nogueira_rows.extend(
            [
                {"source": "primary_cpss", "level": "feature",
                    **nogueira_feature_primary},
                {"source": "primary_cpss", "level": "group",
                    **nogueira_group_primary},
            ]
        )
    nogueira_rows.extend(
        [
            {"source": "supplementary_cv", "level": "feature", **nogueira_feature_cv},
            {"source": "supplementary_cv", "level": "group", **nogueira_group_cv},
        ]
    )
    nogueira_summary_df = pd.DataFrame(nogueira_rows)
    nogueira_summary_df.to_csv("nogueira_primary_summary.csv", index=False)
    plot_group_vs_feature_stability(nogueira_summary_df)

    final_pipeline, final_coef = train_final_model(
        X,
        y,
        groups,
        age_group=age_groups,
        l1_ratios=SGL_L1_RATIOS,
        scaler_method=SCALER_METHOD,
        n_alphas=SGL_N_ALPHAS,
        eps=SGL_EPS,
    )
    feature_df, group_df, region_df, band_df = create_results_tables(
        final_coef, feature_names, groups, group_names, results['stability']
    )
    plot_results(final_coef, feature_names, groups,
                 group_names, results['stability'], y)
    y_pred_oof = results["y_pred_oof"]
    participant_error_df = export_participant_prediction_errors(
        participant_ids=df["participant_id"].values,
        y_true=y,
        y_pred=y_pred_oof,
        output_path=PARTICIPANT_PREDICTION_ERRORS_CSV,
    )
    plot_predictions(
        y, y_pred_oof, title="Out-of-Fold Predictions vs Actual Age")
    error_results = analyze_errors_by_age_group(y, y_pred_oof, age_groups)
    plot_regression_to_mean_diagnostics(y, y_pred_oof, age_groups)
    quantify_regression_to_mean(y, y_pred_oof)
    selected_df, _ = extract_selected_features(
        final_pipeline, feature_names, groups, group_names
    )
    selected_df.to_csv("selected_features.csv", index=False)

    if cpss_results is not None:
        check_cpss_final_model_concordance(
            final_coef=final_coef,
            cpss_results=cpss_results,
            feature_names=feature_names,
            output_path=CPSS_FINAL_CONCORDANCE_CSV,
        )

    if RUN_MULTI_SEED_STABILITY:
        multi_seed_results = run_multi_seed_stability_analysis(
            X,
            y,
            groups,
            group_names,
            feature_names,
            age_groups,
            n_seeds=MULTI_SEED_N_SEEDS,
            n_outer_folds=SGL_OUTER_FOLDS,
            n_inner_folds=SGL_INNER_FOLDS,
            n_repeats=SGL_OUTER_REPEATS,
            l1_ratios=SGL_L1_RATIOS,
            scaler_method=SCALER_METHOD,
            n_alphas=SGL_N_ALPHAS,
            eps=SGL_EPS,
        )
        stability_summary = (
            pd.DataFrame(
                {
                    "feature": feature_names,
                    "global_stability": multi_seed_results[
                        "global_feature_stability"
                    ],
                    "cross_seed_std": multi_seed_results["seed_stability_std"],
                    "mean_coef": multi_seed_results["mean_coef_global"],
                    "std_coef": multi_seed_results["std_coef_global"],
                }
            )
            .assign(
                reliability=lambda df: compute_reliability_score(
                    df["global_stability"], df["cross_seed_std"]
                ),
                direction=lambda df: np.where(
                    df["mean_coef"] > 0, "positive", "negative"
                ),
            )
            .sort_values("reliability", ascending=False)
        )
        stability_summary.to_csv("legacy_stability_summary.csv", index=False)
        print("\n" + "=" * 60)
        print("TOP 20 MOST RELIABLE FEATURES")
        print("=" * 60)
        for _, row in stability_summary.head(20).iterrows():
            print(
                f"  {row['reliability']:.3f} | "
                f"stab={row['global_stability']:.0%} | "
                f"std={row['cross_seed_std']:.2f} | "
                f"{row['direction']:>8} | "
                f"{row['feature']}"
            )
        plot_multi_seed_stability(
            multi_seed_results, feature_names, group_names
        )
        if not multi_seed_results["stable_features_df"].empty:
            multi_seed_results["stable_features_df"].to_csv(
                "globally_stable_features.csv", index=False
            )
    # run_permutation_test(final_pipeline, X, y, n_permutations=1000)
