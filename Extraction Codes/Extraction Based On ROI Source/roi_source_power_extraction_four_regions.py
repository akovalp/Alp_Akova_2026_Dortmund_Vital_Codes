"""
ROI-Based Source Power Extraction (Four Regions)

Computes source-space power features (band power, spectral flatness, PAF) using
the same ROI definitions as the PLI connectivity code. This ensures direct
comparability between power and connectivity features.

Key alignment with PLI pipeline:
- Same atlas (aparc/Desikan-Killiany)
- Same ROI groupings based on Desikan-Killiany labels
- Same source space resolution (ico-4)
- Same inverse method (eLORETA) with pick_ori='normal'

Outputs:
- ROI-Group Summary CSV: One row per participant with frontal/parietal/temporal/occipital aggregates
- Label-Level Detail CSV: Long format with per-label features
"""

import mne
from mne.datasets import fetch_fsaverage
from mne.minimum_norm import make_inverse_operator, apply_inverse_epochs
from mne.time_frequency import psd_array_multitaper
import numpy as np
import pandas as pd
import os
import warnings
import traceback
from datetime import datetime
from joblib import Parallel, delayed
import gc
from scipy.stats import gmean

# --- CONFIGURATION ---
N_PARTICIPANTS_JOBS = 10
MAIN_DIR = "/Users/alpmac/CodeWorks/Trento/PreProcessedData/Complete"
SESSION = "ses-1"
CONDITIONS = ["EyesClosed", "EyesOpen"]
ACQUISITIONS = ["pre", "post"]
BASE_OUTPUT_NAME_SUMMARY = 'roi_source_power_summary_four_regions'
BASE_OUTPUT_NAME_LABELS = 'roi_source_power_labels_four_regions'

# Modeling parameters (aligned with PLI code)
SNR = 3.0
LAMBDA2 = 1.0 / (SNR ** 2)

FREQUENCY_BANDS = {
    'theta': (4, 8),
    'alpha': (8, 14),
    'beta': (14, 30),
}

# Desikan-Killiany (aparc) label base-names (without -lh/-rh)
# Grouped into 4 main regions requested by the project.
#
# Notes
# - "Cingulate" regions are assigned per request:
#   - rostral/caudal anterior cingulate -> Frontal
#   - posterior/isthmus cingulate -> Parietal
Frontal_LABEL_KEYWORDS = [
    'superiorfrontal',
    'rostralmiddlefrontal',
    'caudalmiddlefrontal',
    'parsopercularis',
    'parstriangularis',
    'parsorbitalis',
    'lateralorbitofrontal',
    'medialorbitofrontal',
    'precentral',
    'paracentral',
    'frontalpole',
    'rostralanteriorcingulate',
    'caudalanteriorcingulate',
]
Parietal_LABEL_KEYWORDS = [
    'superiorparietal',
    'inferiorparietal',
    'supramarginal',
    'postcentral',
    'precuneus',
    'posteriorcingulate',
    'isthmuscingulate',
]
Temporal_LABEL_KEYWORDS = [
    'superiortemporal',
    'middletemporal',
    'inferiortemporal',
    'bankssts',
    'fusiform',
    'transversetemporal',
    'entorhinal',
    'temporalpole',
    'parahippocampal',
]
Occipital_LABEL_KEYWORDS = [
    'lateraloccipital',
    'lingual',
    'cuneus',
    'pericalcarine',
]

# Suppress warnings
warnings.filterwarnings('ignore', message='.*does not conform to MNE naming conventions.*')
warnings.filterwarnings("ignore", message="nperseg =")
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*At least one epoch has multiple events.*")
pd.options.display.float_format = '{:.7e}'.format


def get_participants_condition(main_dir, participant, session, condition, acquisition):
    """Returns the path to the epoched .set file."""
    set_path = os.path.join(main_dir, f"sub-{participant}", session, "eeg",
                            f"sub-{participant}_{session}_task-{condition}_acq-{acquisition}_eeg.set")
    if not os.path.exists(set_path):
        return None
    return set_path


def setup_global_resources(template_path):
    """
    Compute Forward Solution and load atlas labels ONCE.

    Uses ico-4 source space for consistency.
    Restricts labels to source space and builds region indices.

    Returns
    -------
    fwd : Forward
        Forward solution
    noise_cov : Covariance
        Ad-hoc noise covariance
    labels : list of Label
        All valid atlas labels (restricted to source space)
    frontal_indices : list of int
        Indices of frontal labels
    parietal_indices : list of int
        Indices of parietal labels
    temporal_indices : list of int
        Indices of temporal labels
    occipital_indices : list of int
        Indices of occipital labels
    src : SourceSpaces
        Source space
    label_names : list of str
        Names of all valid labels
    label_roi_groups : list of str
        ROI group ('frontal', 'parietal', 'temporal', 'occipital', or 'other') for each label
    """
    print(f"--- Setting up Global Resources using template: {os.path.basename(template_path)} ---")

    # Load template epochs
    epochs_template = mne.io.read_epochs_eeglab(template_path, verbose=False)
    montage = mne.channels.make_standard_montage('standard_1020')
    epochs_template.set_montage(montage, on_missing='ignore')

    # Get fsaverage paths - using ico-4 for consistency
    fs_dir = fetch_fsaverage(verbose=False)
    subjects_dir = fs_dir.parent
    src_path = fs_dir / 'bem' / 'fsaverage-ico-4-src.fif'  # ico-4 as specified
    bem = str(fs_dir / 'bem' / 'fsaverage-5120-5120-5120-bem-sol.fif')

    # Load source space
    src = mne.read_source_spaces(src_path, verbose=False)

    # Compute forward solution
    fwd = mne.make_forward_solution(
        epochs_template.info,
        trans='fsaverage',
        src=src,
        bem=bem,
        eeg=True,
        mindist=5.0,
        n_jobs=-1,
        verbose=False
    )

    noise_cov = mne.make_ad_hoc_cov(epochs_template.info, verbose=False)

    # Load atlas labels
    all_labels = mne.read_labels_from_annot(
        'fsaverage',
        parc='aparc',
        subjects_dir=subjects_dir,
        verbose=False
    )

    # Filter out labels with no vertices and restrict to source space
    excluded_labels = ['unknown', 'corpuscallosum']
    labels = []
    label_names = []
    label_roi_groups = []
    frontal_indices = []
    parietal_indices = []
    temporal_indices = []
    occipital_indices = []

    for label in all_labels:
        # Skip excluded labels
        if any(excl in label.name.lower() for excl in excluded_labels):
            continue

        # Restrict label to source space
        try:
            label_restricted = label.restrict(src)
            # Label.vertices is a 1D array of vertex indices
            if len(label_restricted.vertices) == 0:
                print(f"  Warning: {label.name} has no vertices in source space, skipping")
                continue
        except (ValueError, IndexError) as e:
            print(f"  Warning: {label.name} could not be restricted: {e}")
            continue

        # Determine ROI group
        name = label.name.lower()
        base_name = name.replace('-lh', '').replace('-rh', '')

        if base_name in Frontal_LABEL_KEYWORDS:
            roi_group = 'frontal'
            frontal_indices.append(len(labels))
        elif base_name in Parietal_LABEL_KEYWORDS:
            roi_group = 'parietal'
            parietal_indices.append(len(labels))
        elif base_name in Temporal_LABEL_KEYWORDS:
            roi_group = 'temporal'
            temporal_indices.append(len(labels))
        elif base_name in Occipital_LABEL_KEYWORDS:
            roi_group = 'occipital'
            occipital_indices.append(len(labels))
        else:
            roi_group = 'other'

        labels.append(label_restricted)
        label_names.append(label.name)
        label_roi_groups.append(roi_group)

    print(f"  Total valid labels: {len(labels)}")
    print(f"  Frontal labels: {len(frontal_indices)}")
    print(f"  Parietal labels: {len(parietal_indices)}")
    print(f"  Temporal labels: {len(temporal_indices)}")
    print(f"  Occipital labels: {len(occipital_indices)}")
    n_other = len(labels) - len(frontal_indices) - len(parietal_indices) - len(temporal_indices) - len(occipital_indices)
    print(f"  Other labels: {n_other}")

    return (
        fwd,
        noise_cov,
        labels,
        frontal_indices,
        parietal_indices,
        temporal_indices,
        occipital_indices,
        src,
        label_names,
        label_roi_groups,
    )


def compute_robust_paf(psd, freqs, alpha_range=(8, 14)):
    """
    Extract Peak Alpha Frequency with parabolic interpolation for sub-bin precision.

    Parameters
    ----------
    psd : ndarray
        Power spectral density values
    freqs : ndarray
        Corresponding frequencies
    alpha_range : tuple
        (fmin, fmax) for alpha band

    Returns
    -------
    paf : float
        Peak alpha frequency in Hz
    """
    alpha_mask = (freqs >= alpha_range[0]) & (freqs < alpha_range[1])
    alpha_psd = psd[alpha_mask]
    alpha_freqs = freqs[alpha_mask]

    if len(alpha_psd) == 0:
        return np.nan

    peak_idx = np.argmax(alpha_psd)

    # Parabolic interpolation for sub-bin precision
    if 0 < peak_idx < len(alpha_psd) - 1:
        y0, y1, y2 = alpha_psd[peak_idx-1:peak_idx+2]
        denominator = y0 - 2*y1 + y2
        if denominator != 0:
            delta = 0.5 * (y0 - y2) / denominator
            freq_resolution = alpha_freqs[1] - alpha_freqs[0]
            paf = alpha_freqs[peak_idx] + delta * freq_resolution
        else:
            paf = alpha_freqs[peak_idx]
    else:
        paf = alpha_freqs[peak_idx]

    return float(paf)


def compute_label_spectral_features(label_ts, sfreq, frequency_bands, alpha_band_key='alpha'):
    """
    Compute spectral features for each label from label time courses.

    Uses psd_array_multitaper for consistency with MNE's spectral estimation.

    Parameters
    ----------
    label_ts : ndarray
        Label time courses, shape (n_epochs, n_labels, n_times)
    sfreq : float
        Sampling frequency
    frequency_bands : dict
        Dictionary of {band_name: (fmin, fmax)}
    alpha_band_key : str
        Key for alpha band in frequency_bands (for PAF extraction)

    Returns
    -------
    features : dict
        Dictionary with keys:
        - 'power': dict of {band_name: ndarray (n_labels,)}
        - 'flatness': dict of {band_name: ndarray (n_labels,)}
        - 'paf': ndarray (n_labels,)
    """
    n_epochs, n_labels, n_times = label_ts.shape

    # Determine global frequency range
    fmin_global = min(lo for (lo, hi) in frequency_bands.values())
    fmax_global = max(hi for (lo, hi) in frequency_bands.values())

    # Compute PSD using multitaper
    # psd_array_multitaper handles 3D arrays: (n_epochs, n_labels, n_times) -> (n_epochs, n_labels, n_freqs)
    psds, freqs = psd_array_multitaper(
        label_ts,
        sfreq=sfreq,
        fmin=fmin_global,
        fmax=fmax_global,
        bandwidth=4.0,
        adaptive=False,
        low_bias=True,
        normalization='full',
        n_jobs=1,  # Already parallelized at participant level
        verbose=False
    )

    # Average PSD across epochs first
    mean_psds = np.mean(psds, axis=0)  # Shape: (n_labels, n_freqs)

    # Initialize feature containers
    power_features = {band: np.zeros(n_labels) for band in frequency_bands}
    flatness_features = {band: np.zeros(n_labels) for band in frequency_bands}
    paf_features = np.zeros(n_labels)

    # Build frequency masks
    band_masks = {
        band: (freqs >= lo) & (freqs < hi)
        for band, (lo, hi) in frequency_bands.items()
    }

    # Compute features per label
    for label_idx in range(n_labels):
        label_psd = mean_psds[label_idx, :]  # Shape: (n_freqs,)

        # Band power and flatness
        for band, mask in band_masks.items():
            if not np.any(mask):
                power_features[band][label_idx] = np.nan
                flatness_features[band][label_idx] = np.nan
                continue

            band_psd = label_psd[mask]
            band_freqs = freqs[mask]

            # Band power: integrate using trapezoid rule
            power_features[band][label_idx] = np.trapezoid(band_psd, band_freqs)

            # Spectral flatness: geometric mean / arithmetic mean
            if np.all(band_psd > 0):
                flatness_features[band][label_idx] = gmean(band_psd) / np.mean(band_psd)
            else:
                flatness_features[band][label_idx] = np.nan

        # Peak Alpha Frequency with parabolic interpolation
        alpha_lo, alpha_hi = frequency_bands[alpha_band_key]
        paf_features[label_idx] = compute_robust_paf(label_psd, freqs, (alpha_lo, alpha_hi))

    return {
        'power': power_features,
        'flatness': flatness_features,
        'paf': paf_features
    }


def process_single_condition(condition, set_file, participant_id, inverse_operator,
                             labels, src, frequency_bands, label_names, label_roi_groups,
                             frontal_idx, parietal_idx, temporal_idx, occipital_idx):
    """
    Process a single condition for one participant.

    Returns
    -------
    roi_results : dict
        Aggregated ROI-group level results for the summary CSV
    label_results : list of dict
        Per-label results for the label-level CSV
    error_info : dict or None
        Error details if processing failed
    """
    epochs = None
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning,
                                    message=".*At least one epoch has multiple events.*")
            epochs = mne.io.read_epochs_eeglab(set_file, verbose=False)

        if len(epochs) < 1:
            print(f"[{participant_id}] SKIPPING {condition}: No epochs found", flush=True)
            return {}, [], None

        montage = mne.channels.make_standard_montage('standard_1020')
        epochs.set_montage(montage, on_missing='warn')
        epochs.set_eeg_reference('average', projection=True, verbose=False)
        epochs.apply_proj(verbose=False)

        # Apply inverse solution with pick_ori='normal' (matches PLI workflow)
        stcs = apply_inverse_epochs(
            epochs,
            inverse_operator,
            LAMBDA2,
            method='eLORETA',
            pick_ori='normal',
            return_generator=False,  # Need all epochs for label extraction
            verbose=False
        )

        # Extract label time courses
        # Returns a list of arrays, one per epoch, each with shape (n_labels, n_times)
        label_ts_list = mne.extract_label_time_course(
            stcs,
            labels,
            src,
            mode='mean_flip',
            allow_empty=False,
            return_generator=False,
            verbose=False
        )

        if label_ts_list is None or len(label_ts_list) == 0:
            print(f"[{participant_id}] SKIPPING {condition}: No label time courses extracted", flush=True)
            return {}, [], None

        # Convert list to 3D array: (n_epochs, n_labels, n_times)
        label_ts = np.array(label_ts_list)

        # Compute spectral features
        sfreq = epochs.info['sfreq']
        features = compute_label_spectral_features(label_ts, sfreq, frequency_bands)

        # Build label-level results (long format)
        label_results = []
        for idx, (name, roi_group) in enumerate(zip(label_names, label_roi_groups)):
            row = {
                'Participant': participant_id,
                'Condition': condition,
                'Label': name,
                'ROI_Group': roi_group,
            }
            for band in frequency_bands.keys():
                row[f'{band}_power'] = features['power'][band][idx]
                row[f'{band}_flatness'] = features['flatness'][band][idx]
            row['paf'] = features['paf'][idx]
            label_results.append(row)

        # Build ROI-group aggregated results
        roi_results = {}

        roi_groups = {
            'frontal': frontal_idx,
            'parietal': parietal_idx,
            'temporal': temporal_idx,
            'occipital': occipital_idx,
        }

        for band in frequency_bands.keys():
            band_power_strs = []
            band_flat_strs = []

            for roi_group, roi_idx in roi_groups.items():
                roi_power = np.nanmean(features['power'][band][roi_idx]) if len(roi_idx) else np.nan
                roi_flatness = np.nanmean(features['flatness'][band][roi_idx]) if len(roi_idx) else np.nan

                roi_results[f"{condition}_{band}_{roi_group}_power"] = roi_power
                roi_results[f"{condition}_{band}_{roi_group}_flatness"] = roi_flatness

                band_power_strs.append(f"{roi_group[:1].upper()} Pwr: {roi_power:.2e}")
                band_flat_strs.append(f"{roi_group[:1].upper()} Flat: {roi_flatness:.4f}")

            print(
                f"[{participant_id}] {condition} | {band} -> "
                + ", ".join(band_power_strs)
                + " | "
                + ", ".join(band_flat_strs),
                flush=True,
            )

        # PAF aggregates - use pre-computed indices directly
        paf_strs = []
        for roi_group, roi_idx in roi_groups.items():
            roi_paf = np.nanmean(features['paf'][roi_idx]) if len(roi_idx) else np.nan
            roi_results[f"{condition}_{roi_group}_paf"] = roi_paf
            paf_strs.append(f"{roi_group[:1].upper()}: {roi_paf:.2f} Hz")

        print(f"[{participant_id}] {condition} | PAF -> " + ", ".join(paf_strs), flush=True)

        return roi_results, label_results, None

    except Exception as e:
        error_info = {
            'participant_id': participant_id,
            'condition': condition,
            'file': set_file,
            'exception_type': type(e).__name__,
            'message': str(e),
            'traceback': traceback.format_exc()
        }
        print(f"[{participant_id}] !!! ERROR in {condition}: {e}", flush=True)
        return {}, [], error_info

    finally:
        if epochs is not None:
            del epochs
        gc.collect()


def process_single_participant(participant_id, condition_files, global_fwd, global_noise_cov,
                               labels, frontal_idx, parietal_idx, temporal_idx, occipital_idx, src, frequency_bands,
                               label_names, label_roi_groups):
    """
    Process all conditions for one participant.

    Computes the Inverse Operator ONCE per participant, then processes all conditions.
    """
    print(f"[{participant_id}] Computing inverse operator once for all {len(condition_files)} conditions...", flush=True)

    # Compute inverse operator using first available file
    first_condition = list(condition_files.keys())[0]
    first_file = condition_files[first_condition]

    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning,
                                    message=".*At least one epoch has multiple events.*")
            epochs_template = mne.io.read_epochs_eeglab(first_file, verbose=False)
        montage = mne.channels.make_standard_montage('standard_1020')
        epochs_template.set_montage(montage, on_missing='warn')
        epochs_template.set_eeg_reference('average', projection=True, verbose=False)
        epochs_template.apply_proj(verbose=False)

        inverse_operator = make_inverse_operator(
            epochs_template.info,
            global_fwd,
            global_noise_cov,
            loose=0.2,
            depth=0.8,
            verbose=False
        )
        print(f"[{participant_id}] Inverse operator computed. Processing conditions...", flush=True)

        del epochs_template
        gc.collect()

    except Exception as e:
        print(f"[{participant_id}] !!! CRITICAL: Failed to create inverse operator: {e}", flush=True)
        error_info = {
            'participant_id': participant_id,
            'condition': 'INVERSE_OPERATOR_CREATION',
            'file': first_file,
            'exception_type': type(e).__name__,
            'message': str(e),
            'traceback': traceback.format_exc()
        }
        return {'Participant': participant_id}, [], [error_info]

    # Process all conditions sequentially (within participant)
    row_data = {'Participant': participant_id}
    all_label_results = []
    errors = []

    for condition, set_file in condition_files.items():
        roi_results, label_results, error = process_single_condition(
            condition, set_file, participant_id, inverse_operator,
            labels, src, frequency_bands, label_names, label_roi_groups,
            frontal_idx, parietal_idx, temporal_idx, occipital_idx
        )
        row_data.update(roi_results)
        all_label_results.extend(label_results)
        if error is not None:
            errors.append(error)

    del inverse_operator
    gc.collect()

    print(f"[{participant_id}] Finished.", flush=True)
    return row_data, all_label_results, errors


def get_expected_columns(frequency_bands, conditions, acquisitions):
    """Generate expected column names for validation."""
    expected_cols = []
    for condition in conditions:
        for acquisition in acquisitions:
            cond_label = f"{condition}_{acquisition}"
            for band in frequency_bands.keys():
                for roi_group in ['frontal', 'parietal', 'temporal', 'occipital']:
                    expected_cols.append(f"{cond_label}_{band}_{roi_group}_power")
                    expected_cols.append(f"{cond_label}_{band}_{roi_group}_flatness")
            for roi_group in ['frontal', 'parietal', 'temporal', 'occipital']:
                expected_cols.append(f"{cond_label}_{roi_group}_paf")
    return expected_cols


def load_existing_progress(summary_path, labels_path, expected_columns):
    """Load existing CSV files for resume capability."""
    summary_df = pd.DataFrame()
    labels_df = pd.DataFrame()
    completed_participants = set()

    if summary_path and os.path.exists(summary_path):
        try:
            summary_df = pd.read_csv(summary_path, dtype={'Participant': str})
            print(f"Loaded existing summary with {len(summary_df)} participants.")

            for _, row in summary_df.iterrows():
                participant_id = str(row['Participant']).zfill(3)
                is_complete = all(
                    col in summary_df.columns and pd.notna(row.get(col))
                    for col in expected_columns
                )
                if is_complete:
                    completed_participants.add(participant_id)

            print(f"Found {len(completed_participants)} completed participants to skip.")
        except Exception as e:
            print(f"Error loading summary progress: {e}")

    if labels_path and os.path.exists(labels_path):
        try:
            labels_df = pd.read_csv(labels_path, dtype={'Participant': str})
            print(f"Loaded existing labels file with {len(labels_df)} rows.")
        except Exception as e:
            print(f"Error loading labels progress: {e}")

    return summary_df, labels_df, completed_participants


def save_progress(summary_path, labels_path, all_summary_data, all_label_data):
    """Save progress to CSV files."""
    # Save summary CSV
    if all_summary_data:
        summary_df = pd.DataFrame(all_summary_data)
        formatted_df = summary_df.copy()

        for col in formatted_df.columns:
            if col == 'Participant':
                continue
            if 'power' in col.lower():
                formatted_df[col] = formatted_df[col].apply(
                    lambda x: f"{x:.7e}" if pd.notnull(x) else "")
            elif 'flatness' in col.lower():
                formatted_df[col] = formatted_df[col].apply(
                    lambda x: f"{x:.4f}" if pd.notnull(x) else "")
            elif 'paf' in col.lower():
                formatted_df[col] = formatted_df[col].apply(
                    lambda x: f"{x:.2f}" if pd.notnull(x) else "")

        formatted_df.to_csv(summary_path, index=False)
        print(f"  -> Summary saved to {summary_path} ({len(all_summary_data)} participants)")

    # Save labels CSV
    if all_label_data:
        labels_df = pd.DataFrame(all_label_data)
        formatted_labels = labels_df.copy()

        for col in formatted_labels.columns:
            if col in ['Participant', 'Condition', 'Label', 'ROI_Group']:
                continue
            if 'power' in col.lower():
                formatted_labels[col] = formatted_labels[col].apply(
                    lambda x: f"{x:.7e}" if pd.notnull(x) else "")
            elif 'flatness' in col.lower():
                formatted_labels[col] = formatted_labels[col].apply(
                    lambda x: f"{x:.4f}" if pd.notnull(x) else "")
            elif 'paf' in col.lower():
                formatted_labels[col] = formatted_labels[col].apply(
                    lambda x: f"{x:.2f}" if pd.notnull(x) else "")

        formatted_labels.to_csv(labels_path, index=False)
        print(f"  -> Labels saved to {labels_path} ({len(all_label_data)} rows)")


def get_new_output_path(base_name, extension=".csv"):
    """Generate timestamped output filename."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    return f"{base_name}_{timestamp}{extension}"


def find_latest_file(base_name, directory=".", extension=".csv"):
    """Find the latest file matching the pattern {base_name}_*.csv."""
    import glob
    import re

    pattern = os.path.join(directory, f"{base_name}_*{extension}")
    matching_files = glob.glob(pattern)

    if not matching_files:
        return None

    timestamp_pattern = re.compile(rf"{re.escape(base_name)}_(\d{{8}}_\d{{6}}){re.escape(extension)}$")

    files_with_timestamps = []
    for filepath in matching_files:
        filename = os.path.basename(filepath)
        match = timestamp_pattern.match(filename)
        if match:
            files_with_timestamps.append((filepath, match.group(1)))

    if not files_with_timestamps:
        return None

    files_with_timestamps.sort(key=lambda x: x[1], reverse=True)
    return files_with_timestamps[0][0]


if __name__ == "__main__":
    # 1. Setup paths
    NEW_SUMMARY_OUTPUT = get_new_output_path(BASE_OUTPUT_NAME_SUMMARY)
    NEW_LABELS_OUTPUT = get_new_output_path(BASE_OUTPUT_NAME_LABELS)
    print(f"Summary output: {NEW_SUMMARY_OUTPUT}")
    print(f"Labels output: {NEW_LABELS_OUTPUT}")

    # 2. Find latest existing files for resume
    latest_summary = find_latest_file(BASE_OUTPUT_NAME_SUMMARY)
    latest_labels = find_latest_file(BASE_OUTPUT_NAME_LABELS)

    if latest_summary:
        print(f"Found existing summary file: {latest_summary}")
    if latest_labels:
        print(f"Found existing labels file: {latest_labels}")

    # 3. Generate expected columns and load progress
    expected_columns = get_expected_columns(FREQUENCY_BANDS, CONDITIONS, ACQUISITIONS)
    existing_summary, existing_labels, completed_participants = load_existing_progress(
        latest_summary, latest_labels, expected_columns
    )

    all_summary_data = existing_summary.to_dict('records') if not existing_summary.empty else []
    all_label_data = existing_labels.to_dict('records') if not existing_labels.empty else []

    # 4. Scan for participants
    print(f"\nScanning {MAIN_DIR} for participants...")
    existing_folders = [f for f in os.listdir(MAIN_DIR) if f.startswith('sub-')]
    participant_ids = sorted([f.split('-')[1] for f in existing_folders])

    # 5. Validate files and filter completed participants
    valid_participants = {}
    for participant_id in participant_ids:
        if participant_id in completed_participants:
            print(f"[{participant_id}] Already complete, skipping...")
            continue

        condition_files = {}
        for condition in CONDITIONS:
            for acquisition in ACQUISITIONS:
                fpath = get_participants_condition(MAIN_DIR, participant_id, SESSION, condition, acquisition)
                if fpath:
                    condition_files[f"{condition}_{acquisition}"] = fpath

        if condition_files:
            valid_participants[participant_id] = condition_files

    print(f"Found {len(valid_participants)} participants to process.")

    if not valid_participants:
        print("\n All participants already processed. Nothing to do.")
    else:
        # 6. Setup global resources
        first_sub = list(valid_participants.keys())[0]
        first_file = list(valid_participants[first_sub].values())[0]

        (fwd, noise_cov, labels, frontal_idx, parietal_idx, temporal_idx, occipital_idx,
         src, label_names, label_roi_groups) = setup_global_resources(first_file)
        print("\nGlobal resources ready. Starting processing...\n")

        # 7. Process participants in parallel batches
        all_errors = []
        participants_list = list(valid_participants.items())
        total_participants = len(participants_list)

        for batch_start in range(0, total_participants, N_PARTICIPANTS_JOBS):
            batch_end = min(batch_start + N_PARTICIPANTS_JOBS, total_participants)
            batch = participants_list[batch_start:batch_end]

            print(f"\n{'='*60}")
            print(f"Processing batch: participants {batch_start+1}-{batch_end} of {total_participants}")
            print(f"Participants in batch: {[p[0] for p in batch]}")
            print(f"{'='*60}")

            try:
                batch_results = Parallel(n_jobs=len(batch), verbose=0)(
                    delayed(process_single_participant)(
                        participant_id, files, fwd, noise_cov,
                        labels, frontal_idx, parietal_idx, temporal_idx, occipital_idx, src, FREQUENCY_BANDS,
                        label_names, label_roi_groups
                    )
                    for participant_id, files in batch
                )

                for row_data, label_data, errors in batch_results:
                    all_summary_data.append(row_data)
                    all_label_data.extend(label_data)
                    if errors:
                        all_errors.extend(errors)

                save_progress(NEW_SUMMARY_OUTPUT, NEW_LABELS_OUTPUT, all_summary_data, all_label_data)

            except Exception as e:
                print(f"Batch error: {e}")
                traceback.print_exc()
                save_progress(NEW_SUMMARY_OUTPUT, NEW_LABELS_OUTPUT, all_summary_data, all_label_data)
                continue

        print(f"\n{'='*60}")
        print(f"Processing Complete.")
        print(f"Summary saved to: {NEW_SUMMARY_OUTPUT}")
        print(f"Labels saved to: {NEW_LABELS_OUTPUT}")
        print(f"Total participants: {len(all_summary_data)}")
        print(f"{'='*60}")

        # 8. Save error log if needed
        if all_errors:
            error_log_path = f"roi_source_power_errors_four_regions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            with open(error_log_path, 'w') as f:
                f.write(f"# ROI Source Power Extraction Error Log\n")
                f.write(f"# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"# Total Errors: {len(all_errors)}\n\n")

                for i, err in enumerate(all_errors, 1):
                    f.write(f"ERROR {i}/{len(all_errors)}\n{'─'*50}\n")
                    f.write(f"Participant: {err['participant_id']}\n")
                    f.write(f"Condition: {err['condition']}\n")
                    f.write(f"File: {err['file']}\n")
                    f.write(f"Exception: {err['exception_type']}\n")
                    f.write(f"Message: {err['message']}\n")
                    f.write(f"Traceback:\n{err['traceback']}\n\n")

            print(f"\n  {len(all_errors)} error(s) occurred. Log saved to: {error_log_path}")
        else:
            print(f"\n No errors occurred during processing.")
