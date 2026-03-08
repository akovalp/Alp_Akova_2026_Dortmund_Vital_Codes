"""
Source Space PLI Connectivity Extraction - Four Regions

Computes Phase Lag Index (PLI) connectivity in source space using the
aparc (Desikan-Killiany) atlas for four brain regions: Frontal, Parietal,
Temporal, and Occipital.

Outputs: Within-region wPLI (4 regions) and between-region wPLI (6 pairs)
across 4 frequency bands, 2 conditions, and 2 acquisitions.
"""

import mne
from mne.datasets import fetch_fsaverage
from mne.minimum_norm import make_inverse_operator, apply_inverse_epochs
from mne_connectivity import spectral_connectivity_epochs
import numpy as np
import pandas as pd
import os
import warnings
import traceback
from datetime import datetime
from joblib import Parallel, delayed
import gc

# --- CONFIGURATION ---
# Process N participants at once, each with 4 conditions in parallel
# Total cores used = N_PARTICIPANTS_JOBS * 4 = 8 cores for 2 participants
N_PARTICIPANTS_JOBS = 10
MAIN_DIR = "/Users/alpmac/CodeWorks/Trento/PreProcessedData/Complete"
SESSION = "ses-1"
CONDITIONS = ["EyesClosed", "EyesOpen"]
ACQUISITIONS = ["pre", "post"]
BASE_OUTPUT_NAME = 'source_wpli_connectivity_four_regions_summary'
OUTPUT_DIR = "/Users/alpmac/CodeWorks/Trento/NewdatasetFırstTrıal/Connectivity Codes/ROI Connectivity Source/wPLI Connectivity"

FREQUENCY_BANDS = {
    'theta': (4, 8),
    'alpha': (8, 14),
    'beta': (14, 30),
    'broadband': (4, 30),
}

FRONTAL_LABEL_KEYWORDS = [
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
PARIETAL_LABEL_KEYWORDS = [
    'superiorparietal',
    'inferiorparietal',
    'supramarginal',
    'postcentral',
    'precuneus',
    'posteriorcingulate',
    'isthmuscingulate',
]
TEMPORAL_LABEL_KEYWORDS = [
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
OCCIPITAL_LABEL_KEYWORDS = [
    'lateraloccipital',
    'lingual',
    'cuneus',
    'pericalcarine',
]

# Define region names for iteration
REGION_NAMES = ['frontal', 'parietal', 'temporal', 'occipital']
REGION_KEYWORDS = {
    'frontal': FRONTAL_LABEL_KEYWORDS,
    'parietal': PARIETAL_LABEL_KEYWORDS,
    'temporal': TEMPORAL_LABEL_KEYWORDS,
    'occipital': OCCIPITAL_LABEL_KEYWORDS,
}

warnings.filterwarnings(
    'ignore', message='.*does not conform to MNE naming conventions.*')
warnings.filterwarnings("ignore", message="nperseg =")
warnings.filterwarnings("ignore", category=RuntimeWarning,
                        message=".*At least one epoch has multiple events.*")
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

    Returns
    -------
    fwd : Forward
        Forward solution
    noise_cov : Covariance
        Ad-hoc noise covariance
    labels : list of Label
        All atlas labels
    region_indices : dict
        Dictionary mapping region names to label indices
    src : SourceSpaces
        Source space
    """
    print(
        f"--- Setting up Global Resources using template: {os.path.basename(template_path)} ---")

    # Load template epochs
    epochs_template = mne.io.read_epochs_eeglab(template_path, verbose=False)
    montage = mne.channels.make_standard_montage('standard_1020')
    epochs_template.set_montage(montage, on_missing='ignore')

    # Get fsaverage paths
    fs_dir = fetch_fsaverage(verbose=False)
    subjects_dir = fs_dir.parent
    src_path = fs_dir / 'bem' / 'fsaverage-ico-4-src.fif'
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

    # Filter out labels with no vertices in source space ('unknown', 'corpuscallosum')
    # These labels cause errors in extract_label_time_course
    excluded_labels = ['unknown', 'corpuscallosum']
    labels = [
        label for label in all_labels
        if not any(excl in label.name.lower() for excl in excluded_labels)
    ]
    print(
        f"  Filtered out {len(all_labels) - len(labels)} labels with no vertices (unknown, corpuscallosum)")

    # Build indices for all four regions
    region_indices = {region: [] for region in REGION_NAMES}

    for i, label in enumerate(labels):
        name = label.name.lower()
        base_name = name.replace('-lh', '').replace('-rh', '')

        for region, keywords in REGION_KEYWORDS.items():
            if any(kw in base_name for kw in keywords):
                region_indices[region].append(i)
                break  # Each label belongs to only one region

    print(f"  Total valid labels: {len(labels)}")
    for region in REGION_NAMES:
        print(
            f"  {region.capitalize()} labels: {len(region_indices[region])} "
            f"(from {len(REGION_KEYWORDS[region])} regions × 2 hemispheres)")

    return fwd, noise_cov, labels, region_indices, src


def compute_source_pli(epochs, inverse_operator, labels, src, fmin, fmax):
    """
    Compute source-space PLI connectivity for a frequency band.

    Parameters
    ----------
    epochs : mne.Epochs
        Epoched EEG data
    inverse_operator : InverseOperator
        Pre-computed inverse operator
    labels : list of Label
        Atlas labels
    src : SourceSpaces
        Source space from inverse_operator['src']
    fmin, fmax : float
        Frequency band limits

    Returns
    -------
    pli_matrix : ndarray (n_labels, n_labels)
        PLI connectivity between all label pairs
    """
    snr = 1.0  # Lower SNR for single epochs
    lambda2 = 1.0 / snr**2

    # Apply inverse solution to get source time courses
    stcs = apply_inverse_epochs(
        epochs,
        inverse_operator,
        lambda2,
        method='eLORETA',
        pick_ori='normal',  # Important for phase estimation
        return_generator=True,
        verbose=False
    )

    # Extract label time courses (average within each ROI)
    label_ts = mne.extract_label_time_course(
        stcs,
        labels,
        src,
        mode='mean_flip',  # Reduces sign cancellation
        return_generator=True
    )

    # Compute wPLI connectivity (keep legacy *_pli naming for downstream compatibility)
    sfreq = epochs.info['sfreq']
    con = spectral_connectivity_epochs(
        label_ts,
        method=['wpli'],
        mode='multitaper',
        sfreq=sfreq,
        fmin=fmin,
        fmax=fmax,
        faverage=True,
        mt_adaptive=True,
        n_jobs=1,
        verbose=False
    )

    pli_matrix = con.get_data(output='dense')[:, :, 0]
    return pli_matrix


def get_off_diag_mean(matrix):
    """Calculate mean of off-diagonal elements (excluding self-connections)."""
    mask = ~np.eye(matrix.shape[0], dtype=bool)
    if np.sum(mask) == 0:
        return np.nan
    return np.mean(matrix[mask])


def get_region_pairs():
    """Generate all unique pairs of regions for between-region connectivity."""
    from itertools import combinations
    return list(combinations(REGION_NAMES, 2))


def process_single_condition(condition, set_file, participant_id, inverse_operator,
                             labels, src, region_indices, frequency_bands):
    """
    Process a single condition for one participant.

    Returns
    -------
    condition_results : dict
        PLI values for all bands (4 within-region + 6 between-region per band)
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
            print(
                f"[{participant_id}] SKIPPING {condition}: No epochs found", flush=True)
            return {}, None

        montage = mne.channels.make_standard_montage('standard_1020')
        epochs.set_montage(montage, on_missing='warn')
        epochs.set_eeg_reference('average', projection=True, verbose=False)
        epochs.apply_proj(verbose=False)

        condition_results = {}
        region_pairs = get_region_pairs()

        for band_name, (fmin, fmax) in frequency_bands.items():
            # Compute PLI matrix for all labels
            pli_matrix = compute_source_pli(
                epochs, inverse_operator, labels, src, fmin, fmax
            )

            # Compute within-region PLI for all 4 regions
            within_pli = {}
            for region in REGION_NAMES:
                idx = region_indices[region]
                if len(idx) > 1:
                    region_sub = pli_matrix[np.ix_(idx, idx)]
                    within_pli[region] = get_off_diag_mean(region_sub)
                else:
                    within_pli[region] = np.nan

                # Store result: {Condition}_{Timepoint}_{Freq}_{Region}_within_pli
                col_name = f"{condition}_{band_name}_{region}_within_pli"
                condition_results[col_name] = within_pli[region]

            # Compute between-region PLI for all 6 pairs
            between_pli = {}
            for region1, region2 in region_pairs:
                idx1 = region_indices[region1]
                idx2 = region_indices[region2]
                if len(idx1) > 0 and len(idx2) > 0:
                    cross_sub = pli_matrix[np.ix_(idx1, idx2)]
                    between_pli[(region1, region2)] = np.mean(cross_sub)
                else:
                    between_pli[(region1, region2)] = np.nan

                # Store result: {Condition}_{Timepoint}_{Freq}_{Region1}_{Region2}_between_pli
                col_name = f"{condition}_{band_name}_{region1}_{region2}_between_pli"
                condition_results[col_name] = between_pli[(region1, region2)]

            # Print summary
            within_str = ", ".join(
                [f"{r[:3].upper()}: {within_pli[r]:.4f}" for r in REGION_NAMES])
            print(
                f"[{participant_id}] {condition} | {band_name} -> Within: {within_str}", flush=True)

        return condition_results, None

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
        return {}, error_info

    finally:
        if epochs is not None:
            del epochs
        gc.collect()


def process_single_participant(participant_id, condition_files, global_fwd, global_noise_cov,
                               labels, region_indices, src, frequency_bands):
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
            epochs_template = mne.io.read_epochs_eeglab(
                first_file, verbose=False)
        montage = mne.channels.make_standard_montage('standard_1020')
        epochs_template.set_montage(montage, on_missing='warn')
        epochs_template.set_eeg_reference(
            'average', projection=True, verbose=False)
        epochs_template.apply_proj(verbose=False)

        inverse_operator = make_inverse_operator(
            epochs_template.info,
            global_fwd,
            global_noise_cov,
            loose=0.2,
            depth=0.8,
            verbose=False
        )
        print(
            f"[{participant_id}] Inverse operator computed. Processing conditions...", flush=True)

        del epochs_template
        gc.collect()

    except Exception as e:
        print(
            f"[{participant_id}] !!! CRITICAL: Failed to create inverse operator: {e}", flush=True)
        error_info = {
            'participant_id': participant_id,
            'condition': 'INVERSE_OPERATOR_CREATION',
            'file': first_file,
            'exception_type': type(e).__name__,
            'message': str(e),
            'traceback': traceback.format_exc()
        }
        return {'Participant': participant_id}, [error_info]

    # Process all conditions in parallel
    condition_results = Parallel(n_jobs=4, verbose=0)(
        delayed(process_single_condition)(
            condition, set_file, participant_id, inverse_operator,
            labels, src, region_indices, frequency_bands
        )
        for condition, set_file in condition_files.items()
    )

    del inverse_operator
    gc.collect()

    # Aggregate results
    row_data = {'Participant': participant_id}
    errors = []

    for result_dict, error in condition_results:
        row_data.update(result_dict)
        if error is not None:
            errors.append(error)

    print(f"[{participant_id}] Finished.", flush=True)
    return row_data, errors


def get_expected_columns(frequency_bands, conditions, acquisitions):
    """Generate expected column names for validation."""
    from itertools import combinations
    region_pairs = list(combinations(REGION_NAMES, 2))

    expected_cols = []
    for band in frequency_bands.keys():
        for condition in conditions:
            for acquisition in acquisitions:
                # Naming: {Condition}_{Timepoint}_{Freq}_{Region}_within_pli
                base = f"{condition}_{acquisition}_{band}"

                # Within-region columns (4 regions)
                for region in REGION_NAMES:
                    expected_cols.append(f"{base}_{region}_within_pli")

                # Between-region columns (6 pairs)
                for region1, region2 in region_pairs:
                    expected_cols.append(
                        f"{base}_{region1}_{region2}_between_pli")

    return expected_cols


def load_existing_progress(output_path, expected_columns):
    """Load existing CSV for resume capability."""
    if output_path is None or not os.path.exists(output_path):
        print(f"No existing progress file found. Starting fresh.")
        return pd.DataFrame(), set()

    try:
        existing_df = pd.read_csv(output_path, dtype={'Participant': str})
        print(
            f"Loaded existing progress with {len(existing_df)} participants.")

        completed_participants = set()
        for _, row in existing_df.iterrows():
            participant_id = str(row['Participant']).zfill(3)
            is_complete = all(
                col in existing_df.columns and pd.notna(row.get(col))
                for col in expected_columns
            )
            if is_complete:
                completed_participants.add(participant_id)

        print(
            f"Found {len(completed_participants)} completed participants to skip.")
        return existing_df, completed_participants

    except Exception as e:
        print(f"Error loading existing progress: {e}. Starting fresh.")
        return pd.DataFrame(), set()


def save_progress(output_path, all_row_data):
    """Save progress to CSV."""
    if not all_row_data:
        return

    results_df = pd.DataFrame(all_row_data)

    # Format PLI values
    formatted_df = results_df.copy()
    for col in formatted_df.columns:
        if col != 'Participant':
            formatted_df[col] = formatted_df[col].apply(
                lambda x: f"{x:.6f}" if pd.notnull(x) else "")

    formatted_df.to_csv(output_path, index=False)
    print(
        f"  -> Progress saved to {output_path} ({len(all_row_data)} participants)")


def get_new_output_path(base_name, output_dir, extension=".csv"):
    """Generate timestamped output filename in the output directory."""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    return os.path.join(output_dir, f"{base_name}_{timestamp}{extension}")


def find_latest_summary_file(base_name, directory="."):
    """
    Find the latest summary file matching the pattern {base_name}_*.csv.

    Looks for files with timestamps in the format YYYYMMDD_HHMMSS and returns
    the most recent one based on the timestamp in the filename.

    Parameters
    ----------
    base_name : str
        Base name prefix to search for (e.g., 'source_pli_connectivity_summary')
    directory : str
        Directory to search in (default: current directory)

    Returns
    -------
    str or None
        Path to the latest summary file, or None if no matching files found
    """
    import glob
    import re

    # Pattern: base_name_YYYYMMDD_HHMMSS.csv
    pattern = os.path.join(directory, f"{base_name}_*.csv")
    matching_files = glob.glob(pattern)

    if not matching_files:
        return None

    # Extract timestamp from filename and sort
    timestamp_pattern = re.compile(
        rf"{re.escape(base_name)}_(\d{{8}}_\d{{6}})\.csv$")

    files_with_timestamps = []
    for filepath in matching_files:
        filename = os.path.basename(filepath)
        match = timestamp_pattern.match(filename)
        if match:
            timestamp_str = match.group(1)
            files_with_timestamps.append((filepath, timestamp_str))

    if not files_with_timestamps:
        return None

    # Sort by timestamp (string comparison works for YYYYMMDD_HHMMSS format)
    files_with_timestamps.sort(key=lambda x: x[1], reverse=True)

    latest_file = files_with_timestamps[0][0]
    return latest_file


if __name__ == "__main__":
    # 1. Setup paths
    NEW_OUTPUT = get_new_output_path(BASE_OUTPUT_NAME, OUTPUT_DIR)
    print(f"Output will be saved to: {NEW_OUTPUT}")

    # 2. Find the latest existing summary file to check for completed participants
    latest_summary = find_latest_summary_file(BASE_OUTPUT_NAME, OUTPUT_DIR)
    if latest_summary:
        print(f"Found existing summary file: {latest_summary}")
    else:
        print("No existing summary file found. Starting fresh.")

    # 3. Generate expected columns
    expected_columns = get_expected_columns(
        FREQUENCY_BANDS, CONDITIONS, ACQUISITIONS)
    existing_df, completed_participants = load_existing_progress(
        latest_summary, expected_columns) if latest_summary else (pd.DataFrame(), set())

    all_row_data = existing_df.to_dict(
        'records') if not existing_df.empty else []

    # 3. Scan for participants
    print(f"\nScanning {MAIN_DIR} for participants...")
    existing_folders = [f for f in os.listdir(
        MAIN_DIR) if f.startswith('sub-')]
    participant_ids = sorted([f.split('-')[1] for f in existing_folders])

    # 4. Validate files and filter completed participants
    valid_participants = {}
    for participant_id in participant_ids:
        if participant_id in completed_participants:
            print(f"[{participant_id}] Already complete, skipping...")
            continue

        condition_files = {}
        for condition in CONDITIONS:
            for acquisition in ACQUISITIONS:
                fpath = get_participants_condition(
                    MAIN_DIR, participant_id, SESSION, condition, acquisition)
                if fpath:
                    condition_files[f"{condition}_{acquisition}"] = fpath

        if condition_files:
            valid_participants[participant_id] = condition_files

    print(f"Found {len(valid_participants)} participants to process.")

    if not valid_participants:
        print("\n✓ All participants already processed. Nothing to do.")
    else:
        # 5. Setup global resources (forward solution, labels)
        first_sub = list(valid_participants.keys())[0]
        first_file = list(valid_participants[first_sub].values())[0]

        fwd, noise_cov, labels, region_indices, src = setup_global_resources(
            first_file)
        print("\nGlobal resources ready. Starting processing...\n")

        # 6. Process participants in parallel (N_PARTICIPANTS_JOBS at a time)
        all_errors = []
        participants_list = list(valid_participants.items())
        total_participants = len(participants_list)

        # Process in batches of N_PARTICIPANTS_JOBS
        for batch_start in range(0, total_participants, N_PARTICIPANTS_JOBS):
            batch_end = min(batch_start + N_PARTICIPANTS_JOBS,
                            total_participants)
            batch = participants_list[batch_start:batch_end]

            print(f"\n{'='*60}")
            print(
                f"Processing batch: participants {batch_start+1}-{batch_end} of {total_participants}")
            print(f"Participants in batch: {[p[0] for p in batch]}")
            print(f"{'='*60}")

            try:
                # Process N participants in parallel
                batch_results = Parallel(n_jobs=len(batch), verbose=0)(
                    delayed(process_single_participant)(
                        participant_id, files, fwd, noise_cov,
                        labels, region_indices, src, FREQUENCY_BANDS
                    )
                    for participant_id, files in batch
                )

                # Collect results from batch
                for row_data, errors in batch_results:
                    all_row_data.append(row_data)
                    if errors:
                        all_errors.extend(errors)

                # Save after each batch
                save_progress(NEW_OUTPUT, all_row_data)

            except Exception as e:
                print(f"Batch error: {e}")
                traceback.print_exc()
                save_progress(NEW_OUTPUT, all_row_data)
                continue

        print(f"\n{'='*60}")
        print(f"Processing Complete. Results saved to {NEW_OUTPUT}")
        print(f"Total participants: {len(all_row_data)}")
        print(f"{'='*60}")

        # 7. Save error log if needed
        if all_errors:
            error_log_path = f"source_pli_errors_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            with open(error_log_path, 'w') as f:
                f.write(f"# Source PLI Connectivity Error Log\n")
                f.write(
                    f"# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"# Total Errors: {len(all_errors)}\n\n")

                for i, err in enumerate(all_errors, 1):
                    f.write(f"ERROR {i}/{len(all_errors)}\n{'─'*50}\n")
                    f.write(f"Participant: {err['participant_id']}\n")
                    f.write(f"Condition: {err['condition']}\n")
                    f.write(f"File: {err['file']}\n")
                    f.write(f"Exception: {err['exception_type']}\n")
                    f.write(f"Message: {err['message']}\n")
                    f.write(f"Traceback:\n{err['traceback']}\n\n")

            print(
                f"\n {len(all_errors)} error(s) occurred. Log saved to: {error_log_path}")
        else:
            print(f"\n✓ No errors occurred during processing.")
