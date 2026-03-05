"""
Sensor Space wPLI Connectivity Extraction - Four Regions

Refactor of the original sensor-space connectivity script:
- Uses four fixed sensor regions (frontal/parietal/temporal/occipital)
- Computes wPLI (not PLI)
- Exports source-compatible column naming while keeping *_pli suffix for legacy use
- Computes both within-region and between-region connectivity
"""

import os
import gc
import re
import warnings
import traceback
from datetime import datetime
from itertools import combinations
from pathlib import Path
from joblib import Parallel, delayed

import mne
import numpy as np
import pandas as pd


# --- CONFIGURATION ---
MAIN_DIR = "/Users/alpmac/CodeWorks/Trento/PreProcessedData/Complete"
SESSION = "ses-1"
CONDITIONS = ["EyesClosed", "EyesOpen"]
ACQUISITIONS = ["pre", "post"]

FREQUENCY_BANDS = {
    "theta": (4, 8),
    "alpha": (8, 14),
    "beta": (14, 30),
    "broadband": (4, 30),
}

REGION_NAMES = ["frontal", "parietal", "temporal", "occipital"]
SENSOR_REGION_CHANNELS = {
    "frontal": [
        "Fp1", "Fp2", "AF7", "AF3", "AF4", "AF8",
        "F7", "F5", "F3", "F1", "Fz", "F2", "F4", "F6", "F8",
        "FC5", "FC3", "FC1", "FC2", "FC4", "FC6",
    ],
    "temporal": [
        "FT9", "FT7", "T7", "TP7", "TP9",
        "FT8", "T8", "TP8", "TP10", "FT10",
    ],
    "parietal": [
        "C5", "C3", "C1", "Cz", "C2", "C4", "C6",
        "CP5", "CP3", "CP1", "CPz", "CP2", "CP4", "CP6",
        "P7", "P5", "P3", "P1", "Pz", "P2", "P4", "P6", "P8",
    ],
    "occipital": [
        "PO9", "PO7", "PO3", "POz", "PO4", "PO8", "PO10",
        "O1", "Oz", "O2",
    ],
}

EXPECTED_64_CHANNELS = {
    "AF3", "AF4", "AF7", "AF8",
    "C1", "C2", "C3", "C4", "C5", "C6",
    "CP1", "CP2", "CP3", "CP4", "CP5", "CP6", "CPz", "Cz",
    "F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8",
    "FC1", "FC2", "FC3", "FC4", "FC5", "FC6",
    "FT10", "FT7", "FT8", "FT9",
    "Fp1", "Fp2", "Fz",
    "O1", "O2", "Oz",
    "P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8",
    "PO10", "PO3", "PO4", "PO7", "PO8", "PO9", "POz",
    "Pz",
    "T7", "T8",
    "TP10", "TP7", "TP8", "TP9",
}

# Keep current output location policy from the previous sensor script.
OUTPUT_DIR = "/Users/alpmac/CodeWorks/Trento/Dortmund_Vital_Alp_Akova_Clean/Extraction Based on Connectivity Electrode"
BASE_OUTPUT_NAME = "sensor_wpli_connectivity_four_regions_summary"
COMPAT_ALIAS_NAME = "pli_connectivity_summary.csv"

# Process N participants in each batch.
N_PARTICIPANTS_JOBS = 10


warnings.filterwarnings(
    "ignore", message=".*does not conform to MNE naming conventions.*")
warnings.filterwarnings("ignore", message="nperseg =")
warnings.filterwarnings(
    "ignore",
    category=RuntimeWarning,
    message=".*At least one epoch has multiple events.*",
)
warnings.filterwarnings(
    "ignore",
    category=RuntimeWarning,
    message=".*There were no Annotations stored in.*",
)
pd.options.display.float_format = "{:.7e}".format


def require_mne_connectivity():
    """Fail fast with actionable guidance when dependency is missing."""
    try:
        from mne_connectivity import spectral_connectivity_epochs  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: mne-connectivity.\n"
            "Install it before running this script, e.g.:\n"
            "  pip install mne-connectivity\n"
            "or use your project environment manager."
        ) from exc


def validate_region_channel_map(region_map, expected_channels):
    """Validate fixed 64-channel map and one-to-one region assignment."""
    all_channels = []
    for region in REGION_NAMES:
        if region not in region_map:
            raise ValueError(f"Missing region in channel map: {region}")
        all_channels.extend(region_map[region])

    duplicate_channels = sorted(
        {ch for ch in all_channels if all_channels.count(ch) > 1})
    if duplicate_channels:
        raise ValueError(
            "Channel map has duplicate assignments: " +
            ", ".join(duplicate_channels)
        )

    if len(all_channels) != 64:
        raise ValueError(
            f"Channel map must contain 64 channel assignments, found {len(all_channels)}")

    mapped_set = set(all_channels)
    missing_from_map = sorted(expected_channels - mapped_set)
    extra_in_map = sorted(mapped_set - expected_channels)
    if missing_from_map or extra_in_map:
        raise ValueError(
            "Channel map mismatch with expected 64-channel set. "
            f"Missing: {missing_from_map}; Extra: {extra_in_map}"
        )


def canonical_participant_id(participant_id):
    """Normalize IDs to sub-### format."""
    pid = str(participant_id).strip()
    if pid.startswith("sub-"):
        return pid
    if pid.isdigit():
        return f"sub-{pid.zfill(3)}"
    return f"sub-{pid}"


def scan_for_participants(main_dir):
    main_path = Path(main_dir)
    if not main_path.exists():
        raise FileNotFoundError(f"Main directory not found: {main_dir}")

    return sorted(
        entry.name for entry in main_path.iterdir()
        if entry.is_dir() and entry.name.startswith("sub-")
    )


def get_participant_condition_path(main_dir, participant_id, session, condition, acquisition):
    participant_id = canonical_participant_id(participant_id)
    filename = f"{participant_id}_{session}_task-{condition}_acq-{acquisition}_eeg.set"
    set_path = os.path.join(main_dir, participant_id, session, "eeg", filename)
    if not os.path.exists(set_path):
        return None
    return set_path


def get_region_pairs():
    return list(combinations(REGION_NAMES, 2))


def get_off_diag_mean(matrix):
    if matrix.shape[0] <= 1:
        return np.nan
    mask = ~np.eye(matrix.shape[0], dtype=bool)
    if not np.any(mask):
        return np.nan
    return float(np.mean(matrix[mask]))


def get_region_channel_indices(ch_names, region_map):
    """Return indices and missing channels by region for the current file."""
    ch_to_idx = {ch: idx for idx, ch in enumerate(ch_names)}
    region_indices = {}
    region_missing = {}
    for region in REGION_NAMES:
        mapped = region_map[region]
        present = [ch_to_idx[ch] for ch in mapped if ch in ch_to_idx]
        missing = [ch for ch in mapped if ch not in ch_to_idx]
        region_indices[region] = present
        region_missing[region] = missing
    return region_indices, region_missing


def compute_sensor_wpli_matrix(epochs, fmin, fmax):
    """Compute dense sensor-space wPLI matrix for one band."""
    from mne_connectivity import spectral_connectivity_epochs

    con = spectral_connectivity_epochs(
        epochs,
        method=["wpli"],
        mode="multitaper",
        sfreq=epochs.info["sfreq"],
        fmin=fmin,
        fmax=fmax,
        faverage=True,
        mt_adaptive=True,
        n_jobs=1,
        verbose=False,
    )
    return con.get_data(output="dense")[:, :, 0]


def process_single_condition(condition, set_file, participant_id, frequency_bands, region_map):
    """
    Process one condition for one participant.

    Returns
    -------
    condition_results : dict
        Connectivity values for all bands/regions in source-compatible naming.
    error_info : dict or None
        Error details if processing failed.
    """
    epochs = None
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                category=RuntimeWarning,
                message=".*At least one epoch has multiple events.*",
            )
            epochs = mne.io.read_epochs_eeglab(set_file, verbose=False)

        if len(epochs) < 1:
            print(
                f"[{participant_id}] SKIPPING {condition}: No epochs found", flush=True)
            return {}, None

        # Use EEG channels only. Keep loaded referencing unchanged by design.
        epochs = epochs.copy().pick("eeg")
        ch_names = list(epochs.info["ch_names"])
        region_indices, region_missing = get_region_channel_indices(
            ch_names, region_map)

        for region in REGION_NAMES:
            if region_missing[region]:
                print(
                    f"[{participant_id}] {condition} | WARN {region} missing channels: {region_missing[region]}",
                    flush=True,
                )

        condition_results = {}
        region_pairs = get_region_pairs()

        for band_name, (fmin, fmax) in frequency_bands.items():
            wpli_matrix = compute_sensor_wpli_matrix(epochs, fmin, fmax)

            within_vals = {}
            for region in REGION_NAMES:
                idx = region_indices[region]
                if len(idx) > 1:
                    region_sub = wpli_matrix[np.ix_(idx, idx)]
                    within_vals[region] = get_off_diag_mean(region_sub)
                else:
                    within_vals[region] = np.nan

                col_name = f"{condition}_{band_name}_{region}_within_pli"
                condition_results[col_name] = within_vals[region]

            for region1, region2 in region_pairs:
                idx1 = region_indices[region1]
                idx2 = region_indices[region2]
                if len(idx1) > 0 and len(idx2) > 0:
                    cross_sub = wpli_matrix[np.ix_(idx1, idx2)]
                    between_val = float(np.mean(cross_sub))
                else:
                    between_val = np.nan

                col_name = f"{condition}_{band_name}_{region1}_{region2}_between_pli"
                condition_results[col_name] = between_val

            within_str = ", ".join(
                [f"{region[:3].upper()}: {within_vals[region]:.4f}" for region in REGION_NAMES]
            )
            print(
                f"[{participant_id}] {condition} | {band_name} -> Within: {within_str}",
                flush=True,
            )

        return condition_results, None

    except Exception as exc:
        error_info = {
            "participant_id": participant_id,
            "condition": condition,
            "file": set_file,
            "exception_type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
        print(f"[{participant_id}] !!! ERROR in {condition}: {exc}", flush=True)
        return {}, error_info

    finally:
        if epochs is not None:
            del epochs
        gc.collect()


def process_single_participant(participant_id, condition_files, frequency_bands, region_map):
    """Process all conditions for one participant."""
    row_data = {"Participant": participant_id}
    errors = []

    for condition, set_file in condition_files.items():
        condition_results, error = process_single_condition(
            condition, set_file, participant_id, frequency_bands, region_map
        )
        row_data.update(condition_results)
        if error is not None:
            errors.append(error)

    print(f"[{participant_id}] Finished.", flush=True)
    return row_data, errors


def get_expected_columns(frequency_bands, conditions, acquisitions):
    region_pairs = get_region_pairs()
    expected_cols = []
    for band in frequency_bands.keys():
        for condition in conditions:
            for acquisition in acquisitions:
                base = f"{condition}_{acquisition}_{band}"
                for region in REGION_NAMES:
                    expected_cols.append(f"{base}_{region}_within_pli")
                for region1, region2 in region_pairs:
                    expected_cols.append(
                        f"{base}_{region1}_{region2}_between_pli")
    return expected_cols


def load_existing_progress(output_path, expected_columns):
    if output_path is None or not os.path.exists(output_path):
        print("No existing progress file found. Starting fresh.")
        return pd.DataFrame(), set()

    try:
        existing_df = pd.read_csv(output_path, dtype={"Participant": str})
        print(
            f"Loaded existing progress with {len(existing_df)} participants.")

        completed_participants = set()
        for _, row in existing_df.iterrows():
            participant_id = canonical_participant_id(row["Participant"])
            is_complete = all(
                col in existing_df.columns and pd.notna(row.get(col))
                for col in expected_columns
            )
            if is_complete:
                completed_participants.add(participant_id)

        print(
            f"Found {len(completed_participants)} completed participants to skip.")
        return existing_df, completed_participants
    except Exception as exc:
        print(f"Error loading existing progress: {exc}. Starting fresh.")
        return pd.DataFrame(), set()


def save_progress(output_path, compat_alias_path, all_row_data):
    if not all_row_data:
        return

    results_df = pd.DataFrame(all_row_data)
    formatted_df = results_df.copy()
    for col in formatted_df.columns:
        if col != "Participant":
            formatted_df[col] = formatted_df[col].apply(
                lambda x: f"{x:.6f}" if pd.notnull(x) else ""
            )

    formatted_df.to_csv(output_path, index=False)
    formatted_df.to_csv(compat_alias_path, index=False)
    print(
        f"  -> Progress saved to {output_path} ({len(all_row_data)} participants)")


def get_new_output_path(base_name, output_dir, extension=".csv"):
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(output_dir, f"{base_name}_{timestamp}{extension}")


def find_latest_summary_file(base_name, directory="."):
    pattern = os.path.join(directory, f"{base_name}_*.csv")
    matching_files = [
        fp for fp in Path(directory).glob(f"{base_name}_*.csv")
        if fp.is_file()
    ]
    if not matching_files:
        return None

    ts_pattern = re.compile(rf"{re.escape(base_name)}_(\d{{8}}_\d{{6}})\.csv$")
    files_with_ts = []
    for fp in matching_files:
        match = ts_pattern.match(fp.name)
        if match:
            files_with_ts.append((str(fp), match.group(1)))

    if not files_with_ts:
        return None

    files_with_ts.sort(key=lambda x: x[1], reverse=True)
    return files_with_ts[0][0]


if __name__ == "__main__":
    validate_region_channel_map(SENSOR_REGION_CHANNELS, EXPECTED_64_CHANNELS)
    require_mne_connectivity()

    new_output = get_new_output_path(BASE_OUTPUT_NAME, OUTPUT_DIR)
    latest_alias = os.path.join(OUTPUT_DIR, f"{BASE_OUTPUT_NAME}.csv")
    compat_alias = os.path.join(OUTPUT_DIR, COMPAT_ALIAS_NAME)
    print(f"Output will be saved to: {new_output}")

    latest_summary = find_latest_summary_file(BASE_OUTPUT_NAME, OUTPUT_DIR)
    if latest_summary:
        print(f"Found existing summary file: {latest_summary}")
    else:
        print("No existing summary file found. Starting fresh.")

    expected_columns = get_expected_columns(
        FREQUENCY_BANDS, CONDITIONS, ACQUISITIONS)
    existing_df, completed_participants = load_existing_progress(
        latest_summary, expected_columns
    ) if latest_summary else (pd.DataFrame(), set())

    all_row_data = existing_df.to_dict(
        "records") if not existing_df.empty else []

    print(f"\nScanning {MAIN_DIR} for participants...")
    participant_ids = scan_for_participants(MAIN_DIR)

    valid_participants = {}
    for participant_id in participant_ids:
        participant_id = canonical_participant_id(participant_id)
        if participant_id in completed_participants:
            print(f"[{participant_id}] Already complete, skipping...")
            continue

        condition_files = {}
        for condition in CONDITIONS:
            for acquisition in ACQUISITIONS:
                fpath = get_participant_condition_path(
                    MAIN_DIR, participant_id, SESSION, condition, acquisition
                )
                if fpath:
                    condition_files[f"{condition}_{acquisition}"] = fpath

        if condition_files:
            valid_participants[participant_id] = condition_files

    print(f"Found {len(valid_participants)} participants to process.")
    if not valid_participants:
        print("\nAll participants already processed. Nothing to do.")
    else:
        all_errors = []
        participants_list = list(valid_participants.items())
        total_participants = len(participants_list)

        for batch_start in range(0, total_participants, N_PARTICIPANTS_JOBS):
            batch_end = min(batch_start + N_PARTICIPANTS_JOBS,
                            total_participants)
            batch = participants_list[batch_start:batch_end]
            print(f"\n{'=' * 60}")
            print(
                f"Processing batch: participants {batch_start + 1}-{batch_end} of {total_participants}"
            )
            print(f"Participants in batch: {[pid for pid, _ in batch]}")
            print(f"{'=' * 60}")

            try:
                batch_results = Parallel(n_jobs=len(batch), verbose=0)(
                    delayed(process_single_participant)(
                        participant_id,
                        condition_files,
                        FREQUENCY_BANDS,
                        SENSOR_REGION_CHANNELS,
                    )
                    for participant_id, condition_files in batch
                )
                for row_data, errors in batch_results:
                    all_row_data.append(row_data)
                    if errors:
                        all_errors.extend(errors)
            except Exception as exc:
                print(f"Batch-level error: {exc}")
                traceback.print_exc()

            # Save progress after each batch.
            save_progress(new_output, compat_alias, all_row_data)
            # Keep non-timestamped latest alias in sync.
            if os.path.exists(new_output):
                pd.read_csv(new_output).to_csv(latest_alias, index=False)

        print(f"\n{'=' * 60}")
        print(f"Processing Complete. Results saved to {new_output}")
        print(f"Total participants: {len(all_row_data)}")
        print(f"{'=' * 60}")

        if all_errors:
            error_log_path = os.path.join(
                OUTPUT_DIR,
                f"sensor_wpli_errors_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            )
            with open(error_log_path, "w", encoding="utf-8") as handle:
                handle.write("# Sensor wPLI Connectivity Error Log\n")
                handle.write(
                    f"# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                handle.write(f"# Total Errors: {len(all_errors)}\n\n")
                for i, err in enumerate(all_errors, start=1):
                    handle.write(f"ERROR {i}/{len(all_errors)}\n{'-' * 50}\n")
                    handle.write(f"Participant: {err['participant_id']}\n")
                    handle.write(f"Condition: {err['condition']}\n")
                    handle.write(f"File: {err['file']}\n")
                    handle.write(f"Exception: {err['exception_type']}\n")
                    handle.write(f"Message: {err['message']}\n")
                    handle.write(f"Traceback:\n{err['traceback']}\n\n")
            print(
                f"\n{len(all_errors)} error(s) occurred. Log saved to: {error_log_path}")
        else:
            print("\nNo errors occurred during processing.")
