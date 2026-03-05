"""
ROI-Based Sensor-Space Power Extraction (Four Regions, Summary Only)

This script extracts sensor-space spectral features from epoched EEG (.set) files
using fixed channel groups for four regions:
frontal, parietal, temporal, occipital.

Features per condition:
- Band power (theta/alpha/beta) per region
- Spectral flatness (Wiener entropy) per band and region
- Peak alpha frequency (PAF) per region

Output:
- Summary CSV only (one row per participant), with source-compatible column names.
"""

import argparse
from datetime import datetime
from pathlib import Path
import os
import warnings

import mne
import numpy as np
import pandas as pd
from scipy.stats import gmean


# =============================================================================
# CONFIGURATION
# =============================================================================

MAIN_DIR = "/Users/alpmac/CodeWorks/Trento/PreProcessedData/Complete"
SESSION = "ses-1"
CONDITIONS = ["EyesClosed", "EyesOpen"]
ACQUISITIONS = ["pre", "post"]

FREQUENCY_BANDS = {
    "theta": (4, 8),
    "alpha": (8, 14),
    "beta": (14, 30),
}

PSD_BANDWIDTH = 1
BASE_OUTPUT_NAME_SUMMARY = "roi_sensor_power_summary_four_regions"

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

# Suppress noisy warnings to keep logs usable in large runs.
warnings.filterwarnings(
    "ignore", message=".*does not conform to MNE naming conventions.*"
)
warnings.filterwarnings("ignore", message="nperseg =")
warnings.filterwarnings(
    "ignore",
    category=RuntimeWarning,
    message=".*At least one epoch has multiple events.*",
)
pd.options.display.float_format = "{:.7e}".format


def _all_condition_labels():
    return [f"{task}_{acq}" for task in CONDITIONS for acq in ACQUISITIONS]


def validate_region_channel_map(region_map, expected_channels):
    """Assert 64-channel fixed map integrity and one-to-one assignment."""
    all_channels = []
    for region in ("frontal", "parietal", "temporal", "occipital"):
        if region not in region_map:
            raise ValueError(f"Missing region in channel map: {region}")
        all_channels.extend(region_map[region])

    duplicate_channels = sorted({ch for ch in all_channels if all_channels.count(ch) > 1})
    if duplicate_channels:
        raise ValueError(
            "Channel map has duplicate assignments: "
            + ", ".join(duplicate_channels)
        )

    mapped_set = set(all_channels)
    if len(all_channels) != 64:
        raise ValueError(f"Channel map must contain 64 assignments, found {len(all_channels)}")

    missing_from_map = sorted(expected_channels - mapped_set)
    unexpected_in_map = sorted(mapped_set - expected_channels)
    if missing_from_map or unexpected_in_map:
        raise ValueError(
            "Channel map does not match expected 64-channel set. "
            f"Missing: {missing_from_map}; Unexpected: {unexpected_in_map}"
        )


def parse_participant_filter(participants_arg):
    """Parse comma-separated participant IDs with/without 'sub-' prefix."""
    if not participants_arg:
        return None

    tokens = [token.strip() for token in participants_arg.split(",") if token.strip()]
    if not tokens:
        return None

    filters = set()
    for token in tokens:
        if token.startswith("sub-"):
            filters.add(token)
            filters.add(token.replace("sub-", "", 1))
        else:
            filters.add(token)
            filters.add(f"sub-{token}")
    return filters


def participant_in_filter(participant_id, participant_filter):
    if participant_filter is None:
        return True
    plain_id = participant_id.replace("sub-", "", 1)
    return participant_id in participant_filter or plain_id in participant_filter


def parse_condition_labels(condition_labels_arg):
    """Parse comma-separated condition labels; default is all task-acq labels."""
    if not condition_labels_arg:
        return _all_condition_labels()

    requested = [
        token.strip() for token in condition_labels_arg.split(",") if token.strip()
    ]
    valid = set(_all_condition_labels())
    invalid = sorted(set(requested) - valid)
    if invalid:
        raise ValueError(
            "Invalid condition labels: "
            + ", ".join(invalid)
            + f". Valid labels: {sorted(valid)}"
        )
    return requested


def normalize_participant_id(participant_id):
    if participant_id.startswith("sub-"):
        return participant_id
    return f"sub-{participant_id}"


def get_set_file_path(main_dir, participant_id, session, task, acq):
    participant_id = normalize_participant_id(participant_id)
    filename = f"{participant_id}_{session}_task-{task}_acq-{acq}_eeg.set"
    set_path = Path(main_dir) / participant_id / session / "eeg" / filename
    if not set_path.exists():
        return None
    return str(set_path)


def scan_for_participants(main_dir):
    main_path = Path(main_dir)
    if not main_path.exists():
        raise FileNotFoundError(f"Main directory not found: {main_dir}")

    participant_folders = [
        entry.name for entry in main_path.iterdir()
        if entry.is_dir() and entry.name.startswith("sub-")
    ]
    return sorted(participant_folders)


def load_epochs(set_path):
    epochs = mne.io.read_epochs_eeglab(set_path, verbose=False)
    return epochs


def compute_psd_multitaper(epochs, bandwidth=PSD_BANDWIDTH):
    return epochs.compute_psd(method="multitaper", bandwidth=bandwidth, verbose=False)


def _get_region_channel_indices(ch_names, region_map):
    index_by_channel = {ch: idx for idx, ch in enumerate(ch_names)}
    region_indices = {}
    region_missing = {}
    for region, mapped_channels in region_map.items():
        present = [ch for ch in mapped_channels if ch in index_by_channel]
        missing = [ch for ch in mapped_channels if ch not in index_by_channel]
        region_indices[region] = [index_by_channel[ch] for ch in present]
        region_missing[region] = missing
    return region_indices, region_missing


def extract_region_band_power(psd_data, freqs, ch_indices, freq_band):
    if not ch_indices:
        return np.nan
    low_freq, high_freq = freq_band
    idx_band = np.logical_and(freqs >= low_freq, freqs <= high_freq)
    if not np.any(idx_band):
        return np.nan
    band_data = psd_data[:, ch_indices, :][:, :, idx_band]
    return float(np.mean(band_data))


def extract_region_spectral_flatness(psd_data, freqs, ch_indices, freq_band):
    if not ch_indices:
        return np.nan
    low_freq, high_freq = freq_band
    idx_band = np.logical_and(freqs >= low_freq, freqs <= high_freq)
    if not np.any(idx_band):
        return np.nan

    band_data = psd_data[:, ch_indices, :][:, :, idx_band]
    geometric_mean = gmean(band_data, axis=2)
    arithmetic_mean = np.mean(band_data, axis=2)
    with np.errstate(divide="ignore", invalid="ignore"):
        flatness_epochs = geometric_mean / arithmetic_mean
    channel_flatness = np.nanmean(flatness_epochs, axis=0)
    return float(np.nanmean(channel_flatness))


def extract_region_peak_alpha_frequency(psd_data, freqs, ch_indices, alpha_band=(8, 14)):
    if not ch_indices:
        return np.nan

    mean_psd = np.mean(psd_data[:, ch_indices, :], axis=(0, 1))
    alpha_mask = (freqs >= alpha_band[0]) & (freqs <= alpha_band[1])
    if not np.any(alpha_mask):
        return np.nan

    alpha_psd = mean_psd[alpha_mask]
    alpha_freqs = freqs[alpha_mask]
    if alpha_psd.size == 0 or np.all(np.isnan(alpha_psd)):
        return np.nan

    peak_idx = int(np.nanargmax(alpha_psd))
    return float(alpha_freqs[peak_idx])


def compute_condition_region_features(psd, region_map, frequency_bands):
    psd_data = psd.get_data()
    freqs = psd.freqs
    ch_names = psd.ch_names
    regions = list(region_map.keys())

    region_indices, region_missing = _get_region_channel_indices(ch_names, region_map)

    results = {
        "power": {band: {region: np.nan for region in regions} for band in frequency_bands},
        "flatness": {band: {region: np.nan for region in regions} for band in frequency_bands},
        "paf": {region: np.nan for region in regions},
    }

    alpha_band = frequency_bands["alpha"]

    for region in regions:
        present_indices = region_indices[region]
        if region_missing[region]:
            print(
                f"      [WARN] {region}: missing channels -> {region_missing[region]}"
            )
        if not present_indices:
            print(f"      [WARN] {region}: no mapped channels available in this file.")
            continue

        for band, freq_range in frequency_bands.items():
            power_value = extract_region_band_power(
                psd_data, freqs, present_indices, freq_range
            )
            flatness_value = extract_region_spectral_flatness(
                psd_data, freqs, present_indices, freq_range
            )
            results["power"][band][region] = power_value
            results["flatness"][band][region] = flatness_value

        paf_value = extract_region_peak_alpha_frequency(
            psd_data, freqs, present_indices, alpha_band
        )
        results["paf"][region] = paf_value

    return results


def get_expected_columns(condition_labels, frequency_bands, regions):
    expected_cols = []
    for condition_label in condition_labels:
        for band in frequency_bands.keys():
            for region in regions:
                expected_cols.append(f"{condition_label}_{band}_{region}_power")
                expected_cols.append(f"{condition_label}_{band}_{region}_flatness")
        for region in regions:
            expected_cols.append(f"{condition_label}_{region}_paf")
    return expected_cols


def condition_label_to_task_acq(condition_label):
    parts = condition_label.split("_")
    if len(parts) != 2:
        raise ValueError(f"Invalid condition label: {condition_label}")
    return parts[0], parts[1]


def set_condition_nan_values(row_data, condition_label, frequency_bands, regions):
    for band in frequency_bands.keys():
        for region in regions:
            row_data[f"{condition_label}_{band}_{region}_power"] = np.nan
            row_data[f"{condition_label}_{band}_{region}_flatness"] = np.nan
    for region in regions:
        row_data[f"{condition_label}_{region}_paf"] = np.nan


def apply_condition_results_to_row(
    row_data, condition_label, features, frequency_bands, regions
):
    for band in frequency_bands.keys():
        for region in regions:
            row_data[f"{condition_label}_{band}_{region}_power"] = features["power"][band][region]
            row_data[f"{condition_label}_{band}_{region}_flatness"] = features["flatness"][band][region]
    for region in regions:
        row_data[f"{condition_label}_{region}_paf"] = features["paf"][region]


def format_summary_dataframe(df):
    formatted = df.copy()
    for col in formatted.columns:
        if col == "Participant":
            continue
        if col.endswith("_power"):
            formatted[col] = formatted[col].apply(
                lambda x: f"{x:.7e}" if pd.notnull(x) else ""
            )
        elif col.endswith("_flatness"):
            formatted[col] = formatted[col].apply(
                lambda x: f"{x:.4f}" if pd.notnull(x) else ""
            )
        elif col.endswith("_paf"):
            formatted[col] = formatted[col].apply(
                lambda x: f"{x:.2f}" if pd.notnull(x) else ""
            )
    return formatted


def run_extraction(
    main_dir=MAIN_DIR,
    session=SESSION,
    participants_filter=None,
    max_participants=None,
    condition_labels=None,
    output_dir=".",
):
    validate_region_channel_map(SENSOR_REGION_CHANNELS, EXPECTED_64_CHANNELS)

    regions = ["frontal", "parietal", "temporal", "occipital"]
    if condition_labels is None:
        condition_labels = _all_condition_labels()

    print("=" * 80)
    print("ROI SENSOR POWER EXTRACTION (FOUR REGIONS, SUMMARY ONLY)")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    print(f"Main directory: {main_dir}")
    print(f"Session: {session}")
    print(f"Condition labels: {condition_labels}")
    print(f"Frequency bands: {FREQUENCY_BANDS}")
    print(f"Regions: {regions}")

    participant_ids = scan_for_participants(main_dir)
    participant_ids = [
        pid for pid in participant_ids
        if participant_in_filter(pid, participants_filter)
    ]
    if max_participants is not None:
        participant_ids = participant_ids[:max_participants]

    print(f"\nFound {len(participant_ids)} participant(s) after filtering.")
    if not participant_ids:
        print("No participants to process.")
        return None

    valid_participants = {}
    for participant_id in participant_ids:
        condition_files = {}
        missing = []
        for condition_label in condition_labels:
            task, acq = condition_label_to_task_acq(condition_label)
            set_path = get_set_file_path(main_dir, participant_id, session, task, acq)
            if set_path is None:
                missing.append(condition_label)
            else:
                condition_files[condition_label] = set_path

        if not condition_files:
            print(f"  [SKIP] {participant_id}: no requested conditions found.")
            continue
        if missing:
            print(f"  [WARN] {participant_id}: missing conditions {missing}")
        valid_participants[participant_id] = condition_files

    if not valid_participants:
        print("No valid participants with available files.")
        return None

    summary_rows = []
    for idx, (participant_id, condition_files) in enumerate(valid_participants.items(), start=1):
        print(f"\n[{idx}/{len(valid_participants)}] Processing {participant_id}")
        row_data = {"Participant": participant_id}

        for condition_label in condition_labels:
            if condition_label not in condition_files:
                set_condition_nan_values(row_data, condition_label, FREQUENCY_BANDS, regions)
                continue

            set_path = condition_files[condition_label]
            print(f"  Condition: {condition_label}")
            try:
                epochs = load_epochs(set_path)
                psd = compute_psd_multitaper(epochs, bandwidth=PSD_BANDWIDTH)
                features = compute_condition_region_features(
                    psd, SENSOR_REGION_CHANNELS, FREQUENCY_BANDS
                )
                apply_condition_results_to_row(
                    row_data, condition_label, features, FREQUENCY_BANDS, regions
                )

                for band in FREQUENCY_BANDS.keys():
                    summary_chunks = []
                    for region in regions:
                        pwr = features["power"][band][region]
                        flat = features["flatness"][band][region]
                        summary_chunks.append(f"{region[:1].upper()} Pwr: {pwr:.2e}, Flat: {flat:.4f}")
                    print(f"    {band}: " + " | ".join(summary_chunks))
                paf_str = ", ".join(
                    [f"{region[:1].upper()}: {features['paf'][region]:.2f} Hz" for region in regions]
                )
                print(f"    PAF: {paf_str}")
            except Exception as exc:
                print(f"    [ERROR] {condition_label}: {type(exc).__name__}: {exc}")
                set_condition_nan_values(row_data, condition_label, FREQUENCY_BANDS, regions)

        summary_rows.append(row_data)

    summary_df = pd.DataFrame(summary_rows)
    expected_cols = get_expected_columns(condition_labels, FREQUENCY_BANDS, regions)
    for col in expected_cols:
        if col not in summary_df.columns:
            summary_df[col] = np.nan

    ordered_columns = ["Participant"] + expected_cols
    summary_df = summary_df[ordered_columns]
    formatted_summary_df = format_summary_dataframe(summary_df)

    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_timestamped = os.path.join(
        output_dir, f"{BASE_OUTPUT_NAME_SUMMARY}_{timestamp}.csv"
    )
    output_latest = os.path.join(output_dir, f"{BASE_OUTPUT_NAME_SUMMARY}.csv")

    formatted_summary_df.to_csv(output_timestamped, index=False)
    formatted_summary_df.to_csv(output_latest, index=False)

    print("\n" + "-" * 80)
    print("Completed.")
    print(f"Summary rows: {len(formatted_summary_df)}")
    print(f"Summary columns: {len(formatted_summary_df.columns)}")
    print(f"Timestamped output: {output_timestamped}")
    print(f"Latest alias output: {output_latest}")
    print("-" * 80)

    return output_timestamped


def build_parser():
    parser = argparse.ArgumentParser(
        description="Extract ROI sensor-space power/flatness/PAF across four regions."
    )
    parser.add_argument(
        "--main-dir",
        default=MAIN_DIR,
        help=f"Main preprocessed directory (default: {MAIN_DIR})",
    )
    parser.add_argument(
        "--session",
        default=SESSION,
        help=f"Session label (default: {SESSION})",
    )
    parser.add_argument(
        "--participants",
        default=None,
        help="Comma-separated participant IDs (e.g., 'sub-001,sub-002' or '001,002').",
    )
    parser.add_argument(
        "--max-participants",
        type=int,
        default=None,
        help="Limit number of participants after filtering.",
    )
    parser.add_argument(
        "--condition-labels",
        default=None,
        help=(
            "Comma-separated condition labels from "
            f"{_all_condition_labels()} (default: all)."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=".",
        help="Directory for summary CSV outputs (default: current directory).",
    )
    parser.add_argument(
        "--validate-map-only",
        action="store_true",
        help="Run channel map integrity checks and exit.",
    )
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()

    try:
        validate_region_channel_map(SENSOR_REGION_CHANNELS, EXPECTED_64_CHANNELS)
    except Exception as exc:
        raise SystemExit(f"Region-channel map validation failed: {exc}") from exc

    if args.validate_map_only:
        print("Region-channel map validation passed.")
        raise SystemExit(0)

    participant_filter = parse_participant_filter(args.participants)
    condition_labels = parse_condition_labels(args.condition_labels)

    run_extraction(
        main_dir=args.main_dir,
        session=args.session,
        participants_filter=participant_filter,
        max_participants=args.max_participants,
        condition_labels=condition_labels,
        output_dir=args.output_dir,
    )
