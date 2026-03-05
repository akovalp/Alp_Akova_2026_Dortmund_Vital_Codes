# ROI-Based Source-Space wPLI Connectivity (Four Regions): Methodological Details

## 1. Overview

This document describes the comprehensive methodology for extracting region-of-interest (ROI) based source-space weighted Phase Lag Index (wPLI) connectivity from electroencephalographic (EEG) data using a four-region parcellation scheme. The pipeline computes functional connectivity metrics within and between four canonical brain regions: Frontal, Parietal, Temporal, and Occipital. This extends the traditional anterior-posterior dichotomy to provide a more granular characterization of cortical connectivity patterns.

### 1.1 Primary Objectives

- Quantify phase-synchronization between four anatomically defined brain regions using wPLI
- Extract connectivity metrics for within-region coupling (4 regions) and between-region coupling (6 pairs)
- Provide robust connectivity estimates across three canonical frequency bands (theta, alpha, beta)
- Enable comprehensive network analysis beyond simple anterior-posterior dichotomies

### 1.2 What is wPLI?

**Weighted Phase Lag Index (wPLI)** quantifies consistent non-zero phase-lag relationships between two signals while weighting contributions by the magnitude of the imaginary cross-spectrum. Compared with PLI, wPLI down-weights small near-zero-lag imaginary components and is generally more robust to noise.

**Mathematical Definition**:

```
wPLI = |E[|Im(S)| sign(Im(S))]| / E[|Im(S)|]
```

Where:
- **S** is the cross-spectrum between two signals
- **Im(S)** is the imaginary part of the cross-spectrum
- **sign()** returns +1 for positive imaginary parts, -1 for negative
- **E[]** denotes the expected value (average across time/epochs)

**Key Properties**:
- **Range**: 0 to 1 (0 = no phase coupling, 1 = strong consistent phase-lag coupling)
- **Volume conduction immunity**: wPLI is insensitive to zero-lag coupling (volume-conducted signals)
- **Robustness**: Less sensitive than PLI to small, noisy imaginary cross-spectrum values

---

## 2. Data Structure and Input Requirements

### 2.1 Data Organization

The pipeline expects a BIDS-derivative directory structure:

```
<MAIN_DIR>/
└── sub-<participant_id>/
    └── ses-1/
        └── eeg/
            └── sub-<participant_id>_ses-1_task-<condition>_acq-<acquisition>_eeg.set
```

**Parameters:**
- **Main Directory**: Root path to preprocessed data (`/Users/alpmac/CodeWorks/Trento/PreProcessedData/Complete`)
- **Session**: Fixed as `ses-1`
- **Conditions**: `EyesClosed`, `EyesOpen`
- **Acquisitions**: `pre`, `post` (representing experimental timepoints)

### 2.2 Input Data Format

- **File Format**: EEGLAB `.set` files containing epoched EEG data
- **Epoch Structure**: Continuous data segmented into discrete epochs during preprocessing
- **Channel Configuration**: Standard 10-20 electrode montage (64+ channels recommended)
- **Preprocessing Status**: Data should be preprocessed (filtered, artifact-corrected, epoched) prior to source analysis

---

## 3. Source Modeling Methodology

### 3.1 Forward Solution Computation

#### 3.1.1 Source Space Specification

**Template Brain**: `fsaverage` (FreeSurfer average brain)

**Source Space Resolution**: ico-4 subdivision
- Provides approximately 2,562 vertices per hemisphere (5,124 total)
- Represents a balance between spatial resolution and computational efficiency
- Equivalent to ~10 mm average source spacing

**Source Space Path**: `fsaverage/bem/fsaverage-ico-4-src.fif`

#### 3.1.2 Boundary Element Model (BEM)

**BEM Configuration**: Three-layer BEM (inner skull / outer skull / scalp surfaces)
- **Inner Skull Mesh**: 5,120 vertices
- **Outer Skull Mesh**: 5,120 vertices  
- **Scalp Mesh**: 5,120 vertices

**BEM Solution Path**: `fsaverage/bem/fsaverage-5120-5120-5120-bem-sol.fif`

**Rationale**: The three-layer BEM provides an accurate representation of conductivity profiles (brain, skull, scalp) while maintaining computational tractability for large-scale batch processing.

#### 3.1.3 Forward Solution Parameters

```python
mne.make_forward_solution(
    info=epochs.info,
    trans='fsaverage',                    # fsaverage head-to-MRI transformation
    src=src,                              # ico-4 source space
    bem=bem,                              # Three-layer BEM
    eeg=True,                             # Include EEG forward model
    mindist=5.0,                          # Minimum source-to-cortex distance (mm)
    n_jobs=-1,                            # Parallel computation
)
```

**Key Parameters**:
- **mindist=5.0 mm**: Excludes sources within 5 mm of the cortical surface to avoid modeling artifacts from sulcal geometry
- **trans='fsaverage'**: Uses the standard fsaverage head-to-MRI transformation, suitable when individual T1-weighted structural MRI is unavailable

### 3.2 Inverse Solution Computation

#### 3.2.1 Inverse Method: eLORETA

**Method Selection**: Exact Low-Resolution Brain Electromagnetic Tomography (eLORETA)

**Theoretical Foundation**:
- eLORETA is a weighted minimum norm estimate that provides zero localization error in the noise-free limit
- Unlike standard MNE or sLORETA, eLORETA achieves exact localization even for deep sources
- The method iteratively reweights source covariance based on the current solution, achieving improved spatial resolution

**Mathematical Formulation**:

Given the forward model **G** (sensors × sources) and sensor data **Φ**, the inverse solution **J** is:

```
J = W * G^T * (G * W * G^T + C)^(-1) * Φ
```

Where:
- **W** is the diagonal weighting matrix with elements w_i
- **C** is the sensor noise covariance
- Weights are updated iteratively: w_i = |J_i|^-k (typically k=2)

#### 3.2.2 Regularization Parameters

**Signal-to-Noise Ratio (SNR)**: 1.0 (for single epochs)
- Lower SNR provides appropriate regularization for single-epoch inverse solutions
- **Regularization Parameter (λ²)**: λ² = 1/SNR² = 1.0

**Rationale**: Single epochs have lower SNR than averaged data, requiring stronger regularization to ensure stable source estimates.

#### 3.2.3 Orientation Constraint

**Parameter**: `pick_ori='normal'`

**Explanation**: 
- Constrains source dipoles to be oriented perpendicular to the cortical surface
- Reduces the solution space by 66% (from 3D vectors to 1D scalars)
- Physiologically justified as pyramidal neurons (primary EEG generators) are predominantly perpendicularly oriented
- Critical for accurate phase estimation in connectivity analysis

#### 3.2.4 Inverse Operator Creation

```python
make_inverse_operator(
    info=epochs.info,
    forward=fwd,
    noise_cov=noise_cov,
    loose=0.2,                            # Orientation loose constraint
    depth=0.8,                            # Depth weighting exponent
)
```

**depth=0.8**: Depth weighting compensates for the inherent bias of EEG toward superficial sources by assigning higher weights to deeper sources during inverse computation.

**loose=0.2**: When combined with `pick_ori='normal'`, this parameter has minimal effect but is retained for compatibility with the inverse operator interface.

### 3.3 Noise Covariance Estimation

**Method**: Ad-hoc noise covariance (`mne.make_ad_hoc_cov`)

**Rationale**: When preprocessed epochs lack dedicated baseline periods for empirical noise estimation, an ad-hoc covariance provides a reasonable approximation:
- Diagonal covariance assumes uncorrelated noise across sensors
- Uses default MNE ad-hoc covariance parameters

**Note**: For optimal results, empirical noise covariance should be computed from prestimulus baseline periods when available.

---

## 4. Anatomical Parcellation and ROI Definitions

### 4.1 Atlas Specification

**Atlas**: Desikan-Killiany (aparc)
- 34 cortical regions per hemisphere (68 total regions)
- Based on gyral and sulcal boundaries from FreeSurfer cortical reconstruction
- Widely adopted in neuroimaging for its anatomical interpretability

**Subjects Directory**: Derived from `fsaverage` template

### 4.2 Label Processing Pipeline

#### 4.2.1 Label Loading

```python
all_labels = mne.read_labels_from_annot(
    'fsaverage',
    parc='aparc',
    subjects_dir=subjects_dir,
)
```

This loads all cortical labels defined in the Desikan-Killiany atlas.

#### 4.2.2 Label Filtering and Restriction

**Excluded Labels**: 
- `unknown`: Undefined regions
- `corpuscallosum`: Non-cortical structure (white matter tract)

**Restriction to Source Space**:
- Each anatomical label is intersected with the ico-4 source space vertices
- Labels with zero vertices in the source space are excluded
- This ensures that all extracted connectivity estimates correspond to valid source locations

**Rationale**: The ico-4 source space has lower resolution than the native surface mesh; restricting labels prevents inclusion of anatomical regions that lack corresponding source dipoles.

### 4.3 Four-Region Parcellation Scheme

The pipeline classifies cortical labels into four distinct anatomical regions: **Frontal**, **Parietal**, **Temporal**, and **Occipital**. This provides a more granular parcellation than the traditional anterior-posterior dichotomy.

#### 4.3.1 Frontal Region

**Definition**: Comprises prefrontal, premotor, motor, and cingulate regions

**Included Labels** (both hemispheres):
- `superiorfrontal` (BA6, BA8)
- `rostralmiddlefrontal` (BA10, BA46)
- `caudalmiddlefrontal` (BA6, BA8, BA9)
- `parsopercularis` (BA44)
- `parstriangularis` (BA45)
- `parsorbitalis` (BA47)
- `lateralorbitofrontal` (BA11, BA47)
- `medialorbitofrontal` (BA11, BA12, BA13)
- `precentral` (BA4, primary motor cortex)
- `paracentral` (BA5, BA31)
- `frontalpole` (BA10)
- `rostralanteriorcingulate` (BA24, BA32)
- `caudalanteriorcingulate` (BA24, BA32)

**Count**: 13 regions × 2 hemispheres = 26 labels

**Functional Significance**: These regions subserve executive functions, working memory, motor planning, decision-making, and cognitive control. The frontal lobe is crucial for goal-directed behavior and serves as the hub of cognitive control networks.

#### 4.3.2 Parietal Region

**Definition**: Comprises somatosensory, superior, and inferior parietal regions

**Included Labels** (both hemispheres):
- `superiorparietal` (BA7)
- `inferiorparietal` (BA39, BA40)
- `supramarginal` (BA40)
- `postcentral` (BA1, BA2, BA3 - primary somatosensory)
- `precuneus` (BA7, BA31)
- `posteriorcingulate` (BA23, BA31)
- `isthmuscingulate` (BA26, BA29, BA30)

**Count**: 7 regions × 2 hemispheres = 14 labels

**Functional Significance**: These regions support visuospatial processing, sensory integration, attention, and multimodal information processing. The parietal lobe serves as a critical hub for the dorsal attention network and sensorimotor integration.

#### 4.3.3 Temporal Region

**Definition**: Comprises lateral, medial, and inferior temporal regions

**Included Labels** (both hemispheres):
- `superiortemporal` (BA22, BA41, BA42)
- `middletemporal` (BA21)
- `inferiortemporal` (BA20)
- `bankssts` (superior temporal sulcus banks)
- `fusiform` (BA37)
- `transversetemporal` (BA41, primary auditory cortex)
- `entorhinal` (BA28, BA34)
- `temporalpole` (BA38)
- `parahippocampal` (BA35, BA36)

**Count**: 9 regions × 2 hemispheres = 18 labels

**Functional Significance**: These regions are essential for auditory processing, language comprehension, memory encoding, and visual object recognition. The temporal lobe plays a central role in the ventral visual stream and memory networks.

#### 4.3.4 Occipital Region

**Definition**: Comprises primary and association visual cortex regions

**Included Labels** (both hemispheres):
- `lateraloccipital` (BA18, BA19)
- `lingual` (BA18, BA19)
- `cuneus` (BA17, BA18)
- `pericalcarine` (BA17, primary visual cortex)

**Count**: 4 regions × 2 hemispheres = 8 labels

**Functional Significance**: These regions constitute the primary visual cortex and visual association areas. The occipital lobe is the endpoint of the retinogeniculostriate pathway and is essential for all visual processing.

### 4.4 Total Label Count Summary

| Region | Regions per Hemisphere | Total Labels | Functional Network |
|--------|----------------------|--------------|-------------------|
| Frontal | 13 | 26 | Executive/Motor |
| Parietal | 7 | 14 | Attention/Sensory |
| Temporal | 9 | 18 | Auditory/Memory/Language |
| Occipital | 4 | 8 | Visual |
| **Total** | **33** | **66** | - |

*Note: Two labels (unknown, corpuscallosum) are excluded from the original 68 Desikan-Killiany labels, leaving 66 valid cortical labels.*

### 4.5 Hemispheric Handling

**Bilateral Processing**: Labels from both left (LH) and right (RH) hemispheres are included in their respective regions

**Label Naming Convention**: Labels are suffixed with `-lh` or `-rh` in the annotation system

**Connectivity Aggregation**: For network-level wPLI computation:
- Within-region connectivity averages over all label pairs within the same region
- Between-region connectivity averages over all cross-region label pairs
- All analyses include bilateral contributions from both hemispheres

---

## 5. Label Time Course Extraction

### 5.1 Source Reconstruction at Epoch Level

**Procedure**: Apply the inverse operator to each epoch independently

```python
stcs = apply_inverse_epochs(
    epochs,
    inverse_operator,
    lambda2=1.0,                          # SNR = 1.0 for single epochs
    method='eLORETA',
    pick_ori='normal',
    return_generator=True,                # Memory-efficient streaming
)
```

**Output**: SourceEstimate generator yielding source time courses at all 5,124 vertices for each epoch

**Rationale**: 
- Processing epochs individually preserves trial-to-trial variability essential for wPLI computation
- Generator pattern enables memory-efficient processing of large datasets
- Lower SNR (1.0) provides appropriate regularization for single-epoch inverse solutions

### 5.2 Label Time Course Computation

**Method**: `mne.extract_label_time_course`

**Parameters**:
```python
mne.extract_label_time_course(
    stcs,
    labels,
    src,
    mode='mean_flip',                     # Critical for connectivity analysis
    return_generator=True,
)
```

**Mode: 'mean_flip'**

**Explanation**: 
- Averages source time courses across all vertices within each anatomical label
- The 'flip' component accounts for source orientation: signals are flipped such that the orientation direction is consistent across all vertices
- This prevents cancellation of oppositely oriented dipoles within the same region
- Yields a single time series per label per epoch
- Essential for accurate phase estimation in connectivity analysis

**Mathematical Formulation**:

For label **L** containing vertices {v₁, v₂, ..., vₙ} with source time courses **s₁(t), s₂(t), ..., sₙ(t)**:

```
s_L(t) = (1/n) * Σ |sign(J_v) · s_v(t)|
```

Where **J_v** is the source dipole orientation at vertex **v**.

**Output Dimensions**: Generator yielding (n_labels, n_times) arrays per epoch

---

## 6. wPLI Connectivity Computation

### 6.1 Spectral Connectivity Estimation

#### 6.1.1 Method: Multitaper Spectral Connectivity

**Function**: `spectral_connectivity_epochs` (MNE-Connectivity)

**Rationale**: Multitaper methods provide optimal control over the trade-off between spectral resolution and variance reduction by using orthogonal tapers. For connectivity analysis, this is crucial because:
- wPLI requires reliable phase estimates across trials
- Multitaper averaging reduces variance in phase estimates
- Frequency-specific connectivity can be isolated

**Parameters**:
```python
con = spectral_connectivity_epochs(
    label_ts,                            # Generator of label time courses
    method=['wpli'],                     # Weighted Phase Lag Index
    mode='multitaper',                   # Multitaper spectral estimation
    sfreq=sfreq,                         # Sampling frequency
    fmin=fmin,                           # Band minimum
    fmax=fmax,                           # Band maximum
    faverage=True,                       # Average connectivity across band
    mt_adaptive=True,                    # Adaptive taper weighting
    n_jobs=1,
)
```

**Detailed Parameter Explanations**:

**method=['wpli']**:
- Computes weighted Phase Lag Index specifically
- wPLI = |E[|Im(S)| sign(Im(S))]| / E[|Im(S)|], where S is the cross-spectrum
- Immune to zero-lag volume conduction and more robust to small noisy phase differences than PLI

**mode='multitaper'**:
- Uses discrete prolate spheroidal sequences (DPSS, Slepian tapers)
- Provides multiple independent estimates of the cross-spectrum
- Reduces variance in connectivity estimates through taper averaging

**faverage=True**:
- Averages wPLI across all frequencies within the specified band
- Returns single connectivity value per band (not frequency-resolved)
- Appropriate when band-level summary is desired rather than frequency-specific connectivity

**mt_adaptive=True**:
- Uses adaptive weighting of tapers based on spectral concentration
- Optimizes the multitaper estimator for varying spectral content
- Increases accuracy of phase estimates, particularly important for wPLI

### 6.2 Frequency Bands

**Canonical EEG Frequency Bands**:

| Band | Frequency Range | Primary Neural Correlates |
|------|-----------------|-------------------------|
| Theta | 4 - 8 Hz | Drowsiness, working memory, encoding |
| Alpha | 8 - 14 Hz | Idling, inhibition, visual attention |
| Beta | 14 - 30 Hz | Active thinking, focus, motor control |

**Rationale**: These bands represent well-established oscillatory phenomena with distinct functional roles in cognition and perception. Connectivity within these bands may reflect distinct neural communication mechanisms.

### 6.3 Connectivity Matrix Extraction

**Output Structure**:
The `spectral_connectivity_epochs` function returns a connectivity object with dimensions (n_labels, n_labels, n_bands). With `faverage=True`, this yields a single wPLI value per label pair per frequency band.

**Matrix Properties**:
- **Shape**: (n_labels, n_labels) where n_labels = 66 (after filtering)
- **Symmetry**: wPLI matrices are symmetric (wPLI(A→B) = wPLI(B→A))
- **Diagonal**: Self-connectivity values are not analyzed and are excluded from regional summaries
- **Range**: [0, 1] where 0 = no phase coupling, 1 = strong consistent phase-lag coupling

### 6.4 Regional Connectivity Aggregation

From the full label-level wPLI matrix, ten connectivity metrics are extracted: 4 within-region and 6 between-region measures.

#### 6.4.1 Within-Region Connectivity

**Definition**: Mean wPLI between all pairs of labels within the same region

**Computation**:
```python
# For each region
idx = region_indices[region]                    # Get label indices for region
region_sub = pli_matrix[np.ix_(idx, idx)]       # Extract submatrix
within_pli = get_off_diag_mean(region_sub)      # Mean of off-diagonal elements
```

Where `get_off_diag_mean` computes:
```
mean = Σ wPLI(i,j) / (n_pairs) for all i≠j in region labels
```

**Four Within-Region Metrics**:
1. **Frontal Within-Region wPLI**: Mean connectivity within frontal regions (26 labels)
2. **Parietal Within-Region wPLI**: Mean connectivity within parietal regions (14 labels)
3. **Temporal Within-Region wPLI**: Mean connectivity within temporal regions (18 labels)
4. **Occipital Within-Region wPLI**: Mean connectivity within occipital regions (8 labels)

**Interpretation**: Reflects functional coupling within each anatomical region. Higher values indicate stronger phase synchronization between regions belonging to the same functional network.

#### 6.4.2 Between-Region Connectivity

**Definition**: Mean wPLI between all pairs of labels from two different regions

**Six Unique Region Pairs** (combinations of 4 regions taken 2 at a time):
1. **Frontal-Parietal**
2. **Frontal-Temporal**
3. **Frontal-Occipital**
4. **Parietal-Temporal**
5. **Parietal-Occipital**
6. **Temporal-Occipital**

**Computation**:
```python
# For each region pair
idx1 = region_indices[region1]                  # Get label indices for region 1
idx2 = region_indices[region2]                  # Get label indices for region 2
cross_sub = pli_matrix[np.ix_(idx1, idx2)]      # Extract cross-region submatrix
between_pli = np.mean(cross_sub)                # Mean of all cross-region pairs
```

**Interpretation**: Reflects long-range functional coupling between different anatomical networks. These cross-region connections are particularly sensitive to:
- **Frontal-Parietal**: Cognitive control and attention networks
- **Frontal-Temporal**: Language and memory integration
- **Frontal-Occipital**: Top-down visual attention
- **Parietal-Temporal**: Sensorimotor-auditory integration
- **Parietal-Occipital**: Dorsal visual stream
- **Temporal-Occipital**: Ventral visual stream

### 6.5 Connectivity Matrix Summary

**Within-Region Metrics** (4):
- Frontal-Frontal: 26 × 25 / 2 = 325 unique pairs
- Parietal-Parietal: 14 × 13 / 2 = 91 unique pairs
- Temporal-Temporal: 18 × 17 / 2 = 153 unique pairs
- Occipital-Occipital: 8 × 7 / 2 = 28 unique pairs

**Between-Region Metrics** (6):
- Frontal-Parietal: 26 × 14 = 364 unique pairs
- Frontal-Temporal: 26 × 18 = 468 unique pairs
- Frontal-Occipital: 26 × 8 = 208 unique pairs
- Parietal-Temporal: 14 × 18 = 252 unique pairs
- Parietal-Occipital: 14 × 8 = 112 unique pairs
- Temporal-Occipital: 18 × 8 = 144 unique pairs

**Total Unique Pairs Analyzed**: 325 + 91 + 153 + 28 + 364 + 468 + 208 + 252 + 112 + 144 = 2,145 pairs

---

## 7. Data Aggregation and Output

### 7.1 Output Metrics

For each participant, condition, acquisition, and frequency band, the pipeline computes:

**Within-Region Metrics (4)**:
1. Frontal within-region wPLI
2. Parietal within-region wPLI
3. Temporal within-region wPLI
4. Occipital within-region wPLI

**Between-Region Metrics (6)**:
1. Frontal-Parietal between-region wPLI
2. Frontal-Temporal between-region wPLI
3. Frontal-Occipital between-region wPLI
4. Parietal-Temporal between-region wPLI
5. Parietal-Occipital between-region wPLI
6. Temporal-Occipital between-region wPLI

**Total per Band**: 10 connectivity metrics
**Total per Condition**: 10 metrics × 3 bands = 30 metrics
**Total per Participant**: 30 metrics × 4 conditions = 120 metrics

### 7.2 Naming Convention

**Pattern (legacy compatibility)**: `{Condition}_{Acquisition}_{Band}_{Regions}_pli`

**Important**: The `_pli` suffix is intentionally retained in column names for downstream compatibility, but values are computed using `method=['wpli']`.

**Within-Region Examples**:
- `EyesClosed_pre_theta_frontal_within_pli`
- `EyesOpen_post_alpha_parietal_within_pli`
- `EyesClosed_pre_beta_temporal_within_pli`
- `EyesOpen_post_theta_occipital_within_pli`

**Between-Region Examples**:
- `EyesClosed_pre_theta_frontal_parietal_between_pli`
- `EyesOpen_post_alpha_frontal_temporal_between_pli`
- `EyesClosed_pre_beta_parietal_occipital_between_pli`

### 7.3 Output Format

**Wide-Format Table**: One row per participant with columns for all condition-band-metric combinations

**Columns**:
- `Participant`: Participant identifier
- 120 metric columns (3 bands × 2 conditions × 2 acquisitions × 10 metrics)

**Example Row Structure**:
| Participant | EyesClosed_pre_theta_frontal_within_pli | EyesClosed_pre_theta_parietal_within_pli | ... | EyesOpen_post_beta_temporal_occipital_between_pli |
|-------------|----------------------------------------|-----------------------------------------|-----|--------------------------------------------------|
| sub-001 | 0.123456 | 0.234567 | ... | 0.345678 |

**Precision Formatting**:
- Connectivity (wPLI) values: 6 decimal places (e.g., 0.123456)
- Ensures sub-threshold precision for statistical analysis

### 7.4 File Naming and Version Control

**Timestamp Convention**: `YYYYMMDD_HHMMSS` format

**Output Files**:
- Summary: `source_pli_connectivity_four_regions_summary_<timestamp>.csv`
- Error Log: `source_pli_errors_<timestamp>.txt` (if errors occur)

**Resume Capability**: 
- The pipeline detects the most recent output file and continues from the last completed participant
- Participants with complete data across all conditions are skipped
- Incomplete or missing participants are processed

---

## 8. Computational Implementation

### 8.1 Parallel Processing Architecture

**Strategy**: Participant-level parallelization with condition-level nested parallelism

**Batch Size**: 10 participants per batch (`N_PARTICIPANTS_JOBS = 10`)

**Parallelization Framework**: `joblib.Parallel`

**Hierarchy**:
1. **Batch Level**: Participants processed in batches of 10
2. **Participant Level**: Each participant's inverse operator computed once
3. **Condition Level**: Within each participant, 4 conditions processed in parallel (4 workers)
4. **Frequency Level**: Within each condition, 3 frequency bands processed sequentially

**Total Parallelism**: For 10 participants with 4 conditions each:
- 10 participants × 4 conditions = 40 parallel workers maximum
- This uses available CPU cores efficiently while preventing memory exhaustion

**Rationale**:
- Parallelization at condition level within participants allows inverse operator reuse
- Batches prevent excessive memory consumption from loading multiple participants simultaneously
- Sequential band processing within conditions reduces overhead for connectivity computation

### 8.2 Memory Management

**Garbage Collection**: Explicit cleanup after each participant and condition

```python
del epochs
del inverse_operator
del stcs
gc.collect()
```

**Resource Sharing**: Forward solution, noise covariance, labels, and source space computed once globally and shared across all participants in a batch

**Generator Patterns**: 
- `apply_inverse_epochs` with `return_generator=True` streams source estimates
- `extract_label_time_course` with `return_generator=True` streams label time courses
- This eliminates need to store full source-space data in memory

**Efficiency**: 
- Global resources computed once per batch (~15-30 seconds overhead)
- Inverse operator computed once per participant (~5-10 seconds overhead)
- Minimal memory duplication across parallel workers

### 8.3 Error Handling

**Exception Scope**: Individual participant-condition failures do not halt the pipeline

**Error Logging**: Comprehensive error information captured:
- Participant ID
- Condition
- File path
- Exception type
- Error message
- Full traceback

**Output**: Errors saved to timestamped log file for review and debugging

---

## 9. Parameter Justification

### 9.1 Source Modeling Parameters

| Parameter | Value | Justification |
|-----------|-------|---------------|
| Source Space | ico-4 | Balance of resolution and efficiency |
| Inverse Method | eLORETA | Zero localization error; improved spatial resolution |
| Orientation Constraint | `normal` | Physiologically justified; critical for phase estimation |
| SNR (epoch) | 1.0 | Appropriate regularization for single-epoch solutions |
| Depth Weighting | 0.8 | Compensates for superficial bias in EEG |

### 9.2 Connectivity Analysis Parameters

| Parameter | Value | Justification |
|-----------|-------|---------------|
| Method | wPLI | Volume conduction immune; robust weighted phase-lag connectivity estimate |
| Spectral Method | Multitaper | Optimal variance-resolution trade-off for phase estimation |
| Adaptive Tapers | True | Frequency-dependent taper weighting improves accuracy |
| Frequency Range | 4-30 Hz | Covers theta, alpha, beta bands relevant to cognition |
| Band Averaging | True | Provides stable summary per canonical band |

### 9.3 Regional Parcellation Parameters

| Parameter | Value | Justification |
|-----------|-------|---------------|
| Number of Regions | 4 | Comprehensive coverage of major cortical lobes |
| Within-Region Metrics | 4 | One per region for network-specific analysis |
| Between-Region Metrics | 6 | All unique pairs (C(4,2) = 6) for complete network characterization |
| Bilateral Inclusion | Yes | Both hemispheres for comprehensive coverage |

---

## 10. Comparison with Two-Region Approach

### 10.1 Methodological Differences

| Aspect | Two-Region Approach | Four-Region Approach |
|--------|-------------------|---------------------|
| Regions | Anterior (Frontal), Posterior (Parietal+Temporal+Occipital) | Frontal, Parietal, Temporal, Occipital |
| Within-Region Metrics | 2 | 4 |
| Between-Region Metrics | 1 (Anterior-Posterior) | 6 (all unique pairs) |
| Total Metrics | 3 | 10 |
| Spatial Granularity | Coarse | Fine |
| Anatomical Specificity | Low | High |

### 10.2 Anatomical Composition Comparison

**Two-Region Anterior** = Four-Region Frontal (identical)
**Two-Region Posterior** = Four-Region Parietal + Temporal + Occipital (aggregated)

The four-region approach disaggregates the posterior region into three distinct anatomical-functional networks, enabling more specific hypotheses about:
- Visual processing (Occipital)
- Attention and sensorimotor integration (Parietal)
- Auditory and memory processing (Temporal)

### 10.3 Output Metrics Comparison

**Two-Region Outputs (3 metrics)**:
- Within-Anterior wPLI
- Within-Posterior wPLI
- Anterior-Posterior Between wPLI

**Four-Region Outputs (10 metrics)**:
- Within-Frontal wPLI
- Within-Parietal wPLI
- Within-Temporal wPLI
- Within-Occipital wPLI
- Frontal-Parietal Between wPLI
- Frontal-Temporal Between wPLI
- Frontal-Occipital Between wPLI
- Parietal-Temporal Between wPLI
- Parietal-Occipital Between wPLI
- Temporal-Occipital Between wPLI

---

## 11. Validation and Quality Control

### 11.1 Expected Output Validation

**Participants**: Verify expected number of participants processed

**Conditions**: Verify 4 conditions per participant (2 tasks × 2 acquisitions)

**Connectivity Values**: 
- wPLI ranges 0-1 (expect ~0.05-0.40 for typical resting-state EEG)
- Values near 0 indicate no phase coupling
- Values near 1 indicate strong phase locking (rare in resting-state)

**Spatial Patterns**: 
- Posterior (Parietal/Occipital) alpha connectivity typically stronger in eyes-closed condition
- Within-region connectivity typically stronger than between-region
- Frontal theta connectivity may be elevated during cognitive tasks
- Temporal connectivity may show task-specific modulation

### 11.2 Data Quality Checks

**wPLI Value Ranges**:
- Should be 0-1 (negative values indicate computational error)
- Values >0.5 uncommon in resting-state (check for artifacts)
- Values <0.01 may indicate poor signal quality or over-regularization

**Missing Data**: Investigate participants/conditions with NaN values
- Common causes: bad epochs, insufficient data, processing errors

**Cross-Band Consistency**:
- Connectivity patterns should be relatively consistent across bands
- Dramatic differences may indicate artifacts or processing errors

**Regional Consistency**:
- Within-region values should generally be higher than between-region values
- Occipital within-region connectivity often strongest in alpha band during rest
- Frontal connectivity often strongest in theta band

---

**Document Version**: 1.1  
**Last Updated**: 2026-02-10  
**Corresponding Script**: `source_pli_connectivity_four_regions.py`  
**Compatible with**: MNE-Python ≥ 1.0, MNE-Connectivity ≥ 0.5, joblib ≥ 1.0
