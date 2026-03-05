# ROI-Based Source Power Extraction: Methodological Details

## 1. Overview

This document describes the comprehensive methodology for extracting region-of-interest (ROI) based source-space power features from electroencephalographic (EEG) data. The pipeline computes spectral power features in canonical frequency bands (theta, alpha, beta) at the source level using anatomically defined regions from the Desikan-Killiany cortical atlas. The methodology ensures direct comparability with phase-lag index (PLI) connectivity analysis by employing identical source modeling parameters, ROI definitions, and spatial resolution specifications.

### 1.1 Primary Objectives

- Quantify oscillatory power characteristics in source space across anatomically defined brain regions
- Extract canonical spectral metrics: band power, spectral flatness, and peak alpha frequency (PAF)
- Provide both region-level (frontal/parietal/temporal/occipital) and label-level feature extraction
- Ensure methodological consistency with companion connectivity analyses for integrative multimodal studies

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
- **Main Directory**: Root path to preprocessed data
- **Session**: Fixed as `ses-1`
- **Conditions**: `EyesClosed`, `EyesOpen`
- **Acquisitions**: `pre`, `post` (representing experimental timepoints)

### 2.2 Input Data Format

- **File Format**: EEGLAB `.set` files containing epoched EEG data
- **Epoch Structure**: Continuous data segmented into discrete epochs during preprocessing
- **Channel Configuration**: Standard 10-20 electrode montage (64+ channels recommended)
- **Preprocessing Status**: Data should be preprocessed (filtered, artifact-corrected, epoched) prior to source analysis

## 3. Source Modeling Methodology

### 3.1 Forward Solution Computation

#### 3.1.1 Source Space Specification

**Template Brain**: `fsaverage` (FreeSurfer average brain)

**Source Space Resolution**: ico-4 subdivision
- Provides approximately 2,562 vertices per hemisphere
- Represents a balance between spatial resolution and computational efficiency
- Equivalent to ~10 mm average source spacing
- Consistent with companion PLI connectivity pipeline for direct feature comparability

**Source Space Path**: `fsaverage/bem/fsaverage-ico-4-src.fif`

#### 3.1.2 Boundary Element Model (BEM)

**BEM Configuration**: Three-layer BEM (inner skull / outer skull / scalp surfaces)
- **Inner Skull Mesh**: 5,120 vertices (5120 source space)
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

**Signal-to-Noise Ratio (SNR)**: 3.0
- Reflects typical SNR in well-preprocessed EEG data
- Determines the regularization strength

**Regularization Parameter (λ²)**: λ² = 1/SNR² = 1/9 ≈ 0.111

**Rationale**: This SNR value is standard for EEG source reconstruction, providing a balance between data fit and solution stability without over-regularizing.

#### 3.2.3 Orientation Constraint

**Parameter**: `pick_ori='normal'`

**Explanation**: 
- Constrains source dipoles to be oriented perpendicular to the cortical surface
- Reduces the solution space by 66% (from 3D vectors to 1D scalars)
- Physiologically justified as pyramidal neurons (primary EEG generators) are predominantly perpendicularly oriented
- Matches companion PLI workflow for feature consistency

#### 3.2.4 Additional Inverse Parameters

```python
make_inverse_operator(
    info=epochs.info,
    forward=fwd,
    noise_cov=noise_cov,
    loose=0.2,                            # Orientation loose constraint (unused with pick_ori='normal')
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

## 4. Anatomical Parcellation and ROI Definitions

### 4.1 Atlas Specification

**Atlas**: Desikan-Killiany (aparc)
- 34 cortical regions per hemisphere (68 total regions)
- Based on gyral and sulcal boundaries from FreeSurfer cortical reconstruction
- Widely adopted in neuroimaging for its anatomical interpretability

**Subjects Directory**: Derived from `fsaverage` template
**Parcellation File**: `fsaverage/label/aparc+aseg.mgz` (implicit through MNE's annotation system)

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
- This ensures that all extracted features correspond to valid source locations

**Rationale**: The ico-4 source space has lower resolution than the native surface mesh; restricting labels prevents inclusion of anatomical regions that lack corresponding source dipoles.

### 4.3 ROI Grouping Scheme

The pipeline classifies cortical labels into **four functional-anatomical groups**: **Frontal**, **Parietal**, **Temporal**, and **Occipital**. This four-region scheme aligns with the PLI connectivity analysis pipeline.

#### 4.3.1 Frontal Group

**Definition**: Comprises prefrontal, premotor, motor, and anterior cingulate regions

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
- `paracentral` (supplementary motor area)
- `frontalpole` (BA10)
- `rostralanteriorcingulate` (BA24, BA32)
- `caudalanteriorcingulate` (BA24, BA32)

**Functional Significance**: These regions subserve executive functions, working memory, motor planning, decision-making, and emotional regulation. Power modulations in these regions are frequently observed in cognitive and affective tasks.

**Note**: Anterior cingulate regions are assigned to the frontal group per project specifications, separating them from posterior cingulate regions assigned to the parietal group.

#### 4.3.2 Parietal Group

**Definition**: Comprises somatosensory, association, and posterior cingulate regions

**Included Labels** (both hemispheres):
- `superiorparietal` (BA7)
- `inferiorparietal` (BA39, BA40)
- `supramarginal` (BA40)
- `postcentral` (BA1, BA2, BA3, primary somatosensory cortex)
- `precuneus` (BA7, BA31)
- `posteriorcingulate` (BA23, BA31)
- `isthmuscingulate` (BA30)

**Functional Significance**: These regions support visuospatial processing, sensory integration, attention, self-referential processing, and the default mode network. The posterior cingulate and precuneus are key hubs of the default mode network.

**Note**: Posterior cingulate and isthmus cingulate are assigned to the parietal group per project specifications, distinct from anterior cingulate regions in the frontal group.

#### 4.3.3 Temporal Group

**Definition**: Comprises lateral and medial temporal lobe structures

**Included Labels** (both hemispheres):
- `superiortemporal` (BA22, auditory association cortex)
- `middletemporal` (BA21)
- `inferiortemporal` (BA20)
- `bankssts` (banks of the superior temporal sulcus)
- `fusiform` (BA37, visual word form area)
- `transversetemporal` (BA41, BA42, primary auditory cortex)
- `entorhinal` (BA28, BA34, memory encoding)
- `temporalpole` (BA38)
- `parahippocampal` (BA35, BA36, memory processing)

**Functional Significance**: These regions support auditory processing, language comprehension, memory encoding and retrieval, face/object recognition, and emotional processing. The temporal lobe contains critical structures for declarative memory (hippocampal formation, entorhinal cortex).

#### 4.3.4 Occipital Group

**Definition**: Comprises primary and association visual cortices

**Included Labels** (both hemispheres):
- `lateraloccipital` (BA18, BA19, visual association)
- `lingual` (BA18, BA19, visual processing)
- `cuneus` (BA17, BA18, primary/secondary visual cortex)
- `pericalcarine` (BA17, primary visual cortex)

**Functional Significance**: These regions constitute the visual cortex, processing visual information from primary reception (V1 in pericalcarine/cuneus) through higher-order visual association areas. Alpha band power is particularly prominent in these regions during eyes-closed resting states.

#### 4.3.5 Other Group

**Definition**: All remaining cortical labels not classified into the four main regions

**Includes**: Insular cortex, remaining cingulate subdivisions not explicitly assigned, and any labels not matching the keyword lists above

**Note**: These regions are included in label-level output but not aggregated in the ROI-group summary.

### 4.4 Hemispheric Handling

**Bilateral Processing**: Labels from both left (LH) and right (RH) hemispheres are included in their respective ROI groups

**Label Naming Convention**: Labels are suffixed with `-lh` or `-rh` in the output

**Aggregation**: For ROI-group level summaries, power features are averaged across all labels within the group regardless of hemisphere, providing bilateral representations of each network.

## 5. Label Time Course Extraction

### 5.1 Source Reconstruction at Epoch Level

**Procedure**: Apply the inverse operator to each epoch independently

```python
stcs = apply_inverse_epochs(
    epochs,
    inverse_operator,
    lambda2=LAMBDA2,
    method='eLORETA',
    pick_ori='normal',
    return_generator=False,
)
```

**Output**: SourceEstimate objects for each epoch containing source time courses at all source space vertices

**Rationale**: Processing epochs individually allows for the preservation of trial-to-trial variability, which is essential for computing epoch-averaged spectral estimates with appropriate variance estimates.

### 5.2 Label Time Course Computation

**Method**: `mne.extract_label_time_course`

**Parameters**:
```python
mne.extract_label_time_course(
    stcs,
    labels,
    src,
    mode='mean_flip',
    allow_empty=False,
)
```

**Mode: 'mean_flip'**

**Explanation**: 
- Averages source time courses across all vertices within each anatomical label
- The 'flip' component accounts for source orientation: signals are flipped such that the orientation direction is consistent across all vertices
- This prevents cancellation of oppositely oriented dipoles within the same region
- Yields a single time series per label per epoch

**Mathematical Formulation**:

For label **L** containing vertices {v₁, v₂, ..., vₙ} with source time courses **s₁(t), s₂(t), ..., sₙ(t)**:

```
s_L(t) = (1/n) * Σ |sign(J_v) · s_v(t)|
```

Where **J_v** is the source dipole orientation at vertex **v**.

**Output Dimensions**: (n_epochs, n_labels, n_times)

## 6. Spectral Feature Extraction

### 6.1 Power Spectral Density Estimation

#### 6.1.1 Method: Multitaper Spectral Estimation

**Function**: `psd_array_multitaper`

**Rationale**: Multitaper methods provide optimal control over the trade-off between spectral resolution and variance reduction by using orthogonal tapers.

**Parameters**:
```python
psds, freqs = psd_array_multitaper(
    label_ts,                              # (n_epochs, n_labels, n_times)
    sfreq=sfreq,
    fmin=4.0,                              # Global minimum frequency (theta)
    fmax=30.0,                             # Global maximum frequency (beta)
    bandwidth=4.0,                        # Time-bandwidth product
    adaptive=False,
    low_bias=True,
    normalization='full',
    n_jobs=1,
)
```

**Detailed Parameter Explanations**:

**bandwidth=4.0 (Time-Bandwidth Product)**:
- Provides frequency smoothing of approximately 4 Hz
- The effective time-bandwidth product and number of tapers depend on epoch duration
- Each taper provides an independent estimate of the PSD
- Final PSD is the average across tapers, reducing variance

**adaptive=False**:
- Uses a fixed set of tapers across all frequencies
- Simplifies computation and ensures consistent degrees of freedom
- Adaptive weighting (adaptive=True) would use frequency-dependent tapers, increasing complexity

**low_bias=True**:
- Excludes tapers with significant spectral leakage at frequency band edges
- Ensures accurate power estimates at band boundaries

**normalization='full'**:
- PSD is normalized to units of V²/Hz
- Provides absolute power estimates (not relative or decibel-scaled)

**Spectral Resolution**:
- Resolution ≈ bandwidth / n_samples
- For typical epochs (e.g., 2-second epochs at 500 Hz sampling): 4 Hz / 1000 samples ≈ 0.004 Hz frequency bins

#### 6.1.2 Epoch Averaging

**Procedure**: Compute mean PSD across epochs for each label

```python
mean_psds = np.mean(psds, axis=0)  # Shape: (n_labels, n_freqs)
```

**Rationale**: 
- Epoch averaging reduces variance of spectral estimates
- Provides stable representation of condition-specific oscillatory power
- Preserves label-specific spectral characteristics

**Mathematical Formulation**:

```
PSD_mean(L, f) = (1/N) * Σ PSD_i(L, f)
```

Where N is the number of epochs, L is the label index, and f is frequency.

### 6.2 Frequency Bands

**Canonical EEG Frequency Bands**:

| Band  | Frequency Range | Primary Neural Correlates             |
| ----- | --------------- | ------------------------------------- |
| Theta | 4 - 8 Hz        | Drowsiness, working memory, encoding  |
| Alpha | 8 - 14 Hz       | Idling, inhibition, visual attention  |
| Beta  | 14 - 30 Hz      | Active thinking, focus, motor control |

**Rationale**: These bands represent well-established oscillatory phenomena with distinct functional roles in cognition and perception.

### 6.3 Band Power Computation

**Method**: Numerical integration using the trapezoidal rule

**Implementation**:
```python
band_psd = label_psd[mask]
band_freqs = freqs[mask]
power = np.trapezoid(band_psd, band_freqs)
```

**Mathematical Formulation**:

For frequency band [f_min, f_max]:

```
Power = ∫[f_min,f_max] PSD(f) df
```

Approximated discretely as:

```
Power ≈ Σ[i=1 to n-1] (f[i+1] - f[i]) * (PSD[i] + PSD[i+1]) / 2
```

**Units**: V² (since PSD has units V²/Hz)

**Interpretation**: Represents the total oscillatory power within the specified frequency band, analogous to area under the PSD curve.

### 6.4 Spectral Flatness Computation

**Definition**: Ratio of geometric mean to arithmetic mean of PSD values within a frequency band

**Mathematical Formulation**:

```
Flatness = (∏[i=1 to n] PSD[i])^(1/n) / (1/n) * Σ PSD[i]
         = geometric mean / arithmetic mean
```

**Implementation**:
```python
flatness = gmean(band_psd) / np.mean(band_psd)
```

**Interpretation**:
- **Flatness ≈ 1.0**: Flat spectrum (white noise-like, no dominant frequencies)
- **Flatness ≈ 0.0**: Peaked spectrum (strong oscillatory activity at specific frequencies)
- **Values**: Typically ranges 0.1 - 0.5 for EEG showing prominent rhythmic activity

**Rationale**: Spectral flatness quantifies the "peakedness" of the spectrum, providing a complementary metric to band power. It is sensitive to the organization of spectral energy and is less influenced by absolute power differences.

**Note**: Flatness is undefined for PSD values ≤ 0. Such cases are assigned NaN (Not a Number).

### 6.5 Peak Alpha Frequency (PAF) Extraction

#### 6.5.1 Definition

**Peak Alpha Frequency (PAF)**: The frequency of maximum power within the alpha band (8-14 Hz)

**Functional Significance**: 
- PAF is an individual alpha frequency (IAF) trait marker
- Associated with cognitive performance, neural efficiency, and thalamocortical dynamics
- Shows task-dependent modulation and age-related changes

#### 6.5.2 Extraction Method: Parabolic Interpolation

**Rationale**: Simple peak finding on discrete frequency bins is limited by spectral resolution. Parabolic interpolation provides sub-bin frequency resolution.

**Algorithm**:

1. Identify discrete peak index within alpha band:
   ```
   peak_idx = argmax(PSD[f] for f in [8, 14))
   ```

2. If peak is not at band edge, perform parabolic interpolation using three points:
   - y₀ = PSD[peak_idx - 1]
   - y₁ = PSD[peak_idx]
   - y₂ = PSD[peak_idx + 1]

3. Compute fractional offset δ:
   ```
   δ = 0.5 * (y₀ - y₂) / (y₀ - 2*y₁ + y₂)
   ```

4. Final interpolated PAF:
   ```
   PAF = freqs[peak_idx] + δ * Δf
   ```

Where Δf is the frequency resolution (freqs[1] - freqs[0])

**Edge Cases**:
- If peak is at band edge (first or last frequency bin in alpha range), no interpolation is performed
- If denominator (y₀ - 2*y₁ + y₂) = 0, parabolic fit fails; discrete peak frequency is used

**Precision**: Interpolation typically provides frequency resolution of ~0.01 Hz, compared to discrete bin resolution of ~0.1-0.5 Hz

## 7. Data Aggregation and Output

### 7.1 Label-Level Features

**Output Format**: Long-format table with one row per participant-condition-label combination

**Columns**:
- `Participant`: Participant identifier
- `Condition`: Experimental condition (e.g., `EyesClosed_pre`, `EyesOpen_post`)
- `Label`: Anatomical label name (e.g., `superiorfrontal-lh`)
- `ROI_Group`: ROI group classification (`frontal`, `parietal`, `temporal`, `occipital`, `other`)
- `theta_power`: Integrated power in theta band (4-8 Hz)
- `alpha_power`: Integrated power in alpha band (8-14 Hz)
- `beta_power`: Integrated power in beta band (14-30 Hz)
- `theta_flatness`: Spectral flatness in theta band
- `alpha_flatness`: Spectral flatness in alpha band
- `beta_flatness`: Spectral flatness in beta band
- `paf`: Peak alpha frequency (Hz)

**Precision Formatting**:
- Power values: Scientific notation (e.g., 1.2345678e-05)
- Flatness values: 4 decimal places (e.g., 0.1234)
- PAF values: 2 decimal places (e.g., 10.23)

**Purpose**: Enables fine-grained analysis of regional spectral properties and examination of lateralized effects.

### 7.2 ROI-Group Aggregated Features

**Output Format**: Wide-format table with one row per participant

**Columns**:
- `Participant`: Participant identifier
- One column per condition-band-metric-roi combination:
  - Examples: `EyesClosed_pre_theta_frontal_power`, `EyesOpen_post_alpha_parietal_paf`
  - Four ROI groups: `frontal`, `parietal`, `temporal`, `occipital`
  - Three bands: `theta`, `alpha`, `beta`
  - Two metrics per band: `power`, `flatness`
  - One metric across bands: `paf` (peak alpha frequency)

**Aggregation Method**: Arithmetic mean across all labels belonging to the ROI group

**Implementation**:
```python
frontal_power = np.nanmean(features['power']['theta'][frontal_idx])
parietal_power = np.nanmean(features['power']['theta'][parietal_idx])
temporal_power = np.nanmean(features['power']['theta'][temporal_idx])
occipital_power = np.nanmean(features['power']['theta'][occipital_idx])
```

**Handling of Missing Data**: 
- `np.nanmean` excludes NaN values (labels with invalid features)
- If all values in a group are NaN, result is NaN

**Purpose**: Provides concise summary statistics suitable for:
- Statistical group comparisons
- Feature engineering for machine learning
- Integration with connectivity features at network level
- Direct comparison with PLI connectivity measures from the same four regions

### 7.3 File Naming and Version Control

**Timestamp Convention**: `YYYYMMDD_HHMMSS` format

**Output Files**:
- Summary: `roi_source_power_summary_four_regions_<timestamp>.csv`
- Labels: `roi_source_power_labels_four_regions_<timestamp>.csv`
- Error Log: `roi_source_power_errors_four_regions_<timestamp>.txt` (if errors occur)

**Resume Capability**: 
- The pipeline detects the most recent output file and continues from the last completed participant
- Participants with complete data across all conditions are skipped
- Incomplete or missing participants are processed

## 8. Post-Processing: Relative Power Computation

This section describes the post-processing analysis performed on the extracted source-space power features to derive **relative power metrics**. Relative power normalizes absolute power values within each participant and condition, transforming the analysis from "how much power does this person have?" to "how is this person's power distributed across bands and regions?" This approach addresses individual differences in absolute power that arise from anatomical factors such as skull thickness, cortical folding patterns, and electrode impedance.

### 8.1 Band-Region Relative Power

**Definition**: The proportion of total spectral power contributed by each specific band-region combination within a given condition-session recording.

**Computation**: For each participant and condition-session combination (e.g., `EyesClosed_pre`), the pipeline first computes a **grand total power** by summing all band-region power values:

```python
total_power = Σ (power for all bands in [theta, alpha, beta] and all regions in [frontal, parietal, temporal, occipital])
```

Then, each individual band-region power value is divided by this grand total:

```
relative_power(band, region) = band_region_power / total_power
```

**Output Values**: Twelve relative power values per condition-session (3 bands × 4 regions):
- `{condition}_{session}_theta_frontal_rel`
- `{condition}_{session}_theta_parietal_rel`
- `{condition}_{session}_theta_temporal_rel`
- `{condition}_{session}_theta_occipital_rel`
- `{condition}_{session}_alpha_frontal_rel`
- `{condition}_{session}_alpha_parietal_rel`
- `{condition}_{session}_alpha_temporal_rel`
- `{condition}_{session}_alpha_occipital_rel`
- `{condition}_{session}_beta_frontal_rel`
- `{condition}_{session}_beta_parietal_rel`
- `{condition}_{session}_beta_temporal_rel`
- `{condition}_{session}_beta_occipital_rel`

**Interpretation**: These values represent the percentage of total spectral power (4-30 Hz) originating from each specific band-region combination. By definition, these twelve values sum to 1.0 (or 100%) for each condition-session recording. For example, if `EyesClosed_pre_alpha_occipital_rel = 0.25`, this indicates that 25% of the participant's total spectral power in the eyes-closed pre-task condition comes from occipital alpha activity.

**Rationale**: Band-region relative power captures the full spectral distribution across both frequency dimensions (theta, alpha, beta) and spatial dimensions (frontal, parietal, temporal, occipital). This comprehensive view is essential for understanding the complete oscillatory profile while controlling for individual differences in absolute power magnitude.

### 8.2 Region Relative Power

**Definition**: The proportion of total spectral power contributed by each broad anatomical region, collapsed across all frequency bands.

**Computation**: For each condition-session, the pipeline first aggregates power within each region by summing across all frequency bands:

```python
frontal_total = Σ power(frontal, band) for band in [theta, alpha, beta]
parietal_total = Σ power(parietal, band) for band in [theta, alpha, beta]
temporal_total = Σ power(temporal, band) for band in [theta, alpha, beta]
occipital_total = Σ power(occipital, band) for band in [theta, alpha, beta]
grand_total = frontal_total + parietal_total + temporal_total + occipital_total
```

Then, regional relative power is computed as:

```
frontal_region_rel = frontal_total / grand_total
parietal_region_rel = parietal_total / grand_total
temporal_region_rel = temporal_total / grand_total
occipital_region_rel = occipital_total / grand_total
```

**Output Values**: Four relative power values per condition-session:
- `{condition}_{session}_frontal_region_rel`
- `{condition}_{session}_parietal_region_rel`
- `{condition}_{session}_temporal_region_rel`
- `{condition}_{session}_occipital_region_rel`

**Interpretation**: These values represent the band-agnostic distribution of oscillatory activity across the four major brain networks. By definition, these four values sum to 1.0 for each condition-session recording. For example, if `EyesClosed_pre_occipital_region_rel = 0.35`, this indicates that 35% of total oscillatory power (regardless of frequency band) originates from occipital regions.

**Rationale**: Region relative power provides a simplified, band-agnostic measure of the distribution of neural activity across the four major lobes. This metric is particularly useful when the research question focuses on broad network-level differences (e.g., "Is there a shift from posterior-dominant to frontal-dominant activity?") rather than frequency-specific modulations. It captures the overall spatial distribution of neural activity while remaining robust to individual differences in absolute power.

### 8.3 Advantages of Relative Power Analysis

**Normalization of Individual Differences**: Absolute EEG power varies substantially across individuals due to:
- Skull thickness and conductivity variations
- Cortical folding patterns and brain size
- Electrode placement and impedance differences
- Recording equipment and environment variations

By converting to relative power, these nuisance variables are removed, allowing comparison of functional brain states across participants.

**State-Dependent Distribution Analysis**: Relative power shifts the analytical focus from absolute magnitude to **power distribution patterns**. This is particularly valuable for examining:
- Changes in the theta/alpha/beta balance with aging or pathology
- Task-dependent shifts between frontal (executive) and posterior (sensory) dominance
- Individual differences in neural efficiency and oscillatory organization
- Regional specialization across the four lobes

**Statistical Properties**: Relative power values are bounded [0, 1] and naturally sum to 1.0 within each normalization unit (condition-session). This bounded nature makes them suitable for specific statistical approaches such as:
- Compositional data analysis (log-ratio transformations)
- Mixed-effects models with proportional outcomes
- Multi-level modeling with participant-level random effects

### 8.4 Integration with Statistical Analysis

The relative power features are typically exported as a cleaned dataset suitable for statistical analysis in software such as JASP or R. The export process includes:

1. **Column Selection**: Retaining only relative power columns (suffix `_rel`) and participant identifiers
2. **Participant ID Formatting**: Converting numeric participant IDs to BIDS-compliant format (e.g., `001` → `sub-001`)
3. **Demographic Merging**: Integrating with `participants.tsv` to include covariates such as age
4. **Age Group Stratification**: Creating categorical age groups (Young, Middle-aged, Old) for developmental or aging studies

**Typical Output File**: `ROI_SOURCE_SPACE_POWER_JASP.csv` - formatted for direct import into statistical software with one row per participant and relative power features as columns.

### 8.5 Parameter Summary: Relative Power

| Metric | Scope | Values per Condition | Sum Constraint | Primary Use Case |
|--------|-------|---------------------|----------------|------------------|
| Band-Region Relative Power | Band × Region | 12 (3 bands × 4 regions) | 1.0 (100%) | Full spectral-spatial distribution analysis |
| Region Relative Power | Region only | 4 (frontal, parietal, temporal, occipital) | 1.0 (100%) | Lobe-level distribution analysis |

## 9. Computational Implementation

### 9.1 Parallel Processing Architecture

**Strategy**: Participant-level parallelization

**Batch Size**: 10 participants per batch (`N_PARTICIPANTS_JOBS = 10`)

**Parallelization Framework**: `joblib.Parallel`

**Hierarchy**:
1. **Batch Level**: Participants processed in batches of 10
2. **Participant Level**: Each participant's conditions processed sequentially
3. **Condition Level**: Within each participant, all conditions processed

**Rationale**:
- Parallelization at participant level maximizes CPU utilization
- Batches prevent excessive memory consumption
- Sequential condition processing within participants allows inverse operator reuse

### 9.2 Memory Management

**Garbage Collection**: Explicit cleanup after each participant

```python
del epochs
del inverse_operator
gc.collect()
```

**Resource Sharing**: Forward solution, noise covariance, labels, and source space computed once globally

**Efficiency**: 
- Global resources computed once per batch (~15-30 seconds overhead)
- Inverse operator computed once per participant (~5-10 seconds overhead)
- Minimal memory duplication across parallel workers

### 9.3 Error Handling

**Exception Scope**: Individual participant-condition failures do not halt the pipeline

**Error Logging**: Comprehensive error information captured:
- Participant ID
- Condition
- File path
- Exception type
- Error message
- Full traceback

**Output**: Errors saved to timestamped log file for review and debugging

## 10. Parameter Justification and Recommendations

### 10.1 Source Modeling Parameters

| Parameter              | Value    | Justification                                                         |
| ---------------------- | -------- | --------------------------------------------------------------------- |
| Source Space           | ico-4    | Balance of resolution and efficiency; consistent with PLI pipeline    |
| Inverse Method         | eLORETA  | Zero localization error; improved spatial resolution over MNE/sLORETA |
| Orientation Constraint | `normal` | Physiologically justified; reduces solution space                     |
| SNR                    | 3.0      | Standard for EEG; provides appropriate regularization                 |
| Depth Weighting        | 0.8      | Compensates for superficial bias in EEG                               |

### 10.2 Spectral Analysis Parameters

| Parameter              | Value      | Justification                                          |
| ---------------------- | ---------- | ------------------------------------------------------ |
| Method                 | Multitaper | Optimal variance-resolution trade-off                  |
| Time-Bandwidth Product | 4.0        | Good balance for EEG spectral analysis                 |
| Frequency Range        | 4-30 Hz    | Covers theta, alpha, beta bands relevant to cognition  |
| Normalization          | `full`     | Absolute power (V²/Hz) for cross-subject comparability |

### 10.3 ROI Configuration Summary

| ROI Group  | Typical Label Count | Primary Functions                                  |
| ---------- | ------------------- | -------------------------------------------------- |
| Frontal    | 26 (13 × 2)         | Executive function, motor control, decision-making |
| Parietal   | 14 (7 × 2)          | Sensory integration, attention, default mode       |
| Temporal   | 18 (9 × 2)          | Auditory processing, memory, language              |
| Occipital  | 8 (4 × 2)           | Visual processing                                  |
| Other      | ~2-4                | Insula, unassigned regions                         |

**ROI Label Counts**: Verify appropriate distribution:
- Frontal: 26 labels (13 × 2 hemispheres, including anterior cingulate)
- Parietal: 14 labels (7 × 2 hemispheres, including posterior cingulate)
- Temporal: 18 labels (9 × 2 hemispheres)
- Occipital: 8 labels (4 × 2 hemispheres)
- Other: 2-4 labels (insula, remaining regions)

---

**Document Version**: 2.0  
**Last Updated**: 2026-02-04  
**Corresponding Script**: `roi_source_power_extraction_four_regions.py`  
**Compatible with**: MNE-Python ≥ 1.0
