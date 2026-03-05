# ROI-Based Electrode Power Extraction: Methodological Details

## 1. Overview

This document describes the methodology for extracting region-of-interest (ROI) based sensor-space power features from electroencephalographic (EEG) data.

**Please Note:** This pipeline follows the exact same extraction logic and methodological principles as the source-space extraction pipeline described in [ROI Source Methods](../Extraction%20Based%20On%20ROI%20Source/methods_roi.md).

The fundamental difference is that the spectral power features (band power, spectral flatness, and peak alpha frequency) are computed directly in **sensor space** (at the electrode level) rather than being reconstructed in **source space**.

Consequently, the ROI groupings (Frontal, Parietal, Temporal, Occipital) are defined based on predefined 64-channel EEG electrode groups mapped to those anatomical regions, rather than using the Desikan-Killiany cortical atlas labels. 

For full details on the spectral feature extraction methods (Multitaper PSD estimation, frequency bands, band power computation, spectral flatness, Peak Alpha Frequency, and relative power computation), please refer to the corresponding sections in the source space methodology document.
