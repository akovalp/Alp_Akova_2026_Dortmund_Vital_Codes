# Dortmund Vital Study Analysis

This repository contains the cleaned-up version of my initial Dortmund Vital study analysis. It includes:

- **A)** All extraction logic conducted so far (source power, spectral flatness, connectivity)
- **B)** All machine learning analyses performed on these extracted features

## Project Structure

This section provides descriptions of the main folders in this repository.

### Extraction Codes

For every extraction method, I have implemented the same (or similar) metrics in both sensor and source space. The rationale is to enable one-to-one comparisons of source-derived vs. electrode-derived features in machine learning pipelines.

#### Extraction Based on ROI (Source and Electrode)

This folder contains code for extracting source and electrode power and related metrics from four dedicated regions of interest (ROIs) in the sample. Each subfolder includes a `methods.md` file with a technical overview of the tools and analyses used.

#### Extraction Based on Connectivity

This folder stores wPLI (weighted Phase Lag Index) extracted connectivity metrics for both source and electrode space.

### Notebooks

I extensively use Jupyter notebooks for all analyses, normalization, and dataset construction. The notebooks directory contains subdirectories with descriptive names, each including documentation of what has been achieved to facilitate testing and further discussion.


