# Multilingual Transformer Neuroanatomy

Code and paper-facing artifacts for:

**Inside the Multilingual Transformer: Computational Neuroanatomy of Shared and Language-Specific Brain Alignment**

This repository studies how internal computations in multilingual transformers align with shared and language-specific brain responses during naturalistic story listening. It is a clean public research release: the visible surface is limited to code, lightweight manifests, tests, and paper-facing figures and tables.

## Overview

The central question is:

**Which internal transformer computations and token-level signals generate multilingual `SHARED > SPECIFIC` brain alignment, and why does that effect extend into auditory cortex?**

The analyses use LPPC / Le Petit Prince multilingual naturalistic fMRI, XLM-R-base, and the encoder of NLLB-200-distilled-600M. The public repo focuses on the mechanistic evidence surface behind the paper rather than the full internal working tree used during development.

## Main contributions

- state-level analysis of multilingual transformer blocks rather than final hidden states only
- leave-target-out `SHARED` versus orthogonalized `SPECIFIC` decomposition across English, French, and Chinese
- representative-state selection, token attribution, and deletion validation tied to paper-ready outputs
- public release of lightweight manifests, executable analysis code, and the full paper-facing figure and table set

## Key findings at a glance

- the originally planned semantic FFN-preference headline is not supported by the confirmatory primary table
- auditory attention-side preference is positive in most model-language cells and Holm-significant in `4 / 6`
- deletion validation is complete, and top-attributed deletion exceeds matched-random deletion in every public deletion-summary row

The current evidence therefore supports a stronger attention-side mechanistic account than the original FFN-centered semantic expectation.

## Repository layout

- `src/`: analysis code for manifests, features, encoding, statistics, attribution, and figure generation
- `conf/`: sanitized public configuration
- `tests/`: lightweight regression tests for the public code paths
- `data/manifests/`: frozen structural manifests for the sample, runs, ROIs, sentence spans, and multilingual triplets
- `outputs/tables/`: paper-facing CSV tables
- `outputs/figures/`: paper-facing PNG and PDF figures
- `templates/`: public-safe configuration templates

## Results and artifacts

Primary paper-facing assets:

- [table03_primary_confirmatory_stats.csv](outputs/tables/table03_primary_confirmatory_stats.csv)
- [table04_representative_roi_summaries.csv](outputs/tables/table04_representative_roi_summaries.csv)
- [table05_token_class_attribution.csv](outputs/tables/table05_token_class_attribution.csv)
- [table06_deletion_validation.csv](outputs/tables/table06_deletion_validation.csv)
- [fig03_state_depth_heatmaps.png](outputs/figures/fig03_state_depth_heatmaps.png)
- [fig04_primary_state_family_tests.png](outputs/figures/fig04_primary_state_family_tests.png)
- [fig05_representative_roi_curves.png](outputs/figures/fig05_representative_roi_curves.png)
- [fig07_token_class_mass.png](outputs/figures/fig07_token_class_mass.png)
- [fig08_deletion_validation.png](outputs/figures/fig08_deletion_validation.png)
- [fig09_roi_preference_map.png](outputs/figures/fig09_roi_preference_map.png)

## Data access

This repository includes only lightweight public manifests and summary outputs. For scope and boundaries, see [DATA_ACCESS.md](DATA_ACCESS.md).

## Reproducing the public release

This release is designed for artifact inspection, public-code review, and lightweight validation rather than full raw-data recomputation from scratch. See [REPRODUCE.md](REPRODUCE.md).

## What stays archived locally

The public repo intentionally excludes:

- internal status documents and operational notes
- GPT handoff prompts, review memos, and working commentary
- provenance bundles, checkpoint logs, and subject-level runtime shards
- heavy mirrored prior-repo assets and large intermediate caches
- local manuscript packaging experiments and other development-only materials

## Citation

If you use this repository, please cite the accompanying paper once the final manuscript is public. A machine-readable citation record is provided in [CITATION.cff](CITATION.cff).
