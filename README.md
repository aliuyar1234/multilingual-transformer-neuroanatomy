# Multilingual Transformer Neuroanatomy

[![DOI](https://zenodo.org/badge/1187629245.svg)](https://doi.org/10.5281/zenodo.19185601)
[![Paper PDF](https://img.shields.io/badge/Paper-PDF-B31B1B?style=flat&logo=adobeacrobatreader&logoColor=white)](https://github.com/aliuyar1234/multilingual-transformer-neuroanatomy/raw/main/paper/attention-side-transformer-brain-alignment.pdf)
[![Reproduce](https://img.shields.io/badge/Reproduce-Guide-0A7F5A?style=flat)](REPRODUCE.md)
[![Data Access](https://img.shields.io/badge/Data-Access-005F8F?style=flat)](DATA_ACCESS.md)

Code, paper-facing artifacts, and public release materials for:

**Inside the Multilingual Transformer: Computational Neuroanatomy of Shared and Language-Specific Brain Alignment**

This repository studies how internal computations in multilingual transformers align with shared and language-specific brain responses during naturalistic story listening. The project is framed as a mechanistic multilingual neuroscience study rather than a model-benchmark comparison: the goal is to identify which internal transformer computations and token-level signals help explain multilingual `SHARED > SPECIFIC` brain alignment, and why that effect extends into auditory cortex.

The released public surface is intentionally compact. It provides the executable analysis code, lightweight public manifests, tests, paper-facing figures and tables, and the final paper PDF without exposing the full internal development workspace.

## At a Glance

- **Scientific scope:** computational neuroanatomy of multilingual transformer internals under naturalistic fMRI
- **Dataset:** LPPC / Le Petit Prince multilingual naturalistic fMRI
- **Languages:** English, French, Chinese
- **Model families:** XLM-R-base and the encoder of NLLB-200-distilled-600M
- **Main analytical lens:** internal-state, token-attribution, and deletion-validation analysis of `SHARED` versus `SPECIFIC` alignment

## Main Paper

- [Download the paper PDF](https://github.com/aliuyar1234/multilingual-transformer-neuroanatomy/raw/main/paper/attention-side-transformer-brain-alignment.pdf)
- [Machine-readable citation](CITATION.cff)

## Central Question

The core question behind this release is:

**Which internal transformer computations and token-level signals generate multilingual `SHARED > SPECIFIC` brain alignment, and what explains the extension of that effect into auditory cortex?**

Rather than treating multilingual brain alignment as a single scalar outcome, this repository analyzes internal model states, representative-state structure, token-class attribution, and deletion-based validation to identify which computational families contribute to the observed brain-side effects.

## Main Contributions

- state-level analysis of multilingual transformer blocks rather than final hidden states alone
- leave-target-out `SHARED` versus orthogonalized `SPECIFIC` decomposition across English, French, and Chinese
- representative-state selection tied directly to paper-facing ROI summaries
- token attribution analyses for multilingual alignment-relevant signal mass
- deletion validation that tests whether high-attribution tokens matter more than matched random controls
- public release of lightweight manifests, executable analysis code, tests, and the full figure/table surface behind the paper

## Key Findings at a Glance

- the originally planned semantic FFN-preference headline is not supported by the confirmatory primary table
- auditory attention-side preference is positive in most model-language cells and Holm-significant in `4 / 6`
- deletion validation is complete, and top-attributed deletion exceeds matched-random deletion in every released deletion-summary row
- the public evidence surface therefore supports a stronger attention-side mechanistic account than the original FFN-centered semantic expectation

## Repository Layout

- `src/`: analysis code for manifests, features, encoding, statistics, attribution, and figure generation
- `conf/`: sanitized public configuration for the released analysis surface
- `tests/`: lightweight regression tests for public code paths
- `data/manifests/`: frozen structural manifests for the sample, runs, ROIs, sentence spans, and multilingual triplets
- `outputs/tables/`: paper-facing CSV tables
- `outputs/figures/`: paper-facing PNG and PDF figures
- `paper/`: public paper PDF
- `templates/`: public-safe configuration templates
- `DATA_ACCESS.md`: scope and boundary document for included versus archived assets
- `REPRODUCE.md`: reproduction and lightweight validation guidance

## Results and Artifacts

Primary paper-facing assets include:

- [paper/attention-side-transformer-brain-alignment.pdf](paper/attention-side-transformer-brain-alignment.pdf)
- [outputs/tables/table03_primary_confirmatory_stats.csv](outputs/tables/table03_primary_confirmatory_stats.csv)
- [outputs/tables/table04_representative_roi_summaries.csv](outputs/tables/table04_representative_roi_summaries.csv)
- [outputs/tables/table05_token_class_attribution.csv](outputs/tables/table05_token_class_attribution.csv)
- [outputs/tables/table06_deletion_validation.csv](outputs/tables/table06_deletion_validation.csv)
- [outputs/figures/fig03_state_depth_heatmaps.png](outputs/figures/fig03_state_depth_heatmaps.png)
- [outputs/figures/fig04_primary_state_family_tests.png](outputs/figures/fig04_primary_state_family_tests.png)
- [outputs/figures/fig05_representative_roi_curves.png](outputs/figures/fig05_representative_roi_curves.png)
- [outputs/figures/fig07_token_class_mass.png](outputs/figures/fig07_token_class_mass.png)
- [outputs/figures/fig08_deletion_validation.png](outputs/figures/fig08_deletion_validation.png)
- [outputs/figures/fig09_roi_preference_map.png](outputs/figures/fig09_roi_preference_map.png)

## Reading Order

If you want the fastest route through the release, start with:

1. [paper/attention-side-transformer-brain-alignment.pdf](paper/attention-side-transformer-brain-alignment.pdf)
2. [outputs/tables/table03_primary_confirmatory_stats.csv](outputs/tables/table03_primary_confirmatory_stats.csv)
3. [outputs/tables/table06_deletion_validation.csv](outputs/tables/table06_deletion_validation.csv)
4. [outputs/figures/fig04_primary_state_family_tests.png](outputs/figures/fig04_primary_state_family_tests.png)
5. [outputs/figures/fig08_deletion_validation.png](outputs/figures/fig08_deletion_validation.png)
6. [REPRODUCE.md](REPRODUCE.md)
7. [DATA_ACCESS.md](DATA_ACCESS.md)

## Reproduction Scope

This repository is designed for artifact inspection, public-code review, and lightweight validation of the released analysis surface. It supports:

- inspection of the implementation in `src/`
- inspection of the public configuration in `conf/`
- inspection of frozen manifests in `data/manifests/`
- inspection of released figures and tables in `outputs/`
- lightweight regression testing via `pytest -q`

It is not a full mirror of the private development workspace or a complete raw-data execution environment.

## Release Boundaries

The public repository intentionally excludes:

- heavy raw or mirrored dataset assets
- large intermediate feature caches
- subject-level runtime shards
- checkpoint logs and provenance bundles
- development-only manuscript packaging materials
- internal operational notes, review memos, and handoff commentary

For exact scope boundaries, see [DATA_ACCESS.md](DATA_ACCESS.md). For the intended public validation surface, see [REPRODUCE.md](REPRODUCE.md).

## Citation

If you use this repository, please cite the accompanying paper and the Zenodo archive linked above. A machine-readable citation record is available in [CITATION.cff](CITATION.cff).
