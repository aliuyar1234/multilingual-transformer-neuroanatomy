# Data Access

This repository is built around LPPC / Le Petit Prince multilingual naturalistic fMRI and public-safe derived manifests for the mechanistic analyses in the paper.

## Included in this public release

- frozen lightweight manifests under `data/manifests/`
- paper-facing summary outputs under `outputs/tables/` and `outputs/figures/`
- sanitized configuration paths that document expected project structure without exposing workstation-specific locations

Included manifest types:

- canonical run manifest
- dataset and sample manifests
- ROI manifests and ROI metadata
- sentence-span manifests for `EN`, `FR`, and `ZH`
- multilingual triplet manifest

## Archived locally, not published here

- heavy raw or mirrored dataset assets
- large intermediate feature caches
- subject-level output shards
- runtime logs and provenance bundles
- mirrored prior-repo working trees
- development-only manuscript packaging materials

## Intended use

The public repo is designed for:

- paper review
- code inspection
- artifact inspection
- lightweight validation against the released figures, tables, and manifests

It is not a full raw-data mirror and not a complete internal compute workspace.

For reproduction scope, see [REPRODUCE.md](REPRODUCE.md).
