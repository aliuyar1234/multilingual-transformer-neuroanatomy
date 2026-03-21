# Reproducing the Public Release

This repository accompanies the paper as a public research artifact. It supports code inspection, figure and table review, and lightweight validation of the released analysis surface.

## What you can do from this repo

- inspect the public implementation in `src/`
- inspect the public configuration in `conf/`
- inspect frozen manifests in `data/manifests/`
- inspect the generated figures in `outputs/figures/`
- inspect the generated tables in `outputs/tables/`
- run the public regression tests in `tests/`

## Minimal public validation

Typical lightweight checks:

1. Start with [README.md](README.md).
2. Inspect `outputs/tables/` and `outputs/figures/`.
3. Review the implementation in `src/`.
4. Run `pytest -q` from the repository root.

## What this public repo is not

This is not the full internal execution workspace used during development. It does not include:

- heavy mirrored prior-repo assets
- internal provenance bundles
- runtime and checkpoint logs
- subject-level shards and scratch outputs
- large feature caches
- development-only manuscript packaging materials

## Release status

At the time of this public release:

- all paper-facing figures `fig01` to `fig09` are present
- all paper-facing tables `table01` to `table06` are present
- the released outputs reflect the completed mechanistic evidence surface behind the paper

## Configuration note

`conf/paths.yaml` is sanitized for public release. It documents the expected project layout rather than the original workstation-specific paths used during development.
