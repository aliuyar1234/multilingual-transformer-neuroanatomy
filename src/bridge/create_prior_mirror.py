from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from src.utils.config import load_config


COPY_DIRS = (
    "src",
    "configs",
    "docs",
    "tasks",
    "scripts",
)

COPY_FILES = (
    "pyproject.toml",
    "README.md",
    "AGENTS.md",
    "ERRATA_AND_IMPLEMENTATION_DECISIONS.md",
    "METHOD_RATIONALE.md",
    "SSOT.md",
)

COPY_DATA_FILES = (
    "data/interim/lppc_run_manifest.parquet",
    "data/interim/sentence_spans_en.parquet",
    "data/interim/sentence_spans_fr.parquet",
    "data/interim/sentence_spans_zh.parquet",
    "data/interim/embeddings/embedding_manifest.parquet",
    "data/interim/roi/roi_metadata.parquet",
    "data/interim/roi/harvard_oxford_cortical_resampled_to_lppc_bold.nii.gz",
    "data/interim/roi/en_roi_target_manifest.parquet",
    "data/interim/roi/fr_roi_target_manifest.parquet",
    "data/interim/roi/zh_roi_target_manifest.parquet",
    "data/processed/alignment_triplets.parquet",
    "data/processed/alignment_triplets_qc.parquet",
    "data/processed/features/feature_manifest.parquet",
)

COPY_DATA_DIRS = (
    "data/raw/ds003643/annotation",
)


def _copy_file(src_root: Path, dst_root: Path, relative: str) -> None:
    src = src_root / relative
    dst = dst_root / relative
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _copy_dir(src_root: Path, dst_root: Path, relative: str) -> None:
    src = src_root / relative
    dst = dst_root / relative
    shutil.copytree(src, dst, dirs_exist_ok=True)


def create_prior_mirror(config: dict, *, mirror_root: Path | None = None) -> Path:
    src_root = Path(config["paths"]["prior_repo_root"]).resolve()
    dst_root = mirror_root or Path(config["paths"]["prior_mirror_root"]).resolve()
    dst_root.mkdir(parents=True, exist_ok=True)

    for relative in COPY_DIRS:
        _copy_dir(src_root, dst_root, relative)
    for relative in COPY_FILES:
        _copy_file(src_root, dst_root, relative)
    for relative in COPY_DATA_FILES:
        _copy_file(src_root, dst_root, relative)
    for relative in COPY_DATA_DIRS:
        _copy_dir(src_root, dst_root, relative)

    for relative in ("outputs/logs", "outputs/stats", "outputs/figures", "outputs/tables"):
        (dst_root / relative).mkdir(parents=True, exist_ok=True)

    return dst_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a lightweight local mirror of the prior repo.")
    parser.add_argument("--config", default="conf/base.yaml", help="Path to sequel config.")
    parser.add_argument("--mirror-root", default=None, help="Optional override path for the local mirror.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    mirror_root = None if args.mirror_root is None else Path(args.mirror_root).resolve()
    result = create_prior_mirror(config, mirror_root=mirror_root)
    print(f"Created/updated local prior mirror at {result}")


if __name__ == "__main__":
    main()
