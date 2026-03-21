from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class InventorySpec:
    category: str
    name: str
    priority: int
    expected_paths: tuple[str, ...]
    sequel_action: str
    notes: str


@dataclass(frozen=True)
class InventoryRow:
    category: str
    name: str
    priority: int
    status: str
    found_paths: tuple[str, ...]
    missing_paths: tuple[str, ...]
    sequel_action: str
    notes: str


INVENTORY_SPECS: tuple[InventorySpec, ...] = (
    InventorySpec(
        category="manifest",
        name="canonical_run_manifest",
        priority=1,
        expected_paths=("data/interim/lppc_run_manifest.parquet",),
        sequel_action="Reuse as the baseline source for canonical run discovery and remapping; emit sequel copy under data/manifests/.",
        notes="Core LPPC raw-run to canonical-run mapping artifact.",
    ),
    InventorySpec(
        category="manifest",
        name="sentence_spans_en",
        priority=2,
        expected_paths=("data/interim/sentence_spans_en.parquet",),
        sequel_action="Reuse row schema and source text/timing behavior for English sentence spans.",
        notes="English sentence spans appear to be finalized.",
    ),
    InventorySpec(
        category="manifest",
        name="sentence_spans_fr",
        priority=2,
        expected_paths=("data/interim/sentence_spans_fr.parquet",),
        sequel_action="Reuse row schema and source text/timing behavior for French sentence spans.",
        notes="French sentence spans appear to be finalized.",
    ),
    InventorySpec(
        category="manifest",
        name="sentence_spans_zh",
        priority=2,
        expected_paths=("data/interim/sentence_spans_zh.parquet",),
        sequel_action="Reuse row schema and source text/timing behavior for Chinese sentence spans.",
        notes="Chinese sentence spans appear to be finalized.",
    ),
    InventorySpec(
        category="manifest",
        name="triplet_table",
        priority=3,
        expected_paths=(
            "data/processed/alignment_triplets.parquet",
            "data/processed/alignment_triplets_qc.parquet",
        ),
        sequel_action="Reuse as the primary multilingual triplet baseline unless a documented rebuild is required.",
        notes="Alignment table plus QC table for SHARED/SPECIFIC factorization.",
    ),
    InventorySpec(
        category="manifest",
        name="roi_artifacts",
        priority=4,
        expected_paths=(
            "data/interim/roi/harvard_oxford_cortical_resampled_to_lppc_bold.nii.gz",
            "data/interim/roi/roi_metadata.parquet",
            "data/interim/roi/en_roi_target_manifest.parquet",
            "data/interim/roi/fr_roi_target_manifest.parquet",
            "data/interim/roi/zh_roi_target_manifest.parquet",
        ),
        sequel_action="Reuse atlas resampling, ROI metadata, and per-language ROI target manifests to reconstruct the sequel ROI manifest/sample manifest.",
        notes="No single frozen sample manifest was found; per-language ROI target manifests appear to carry the inclusion surface.",
    ),
    InventorySpec(
        category="artifact",
        name="embedding_manifests",
        priority=5,
        expected_paths=(
            "data/interim/embeddings/embedding_manifest.parquet",
            "data/interim/embeddings/xlmr/triplet_ids.parquet",
            "data/interim/embeddings/nllb_encoder/triplet_ids.parquet",
        ),
        sequel_action="Reuse as bridge-level output-state extraction references only; sequel internal-state caches will need a new manifest layout.",
        notes="Prior repo caches returned hidden-state embeddings, not internal block states.",
    ),
    InventorySpec(
        category="code",
        name="lppc_discovery_module",
        priority=6,
        expected_paths=("src/brain_subspace_paper/data/inspect_lppc.py",),
        sequel_action="Port run discovery and manifest-building logic with minimal changes.",
        notes="Candidate implementation for LPPC file discovery and canonical run manifest construction.",
    ),
    InventorySpec(
        category="code",
        name="sentence_span_module",
        priority=7,
        expected_paths=("src/brain_subspace_paper/data/sentence_spans.py",),
        sequel_action="Port or wrap the sentence span construction code; keep reconstruction fallback aligned with this implementation.",
        notes="Candidate implementation for public annotation plus word-timing alignment.",
    ),
    InventorySpec(
        category="code",
        name="alignment_module",
        priority=8,
        expected_paths=("src/brain_subspace_paper/data/alignment.py",),
        sequel_action="Port triplet alignment and QC logic if the saved triplet table needs to be rebuilt.",
        notes="Contains monotonic pairwise DP, tri-lingual reconciliation, and QC/report generation.",
    ),
    InventorySpec(
        category="code",
        name="roi_target_module",
        priority=9,
        expected_paths=("src/brain_subspace_paper/roi/targets.py",),
        sequel_action="Port ROI atlas resampling and ROI-average time-series extraction logic.",
        notes="Candidate implementation for Harvard-Oxford resampling and ROI target construction.",
    ),
    InventorySpec(
        category="code",
        name="embedding_extraction_module",
        priority=10,
        expected_paths=("src/brain_subspace_paper/models/extraction.py",),
        sequel_action="Reuse tokenizer/model loading, batching, pooling, and manifest conventions as the OUTPUT-state bridge path; extend for internal-state hooks in sequel adapters.",
        notes="Prior extraction is for returned hidden states, not internal states.",
    ),
    InventorySpec(
        category="code",
        name="decomposition_module",
        priority=11,
        expected_paths=("src/brain_subspace_paper/features/decomposition.py",),
        sequel_action="Port SHARED/SPECIFIC/FULL construction and run-local mismatch generation, then refresh mismatch counts in the sequel.",
        notes="Important refresh target: prior repo includes MISMATCHED_SHARED infrastructure but sequel must raise mismatch shuffles for representative controls.",
    ),
    InventorySpec(
        category="code",
        name="encoding_module",
        priority=12,
        expected_paths=("src/brain_subspace_paper/encoding/xlmr_roi_pipeline.py",),
        sequel_action="Port the nested leave-one-run-out ROI encoding backbone and generalize from layer to block-state indexing.",
        notes="Critical backbone: train-only nuisance residualization, scaling, PCA, alpha selection, subject/group outputs, and confirmatory stats.",
    ),
    InventorySpec(
        category="code",
        name="paper_stats_module",
        priority=13,
        expected_paths=("src/brain_subspace_paper/stats/paper_level.py",),
        sequel_action="Reuse aggregation/reporting patterns selectively; sequel will need a new primary-test surface.",
        notes="Useful for bridge comparisons and downstream reporting conventions.",
    ),
    InventorySpec(
        category="code",
        name="figure_table_modules",
        priority=14,
        expected_paths=(
            "src/brain_subspace_paper/viz/figures.py",
            "src/brain_subspace_paper/viz/tables.py",
        ),
        sequel_action="Reuse publication output conventions and provenance style; do not reuse the old figure plan directly.",
        notes="Good source for layout, styling, and manuscript-facing provenance discipline.",
    ),
    InventorySpec(
        category="output",
        name="canonical_stats_outputs",
        priority=15,
        expected_paths=(
            "outputs/stats/subject_level_roi_results.parquet",
            "outputs/stats/group_level_roi_results.parquet",
            "outputs/stats/roi_condition_stats.parquet",
            "outputs/stats/roi_family_effect_panels.parquet",
        ),
        sequel_action="Use for bridge comparisons and output-schema continuity checks.",
        notes="Prior-paper released subject/group-level ROI stats are present.",
    ),
    InventorySpec(
        category="output",
        name="canonical_tables_figures",
        priority=16,
        expected_paths=(
            "outputs/tables/table01_dataset_summary.csv",
            "outputs/tables/table03_main_confirmatory_stats.csv",
            "outputs/figures/fig04_main_confirmatory_roi_families.png",
            "outputs/manuscript/claim_evidence_map.md",
        ),
        sequel_action="Use as manuscript-facing bridge artifacts and provenance examples.",
        notes="Useful for comparing sequel bridge outputs to the prior paper release surface.",
    ),
)


def _inventory_rows(prior_repo: Path) -> list[InventoryRow]:
    rows: list[InventoryRow] = []
    for spec in INVENTORY_SPECS:
        found_paths: list[str] = []
        missing_paths: list[str] = []
        for rel_path in spec.expected_paths:
            full_path = prior_repo / rel_path
            if full_path.exists():
                found_paths.append(rel_path)
            else:
                missing_paths.append(rel_path)
        if found_paths and not missing_paths:
            status = "found"
        elif found_paths:
            status = "partial"
        else:
            status = "missing"
        rows.append(
            InventoryRow(
                category=spec.category,
                name=spec.name,
                priority=spec.priority,
                status=status,
                found_paths=tuple(found_paths),
                missing_paths=tuple(missing_paths),
                sequel_action=spec.sequel_action,
                notes=spec.notes,
            )
        )
    return sorted(rows, key=lambda row: (row.priority, row.category, row.name))


def _write_csv(rows: Iterable[InventoryRow], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "priority",
                "category",
                "name",
                "status",
                "found_paths",
                "missing_paths",
                "sequel_action",
                "notes",
            ),
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "priority": row.priority,
                    "category": row.category,
                    "name": row.name,
                    "status": row.status,
                    "found_paths": " | ".join(row.found_paths),
                    "missing_paths": " | ".join(row.missing_paths),
                    "sequel_action": row.sequel_action,
                    "notes": row.notes,
                }
            )


def _write_markdown(rows: list[InventoryRow], prior_repo: Path, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    found_count = sum(row.status == "found" for row in rows)
    partial_count = sum(row.status == "partial" for row in rows)
    missing_count = sum(row.status == "missing" for row in rows)
    lines: list[str] = [
        "# Prior Repo Inventory",
        "",
        f"- Prior repo root: `{prior_repo}`",
        f"- Inventory date: `2026-03-17`",
        f"- Specs checked: `{len(rows)}`",
        f"- Found: `{found_count}`",
        f"- Partial: `{partial_count}`",
        f"- Missing: `{missing_count}`",
        "",
        "## Key salvage conclusions",
        "",
        "- The prior repo contains strong candidate implementations for canonical run discovery, sentence spans, multilingual triplet alignment, ROI target construction, SHARED/SPECIFIC decomposition, nested ROI encoding, and publication figures/tables.",
        "- The prior repo appears to expose per-language ROI target manifests rather than one single frozen sample manifest file; the sequel should reconstruct `data/manifests/sample_manifest.parquet` from the ROI target manifests plus the canonical run manifest.",
        "- The prior repo contains released subject/group result tables and manuscript-facing provenance outputs that should anchor the sequel bridge figure and result-schema compatibility checks.",
        "- Known refresh targets remain explicit: sparse NLLB layer coverage and the lightweight mismatch checkpoint behavior must not be carried forward unchanged.",
        "",
        "## Inventory table",
        "",
        "| Priority | Category | Name | Status | Found paths | Missing paths |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        found = "<br>".join(row.found_paths) if row.found_paths else "-"
        missing = "<br>".join(row.missing_paths) if row.missing_paths else "-"
        lines.append(
            f"| {row.priority} | {row.category} | {row.name} | {row.status} | {found} | {missing} |"
        )
    lines.extend(
        [
            "",
            "## Recommended sequel reuse map",
            "",
        ]
    )
    for row in rows:
        lines.extend(
            [
                f"### {row.name}",
                "",
                f"- Status: `{row.status}`",
                f"- Action: {row.sequel_action}",
                f"- Notes: {row.notes}",
                "",
            ]
        )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def build_inventory(prior_repo: Path, inventory_md: Path, module_map_csv: Path) -> tuple[Path, Path]:
    rows = _inventory_rows(prior_repo)
    _write_markdown(rows, prior_repo, inventory_md)
    _write_csv(rows, module_map_csv)
    return inventory_md, module_map_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect the prior multilingual SHARED/SPECIFIC repo.")
    parser.add_argument(
        "--prior-repo",
        type=Path,
        required=True,
        help="Path to the prior repo root.",
    )
    parser.add_argument(
        "--inventory-md",
        type=Path,
        default=Path("outputs/provenance/prior_repo_inventory.md"),
        help="Markdown inventory output path.",
    )
    parser.add_argument(
        "--module-map-csv",
        type=Path,
        default=Path("outputs/provenance/prior_repo_module_map.csv"),
        help="CSV module map output path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prior_repo = args.prior_repo.resolve()
    if not prior_repo.exists():
        raise FileNotFoundError(f"Prior repo does not exist: {prior_repo}")
    inventory_md, module_map_csv = build_inventory(
        prior_repo=prior_repo,
        inventory_md=args.inventory_md,
        module_map_csv=args.module_map_csv,
    )
    print(f"Wrote inventory markdown to {inventory_md}")
    print(f"Wrote module map CSV to {module_map_csv}")


if __name__ == "__main__":
    main()
