from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Sequence

import yaml

from src.encoding.run_sequel_fastpath import PRIMARY_STATES, run_sequel_fastpath
from src.utils.config import load_config


def _local_outputs_root(config: dict[str, Any]) -> Path:
    return Path(config["paths"]["local_outputs_root"]).resolve()


def _provenance_path(config: dict[str, Any]) -> Path:
    return _local_outputs_root(config) / "provenance" / "full_paper_pipeline.md"


def _paper_artifact_manifest_yaml_path(config: dict[str, Any]) -> Path:
    return _local_outputs_root(config) / "provenance" / "paper_artifact_manifest.yaml"


def _normalize_models(models: Sequence[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(str(model).lower() for model in models))


def _normalize_languages(languages: Sequence[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(str(language).upper() for language in languages))


def _normalize_states(states: Sequence[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(str(state).upper() for state in states))


def _run_refresh_job_manifest_stage(config: dict[str, Any]) -> dict[str, Path]:
    from src.encoding.build_refresh_job_manifest import build_refresh_job_manifest

    return build_refresh_job_manifest(config)


def _run_refresh_table_stage(config: dict[str, Any]) -> dict[str, Path]:
    from src.stats.generate_refresh_tables import generate_refresh_tables

    return generate_refresh_tables(config)


def _run_representatives_stage(config: dict[str, Any]) -> dict[str, Path]:
    from src.attribution.select_representative_states import select_representative_states

    return select_representative_states(config)


def _run_attribution_stage(config: dict[str, Any]) -> dict[str, Path]:
    from src.attribution.compute_word_attributions import compute_word_attributions

    return compute_word_attributions(config)


def _run_deletion_stage(config: dict[str, Any]) -> dict[str, Path]:
    from src.attribution.run_deletion_validation import run_deletion_validation

    return run_deletion_validation(config)


def _run_tables_stage(config: dict[str, Any]) -> dict[str, Path]:
    from src.stats.generate_all_tables import generate_all_tables

    return generate_all_tables(config)


def _run_figures_stage(config: dict[str, Any]) -> dict[str, Path]:
    from src.plots.generate_all_figures import generate_all_figures

    return generate_all_figures(config)


def _run_claim_map_stage(config: dict[str, Any]) -> Path:
    from src.manuscript.build_claim_evidence_map import build_claim_evidence_map

    return build_claim_evidence_map(config)


def _run_draft_stage(config: dict[str, Any]) -> dict[str, Path]:
    from src.manuscript.write_draft import write_draft

    return write_draft(config)


def _run_artifact_manifest_stage(config: dict[str, Any]) -> dict[str, Path]:
    from src.manuscript.build_paper_artifact_manifest import build_paper_artifact_manifest

    return build_paper_artifact_manifest(config)


def _assert_required_artifacts_complete(config: dict[str, Any]) -> None:
    manifest_path = _paper_artifact_manifest_yaml_path(config)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Paper artifact manifest missing after full pipeline run: {manifest_path}")
    with manifest_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not bool(payload.get("paper_ready", False)):
        raise RuntimeError(
            "Full paper pipeline completed but required artifacts are still missing. "
            f"See {manifest_path} for the exact missing required figures/claims."
        )


def _write_provenance(
    config: dict[str, Any],
    *,
    models: Sequence[str],
    languages: Sequence[str],
    states: Sequence[str],
    max_subjects: int | None,
    require_complete_artifacts: bool,
    stage_outputs: dict[str, Path],
) -> Path:
    provenance_path = _provenance_path(config)
    provenance_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Full Paper Pipeline",
        "",
        "- Purpose: run the explicit end-to-end paper artifact contract for the sequel pipeline.",
        "- Stage order: refresh job manifest -> sequel fastpath core -> refresh tables -> representative states -> attribution -> deletion -> tables -> figures -> claim map -> draft -> paper artifact manifest",
        f"- models: `{', '.join(models)}`",
        f"- languages: `{', '.join(languages)}`",
        f"- states: `{', '.join(states)}`",
        f"- max_subjects: `{max_subjects if max_subjects is not None else 'all included'}`",
        f"- require_complete_artifacts: `{require_complete_artifacts}`",
        "",
        "## Outputs",
        "",
    ]
    for name, path in stage_outputs.items():
        lines.append(f"- {name}: `{path}`")
    provenance_path.write_text("\n".join(lines), encoding="utf-8")
    return provenance_path


def run_full_paper_pipeline(
    config: dict[str, Any],
    *,
    config_path: str = "conf/base.yaml",
    models: Sequence[str] = ("xlmr", "nllb"),
    languages: Sequence[str] = ("EN", "FR", "ZH"),
    states: Sequence[str] = PRIMARY_STATES,
    max_subjects: int | None = None,
    require_complete_artifacts: bool = False,
) -> dict[str, Path]:
    normalized_models = _normalize_models(models)
    normalized_languages = _normalize_languages(languages)
    normalized_states = _normalize_states(states)

    outputs: dict[str, Path] = {}

    try:
        refresh_manifest_outputs = _run_refresh_job_manifest_stage(config)
        outputs.update({f"bridge_manifest_{name}": path for name, path in refresh_manifest_outputs.items()})
    except FileNotFoundError:
        pass

    fastpath_outputs = run_sequel_fastpath(
        config,
        config_path=config_path,
        models=normalized_models,
        languages=normalized_languages,
        states=normalized_states,
        max_subjects=max_subjects,
        include_downstream=False,
    )
    outputs.update({f"fastpath_{name}": path for name, path in fastpath_outputs.items()})

    refresh_tables = _run_refresh_table_stage(config)
    outputs.update({f"refresh_tables_{name}": path for name, path in refresh_tables.items()})

    representative_outputs = _run_representatives_stage(config)
    outputs.update({f"representatives_{name}": path for name, path in representative_outputs.items()})

    attribution_outputs = _run_attribution_stage(config)
    outputs.update({f"attribution_{name}": path for name, path in attribution_outputs.items()})

    deletion_outputs = _run_deletion_stage(config)
    outputs.update({f"deletion_{name}": path for name, path in deletion_outputs.items()})

    table_outputs = _run_tables_stage(config)
    outputs.update({f"tables_{name}": path for name, path in table_outputs.items()})

    figure_outputs = _run_figures_stage(config)
    outputs.update({f"figures_{name}": path for name, path in figure_outputs.items()})

    claim_map_path = _run_claim_map_stage(config)
    outputs["claim_evidence_map"] = claim_map_path

    draft_outputs = _run_draft_stage(config)
    outputs.update({f"draft_{name}": path for name, path in draft_outputs.items()})

    artifact_manifest_outputs = _run_artifact_manifest_stage(config)
    outputs.update({f"artifact_manifest_{name}": path for name, path in artifact_manifest_outputs.items()})

    if require_complete_artifacts:
        _assert_required_artifacts_complete(config)

    provenance_path = _write_provenance(
        config,
        models=normalized_models,
        languages=normalized_languages,
        states=normalized_states,
        max_subjects=max_subjects,
        require_complete_artifacts=require_complete_artifacts,
        stage_outputs=outputs,
    )
    outputs["provenance"] = provenance_path
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the explicit full-paper sequel artifact pipeline.")
    parser.add_argument("--config", default="conf/base.yaml", help="Path to sequel config.")
    parser.add_argument("--models", nargs="+", default=["xlmr", "nllb"], choices=["xlmr", "nllb"])
    parser.add_argument("--languages", nargs="+", default=["EN", "FR", "ZH"], choices=["EN", "FR", "ZH", "en", "fr", "zh"])
    parser.add_argument("--states", nargs="+", default=list(PRIMARY_STATES), choices=list(PRIMARY_STATES) + ["INPUT"])
    parser.add_argument("--max-subjects", type=int, default=None)
    parser.add_argument(
        "--require-complete-artifacts",
        action="store_true",
        help="Fail if any required figure or required claim is still missing after the pipeline finishes.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    outputs = run_full_paper_pipeline(
        config,
        config_path=args.config,
        models=tuple(args.models),
        languages=tuple(args.languages),
        states=tuple(args.states),
        max_subjects=args.max_subjects,
        require_complete_artifacts=args.require_complete_artifacts,
    )
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
