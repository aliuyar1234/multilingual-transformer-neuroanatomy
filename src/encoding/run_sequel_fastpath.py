from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Sequence

from src.utils.config import load_config


PRIMARY_STATES = ("ATTN", "POST_ATTN", "FFN", "OUTPUT")
PRIMARY_CONDITIONS = ("SHARED", "SPECIFIC")


def _local_outputs_root(config: dict[str, Any]) -> Path:
    return Path(config["paths"]["local_outputs_root"]).resolve()


def _provenance_path(config: dict[str, Any]) -> Path:
    return _local_outputs_root(config) / "provenance" / "sequel_fastpath.md"


def _normalize_models(models: Sequence[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(str(model).lower() for model in models))


def _normalize_languages(languages: Sequence[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(str(language).upper() for language in languages))


def _normalize_states(states: Sequence[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(str(state).upper() for state in states))


def _parallel_stage_config(config: dict[str, Any], stage: str) -> dict[str, Any]:
    return dict(config.get("parallel", {}).get(stage, {}))


def _run_extract_stage(
    config: dict[str, Any],
    *,
    config_path: str,
    models: Sequence[str],
    languages: Sequence[str],
    states: Sequence[str],
) -> dict[str, Path]:
    parallel_cfg = _parallel_stage_config(config, "extraction")
    use_parallel = bool(parallel_cfg.get("enabled", True)) and (len(models) * len(languages) > 1)
    if use_parallel:
        from src.features.run_parallel_internal_state_extraction import run_parallel_internal_state_extraction

        return run_parallel_internal_state_extraction(
            config,
            config_path=config_path,
            models=tuple(models),
            languages=tuple(languages),
            states=tuple(states),
            max_parallel=int(parallel_cfg.get("max_parallel", 3)),
            max_tokens_per_batch=int(parallel_cfg.get("max_tokens_per_batch", 2048)),
            cache_dtype=str(parallel_cfg["cache_dtype"]) if "cache_dtype" in parallel_cfg else None,
        )

    # Warm-import torch before the extractor module so the CUDA DLL stack is initialized
    # in a predictable order on Windows.
    import torch  # noqa: F401

    from src.features.extract_internal_states import extract_internal_states

    return extract_internal_states(
        config,
        models=tuple(models),
        languages=tuple(languages),
        states=tuple(states),
    )


def _run_build_shared_specific_stage(
    config: dict[str, Any],
    *,
    states: Sequence[str],
) -> dict[str, Path]:
    from src.features.build_shared_specific import build_shared_specific_caches

    return build_shared_specific_caches(
        config,
        states=tuple(states),
        include_full=False,
        include_raw_output=True,
        include_raw_all_states=False,
    )


def _run_state_sweep_stage(
    config: dict[str, Any],
    *,
    config_path: str,
    models: Sequence[str],
    languages: Sequence[str],
    states: Sequence[str],
    max_subjects: int | None,
) -> dict[str, Path]:
    parallel_cfg = _parallel_stage_config(config, "sweep")
    use_parallel = bool(parallel_cfg.get("enabled", True)) and (len(languages) > 1)
    if use_parallel:
        from src.encoding.run_parallel_state_sweep import run_parallel_state_sweep

        return run_parallel_state_sweep(
            config,
            config_path=config_path,
            models=tuple(models),
            languages=tuple(languages),
            states=tuple(states),
            conditions=PRIMARY_CONDITIONS,
            max_subjects=max_subjects,
            max_parallel=int(parallel_cfg.get("max_parallel", 3)),
        )

    from src.encoding.run_state_sweep import run_state_sweep

    return run_state_sweep(
        config,
        models=tuple(models),
        languages=tuple(languages),
        states=tuple(states),
        conditions=PRIMARY_CONDITIONS,
        max_subjects=max_subjects,
    )


def _run_primary_tests_stage(
    config: dict[str, Any],
    *,
    models: Sequence[str],
    languages: Sequence[str],
) -> dict[str, Path]:
    from src.stats.run_primary_tests import run_primary_tests

    return run_primary_tests(
        config,
        expected_models=tuple(models),
        expected_languages=tuple(languages),
    )


def _run_downstream_stage(config: dict[str, Any]) -> dict[str, Path]:
    from src.attribution.compute_word_attributions import compute_word_attributions
    from src.attribution.run_deletion_validation import run_deletion_validation
    from src.attribution.select_representative_states import select_representative_states
    from src.manuscript.build_claim_evidence_map import build_claim_evidence_map
    from src.manuscript.write_draft import write_draft
    from src.plots.generate_all_figures import generate_all_figures
    from src.stats.generate_all_tables import generate_all_tables

    outputs: dict[str, Path] = {}
    representative_outputs = select_representative_states(config)
    outputs.update({f"representatives_{name}": path for name, path in representative_outputs.items()})
    attribution_outputs = compute_word_attributions(config)
    outputs.update({f"attribution_{name}": path for name, path in attribution_outputs.items()})
    deletion_outputs = run_deletion_validation(config)
    outputs.update({f"deletion_{name}": path for name, path in deletion_outputs.items()})
    table_outputs = generate_all_tables(config)
    outputs.update({f"tables_{name}": path for name, path in table_outputs.items()})
    claim_map_path = build_claim_evidence_map(config)
    outputs["claim_evidence_map"] = claim_map_path
    figure_outputs = generate_all_figures(config)
    outputs.update({f"figures_{name}": path for name, path in figure_outputs.items()})
    draft_outputs = write_draft(config)
    outputs.update({f"draft_{name}": path for name, path in draft_outputs.items()})
    return outputs


def _write_provenance(
    config: dict[str, Any],
    *,
    models: Sequence[str],
    languages: Sequence[str],
    states: Sequence[str],
    max_subjects: int | None,
    include_downstream: bool,
    stage_outputs: dict[str, Path],
) -> Path:
    provenance_path = _provenance_path(config)
    provenance_path.parent.mkdir(parents=True, exist_ok=True)
    extraction_parallel = _parallel_stage_config(config, "extraction")
    sweep_parallel = _parallel_stage_config(config, "sweep")
    lines = [
        "# Sequel Fastpath",
        "",
        "- Purpose: run the sequel-native mechanistic path without treating the OUTPUT bridge as a standalone long-running campaign.",
        "- Stage order: extract pooled states -> build SHARED/SPECIFIC caches -> run state sweep -> run primary tests"
        + (" -> downstream attribution/deletion/tables/figures/manuscript" if include_downstream else ""),
        f"- models: `{', '.join(models)}`",
        f"- languages: `{', '.join(languages)}`",
        f"- states: `{', '.join(states)}`",
        f"- conditions: `{', '.join(PRIMARY_CONDITIONS)}`",
        f"- max_subjects: `{max_subjects if max_subjects is not None else 'all included'}`",
        f"- include_downstream: `{include_downstream}`",
        f"- extraction_parallel: `enabled={bool(extraction_parallel.get('enabled', True))}, max_parallel={int(extraction_parallel.get('max_parallel', max(1, len(models) * len(languages))))}, cache_dtype={extraction_parallel.get('cache_dtype', config.get('features', {}).get('cache_dtype', 'float32'))}`",
        f"- sweep_parallel: `enabled={bool(sweep_parallel.get('enabled', True))}, max_parallel={int(sweep_parallel.get('max_parallel', max(1, len(languages))))}`",
        "",
        "## Outputs",
        "",
    ]
    for name, path in stage_outputs.items():
        lines.append(f"- {name}: `{path}`")
    provenance_path.write_text("\n".join(lines), encoding="utf-8")
    return provenance_path


def run_sequel_fastpath(
    config: dict[str, Any],
    *,
    config_path: str = "conf/base.yaml",
    models: Sequence[str] = ("xlmr", "nllb"),
    languages: Sequence[str] = ("EN", "FR", "ZH"),
    states: Sequence[str] = PRIMARY_STATES,
    max_subjects: int | None = None,
    include_downstream: bool = False,
) -> dict[str, Path]:
    normalized_models = _normalize_models(models)
    normalized_languages = _normalize_languages(languages)
    normalized_states = _normalize_states(states)

    outputs: dict[str, Path] = {}
    extraction_outputs = _run_extract_stage(
        config,
        config_path=config_path,
        models=normalized_models,
        languages=normalized_languages,
        states=normalized_states,
    )
    outputs.update({f"extract_{name}": path for name, path in extraction_outputs.items()})

    feature_outputs = _run_build_shared_specific_stage(
        config,
        states=normalized_states,
    )
    outputs.update({f"features_{name}": path for name, path in feature_outputs.items()})

    sweep_outputs = _run_state_sweep_stage(
        config,
        config_path=config_path,
        models=normalized_models,
        languages=normalized_languages,
        states=normalized_states,
        max_subjects=max_subjects,
    )
    outputs.update({f"sweep_{name}": path for name, path in sweep_outputs.items()})

    primary_outputs = _run_primary_tests_stage(
        config,
        models=normalized_models,
        languages=normalized_languages,
    )
    outputs.update({f"primary_{name}": path for name, path in primary_outputs.items()})

    if include_downstream:
        outputs.update(_run_downstream_stage(config))

    provenance_path = _write_provenance(
        config,
        models=normalized_models,
        languages=normalized_languages,
        states=normalized_states,
        max_subjects=max_subjects,
        include_downstream=include_downstream,
        stage_outputs=outputs,
    )
    outputs["provenance"] = provenance_path
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the sequel-native fastpath: pooled-state extraction, SHARED/SPECIFIC build, state sweep, and primary tests."
    )
    parser.add_argument("--config", default="conf/base.yaml", help="Path to sequel config.")
    parser.add_argument("--models", nargs="+", default=["xlmr", "nllb"], choices=["xlmr", "nllb"])
    parser.add_argument("--languages", nargs="+", default=["EN", "FR", "ZH"], choices=["EN", "FR", "ZH", "en", "fr", "zh"])
    parser.add_argument("--states", nargs="+", default=list(PRIMARY_STATES), choices=list(PRIMARY_STATES) + ["INPUT"])
    parser.add_argument("--max-subjects", type=int, default=None)
    parser.add_argument(
        "--include-downstream",
        action="store_true",
        help="Also run representative selection, attribution, deletion, tables, figures, claim map, and draft writing.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    outputs = run_sequel_fastpath(
        config,
        models=tuple(args.models),
        languages=tuple(args.languages),
        states=tuple(args.states),
        max_subjects=args.max_subjects,
        include_downstream=args.include_downstream,
        config_path=args.config,
    )
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
