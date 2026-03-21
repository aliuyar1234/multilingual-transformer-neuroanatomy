from __future__ import annotations

import argparse
from pathlib import Path

from src.bridge.prior_runtime import prior_python, prior_repo_root, run_prior_repo_python
from src.utils.config import load_config


MODEL_TO_FUNCTION = {
    "xlmr": "run_xlmr_roi_pipeline",
    "nllb": "run_nllb_roi_pipeline",
}


def _prior_refresh_config(config: dict) -> dict:
    return dict(config.get("bridge", {}).get("prior_refresh", {}))


def _default_mismatch_shuffles(config: dict, *, mismatch_only: bool, layer_indices: tuple[int, ...] | None) -> int:
    refresh_cfg = _prior_refresh_config(config)
    if mismatch_only:
        return int(refresh_cfg.get("representative_mismatch_shuffles_default", 5))
    if layer_indices is not None:
        legacy_sparse = {4, 6, 8}
        if any(int(layer) not in legacy_sparse for layer in layer_indices):
            return int(refresh_cfg.get("missing_layer_mismatch_shuffles_default", 1))
    return int(refresh_cfg.get("dense_refresh_mismatch_shuffles_default", 20))


def _prior_subject_output(prior_repo_root: Path, model: str, output_tag: str) -> Path:
    stem = "nllb" if model == "nllb" else model
    return prior_repo_root / "outputs" / "stats" / f"{stem}_subject_level_roi_results__{output_tag}.parquet"


def _prior_group_output(prior_repo_root: Path, model: str, output_tag: str) -> Path:
    stem = "nllb" if model == "nllb" else model
    return prior_repo_root / "outputs" / "stats" / f"{stem}_group_level_roi_results__{output_tag}.parquet"


def _prior_confirmatory_output(prior_repo_root: Path, model: str, output_tag: str) -> Path:
    stem = "nllb" if model == "nllb" else model
    return prior_repo_root / "outputs" / "stats" / f"{stem}_confirmatory_effects__{output_tag}.parquet"


def _build_inline_code(
    *,
    prior_repo_root: Path,
    model: str,
    languages: tuple[str, ...],
    layer_indices: tuple[int, ...] | None,
    mismatch_shuffles: int,
    output_tag: str,
    max_subjects: int | None,
    subject_only: bool,
    resume: bool,
    mismatch_only: bool,
) -> str:
    function_name = MODEL_TO_FUNCTION[model]
    language_list = repr(tuple(languages))
    max_subjects_repr = "None" if max_subjects is None else str(int(max_subjects))
    layer_repr = "None" if layer_indices is None else repr(tuple(int(layer) for layer in layer_indices))
    return (
        "import sys\n"
        f"sys.path.insert(0, {repr(str((prior_repo_root / 'src').resolve()))})\n"
        "import brain_subspace_paper.encoding.xlmr_roi_pipeline as pipeline_module\n"
        f"pipeline_module.TEXT_CONDITIONS = {repr(()) if mismatch_only else 'pipeline_module.TEXT_CONDITIONS'}\n"
        f"summary = pipeline_module.{function_name}(\n"
        f"    languages={language_list},\n"
        f"    max_subjects={max_subjects_repr},\n"
        f"    layer_indices={layer_repr},\n"
        f"    mismatch_shuffles={int(mismatch_shuffles)},\n"
        f"    output_tag={repr(output_tag)},\n"
        f"    subject_only={subject_only},\n"
        f"    resume={resume},\n"
        ")\n"
        "print(summary)\n"
    )


def run_refresh_baseline(
    config: dict,
    *,
    models: tuple[str, ...],
    languages: tuple[str, ...],
    layer_indices: tuple[int, ...] | None,
    mismatch_shuffles: int,
    output_tag: str,
    max_subjects: int | None,
    subject_only: bool,
    resume: bool,
    mismatch_only: bool,
    timeout_ms: int | None = None,
) -> list[Path]:
    execution_root = prior_repo_root(config)
    python_exe = prior_python(execution_root, config)
    produced_paths: list[Path] = []

    for model in models:
        code = _build_inline_code(
            prior_repo_root=execution_root,
            model=model,
            languages=languages,
            layer_indices=layer_indices,
            mismatch_shuffles=mismatch_shuffles,
            output_tag=output_tag,
            max_subjects=max_subjects,
            subject_only=subject_only,
            resume=resume,
            mismatch_only=mismatch_only,
        )
        completed = run_prior_repo_python(
            code,
            prior_repo_root=execution_root,
            python_exe=python_exe,
            timeout_ms=timeout_ms,
        )
        print(completed.stdout.strip())
        if completed.stderr.strip():
            print(completed.stderr.strip())

        if subject_only:
            produced_paths.append(_prior_subject_output(execution_root, model, output_tag))
        else:
            produced_paths.extend(
                [
                    _prior_subject_output(execution_root, model, output_tag),
                    _prior_group_output(execution_root, model, output_tag),
                    _prior_confirmatory_output(execution_root, model, output_tag),
                ]
            )
    return produced_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run refreshed OUTPUT-state baseline jobs through the prior repo pipeline.")
    parser.add_argument("--config", default="conf/base.yaml", help="Path to sequel config.")
    parser.add_argument("--models", nargs="+", default=["xlmr", "nllb"], choices=["xlmr", "nllb"])
    parser.add_argument("--languages", nargs="+", default=["en", "fr", "zh"], choices=["en", "fr", "zh"])
    parser.add_argument("--layers", nargs="+", type=int, default=None, help="Optional explicit layer indices.")
    parser.add_argument("--mismatch-shuffles", type=int, default=None)
    parser.add_argument("--output-tag", default="sequel_refresh_output")
    parser.add_argument("--max-subjects", type=int, default=None)
    parser.add_argument("--subject-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--mismatch-only", action="store_true", help="Run only acoustic_only plus mismatched_shared.")
    parser.add_argument("--timeout-ms", type=int, default=None, help="Optional timeout for each model invocation.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    layer_indices = None if args.layers is None else tuple(args.layers)
    mismatch_shuffles = (
        int(args.mismatch_shuffles)
        if args.mismatch_shuffles is not None
        else _default_mismatch_shuffles(config, mismatch_only=args.mismatch_only, layer_indices=layer_indices)
    )
    produced = run_refresh_baseline(
        config,
        models=tuple(args.models),
        languages=tuple(args.languages),
        layer_indices=layer_indices,
        mismatch_shuffles=mismatch_shuffles,
        output_tag=args.output_tag,
        max_subjects=args.max_subjects,
        subject_only=args.subject_only,
        resume=args.resume,
        mismatch_only=args.mismatch_only,
        timeout_ms=args.timeout_ms,
    )
    for path in produced:
        print(f"produced_or_expected: {path}")


if __name__ == "__main__":
    main()
