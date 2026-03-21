from __future__ import annotations

import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src.encoding.advance_refresh_checkpoint import _count_subjects, _subject_output_path
from src.encoding.run_refresh_baseline import _default_mismatch_shuffles
from src.utils.config import load_config


@dataclass(frozen=True)
class RefreshLane:
    language: str
    output_tag: str
    target_subjects: int


def _sample_manifest_path(config: dict[str, Any]) -> Path:
    manifests_root = Path(config["paths"]["local_manifests_root"]).resolve()
    return manifests_root / "sample_manifest.parquet"


def _load_language_targets(config: dict[str, Any], *, languages: tuple[str, ...]) -> dict[str, int]:
    manifest_path = _sample_manifest_path(config)
    sample_df = pd.read_parquet(manifest_path, columns=["language", "subject_id", "included"]).copy()
    included = sample_df.loc[sample_df["included"].astype(bool)].copy()
    counts = included.groupby("language", sort=False)["subject_id"].nunique().astype(int).to_dict()
    targets = {language: int(counts[language.upper()]) for language in languages}
    missing = [language for language, count in targets.items() if count <= 0]
    if missing:
        raise RuntimeError(f"Missing included-subject targets for languages: {missing!r}")
    return targets


def _build_lanes(
    config: dict[str, Any],
    *,
    languages: tuple[str, ...],
    output_tag_template: str,
) -> tuple[RefreshLane, ...]:
    targets = _load_language_targets(config, languages=languages)
    return tuple(
        RefreshLane(
            language=language,
            output_tag=output_tag_template.format(language=language),
            target_subjects=targets[language],
        )
        for language in languages
    )


def _lane_subjects_completed(config: dict[str, Any], *, model: str, lane: RefreshLane) -> int:
    return _count_subjects(_subject_output_path(config, model=model, output_tag=lane.output_tag))


def _run_lane_once(
    *,
    config_path: str,
    lane: RefreshLane,
    model: str,
    layers: tuple[int, ...],
    mismatch_shuffles: int,
    pass_timeout_ms: int,
    passes_per_round: int,
    mismatch_only: bool,
) -> dict[str, Any]:
    command = [
        sys.executable,
        "-m",
        "src.encoding.advance_refresh_checkpoint",
        "--config",
        config_path,
        "--model",
        model,
        "--language",
        lane.language,
        "--output-tag",
        lane.output_tag,
        "--pass-timeout-ms",
        str(int(pass_timeout_ms)),
        "--passes",
        str(int(passes_per_round)),
        "--mismatch-shuffles",
        str(int(mismatch_shuffles)),
        "--layers",
        *[str(int(layer)) for layer in layers],
    ]
    if mismatch_only:
        command.append("--mismatch-only")
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    stdout = completed.stdout.strip()
    if not stdout:
        raise RuntimeError(f"Lane {lane.language!r} returned no JSON summary.")
    return json.loads(stdout)


def _synthesize_lane_partial(
    *,
    config_path: str,
    lane: RefreshLane,
    model: str,
    completed_subjects: int,
) -> dict[str, str]:
    synthesis_tag = f"sequel_refresh_output_{model}_dense_partial_{lane.language}_subject{completed_subjects}"
    command = [
        sys.executable,
        "-m",
        "src.encoding.synthesize_refresh_baseline",
        "--config",
        config_path,
        "--model",
        model,
        "--chunk-glob",
        f"{model}_subject_level_roi_results__{lane.output_tag}.parquet",
        "--output-tag",
        synthesis_tag,
    ]
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    outputs: dict[str, str] = {}
    for line in completed.stdout.splitlines():
        if ":" not in line:
            continue
        name, value = line.split(":", 1)
        outputs[name.strip()] = value.strip()
    outputs["output_tag"] = synthesis_tag
    return outputs


def run_parallel_refresh_checkpoint(
    config: dict[str, Any],
    *,
    config_path: str,
    model: str,
    languages: tuple[str, ...],
    layers: tuple[int, ...],
    output_tag_template: str,
    max_parallel: int,
    mismatch_shuffles: int,
    pass_timeout_ms: int,
    passes_per_round: int,
    max_rounds: int,
    stop_after_no_progress_rounds: int,
    mismatch_only: bool = False,
    synthesize_partials: bool = False,
) -> dict[str, Any]:
    lanes = _build_lanes(config, languages=languages, output_tag_template=output_tag_template)
    summary: dict[str, Any] = {
        "model": model,
        "languages": list(languages),
        "layers": list(layers),
        "mismatch_shuffles": int(mismatch_shuffles),
        "pass_timeout_ms": int(pass_timeout_ms),
        "passes_per_round": int(passes_per_round),
        "max_parallel": int(max_parallel),
        "max_rounds": int(max_rounds),
        "stop_after_no_progress_rounds": int(stop_after_no_progress_rounds),
        "mismatch_only": bool(mismatch_only),
        "synthesize_partials": bool(synthesize_partials),
        "lanes": [
            {
                "language": lane.language,
                "output_tag": lane.output_tag,
                "target_subjects": lane.target_subjects,
            }
            for lane in lanes
        ],
        "rounds": [],
        "status": "running",
    }

    idle_rounds = 0
    for round_index in range(1, max_rounds + 1):
        pending_lanes = [
            lane
            for lane in lanes
            if _lane_subjects_completed(config, model=model, lane=lane) < lane.target_subjects
        ]
        if not pending_lanes:
            summary["status"] = "completed"
            break

        round_rows: list[dict[str, Any]] = []
        before_counts = {
            lane.language: _lane_subjects_completed(config, model=model, lane=lane) for lane in pending_lanes
        }

        with ThreadPoolExecutor(max_workers=min(max_parallel, len(pending_lanes))) as executor:
            future_to_lane = {
                executor.submit(
                    _run_lane_once,
                    config_path=config_path,
                    lane=lane,
                    model=model,
                    layers=layers,
                    mismatch_shuffles=mismatch_shuffles,
                    pass_timeout_ms=pass_timeout_ms,
                    passes_per_round=passes_per_round,
                    mismatch_only=mismatch_only,
                ): lane
                for lane in pending_lanes
            }
            for future in as_completed(future_to_lane):
                lane = future_to_lane[future]
                before = before_counts[lane.language]
                try:
                    lane_summary = future.result()
                except Exception as exc:
                    round_rows.append(
                        {
                            "language": lane.language,
                            "output_tag": lane.output_tag,
                            "status": "failed",
                            "error": str(exc),
                            "target_subjects": lane.target_subjects,
                        }
                    )
                    continue
                after = _lane_subjects_completed(config, model=model, lane=lane)
                pass_statuses = {
                    str(payload.get("status", "")).lower()
                    for payload in lane_summary.get("passes", [])
                }
                if "failed" in pass_statuses:
                    lane_status = "failed"
                elif after >= lane.target_subjects:
                    lane_status = "completed"
                else:
                    lane_status = "incomplete"
                lane_row = {
                    "language": lane.language,
                    "output_tag": lane.output_tag,
                    "status": lane_status,
                    "subjects_before_round": before,
                    "subjects_after_round": after,
                    "subjects_added": after - before,
                    "target_subjects": lane.target_subjects,
                    "lane_summary": lane_summary,
                }
                if synthesize_partials and after > before:
                    try:
                        lane_row["synthesized_outputs"] = _synthesize_lane_partial(
                            config_path=config_path,
                            lane=lane,
                            model=model,
                            completed_subjects=after,
                        )
                    except Exception as exc:
                        lane_row["synthesis_error"] = str(exc)
                round_rows.append(lane_row)

        total_added = sum(int(row.get("subjects_added", 0)) for row in round_rows)
        idle_rounds = idle_rounds + 1 if total_added <= 0 else 0
        summary["rounds"].append(
            {
                "round_index": round_index,
                "total_subjects_added": total_added,
                "lanes": sorted(round_rows, key=lambda row: row["language"]),
            }
        )

        if any(row["status"] == "failed" for row in round_rows):
            summary["status"] = "failed"
            break

        if all(_lane_subjects_completed(config, model=model, lane=lane) >= lane.target_subjects for lane in lanes):
            summary["status"] = "completed"
            break

        if idle_rounds >= stop_after_no_progress_rounds:
            summary["status"] = "stalled"
            break
    else:
        summary["status"] = "max_rounds_reached"

    summary["final_subject_counts"] = {
        lane.language: _lane_subjects_completed(config, model=model, lane=lane) for lane in lanes
    }
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run resumable refresh checkpoint lanes in parallel until completion, stall, or max-round limit."
    )
    parser.add_argument("--config", default="conf/base.yaml", help="Path to sequel config.")
    parser.add_argument("--model", default="nllb", choices=["xlmr", "nllb"])
    parser.add_argument("--languages", nargs="+", default=["en", "fr", "zh"], choices=["en", "fr", "zh"])
    parser.add_argument("--layers", nargs="+", type=int, required=True)
    parser.add_argument(
        "--output-tag-template",
        default="sequel_refresh_nllb_missing_layers_{language}",
        help="Python format string with {language}.",
    )
    parser.add_argument("--max-parallel", type=int, default=3)
    parser.add_argument("--mismatch-shuffles", type=int, default=None)
    parser.add_argument("--pass-timeout-ms", type=int, default=900000)
    parser.add_argument("--passes-per-round", type=int, default=1)
    parser.add_argument("--max-rounds", type=int, default=100)
    parser.add_argument("--stop-after-no-progress-rounds", type=int, default=2)
    parser.add_argument("--mismatch-only", action="store_true")
    parser.add_argument("--synthesize-partials", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    mismatch_shuffles = (
        int(args.mismatch_shuffles)
        if args.mismatch_shuffles is not None
        else _default_mismatch_shuffles(
            config,
            mismatch_only=args.mismatch_only,
            layer_indices=tuple(args.layers),
        )
    )
    summary = run_parallel_refresh_checkpoint(
        config,
        config_path=args.config,
        model=args.model,
        languages=tuple(args.languages),
        layers=tuple(args.layers),
        output_tag_template=args.output_tag_template,
        max_parallel=args.max_parallel,
        mismatch_shuffles=mismatch_shuffles,
        pass_timeout_ms=args.pass_timeout_ms,
        passes_per_round=args.passes_per_round,
        max_rounds=args.max_rounds,
        stop_after_no_progress_rounds=args.stop_after_no_progress_rounds,
        mismatch_only=args.mismatch_only,
        synthesize_partials=args.synthesize_partials,
    )
    print(json.dumps(summary, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
