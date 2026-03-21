from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import UTC, datetime
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Sequence

from src.encoding.merge_state_sweep_shards import merge_state_sweep_shards
from src.encoding.run_state_sweep import _group_results_path, _provenance_path as _sweep_provenance_path, _subject_results_path
from src.utils.config import load_config


@dataclass(frozen=True, slots=True)
class SweepShard:
    language: str
    output_tag: str


PRIMARY_STATES = ("ATTN", "POST_ATTN", "FFN", "OUTPUT")
PRIMARY_CONDITIONS = ("SHARED", "SPECIFIC")


def _local_outputs_root(config: dict[str, Any]) -> Path:
    return Path(config["paths"]["local_outputs_root"]).resolve()


def _provenance_path(config: dict[str, Any]) -> Path:
    return _local_outputs_root(config) / "provenance" / "parallel_state_sweep.md"


def _manifest_path(config: dict[str, Any]) -> Path:
    return _local_outputs_root(config) / "provenance" / "parallel_state_sweep_manifest.json"


def _now_utc() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _build_shards(languages: Sequence[str]) -> tuple[SweepShard, ...]:
    return tuple(
        SweepShard(language=str(language).upper(), output_tag=str(language).lower())
        for language in tuple(dict.fromkeys(str(language).upper() for language in languages))
    )


def _write_manifest(
    config: dict[str, Any],
    *,
    models: Sequence[str],
    languages: Sequence[str],
    states: Sequence[str],
    conditions: Sequence[str],
    max_subjects: int | None,
    max_parallel: int,
    shards: Sequence[SweepShard],
) -> Path:
    manifest_path = _manifest_path(config)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "run_id": _now_utc(),
        "models": [str(model) for model in models],
        "languages": [str(language).upper() for language in languages],
        "states": [str(state).upper() for state in states],
        "conditions": [str(condition).upper() for condition in conditions],
        "max_subjects": max_subjects,
        "max_parallel": int(max_parallel),
        "shards": [
            {
                "language": shard.language,
                "output_tag": shard.output_tag,
                "subject_results_path": str(_subject_results_path(config, shard.output_tag)),
                "group_results_path": str(_group_results_path(config, shard.output_tag)),
                "provenance_path": str(_sweep_provenance_path(config, shard.output_tag)),
            }
            for shard in shards
        ],
    }
    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return manifest_path


def _shard_command(
    *,
    config_path: str,
    shard: SweepShard,
    models: Sequence[str],
    states: Sequence[str],
    conditions: Sequence[str],
    max_subjects: int | None,
) -> list[str]:
    command = [
        sys.executable,
        "-u",
        "-m",
        "src.encoding.run_state_sweep",
        "--config",
        config_path,
        "--models",
        *[str(model) for model in models],
        "--languages",
        str(shard.language),
        "--states",
        *[str(state).upper() for state in states],
        "--conditions",
        *[str(condition).upper() for condition in conditions],
        "--output-tag",
        str(shard.output_tag),
    ]
    if max_subjects is not None:
        command.extend(["--max-subjects", str(int(max_subjects))])
    return command


def _run_shard(
    *,
    config_path: str,
    shard: SweepShard,
    models: Sequence[str],
    states: Sequence[str],
    conditions: Sequence[str],
    max_subjects: int | None,
) -> dict[str, object]:
    completed = subprocess.run(
        _shard_command(
            config_path=config_path,
            shard=shard,
            models=models,
            states=states,
            conditions=conditions,
            max_subjects=max_subjects,
        ),
        check=True,
        capture_output=True,
        text=True,
    )
    return {
        "language": shard.language,
        "output_tag": shard.output_tag,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }


def run_parallel_state_sweep(
    config: dict[str, Any],
    *,
    config_path: str = "conf/base.yaml",
    models: Sequence[str] = ("xlmr", "nllb"),
    languages: Sequence[str] = ("EN", "FR", "ZH"),
    states: Sequence[str] = PRIMARY_STATES,
    conditions: Sequence[str] = PRIMARY_CONDITIONS,
    max_subjects: int | None = None,
    max_parallel: int = 3,
) -> dict[str, Path]:
    shards = _build_shards(languages)
    if max_parallel < 1:
        raise ValueError("max_parallel must be positive.")
    manifest_path = _write_manifest(
        config,
        models=tuple(dict.fromkeys(str(model) for model in models)),
        languages=tuple(dict.fromkeys(str(language).upper() for language in languages)),
        states=tuple(dict.fromkeys(str(state).upper() for state in states)),
        conditions=tuple(dict.fromkeys(str(condition).upper() for condition in conditions)),
        max_subjects=max_subjects,
        max_parallel=max_parallel,
        shards=shards,
    )

    shard_results: list[dict[str, object]] = []
    with ThreadPoolExecutor(max_workers=min(int(max_parallel), len(shards))) as executor:
        futures = {
            executor.submit(
                _run_shard,
                config_path=config_path,
                shard=shard,
                models=tuple(dict.fromkeys(str(model) for model in models)),
                states=tuple(dict.fromkeys(str(state).upper() for state in states)),
                conditions=tuple(dict.fromkeys(str(condition).upper() for condition in conditions)),
                max_subjects=max_subjects,
            ): shard
            for shard in shards
        }
        for future in as_completed(futures):
            shard_results.append(future.result())

    outputs = merge_state_sweep_shards(config, manifest_path=manifest_path)
    provenance_path = _provenance_path(config)
    provenance_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Parallel State Sweep",
        "",
        f"- run_manifest: `{manifest_path}`",
        f"- shards: `{len(shards)}`",
        f"- max_parallel: `{int(max_parallel)}`",
        f"- models: `{', '.join(tuple(dict.fromkeys(str(model) for model in models)))}`",
        f"- states: `{', '.join(tuple(dict.fromkeys(str(state).upper() for state in states)))}`",
        f"- conditions: `{', '.join(tuple(dict.fromkeys(str(condition).upper() for condition in conditions)))}`",
        f"- max_subjects: `{max_subjects if max_subjects is not None else 'all included'}`",
        "",
        "## Completed Shards",
        "",
    ]
    for result in sorted(shard_results, key=lambda item: str(item["language"])):
        lines.append(f"- {result['language']} -> `{result['output_tag']}`")
    provenance_path.write_text("\n".join(lines), encoding="utf-8")
    outputs["parallel_provenance"] = provenance_path
    outputs["run_manifest"] = manifest_path
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run language-sharded sequel state sweeps in parallel, then merge canonical outputs.")
    parser.add_argument("--config", default="conf/base.yaml", help="Path to sequel config.")
    parser.add_argument("--models", nargs="+", default=["xlmr", "nllb"], choices=["xlmr", "nllb"])
    parser.add_argument("--languages", nargs="+", default=["EN", "FR", "ZH"], choices=["EN", "FR", "ZH", "en", "fr", "zh"])
    parser.add_argument("--states", nargs="+", default=list(PRIMARY_STATES), choices=list(PRIMARY_STATES) + ["INPUT"])
    parser.add_argument("--conditions", nargs="+", default=list(PRIMARY_CONDITIONS), choices=["SHARED", "SPECIFIC", "RAW", "FULL"])
    parser.add_argument("--max-subjects", type=int, default=None)
    parser.add_argument("--max-parallel", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    parallel_cfg = dict(config.get("parallel", {}).get("sweep", {}))
    outputs = run_parallel_state_sweep(
        config,
        config_path=args.config,
        models=tuple(args.models),
        languages=tuple(args.languages),
        states=tuple(args.states),
        conditions=tuple(args.conditions),
        max_subjects=args.max_subjects,
        max_parallel=int(args.max_parallel if args.max_parallel is not None else parallel_cfg.get("max_parallel", 3)),
    )
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
