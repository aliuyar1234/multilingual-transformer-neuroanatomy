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

from src.features.consolidate_internal_state_shards import (
    _output_state_manifest_path,
    _state_adapter_note_path,
    _state_cache_manifest_path,
    _state_extraction_note_path,
    _state_extraction_qc_path,
    _token_metadata_path,
    _triplet_ids_path,
    consolidate_internal_state_shards,
)
from src.utils.config import load_config


@dataclass(frozen=True, slots=True)
class ExtractionShard:
    model: str
    languages: tuple[str, ...]
    artifact_tag: str


def _local_outputs_root(config: dict[str, Any]) -> Path:
    return Path(config["paths"]["local_outputs_root"]).resolve()


def _provenance_path(config: dict[str, Any]) -> Path:
    return _local_outputs_root(config) / "provenance" / "parallel_state_extraction.md"


def _manifest_path(config: dict[str, Any]) -> Path:
    return _local_outputs_root(config) / "provenance" / "parallel_state_extraction_manifest.json"


def _now_utc() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _build_shards(*, models: Sequence[str], languages: Sequence[str]) -> tuple[ExtractionShard, ...]:
    normalized_languages = tuple(dict.fromkeys(str(value).upper() for value in languages))
    shards: list[ExtractionShard] = []
    for model in tuple(dict.fromkeys(str(value).lower() for value in models)):
        shards.append(ExtractionShard(model=model, languages=normalized_languages, artifact_tag=str(model)))
    return tuple(shards)


def _write_manifest(
    config: dict[str, Any],
    *,
    models: Sequence[str],
    languages: Sequence[str],
    states: Sequence[str] | None,
    max_parallel: int,
    max_tokens_per_batch: int,
    cache_dtype: str | None,
    shards: Sequence[ExtractionShard],
) -> Path:
    manifest_path = _manifest_path(config)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "run_id": _now_utc(),
        "models": [str(model) for model in models],
        "languages": [str(language).upper() for language in languages],
        "states": [str(state).upper() for state in states] if states else None,
        "max_parallel": int(max_parallel),
        "max_tokens_per_batch": int(max_tokens_per_batch),
        "cache_dtype": cache_dtype,
        "shards": [
            {
                "model": shard.model,
                "languages": [str(language) for language in shard.languages],
                "artifact_tag": shard.artifact_tag,
                "state_manifest_path": str(_state_cache_manifest_path(config, shard.artifact_tag)),
                "output_manifest_path": str(_output_state_manifest_path(config, shard.artifact_tag)),
                "token_metadata_path": str(_token_metadata_path(config, shard.artifact_tag)),
                "triplet_ids_path": str(_triplet_ids_path(config, shard.artifact_tag)),
                "qc_path": str(_state_extraction_qc_path(config, shard.artifact_tag)),
                "provenance_path": str(_state_extraction_note_path(config, shard.artifact_tag)),
                "adapter_notes_path": str(_state_adapter_note_path(config, shard.artifact_tag)),
            }
            for shard in shards
        ],
    }
    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return manifest_path


def _shard_command(
    *,
    config_path: str,
    shard: ExtractionShard,
    states: Sequence[str] | None,
    max_tokens_per_batch: int,
    cache_dtype: str | None,
) -> list[str]:
    argv = [
        "extract_internal_states",
        "--config",
        config_path,
        "--models",
        str(shard.model),
        "--languages",
        *[str(language) for language in shard.languages],
        "--max-tokens-per-batch",
        str(int(max_tokens_per_batch)),
        "--artifact-tag",
        str(shard.artifact_tag),
    ]
    if states:
        argv.extend(["--states", *[str(state).upper() for state in states]])
    if cache_dtype:
        argv.extend(["--cache-dtype", str(cache_dtype)])
    inline_code = (
        "import torch\n"
        "import sys\n"
        "from src.features.extract_internal_states import main\n"
        f"sys.argv = {argv!r}\n"
        "main()\n"
    )
    return [sys.executable, "-u", "-c", inline_code]


def _run_shard(
    *,
    config_path: str,
    shard: ExtractionShard,
    states: Sequence[str] | None,
    max_tokens_per_batch: int,
    cache_dtype: str | None,
) -> dict[str, object]:
    command = _shard_command(
        config_path=config_path,
        shard=shard,
        states=states,
        max_tokens_per_batch=max_tokens_per_batch,
        cache_dtype=cache_dtype,
    )
    completed = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
    )
    return {
        "model": shard.model,
        "languages": shard.languages,
        "artifact_tag": shard.artifact_tag,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }


def run_parallel_internal_state_extraction(
    config: dict[str, Any],
    *,
    config_path: str = "conf/base.yaml",
    models: Sequence[str] = ("xlmr", "nllb"),
    languages: Sequence[str] = ("EN", "FR", "ZH"),
    states: Sequence[str] | None = None,
    max_parallel: int = 1,
    max_tokens_per_batch: int = 2048,
    cache_dtype: str | None = None,
) -> dict[str, Path]:
    shards = _build_shards(models=models, languages=languages)
    if max_parallel < 1:
        raise ValueError("max_parallel must be positive.")
    manifest_path = _write_manifest(
        config,
        models=models,
        languages=languages,
        states=states,
        max_parallel=max_parallel,
        max_tokens_per_batch=max_tokens_per_batch,
        cache_dtype=cache_dtype,
        shards=shards,
    )

    shard_results: list[dict[str, object]] = []
    with ThreadPoolExecutor(max_workers=min(int(max_parallel), len(shards))) as executor:
        futures = {
            executor.submit(
                _run_shard,
                config_path=config_path,
                shard=shard,
                states=states,
                max_tokens_per_batch=max_tokens_per_batch,
                cache_dtype=cache_dtype,
            ): shard
            for shard in shards
        }
        for future in as_completed(futures):
            shard_results.append(future.result())

    outputs = consolidate_internal_state_shards(config, manifest_path=manifest_path)
    provenance_path = _provenance_path(config)
    provenance_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Parallel Internal State Extraction",
        "",
        f"- run_manifest: `{manifest_path}`",
        f"- shards: `{len(shards)}`",
        f"- max_parallel: `{int(max_parallel)}`",
        f"- states: `{', '.join(states) if states else 'config default'}`",
        f"- cache_dtype: `{cache_dtype or 'config default'}`",
        f"- max_tokens_per_batch: `{int(max_tokens_per_batch)}`",
        "",
        "## Completed Shards",
        "",
    ]
    for result in sorted(shard_results, key=lambda item: str(item["model"])):
        lines.append(f"- {result['model']}:{', '.join(result['languages'])} -> `{result['artifact_tag']}`")
    provenance_path.write_text("\n".join(lines), encoding="utf-8")
    outputs["parallel_provenance"] = provenance_path
    outputs["run_manifest"] = manifest_path
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run model-resident internal-state extraction shards in parallel, then consolidate.")
    parser.add_argument("--config", default="conf/base.yaml", help="Path to sequel config.")
    parser.add_argument("--models", nargs="+", default=["xlmr", "nllb"], choices=["xlmr", "nllb"])
    parser.add_argument("--languages", nargs="+", default=["EN", "FR", "ZH"], choices=["EN", "FR", "ZH", "en", "fr", "zh"])
    parser.add_argument("--states", nargs="+", default=None, choices=["INPUT", "ATTN", "POST_ATTN", "FFN", "OUTPUT"])
    parser.add_argument("--max-parallel", type=int, default=None)
    parser.add_argument("--max-tokens-per-batch", type=int, default=None)
    parser.add_argument("--cache-dtype", default=None, choices=["float16", "float32", "fp16", "fp32"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    parallel_cfg = dict(config.get("parallel", {}).get("extraction", {}))
    outputs = run_parallel_internal_state_extraction(
        config,
        config_path=args.config,
        models=tuple(args.models),
        languages=tuple(args.languages),
        states=None if args.states is None else tuple(args.states),
        max_parallel=int(args.max_parallel if args.max_parallel is not None else parallel_cfg.get("max_parallel", 1)),
        max_tokens_per_batch=int(
            args.max_tokens_per_batch
            if args.max_tokens_per_batch is not None
            else parallel_cfg.get("max_tokens_per_batch", 2048)
        ),
        cache_dtype=args.cache_dtype or (str(parallel_cfg["cache_dtype"]) if "cache_dtype" in parallel_cfg else None),
    )
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
