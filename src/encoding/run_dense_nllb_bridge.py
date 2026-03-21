from __future__ import annotations

import argparse
import json

from src.encoding.run_parallel_refresh_checkpoint import run_parallel_refresh_checkpoint
from src.utils.config import load_config


DEFAULT_LANGUAGES = ("en", "fr", "zh")
DEFAULT_LAYERS = (0, 1, 2, 3, 5, 7, 9, 10, 11, 12)
DEFAULT_OUTPUT_TAG_TEMPLATE = "sequel_refresh_nllb_missing_layers_{language}"
DEFAULT_MAX_PARALLEL = 3
DEFAULT_PASS_TIMEOUT_MS = 1800000
DEFAULT_PASSES_PER_ROUND = 1
DEFAULT_MAX_ROUNDS = 100
DEFAULT_STOP_AFTER_NO_PROGRESS_ROUNDS = 2
DEFAULT_MISMATCH_SHUFFLES = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Short, resumable entry point for the dense NLLB OUTPUT bridge refresh."
    )
    parser.add_argument("--config", default="conf/base.yaml", help="Path to sequel config.")
    parser.add_argument("--max-parallel", type=int, default=DEFAULT_MAX_PARALLEL)
    parser.add_argument("--pass-timeout-ms", type=int, default=DEFAULT_PASS_TIMEOUT_MS)
    parser.add_argument("--passes-per-round", type=int, default=DEFAULT_PASSES_PER_ROUND)
    parser.add_argument("--max-rounds", type=int, default=DEFAULT_MAX_ROUNDS)
    parser.add_argument(
        "--stop-after-no-progress-rounds",
        type=int,
        default=DEFAULT_STOP_AFTER_NO_PROGRESS_ROUNDS,
    )
    parser.add_argument("--mismatch-shuffles", type=int, default=DEFAULT_MISMATCH_SHUFFLES)
    parser.add_argument("--no-synthesize-partials", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    summary = run_parallel_refresh_checkpoint(
        config,
        config_path=args.config,
        model="nllb",
        languages=DEFAULT_LANGUAGES,
        layers=DEFAULT_LAYERS,
        output_tag_template=DEFAULT_OUTPUT_TAG_TEMPLATE,
        max_parallel=args.max_parallel,
        mismatch_shuffles=args.mismatch_shuffles,
        pass_timeout_ms=args.pass_timeout_ms,
        passes_per_round=args.passes_per_round,
        max_rounds=args.max_rounds,
        stop_after_no_progress_rounds=args.stop_after_no_progress_rounds,
        mismatch_only=False,
        synthesize_partials=not args.no_synthesize_partials,
    )
    print(json.dumps(summary, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
