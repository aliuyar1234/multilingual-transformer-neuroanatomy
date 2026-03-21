from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.utils.config import load_config


MODEL_MAP = {"xlmr": "xlmr", "nllb_encoder": "nllb"}
LANGUAGE_MAP = {"en": "EN", "fr": "FR", "zh": "ZH"}


def build_output_state_manifest(config: dict) -> Path:
    prior_root = Path(config["paths"]["prior_data_root"]).resolve()
    output_root = Path(config["paths"]["local_outputs_root"]).resolve()
    source_path = prior_root / "interim" / "embeddings" / "embedding_manifest.parquet"
    target_path = output_root / "caches" / "features" / "output_state_manifest.parquet"
    target_path.parent.mkdir(parents=True, exist_ok=True)

    source_df = pd.read_parquet(source_path).copy()
    manifest_df = pd.DataFrame(
        {
            "model": source_df["model"].map(MODEL_MAP),
            "language": source_df["language"].map(LANGUAGE_MAP),
            "block": source_df["layer_index"].astype(int),
            "block_depth_norm": source_df["layer_depth"].astype(float),
            "state": "OUTPUT",
            "n_rows": source_df["n_rows"].astype(int),
            "hidden_size": source_df["hidden_size"].astype(int),
            "dtype": source_df["dtype"].astype(str),
            "source_array_path": source_df["filepath"].astype(str),
            "source_artifact": str(source_path),
        }
    ).sort_values(["model", "language", "block"]).reset_index(drop=True)
    manifest_df.to_parquet(target_path, index=False)
    return target_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bridge prior hidden-state embeddings into sequel OUTPUT-state cache manifest.")
    parser.add_argument("--config", default="conf/base.yaml", help="Path to sequel config.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    target_path = build_output_state_manifest(config)
    print(f"Wrote OUTPUT-state manifest to {target_path}")


if __name__ == "__main__":
    main()
