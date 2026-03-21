from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

# Warm-import torch before numpy / extractor imports so CUDA DLL initialization stays stable on Windows.
import torch
import numpy as np

from src.features.extract_internal_states import (
    _batch_index_groups,
    _load_model_bundle,
    _pool_state_tensor,
    _tokenizer_kwargs_for_batch,
    _tokenizer_kwargs_for_lengths,
    run_model_forward_with_state_capture,
)
from src.utils.config import load_config


def _encode_text_batch(
    *,
    config: dict[str, Any],
    model: str,
    language: str,
    state: str,
    block: int,
    texts: list[str],
) -> np.ndarray:
    bundle = _load_model_bundle(model, str(config["paths"]["prior_models_root"]))
    tokenizer = bundle.tokenizer
    length_kwargs = _tokenizer_kwargs_for_lengths(model_key=model, tokenizer=tokenizer, language=language)
    token_lengths = [len(tokenizer(text, **length_kwargs)["input_ids"]) for text in texts]
    batches = _batch_index_groups(
        token_lengths,
        max_tokens_per_batch=int(config.get("features", {}).get("max_tokens_per_batch", 2048)),
    )

    pooled = np.zeros((len(texts), int(bundle.hidden_size)), dtype=np.float32)
    batch_kwargs = _tokenizer_kwargs_for_batch(model_key=model, tokenizer=tokenizer, language=language)
    for batch_indices in batches:
        batch_texts = [texts[index] for index in batch_indices]
        encoded = tokenizer(batch_texts, **batch_kwargs)
        offset_mapping = encoded.pop("offset_mapping", None)
        _ = offset_mapping
        encoded = {key: value.to(bundle.device) for key, value in encoded.items()}
        pooling_mask = encoded["attention_mask"].bool() & ~encoded["special_tokens_mask"].bool()
        state_tensors, _ = run_model_forward_with_state_capture(
            model_key=model,
            model=bundle.model,
            input_ids=encoded["input_ids"],
            attention_mask=encoded["attention_mask"],
            requested_states=(state,),
        )
        batch_pooled = _pool_state_tensor(
            state_tensors[(state, int(block))],
            pooling_mask,
            normalize=bool(str(config.get("features", {}).get("normalize", "l2")).lower() == "l2"),
            eps=float(config.get("features", {}).get("eps", 1.0e-8)),
        )
        for row_offset, text_index in enumerate(batch_indices):
            pooled[text_index] = batch_pooled[row_offset]
    return pooled


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch-encode mutated texts for attribution under a specific state/block.")
    parser.add_argument("--config", default="conf/base.yaml", help="Path to sequel config.")
    parser.add_argument("--model", required=True, choices=["xlmr", "nllb"])
    parser.add_argument("--language", choices=["EN", "FR", "ZH", "en", "fr", "zh"])
    parser.add_argument("--state")
    parser.add_argument("--block", type=int)
    parser.add_argument("--input-json", help="JSON file containing a list of texts.")
    parser.add_argument("--output-npy", help="Destination .npy path for pooled vectors.")
    parser.add_argument("--worker", action="store_true", help="Run as a persistent stdin/stdout worker.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.worker:
        for line in sys.stdin:
            payload = json.loads(line)
            if str(payload.get("op", "")).lower() == "shutdown":
                sys.stdout.write(json.dumps({"status": "bye"}) + "\n")
                sys.stdout.flush()
                return
            try:
                output_path = Path(str(payload["output_npy"])).resolve()
                pooled = _encode_text_batch(
                    config=config,
                    model=str(args.model),
                    language=str(payload["language"]).upper(),
                    state=str(payload["state"]).upper(),
                    block=int(payload["block"]),
                    texts=[str(text) for text in payload["texts"]],
                )
                output_path.parent.mkdir(parents=True, exist_ok=True)
                np.save(output_path, pooled.astype(np.float32, copy=False))
                sys.stdout.write(json.dumps({"status": "ok", "n_rows": int(len(pooled))}) + "\n")
                sys.stdout.flush()
            except Exception as exc:  # pragma: no cover - exercised through parent worker integration
                sys.stdout.write(json.dumps({"status": "error", "error": str(exc)}) + "\n")
                sys.stdout.flush()
        return

    if not all((args.language, args.state, args.block is not None, args.input_json, args.output_npy)):
        raise SystemExit("--language, --state, --block, --input-json, and --output-npy are required unless --worker is used")
    input_path = Path(args.input_json).resolve()
    output_path = Path(args.output_npy).resolve()
    texts = json.loads(input_path.read_text(encoding="utf-8"))
    pooled = _encode_text_batch(
        config=config,
        model=str(args.model),
        language=str(args.language).upper(),
        state=str(args.state).upper(),
        block=int(args.block),
        texts=[str(text) for text in texts],
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, pooled.astype(np.float32, copy=False))


if __name__ == "__main__":
    main()
