from __future__ import annotations

import argparse
import json
from contextlib import AbstractContextManager
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Sequence

# On Windows, importing torch before numpy/pandas avoids intermittent CUDA DLL load failures.
import torch
import numpy as np
import pandas as pd
from transformers import AutoModel, AutoTokenizer

from src.utils.config import load_config


LANGUAGE_TEXT_COLUMNS = {
    "EN": "en_text",
    "FR": "fr_text",
    "ZH": "zh_text",
}
LOWER_TO_UPPER_LANGUAGE = {language.lower(): language for language in LANGUAGE_TEXT_COLUMNS}
NLLB_LANGUAGE_CODES = {
    "EN": "eng_Latn",
    "FR": "fra_Latn",
    "ZH": "zho_Hans",
}
MODEL_DIR_NAMES = {
    "xlmr": "xlmr",
    "nllb": "nllb_encoder",
}
VALID_STATES = ("INPUT", "ATTN", "POST_ATTN", "FFN", "OUTPUT")


def _now_utc() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _normalize_rows(array: np.ndarray, *, eps: float) -> np.ndarray:
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    norms = np.clip(norms, eps, None)
    return (array / norms).astype(np.float32, copy=False)


def _batch_index_groups(token_lengths: Sequence[int], *, max_tokens_per_batch: int) -> list[list[int]]:
    groups: list[list[int]] = []
    current: list[int] = []
    current_tokens = 0
    for row_index, token_length in enumerate(token_lengths):
        token_cost = max(1, int(token_length))
        if current and current_tokens + token_cost > max_tokens_per_batch:
            groups.append(current)
            current = [row_index]
            current_tokens = token_cost
            continue
        current.append(row_index)
        current_tokens += token_cost
    if current:
        groups.append(current)
    return groups


def _extract_tensor(output: Any) -> torch.Tensor:
    if isinstance(output, tuple):
        return output[0]
    if isinstance(output, list):
        return output[0]
    if torch.is_tensor(output):
        return output
    raise TypeError(f"Unsupported hook output type: {type(output)!r}")


@dataclass(frozen=True, slots=True)
class LoadedModelBundle:
    model_key: str
    model_dir: Path
    tokenizer: Any
    model: Any
    encoder: Any
    encoder_layers: Sequence[torch.nn.Module]
    device: torch.device
    hidden_size: int
    n_blocks: int


@dataclass(frozen=True, slots=True)
class EncodedStateBundle:
    block_states: dict[int, dict[str, np.ndarray]]
    token_strings: tuple[str, ...]
    token_offsets: tuple[tuple[int, int], ...]


@dataclass(slots=True)
class ModelStateAdapter:
    model_name: str
    model: Any
    tokenizer: Any
    state_names: tuple[str, ...]
    n_blocks: int
    _last_output_equivalence: dict[int, float] | None = None

    def encode_text_with_states(self, text: str) -> EncodedStateBundle:
        kwargs: dict[str, Any] = {
            "return_tensors": "pt",
            "return_special_tokens_mask": True,
        }
        if getattr(self.tokenizer, "is_fast", False):
            kwargs["return_offsets_mapping"] = True
        try:
            encoded = self.tokenizer(text, **kwargs)
            offset_mapping = encoded.pop("offset_mapping", None)
        except (NotImplementedError, ValueError):
            kwargs.pop("return_offsets_mapping", None)
            encoded = self.tokenizer(text, **kwargs)
            offset_mapping = None
        model_device = next(self.model.parameters()).device
        state_tensors, outputs = run_model_forward_with_state_capture(
            model_key=self.model_name,
            model=self.model,
            input_ids=encoded["input_ids"].to(model_device),
            attention_mask=encoded["attention_mask"].to(model_device),
            requested_states=self.state_names,
        )
        block_states: dict[int, dict[str, np.ndarray]] = {}
        output_equivalence: dict[int, float] = {}
        for block_index in range(self.n_blocks):
            block_state_map: dict[str, np.ndarray] = {}
            for state_name in self.state_names:
                tensor = state_tensors[(state_name, block_index)][0].detach().cpu().numpy().astype(np.float32, copy=False)
                block_state_map[state_name] = tensor
            block_states[block_index] = block_state_map
            difference = np.abs(block_state_map["OUTPUT"] - outputs.hidden_states[block_index + 1][0].detach().cpu().numpy())
            output_equivalence[block_index] = float(difference.max())

        token_ids = encoded["input_ids"][0].detach().cpu().tolist()
        token_strings = tuple(self.tokenizer.convert_ids_to_tokens(token_ids))
        if offset_mapping is not None:
            token_offsets = tuple(tuple(int(value) for value in pair) for pair in offset_mapping[0].tolist())
        else:
            token_offsets = tuple((-1, -1) for _ in token_ids)

        self._last_output_equivalence = output_equivalence
        return EncodedStateBundle(
            block_states=block_states,
            token_strings=token_strings,
            token_offsets=token_offsets,
        )

    def verify_output_equivalence(self) -> dict[int, float]:
        return dict(self._last_output_equivalence or {})


@lru_cache(maxsize=4)
def _load_model_bundle(model_key: str, prior_models_root: str) -> LoadedModelBundle:
    if model_key not in MODEL_DIR_NAMES:
        raise KeyError(f"Unsupported model key: {model_key}")
    model_dir = Path(prior_models_root).resolve() / MODEL_DIR_NAMES[model_key]
    if not model_dir.exists():
        raise FileNotFoundError(f"Local model directory missing: {model_dir}")

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), use_fast=True)
    model = AutoModel.from_pretrained(str(model_dir))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    if model_key == "nllb":
        if not hasattr(model, "get_encoder"):
            raise RuntimeError("Expected NLLB AutoModel to expose get_encoder().")
        encoder = model.get_encoder()
        encoder_layers = tuple(encoder.layers)
        hidden_size = int(model.config.d_model)
    else:
        encoder = model
        encoder_layers = tuple(model.encoder.layer)
        hidden_size = int(model.config.hidden_size)

    return LoadedModelBundle(
        model_key=model_key,
        model_dir=model_dir,
        tokenizer=tokenizer,
        model=model,
        encoder=encoder,
        encoder_layers=encoder_layers,
        device=device,
        hidden_size=hidden_size,
        n_blocks=len(encoder_layers),
    )


class StateCaptureController(AbstractContextManager["StateCaptureController"]):
    def __init__(self, *, model_key: str, encoder_layers: Sequence[torch.nn.Module], requested_states: Iterable[str]) -> None:
        unknown_states = sorted(set(requested_states).difference(VALID_STATES))
        if unknown_states:
            raise ValueError(f"Unsupported state names: {unknown_states}")

        self.model_key = model_key
        self.encoder_layers = tuple(encoder_layers)
        self.requested_states = tuple(dict.fromkeys(str(state).upper() for state in requested_states))
        self.handles: list[Any] = []
        self.batch_states: dict[tuple[str, int], torch.Tensor] = {}

    def _record(self, state: str, block_index: int, tensor: torch.Tensor) -> None:
        self.batch_states[(state, int(block_index))] = tensor.detach().clone()

    def clear(self) -> None:
        self.batch_states = {}

    def __enter__(self) -> StateCaptureController:
        for block_index, layer in enumerate(self.encoder_layers):
            if "INPUT" in self.requested_states:
                self.handles.append(
                    layer.register_forward_pre_hook(
                        lambda _module, inputs, *, block_index=block_index: self._record(
                            "INPUT",
                            block_index,
                            _extract_tensor(inputs[0]),
                        )
                    )
                )

            if self.model_key == "xlmr":
                if "ATTN" in self.requested_states:
                    self.handles.append(
                        layer.attention.output.dropout.register_forward_hook(
                            lambda _module, _inputs, output, *, block_index=block_index: self._record(
                                "ATTN",
                                block_index,
                                _extract_tensor(output),
                            )
                        )
                    )
                if "POST_ATTN" in self.requested_states:
                    self.handles.append(
                        layer.attention.register_forward_hook(
                            lambda _module, _inputs, output, *, block_index=block_index: self._record(
                                "POST_ATTN",
                                block_index,
                                _extract_tensor(output),
                            )
                        )
                    )
                if "FFN" in self.requested_states:
                    self.handles.append(
                        layer.output.dropout.register_forward_hook(
                            lambda _module, _inputs, output, *, block_index=block_index: self._record(
                                "FFN",
                                block_index,
                                _extract_tensor(output),
                            )
                        )
                    )
            else:
                if "ATTN" in self.requested_states:
                    self.handles.append(
                        layer.self_attn.register_forward_hook(
                            lambda _module, _inputs, output, *, block_index=block_index: self._record(
                                "ATTN",
                                block_index,
                                _extract_tensor(output),
                            )
                        )
                    )
                if "POST_ATTN" in self.requested_states:
                    self.handles.append(
                        layer.final_layer_norm.register_forward_pre_hook(
                            lambda _module, inputs, *, block_index=block_index: self._record(
                                "POST_ATTN",
                                block_index,
                                _extract_tensor(inputs[0]),
                            )
                        )
                    )
                if "FFN" in self.requested_states:
                    self.handles.append(
                        layer.fc2.register_forward_hook(
                            lambda _module, _inputs, output, *, block_index=block_index: self._record(
                                "FFN",
                                block_index,
                                _extract_tensor(output),
                            )
                        )
                    )

            if "OUTPUT" in self.requested_states:
                self.handles.append(
                    layer.register_forward_hook(
                        lambda _module, _inputs, output, *, block_index=block_index: self._record(
                            "OUTPUT",
                            block_index,
                            _extract_tensor(output),
                        )
                    )
                )
        return self

    def __exit__(self, exc_type: object, exc: object, exc_tb: object) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def build_model_state_adapter(
    *,
    model_name: str,
    model: Any,
    tokenizer: Any,
    state_names: Iterable[str] = VALID_STATES,
) -> ModelStateAdapter:
    requested_states = tuple(dict.fromkeys(str(state).upper() for state in state_names))
    if model_name == "nllb":
        n_blocks = len(model.get_encoder().layers)
    else:
        n_blocks = len(model.encoder.layer)
    return ModelStateAdapter(
        model_name=model_name,
        model=model,
        tokenizer=tokenizer,
        state_names=requested_states,
        n_blocks=n_blocks,
    )


def run_model_forward_with_state_capture(
    *,
    model_key: str,
    model: Any,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    requested_states: Iterable[str],
) -> tuple[dict[tuple[str, int], torch.Tensor], Any]:
    model.eval()
    if model_key == "nllb":
        if not hasattr(model, "get_encoder"):
            raise RuntimeError("Expected NLLB model to expose get_encoder().")
        encoder = model.get_encoder()
        controller = StateCaptureController(
            model_key=model_key,
            encoder_layers=tuple(encoder.layers),
            requested_states=requested_states,
        )
        with controller, torch.no_grad():
            outputs = encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )
        if controller.batch_states:
            controller.batch_states[("OUTPUT", len(encoder.layers) - 1)] = outputs.hidden_states[-1].detach().clone()
    else:
        controller = StateCaptureController(
            model_key=model_key,
            encoder_layers=tuple(model.encoder.layer),
            requested_states=requested_states,
        )
        with controller, torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )
    return dict(controller.batch_states), outputs


def _triplets_path(config: dict[str, Any]) -> Path:
    return Path(config["paths"]["local_manifests_root"]).resolve() / "triplets.parquet"


def _tagged_path(path: Path, artifact_tag: str | None) -> Path:
    if not artifact_tag:
        return path
    suffix = "".join(path.suffixes)
    stem = path.name[: -len(suffix)] if suffix else path.name
    tagged_name = f"{stem}__{artifact_tag}{suffix}"
    return path.with_name(tagged_name)


def _state_cache_manifest_path(config: dict[str, Any], artifact_tag: str | None = None) -> Path:
    path = Path(config["paths"]["local_outputs_root"]).resolve() / "caches" / "features" / "state_cache_manifest.parquet"
    return _tagged_path(path, artifact_tag)


def _output_state_manifest_path(config: dict[str, Any], artifact_tag: str | None = None) -> Path:
    path = Path(config["paths"]["local_outputs_root"]).resolve() / "caches" / "features" / "output_state_manifest.parquet"
    return _tagged_path(path, artifact_tag)


def _token_metadata_path(config: dict[str, Any], artifact_tag: str | None = None) -> Path:
    path = Path(config["paths"]["local_outputs_root"]).resolve() / "caches" / "features" / "tokenization_metadata.parquet"
    return _tagged_path(path, artifact_tag)


def _triplet_ids_path(config: dict[str, Any], artifact_tag: str | None = None) -> Path:
    path = Path(config["paths"]["local_outputs_root"]).resolve() / "caches" / "features" / "triplet_ids.parquet"
    return _tagged_path(path, artifact_tag)


def _state_extraction_qc_path(config: dict[str, Any], artifact_tag: str | None = None) -> Path:
    path = Path(config["paths"]["local_outputs_root"]).resolve() / "qc" / "state_extraction_equivalence.parquet"
    return _tagged_path(path, artifact_tag)


def _state_extraction_note_path(config: dict[str, Any], artifact_tag: str | None = None) -> Path:
    path = Path(config["paths"]["local_outputs_root"]).resolve() / "provenance" / "state_extraction.md"
    return _tagged_path(path, artifact_tag)


def _state_adapter_note_path(config: dict[str, Any], artifact_tag: str | None = None) -> Path:
    path = Path(config["paths"]["local_outputs_root"]).resolve() / "provenance" / "state_adapter_notes.md"
    return _tagged_path(path, artifact_tag)


def _tokenizer_kwargs_for_lengths(*, model_key: str, tokenizer: Any, language: str) -> dict[str, Any]:
    if model_key == "nllb":
        tokenizer.src_lang = NLLB_LANGUAGE_CODES[language]
    return {"add_special_tokens": True, "truncation": True}


def _tokenizer_kwargs_for_batch(*, model_key: str, tokenizer: Any, language: str) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "padding": True,
        "truncation": True,
        "return_special_tokens_mask": True,
        "return_tensors": "pt",
    }
    if tokenizer.is_fast:
        kwargs["return_offsets_mapping"] = True
    if model_key == "nllb":
        tokenizer.src_lang = NLLB_LANGUAGE_CODES[language]
    return kwargs


def _select_languages(languages: Iterable[str]) -> tuple[str, ...]:
    selected: list[str] = []
    for language in languages:
        upper = LOWER_TO_UPPER_LANGUAGE.get(str(language).lower(), str(language).upper())
        if upper not in LANGUAGE_TEXT_COLUMNS:
            raise KeyError(f"Unsupported language: {language!r}")
        selected.append(upper)
    return tuple(dict.fromkeys(selected))


def _select_states(config: dict[str, Any], states: Iterable[str] | None) -> tuple[str, ...]:
    raw_states = list(config.get("features", {}).get("state_names", [])) if states is None else list(states)
    selected = [str(state).upper() for state in raw_states]
    unknown = sorted(set(selected).difference(VALID_STATES))
    if unknown:
        raise KeyError(f"Unsupported state names requested: {unknown}")
    return tuple(dict.fromkeys(selected))


def _cache_dtype(config: dict[str, Any], override: str | None) -> tuple[np.dtype[Any], str]:
    dtype_name = str(override or config.get("features", {}).get("cache_dtype", "float32")).lower()
    mapping = {
        "float16": (np.float16, "float16"),
        "fp16": (np.float16, "float16"),
        "float32": (np.float32, "float32"),
        "fp32": (np.float32, "float32"),
    }
    if dtype_name not in mapping:
        raise ValueError(f"Unsupported cache dtype: {dtype_name}")
    return mapping[dtype_name]


def _pool_state_tensor(
    state_tensor: torch.Tensor,
    pooling_mask: torch.Tensor,
    *,
    normalize: bool,
    eps: float,
) -> np.ndarray:
    token_counts = pooling_mask.sum(dim=1).clamp(min=1)
    masked = state_tensor * pooling_mask.unsqueeze(-1)
    pooled = masked.sum(dim=1) / token_counts.unsqueeze(-1)
    pooled_array = pooled.detach().cpu().to(torch.float32).numpy().astype(np.float32, copy=False)
    if normalize:
        pooled_array = _normalize_rows(pooled_array, eps=eps)
    return pooled_array


def _empty_cache_arrays(
    *,
    requested_states: Sequence[str],
    n_blocks: int,
    n_rows: int,
    hidden_size: int,
) -> dict[tuple[str, int], np.ndarray]:
    return {
        (state, block_index): np.zeros((n_rows, hidden_size), dtype=np.float32)
        for state in requested_states
        for block_index in range(n_blocks)
    }


def _array_output_path(config: dict[str, Any], *, model_key: str, language: str, state: str, block_index: int) -> Path:
    return (
        Path(config["paths"]["local_outputs_root"]).resolve()
        / "caches"
        / "features"
        / model_key
        / language
        / state
        / f"block_{block_index:02d}.npy"
    )


def _save_state_arrays(
    config: dict[str, Any],
    *,
    model_key: str,
    language: str,
    arrays_by_state_block: dict[tuple[str, int], np.ndarray],
    dtype_label: str,
    cache_dtype: np.dtype[Any],
    n_blocks: int,
    row_index_source: Path,
    source_artifact: Path,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for block_index in range(n_blocks):
        block_depth_norm = 0.0 if n_blocks <= 1 else float(block_index / (n_blocks - 1))
        for state in VALID_STATES:
            key = (state, block_index)
            if key not in arrays_by_state_block:
                continue
            output_path = _array_output_path(
                config,
                model_key=model_key,
                language=language,
                state=state,
                block_index=block_index,
            )
            output_path.parent.mkdir(parents=True, exist_ok=True)
            cache_array = arrays_by_state_block[key].astype(cache_dtype, copy=False)
            np.save(output_path, cache_array)
            rows.append(
                {
                    "model": model_key,
                    "language": language,
                    "block": block_index,
                    "block_depth_norm": block_depth_norm,
                    "state": state,
                    "pooling": "sentence_mean",
                    "n_rows": int(cache_array.shape[0]),
                    "hidden_size": int(cache_array.shape[1]),
                    "dtype": dtype_label,
                    "source_array_path": str(output_path),
                    "row_index_source": str(row_index_source),
                    "source_artifact": str(source_artifact),
                }
            )
    return rows


def _merge_on_scope(
    existing: pd.DataFrame,
    incoming: pd.DataFrame,
    *,
    scope_columns: Sequence[str],
) -> pd.DataFrame:
    if existing.empty:
        return incoming.copy()

    scope_rows = incoming.loc[:, list(scope_columns)].drop_duplicates().copy()
    merged = existing.merge(scope_rows.assign(_drop=True), on=list(scope_columns), how="left")
    merged = merged.loc[merged["_drop"].isna()].drop(columns=["_drop"])
    return pd.concat([merged, incoming], ignore_index=True)


def _write_manifest_with_merge(
    path: Path,
    incoming: pd.DataFrame,
    *,
    scope_columns: Sequence[str],
    sort_columns: Sequence[str],
) -> pd.DataFrame:
    if path.exists():
        existing = pd.read_parquet(path).copy()
        merged = _merge_on_scope(existing, incoming, scope_columns=scope_columns)
    else:
        merged = incoming.copy()
    merged = merged.sort_values(list(sort_columns)).reset_index(drop=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(path, index=False)
    return merged


def _write_state_adapter_notes(config: dict[str, Any], *, artifact_tag: str | None = None) -> Path:
    note_path = _state_adapter_note_path(config, artifact_tag)
    note_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# State Adapter Notes",
        "",
        f"- generated_utc: `{_now_utc()}`",
        "",
        "## XLM-R / Roberta",
        "",
        "- `INPUT`: encoder-layer forward pre-hook",
        "- `ATTN`: `layer.attention.output.dropout` output, i.e. the attention-side delta before residual normalization",
        "- `POST_ATTN`: `layer.attention` output, i.e. post-attention residual state",
        "- `FFN`: `layer.output.dropout` output, i.e. FFN-side delta before residual normalization",
        "- `OUTPUT`: encoder-layer forward output",
        "",
        "## NLLB / M2M100 encoder",
        "",
        "- `INPUT`: encoder-layer forward pre-hook",
        "- `ATTN`: `layer.self_attn` output, which is the attention branch update in eval mode",
        "- `POST_ATTN`: `layer.final_layer_norm` forward pre-hook input, i.e. the post-attention residual state",
        "- `FFN`: `layer.fc2` output, which is the FFN branch update in eval mode",
        "- `OUTPUT`: encoder-layer forward output",
        "",
        "## Shared pooling rule",
        "",
        "- remove special tokens and padding with `attention_mask & ~special_tokens_mask`",
        "- mean-pool over the remaining tokens",
        "- row-wise L2 normalize pooled sentence vectors when configured",
        "",
    ]
    note_path.write_text("\n".join(lines), encoding="utf-8")
    return note_path


def extract_internal_states(
    config: dict[str, Any],
    *,
    models: Sequence[str] = ("xlmr", "nllb"),
    languages: Sequence[str] = ("EN", "FR", "ZH"),
    states: Iterable[str] | None = None,
    max_tokens_per_batch: int = 2048,
    cache_dtype_override: str | None = None,
    artifact_tag: str | None = None,
) -> dict[str, Path]:
    triplets_path = _triplets_path(config)
    if not triplets_path.exists():
        raise FileNotFoundError(f"Triplet manifest missing: {triplets_path}")

    triplets = pd.read_parquet(triplets_path).sort_values("triplet_id").reset_index(drop=True)
    triplets["triplet_row_index"] = np.arange(len(triplets), dtype=np.int64)

    selected_languages = _select_languages(languages)
    requested_states = _select_states(config, states)
    cache_dtype, dtype_label = _cache_dtype(config, cache_dtype_override)
    normalize = str(config.get("features", {}).get("normalize", "l2")).lower() == "l2"
    eps = float(config.get("features", {}).get("eps", 1.0e-8))

    prior_models_root = str(Path(config["paths"]["prior_models_root"]).resolve())
    manifest_rows: list[dict[str, Any]] = []
    token_metadata_rows: list[dict[str, Any]] = []
    qc_accumulator: dict[tuple[str, str, int], dict[str, float]] = {}
    offset_mapping_fallbacks: list[str] = []

    for model_key in models:
        bundle = _load_model_bundle(model_key, prior_models_root)
        for language in selected_languages:
            text_column = LANGUAGE_TEXT_COLUMNS[language]
            texts = triplets[text_column].astype(str).tolist()
            length_kwargs = _tokenizer_kwargs_for_lengths(
                model_key=model_key,
                tokenizer=bundle.tokenizer,
                language=language,
            )
            token_lengths = [len(ids) for ids in bundle.tokenizer(texts, **length_kwargs)["input_ids"]]
            batch_groups = _batch_index_groups(token_lengths, max_tokens_per_batch=max_tokens_per_batch)
            arrays_by_state_block = _empty_cache_arrays(
                requested_states=requested_states,
                n_blocks=bundle.n_blocks,
                n_rows=len(triplets),
                hidden_size=bundle.hidden_size,
            )

            for batch_indices in batch_groups:
                batch_triplets = triplets.iloc[batch_indices].copy().reset_index(drop=True)
                batch_texts = batch_triplets[text_column].astype(str).tolist()
                batch_kwargs = _tokenizer_kwargs_for_batch(
                    model_key=model_key,
                    tokenizer=bundle.tokenizer,
                    language=language,
                )
                try:
                    encoded = bundle.tokenizer(batch_texts, **batch_kwargs)
                    offset_mapping = encoded.pop("offset_mapping", None)
                except (NotImplementedError, ValueError):
                    batch_kwargs.pop("return_offsets_mapping", None)
                    encoded = bundle.tokenizer(batch_texts, **batch_kwargs)
                    offset_mapping = None
                if offset_mapping is None and bundle.tokenizer.is_fast:
                    offset_mapping_fallbacks.append(f"{model_key}:{language}")
                encoded_tensors = {
                    key: value.to(bundle.device)
                    for key, value in encoded.items()
                    if hasattr(value, "to")
                }
                pooling_mask = encoded_tensors["attention_mask"].bool() & ~encoded_tensors["special_tokens_mask"].bool()

                state_tensors, outputs = run_model_forward_with_state_capture(
                    model_key=model_key,
                    model=bundle.model,
                    input_ids=encoded_tensors["input_ids"],
                    attention_mask=encoded_tensors["attention_mask"],
                    requested_states=requested_states,
                )

                for block_index in range(bundle.n_blocks):
                    captured_output = state_tensors.get(("OUTPUT", block_index))
                    if captured_output is None:
                        continue
                    difference = (captured_output - outputs.hidden_states[block_index + 1].detach()).abs()
                    key = (model_key, language, block_index)
                    stats = qc_accumulator.setdefault(
                        key,
                        {"abs_sum": 0.0, "n_values": 0.0, "abs_max": 0.0, "n_batches": 0.0},
                    )
                    stats["abs_sum"] += float(difference.sum().item())
                    stats["n_values"] += float(difference.numel())
                    stats["abs_max"] = max(stats["abs_max"], float(difference.max().item()))
                    stats["n_batches"] += 1.0

                for (state, block_index), tensor in state_tensors.items():
                    pooled = _pool_state_tensor(tensor, pooling_mask, normalize=normalize, eps=eps)
                    destination_rows = batch_triplets["triplet_row_index"].to_numpy(dtype=np.int64)
                    arrays_by_state_block[(state, block_index)][destination_rows] = pooled

                for row_offset, row in enumerate(batch_triplets.itertuples(index=False)):
                    sequence_length = int(encoded_tensors["attention_mask"][row_offset].sum().item())
                    pooled_positions = torch.nonzero(pooling_mask[row_offset], as_tuple=False).flatten()
                    first_idx = int(pooled_positions[0].item()) if len(pooled_positions) else -1
                    last_idx = int(pooled_positions[-1].item()) if len(pooled_positions) else -1
                    token_ids = encoded_tensors["input_ids"][row_offset, :sequence_length].detach().cpu().tolist()
                    token_strings = bundle.tokenizer.convert_ids_to_tokens(token_ids)
                    row_offsets = offset_mapping[row_offset][:sequence_length].tolist() if offset_mapping is not None else []
                    token_metadata_rows.append(
                        {
                            "model": model_key,
                            "language": language,
                            "triplet_id": int(row.triplet_id),
                            "text": getattr(row, text_column),
                            "sequence_length": sequence_length,
                            "n_tokens_pooled": int(pooling_mask[row_offset].sum().item()),
                            "first_pooled_token_index": first_idx,
                            "last_pooled_token_index": last_idx,
                            "token_ids_json": json.dumps(token_ids),
                            "token_strings_json": json.dumps(token_strings, ensure_ascii=False),
                            "char_offsets_json": json.dumps(row_offsets, ensure_ascii=False),
                            "pooled_token_indices_json": json.dumps(pooled_positions.detach().cpu().tolist()),
                            "has_offset_mapping": bool(offset_mapping is not None),
                            "source_model_dir": str(bundle.model_dir),
                        }
                    )

            manifest_rows.extend(
                _save_state_arrays(
                    config,
                    model_key=model_key,
                    language=language,
                    arrays_by_state_block=arrays_by_state_block,
                    dtype_label=dtype_label,
                    cache_dtype=cache_dtype,
                    n_blocks=bundle.n_blocks,
                    row_index_source=triplets_path,
                    source_artifact=bundle.model_dir,
                )
            )

    state_manifest_df = pd.DataFrame(manifest_rows)
    state_manifest_path = _state_cache_manifest_path(config, artifact_tag)
    merged_manifest_df = _write_manifest_with_merge(
        state_manifest_path,
        state_manifest_df,
        scope_columns=("model", "language", "state", "block"),
        sort_columns=("model", "language", "state", "block"),
    )

    output_manifest_df = merged_manifest_df.loc[merged_manifest_df["state"] == "OUTPUT"].copy()
    output_manifest_df = output_manifest_df.loc[
        :,
        [
            "model",
            "language",
            "block",
            "block_depth_norm",
            "state",
            "n_rows",
            "hidden_size",
            "dtype",
            "source_array_path",
            "source_artifact",
        ],
    ]
    output_manifest_path = _output_state_manifest_path(config, artifact_tag)
    output_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    output_manifest_df.to_parquet(output_manifest_path, index=False)

    token_metadata_df = pd.DataFrame(token_metadata_rows)
    token_metadata_path = _token_metadata_path(config, artifact_tag)
    _write_manifest_with_merge(
        token_metadata_path,
        token_metadata_df,
        scope_columns=("model", "language", "triplet_id"),
        sort_columns=("model", "language", "triplet_id"),
    )

    triplet_ids_path = _triplet_ids_path(config, artifact_tag)
    triplet_ids_path.parent.mkdir(parents=True, exist_ok=True)
    triplets.loc[:, ["triplet_id"]].to_parquet(triplet_ids_path, index=False)

    qc_rows = [
        {
            "model": model_key,
            "language": language,
            "block": block_index,
            "state": "OUTPUT",
            "mean_abs_diff": stats["abs_sum"] / stats["n_values"] if stats["n_values"] > 0 else 0.0,
            "max_abs_diff": stats["abs_max"],
            "n_batches": int(stats["n_batches"]),
        }
        for (model_key, language, block_index), stats in sorted(qc_accumulator.items())
    ]
    qc_df = pd.DataFrame(qc_rows)
    qc_path = _state_extraction_qc_path(config, artifact_tag)
    qc_path.parent.mkdir(parents=True, exist_ok=True)
    qc_df.to_parquet(qc_path, index=False)

    adapter_note_path = _write_state_adapter_notes(config, artifact_tag=artifact_tag)
    provenance_path = _state_extraction_note_path(config, artifact_tag)
    provenance_path.parent.mkdir(parents=True, exist_ok=True)
    provenance_lines = [
        "# State Extraction",
        "",
        f"- generated_utc: `{_now_utc()}`",
        f"- triplet_manifest: `{triplets_path}`",
        f"- state_cache_manifest: `{state_manifest_path}`",
        f"- output_state_manifest: `{output_manifest_path}`",
        f"- token_metadata: `{token_metadata_path}`",
        f"- output_equivalence_qc: `{qc_path}`",
        f"- adapter_notes: `{adapter_note_path}`",
        f"- models: `{', '.join(models)}`",
        f"- languages: `{', '.join(selected_languages)}`",
        f"- requested_states: `{', '.join(requested_states)}`",
        f"- pooling: `sentence_mean`",
        f"- normalization: `{'l2' if normalize else 'none'}`",
        f"- cache_dtype: `{dtype_label}`",
        f"- max_tokens_per_batch: `{int(max_tokens_per_batch)}`",
        f"- artifact_tag: `{artifact_tag or 'canonical'}`",
        "",
        "## Notes",
        "",
        "- OUTPUT equivalence is checked against the model's returned hidden-state stack at every block.",
        "- Token metadata stores IDs, token strings, pooled-token indices, and character offsets when the tokenizer exposes them.",
    ]
    if offset_mapping_fallbacks:
        provenance_lines.extend(
            [
                "",
                "## Fallbacks",
                "",
                f"- Offset mappings were unavailable for: `{', '.join(sorted(set(offset_mapping_fallbacks)))}`.",
            ]
        )
    provenance_path.write_text("\n".join(provenance_lines), encoding="utf-8")

    return {
        "state_manifest": state_manifest_path,
        "output_manifest": output_manifest_path,
        "token_metadata": token_metadata_path,
        "triplet_ids": triplet_ids_path,
        "qc": qc_path,
        "provenance": provenance_path,
        "adapter_notes": adapter_note_path,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract pooled internal transformer states for the sequel pipeline.")
    parser.add_argument("--config", default="conf/base.yaml", help="Path to sequel config.")
    parser.add_argument("--models", nargs="+", default=["xlmr", "nllb"], choices=["xlmr", "nllb"])
    parser.add_argument("--languages", nargs="+", default=["en", "fr", "zh"], choices=["en", "fr", "zh", "EN", "FR", "ZH"])
    parser.add_argument("--states", nargs="+", default=None, choices=list(VALID_STATES))
    parser.add_argument("--max-tokens-per-batch", type=int, default=2048)
    parser.add_argument("--cache-dtype", default=None, choices=["float16", "float32", "fp16", "fp32"])
    parser.add_argument("--artifact-tag", default=None, help="Optional shard tag for metadata/provenance outputs.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    outputs = extract_internal_states(
        config,
        models=tuple(args.models),
        languages=tuple(args.languages),
        states=None if args.states is None else tuple(args.states),
        max_tokens_per_batch=args.max_tokens_per_batch,
        cache_dtype_override=args.cache_dtype,
        artifact_tag=args.artifact_tag,
    )
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
