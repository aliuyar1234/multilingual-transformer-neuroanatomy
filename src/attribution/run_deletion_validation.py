from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import UTC, datetime
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import regex

from src.attribution.compute_word_attributions import (
    LANGUAGE_TEXT_COLUMNS,
    _append_fallback,
    _delete_span,
    _fit_fold_models,
    _load_state_array_lookup,
    _load_triplets,
    _pearson_r,
    _prediction_contribution,
    _prepare_subject_family,
    _run_mutation_encoder,
    _word_attributions_path,
)
from src.stats.run_primary_tests import _project_seed, _stable_seed_offset
from src.utils.config import load_config


DELETION_VALIDATION_COLUMNS = [
    "subject_id",
    "model",
    "language",
    "roi_family",
    "representative_block",
    "representative_state",
    "k",
    "base_z",
    "topk_z",
    "random_z_mean",
    "delta_topk",
    "delta_random",
    "delta_diff",
    "n_random_draws",
    "matching_policy",
]
NON_PUNCT_TOKEN_CLASSES = {"CONTENT", "FUNCTION", "EDGE"}


@dataclass(slots=True)
class SettingMutationCache:
    config: dict[str, Any]
    model: str
    state: str
    block: int
    batch_size: int
    vectors_by_language: dict[str, dict[str, np.ndarray]]

    def prime(self, sentence_language: str, texts: list[str]) -> None:
        language = str(sentence_language)
        language_cache = self.vectors_by_language.setdefault(language, {})
        missing = sorted({str(text) for text in texts if str(text) not in language_cache})
        if not missing:
            return
        encoded = _run_mutation_encoder_batched(
            self.config,
            model=self.model,
            language=language,
            state=self.state,
            block=self.block,
            texts=missing,
            batch_size=self.batch_size,
        )
        for text, vector in zip(missing, encoded, strict=False):
            language_cache[str(text)] = np.asarray(vector, dtype=np.float32)

    def vector(self, sentence_language: str, text: str) -> np.ndarray:
        return np.asarray(
            self.vectors_by_language[str(sentence_language)][str(text)],
            dtype=np.float32,
        )


def _local_outputs_root(config: dict[str, Any]) -> Path:
    return Path(config["paths"]["local_outputs_root"]).resolve()


def _deletion_validation_path(config: dict[str, Any]) -> Path:
    return _local_outputs_root(config) / "subject_results" / "deletion_validation.parquet"


def _provenance_path(config: dict[str, Any]) -> Path:
    return _local_outputs_root(config) / "provenance" / "deletion_validation.md"


def _deletion_shards_root(config: dict[str, Any]) -> Path:
    return _local_outputs_root(config) / "subject_results" / "deletion_validation_shards"


def _deletion_progress_logs_root(config: dict[str, Any]) -> Path:
    return _local_outputs_root(config) / "provenance" / "deletion_progress_logs"


def _slug(value: str) -> str:
    normalized = regex.sub(r"[^A-Za-z0-9]+", "_", str(value).strip())
    normalized = normalized.strip("_")
    return normalized.lower() or "na"


def _now_utc() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _deletion_shard_path(
    config: dict[str, Any],
    *,
    model: str,
    target_language: str,
    roi_family: str,
    block: int,
    state: str,
) -> Path:
    filename = (
        f"deletion_validation__{_slug(model)}__{_slug(target_language)}__{_slug(roi_family)}"
        f"__b{int(block):02d}__{_slug(state)}.parquet"
    )
    return _deletion_shards_root(config) / filename


def _deletion_shard_metadata_path(
    config: dict[str, Any],
    *,
    model: str,
    target_language: str,
    roi_family: str,
    block: int,
    state: str,
) -> Path:
    shard_path = _deletion_shard_path(
        config,
        model=model,
        target_language=target_language,
        roi_family=roi_family,
        block=block,
        state=state,
    )
    return shard_path.with_name(f"{shard_path.stem}__metadata.json")


def _deletion_progress_log_path(
    config: dict[str, Any],
    *,
    model: str,
    target_language: str,
    roi_family: str,
    block: int,
    state: str,
) -> Path:
    root = _deletion_progress_logs_root(config)
    root.mkdir(parents=True, exist_ok=True)
    filename = (
        f"deletion_validation__{_slug(model)}__{_slug(target_language)}__{_slug(roi_family)}"
        f"__b{int(block):02d}__{_slug(state)}.jsonl"
    )
    return root / filename


def _write_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=True) + "\n")


def _setting_label(*, model: str, target_language: str, roi_family: str, block: int, state: str) -> str:
    return f"{model}/{target_language}/{roi_family}/b{int(block):02d}/{state}"


def _empty_deletion_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=DELETION_VALIDATION_COLUMNS)


def _frame_from_setting_rows(
    setting_rows_by_key: dict[tuple[str, int], dict[str, object]],
) -> pd.DataFrame:
    if not setting_rows_by_key:
        return _empty_deletion_frame()
    frame = pd.DataFrame(list(setting_rows_by_key.values()), columns=DELETION_VALIDATION_COLUMNS)
    return frame.sort_values(
        ["subject_id", "k", "model", "language", "roi_family", "representative_block", "representative_state"]
    ).reset_index(drop=True)


def _setting_rows_by_key_from_frame(frame: pd.DataFrame) -> dict[tuple[str, int], dict[str, object]]:
    rows_by_key: dict[tuple[str, int], dict[str, object]] = {}
    if frame.empty:
        return rows_by_key
    for row in frame.itertuples(index=False):
        rows_by_key[(str(row.subject_id), int(row.k))] = {
            column: getattr(row, column)
            for column in DELETION_VALIDATION_COLUMNS
        }
    return rows_by_key


def _persist_setting_checkpoint(
    *,
    shard_path: Path,
    metadata_path: Path,
    setting_rows_by_key: dict[tuple[str, int], dict[str, object]],
    metadata: dict[str, Any],
) -> pd.DataFrame:
    frame = _frame_from_setting_rows(setting_rows_by_key)
    metadata["rows_written"] = int(len(frame))
    metadata["updated_utc"] = _now_utc()
    _write_parquet(frame, shard_path)
    _write_json(metadata_path, metadata)
    return frame


def _progress_payload(
    *,
    event: str,
    model: str,
    target_language: str,
    roi_family: str,
    block: int,
    state: str,
    **extra: Any,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "timestamp_utc": _now_utc(),
        "event": str(event),
        "model": str(model),
        "target_language": str(target_language),
        "roi_family": str(roi_family),
        "block": int(block),
        "state": str(state),
        "setting_label": _setting_label(
            model=model,
            target_language=target_language,
            roi_family=roi_family,
            block=block,
            state=state,
        ),
    }
    payload.update(extra)
    return payload


def _emit_progress(
    log_path: Path,
    *,
    event: str,
    model: str,
    target_language: str,
    roi_family: str,
    block: int,
    state: str,
    console_message: str | None = None,
    **extra: Any,
) -> None:
    payload = _progress_payload(
        event=event,
        model=model,
        target_language=target_language,
        roi_family=roi_family,
        block=block,
        state=state,
        **extra,
    )
    _append_jsonl(log_path, payload)
    if console_message:
        print(console_message, flush=True)


def _stable_request_offset(*parts: str) -> int:
    return _stable_seed_offset(*parts)


def _load_word_attributions(config: dict[str, Any]) -> pd.DataFrame:
    path = _word_attributions_path(config)
    if not path.exists():
        raise FileNotFoundError(
            f"Word attributions missing: {path}. Run src.attribution.compute_word_attributions first."
        )
    df = pd.read_parquet(path).copy()
    if df.empty:
        raise RuntimeError("Word attribution table is empty; deletion validation cannot proceed.")
    return df


def _adaptive_k3_config(config: dict[str, Any]) -> dict[str, Any]:
    return dict(config.get("attribution", {}).get("adaptive_k3", {}))


def _should_run_k(
    *,
    k: int,
    prior_rows_by_k: dict[int, dict[str, object]],
    adaptive_cfg: dict[str, Any],
) -> tuple[bool, str]:
    if int(k) != 3 or not bool(adaptive_cfg.get("enabled", True)):
        return True, "enabled"
    if 2 not in prior_rows_by_k:
        return False, "adaptive_k3_requires_k2"
    min_gain = float(adaptive_cfg.get("min_delta_diff_gain", 0.0))
    previous_delta = float(prior_rows_by_k[2]["delta_diff"])
    baseline_delta = float(prior_rows_by_k.get(1, {}).get("delta_diff", 0.0))
    if previous_delta - baseline_delta <= min_gain:
        return False, "adaptive_k3_skipped_no_incremental_gain"
    return True, "adaptive_k3_triggered"


def _text_for_triplet(triplet_row: pd.Series, language: str) -> str:
    return str(triplet_row[LANGUAGE_TEXT_COLUMNS[str(language)]])


def _delete_multiple_spans(
    text: str,
    spans: list[tuple[int, int]],
    *,
    language: str,
) -> str:
    mutated = str(text)
    for start, end in sorted(spans, key=lambda item: (int(item[0]), int(item[1])), reverse=True):
        mutated = _delete_span(mutated, int(start), int(end), language=language)
    if language in {"EN", "FR"}:
        mutated = regex.sub(r"\s+", " ", mutated).strip()
    return mutated


def _row_keys(frame: pd.DataFrame) -> set[tuple[str, int, int, int]]:
    return {
        (
            str(row.sentence_language),
            int(row.word_index),
            int(row.word_char_start),
            int(row.word_char_end),
        )
        for row in frame.itertuples(index=False)
    }


def _sample_without_replacement(
    frame: pd.DataFrame,
    *,
    n: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    if n <= 0:
        return frame.iloc[0:0].copy()
    chosen = rng.choice(frame.index.to_numpy(dtype=np.int64), size=int(n), replace=False)
    return frame.loc[np.asarray(chosen, dtype=np.int64)].copy()


def _exact_language_class_match(
    pool: pd.DataFrame,
    selected_topk: pd.DataFrame,
    *,
    rng: np.random.Generator,
) -> pd.DataFrame | None:
    pieces: list[pd.DataFrame] = []
    grouped = selected_topk.groupby(["sentence_language", "token_class"], as_index=False).size()
    for row in grouped.itertuples(index=False):
        subset = pool.loc[
            (pool["sentence_language"].astype(str) == str(row.sentence_language))
            & (pool["token_class"].astype(str) == str(row.token_class))
        ].copy()
        if len(subset) < int(row.size):
            return None
        pieces.append(_sample_without_replacement(subset, n=int(row.size), rng=rng))
    if not pieces:
        return selected_topk.iloc[0:0].copy()
    return pd.concat(pieces, ignore_index=False).reset_index(drop=True)


def _language_allocation_match(
    pool: pd.DataFrame,
    selected_topk: pd.DataFrame,
    *,
    rng: np.random.Generator,
    prefer_non_punct: bool,
) -> pd.DataFrame | None:
    pieces: list[pd.DataFrame] = []
    grouped = selected_topk.groupby("sentence_language", as_index=False).size()
    for row in grouped.itertuples(index=False):
        subset = pool.loc[pool["sentence_language"].astype(str) == str(row.sentence_language)].copy()
        if len(subset) < int(row.size):
            return None
        if prefer_non_punct:
            preferred = subset.loc[subset["token_class"].astype(str).isin(NON_PUNCT_TOKEN_CLASSES)].copy()
            if len(preferred) >= int(row.size):
                pieces.append(_sample_without_replacement(preferred, n=int(row.size), rng=rng))
                continue
        pieces.append(_sample_without_replacement(subset, n=int(row.size), rng=rng))
    if not pieces:
        return selected_topk.iloc[0:0].copy()
    return pd.concat(pieces, ignore_index=False).reset_index(drop=True)


def _total_count_match(
    pool: pd.DataFrame,
    *,
    n: int,
    rng: np.random.Generator,
    prefer_non_punct: bool,
) -> pd.DataFrame | None:
    subset = pool.copy()
    if prefer_non_punct:
        preferred = subset.loc[subset["token_class"].astype(str).isin(NON_PUNCT_TOKEN_CLASSES)].copy()
        if len(preferred) >= int(n):
            subset = preferred
    if len(subset) < int(n):
        return None
    return _sample_without_replacement(subset, n=int(n), rng=rng).reset_index(drop=True)


def _matched_random_selection(
    candidates: pd.DataFrame,
    selected_topk: pd.DataFrame,
    *,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, str]:
    top_keys = _row_keys(selected_topk)
    pool_without_overlap = candidates.loc[
        ~candidates.apply(
            lambda row: (
                str(row["sentence_language"]),
                int(row["word_index"]),
                int(row["word_char_start"]),
                int(row["word_char_end"]),
            )
            in top_keys,
            axis=1,
        )
    ].copy()
    n_selected = int(len(selected_topk))

    exact = _exact_language_class_match(pool_without_overlap, selected_topk, rng=rng)
    if exact is not None:
        return exact, "exact_language_token_class_no_overlap"

    language_only = _language_allocation_match(
        pool_without_overlap,
        selected_topk,
        rng=rng,
        prefer_non_punct=True,
    )
    if language_only is not None:
        return language_only, "language_allocation_nonpunct_no_overlap"

    language_only_overlap = _language_allocation_match(
        candidates,
        selected_topk,
        rng=rng,
        prefer_non_punct=True,
    )
    if language_only_overlap is not None:
        return language_only_overlap, "language_allocation_nonpunct_overlap_allowed"

    total_non_punct = _total_count_match(
        pool_without_overlap,
        n=n_selected,
        rng=rng,
        prefer_non_punct=True,
    )
    if total_non_punct is not None:
        return total_non_punct, "total_count_nonpunct_no_overlap"

    total_any = _total_count_match(
        candidates,
        n=n_selected,
        rng=rng,
        prefer_non_punct=False,
    )
    if total_any is not None:
        return total_any, "total_count_any_overlap_allowed"

    return selected_topk.copy(), "fallback_topk_reused"


def _select_topk_by_triplet(
    attribution_df: pd.DataFrame,
    *,
    k: int,
) -> dict[int, pd.DataFrame]:
    selections: dict[int, pd.DataFrame] = {}
    for triplet_row_index, triplet_df in attribution_df.groupby("triplet_row_index", sort=False):
        ordered = triplet_df.sort_values(
            ["attribution_score", "sentence_language", "word_char_start", "word_index"],
            ascending=[False, True, True, True],
        ).reset_index(drop=True)
        selected = ordered.head(int(k)).copy()
        if not selected.empty:
            selections[int(triplet_row_index)] = selected
    return selections


def _build_mutation_requests(
    triplets: pd.DataFrame,
    selections: dict[int, pd.DataFrame],
) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    triplet_lookup = triplets.set_index("triplet_row_index", drop=False)
    requests: list[dict[str, Any]] = []
    language_requests: dict[str, list[dict[str, Any]]] = {}
    for triplet_row_index, selected in selections.items():
        triplet_row = triplet_lookup.loc[int(triplet_row_index)]
        if isinstance(triplet_row, pd.DataFrame):
            triplet_row = triplet_row.iloc[0]
        request_id = f"triplet_{int(triplet_row_index)}"
        request_languages: list[str] = []
        for sentence_language, sentence_rows in selected.groupby("sentence_language", sort=False):
            original_text = _text_for_triplet(triplet_row, str(sentence_language))
            spans = [
                (int(row.word_char_start), int(row.word_char_end))
                for row in sentence_rows.itertuples(index=False)
            ]
            mutated_text = _delete_multiple_spans(
                original_text,
                spans,
                language=str(sentence_language),
            )
            language_requests.setdefault(str(sentence_language), []).append(
                {
                    "request_id": request_id,
                    "triplet_row_index": int(triplet_row_index),
                    "sentence_language": str(sentence_language),
                    "mutated_text": mutated_text,
                }
            )
            request_languages.append(str(sentence_language))
        requests.append(
            {
                "request_id": request_id,
                "triplet_row_index": int(triplet_row_index),
                "sentence_languages": tuple(request_languages),
            }
        )
    return requests, language_requests


def _run_mutation_encoder_batched(
    config: dict[str, Any],
    *,
    model: str,
    language: str,
    state: str,
    block: int,
    texts: list[str],
    batch_size: int,
) -> np.ndarray:
    if not texts:
        return np.zeros((0, 0), dtype=np.float32)
    chunks: list[np.ndarray] = []
    for start in range(0, len(texts), int(batch_size)):
        stop = min(len(texts), start + int(batch_size))
        chunks.append(
            _run_mutation_encoder(
                config,
                model=model,
                language=language,
                state=state,
                block=block,
                texts=texts[start:stop],
            )
        )
    return np.vstack(chunks).astype(np.float32, copy=False)


def _build_subject_mutation_plan(
    *,
    subject_df: pd.DataFrame,
    triplets: pd.DataFrame,
    deletion_ks: list[int],
    n_random_draws: int,
    seed: int,
    model: str,
    target_language: str,
    roi_family: str,
    subject_id: str,
) -> tuple[dict[int, dict[str, Any]], dict[str, int]]:
    plans_by_k: dict[int, dict[str, Any]] = {}
    fallback_counts: dict[str, int] = {}
    for k in sorted(int(value) for value in deletion_ks):
        topk_selections = _select_topk_by_triplet(subject_df, k=int(k))
        if not topk_selections:
            continue
        topk_requests, topk_language_requests = _build_mutation_requests(triplets, topk_selections)
        rng = np.random.default_rng(
            int(seed) + _stable_request_offset(model, target_language, roi_family, str(subject_id), str(k))
        )
        random_draws: list[dict[str, Any]] = []
        for _ in range(int(n_random_draws)):
            random_selections: dict[int, pd.DataFrame] = {}
            draw_policies: set[str] = set()
            for triplet_row_index, triplet_df in subject_df.groupby("triplet_row_index", sort=False):
                ordered = triplet_df.sort_values(
                    ["attribution_score", "sentence_language", "word_char_start", "word_index"],
                    ascending=[False, True, True, True],
                ).reset_index(drop=True)
                selected_topk = ordered.head(int(k)).copy()
                if selected_topk.empty:
                    continue
                random_selected, policy = _matched_random_selection(
                    ordered,
                    selected_topk,
                    rng=rng,
                )
                random_selections[int(triplet_row_index)] = random_selected
                draw_policies.add(str(policy))
                fallback_counts[str(policy)] = fallback_counts.get(str(policy), 0) + 1
            if not random_selections:
                continue
            requests, language_requests = _build_mutation_requests(triplets, random_selections)
            random_draws.append(
                {
                    "requests": requests,
                    "language_requests": language_requests,
                    "policy_label": "+".join(sorted(draw_policies)),
                }
            )
        plans_by_k[int(k)] = {
            "topk_requests": topk_requests,
            "topk_language_requests": topk_language_requests,
            "random_draws": random_draws,
        }
    return plans_by_k, fallback_counts


def _mutated_target_vectors(
    config: dict[str, Any],
    *,
    model: str,
    target_language: str,
    state: str,
    block: int,
    state_arrays: dict[str, np.ndarray],
    requests: list[dict[str, Any]],
    language_requests: dict[str, list[dict[str, Any]]],
    batch_size: int,
    mutation_vector_cache: SettingMutationCache | None = None,
) -> dict[int, np.ndarray]:
    source_languages = [language for language in LANGUAGE_TEXT_COLUMNS if language != str(target_language)]
    encoded_lookup: dict[tuple[str, str], np.ndarray] = {}
    for sentence_language, language_rows in language_requests.items():
        if mutation_vector_cache is not None:
            mutation_vector_cache.prime(
                str(sentence_language),
                [str(row["mutated_text"]) for row in language_rows],
            )
            for row in language_rows:
                encoded_lookup[(str(row["request_id"]), str(sentence_language))] = mutation_vector_cache.vector(
                    str(sentence_language),
                    str(row["mutated_text"]),
                )
            continue
        encoded = _run_mutation_encoder_batched(
            config,
            model=model,
            language=str(sentence_language),
            state=state,
            block=block,
            texts=[str(row["mutated_text"]) for row in language_rows],
            batch_size=batch_size,
        )
        for row, vector in zip(language_rows, encoded, strict=False):
            encoded_lookup[(str(row["request_id"]), str(sentence_language))] = np.asarray(vector, dtype=np.float32)

    mutated_vectors: dict[int, np.ndarray] = {}
    for request in requests:
        triplet_row_index = int(request["triplet_row_index"])
        per_language: list[np.ndarray] = []
        for source_language in source_languages:
            encoded = encoded_lookup.get((str(request["request_id"]), str(source_language)))
            if encoded is None:
                encoded = state_arrays[str(source_language)][triplet_row_index]
            per_language.append(np.asarray(encoded, dtype=np.float32))
        mutated_vectors[int(triplet_row_index)] = (
            (per_language[0] + per_language[1]) / 2.0
        ).astype(np.float32, copy=False)
    return mutated_vectors


def _build_base_prediction_cache(
    fold_models: dict[int, Any],
    *,
    shared_array: np.ndarray,
) -> tuple[dict[int, dict[int, np.ndarray]], float]:
    contribution_cache: dict[int, dict[int, np.ndarray]] = {}
    z_scores: list[float] = []
    for run_index, fold in fold_models.items():
        contribution_cache[int(run_index)] = {}
        for triplet_row_index in fold.row_to_local_index:
            contribution_cache[int(run_index)][int(triplet_row_index)] = _prediction_contribution(
                fold,
                triplet_row_index=int(triplet_row_index),
                feature_vector=shared_array[int(triplet_row_index)],
            )
        z_scores.append(float(np.arctanh(_pearson_r(fold.actual, fold.base_prediction))))
    return contribution_cache, float(np.mean(z_scores)) if z_scores else 0.0


def _warm_start_mutation_encoding(
    config: dict[str, Any],
    *,
    model: str,
    target_language: str,
    state: str,
    block: int,
    triplets: pd.DataFrame,
    batch_size: int,
) -> None:
    if triplets.empty:
        return
    first_triplet = triplets.iloc[0]
    for sentence_language in LANGUAGE_TEXT_COLUMNS:
        if str(sentence_language) == str(target_language):
            continue
        text = str(first_triplet[LANGUAGE_TEXT_COLUMNS[str(sentence_language)]])
        if not text:
            continue
        _run_mutation_encoder_batched(
            config,
            model=model,
            language=str(sentence_language),
            state=state,
            block=block,
            texts=[text],
            batch_size=max(1, int(batch_size)),
        )


def _score_mutated_predictions(
    fold_models: dict[int, Any],
    *,
    base_contributions: dict[int, dict[int, np.ndarray]],
    mutated_vectors: dict[int, np.ndarray],
    triplet_to_run: dict[int, int],
) -> float:
    z_scores: list[float] = []
    mutated_predictions = {
        int(run_index): fold.base_prediction.astype(np.float32, copy=True)
        for run_index, fold in fold_models.items()
    }
    for triplet_row_index, mutated_vector in mutated_vectors.items():
        run_index = int(triplet_to_run[int(triplet_row_index)])
        fold = fold_models[int(run_index)]
        base = base_contributions[int(run_index)][int(triplet_row_index)]
        updated = _prediction_contribution(
            fold,
            triplet_row_index=int(triplet_row_index),
            feature_vector=np.asarray(mutated_vector, dtype=np.float32),
        )
        mutated_predictions[int(run_index)] = mutated_predictions[int(run_index)] - base + updated
    for run_index, fold in fold_models.items():
        z_scores.append(float(np.arctanh(_pearson_r(fold.actual, mutated_predictions[int(run_index)]))))
    return float(np.mean(z_scores)) if z_scores else 0.0


def _evaluate_subject_setting(
    config: dict[str, Any],
    *,
    model: str,
    target_language: str,
    roi_family: str,
    block: int,
    state: str,
    prepared: Any,
    subject_df: pd.DataFrame,
    triplets: pd.DataFrame,
    triplet_to_run: dict[int, int],
    state_arrays: dict[str, np.ndarray],
    shared_array: np.ndarray,
    alpha_values: np.ndarray,
    max_pcs: int,
    variance_threshold: float,
    deletion_ks: list[int],
    adaptive_k3_cfg: dict[str, Any],
    n_random_draws: int,
    batch_size: int,
    seed: int,
    mutation_vector_cache: SettingMutationCache | None = None,
    existing_rows_by_k: dict[int, dict[str, object]] | None = None,
    row_callback: Callable[[dict[str, object]], None] | None = None,
) -> tuple[pd.DataFrame, dict[str, int]]:
    subject_id = str(prepared.subject_id)
    fold_models = _fit_fold_models(
        prepared,
        feature_array=shared_array,
        max_pcs=max_pcs,
        variance_threshold=variance_threshold,
        alpha_values=alpha_values,
        random_state=seed,
    )
    base_contributions, base_z = _build_base_prediction_cache(
        fold_models,
        shared_array=shared_array,
    )

    rows: list[dict[str, object]] = []
    plans_by_k, fallback_counts = _build_subject_mutation_plan(
        subject_df=subject_df,
        triplets=triplets,
        deletion_ks=deletion_ks,
        n_random_draws=n_random_draws,
        seed=seed,
        model=model,
        target_language=target_language,
        roi_family=roi_family,
        subject_id=subject_id,
    )
    rows_by_k: dict[int, dict[str, object]] = {
        int(k): dict(row)
        for k, row in (existing_rows_by_k or {}).items()
    }
    for k in sorted(int(value) for value in deletion_ks):
        plan = plans_by_k.get(int(k))
        if plan is None:
            continue
        if int(k) in rows_by_k:
            continue
        should_run, policy_label = _should_run_k(
            k=int(k),
            prior_rows_by_k=rows_by_k,
            adaptive_cfg=adaptive_k3_cfg,
        )
        if not should_run:
            fallback_counts[str(policy_label)] = fallback_counts.get(str(policy_label), 0) + 1
            continue
        topk_vectors = _mutated_target_vectors(
            config,
            model=model,
            target_language=target_language,
            state=state,
            block=block,
            state_arrays=state_arrays,
            requests=plan["topk_requests"],
            language_requests=plan["topk_language_requests"],
            batch_size=batch_size,
            mutation_vector_cache=mutation_vector_cache,
        )
        topk_z = _score_mutated_predictions(
            fold_models,
            base_contributions=base_contributions,
            mutated_vectors=topk_vectors,
            triplet_to_run=triplet_to_run,
        )

        random_draw_scores: list[float] = []
        policy_labels: list[str] = []
        for draw in plan["random_draws"]:
            mutated_vectors = _mutated_target_vectors(
                config,
                model=model,
                target_language=target_language,
                state=state,
                block=block,
                state_arrays=state_arrays,
                requests=draw["requests"],
                language_requests=draw["language_requests"],
                batch_size=batch_size,
                mutation_vector_cache=mutation_vector_cache,
            )
            random_draw_scores.append(
                _score_mutated_predictions(
                    fold_models,
                    base_contributions=base_contributions,
                    mutated_vectors=mutated_vectors,
                    triplet_to_run=triplet_to_run,
                )
            )
            policy_labels.append(str(draw["policy_label"]))

        if not random_draw_scores:
            continue

        random_z_mean = float(np.mean(random_draw_scores))
        delta_topk = float(base_z - topk_z)
        delta_random = float(base_z - random_z_mean)
        row = {
            "subject_id": str(subject_id),
            "model": model,
            "language": target_language,
            "roi_family": roi_family,
            "representative_block": block,
            "representative_state": state,
            "k": int(k),
            "base_z": float(base_z),
            "topk_z": float(topk_z),
            "random_z_mean": random_z_mean,
            "delta_topk": delta_topk,
            "delta_random": delta_random,
            "delta_diff": float(delta_topk - delta_random),
            "n_random_draws": int(len(random_draw_scores)),
            "matching_policy": ", ".join(sorted(set(policy_labels))),
        }
        rows.append(row)
        rows_by_k[int(k)] = row
        if row_callback is not None:
            row_callback(dict(row))
    return pd.DataFrame(rows, columns=DELETION_VALIDATION_COLUMNS), fallback_counts


def run_deletion_validation(
    config: dict[str, Any],
    *,
    models: list[str] | None = None,
    target_languages: list[str] | None = None,
    roi_families: list[str] | None = None,
    max_settings: int | None = None,
    resume_shards: bool = True,
) -> dict[str, Path]:
    triplets = _load_triplets(config)
    word_df = _load_word_attributions(config)
    subset_mode = any(
        (
            bool(models),
            bool(target_languages),
            bool(roi_families),
            max_settings is not None,
        )
    )
    triplet_to_run = {
        int(row.triplet_row_index): int(row.canonical_run)
        for row in triplets.itertuples(index=False)
    }
    deletion_cfg = config.get("attribution", {})
    deletion_ks = [int(value) for value in deletion_cfg.get("deletion_ks", [1, 2, 3])]
    adaptive_k3_cfg = _adaptive_k3_config(config)
    n_random_draws = int(deletion_cfg.get("matched_random_draws", 100))
    batch_size = int(deletion_cfg.get("mutation_batch_size", 256))
    alpha_values = np.logspace(-2.0, 6.0, 15).astype(np.float64)
    max_pcs = int(config.get("encoding", {}).get("max_pcs", 128))
    variance_threshold = float(config.get("encoding", {}).get("variance_threshold", 0.95))
    seed = _project_seed(config)
    representative_settings = (
        word_df.loc[word_df["mapping_status"].astype(str) == "OK", [
            "model",
            "target_language",
            "roi_family",
            "representative_block",
            "representative_state",
        ]]
        .drop_duplicates()
        .sort_values(["model", "target_language", "roi_family", "representative_block", "representative_state"])
        .reset_index(drop=True)
    )
    if models:
        model_filter = {str(value).lower() for value in models}
        representative_settings = representative_settings.loc[
            representative_settings["model"].astype(str).str.lower().isin(model_filter)
        ].copy()
    if target_languages:
        language_filter = {str(value).upper() for value in target_languages}
        representative_settings = representative_settings.loc[
            representative_settings["target_language"].astype(str).str.upper().isin(language_filter)
        ].copy()
    if roi_families:
        family_filter = {str(value).upper() for value in roi_families}
        representative_settings = representative_settings.loc[
            representative_settings["roi_family"].astype(str).str.upper().isin(family_filter)
        ].copy()
    representative_settings = representative_settings.reset_index(drop=True)
    if max_settings is not None:
        representative_settings = representative_settings.head(int(max_settings)).copy()

    deletion_rows: list[dict[str, object]] = []
    fallback_counts: dict[str, int] = {}
    processed_shard_paths: list[Path] = []
    prepared_cache: dict[tuple[str, str, str], Any] = {}
    for rep in representative_settings.itertuples(index=False):
        model = str(rep.model)
        target_language = str(rep.target_language)
        roi_family = str(rep.roi_family)
        block = int(rep.representative_block)
        state = str(rep.representative_state)
        shard_path = _deletion_shard_path(
            config,
            model=model,
            target_language=target_language,
            roi_family=roi_family,
            block=block,
            state=state,
        )
        metadata_path = _deletion_shard_metadata_path(
            config,
            model=model,
            target_language=target_language,
            roi_family=roi_family,
            block=block,
            state=state,
        )
        progress_log_path = _deletion_progress_log_path(
            config,
            model=model,
            target_language=target_language,
            roi_family=roi_family,
            block=block,
            state=state,
        )
        processed_shard_paths.append(shard_path)

        if not resume_shards:
            for path in (shard_path, metadata_path, progress_log_path):
                if path.exists():
                    path.unlink()

        existing_metadata = _read_json(metadata_path) if resume_shards else None
        existing_shard_df = pd.read_parquet(shard_path).copy() if shard_path.exists() else _empty_deletion_frame()
        if resume_shards and shard_path.exists() and existing_metadata is None:
            if not existing_shard_df.empty:
                deletion_rows.extend(existing_shard_df.to_dict(orient="records"))
            continue

        state_arrays = _load_state_array_lookup(config, model=model, state=state, block=block)
        source_languages = [language for language in LANGUAGE_TEXT_COLUMNS if language != target_language]
        shared_array = (
            (state_arrays[str(source_languages[0])] + state_arrays[str(source_languages[1])]) / 2.0
        ).astype(np.float32, copy=False)

        subject_candidates = word_df.loc[
            (word_df["model"].astype(str) == model)
            & (word_df["target_language"].astype(str) == target_language)
            & (word_df["roi_family"].astype(str) == roi_family)
            & (word_df["condition"].astype(str) == "SHARED")
            & (word_df["mapping_status"].astype(str) == "OK")
        ].copy()
        if subject_candidates.empty:
            _append_fallback(
                config,
                component="deletion_validation",
                decision="representative_setting_skipped_no_attributions",
                reason=(
                    f"No SHARED attribution rows were available for model={model}, "
                    f"language={target_language}, roi_family={roi_family}."
                ),
            )
            continue
        if "triplet_row_index" not in subject_candidates.columns:
            triplet_lookup = triplets.loc[:, ["triplet_id", "triplet_row_index"]].copy()
            subject_candidates = subject_candidates.merge(triplet_lookup, on="triplet_id", how="left")
            subject_candidates = subject_candidates.dropna(subset=["triplet_row_index"]).copy()
        subject_ids = sorted(subject_candidates["subject_id"].astype(str).unique().tolist())
        started_utc = (
            str(existing_metadata.get("started_utc"))
            if existing_metadata and existing_metadata.get("started_utc")
            else _now_utc()
        )
        completed_subject_ids = {
            str(value)
            for value in (
                existing_metadata.get("completed_subject_ids", [])
                if existing_metadata is not None
                else []
            )
        }
        subject_progress: dict[str, dict[str, Any]] = {
            str(subject_id): {
                "status": str(payload.get("status", "partial")),
                "completed_ks": sorted({int(value) for value in payload.get("completed_ks", [])}),
                "updated_utc": str(payload.get("updated_utc", started_utc)),
            }
            for subject_id, payload in ((existing_metadata or {}).get("subject_progress", {}) or {}).items()
        }
        setting_rows_by_key = _setting_rows_by_key_from_frame(existing_shard_df)
        for existing_subject_id, existing_subject_df in existing_shard_df.groupby("subject_id", sort=False):
            subject_progress.setdefault(
                str(existing_subject_id),
                {
                    "status": "partial",
                    "completed_ks": sorted(existing_subject_df["k"].astype(int).tolist()),
                    "updated_utc": started_utc,
                },
            )
        metadata: dict[str, Any] = {
            "status": "running",
            "model": model,
            "target_language": target_language,
            "roi_family": roi_family,
            "representative_block": int(block),
            "representative_state": state,
            "started_utc": started_utc,
            "updated_utc": _now_utc(),
            "resume_shards": bool(resume_shards),
            "resume_count": int((existing_metadata or {}).get("resume_count", 0)),
            "subject_ids_total": int(len(subject_ids)),
            "subject_ids": list(subject_ids),
            "subjects_completed": int(len(completed_subject_ids)),
            "completed_subject_ids": sorted(completed_subject_ids),
            "current_subject_id": None,
            "current_subject_index": 0,
            "rows_written": int(len(existing_shard_df)),
            "deletion_ks": [int(value) for value in deletion_ks],
            "subject_progress": subject_progress,
            "progress_log_path": str(progress_log_path),
            "shard_path": str(shard_path),
            "last_error": None,
        }
        if resume_shards and (existing_metadata is not None or not existing_shard_df.empty):
            metadata["resume_count"] = int(metadata["resume_count"]) + 1
            _emit_progress(
                progress_log_path,
                event="setting_resumed",
                model=model,
                target_language=target_language,
                roi_family=roi_family,
                block=block,
                state=state,
                console_message=(
                    f"[deletion] resuming {_setting_label(model=model, target_language=target_language, roi_family=roi_family, block=block, state=state)} "
                    f"with {len(completed_subject_ids)}/{len(subject_ids)} subjects complete and {len(existing_shard_df)} rows saved"
                ),
                completed_subjects=int(len(completed_subject_ids)),
                subject_ids_total=int(len(subject_ids)),
                rows_written=int(len(existing_shard_df)),
            )
        else:
            _emit_progress(
                progress_log_path,
                event="setting_started",
                model=model,
                target_language=target_language,
                roi_family=roi_family,
                block=block,
                state=state,
                console_message=(
                    f"[deletion] starting {_setting_label(model=model, target_language=target_language, roi_family=roi_family, block=block, state=state)} "
                    f"for {len(subject_ids)} subjects"
                ),
                completed_subjects=0,
                subject_ids_total=int(len(subject_ids)),
                rows_written=int(len(existing_shard_df)),
            )
        _persist_setting_checkpoint(
            shard_path=shard_path,
            metadata_path=metadata_path,
            setting_rows_by_key=setting_rows_by_key,
            metadata=metadata,
        )
        mutation_vector_cache = SettingMutationCache(
            config=config,
            model=model,
            state=state,
            block=block,
            batch_size=batch_size,
            vectors_by_language={},
        )
        _warm_start_mutation_encoding(
            config,
            model=model,
            target_language=target_language,
            state=state,
            block=block,
            triplets=triplets,
            batch_size=batch_size,
        )
        try:
            for subject_index, subject_id in enumerate(subject_ids, start=1):
                if str(subject_id) in completed_subject_ids:
                    continue
                subject_df = subject_candidates.loc[
                    subject_candidates["subject_id"].astype(str) == str(subject_id)
                ].copy()
                if subject_df.empty:
                    continue
                existing_subject_rows = {
                    int(k): dict(row)
                    for (existing_subject_id, k), row in setting_rows_by_key.items()
                    if str(existing_subject_id) == str(subject_id)
                }
                subject_progress[str(subject_id)] = {
                    "status": "running",
                    "completed_ks": sorted(existing_subject_rows.keys()),
                    "updated_utc": _now_utc(),
                }
                metadata["current_subject_id"] = str(subject_id)
                metadata["current_subject_index"] = int(subject_index)
                metadata["subject_progress"] = subject_progress
                _persist_setting_checkpoint(
                    shard_path=shard_path,
                    metadata_path=metadata_path,
                    setting_rows_by_key=setting_rows_by_key,
                    metadata=metadata,
                )
                _emit_progress(
                    progress_log_path,
                    event="subject_started",
                    model=model,
                    target_language=target_language,
                    roi_family=roi_family,
                    block=block,
                    state=state,
                    console_message=(
                        f"[deletion] {_setting_label(model=model, target_language=target_language, roi_family=roi_family, block=block, state=state)} "
                        f"subject {subject_index}/{len(subject_ids)} {subject_id} started"
                    ),
                    subject_id=str(subject_id),
                    subject_index=int(subject_index),
                    subject_ids_total=int(len(subject_ids)),
                    completed_ks=sorted(existing_subject_rows.keys()),
                )
                prepared_key = (target_language, str(subject_id), roi_family)
                prepared = prepared_cache.get(prepared_key)
                if prepared is None:
                    prepared = _prepare_subject_family(
                        config,
                        target_language=target_language,
                        subject_id=str(subject_id),
                        roi_family=roi_family,
                    )
                    prepared_cache[prepared_key] = prepared

                def _row_callback(row: dict[str, object]) -> None:
                    row_key = (str(row["subject_id"]), int(row["k"]))
                    setting_rows_by_key[row_key] = dict(row)
                    completed_ks = sorted(
                        int(k)
                        for (existing_subject_id, k) in setting_rows_by_key
                        if str(existing_subject_id) == str(subject_id)
                    )
                    subject_progress[str(subject_id)] = {
                        "status": "running",
                        "completed_ks": completed_ks,
                        "updated_utc": _now_utc(),
                    }
                    metadata["subject_progress"] = subject_progress
                    setting_df_live = _persist_setting_checkpoint(
                        shard_path=shard_path,
                        metadata_path=metadata_path,
                        setting_rows_by_key=setting_rows_by_key,
                        metadata=metadata,
                    )
                    _emit_progress(
                        progress_log_path,
                        event="k_completed",
                        model=model,
                        target_language=target_language,
                        roi_family=roi_family,
                        block=block,
                        state=state,
                        console_message=(
                            f"[deletion] {_setting_label(model=model, target_language=target_language, roi_family=roi_family, block=block, state=state)} "
                            f"subject {subject_index}/{len(subject_ids)} {subject_id} completed k={int(row['k'])} "
                            f"(rows={len(setting_df_live)}, delta_diff={float(row['delta_diff']):.4f})"
                        ),
                        subject_id=str(subject_id),
                        subject_index=int(subject_index),
                        subject_ids_total=int(len(subject_ids)),
                        k=int(row["k"]),
                        rows_written=int(len(setting_df_live)),
                        delta_diff=float(row["delta_diff"]),
                    )

                subject_rows, subject_fallbacks = _evaluate_subject_setting(
                    config,
                    model=model,
                    target_language=target_language,
                    roi_family=roi_family,
                    block=block,
                    state=state,
                    prepared=prepared,
                    subject_df=subject_df,
                    triplets=triplets,
                    triplet_to_run=triplet_to_run,
                    state_arrays=state_arrays,
                    shared_array=shared_array,
                    alpha_values=alpha_values,
                    max_pcs=max_pcs,
                    variance_threshold=variance_threshold,
                    deletion_ks=deletion_ks,
                    adaptive_k3_cfg=adaptive_k3_cfg,
                    n_random_draws=n_random_draws,
                    batch_size=batch_size,
                    seed=seed,
                    mutation_vector_cache=mutation_vector_cache,
                    existing_rows_by_k=existing_subject_rows,
                    row_callback=_row_callback,
                )
                for policy, count in subject_fallbacks.items():
                    fallback_counts[str(policy)] = fallback_counts.get(str(policy), 0) + int(count)
                completed_subject_ids.add(str(subject_id))
                subject_progress[str(subject_id)] = {
                    "status": "completed",
                    "completed_ks": sorted(
                        int(k)
                        for (existing_subject_id, k) in setting_rows_by_key
                        if str(existing_subject_id) == str(subject_id)
                    ),
                    "updated_utc": _now_utc(),
                }
                metadata["completed_subject_ids"] = sorted(completed_subject_ids)
                metadata["subjects_completed"] = int(len(completed_subject_ids))
                metadata["current_subject_id"] = None
                metadata["current_subject_index"] = 0
                metadata["subject_progress"] = subject_progress
                setting_df_live = _persist_setting_checkpoint(
                    shard_path=shard_path,
                    metadata_path=metadata_path,
                    setting_rows_by_key=setting_rows_by_key,
                    metadata=metadata,
                )
                _emit_progress(
                    progress_log_path,
                    event="subject_completed",
                    model=model,
                    target_language=target_language,
                    roi_family=roi_family,
                    block=block,
                    state=state,
                    console_message=(
                        f"[deletion] {_setting_label(model=model, target_language=target_language, roi_family=roi_family, block=block, state=state)} "
                        f"subject {subject_index}/{len(subject_ids)} {subject_id} finished "
                        f"({len(subject_rows)} new rows, total_rows={len(setting_df_live)})"
                    ),
                    subject_id=str(subject_id),
                    subject_index=int(subject_index),
                    subject_ids_total=int(len(subject_ids)),
                    new_rows=int(len(subject_rows)),
                    rows_written=int(len(setting_df_live)),
                    completed_subjects=int(len(completed_subject_ids)),
                )
        except BaseException as exc:
            metadata["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
            metadata["last_error"] = str(exc)
            metadata["subject_progress"] = subject_progress
            _persist_setting_checkpoint(
                shard_path=shard_path,
                metadata_path=metadata_path,
                setting_rows_by_key=setting_rows_by_key,
                metadata=metadata,
            )
            _emit_progress(
                progress_log_path,
                event="setting_failed",
                model=model,
                target_language=target_language,
                roi_family=roi_family,
                block=block,
                state=state,
                console_message=(
                    f"[deletion] {_setting_label(model=model, target_language=target_language, roi_family=roi_family, block=block, state=state)} "
                    f"stopped with status={metadata['status']} after {len(completed_subject_ids)}/{len(subject_ids)} completed subjects"
                ),
                completed_subjects=int(len(completed_subject_ids)),
                subject_ids_total=int(len(subject_ids)),
                rows_written=int(len(setting_rows_by_key)),
                error=str(exc),
            )
            raise
        metadata["status"] = "completed"
        metadata["completed_subject_ids"] = sorted(completed_subject_ids)
        metadata["subjects_completed"] = int(len(completed_subject_ids))
        metadata["current_subject_id"] = None
        metadata["current_subject_index"] = 0
        metadata["subject_progress"] = subject_progress
        setting_df = _persist_setting_checkpoint(
            shard_path=shard_path,
            metadata_path=metadata_path,
            setting_rows_by_key=setting_rows_by_key,
            metadata=metadata,
        )
        _emit_progress(
            progress_log_path,
            event="setting_completed",
            model=model,
            target_language=target_language,
            roi_family=roi_family,
            block=block,
            state=state,
            console_message=(
                f"[deletion] completed {_setting_label(model=model, target_language=target_language, roi_family=roi_family, block=block, state=state)} "
                f"with {len(setting_df)} rows across {len(completed_subject_ids)}/{len(subject_ids)} subjects"
            ),
            completed_subjects=int(len(completed_subject_ids)),
            subject_ids_total=int(len(subject_ids)),
            rows_written=int(len(setting_df)),
        )
        if not setting_df.empty:
            deletion_rows.extend(setting_df.to_dict(orient="records"))

    deletion_df = pd.DataFrame(deletion_rows, columns=DELETION_VALIDATION_COLUMNS)
    deletion_path = (
        _local_outputs_root(config) / "subject_results" / "deletion_validation_subset.parquet"
        if subset_mode
        else _deletion_validation_path(config)
    )
    _write_parquet(deletion_df, deletion_path)

    if any("fallback" in policy or "overlap" in policy for policy in fallback_counts):
        _append_fallback(
            config,
            component="deletion_validation",
            decision="matched_random_fallbacks_used",
            reason="Matched-random sampling occasionally fell back to overlap or looser allocation matching; see deletion_validation.md for policy counts.",
        )

    provenance_path = (
        _local_outputs_root(config) / "provenance" / "deletion_validation_subset.md"
        if subset_mode
        else _provenance_path(config)
    )
    provenance_path.parent.mkdir(parents=True, exist_ok=True)
    provenance_lines = [
        "# Deletion Validation",
        "",
        f"- deletion_validation: `{deletion_path}`",
        f"- n_rows: `{len(deletion_df)}`",
        f"- ks: `{', '.join(str(value) for value in deletion_ks)}`",
        f"- adaptive_k3_enabled: `{bool(adaptive_k3_cfg.get('enabled', True))}`",
        f"- adaptive_k3_min_delta_diff_gain: `{float(adaptive_k3_cfg.get('min_delta_diff_gain', 0.0))}`",
        f"- matched_random_draws: `{n_random_draws}`",
        f"- mutation_batch_size: `{batch_size}`",
        f"- representative_settings_processed: `{len(representative_settings)}`",
        f"- representative_setting_shards: `{len(processed_shard_paths)}`",
        f"- resume_shards: `{bool(resume_shards)}`",
        f"- subset_mode: `{subset_mode}`",
        f"- shard_metadata_pattern: `{_deletion_shards_root(config) / 'deletion_validation__*__metadata.json'}`",
        f"- progress_logs_root: `{_deletion_progress_logs_root(config)}`",
        "",
        "## Matching Policy Counts",
        "",
    ]
    if fallback_counts:
        provenance_lines.extend(
            [
                f"- `{policy}`: `{count}`"
                for policy, count in sorted(fallback_counts.items())
            ]
        )
    else:
        provenance_lines.append("- `none_recorded`: `0`")
    provenance_path.write_text("\n".join(provenance_lines) + "\n", encoding="utf-8")

    return {
        "deletion_validation": deletion_path,
        "provenance": provenance_path,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run top-k versus matched-random deletion validation.")
    parser.add_argument("--config", default="conf/base.yaml", help="Path to sequel config.")
    parser.add_argument("--model", action="append", dest="models", help="Optional model filter; repeatable.")
    parser.add_argument("--target-language", action="append", dest="target_languages", help="Optional target-language filter; repeatable.")
    parser.add_argument("--roi-family", action="append", dest="roi_families", help="Optional ROI-family filter; repeatable.")
    parser.add_argument("--max-settings", type=int, default=None, help="Optional cap on representative settings to process.")
    parser.add_argument("--no-resume-shards", action="store_true", help="Recompute representative-setting shards even if they already exist.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    outputs = run_deletion_validation(
        config,
        models=args.models,
        target_languages=args.target_languages,
        roi_families=args.roi_families,
        max_settings=args.max_settings,
        resume_shards=not bool(args.no_resume_shards),
    )
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
