from __future__ import annotations

import atexit
import argparse
from dataclasses import dataclass
from datetime import UTC, datetime
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any
import uuid

import numpy as np
import pandas as pd
import regex

from src.encoding.design_matrix_operator import build_language_sentence_to_bold_operator
from src.encoding.multi_target_ridge import (
    fit_pca_train_test as _shared_fit_pca_train_test,
    residualize_train_test as _shared_residualize_train_test,
    ridge_coefficients as _shared_ridge_coefficients,
    select_alpha_per_target as _shared_select_alpha_per_target,
    standardize_train_test as _shared_standardize_train_test,
)
from src.encoding.run_state_sweep import (
    LANGUAGE_SPECS,
    _align_run_frame_columns,
    _build_run_nuisance_and_acoustic,
    _load_annotation_tables,
    _run_scan_counts,
)
from src.utils.config import load_config


LANGUAGE_TEXT_COLUMNS = {
    "EN": "en_text",
    "FR": "fr_text",
    "ZH": "zh_text",
}
WORD_ATTRIBUTION_COLUMNS = [
    "subject_id",
    "model",
    "target_language",
    "roi_family",
    "triplet_id",
    "triplet_row_index",
    "canonical_run",
    "token_subset_name",
    "token_subset_rank",
    "representative_block",
    "representative_state",
    "condition",
    "sentence_language",
    "word_index",
    "word_text",
    "word_char_start",
    "word_char_end",
    "upos",
    "token_class",
    "attribution_score",
    "mapping_status",
    "n_subwords_deleted",
]
MAPPING_FAILURE_COLUMNS = [
    "subject_id",
    "model",
    "target_language",
    "roi_family",
    "triplet_id",
    "triplet_row_index",
    "canonical_run",
    "token_subset_name",
    "token_subset_rank",
    "representative_block",
    "representative_state",
    "condition",
    "sentence_language",
    "word_index",
    "word_text",
    "word_char_start",
    "word_char_end",
    "failure_reason",
    "mapping_status",
]
LATIN_TOKEN_RE = regex.compile(r"\p{L}+(?:['’-]\p{L}+)*|\p{N}+|[^\p{L}\p{N}\s]", flags=regex.VERSION1)
ZH_TOKEN_RE = regex.compile(r"\p{Han}|[\p{L}\p{N}]+|[^\p{L}\p{N}\s]", flags=regex.VERSION1)
PUNCT_RE = regex.compile(r"^\p{P}+$", flags=regex.VERSION1)
EN_FUNCTION_WORDS = {
    "a", "an", "the", "and", "or", "but", "if", "then", "than", "that", "this", "these", "those",
    "to", "of", "in", "on", "at", "for", "from", "with", "by", "as", "is", "are", "was", "were",
    "be", "been", "being", "am", "do", "does", "did", "have", "has", "had", "will", "would",
    "can", "could", "should", "may", "might", "must", "not", "no", "nor", "it", "its", "they",
    "them", "their", "he", "she", "we", "you", "i", "me", "my", "our", "your",
}
FR_FUNCTION_WORDS = {
    "le", "la", "les", "un", "une", "des", "de", "du", "et", "ou", "mais", "donc", "or", "ni", "car",
    "a", "au", "aux", "en", "dans", "sur", "sous", "pour", "par", "avec", "sans", "ce", "cet", "cette",
    "ces", "je", "tu", "il", "elle", "nous", "vous", "ils", "elles", "me", "te", "se", "ne", "pas",
    "est", "sont", "etait", "etre", "avoir", "ont", "que", "qui", "quoi", "dont",
}
SEMANTIC_ROI_FAMILY = "SEMANTIC"
AUDITORY_ROI_FAMILY = "AUDITORY"
AUDITORY_HYPOTHESIS = "H2_auditory_attention_preference"
TOKEN_SUBSET_COLUMNS = [
    "target_language",
    "canonical_run",
    "triplet_id",
    "triplet_row_index",
    "token_subset_name",
    "token_subset_rank",
]


def _now_utc() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _local_outputs_root(config: dict[str, Any]) -> Path:
    return Path(config["paths"]["local_outputs_root"]).resolve()


def _local_manifests_root(config: dict[str, Any]) -> Path:
    return Path(config["paths"]["local_manifests_root"]).resolve()


def _triplets_path(config: dict[str, Any]) -> Path:
    return _local_manifests_root(config) / "triplets.parquet"


def _sample_manifest_path(config: dict[str, Any]) -> Path:
    return _local_manifests_root(config) / "sample_manifest.parquet"


def _roi_manifest_path(config: dict[str, Any]) -> Path:
    return _local_manifests_root(config) / "roi_manifest.parquet"


def _roi_metadata_path(config: dict[str, Any]) -> Path:
    return _local_manifests_root(config) / "roi_metadata.parquet"


def _state_cache_manifest_path(config: dict[str, Any]) -> Path:
    return _local_outputs_root(config) / "caches" / "features" / "state_cache_manifest.parquet"


def _tokenization_metadata_path(config: dict[str, Any]) -> Path:
    return _local_outputs_root(config) / "caches" / "features" / "tokenization_metadata.parquet"


def _representative_states_path(config: dict[str, Any]) -> Path:
    return _local_outputs_root(config) / "group_results" / "representative_states.parquet"


def _primary_effect_tables_path(config: dict[str, Any]) -> Path:
    return _local_outputs_root(config) / "group_results" / "primary_effect_tables.parquet"


def _word_attributions_path(config: dict[str, Any]) -> Path:
    return _local_outputs_root(config) / "subject_results" / "word_attributions.parquet"


def _mapping_failures_path(config: dict[str, Any]) -> Path:
    return _local_outputs_root(config) / "qc" / "word_to_subword_mapping_failures.parquet"


def _token_scope_representatives_path(config: dict[str, Any]) -> Path:
    return _local_outputs_root(config) / "group_results" / "representative_states_token_scope.parquet"


def _token_subset_manifest_path(config: dict[str, Any]) -> Path:
    return _local_outputs_root(config) / "provenance" / "token_triplet_subset.parquet"


def _provenance_path(config: dict[str, Any]) -> Path:
    return _local_outputs_root(config) / "provenance" / "word_attributions.md"


def _fallback_decisions_path(config: dict[str, Any]) -> Path:
    return _local_outputs_root(config) / "provenance" / "fallback_decisions.md"


def _write_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)


def _mutation_worker_tmp_root(config: dict[str, Any]) -> Path:
    path = _local_outputs_root(config) / ".mutation_worker_tmp"
    path.mkdir(parents=True, exist_ok=True)
    return path


@dataclass(slots=True)
class _MutationEncoderWorker:
    config: dict[str, Any]
    model: str
    process: subprocess.Popen[str]
    stderr_path: Path

    def encode(
        self,
        *,
        language: str,
        state: str,
        block: int,
        texts: list[str],
    ) -> np.ndarray:
        if self.process.poll() is not None or self.process.stdin is None or self.process.stdout is None:
            raise RuntimeError(
                f"Persistent mutation worker for model={self.model} is not running. See {self.stderr_path}."
            )
        output_npy = _mutation_worker_tmp_root(self.config) / f"{self.model}_{uuid.uuid4().hex}.npy"
        try:
            request = {
                "language": str(language).upper(),
                "state": str(state).upper(),
                "block": int(block),
                "texts": [str(text) for text in texts],
                "output_npy": str(output_npy),
            }
            self.process.stdin.write(json.dumps(request) + "\n")
            self.process.stdin.flush()
            response_line = self.process.stdout.readline()
            if not response_line:
                stderr_tail = ""
                if self.stderr_path.exists():
                    stderr_tail = self.stderr_path.read_text(encoding="utf-8", errors="replace")[-1000:]
                raise RuntimeError(
                    f"Persistent mutation worker for model={self.model} did not return a response. "
                    f"stderr tail: {stderr_tail}"
                )
            response = json.loads(response_line)
            if response.get("status") != "ok":
                raise RuntimeError(
                    f"Persistent mutation worker for model={self.model} failed: {response.get('error')}"
                )
            return np.load(output_npy).astype(np.float32, copy=False)
        finally:
            try:
                output_npy.unlink(missing_ok=True)
            except OSError:
                pass

    def close(self) -> None:
        if self.process.poll() is not None:
            return
        if self.process.stdin is not None:
            try:
                self.process.stdin.write(json.dumps({"op": "shutdown"}) + "\n")
                self.process.stdin.flush()
            except OSError:
                pass
            try:
                self.process.stdin.close()
            except OSError:
                pass
        try:
            self.process.wait(timeout=5)
        except Exception:
            self.process.kill()


_MUTATION_ENCODER_WORKERS: dict[str, _MutationEncoderWorker] = {}


def _close_mutation_encoder_workers() -> None:
    for worker in list(_MUTATION_ENCODER_WORKERS.values()):
        worker.close()
    _MUTATION_ENCODER_WORKERS.clear()


atexit.register(_close_mutation_encoder_workers)


def _get_mutation_encoder_worker(config: dict[str, Any], *, model: str) -> _MutationEncoderWorker:
    key = str(model)
    worker = _MUTATION_ENCODER_WORKERS.get(key)
    if worker is not None and worker.process.poll() is None:
        return worker

    logs_root = _local_outputs_root(config) / "provenance" / "run_logs"
    logs_root.mkdir(parents=True, exist_ok=True)
    stderr_path = logs_root / f"mutation_worker__{key}.err.log"
    stderr_handle = stderr_path.open("a", encoding="utf-8")
    command = [
        sys.executable,
        "-m",
        "src.attribution._encode_mutations",
        "--worker",
        "--config",
        "conf/base.yaml",
        "--model",
        key,
    ]
    process = subprocess.Popen(
        command,
        cwd=str(_repo_root()),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=stderr_handle,
        text=True,
        encoding="utf-8",
        bufsize=1,
    )
    worker = _MutationEncoderWorker(
        config=config,
        model=key,
        process=process,
        stderr_path=stderr_path,
    )
    _MUTATION_ENCODER_WORKERS[key] = worker
    return worker


def _token_subset_config(config: dict[str, Any]) -> dict[str, Any]:
    return dict(config.get("attribution", {}).get("token_subset", {}))


def _token_scope_config(config: dict[str, Any]) -> dict[str, Any]:
    return dict(config.get("attribution", {}).get("token_scope", {}))


def _token_subset_name(config: dict[str, Any]) -> str:
    return str(_token_subset_config(config).get("name", "strict_1_1_1_v1"))


def _token_subset_triplets_per_run(config: dict[str, Any]) -> int:
    return int(_token_subset_config(config).get("triplets_per_run", 15))


def _primary_support_threshold(config: dict[str, Any]) -> float:
    return float(_token_scope_config(config).get("support_p_threshold", 0.05))


def _load_triplets(config: dict[str, Any]) -> pd.DataFrame:
    triplets = pd.read_parquet(_triplets_path(config)).copy()
    if "triplet_row_index" not in triplets.columns:
        triplets["triplet_row_index"] = np.arange(len(triplets), dtype=np.int64)
    triplets["triplet_id"] = triplets["triplet_id"].astype(str)
    triplets["triplet_row_index"] = triplets["triplet_row_index"].astype(np.int64)
    if "canonical_run" in triplets.columns:
        triplets["canonical_run"] = triplets["canonical_run"].astype(np.int64)
    return triplets.sort_values("triplet_row_index").reset_index(drop=True)


def _load_representative_states(config: dict[str, Any]) -> pd.DataFrame:
    path = _representative_states_path(config)
    if not path.exists():
        raise FileNotFoundError(
            f"Representative states missing: {path}. Run src.attribution.select_representative_states first."
        )
    return pd.read_parquet(path).copy()


def _build_token_subset_manifest(config: dict[str, Any], triplets: pd.DataFrame) -> pd.DataFrame:
    subset_name = _token_subset_name(config)
    triplets_per_run = _token_subset_triplets_per_run(config)
    subset_cfg = _token_subset_config(config)
    triplets = triplets.copy()
    if "triplet_row_index" not in triplets.columns:
        triplets["triplet_row_index"] = np.arange(len(triplets), dtype=np.int64)
    target_languages = tuple(
        str(language)
        for language in subset_cfg.get("target_languages", config.get("dataset", {}).get("languages", ["EN", "FR", "ZH"]))
    )
    rows: list[dict[str, Any]] = []

    ordered = triplets.sort_values(["canonical_run", "triplet_row_index"]).reset_index(drop=True)
    for canonical_run, run_df in ordered.groupby("canonical_run", sort=True):
        chosen = run_df.head(max(0, triplets_per_run)).reset_index(drop=True)
        if len(chosen) < triplets_per_run:
            _append_fallback(
                config,
                component="attribution",
                decision="token_subset_run_underfilled",
                reason=(
                    f"Canonical run {int(canonical_run)} only contained {len(chosen)} aligned triplets, "
                    f"so the fixed token subset could not reach the requested {triplets_per_run} rows."
                ),
            )
        for target_language in target_languages:
            for selection_rank, row in enumerate(chosen.itertuples(index=False), start=1):
                rows.append(
                    {
                        "target_language": str(target_language),
                        "canonical_run": int(canonical_run),
                        "triplet_id": str(row.triplet_id),
                        "triplet_row_index": int(row.triplet_row_index),
                        "token_subset_name": subset_name,
                        "token_subset_rank": int(selection_rank),
                    }
                )
    return pd.DataFrame(rows, columns=TOKEN_SUBSET_COLUMNS)


def _load_or_create_token_subset_manifest(config: dict[str, Any], triplets: pd.DataFrame) -> pd.DataFrame:
    path = _token_subset_manifest_path(config)
    expected_subset = _build_token_subset_manifest(config, triplets)

    def _normalize_subset(frame: pd.DataFrame) -> pd.DataFrame:
        normalized = frame.loc[:, TOKEN_SUBSET_COLUMNS].copy()
        normalized["target_language"] = normalized["target_language"].astype(str)
        normalized["canonical_run"] = normalized["canonical_run"].astype(int)
        normalized["triplet_id"] = normalized["triplet_id"].astype(str)
        normalized["triplet_row_index"] = normalized["triplet_row_index"].astype(int)
        normalized["token_subset_name"] = normalized["token_subset_name"].astype(str)
        normalized["token_subset_rank"] = normalized["token_subset_rank"].astype(int)
        return normalized.sort_values(
            ["target_language", "canonical_run", "token_subset_rank", "triplet_row_index", "triplet_id"]
        ).reset_index(drop=True)

    if path.exists():
        existing = pd.read_parquet(path).copy()
        required = set(TOKEN_SUBSET_COLUMNS)
        if required.issubset(existing.columns) and _normalize_subset(existing).equals(_normalize_subset(expected_subset)):
            return existing
    _write_parquet(expected_subset, path)
    return expected_subset


def _subset_triplets_for_target_language(
    triplets: pd.DataFrame,
    subset_manifest: pd.DataFrame,
    *,
    target_language: str,
) -> pd.DataFrame:
    triplets = triplets.copy()
    if "triplet_row_index" not in triplets.columns:
        triplets["triplet_row_index"] = np.arange(len(triplets), dtype=np.int64)
    triplets["triplet_id"] = triplets["triplet_id"].astype(str)
    triplets["triplet_row_index"] = triplets["triplet_row_index"].astype(np.int64)
    triplets["canonical_run"] = triplets["canonical_run"].astype(np.int64)
    subset_rows = subset_manifest.loc[
        subset_manifest["target_language"].astype(str) == str(target_language),
        :,
    ].copy()
    if subset_rows.empty:
        raise RuntimeError(f"No token subset rows were available for target_language={target_language}.")
    subset_rows["triplet_id"] = subset_rows["triplet_id"].astype(str)
    subset_rows["triplet_row_index"] = subset_rows["triplet_row_index"].astype(np.int64)
    subset_rows["canonical_run"] = subset_rows["canonical_run"].astype(np.int64)
    merged = subset_rows.merge(
        triplets,
        on=["triplet_id", "triplet_row_index", "canonical_run"],
        how="left",
    )
    merged = merged.dropna(subset=["triplet_id", "triplet_row_index"]).copy()
    return merged.sort_values(["canonical_run", "token_subset_rank", "triplet_row_index"]).reset_index(drop=True)


def _primary_effect_support_lookup(config: dict[str, Any]) -> pd.DataFrame | None:
    path = _primary_effect_tables_path(config)
    if not path.exists():
        return None
    return pd.read_parquet(path).copy()


def _auditory_token_supports_expansion(
    config: dict[str, Any],
    primary_effects: pd.DataFrame | None,
    *,
    model: str,
    target_language: str,
) -> tuple[bool, str]:
    if primary_effects is None:
        return False, "primary_effect_tables_missing"
    hypothesis = str(_token_scope_config(config).get("auditory_hypothesis", AUDITORY_HYPOTHESIS))
    require_positive_effect = bool(_token_scope_config(config).get("require_positive_effect", True))
    support_row = primary_effects.loc[
        (primary_effects["model"].astype(str) == str(model))
        & (primary_effects["language"].astype(str) == str(target_language))
        & (primary_effects["hypothesis"].astype(str) == hypothesis)
    ].copy()
    if support_row.empty:
        return False, "auditory_primary_cell_missing"
    row = support_row.iloc[0]
    p_holm = float(row.get("p_holm", np.nan))
    effect_mean = float(row.get("effect_mean", np.nan))
    if np.isnan(p_holm) or p_holm >= _primary_support_threshold(config):
        return False, "auditory_primary_not_significant"
    if require_positive_effect and (np.isnan(effect_mean) or effect_mean <= 0.0):
        return False, "auditory_primary_nonpositive"
    return True, "auditory_primary_supported"


def _filter_representatives_for_token_scope(
    config: dict[str, Any],
    representatives: pd.DataFrame,
) -> pd.DataFrame:
    scope_cfg = _token_scope_config(config)
    include_semantic = bool(scope_cfg.get("include_semantic", True))
    include_auditory = bool(scope_cfg.get("include_auditory_if_primary_supported", True))
    primary_effects = _primary_effect_support_lookup(config)
    kept_rows: list[dict[str, Any]] = []

    for row in representatives.itertuples(index=False):
        roi_family = str(row.roi_family)
        if roi_family == SEMANTIC_ROI_FAMILY:
            if include_semantic:
                kept_rows.append(row._asdict())
            continue
        if roi_family != AUDITORY_ROI_FAMILY:
            continue
        if not include_auditory:
            _append_fallback(
                config,
                component="attribution",
                decision="auditory_token_scope_disabled",
                reason=(
                    f"Auditory token-level work was disabled in config, so model={row.model}, "
                    f"language={row.language} was not expanded beyond the semantic winners."
                ),
            )
            continue
        supported, reason = _auditory_token_supports_expansion(
            config,
            primary_effects,
            model=str(row.model),
            target_language=str(row.language),
        )
        if supported:
            kept_rows.append(row._asdict())
            continue
        _append_fallback(
            config,
            component="attribution",
            decision="auditory_token_scope_not_expanded",
            reason=(
                f"Auditory token-level work stayed descriptive-only for model={row.model}, "
                f"language={row.language} because {reason}."
            ),
        )
    filtered = pd.DataFrame(kept_rows, columns=representatives.columns)
    if filtered.empty:
        raise RuntimeError("Token-level scope filtering removed every representative setting.")
    return filtered.reset_index(drop=True)


def _load_state_array_lookup(config: dict[str, Any], *, model: str, state: str, block: int) -> dict[str, np.ndarray]:
    manifest_path = _state_cache_manifest_path(config)
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"State cache manifest missing: {manifest_path}. Run src.features.extract_internal_states first."
        )
    manifest = pd.read_parquet(manifest_path).copy()
    subset = manifest.loc[
        (manifest["model"].astype(str) == str(model))
        & (manifest["state"].astype(str) == str(state))
        & (manifest["block"].astype(int) == int(block))
    ].copy()
    arrays: dict[str, np.ndarray] = {}
    for row in subset.itertuples(index=False):
        arrays[str(row.language)] = np.load(row.source_array_path).astype(np.float32, copy=False)
    if set(arrays) != set(LANGUAGE_TEXT_COLUMNS):
        missing = sorted(set(LANGUAGE_TEXT_COLUMNS).difference(arrays))
        raise RuntimeError(
            f"Missing state arrays for model={model}, state={state}, block={block}: {missing}"
        )
    return arrays


def _load_token_metadata_lookup(config: dict[str, Any], *, model: str) -> dict[tuple[str, str], dict[str, Any]]:
    path = _tokenization_metadata_path(config)
    if not path.exists():
        raise FileNotFoundError(
            f"Tokenization metadata missing: {path}. Run src.features.extract_internal_states first."
        )
    df = pd.read_parquet(path).copy()
    df = df.loc[df["model"].astype(str) == str(model)].copy()
    lookup: dict[tuple[str, str], dict[str, Any]] = {}
    for row in df.itertuples(index=False):
        offsets = json.loads(str(row.char_offsets_json))
        lookup[(str(row.language), str(row.triplet_id))] = {
            "text": str(row.text),
            "token_strings": json.loads(str(row.token_strings_json)),
            "char_offsets": [tuple(int(value) for value in pair) for pair in offsets],
            "has_offset_mapping": bool(row.has_offset_mapping),
        }
    return lookup


def _source_languages(target_language: str) -> tuple[str, str]:
    sources = [language for language in LANGUAGE_TEXT_COLUMNS if language != str(target_language)]
    return tuple(sources)  # type: ignore[return-value]


def _append_fallback(config: dict[str, Any], *, component: str, decision: str, reason: str) -> None:
    path = _fallback_decisions_path(config)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(
            f"- timestamp_utc: `{_now_utc()}`\n"
            f"  - component: `{component}`\n"
            f"  - decision: `{decision}`\n"
            f"  - reason: `{reason}`\n"
        )


def _segment_words(text: str, *, language: str) -> list[dict[str, Any]]:
    if language == "ZH":
        matches = list(ZH_TOKEN_RE.finditer(text))
        non_punct_indices = [index for index, match in enumerate(matches) if not PUNCT_RE.match(match.group(0))]
        first_non_punct = non_punct_indices[0] if non_punct_indices else -1
        last_non_punct = non_punct_indices[-1] if non_punct_indices else -1
        return [
            {
                "word_index": word_index,
                "word_text": token,
                "word_char_start": int(match.start()),
                "word_char_end": int(match.end()),
                "upos": None if not PUNCT_RE.match(token) else "PUNCT",
                "token_class": (
                    "PUNCT"
                    if PUNCT_RE.match(token)
                    else "EDGE"
                    if word_index in {first_non_punct, last_non_punct}
                    else "CONTENT"
                ),
            }
            for word_index, match in enumerate(matches)
            for token in [match.group(0)]
        ]

    words: list[dict[str, Any]] = []
    function_words = EN_FUNCTION_WORDS if language == "EN" else FR_FUNCTION_WORDS
    matches = list(LATIN_TOKEN_RE.finditer(text))
    non_punct_indices = [index for index, match in enumerate(matches) if not PUNCT_RE.match(match.group(0))]
    first_non_punct = non_punct_indices[0] if non_punct_indices else -1
    last_non_punct = non_punct_indices[-1] if non_punct_indices else -1
    for word_index, match in enumerate(matches):
        token = match.group(0)
        token_lower = token.lower()
        if PUNCT_RE.match(token):
            token_class = "PUNCT"
        elif word_index in {first_non_punct, last_non_punct}:
            token_class = "EDGE"
        elif token_lower in function_words:
            token_class = "FUNCTION"
        else:
            token_class = "CONTENT"
        words.append(
            {
                "word_index": word_index,
                "word_text": token,
                "word_char_start": int(match.start()),
                "word_char_end": int(match.end()),
                "upos": None if token_class != "PUNCT" else "PUNCT",
                "token_class": token_class,
            }
        )
    return words


def map_words_to_subwords(
    text: str,
    word_spans: list[tuple[int, int]],
    subword_offsets: list[tuple[int, int]],
) -> list[list[int]]:
    _ = text
    mapped: list[list[int]] = []
    for word_start, word_end in word_spans:
        overlapping = [
            index
            for index, (sub_start, sub_end) in enumerate(subword_offsets)
            if int(sub_end) > int(sub_start)
            and int(sub_start) < int(word_end)
            and int(sub_end) > int(word_start)
        ]
        if overlapping:
            overlapping = list(range(min(overlapping), max(overlapping) + 1))
        mapped.append(overlapping)
    return mapped


def _delete_span(text: str, start: int, end: int, *, language: str) -> str:
    mutated = f"{text[:start]}{text[end:]}"
    if language in {"EN", "FR"}:
        mutated = regex.sub(r"\s+", " ", mutated).strip()
    return mutated


def _run_mutation_encoder(
    config: dict[str, Any],
    *,
    model: str,
    language: str,
    state: str,
    block: int,
    texts: list[str],
) -> np.ndarray:
    mode = str(config.get("attribution", {}).get("mutation_encoder_mode", "persistent_worker")).strip().lower()
    if mode not in {"persistent_worker", "subprocess", "auto"}:
        raise ValueError(f"Unsupported attribution.mutation_encoder_mode: {mode}")
    if mode in {"persistent_worker", "auto"}:
        try:
            worker = _get_mutation_encoder_worker(config, model=str(model))
            return worker.encode(
                language=str(language),
                state=str(state),
                block=int(block),
                texts=[str(text) for text in texts],
            ).astype(np.float32, copy=False)
        except Exception:
            if mode == "persistent_worker":
                raise
    prior_python = str(config["paths"]["prior_python"])
    with tempfile.TemporaryDirectory(prefix="attr_mut_") as tmp_dir:
        tmp_root = Path(tmp_dir)
        input_json = tmp_root / "texts.json"
        output_npy = tmp_root / "pooled.npy"
        input_json.write_text(json.dumps(texts, ensure_ascii=False), encoding="utf-8")
        command = [
            prior_python,
            "-m",
            "src.attribution._encode_mutations",
            "--config",
            "conf/base.yaml",
            "--model",
            str(model),
            "--language",
            str(language),
            "--state",
            str(state),
            "--block",
            str(int(block)),
            "--input-json",
            str(input_json),
            "--output-npy",
            str(output_npy),
        ]
        subprocess.run(command, cwd=str(_repo_root()), check=True, capture_output=True, text=True)
        return np.load(output_npy).astype(np.float32, copy=False)


@dataclass(frozen=True, slots=True)
class PreparedSubjectFamily:
    subject_id: str
    target_language: str
    roi_family: str
    operator: Any
    run_order: tuple[int, ...]
    nuisance_by_run: dict[int, np.ndarray]
    y_by_run: dict[int, np.ndarray]
    triplets: pd.DataFrame


@dataclass(frozen=True, slots=True)
class FoldAttributionModel:
    held_out_run: int
    row_to_local_index: dict[int, int]
    basis: np.ndarray
    scale_std: np.ndarray
    pca_components: np.ndarray
    ridge_beta: np.ndarray
    base_prediction: np.ndarray
    actual: np.ndarray
    window_by_triplet: dict[int, np.ndarray]


def _prepare_subject_family(
    config: dict[str, Any],
    *,
    target_language: str,
    subject_id: str,
    roi_family: str,
) -> PreparedSubjectFamily:
    triplets = _load_triplets(config)
    sample_manifest = pd.read_parquet(_sample_manifest_path(config)).copy()
    roi_manifest = pd.read_parquet(_roi_manifest_path(config)).copy()
    roi_metadata = pd.read_parquet(_roi_metadata_path(config)).copy().sort_values("roi_index").reset_index(drop=True)
    included = sample_manifest.loc[
        (sample_manifest["language"].astype(str) == str(target_language))
        & sample_manifest["included"].astype(bool),
        "subject_id",
    ].astype(str)
    if str(subject_id) not in set(included.tolist()):
        raise RuntimeError(f"Subject {subject_id} is not marked included for target_language={target_language}.")

    language_roi_manifest = roi_manifest.loc[roi_manifest["language"].astype(str) == str(target_language)].copy()
    scan_counts_by_run = _run_scan_counts(language_roi_manifest)
    spec = LANGUAGE_SPECS[str(target_language)]
    operator = build_language_sentence_to_bold_operator(
        triplets,
        language=str(target_language),
        scan_counts_by_run=scan_counts_by_run,
        onset_column=spec["onset_col"],
        offset_column=spec["offset_col"],
        tr_seconds=float(config.get("dataset", {}).get("tr_seconds", 2.0)),
        fine_hz=float(config.get("features", {}).get("fine_hz", 10.0)),
    )
    prosody, word_info = _load_annotation_tables(config, str(target_language))
    nuisance_frames_by_run: dict[int, pd.DataFrame] = {}
    acoustic_frames_by_run: dict[int, pd.DataFrame] = {}
    for run_index in sorted(scan_counts_by_run):
        run_triplets = triplets.loc[triplets["canonical_run"].astype(int) == int(run_index)].copy()
        nuisance_df, acoustic_df = _build_run_nuisance_and_acoustic(
            config,
            language=str(target_language),
            run_index=int(run_index),
            n_scans=int(scan_counts_by_run[int(run_index)]),
            sentence_onsets=run_triplets[spec["onset_col"]].to_numpy(dtype=np.float32),
            prosody=prosody,
            word_info=word_info,
            hrf_kernel=operator.hrf_kernel,
        )
        nuisance_frames_by_run[int(run_index)] = nuisance_df
        acoustic_frames_by_run[int(run_index)] = acoustic_df

    nuisance_frames_by_run = _align_run_frame_columns(nuisance_frames_by_run)
    acoustic_frames_by_run = _align_run_frame_columns(acoustic_frames_by_run)
    nuisance_by_run = {
        int(run_index): np.column_stack(
            [
                nuisance_frames_by_run[int(run_index)].to_numpy(dtype=np.float32, copy=False),
                acoustic_frames_by_run[int(run_index)].to_numpy(dtype=np.float32, copy=False),
            ]
        ).astype(np.float32, copy=False)
        for run_index in sorted(scan_counts_by_run)
    }

    family_indices = (
        roi_metadata.loc[roi_metadata["roi_family"].astype(str) == str(roi_family), "roi_index"].astype(int).to_numpy() - 1
    )
    if len(family_indices) == 0:
        raise RuntimeError(f"No ROI indices found for roi_family={roi_family}.")
    subject_runs = language_roi_manifest.loc[
        language_roi_manifest["subject_id"].astype(str) == str(subject_id)
    ].sort_values("canonical_run")
    y_by_run: dict[int, np.ndarray] = {}
    for row in subject_runs.itertuples(index=False):
        run_array = np.load(row.roi_timeseries_path).astype(np.float32, copy=False)
        y_by_run[int(row.canonical_run)] = run_array[:, family_indices].mean(axis=1).astype(np.float32, copy=False)

    return PreparedSubjectFamily(
        subject_id=str(subject_id),
        target_language=str(target_language),
        roi_family=str(roi_family),
        operator=operator,
        run_order=tuple(sorted(y_by_run)),
        nuisance_by_run=nuisance_by_run,
        y_by_run=y_by_run,
        triplets=triplets,
    )


def _residualize_train_test(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    Z_train: np.ndarray,
    Z_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X_train_res, X_test_res, y_train_res, y_test_res = _shared_residualize_train_test(
        X_train,
        X_test,
        y_train,
        y_test,
        Z_train,
        Z_test,
    )
    return (
        X_train_res.astype(np.float32, copy=False),
        X_test_res.astype(np.float32, copy=False),
        y_train_res[:, 0].astype(np.float32, copy=False),
        y_test_res[:, 0].astype(np.float32, copy=False),
    )


def _standardize_train_test(
    X_train: np.ndarray,
    X_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    X_train_std, X_test_std, scaler = _shared_standardize_train_test(X_train, X_test)
    return (
        X_train_std.astype(np.float32, copy=False),
        X_test_std.astype(np.float32, copy=False),
        scaler.std.astype(np.float32, copy=False),
    )


def _fit_pca_train_test(
    X_train: np.ndarray,
    X_test: np.ndarray,
    *,
    max_pcs: int,
    variance_threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    X_train_pca, X_test_pca, transform = _shared_fit_pca_train_test(
        X_train,
        X_test,
        max_components=max_pcs,
        variance_threshold=variance_threshold,
        random_state=1337,
    )
    return (
        X_train_pca.astype(np.float32, copy=False),
        X_test_pca.astype(np.float32, copy=False),
        transform.components.astype(np.float32, copy=False),
    )


def _pearson_r(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = y_true.astype(np.float64, copy=False)
    y_pred = y_pred.astype(np.float64, copy=False)
    y_true = y_true - y_true.mean()
    y_pred = y_pred - y_pred.mean()
    denom = float(np.sqrt(np.sum(y_true * y_true) * np.sum(y_pred * y_pred)))
    if denom <= 0:
        return 0.0
    return float(np.clip(np.sum(y_true * y_pred) / denom, -0.999999, 0.999999))


def _ridge_coefficients(X_train: np.ndarray, y_train: np.ndarray, *, alpha: float) -> np.ndarray:
    return np.asarray(_shared_ridge_coefficients(X_train, y_train, alpha=float(alpha)), dtype=np.float32)


def _select_alpha(
    *,
    run_ids: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    Z: np.ndarray,
    alpha_values: np.ndarray,
    max_pcs: int,
    variance_threshold: float,
    random_state: int,
) -> float:
    alpha = _shared_select_alpha_per_target(
        run_ids=run_ids,
        X=X,
        Y=y[:, None],
        Z=Z,
        alpha_values=alpha_values,
        max_pcs=max_pcs,
        variance_threshold=variance_threshold,
        random_state=random_state,
        target_chunk_size=1,
    )[0]
    return float(alpha)


def _window_indices(
    *,
    onset_sec: float,
    offset_sec: float,
    tr_seconds: float,
    n_scans: int,
) -> np.ndarray:
    scan_times = np.arange(int(n_scans), dtype=np.float32) * float(tr_seconds)
    return np.flatnonzero((scan_times >= float(onset_sec) + 2.0) & (scan_times < float(offset_sec) + 12.0))


def _fit_fold_models(
    prepared: PreparedSubjectFamily,
    *,
    feature_array: np.ndarray,
    max_pcs: int,
    variance_threshold: float,
    alpha_values: np.ndarray,
    random_state: int,
) -> dict[int, FoldAttributionModel]:
    run_designs = prepared.operator.transform_feature_matrix(feature_array)
    run_ids = np.concatenate(
        [np.full(len(prepared.y_by_run[run_index]), int(run_index), dtype=np.int64) for run_index in prepared.run_order]
    )
    X = np.vstack([run_designs[run_index] for run_index in prepared.run_order]).astype(np.float32, copy=False)
    y = np.concatenate([prepared.y_by_run[run_index] for run_index in prepared.run_order]).astype(np.float32, copy=False)
    Z = np.vstack([prepared.nuisance_by_run[run_index] for run_index in prepared.run_order]).astype(np.float32, copy=False)

    fold_models: dict[int, FoldAttributionModel] = {}
    spec = LANGUAGE_SPECS[prepared.target_language]
    tr_seconds = float(prepared.operator.tr_seconds)
    for held_out_run in prepared.run_order:
        train_mask = run_ids != int(held_out_run)
        test_mask = run_ids == int(held_out_run)
        alpha = _select_alpha(
            run_ids=run_ids[train_mask],
            X=X[train_mask],
            y=y[train_mask],
            Z=Z[train_mask],
            alpha_values=alpha_values,
            max_pcs=max_pcs,
            variance_threshold=variance_threshold,
            random_state=random_state,
        )
        X_train_res, X_test_res, y_train_res, y_test_res = _residualize_train_test(
            X[train_mask],
            X[test_mask],
            y[train_mask],
            y[test_mask],
            Z[train_mask],
            Z[test_mask],
        )
        X_train_std, X_test_std, scale_std = _standardize_train_test(X_train_res, X_test_res)
        X_train_pca, X_test_pca, components = _fit_pca_train_test(
            X_train_std,
            X_test_std,
            max_pcs=max_pcs,
            variance_threshold=variance_threshold,
        )
        ridge_beta = _ridge_coefficients(X_train_pca, y_train_res, alpha=alpha)
        base_prediction = (X_test_pca @ ridge_beta).astype(np.float32, copy=False)

        run_triplets = prepared.triplets.loc[prepared.triplets["canonical_run"].astype(int) == int(held_out_run)].copy()
        basis = prepared.operator.basis_by_run[int(held_out_run)].astype(np.float32, copy=False)
        row_indices = prepared.operator.row_indices_by_run[int(held_out_run)].astype(np.int64, copy=False)
        window_by_triplet = {
            int(row.triplet_row_index): _window_indices(
                onset_sec=float(getattr(row, spec["onset_col"])),
                offset_sec=float(getattr(row, spec["offset_col"])),
                tr_seconds=tr_seconds,
                n_scans=int(basis.shape[0]),
            )
            for row in run_triplets.itertuples(index=False)
        }
        fold_models[int(held_out_run)] = FoldAttributionModel(
            held_out_run=int(held_out_run),
            row_to_local_index={int(row_index): local_index for local_index, row_index in enumerate(row_indices.tolist())},
            basis=basis,
            scale_std=scale_std.astype(np.float32, copy=False),
            pca_components=components.astype(np.float32, copy=False),
            ridge_beta=ridge_beta.astype(np.float32, copy=False),
            base_prediction=base_prediction,
            actual=y_test_res.astype(np.float32, copy=False),
            window_by_triplet=window_by_triplet,
        )
    return fold_models


def _prediction_contribution(
    fold: FoldAttributionModel,
    *,
    triplet_row_index: int,
    feature_vector: np.ndarray,
) -> np.ndarray:
    local_index = fold.row_to_local_index[int(triplet_row_index)]
    basis_column = fold.basis[:, local_index : local_index + 1].astype(np.float32, copy=False)
    raw_contribution = basis_column * feature_vector[None, :].astype(np.float32, copy=False)
    standardized = raw_contribution / fold.scale_std
    projected = standardized @ fold.pca_components.T
    return (projected @ fold.ridge_beta).astype(np.float32, copy=False)


def _compute_for_representative_setting(
    config: dict[str, Any],
    *,
    representative_row: dict[str, Any],
    triplets: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    model = str(representative_row["model"])
    target_language = str(representative_row["language"])
    roi_family = str(representative_row["roi_family"])
    block = int(representative_row["block"])
    state = str(representative_row["state"])
    condition = "SHARED"

    token_lookup = _load_token_metadata_lookup(config, model=model)
    state_arrays = _load_state_array_lookup(config, model=model, state=state, block=block)
    source_languages = _source_languages(target_language)
    shared_array = ((state_arrays[source_languages[0]] + state_arrays[source_languages[1]]) / 2.0).astype(np.float32, copy=False)

    candidate_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []
    mutation_requests: dict[str, list[dict[str, Any]]] = {language: [] for language in source_languages}

    for triplet in triplets.itertuples(index=False):
        triplet_key = str(triplet.triplet_id)
        triplet_row_index = int(triplet.triplet_row_index)
        canonical_run = int(triplet.canonical_run)
        token_subset_name = str(getattr(triplet, "token_subset_name", _token_subset_name(config)))
        token_subset_rank = int(getattr(triplet, "token_subset_rank", 0))
        other_language_lookup = {
            source_languages[0]: source_languages[1],
            source_languages[1]: source_languages[0],
        }
        for sentence_language in source_languages:
            metadata = token_lookup.get((sentence_language, triplet_key))
            if metadata is None:
                failure_rows.append(
                    {
                        "subject_id": "ALL",
                        "model": model,
                        "target_language": target_language,
                        "roi_family": roi_family,
                        "triplet_id": triplet.triplet_id,
                        "triplet_row_index": triplet_row_index,
                        "canonical_run": canonical_run,
                        "token_subset_name": token_subset_name,
                        "token_subset_rank": token_subset_rank,
                        "representative_block": block,
                        "representative_state": state,
                        "condition": condition,
                        "sentence_language": sentence_language,
                        "word_index": -1,
                        "word_text": "",
                        "word_char_start": -1,
                        "word_char_end": -1,
                        "failure_reason": "missing_token_metadata",
                        "mapping_status": "FAILED",
                    }
                )
                continue
            if not bool(metadata["has_offset_mapping"]):
                failure_rows.append(
                    {
                        "subject_id": "ALL",
                        "model": model,
                        "target_language": target_language,
                        "roi_family": roi_family,
                        "triplet_id": triplet.triplet_id,
                        "triplet_row_index": triplet_row_index,
                        "canonical_run": canonical_run,
                        "token_subset_name": token_subset_name,
                        "token_subset_rank": token_subset_rank,
                        "representative_block": block,
                        "representative_state": state,
                        "condition": condition,
                        "sentence_language": sentence_language,
                        "word_index": -1,
                        "word_text": "",
                        "word_char_start": -1,
                        "word_char_end": -1,
                        "failure_reason": "offset_mapping_failed",
                        "mapping_status": "FAILED",
                    }
                )
                continue
            try:
                words = _segment_words(str(metadata["text"]), language=sentence_language)
            except RuntimeError as exc:
                failure_rows.append(
                    {
                        "subject_id": "ALL",
                        "model": model,
                        "target_language": target_language,
                        "roi_family": roi_family,
                        "triplet_id": triplet.triplet_id,
                        "triplet_row_index": triplet_row_index,
                        "canonical_run": canonical_run,
                        "token_subset_name": token_subset_name,
                        "token_subset_rank": token_subset_rank,
                        "representative_block": block,
                        "representative_state": state,
                        "condition": condition,
                        "sentence_language": sentence_language,
                        "word_index": -1,
                        "word_text": "",
                        "word_char_start": -1,
                        "word_char_end": -1,
                        "failure_reason": str(exc),
                        "mapping_status": "FAILED",
                    }
                )
                continue
            spans = [(int(word["word_char_start"]), int(word["word_char_end"])) for word in words]
            mapped_subwords = map_words_to_subwords(str(metadata["text"]), spans, list(metadata["char_offsets"]))
            for word, subword_indices in zip(words, mapped_subwords, strict=False):
                if not subword_indices:
                    failure_rows.append(
                        {
                            "subject_id": "ALL",
                            "model": model,
                            "target_language": target_language,
                            "roi_family": roi_family,
                            "triplet_id": triplet.triplet_id,
                            "triplet_row_index": triplet_row_index,
                            "canonical_run": canonical_run,
                            "token_subset_name": token_subset_name,
                            "token_subset_rank": token_subset_rank,
                            "representative_block": block,
                            "representative_state": state,
                            "condition": condition,
                            "sentence_language": sentence_language,
                            "word_index": int(word["word_index"]),
                            "word_text": str(word["word_text"]),
                            "word_char_start": int(word["word_char_start"]),
                            "word_char_end": int(word["word_char_end"]),
                            "failure_reason": "offset_mapping_failed",
                            "mapping_status": "FAILED",
                        }
                    )
                    continue
                mutation_requests[sentence_language].append(
                    {
                        "triplet_row_index": triplet_row_index,
                        "triplet_id": triplet.triplet_id,
                        "canonical_run": canonical_run,
                        "token_subset_name": token_subset_name,
                        "token_subset_rank": token_subset_rank,
                        "sentence_language": sentence_language,
                        "other_source_language": other_language_lookup[sentence_language],
                        "word_index": int(word["word_index"]),
                        "word_text": str(word["word_text"]),
                        "word_char_start": int(word["word_char_start"]),
                        "word_char_end": int(word["word_char_end"]),
                        "upos": word["upos"],
                        "token_class": str(word["token_class"]),
                        "mapping_status": "OK",
                        "n_subwords_deleted": int(len(subword_indices)),
                        "mutated_text": _delete_span(
                            str(metadata["text"]),
                            int(word["word_char_start"]),
                            int(word["word_char_end"]),
                            language=sentence_language,
                        ),
                    }
                )

    mutation_vectors_by_language: dict[str, np.ndarray] = {}
    for sentence_language, requests in mutation_requests.items():
        if not requests:
            mutation_vectors_by_language[sentence_language] = np.zeros((0, shared_array.shape[1]), dtype=np.float32)
            continue
        mutation_vectors_by_language[sentence_language] = _run_mutation_encoder(
            config,
            model=model,
            language=sentence_language,
            state=state,
            block=block,
            texts=[str(request["mutated_text"]) for request in requests],
        )

    enriched_candidates: list[dict[str, Any]] = []
    for sentence_language, requests in mutation_requests.items():
        mutated_vectors = mutation_vectors_by_language[sentence_language]
        for request, mutated_vector in zip(requests, mutated_vectors, strict=False):
            triplet_row_index = int(request["triplet_row_index"])
            other_language = str(request["other_source_language"])
            if sentence_language == source_languages[0]:
                mutated_target_vector = ((mutated_vector + state_arrays[other_language][triplet_row_index]) / 2.0).astype(np.float32, copy=False)
            else:
                mutated_target_vector = ((state_arrays[other_language][triplet_row_index] + mutated_vector) / 2.0).astype(np.float32, copy=False)
            enriched_candidates.append({**request, "mutated_target_vector": mutated_target_vector})

    sample_manifest = pd.read_parquet(_sample_manifest_path(config)).copy()
    subject_ids = (
        sample_manifest.loc[
            (sample_manifest["language"].astype(str) == target_language) & sample_manifest["included"].astype(bool),
            "subject_id",
        ]
        .astype(str)
        .sort_values()
        .tolist()
    )
    max_subjects = config.get("attribution", {}).get("max_subjects")
    if max_subjects is not None:
        subject_ids = subject_ids[: int(max_subjects)]

    alpha_values = np.logspace(-2.0, 6.0, 15).astype(np.float64)
    max_pcs = int(config.get("encoding", {}).get("max_pcs", 128))
    variance_threshold = float(config.get("encoding", {}).get("variance_threshold", 0.95))
    random_state = int(config.get("project", {}).get("seed", 1337))
    for subject_id in subject_ids:
        prepared = _prepare_subject_family(
            config,
            target_language=target_language,
            subject_id=subject_id,
            roi_family=roi_family,
        )
        fold_models = _fit_fold_models(
            prepared,
            feature_array=shared_array,
            max_pcs=max_pcs,
            variance_threshold=variance_threshold,
            alpha_values=alpha_values,
            random_state=random_state,
        )
        for candidate in enriched_candidates:
            fold = fold_models[int(candidate["canonical_run"])]
            triplet_row_index = int(candidate["triplet_row_index"])
            base_contribution = _prediction_contribution(
                fold,
                triplet_row_index=triplet_row_index,
                feature_vector=shared_array[triplet_row_index],
            )
            mutated_contribution = _prediction_contribution(
                fold,
                triplet_row_index=triplet_row_index,
                feature_vector=np.asarray(candidate["mutated_target_vector"], dtype=np.float32),
            )
            mutated_prediction = fold.base_prediction - base_contribution + mutated_contribution
            window = fold.window_by_triplet.get(triplet_row_index)
            if window is None or len(window) == 0:
                continue
            attribution_score = float(
                np.mean(
                    (fold.actual[window] - mutated_prediction[window]) ** 2
                    - (fold.actual[window] - fold.base_prediction[window]) ** 2
                )
            )
            candidate_rows.append(
                {
                    "subject_id": str(subject_id),
                    "model": model,
                    "target_language": target_language,
                    "roi_family": roi_family,
                    "triplet_id": candidate["triplet_id"],
                    "triplet_row_index": int(candidate["triplet_row_index"]),
                    "canonical_run": int(candidate["canonical_run"]),
                    "token_subset_name": str(candidate["token_subset_name"]),
                    "token_subset_rank": int(candidate["token_subset_rank"]),
                    "representative_block": block,
                    "representative_state": state,
                    "condition": condition,
                    "sentence_language": candidate["sentence_language"],
                    "word_index": int(candidate["word_index"]),
                    "word_text": str(candidate["word_text"]),
                    "word_char_start": int(candidate["word_char_start"]),
                    "word_char_end": int(candidate["word_char_end"]),
                    "upos": candidate["upos"],
                    "token_class": str(candidate["token_class"]),
                    "attribution_score": attribution_score,
                    "mapping_status": str(candidate["mapping_status"]),
                    "n_subwords_deleted": int(candidate["n_subwords_deleted"]),
                }
            )
        for failure in failure_rows:
            failure["subject_id"] = str(subject_id)

    return (
        pd.DataFrame(candidate_rows, columns=WORD_ATTRIBUTION_COLUMNS),
        pd.DataFrame(failure_rows, columns=MAPPING_FAILURE_COLUMNS),
    )


def compute_word_attributions(config: dict[str, Any]) -> dict[str, Path]:
    triplets = _load_triplets(config)
    representatives = _load_representative_states(config)
    if representatives.empty:
        raise RuntimeError("Representative states table is empty.")
    requested_representative_count = int(len(representatives))
    representatives = _filter_representatives_for_token_scope(config, representatives)
    token_subset_manifest = _load_or_create_token_subset_manifest(config, triplets)
    token_scope_path = _token_scope_representatives_path(config)
    token_scope_path.parent.mkdir(parents=True, exist_ok=True)
    representatives.to_parquet(token_scope_path, index=False)

    if any(language != "ZH" for language in representatives["language"].astype(str).unique().tolist()):
        _append_fallback(
            config,
            component="attribution",
            decision="zh_character_span_segmentation_fallback",
            reason="Source-language Chinese token rows use a deterministic character-span segmentation fallback with auditable subword-offset mapping.",
        )
    _append_fallback(
        config,
        component="attribution",
        decision="latin_word_fallback_segmentation",
        reason="Stanza and spaCy are unavailable locally, so EN/FR attribution uses a regex word fallback with auditable mapping failures.",
    )

    word_frames: list[pd.DataFrame] = []
    failure_frames: list[pd.DataFrame] = []
    for rep in representatives.itertuples(index=False):
        subset_triplets = _subset_triplets_for_target_language(
            triplets,
            token_subset_manifest,
            target_language=str(rep.language),
        )
        frame, failures = _compute_for_representative_setting(
            config,
            representative_row={
                "model": rep.model,
                "language": rep.language,
                "roi_family": rep.roi_family,
                "block": rep.block,
                "state": rep.state,
            },
            triplets=subset_triplets,
        )
        word_frames.append(frame)
        failure_frames.append(failures)

    word_df = pd.concat(word_frames, ignore_index=True) if word_frames else pd.DataFrame(columns=WORD_ATTRIBUTION_COLUMNS)
    failure_df = pd.concat(failure_frames, ignore_index=True) if failure_frames else pd.DataFrame(columns=MAPPING_FAILURE_COLUMNS)

    word_path = _word_attributions_path(config)
    failure_path = _mapping_failures_path(config)
    word_path.parent.mkdir(parents=True, exist_ok=True)
    failure_path.parent.mkdir(parents=True, exist_ok=True)
    _write_parquet(word_df, word_path)
    _write_parquet(failure_df, failure_path)

    provenance_path = _provenance_path(config)
    provenance_path.parent.mkdir(parents=True, exist_ok=True)
    provenance_path.write_text(
        "\n".join(
            [
                "# Word Attributions",
                "",
                f"- word_attributions: `{word_path}`",
                f"- mapping_failures: `{failure_path}`",
                f"- representative_settings: `{_representative_states_path(config)}`",
                f"- token_scope_representatives: `{token_scope_path}`",
                f"- token_subset_manifest: `{_token_subset_manifest_path(config)}`",
                f"- token_subset_name: `{_token_subset_name(config)}`",
                f"- token_subset_triplets_per_run: `{_token_subset_triplets_per_run(config)}`",
                f"- representative_settings_requested: `{requested_representative_count}`",
                f"- representative_settings_used: `{len(representatives)}`",
                f"- n_rows: `{len(word_df)}`",
                f"- n_failures: `{len(failure_df)}`",
                "- default_condition: `SHARED`",
                "- token_scope_mode: `winner_only`",
            ]
        ),
        encoding="utf-8",
    )
    return {
        "word_attributions": word_path,
        "mapping_failures": failure_path,
        "token_scope_representatives": token_scope_path,
        "token_subset_manifest": _token_subset_manifest_path(config),
        "provenance": provenance_path,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute representative-setting SHARED word attributions.")
    parser.add_argument("--config", default="conf/base.yaml", help="Path to sequel config.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    outputs = compute_word_attributions(config)
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
