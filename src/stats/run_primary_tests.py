from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from src.utils.config import load_config


ATTN_FAMILY = ("ATTN", "POST_ATTN")
FFN_FAMILY = ("FFN", "OUTPUT")
LOCKED_MODELS = ("xlmr", "nllb")
LOCKED_LANGUAGES = ("EN", "FR", "ZH")
LOCKED_HYPOTHESES = ("H1_semantic_ffn_preference", "H2_auditory_attention_preference")


def _stable_seed_offset(*parts: str) -> int:
    payload = "|".join(parts)
    return sum((index + 1) * ord(char) for index, char in enumerate(payload)) % 100_000


def _local_outputs_root(config: dict[str, Any]) -> Path:
    return Path(config["paths"]["local_outputs_root"]).resolve()


def _subject_results_path(config: dict[str, Any]) -> Path:
    return _local_outputs_root(config) / "subject_results" / "state_sweep_subject_results.parquet"


def _primary_effect_tables_path(config: dict[str, Any]) -> Path:
    return _local_outputs_root(config) / "group_results" / "primary_effect_tables.parquet"


def _primary_subject_effects_path(config: dict[str, Any]) -> Path:
    return _local_outputs_root(config) / "group_results" / "primary_subject_effects.parquet"


def _primary_permutation_summary_path(config: dict[str, Any]) -> Path:
    return _local_outputs_root(config) / "group_results" / "primary_permutation_summary.parquet"


def _primary_bootstrap_summary_path(config: dict[str, Any]) -> Path:
    return _local_outputs_root(config) / "group_results" / "primary_bootstrap_summary.parquet"


def _table03_path(config: dict[str, Any]) -> Path:
    return _local_outputs_root(config) / "tables" / "table03_primary_confirmatory_stats.csv"


def _project_seed(config: dict[str, Any]) -> int:
    return int(config.get("project", {}).get("seed", 1337))


def _expected_models(config: dict[str, Any], expected_models: Sequence[str] | None) -> tuple[str, ...]:
    if expected_models is not None:
        return tuple(dict.fromkeys(str(model).lower() for model in expected_models))
    configured = tuple(str(model).lower() for model in config.get("models", {}).keys())
    return configured or LOCKED_MODELS


def _expected_languages(config: dict[str, Any], expected_languages: Sequence[str] | None) -> tuple[str, ...]:
    if expected_languages is not None:
        return tuple(dict.fromkeys(str(language).upper() for language in expected_languages))
    configured = tuple(str(language).upper() for language in config.get("dataset", {}).get("languages", ()))
    return configured or LOCKED_LANGUAGES


def _holm_adjust(p_values: np.ndarray) -> np.ndarray:
    order = np.argsort(p_values)
    adjusted = np.empty_like(p_values, dtype=np.float64)
    m = len(p_values)
    running = 0.0
    for rank, index in enumerate(order):
        scaled = float(p_values[index]) * (m - rank)
        running = max(running, scaled)
        adjusted[index] = min(1.0, running)
    return adjusted


def _sign_flip_p(effect_values: np.ndarray, *, permutations: int, seed: int) -> tuple[float, np.ndarray]:
    rng = np.random.default_rng(seed)
    observed = float(effect_values.mean())
    signs = rng.choice(np.array([-1.0, 1.0], dtype=np.float64), size=(int(permutations), len(effect_values)))
    permuted = (signs * effect_values[None, :]).mean(axis=1)
    p_value = float((1 + np.sum(permuted >= observed)) / (1 + len(permuted)))
    return p_value, permuted.astype(np.float64, copy=False)


def _bootstrap_ci(effect_values: np.ndarray, *, bootstraps: int, seed: int) -> tuple[float, float, np.ndarray]:
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(effect_values), size=(int(bootstraps), len(effect_values)))
    samples = effect_values[indices].mean(axis=1)
    return (
        float(np.quantile(samples, 0.025)),
        float(np.quantile(samples, 0.975)),
        samples.astype(np.float64, copy=False),
    )


def _subject_effects(subject_df: pd.DataFrame) -> pd.DataFrame:
    shared = subject_df.loc[subject_df["condition"].astype(str) == "SHARED"].copy()
    specific = subject_df.loc[subject_df["condition"].astype(str) == "SPECIFIC"].copy()
    key_cols = ["subject_id", "language", "model", "block", "state", "roi", "roi_family"]
    merged = shared.merge(
        specific.loc[:, key_cols + ["z_mean"]].rename(columns={"z_mean": "z_mean_specific"}),
        on=key_cols,
        how="inner",
    )
    merged["delta_z"] = merged["z_mean"].astype(float) - merged["z_mean_specific"].astype(float)
    family_df = (
        merged.groupby(["subject_id", "language", "model", "roi_family", "block", "state"], as_index=False)["delta_z"]
        .mean()
        .sort_values(["language", "model", "subject_id", "roi_family", "block", "state"])
        .reset_index(drop=True)
    )
    return family_df


def _empty_sorted_frame(columns: list[str], sort_columns: list[str]) -> pd.DataFrame:
    _ = sort_columns
    return pd.DataFrame(columns=columns)


def _hypothesis_rows(
    family_df: pd.DataFrame,
    *,
    permutations: int,
    bootstraps: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary_rows: list[dict[str, object]] = []
    subject_rows: list[dict[str, object]] = []
    permutation_rows: list[dict[str, object]] = []
    bootstrap_rows: list[dict[str, object]] = []

    for model in sorted(family_df["model"].astype(str).unique().tolist()):
        for language in sorted(family_df["language"].astype(str).unique().tolist()):
            semantic_df = family_df.loc[
                (family_df["model"].astype(str) == model)
                & (family_df["language"].astype(str) == language)
                & (family_df["roi_family"].astype(str) == "SEMANTIC")
            ].copy()
            auditory_df = family_df.loc[
                (family_df["model"].astype(str) == model)
                & (family_df["language"].astype(str) == language)
                & (family_df["roi_family"].astype(str) == "AUDITORY")
            ].copy()
            for hypothesis, subset, positive_family, negative_family in (
                ("H1_semantic_ffn_preference", semantic_df, FFN_FAMILY, ATTN_FAMILY),
                ("H2_auditory_attention_preference", auditory_df, ATTN_FAMILY, FFN_FAMILY),
            ):
                positive = (
                    subset.loc[subset["state"].astype(str).isin(positive_family)]
                    .groupby("subject_id", as_index=False)["delta_z"]
                    .mean()
                    .rename(columns={"delta_z": "positive_mean"})
                )
                negative = (
                    subset.loc[subset["state"].astype(str).isin(negative_family)]
                    .groupby("subject_id", as_index=False)["delta_z"]
                    .mean()
                    .rename(columns={"delta_z": "negative_mean"})
                )
                effects = positive.merge(negative, on="subject_id", how="inner")
                effects["effect_value"] = effects["positive_mean"] - effects["negative_mean"]
                effect_values = effects["effect_value"].to_numpy(dtype=np.float64)
                if len(effect_values) == 0:
                    continue
                p_value, permuted = _sign_flip_p(
                    effect_values,
                    permutations=permutations,
                    seed=seed + _stable_seed_offset(model, language, hypothesis, "perm"),
                )
                ci_low, ci_high, boot_samples = _bootstrap_ci(
                    effect_values,
                    bootstraps=bootstraps,
                    seed=seed + _stable_seed_offset(hypothesis, language, model, "boot") + 17,
                )
                mean_effect = float(effect_values.mean())
                std_effect = float(effect_values.std(ddof=1)) if len(effect_values) > 1 else 0.0
                se = float(std_effect / np.sqrt(len(effect_values))) if len(effect_values) > 0 else np.nan
                dz = float(mean_effect / std_effect) if std_effect > 0 else 0.0
                summary_rows.append(
                    {
                        "model": model,
                        "language": language,
                        "hypothesis": hypothesis,
                        "effect_mean": mean_effect,
                        "ci_low": ci_low,
                        "ci_high": ci_high,
                        "p_raw": p_value,
                        "p_holm": np.nan,
                        "n_subjects": int(len(effect_values)),
                        "permutation_count": int(permutations),
                        "bootstrap_count": int(bootstraps),
                        "se": se,
                        "dz": dz,
                    }
                )
                for effect_row in effects.itertuples(index=False):
                    subject_rows.append(
                        {
                            "model": model,
                            "language": language,
                            "hypothesis": hypothesis,
                            "subject_id": str(effect_row.subject_id),
                            "positive_mean": float(effect_row.positive_mean),
                            "negative_mean": float(effect_row.negative_mean),
                            "effect_value": float(effect_row.effect_value),
                        }
                    )
                permutation_rows.append(
                    {
                        "model": model,
                        "language": language,
                        "hypothesis": hypothesis,
                        "observed_mean": mean_effect,
                        "permuted_mean": float(np.mean(permuted)),
                        "permuted_std": float(np.std(permuted, ddof=1)) if len(permuted) > 1 else 0.0,
                        "p_raw": p_value,
                        "n_subjects": int(len(effect_values)),
                    }
                )
                bootstrap_rows.append(
                    {
                        "model": model,
                        "language": language,
                        "hypothesis": hypothesis,
                        "ci_low": ci_low,
                        "ci_high": ci_high,
                        "bootstrap_mean": float(np.mean(boot_samples)),
                        "bootstrap_std": float(np.std(boot_samples, ddof=1)) if len(boot_samples) > 1 else 0.0,
                        "n_subjects": int(len(effect_values)),
                    }
                )

    summary_columns = [
        "model",
        "language",
        "hypothesis",
        "effect_mean",
        "ci_low",
        "ci_high",
        "p_raw",
        "p_holm",
        "n_subjects",
        "permutation_count",
        "bootstrap_count",
        "se",
        "dz",
    ]
    subject_columns = [
        "model",
        "language",
        "hypothesis",
        "subject_id",
        "positive_mean",
        "negative_mean",
        "effect_value",
    ]
    permutation_columns = [
        "model",
        "language",
        "hypothesis",
        "observed_mean",
        "permuted_mean",
        "permuted_std",
        "p_raw",
        "n_subjects",
    ]
    bootstrap_columns = [
        "model",
        "language",
        "hypothesis",
        "ci_low",
        "ci_high",
        "bootstrap_mean",
        "bootstrap_std",
        "n_subjects",
    ]

    if summary_rows:
        summary_df = pd.DataFrame(summary_rows).sort_values(["model", "language", "hypothesis"]).reset_index(drop=True)
    else:
        summary_df = _empty_sorted_frame(summary_columns, ["model", "language", "hypothesis"])
    if not summary_df.empty:
        summary_df["p_holm"] = _holm_adjust(summary_df["p_raw"].to_numpy(dtype=np.float64))
    return (
        summary_df,
        (
            pd.DataFrame(subject_rows).sort_values(["model", "language", "hypothesis", "subject_id"]).reset_index(drop=True)
            if subject_rows
            else _empty_sorted_frame(subject_columns, ["model", "language", "hypothesis", "subject_id"])
        ),
        (
            pd.DataFrame(permutation_rows).sort_values(["model", "language", "hypothesis"]).reset_index(drop=True)
            if permutation_rows
            else _empty_sorted_frame(permutation_columns, ["model", "language", "hypothesis"])
        ),
        (
            pd.DataFrame(bootstrap_rows).sort_values(["model", "language", "hypothesis"]).reset_index(drop=True)
            if bootstrap_rows
            else _empty_sorted_frame(bootstrap_columns, ["model", "language", "hypothesis"])
        ),
    )


def _validate_primary_coverage(
    subject_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    *,
    expected_models: Sequence[str],
    expected_languages: Sequence[str],
) -> None:
    observed_models = tuple(sorted(subject_df["model"].astype(str).str.lower().unique().tolist()))
    observed_languages = tuple(sorted(subject_df["language"].astype(str).str.upper().unique().tolist()))
    expected_model_set = tuple(sorted(dict.fromkeys(str(model).lower() for model in expected_models)))
    expected_language_set = tuple(sorted(dict.fromkeys(str(language).upper() for language in expected_languages)))
    if observed_models != expected_model_set:
        raise RuntimeError(
            "Primary tests model coverage mismatch: "
            f"expected {list(expected_model_set)}, got {list(observed_models)}."
        )
    if observed_languages != expected_language_set:
        raise RuntimeError(
            "Primary tests language coverage mismatch: "
            f"expected {list(expected_language_set)}, got {list(observed_languages)}."
        )
    expected_rows = len(expected_model_set) * len(expected_language_set) * len(LOCKED_HYPOTHESES)
    if len(summary_df) != expected_rows:
        raise RuntimeError(
            "Primary tests did not cover every locked model-language hypothesis cell: "
            f"expected {expected_rows}, got {len(summary_df)}."
        )
    observed_cells = {
        (str(row.model).lower(), str(row.language).upper(), str(row.hypothesis))
        for row in summary_df.itertuples(index=False)
    }
    expected_cells = {
        (model, language, hypothesis)
        for model in expected_model_set
        for language in expected_language_set
        for hypothesis in LOCKED_HYPOTHESES
    }
    if observed_cells != expected_cells:
        missing = sorted(expected_cells.difference(observed_cells))
        extra = sorted(observed_cells.difference(expected_cells))
        raise RuntimeError(
            "Primary tests coverage cells mismatch: "
            f"missing={missing[:3]}{' ...' if len(missing) > 3 else ''}, "
            f"extra={extra[:3]}{' ...' if len(extra) > 3 else ''}."
        )


def run_primary_tests(
    config: dict[str, Any],
    *,
    expected_models: Sequence[str] | None = None,
    expected_languages: Sequence[str] | None = None,
) -> dict[str, Path]:
    subject_results_path = _subject_results_path(config)
    if not subject_results_path.exists():
        raise FileNotFoundError(
            f"State sweep subject results missing: {subject_results_path}. Run src.encoding.run_state_sweep first."
        )
    subject_df = pd.read_parquet(subject_results_path).copy()
    family_df = _subject_effects(subject_df)
    permutations = int(config.get("stats", {}).get("permutations", 10000))
    bootstraps = int(config.get("stats", {}).get("bootstraps", 10000))
    seed = _project_seed(config)
    summary_df, subject_effects_df, permutation_df, bootstrap_df = _hypothesis_rows(
        family_df,
        permutations=permutations,
        bootstraps=bootstraps,
        seed=seed,
    )
    _validate_primary_coverage(
        subject_df,
        summary_df,
        expected_models=_expected_models(config, expected_models),
        expected_languages=_expected_languages(config, expected_languages),
    )

    summary_path = _primary_effect_tables_path(config)
    subject_effects_path = _primary_subject_effects_path(config)
    permutation_path = _primary_permutation_summary_path(config)
    bootstrap_path = _primary_bootstrap_summary_path(config)
    table03_path = _table03_path(config)

    for path in (summary_path, subject_effects_path, permutation_path, bootstrap_path, table03_path):
        path.parent.mkdir(parents=True, exist_ok=True)

    summary_df.to_parquet(summary_path, index=False)
    subject_effects_df.to_parquet(subject_effects_path, index=False)
    permutation_df.to_parquet(permutation_path, index=False)
    bootstrap_df.to_parquet(bootstrap_path, index=False)
    summary_df.loc[
        :,
        [
            "model",
            "language",
            "hypothesis",
            "effect_mean",
            "ci_low",
            "ci_high",
            "p_raw",
            "p_holm",
            "n_subjects",
            "permutation_count",
            "bootstrap_count",
        ],
    ].to_csv(table03_path, index=False)

    return {
        "primary_effect_tables": summary_path,
        "primary_subject_effects": subject_effects_path,
        "primary_permutation_summary": permutation_path,
        "primary_bootstrap_summary": bootstrap_path,
        "table03": table03_path,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the primary confirmatory H1/H2 tests.")
    parser.add_argument("--config", default="conf/base.yaml", help="Path to sequel config.")
    parser.add_argument("--models", nargs="+", default=None, choices=["xlmr", "nllb"])
    parser.add_argument("--languages", nargs="+", default=None, choices=["EN", "FR", "ZH", "en", "fr", "zh"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    outputs = run_primary_tests(
        config,
        expected_models=None if args.models is None else tuple(args.models),
        expected_languages=None if args.languages is None else tuple(args.languages),
    )
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
