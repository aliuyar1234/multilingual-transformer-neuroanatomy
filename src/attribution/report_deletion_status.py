from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
import regex

from src.utils.config import load_config


def _local_outputs_root(config: dict[str, Any]) -> Path:
    return Path(config["paths"]["local_outputs_root"]).resolve()


def _local_manifests_root(config: dict[str, Any]) -> Path:
    return Path(config["paths"]["local_manifests_root"]).resolve()


def _representative_token_scope_path(config: dict[str, Any]) -> Path:
    return _local_outputs_root(config) / "group_results" / "representative_states_token_scope.parquet"


def _sample_manifest_path(config: dict[str, Any]) -> Path:
    return _local_manifests_root(config) / "sample_manifest.parquet"


def _deletion_shards_root(config: dict[str, Any]) -> Path:
    return _local_outputs_root(config) / "subject_results" / "deletion_validation_shards"


def _deletion_progress_logs_root(config: dict[str, Any]) -> Path:
    return _local_outputs_root(config) / "provenance" / "deletion_progress_logs"


def _summary_root(config: dict[str, Any]) -> Path:
    return _local_outputs_root(config) / "provenance"


def _slug(value: str) -> str:
    normalized = regex.sub(r"[^A-Za-z0-9]+", "_", str(value).strip())
    normalized = normalized.strip("_")
    return normalized.lower() or "na"


def _summary_stem(
    *,
    models: list[str] | None,
    target_languages: list[str] | None,
    roi_families: list[str] | None,
) -> str:
    suffixes: list[str] = []
    if models:
        suffixes.append(f"model_{'_'.join(sorted(_slug(value) for value in models))}")
    if target_languages:
        suffixes.append(f"language_{'_'.join(sorted(_slug(value) for value in target_languages))}")
    if roi_families:
        suffixes.append(f"roi_{'_'.join(sorted(_slug(value) for value in roi_families))}")
    if not suffixes:
        return "deletion_status_summary"
    return "deletion_status_summary__" + "__".join(suffixes)


def _summary_json_path(
    config: dict[str, Any],
    *,
    models: list[str] | None,
    target_languages: list[str] | None,
    roi_families: list[str] | None,
) -> Path:
    return _summary_root(config) / f"{_summary_stem(models=models, target_languages=target_languages, roi_families=roi_families)}.json"


def _summary_md_path(
    config: dict[str, Any],
    *,
    models: list[str] | None,
    target_languages: list[str] | None,
    roi_families: list[str] | None,
) -> Path:
    return _summary_root(config) / f"{_summary_stem(models=models, target_languages=target_languages, roi_families=roi_families)}.md"


def _setting_label(*, model: str, language: str, roi_family: str, block: int, state: str) -> str:
    return f"{model}/{language}/{roi_family}/b{int(block):02d}/{state}"


def _setting_slug(
    *,
    model: str,
    language: str,
    roi_family: str,
    block: int,
    state: str,
) -> str:
    return (
        f"deletion_validation__{_slug(model)}__{_slug(language)}__{_slug(roi_family)}"
        f"__b{int(block):02d}__{_slug(state)}"
    )


def _shard_path(
    config: dict[str, Any],
    *,
    model: str,
    language: str,
    roi_family: str,
    block: int,
    state: str,
) -> Path:
    return _deletion_shards_root(config) / f"{_setting_slug(model=model, language=language, roi_family=roi_family, block=block, state=state)}.parquet"


def _metadata_path(
    config: dict[str, Any],
    *,
    model: str,
    language: str,
    roi_family: str,
    block: int,
    state: str,
) -> Path:
    return _deletion_shards_root(config) / f"{_setting_slug(model=model, language=language, roi_family=roi_family, block=block, state=state)}__metadata.json"


def _progress_log_path(
    config: dict[str, Any],
    *,
    model: str,
    language: str,
    roi_family: str,
    block: int,
    state: str,
) -> Path:
    return _deletion_progress_logs_root(config) / f"{_setting_slug(model=model, language=language, roi_family=roi_family, block=block, state=state)}.jsonl"


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _load_last_jsonl_event(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    last_line = ""
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                last_line = line
    if not last_line:
        return None
    return json.loads(last_line)


def _load_subject_counts(config: dict[str, Any]) -> dict[str, int]:
    path = _sample_manifest_path(config)
    if not path.exists():
        return {}
    df = pd.read_parquet(path, columns=["language", "subject_id", "included"]).copy()
    included = df.loc[df["included"].astype(bool)].copy()
    counts = included.groupby("language", sort=False)["subject_id"].nunique().astype(int).to_dict()
    return {str(language).upper(): int(count) for language, count in counts.items()}


def _load_expected_settings(config: dict[str, Any]) -> pd.DataFrame:
    path = _representative_token_scope_path(config)
    if not path.exists():
        return pd.DataFrame(columns=["model", "language", "roi_family", "block", "state"])
    df = pd.read_parquet(path).copy()
    return df.loc[:, ["model", "language", "roi_family", "block", "state"]].drop_duplicates().reset_index(drop=True)


def _filter_settings(
    settings: pd.DataFrame,
    *,
    models: list[str] | None,
    target_languages: list[str] | None,
    roi_families: list[str] | None,
) -> pd.DataFrame:
    filtered = settings.copy()
    if models:
        allowed = {str(value).lower() for value in models}
        filtered = filtered.loc[filtered["model"].astype(str).str.lower().isin(allowed)].copy()
    if target_languages:
        allowed = {str(value).upper() for value in target_languages}
        filtered = filtered.loc[filtered["language"].astype(str).str.upper().isin(allowed)].copy()
    if roi_families:
        allowed = {str(value).upper() for value in roi_families}
        filtered = filtered.loc[filtered["roi_family"].astype(str).str.upper().isin(allowed)].copy()
    return filtered.reset_index(drop=True)


def _status_bucket(status: str) -> str:
    normalized = str(status).strip().lower()
    if normalized in {"completed", "completed_legacy"}:
        return "completed"
    if normalized in {"running"}:
        return "running"
    if normalized in {"failed"}:
        return "failed"
    if normalized in {"interrupted"}:
        return "interrupted"
    return "not_started"


def _setting_summary(
    config: dict[str, Any],
    *,
    setting: pd.Series,
    subject_counts: dict[str, int],
    deletion_ks: list[int],
) -> dict[str, Any]:
    model = str(setting["model"])
    language = str(setting["language"]).upper()
    roi_family = str(setting["roi_family"]).upper()
    block = int(setting["block"])
    state = str(setting["state"]).upper()
    label = _setting_label(model=model, language=language, roi_family=roi_family, block=block, state=state)
    shard_path = _shard_path(
        config,
        model=model,
        language=language,
        roi_family=roi_family,
        block=block,
        state=state,
    )
    metadata_path = _metadata_path(
        config,
        model=model,
        language=language,
        roi_family=roi_family,
        block=block,
        state=state,
    )
    progress_path = _progress_log_path(
        config,
        model=model,
        language=language,
        roi_family=roi_family,
        block=block,
        state=state,
    )

    metadata = _load_json(metadata_path)
    last_event = _load_last_jsonl_event(progress_path)
    subject_ids_total = int(subject_counts.get(language, 0))
    completed_subject_ids: list[str] = []
    current_subject_id = None
    rows_written = 0
    resume_count = 0
    subject_progress: dict[str, Any] = {}
    status = "not_started"

    if metadata is not None:
        status = str(metadata.get("status", "running"))
        subject_ids_total = int(metadata.get("subject_ids_total", subject_ids_total))
        completed_subject_ids = [str(value) for value in metadata.get("completed_subject_ids", [])]
        current_subject_id = metadata.get("current_subject_id")
        rows_written = int(metadata.get("rows_written", 0))
        resume_count = int(metadata.get("resume_count", 0))
        subject_progress = dict(metadata.get("subject_progress", {}) or {})
    elif shard_path.exists():
        status = "completed_legacy"
        try:
            rows_written = int(len(pd.read_parquet(shard_path)))
        except Exception:
            rows_written = 0

    completed_subjects = int(len(completed_subject_ids))
    expected_k_slots = int(subject_ids_total * len(deletion_ks))
    progress_ratio = float(rows_written / expected_k_slots) if expected_k_slots > 0 else 0.0
    current_subject_progress = (
        dict(subject_progress.get(str(current_subject_id), {}))
        if current_subject_id is not None
        else {}
    )
    return {
        "model": model,
        "language": language,
        "roi_family": roi_family,
        "block": block,
        "state": state,
        "setting_label": label,
        "status": status,
        "status_bucket": _status_bucket(status),
        "subject_ids_total": subject_ids_total,
        "completed_subjects": completed_subjects,
        "pending_subjects": max(0, int(subject_ids_total - completed_subjects)),
        "rows_written": rows_written,
        "expected_k_slots": expected_k_slots,
        "progress_ratio": progress_ratio,
        "current_subject_id": current_subject_id,
        "current_subject_index": metadata.get("current_subject_index") if metadata else None,
        "current_subject_progress": current_subject_progress,
        "resume_count": resume_count,
        "last_updated_utc": (
            metadata.get("updated_utc")
            if metadata is not None
            else (last_event or {}).get("timestamp_utc")
        ),
        "last_event": last_event,
        "shard_path": str(shard_path),
        "metadata_path": str(metadata_path),
        "progress_log_path": str(progress_path),
    }


def build_deletion_status_summary(
    config: dict[str, Any],
    *,
    models: list[str] | None = None,
    target_languages: list[str] | None = None,
    roi_families: list[str] | None = None,
) -> dict[str, Any]:
    deletion_cfg = dict(config.get("attribution", {}) or {})
    deletion_ks = [int(value) for value in deletion_cfg.get("deletion_ks", [1, 2])]
    settings = _filter_settings(
        _load_expected_settings(config),
        models=models,
        target_languages=target_languages,
        roi_families=roi_families,
    )
    subject_counts = _load_subject_counts(config)
    setting_summaries = [
        _setting_summary(
            config,
            setting=row,
            subject_counts=subject_counts,
            deletion_ks=deletion_ks,
        )
        for _, row in settings.iterrows()
    ]
    bucket_counts = {
        bucket: sum(1 for item in setting_summaries if item["status_bucket"] == bucket)
        for bucket in ["completed", "running", "interrupted", "failed", "not_started"]
    }
    rows_written_total = int(sum(int(item["rows_written"]) for item in setting_summaries))
    expected_k_slots_total = int(sum(int(item["expected_k_slots"]) for item in setting_summaries))
    completed_subjects_total = int(sum(int(item["completed_subjects"]) for item in setting_summaries))
    subject_ids_total = int(sum(int(item["subject_ids_total"]) for item in setting_summaries))
    active_settings = [
        item
        for item in setting_summaries
        if item["status_bucket"] == "running"
    ]
    last_events = [
        item["last_event"]
        for item in setting_summaries
        if item.get("last_event") is not None
    ]
    last_event = max(last_events, key=lambda payload: str(payload.get("timestamp_utc", ""))) if last_events else None
    return {
        "generated_at_utc": pd.Timestamp.utcnow().isoformat(),
        "scope": {
            "models": [str(value) for value in models] if models else [],
            "target_languages": [str(value).upper() for value in target_languages] if target_languages else [],
            "roi_families": [str(value).upper() for value in roi_families] if roi_families else [],
        },
        "settings_total": int(len(setting_summaries)),
        "settings_by_status": bucket_counts,
        "rows_written_total": rows_written_total,
        "expected_k_slots_total": expected_k_slots_total,
        "rows_progress_ratio": (
            float(rows_written_total / expected_k_slots_total)
            if expected_k_slots_total > 0
            else 0.0
        ),
        "completed_subjects_total": completed_subjects_total,
        "subject_ids_total": subject_ids_total,
        "subject_progress_ratio": (
            float(completed_subjects_total / subject_ids_total)
            if subject_ids_total > 0
            else 0.0
        ),
        "configured_deletion_ks": deletion_ks,
        "adaptive_k3_enabled": bool(dict(deletion_cfg.get("adaptive_k3", {}) or {}).get("enabled", True)),
        "active_settings": active_settings,
        "last_event": last_event,
        "settings": setting_summaries,
    }


def render_deletion_status_markdown(summary: dict[str, Any]) -> str:
    settings_by_status = dict(summary.get("settings_by_status", {}) or {})
    active_settings = list(summary.get("active_settings", []) or [])
    lines = [
        "# Deletion Status",
        "",
        f"- generated_at_utc: `{summary.get('generated_at_utc')}`",
        f"- settings_total: `{int(summary.get('settings_total', 0))}`",
        (
            "- settings_by_status: "
            f"`completed={int(settings_by_status.get('completed', 0))}, "
            f"running={int(settings_by_status.get('running', 0))}, "
            f"interrupted={int(settings_by_status.get('interrupted', 0))}, "
            f"failed={int(settings_by_status.get('failed', 0))}, "
            f"not_started={int(settings_by_status.get('not_started', 0))}`"
        ),
        (
            f"- row_progress: `{int(summary.get('rows_written_total', 0))}/"
            f"{int(summary.get('expected_k_slots_total', 0))}` "
            f"(`{float(summary.get('rows_progress_ratio', 0.0)):.2%}`)"
        ),
        (
            f"- subject_progress: `{int(summary.get('completed_subjects_total', 0))}/"
            f"{int(summary.get('subject_ids_total', 0))}` "
            f"(`{float(summary.get('subject_progress_ratio', 0.0)):.2%}`)"
        ),
        (
            f"- configured_ks: "
            f"`{', '.join(str(value) for value in summary.get('configured_deletion_ks', []))}`"
        ),
        f"- adaptive_k3_enabled: `{bool(summary.get('adaptive_k3_enabled', False))}`",
        "",
        "## Active Settings",
        "",
    ]
    if active_settings:
        for item in active_settings:
            current_subject_progress = dict(item.get("current_subject_progress", {}) or {})
            lines.extend(
                [
                    f"- `{item['setting_label']}`",
                    (
                        f"  subject `{item.get('current_subject_index')}/{item.get('subject_ids_total')}` "
                        f"`{item.get('current_subject_id')}`; completed_subjects=`{item.get('completed_subjects')}`; "
                        f"rows_written=`{item.get('rows_written')}`; completed_ks=`{current_subject_progress.get('completed_ks', [])}`"
                    ),
                ]
            )
    else:
        lines.append("- `none`")
    lines.extend(["", "## Next Pending Settings", ""])
    pending = [item for item in summary.get("settings", []) if item.get("status_bucket") == "not_started"]
    if pending:
        for item in pending[:5]:
            lines.append(f"- `{item['setting_label']}`")
    else:
        lines.append("- `none`")
    return "\n".join(lines) + "\n"


def write_deletion_status_artifacts(
    config: dict[str, Any],
    summary: dict[str, Any],
    *,
    models: list[str] | None,
    target_languages: list[str] | None,
    roi_families: list[str] | None,
) -> dict[str, Path]:
    json_path = _summary_json_path(
        config,
        models=models,
        target_languages=target_languages,
        roi_families=roi_families,
    )
    md_path = _summary_md_path(
        config,
        models=models,
        target_languages=target_languages,
        roi_families=roi_families,
    )
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(summary, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(render_deletion_status_markdown(summary), encoding="utf-8")
    return {
        "json": json_path,
        "markdown": md_path,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Report exact deletion-validation progress from checkpoint metadata.")
    parser.add_argument("--config", default="conf/base.yaml", help="Path to sequel config.")
    parser.add_argument("--model", action="append", dest="models", help="Optional model filter; repeatable.")
    parser.add_argument("--target-language", action="append", dest="target_languages", help="Optional target-language filter; repeatable.")
    parser.add_argument("--roi-family", action="append", dest="roi_families", help="Optional ROI-family filter; repeatable.")
    parser.add_argument("--json-only", action="store_true", help="Print JSON summary instead of markdown.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    summary = build_deletion_status_summary(
        config,
        models=args.models,
        target_languages=args.target_languages,
        roi_families=args.roi_families,
    )
    outputs = write_deletion_status_artifacts(
        config,
        summary,
        models=args.models,
        target_languages=args.target_languages,
        roi_families=args.roi_families,
    )
    if args.json_only:
        print(json.dumps(summary, ensure_ascii=True, indent=2))
    else:
        print(render_deletion_status_markdown(summary), end="")
    print(f"summary_json: {outputs['json']}")
    print(f"summary_markdown: {outputs['markdown']}")


if __name__ == "__main__":
    main()
