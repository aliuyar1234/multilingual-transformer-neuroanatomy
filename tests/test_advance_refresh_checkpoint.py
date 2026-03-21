from __future__ import annotations

import importlib
from pathlib import Path

from tests.helpers import make_temp_test_dir


def test_advance_refresh_checkpoint_clears_stale_metadata_and_logs_failure(monkeypatch) -> None:
    module = importlib.import_module("src.encoding.advance_refresh_checkpoint")

    base_path = make_temp_test_dir("advance_refresh_checkpoint")
    outputs_root = base_path / "outputs"
    stats_root = outputs_root / "stats"
    stats_root.mkdir(parents=True, exist_ok=True)

    subject_path = stats_root / "nllb_subject_level_roi_results__tag_zh.parquet"
    metadata_path = stats_root / "nllb_subject_level_roi_results__tag_zh__metadata.json"
    metadata_path.write_text("{}", encoding="utf-8")

    monkeypatch.setattr(module, "_subject_output_path", lambda config, *, model, output_tag: subject_path)
    monkeypatch.setattr(module, "_subject_metadata_path", lambda config, *, model, output_tag: metadata_path)
    monkeypatch.setattr(module, "_count_subjects", lambda path: 0)

    def fake_run_refresh_baseline(*args, **kwargs):
        raise module.subprocess.CalledProcessError(
            returncode=1,
            cmd=["python"],
            stderr="synthetic failure",
        )

    monkeypatch.setattr(module, "run_refresh_baseline", fake_run_refresh_baseline)

    config = {"paths": {"local_outputs_root": str(outputs_root)}}
    summary = module.advance_refresh_checkpoint(
        config,
        model="nllb",
        language="zh",
        layers=(0, 1, 2),
        mismatch_shuffles=1,
        output_tag="tag_zh",
        pass_timeout_ms=1000,
        passes=1,
    )

    try:
        assert not metadata_path.exists()
        assert summary["passes"][0]["status"] == "failed"
        assert "cleanup" in summary["passes"][0]
        assert "synthetic failure" in summary["passes"][0]["error"]
    finally:
        for path in sorted(base_path.rglob("*"), reverse=True):
            if path.is_file():
                path.unlink(missing_ok=True)
            elif path.is_dir():
                path.rmdir()
