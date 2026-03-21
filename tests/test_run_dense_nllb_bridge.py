from __future__ import annotations

import importlib


def test_run_dense_nllb_bridge_uses_short_defaults(monkeypatch, capsys) -> None:
    module = importlib.import_module("src.encoding.run_dense_nllb_bridge")

    calls: list[dict[str, object]] = []

    def fake_load_config(path):
        return {"config_path": path}

    def fake_run_parallel_refresh_checkpoint(config, **kwargs):
        calls.append({"config": config, **kwargs})
        return {"status": "completed", "final_subject_counts": {"en": 31, "fr": 28, "zh": 35}}

    monkeypatch.setattr(module, "load_config", fake_load_config)
    monkeypatch.setattr(module, "run_parallel_refresh_checkpoint", fake_run_parallel_refresh_checkpoint)
    monkeypatch.setattr(
        "sys.argv",
        ["run_dense_nllb_bridge"],
    )

    module.main()
    captured = capsys.readouterr()

    assert len(calls) == 1
    call = calls[0]
    assert call["config"] == {"config_path": "conf/base.yaml"}
    assert call["model"] == "nllb"
    assert call["languages"] == ("en", "fr", "zh")
    assert call["layers"] == (0, 1, 2, 3, 5, 7, 9, 10, 11, 12)
    assert call["output_tag_template"] == "sequel_refresh_nllb_missing_layers_{language}"
    assert call["max_parallel"] == 3
    assert call["mismatch_shuffles"] == 1
    assert call["pass_timeout_ms"] == 1800000
    assert call["synthesize_partials"] is True
    assert '"status": "completed"' in captured.out
