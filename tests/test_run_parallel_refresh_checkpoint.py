from __future__ import annotations

import importlib


def test_run_parallel_refresh_checkpoint_advances_lanes_until_targets(monkeypatch) -> None:
    module = importlib.import_module("src.encoding.run_parallel_refresh_checkpoint")

    counts = {"en": 0, "fr": 0, "zh": 0}
    targets = {"en": 2, "fr": 1, "zh": 1}

    def fake_build_lanes(config, *, languages, output_tag_template):
        _ = (config, output_tag_template)
        return tuple(
            module.RefreshLane(
                language=language,
                output_tag=f"tag_{language}",
                target_subjects=targets[language],
            )
            for language in languages
        )

    def fake_lane_subjects_completed(config, *, model, lane):
        _ = (config, model)
        return counts[lane.language]

    def fake_run_lane_once(**kwargs):
        lane = kwargs["lane"]
        if counts[lane.language] < targets[lane.language]:
            counts[lane.language] += 1
        return {"subjects_total": counts[lane.language], "output_tag": lane.output_tag}

    synth_calls: list[tuple[str, int]] = []

    def fake_synthesize_lane_partial(*, config_path, lane, model, completed_subjects):
        _ = (config_path, model)
        synth_calls.append((lane.language, completed_subjects))
        return {"output_tag": f"synth_{lane.language}_{completed_subjects}"}

    monkeypatch.setattr(module, "_build_lanes", fake_build_lanes)
    monkeypatch.setattr(module, "_lane_subjects_completed", fake_lane_subjects_completed)
    monkeypatch.setattr(module, "_run_lane_once", fake_run_lane_once)
    monkeypatch.setattr(module, "_synthesize_lane_partial", fake_synthesize_lane_partial)

    summary = module.run_parallel_refresh_checkpoint(
        {},
        config_path="conf/base.yaml",
        model="nllb",
        languages=("en", "fr", "zh"),
        layers=(0, 1, 2),
        output_tag_template="ignored_{language}",
        max_parallel=1,
        mismatch_shuffles=1,
        pass_timeout_ms=1,
        passes_per_round=1,
        max_rounds=5,
        stop_after_no_progress_rounds=1,
        synthesize_partials=True,
    )

    assert summary["status"] == "completed"
    assert summary["final_subject_counts"] == targets
    assert len(summary["rounds"]) == 2
    assert synth_calls == [("en", 1), ("fr", 1), ("zh", 1), ("en", 2)]
