"""Operational evidence checks with analytically known routing and costs."""
from dataclasses import asdict
import json

import numpy as np
import pytest

from analysis_results import (compare_policies, formula_evidence, run_analysis,
                              sensitivity_analysis, sha256_file, validate_inputs)
from triage import Scenario


def selected(scenario=None):
    return {"binary": {"lower": .5, "upper": .5},
            "hybrid": {"lower": .3, "upper": .8},
            "selection_split": "val", "scenario": asdict(scenario or Scenario())}


def cache(split, labels=(0, 0, 1, 1), scores=(.1, .4, .6, .9)):
    records = []
    for index, (label, score) in enumerate(zip(labels, scores)):
        box = {"class_id": 0, "xyxy": [.1, .1, .8, .8]}
        records.append({"image_id": f"{split}-{index}", "width": 100, "height": 100,
                        "labels": [box] if label else [],
                        "predictions": [{**box, "confidence": score}] if score else []})
    return {"split": split, "class_names": ["finding", "other"], "positive_class_ids": [0],
            "inference": {"conf": .001, "iou": .7, "imgsz": 640},
            "weights_sha256": "a" * 64, "manifest_sha256": "b" * 64, "records": records}


def documents():
    scenario = {"approved": True, "values": asdict(Scenario())}
    freeze = {"schema_version": 1, "selection_split": "val", "selected": selected(),
              "validation_predictions_sha256": "c" * 64, "scenario_sha256": "d" * 64,
              "weights_sha256": "a" * 64, "manifest_sha256": "b" * 64}
    return cache("val"), cache("test"), freeze, scenario


def test_known_four_policy_costs_and_unresolved_human_observations():
    results = compare_policies([0, 0, 1, 1], [.1, .4, .6, .9], selected(), Scenario())
    routine, human, model, hybrid = results
    assert routine["cost_per_1000"] == pytest.approx(50000)
    assert human["cost_per_1000"] == pytest.approx(5625)
    assert model["cost_per_1000"] == pytest.approx(10)
    assert hybrid["cost_per_1000"] == pytest.approx(2822.5)
    assert hybrid["observed_actions"] == [[1, 1, 0], [0, 1, 1]]
    assert np.asarray(hybrid["expected_resolved_confusion_per_1000"]) == pytest.approx(np.array([[877.5, 22.5], [2.5, 97.5]]))
    assert hybrid["daily_reviews"] == 100
    assert hybrid["daily_review_capacity"] == 60
    assert hybrid["daily_backlog_if_no_extra_staff"] == 40
    assert hybrid["reviewers_needed"] == 2
    assert hybrid["within_capacity"] is False
    assert human["reviewers_needed"] == 4
    assert human["uncertainty"]["direct_binary"] is None
    assert hybrid["uncertainty"]["expected_reviewer_outcome_intervals"] is None
    assert model["uncertainty"]["direct_binary"]["observed_confusion"] == [[2, 0], [0, 2]]
    interval = model["uncertainty"]["direct_binary"]["sensitivity"]["wilson_95"]
    assert .3 < interval[0] < .4 and interval[1] == pytest.approx(1)


def test_formula_worked_numbers_are_scenario_based():
    result = formula_evidence(Scenario())
    assert result["review_cost"]["eur_per_review"] == 2
    assert result["capacity"]["reviews_per_day"] == 60
    assert result["capacity"]["maximum_review_fraction"] == .3
    assert result["hypothetical_human_all"]["fp_per_1000"] == pytest.approx(45)
    assert result["hypothetical_human_all"]["fn_per_1000"] == pytest.approx(5)
    assert result["hypothetical_human_all"]["daily_review_hours"] == pytest.approx(20 / 3)


def test_exact_boundaries_route_consistently_without_simulated_review_outcomes():
    policy = selected()
    policy["hybrid"] = {"lower": .3, "upper": .8}
    result = compare_policies([0, 0, 1, 1], [.3, .8, .3, .8], policy, Scenario())[3]
    assert result["observed_actions"] == [[0, 1, 1], [0, 1, 1]]
    assert result["uncertainty"]["direct_binary"] is None


@pytest.mark.parametrize("change, message", [
    (lambda v, t, f, s: t.update(weights_sha256="e" * 64), "weights_sha256"),
    (lambda v, t, f, s: t.update(manifest_sha256="e" * 64), "manifest_sha256"),
    (lambda v, t, f, s: t.update(inference={"conf": .2}), "inference"),
    (lambda v, t, f, s: t.update(positive_class_ids=[1]), "positive_class_ids"),
    (lambda v, t, f, s: f.update(selection_split="test"), "validation"),
    (lambda v, t, f, s: f.update(validation_predictions_sha256="e" * 64), "supplied file"),
    (lambda v, t, f, s: s.update(approved=False), "approved"),
    (lambda v, t, f, s: t["records"][0].update(image_id="val-0"), "overlapping"),
    (lambda v, t, f, s: f["selected"]["scenario"].update(prevalence=.2), "approved scenario"),
])
def test_mismatched_or_unapproved_evidence_is_rejected(change, message):
    docs = documents()
    change(*docs)
    with pytest.raises(ValueError, match=message):
        validate_inputs(*docs, validation_sha256="c" * 64, scenario_sha256="d" * 64)


def test_sensitivity_selects_only_with_validation_and_keeps_test_thresholds_frozen():
    val_y, val_s = np.array([0, 1]), np.array([.4, .6])
    test_y, test_s = np.array([0, 1]), np.array([.9, .1])
    calls = []
    def selector(labels, scores, scenario):
        assert labels is val_y and scores is val_s
        calls.append(asdict(scenario))
        result = selected(scenario)
        result["binary"] = {"lower": .99, "upper": .99}
        result["hybrid"] = {"lower": .99, "upper": .99}
        return result
    result = sensitivity_analysis(val_y, val_s, test_y, test_s, selected(), Scenario(), selector=selector)
    assert len(result["scenarios"]) == 19
    assert len(calls) == 14  # Identical baseline scenarios are memoized.
    for variant in result["scenarios"]:
        assert variant["frozen_policy_test_projection"][2]["lower"] == .5
        assert variant["validation_only_reselection"]["binary"]["lower"] == .99
        assert variant["validation_only_reselection"]["test_evaluation"] is None
    hours = [r for r in result["scenarios"] if r["factor"] == "staff_hours"]
    assert len({r["frozen_policy_test_projection"][3]["cost_per_1000"] for r in hours}) == 1
    inference = [r for r in result["scenarios"] if r["factor"] == "inference_allowance"]
    assert [r["frozen_policy_test_projection"][2]["inference_cost_per_1000"] for r in inference] == [0., 10., 50.]
    assert all(r["frozen_policy_test_projection"][1]["inference_cost_per_1000"] == 0 for r in inference)


def test_file_pipeline_links_frozen_inputs_and_writes_no_test_sweep(tmp_path):
    validation, test, freeze, scenario = documents()
    paths = [tmp_path / name for name in ("validation.json", "test.json", "frozen.json", "scenario.json")]
    paths[0].write_text(json.dumps(validation))
    paths[1].write_text(json.dumps(test))
    paths[3].write_text(json.dumps(scenario))
    freeze["validation_predictions_sha256"] = sha256_file(paths[0])
    freeze["scenario_sha256"] = sha256_file(paths[3])
    paths[2].write_text(json.dumps(freeze))
    output = tmp_path / "output"
    result = run_analysis(*paths, output, plots=False, selector=lambda y, s, scenario: selected(scenario))
    assert result["sample"]["test_images"] == 4
    assert result["sample"]["observed_test_prevalence"] == .5
    assert result["sample"]["projected_operating_prevalence"] == .1
    assert result["provenance"]["test_predictions_sha256"] == sha256_file(paths[1])
    assert (output / "validation_binary_cost_sweep.csv").is_file()
    assert (output / "operational_comparison.json").is_file()
    assert not list(output.glob("*test*sweep*"))
    # Changed validation scores invalidate the freeze before creating outputs.
    validation["records"][0]["predictions"][0]["confidence"] = .2
    paths[0].write_text(json.dumps(validation))
    with pytest.raises(ValueError, match="supplied file"):
        run_analysis(*paths, tmp_path / "invalid", plots=False)
    assert not (tmp_path / "invalid").exists()
