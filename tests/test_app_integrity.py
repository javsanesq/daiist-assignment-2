"""Synthetic saved-run integrity and UI checks; no model download or data access."""
import copy
import math
import re
from dataclasses import asdict
from types import SimpleNamespace

from PIL import Image, ImageColor
import pytest

import app
from detection_metrics import fixed_threshold_metrics, image_scores, remap_predictions
from pipeline import canonical_sha256, empty_prediction_ap, observed_binary_metrics, write_json
from triage import Scenario, evaluate_policy


NAMES = ["Finding A", "Finding B"]
BOX = {"class_id": 0, "xyxy": [.1, .1, .8, .8]}


def save(root, relative, value):
    write_json(root / relative, value)


def resign_artifacts(root):
    """Let semantic tests bypass the file-digest gate deliberately."""
    path = root / "results/evaluation_provenance.json"
    document = app.read_json(path)
    document["artifact_sha256"] = {name: app.digest(root / "results" / name) for name in app.COMPLETE_ARTIFACTS}
    write_json(path, document)


def change(root, relative, mutate, *, resign=False):
    path = root / relative
    document = app.read_json(path)
    mutate(document)
    write_json(path, document)
    if resign:
        resign_artifacts(root)


@pytest.fixture
def saved_run(tmp_path):
    """A complete artificial run, with no dataset folder or Roboflow key."""
    root = tmp_path
    config = {"dataset": {"approved": True, "workspace": "synthetic", "project": "fixture", "version": 1,
                          "root": "raw-images-do-not-exist", "class_names": NAMES,
                          "positive_class_ids": [0, 1], "accepted_limitations": ["Synthetic test fixture only."]},
              "scenario_path": "scenario.json", "coco_semantic_mapping": {},
              "inference": {"conf": .001, "iou": .7, "imgsz": 768, "max_det": 300}}
    scenario = Scenario()
    save(root, "project_config.json", config)
    save(root, "scenario.json", {"approved": True, "policy_rule_approved": True, "values": asdict(scenario)})
    (root / "weights").mkdir()
    (root / "weights/best.pt").write_bytes(b"synthetic stub, never passed to a real model")
    manifest = {"schema_version": 1, "class_names": NAMES,
                "dataset": {k: config["dataset"][k] for k in ("workspace", "project", "version", "root")}, "splits": {}}
    for split in ("train", "val", "test"):
        manifest["splits"][split] = [
            {"image_id": f"{split}-{i}", "width": 100, "height": 100,
             "labels": [] if i == 0 else [{**copy.deepcopy(BOX), "class_id": i - 1}]}
            for i in range(3)]
    save(root, "results/dataset_manifest.json", manifest)
    bindings = {"weights_sha256": app.digest(root / "weights/best.pt"),
                "manifest_sha256": app.digest(root / "results/dataset_manifest.json"),
                "config_sha256": app.digest(root / "project_config.json")}

    def cache(split):
        records = copy.deepcopy(manifest["splits"][split])
        for i, row in enumerate(records):
            row["predictions"] = [] if i == 0 else [{**copy.deepcopy(row["labels"][0]), "confidence": .8 if i == 1 else .4}]
        return {"schema_version": 1, "split": split, "class_names": NAMES, "positive_class_ids": [0, 1],
                "inference": config["inference"], **bindings,
                "split_manifest_sha256": canonical_sha256(manifest["splits"][split]), "records": records}

    validation, test = cache("val"), cache("test")
    save(root, "results/validation_predictions.json", validation)
    _, labels, scores = image_scores(validation["records"], NAMES, [0, 1])
    binary = evaluate_policy(labels, scores, .5, .5, scenario)
    selected = {"binary": binary, "hybrid": evaluate_policy(labels, scores, .3, .8, scenario),
                "binary_sweep": [evaluate_policy(labels, scores, t, t, scenario) for t in (0, .5, 1.001)],
                "selection_split": "val", "scenario": asdict(scenario)}
    frozen = {"schema_version": 1, "selection_split": "val", "selected": selected, **bindings,
              "validation_predictions_sha256": app.digest(root / "results/validation_predictions.json"),
              "scenario_sha256": app.digest(root / "scenario.json")}
    save(root, "results/frozen_policy.json", frozen)
    freeze_digest = app.digest(root / "results/frozen_policy.json")
    test["frozen_policy_sha256"] = freeze_digest
    save(root, "results/test_predictions.json", test)
    native = {"overall": {"map50": .4, "map50_95": .2}, "split": "test", "inference": config["inference"],
              "per_class": [{"class_id": i, "class_name": name, "instances": 1, "ap50": .4, "ap50_95": .2}
                            for i, name in enumerate(NAMES)]}
    fixed = fixed_threshold_metrics(test["records"], NAMES, .5)
    metrics = {"schema_version": 1, "split": "test", "native": native, "fixed_at_binary_threshold": fixed,
               **bindings, "frozen_policy_sha256": freeze_digest}
    save(root, "results/test_metrics.json", metrics)
    save(root, "results/validation_metrics.json", {"schema_version": 1, "split": "val", "synthetic": True})
    coco_names = [f"COCO category {i}" for i in range(80)]
    raw = copy.deepcopy(test)
    raw.update(weights_sha256="e" * 64, prediction_class_names=coco_names, semantic_mapping={})
    for row in raw["records"]:
        row["predictions"] = [{**copy.deepcopy(BOX), "class_id": 79, "confidence": .9}]
    save(root, "results/coco_predictions_raw.json", raw)
    mapped = {**raw, "records": remap_predictions(raw["records"], coco_names, NAMES, {}),
              "prediction_class_names": NAMES, "source_prediction_class_names": coco_names}
    save(root, "results/coco_predictions_target.json", mapped)
    nothing = {**test, "records": [{**r, "predictions": []} for r in test["records"]],
               "weights_sha256": None, "model": "predict_nothing"}
    save(root, "results/predict_nothing_predictions.json", nothing)
    empty = fixed_threshold_metrics(nothing["records"], NAMES, .5)
    models = {"finetuned": {"name": "Synthetic finetuned", "native": native, "fixed_at_binary_threshold": fixed,
                           "observed_image_level": observed_binary_metrics(labels, scores, .5),
                           "projected_model_only": evaluate_policy(labels, scores, .5, .5, scenario)}}
    for key in ("coco_zero_shot", "predict_nothing"):
        models[key] = {"name": key, "native": empty_prediction_ap(empty), "fixed_at_binary_threshold": empty,
                       "observed_image_level": observed_binary_metrics(labels, scores * 0, 1.001),
                       "projected_no_target_output": evaluate_policy(labels, scores * 0, 1.001, 1.001, scenario,
                                                                    model_used=key == "coco_zero_shot")}
    save(root, "results/model_baselines.json", {"schema_version": 1, "split": "test", "images": 3,
         "same_image_ids": True, "semantic_mapping": {}, "target_taxonomy": NAMES,
         "frozen_policy_sha256": freeze_digest, "raw_coco_prediction_count": 3, "models": models})
    for name in ("operational_comparison.json", "sensitivity_analysis.json"):
        save(root, "results/" + name, {"synthetic_fixture": True})
    save(root, "results/evaluation_provenance.json", {"schema_version": 1, "status": "complete", **bindings,
         "frozen_policy_sha256": freeze_digest, "test_policy_optimization": False, "artifact_sha256": {}})
    resign_artifacts(root)
    return root


class FakeModel:
    names = dict(enumerate(NAMES))

    def __init__(self, *_):
        self.calls = []

    def predict(self, image, **kwargs):
        self.calls.append(kwargs)
        return [SimpleNamespace(boxes=[])]


def dashboard(root, monkeypatch):
    monkeypatch.setattr(app, "YOLO", FakeModel)
    monkeypatch.setattr(app.torch.cuda, "is_available", lambda: False)
    return app.Dashboard(root)


def test_complete_saved_run_works_without_raw_dataset_or_key(saved_run, monkeypatch):
    assert not (saved_run / "raw-images-do-not-exist").exists()
    assert not (saved_run / ".env").exists()
    dash = dashboard(saved_run, monkeypatch)
    assert len(dash.test_ids) == 3 and dash.test_truth.tolist() == [0, 1, 1]
    frame = dash.baseline_frame()
    assert len(frame) == 3 and frame["Test images"].tolist() == [3, 3, 3]
    assert frame["AP50"].tolist() == [.4, 0., 0.]
    figure = dash.object_confusion()
    assert figure.layout.xaxis.title.text == "Predicted class"
    assert figure.layout.yaxis.title.text == "True class"
    assert math.isnan(figure.data[0].z[-1][-1])
    assert figure.data[0].text[-1][-1] == "N/A"
    assert dash.fixed["confusion"]["matrix"][-1][-1] == 0  # Saved counts remain unchanged.
    demo = dash.build()
    assert len(demo.blocks) > 30  # Exercise installed Gradio constructors and event wiring.
    demo.close()


@pytest.mark.parametrize("filename", app.COMPLETE_ARTIFACTS)
def test_changed_completed_artifact_is_rejected_before_model_loading(saved_run, monkeypatch, filename):
    monkeypatch.setattr(app, "YOLO", lambda *_: pytest.fail("Must validate evidence before model loading"))
    path = saved_run / "results" / filename
    path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Completed artifact changed"):
        app.Dashboard(saved_run)


@pytest.mark.parametrize("field,value", [
    ("weights_sha256", "f" * 64), ("manifest_sha256", "f" * 64),
    ("config_sha256", "f" * 64), ("class_names", list(reversed(NAMES))),
    ("positive_class_ids", [1]), ("inference", {"conf": .9}),
    ("split_manifest_sha256", "f" * 64), ("frozen_policy_sha256", "f" * 64),
])
def test_test_cache_metadata_corruption_fails_even_with_updated_artifact_digest(saved_run, field, value):
    change(saved_run, "results/test_predictions.json", lambda d: d.update({field: value}), resign=True)
    with pytest.raises(ValueError):
        app.load_dashboard_artifacts(saved_run)


@pytest.mark.parametrize("mutate", [
    lambda d: d["records"].pop(),
    lambda d: d["records"].append(copy.deepcopy(d["records"][0])),
    lambda d: d["records"][1]["labels"][0].update(class_id=1),
    lambda d: d["records"][1].update(width=101),
])
def test_missing_duplicate_or_relabelled_test_records_are_rejected(saved_run, mutate):
    change(saved_run, "results/test_predictions.json", mutate, resign=True)
    with pytest.raises(ValueError):
        app.load_dashboard_artifacts(saved_run)


def test_partial_evaluation_cannot_claim_real_baselines(saved_run):
    change(saved_run, "results/evaluation_provenance.json", lambda d: d.update(status="evaluating_frozen_policy"))
    with pytest.raises(ValueError, match="not complete"):
        app.load_dashboard_artifacts(saved_run)


def test_missing_coco_artifact_is_not_silently_replaced_with_trivial_output(saved_run):
    (saved_run / "results/coco_predictions_raw.json").unlink()
    with pytest.raises(FileNotFoundError):
        app.load_dashboard_artifacts(saved_run)


def test_false_coco_target_mapping_is_rejected_even_when_digests_are_updated(saved_run):
    change(saved_run, "results/coco_predictions_target.json",
           lambda d: d["records"][0].update(predictions=[{**BOX, "confidence": .9}]), resign=True)
    with pytest.raises(ValueError, match="empty mapping"):
        app.load_dashboard_artifacts(saved_run)


def test_fixed_metrics_and_native_denominators_must_match_test_cache(saved_run):
    change(saved_run, "results/test_metrics.json",
           lambda d: d["native"]["per_class"][0].update(instances=999), resign=True)
    with pytest.raises(ValueError, match="denominators"):
        app.load_dashboard_artifacts(saved_run)


def test_crossing_sliders_and_no_detection_live_inference_keep_declared_boundaries(saved_run, monkeypatch):
    dash = dashboard(saved_run, monkeypatch)
    assert dash.lower_changed(.9, .2)[0] == .9
    assert dash.upper_changed(.9, .2)[0] == .2
    image, text, rows = dash.predict_image(Image.new("RGB", (100, 100)), 0., 0.)
    assert "ACCELERATED CLINICAL REVIEW" in text and rows.empty and image.size == (100, 100)
    kwargs = dash.model.calls[-1]
    assert kwargs["rect"] is False and kwargs["half"] is False and kwargs["augment"] is False
    assert kwargs["conf"] == .001 and kwargs["imgsz"] == 768
    _, text, _ = dash.predict_image(Image.new("RGB", (100, 100)), .3, .8)
    assert "ROUTINE CLINICAL QUEUE" in text


@pytest.mark.parametrize("callback,lower,upper,later_opposite", [
    ("upper_changed", .02, .8, .2),
    ("lower_changed", .2, .8, .9),
    ("upper_changed", .4, .4, .3),
    ("lower_changed", .4, .4, .7),
])
def test_delayed_noncrossing_callback_preserves_newer_opposite_slider(
        saved_run, monkeypatch, callback, lower, upper, later_opposite):
    dash = dashboard(saved_run, monkeypatch)
    response = getattr(dash, callback)(lower, upper)
    # The user changes the opposite slider while this callback is in flight.
    # Only a no-op response leaves that newer client value in place.
    assert response[0] == {"__type__": "update"}
    current_opposite = later_opposite
    if response[0] != {"__type__": "update"}:
        current_opposite = response[0]
    assert current_opposite == later_opposite
    expected = evaluate_policy(dash.truth, dash.scores, lower, upper, dash.scenario)
    assert [list(row) for row in response[3].data[0].z] == expected["observed_actions"]
    assert [shape.x0 for shape in response[5].layout.shapes] == [lower, upper]


@pytest.mark.parametrize("endpoint,callback,lower,upper", [
    ("/change_lower", "lower_changed", .2, .8),
    ("/change_upper", "upper_changed", .2, .8),
    ("/change_lower", "lower_changed", .9, .2),
    ("/change_upper", "upper_changed", .9, .2),
])
def test_dashboard_response_verifier_enforces_crossing_and_noncrossing_contract(
        saved_run, monkeypatch, endpoint, callback, lower, upper):
    from tools.verify_dashboard import verify_policy_response
    dash = dashboard(saved_run, monkeypatch)
    artifacts = app.load_dashboard_artifacts(saved_run)
    response = list(getattr(dash, callback)(lower, upper))
    response[3:] = [{"plot": figure.to_json()} for figure in response[3:]]
    assert verify_policy_response(response, lower, upper, artifacts, endpoint)["passed"]
    if lower > upper:
        response[0] = {"__type__": "update"}
        expected_error = "Crossed slider"
    else:
        response[0] = upper if endpoint == "/change_lower" else lower
        expected_error = "noncrossing callback"
    with pytest.raises(AssertionError, match=expected_error):
        verify_policy_response(response, lower, upper, artifacts, endpoint)


def test_nonfinite_json_is_rejected(tmp_path):
    path = tmp_path / "invalid.json"
    path.write_text('{"value": NaN}')
    with pytest.raises(ValueError, match="Nonfinite"):
        app.read_json(path)


def test_guided_example_uses_cached_predictions_without_running_model_or_altering_source(saved_run, monkeypatch):
    dash = dashboard(saved_run, monkeypatch)
    path = saved_run / "example.png"
    Image.new("RGB", (100, 100), "white").save(path)
    original_digest = app.digest(path)
    dash.demo_examples = {"extra_check": {"path": str(path), "cached_score": .4,
        "labels": [], "predictions": [{**BOX, "confidence": .4}]}}
    original, annotated, card, rows, note = dash.select_example("extra_check", .3, .8)
    assert dash.model.calls == []
    assert app.digest(path) == original_digest
    assert original.getpixel((10, 10)) == (255, 255, 255)
    assert annotated.getpixel((10, 10)) != (255, 255, 255)
    assert 'data-route="EXTRA HUMAN PRIORITY CHECK"' in card
    assert "not a probability of disease" in card and "Saved validation prediction" in card
    assert "No listed finding" in note and "not a random performance sample" in note
    assert len(rows) == 1


def test_first_view_loads_middle_example_without_inference_and_keeps_exploration_collapsed(saved_run, monkeypatch):
    dash = dashboard(saved_run, monkeypatch)
    image_path = saved_run / "example.png"
    Image.new("RGB", (100, 100), "white").save(image_path)
    marker = saved_run / "assets/demo/manifest.json"
    marker.parent.mkdir(parents=True)
    marker.write_text("{}")
    monkeypatch.setattr(app, "load_demo_assets", lambda *_: {"examples": [{
        "id": "extra_check", "path": str(image_path), "cached_score": .4, "labels": [],
        "predictions": [{**BOX, "confidence": .4}]}]})
    demo = dash.build()
    blocks = list(demo.blocks.values())
    assert dash.model.calls == []
    assert any(isinstance(block, app.gr.HTML) and 'data-route="EXTRA HUMAN PRIORITY CHECK"' in str(block.value)
               for block in blocks)
    assert all(not block.open for block in blocks if isinstance(block, app.gr.Accordion))
    source_input = next(block for block in blocks if isinstance(block, app.gr.Image) and block.visible is False)
    assert source_input.format == "png"  # No lossy intermediate before a later real model run.
    assert all("share" not in block.buttons for block in blocks if isinstance(block, app.gr.Image))
    assert any(isinstance(block, app.gr.HTML) and "Assignment prototype" in str(block.value) for block in blocks)
    demo.close()


def test_new_upload_clears_previous_route_until_model_run(saved_run, monkeypatch):
    dash = dashboard(saved_run, monkeypatch)
    path = saved_run / "upload.png"
    Image.new("RGB", (80, 90), "white").save(path)
    original, preview, card, rows, note = dash.prepare_upload(str(path))
    assert original.size == preview.size == (80, 90)
    assert "Your image is ready" in card and "data-route=" not in card
    assert "No published annotation" in note and rows.empty and dash.model.calls == []


def test_route_card_marks_exploration_and_exposes_same_score_as_route_logic(saved_run, monkeypatch):
    dash = dashboard(saved_run, monkeypatch)
    card = dash.prediction_card(.3999999, .4, .8, "Synthetic local check")
    score = float(re.search(r'data-score="([^"]+)"', card).group(1))
    assert score == .3999999
    assert 'data-route="ROUTINE CLINICAL QUEUE"' in card
    assert "Exploratory thresholds" in card
    assert "Approved reference thresholds" in dash.prediction_card(.4, .3, .8, "Synthetic check")


def test_reference_workload_is_derived_from_frozen_test_projection(saved_run, monkeypatch):
    dash = dashboard(saved_run, monkeypatch)
    choice = dash.selected["hybrid"]
    expected = evaluate_policy(dash.test_truth, dash.test_scores, choice["lower"], choice["upper"], dash.scenario)
    panel = dash.reference_workload()
    assert f"{expected['daily_reviews']:.1f}" in panel
    assert f"{dash.scenario.daily_images:,}" in panel
    assert "not an observed hospital day" in panel
    assert "All images still receive clinical assessment" in panel
    assert "1 reviewer with" in panel and "reviewer(s)" not in panel


def test_light_interface_keeps_table_and_primary_button_contrast_under_dark_preference():
    theme = app.dashboard_theme()
    # Exercise the installed theme serializer as well as resolving referenced colour tokens.
    assert ":root.dark" in theme._get_theme_css()

    def luminance(colour):
        channels = [component / 255 for component in ImageColor.getrgb(colour)]
        linear = [channel / 12.92 if channel <= .04045 else ((channel + .055) / 1.055) ** 2.4
                  for channel in channels]
        return sum(value * weight for value, weight in zip(linear, (.2126, .7152, .0722)))

    for suffix in ("", "_dark"):
        text = theme._get_computed_value("table_text_color" + suffix)
        for field in ("table_even_background_fill", "table_odd_background_fill", "table_row_focus"):
            background = theme._get_computed_value(field + suffix)
            light, dark = sorted((luminance(text), luminance(background)), reverse=True)
            assert (light + .05) / (dark + .05) >= 4.5
        assert theme._get_computed_value("button_primary_text_color" + suffix) == "#ffffff"
        assert theme._get_computed_value("button_primary_background_fill" + suffix) == "#147d78"
    assert all(getattr(theme, key) == getattr(theme, key[:-5]) for key in vars(theme)
               if key.endswith("_dark") and hasattr(theme, key[:-5]))
