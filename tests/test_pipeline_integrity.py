"""Small synthetic checks for split integrity and the validation/test boundary."""
import copy
import csv
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest
import yaml

import pipeline as p
from tools.audit_dataset import audit
from triage import Scenario


def tiny_project(tmp_path):
    dataset = tmp_path / "dataset"
    yaml_data = {"names": ["Finding A", "Finding B"], "nc": 2,
                 "roboflow": {"workspace": "example", "project": "medical", "version": 1}}
    dataset.mkdir()
    (dataset / "data.yaml").write_text(yaml.safe_dump(yaml_data))
    for i, dirname in enumerate(("train", "valid", "test")):
        for sub in ("images", "labels"):
            (dataset / dirname / sub).mkdir(parents=True)
        for j in range(2):
            pixels = np.random.default_rng(i * 2 + j).integers(0, 256, (16, 16, 3), dtype=np.uint8)
            stem = f"case_{i}_{j}"
            Image.fromarray(pixels).save(dataset / dirname / "images" / f"{stem}.png")
            (dataset / dirname / "labels" / f"{stem}.txt").write_text("0 0.5 0.5 0.2 0.2\n" if j else "")
    report, rows = audit(dataset)
    p.write_json(tmp_path / "audit.json", report)
    with (tmp_path / "audit.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    config = {"dataset": {"approved": True, "workspace": "example", "project": "medical", "version": 1,
                          "root": "dataset", "format": "yolov8", "class_names": yaml_data["names"],
                          "positive_class_ids": [0, 1], "audit_path": "audit.json", "manifest_path": "audit.csv",
                          "accepted_limitations": ["Synthetic image IDs are not patient IDs."]},
              "training": {"model": "pretrained/yolo11n.pt", "device": 0, "batch": 2},
              "inference": {"conf": .001, "iou": .7, "max_det": 300, "imgsz": 768},
              "scenario_path": "scenario.json"}
    p.write_json(tmp_path / "project_config.json", config)
    p.write_json(tmp_path / "scenario.json", {"approved": True, "policy_rule_approved": False,
                 "selection_rule": "validation_grid_minimum_with_capacity", "values": {}, "adopted_thresholds": None})
    return config, dataset


def test_json_writes_finite_values_and_rejects_untrusted_nonfinite(tmp_path):
    path = tmp_path / "result.json"
    p.write_json(path, {"values": np.array([1., np.nan, np.inf]), "count": np.int64(2)})
    assert p.read_json(path) == {"values": [1., None, None], "count": 2}
    path.write_text('{"invalid": NaN}')
    with pytest.raises(ValueError, match="Nonfinite"):
        p.read_json(path)


def test_runtime_paths_are_absolute_and_source_yaml_unchanged(tmp_path):
    config, dataset = tiny_project(tmp_path)
    original = (dataset / "data.yaml").read_bytes()
    runtime = yaml.safe_load(p.runtime_yaml(config, dataset, tmp_path).read_text())
    assert all(Path(runtime[split]).is_absolute() for split in ("train", "val", "test"))
    assert Path(runtime["val"]).parent.name == "valid"
    assert (dataset / "data.yaml").read_bytes() == original


def test_manifest_preserves_negatives_and_detects_postaudit_label_change(tmp_path):
    config, dataset = tiny_project(tmp_path)
    manifest = p.build_manifest(config, dataset, tmp_path)
    assert [len(manifest["splits"][s]) for s in ("train", "val", "test")] == [2, 2, 2]
    assert len([r for r in manifest["splits"]["val"] if not r["labels"]]) == 1
    label = dataset / manifest["splits"]["val"][1]["label"]
    label.write_text("1 0.5 0.5 0.2 0.2\n")
    with pytest.raises(ValueError, match="Labels changed"):
        p.build_manifest(config, dataset, tmp_path)
    with pytest.raises(ValueError, match="bytes changed"):
        p.verify_manifest_files(manifest, dataset)


def test_manifest_rejects_pixel_changes_even_when_filename_is_identical(tmp_path):
    config, dataset = tiny_project(tmp_path)
    image = next((dataset / "valid/images").iterdir())
    Image.new("RGB", (16, 16), "black").save(image)
    with pytest.raises(ValueError, match="Decoded image changed"):
        p.build_manifest(config, dataset, tmp_path)


def test_full_phase_blocks_before_training_when_policy_not_approved(tmp_path, monkeypatch):
    tiny_project(tmp_path)
    import tools.windows_preflight
    monkeypatch.setattr(tools.windows_preflight, "require_windows", lambda: None)
    monkeypatch.setattr(p, "train_validation", lambda *a: pytest.fail("Training must not start"))
    with pytest.raises(ValueError, match="Approve the validation policy"):
        p.run_pipeline("full", tmp_path)


def test_test_cache_cannot_select_a_policy():
    with pytest.raises(ValueError, match="only.*validation"):
        p.approved_policy_from_validation({"split": "test"}, {}, Scenario())


@pytest.mark.parametrize("changed", ["weights_sha256", "manifest_sha256", "config_sha256", "scenario_sha256", "validation_predictions_sha256"])
def test_frozen_binding_detects_each_changed_artifact(tmp_path, changed):
    config = {"scenario_path": "scenario.json"}
    files = {"weights_sha256": "weights/best.pt", "manifest_sha256": "results/dataset_manifest.json",
             "config_sha256": "project_config.json", "scenario_sha256": "scenario.json",
             "validation_predictions_sha256": "results/validation_predictions.json"}
    frozen = {"schema_version": 1, "selection_split": "val", "selected": {"selection_split": "val"}}
    for key, relative in files.items():
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("original")
        frozen[key] = p.sha256_file(path)
    p.assert_frozen_integrity(frozen, config, tmp_path)
    (tmp_path / files[changed]).write_text("changed")
    with pytest.raises(ValueError, match=changed):
        p.assert_frozen_integrity(frozen, config, tmp_path)


def test_inference_does_not_turn_missing_images_into_negative_results(tmp_path):
    config, dataset = tiny_project(tmp_path)
    manifest = p.build_manifest(config, dataset, tmp_path)
    class EmptyModel:
        def predict(self, **kwargs):
            return iter([])
    with pytest.raises(ValueError, match="omitted approved images"):
        p.cache_predictions(EmptyModel(), config, manifest, "val", tmp_path / "unused.pt", tmp_path)


def ready_validation(tmp_path):
    config, dataset = tiny_project(tmp_path)
    manifest = p.build_manifest(config, dataset, tmp_path)
    weights = tmp_path / "weights/best.pt"
    weights.parent.mkdir()
    weights.write_bytes(b"synthetic checkpoint; never loaded")
    source = tmp_path / config["training"]["model"]
    source.parent.mkdir()
    source.write_bytes(b"synthetic pretrained checkpoint; never loaded")
    cache = {"schema_version": 1, "split": "val", "class_names": config["dataset"]["class_names"],
             "positive_class_ids": [0, 1], "inference": config["inference"],
             "config_sha256": p.sha256_file(tmp_path / "project_config.json"),
             "manifest_sha256": p.sha256_file(tmp_path / "results/dataset_manifest.json"),
             "weights_sha256": p.sha256_file(weights),
             "records": [{"image_id": row["image_id"], "width": row["width"], "height": row["height"],
                          "labels": row["labels"], "predictions": [{**b, "confidence": .8} for b in row["labels"]]}
                         for row in manifest["splits"]["val"]]}
    p.write_json(tmp_path / "results/validation_predictions.json", cache)
    provenance = {k: cache[k] for k in ("weights_sha256", "manifest_sha256", "config_sha256")}
    provenance.update(run_id="synthetic", pretrained_weights_sha256=p.sha256_file(source),
                      validation_predictions_sha256=p.sha256_file(tmp_path / "results/validation_predictions.json"))
    p.write_json(tmp_path / "results/training_provenance.json", provenance)
    p.write_json(tmp_path / "results/validation_metrics.json", {"native": {}})
    scenario_path = tmp_path / "scenario.json"
    document = p.read_json(scenario_path)
    document["policy_rule_approved"] = True
    p.write_json(scenario_path, document)
    return config, manifest, cache


def test_freeze_is_validation_only_and_refuses_later_file_changes(tmp_path):
    config, _, cache = ready_validation(tmp_path)
    frozen = p.freeze_policy(config, tmp_path)
    assert frozen["selection_split"] == "val"
    assert frozen["test_inference_performed_before_freeze"] is False
    assert p.freeze_policy(config, tmp_path) == frozen
    cache["records"][0]["predictions"] = [{"class_id": 1, "confidence": .99, "xyxy": [.1, .1, .2, .2]}]
    p.write_json(tmp_path / "results/validation_predictions.json", cache)
    with pytest.raises(ValueError, match="changed after"):
        p.freeze_policy(config, tmp_path)


def test_freeze_refuses_test_access_without_an_existing_valid_freeze(tmp_path):
    config, _, _ = ready_validation(tmp_path)
    p.write_json(tmp_path / "results/test_predictions.json", {"test_was_accessed": True})
    with pytest.raises(RuntimeError, match="Test artifacts already exist"):
        p.freeze_policy(config, tmp_path)


def test_study_freeze_requires_exact_adoption_and_discloses_reused_test(tmp_path):
    config, _, cache = ready_validation(tmp_path)
    config['model_study'] = {'candidate_id': 'candidate', 'previous_test_already_exposed': True}
    p.write_json(tmp_path / 'project_config.json', config)
    cache['config_sha256'] = p.sha256_file(tmp_path / 'project_config.json')
    p.write_json(tmp_path / 'results/validation_predictions.json', cache)
    provenance = p.read_json(tmp_path / 'results/training_provenance.json')
    provenance.update(config_sha256=cache['config_sha256'], validation_predictions_sha256=p.sha256_file(tmp_path / 'results/validation_predictions.json'))
    p.write_json(tmp_path / 'results/training_provenance.json', provenance)
    with pytest.raises(ValueError, match='recorded user adoption'):
        p.freeze_policy(config, tmp_path)
    scenario = p.read_json(tmp_path / 'scenario.json')
    scenario['model_study_adoption'] = {'approved': True, 'candidate_id': 'candidate', 'weights_sha256': cache['weights_sha256']}
    p.write_json(tmp_path / 'scenario.json', scenario)
    frozen = p.freeze_policy(config, tmp_path)
    assert frozen['test_previously_exposed'] is True
    assert frozen['test_inference_performed_before_freeze'] is False
    assert 'not a new blind test' in frozen['evaluation_scope']


def test_evaluation_freezes_before_either_model_and_runs_real_baseline_path(tmp_path, monkeypatch):
    config, manifest, validation = ready_validation(tmp_path)
    config["coco_semantic_mapping"] = {}
    # The pipeline receives the same validated config; update digests after its explicit fixture change.
    p.write_json(tmp_path / "project_config.json", config)
    digest = p.sha256_file(tmp_path / "project_config.json")
    validation["config_sha256"] = digest
    p.write_json(tmp_path / "results/validation_predictions.json", validation)
    provenance = p.read_json(tmp_path / "results/training_provenance.json")
    provenance.update(config_sha256=digest, validation_predictions_sha256=p.sha256_file(tmp_path / "results/validation_predictions.json"))
    p.write_json(tmp_path / "results/training_provenance.json", provenance)
    calls = []
    def require_freeze():
        frozen = p.read_json(tmp_path / "results/frozen_policy.json")
        p.assert_frozen_integrity(frozen, config, tmp_path)
    class FakeModel:
        def __init__(self, path):
            require_freeze()
            self.source = Path(path).name != "best.pt"
            names = [f"COCO {i}" for i in range(80)] if self.source else config["dataset"]["class_names"]
            self.names = dict(enumerate(names))
    def native(model, cfg, data_yaml, split, root, run_id):
        require_freeze()
        calls.append(("native", split, model.source))
        return {"overall": {"map50": 1., "map50_95": 1.}, "per_class": []}
    def predictions(model, cfg, data, split, weights, root, **kwargs):
        require_freeze()
        calls.append(("predict", split, model.source))
        records = [{"image_id": r["image_id"], "width": r["width"], "height": r["height"], "labels": r["labels"],
                    "predictions": [{"class_id": 0, "confidence": .8, "xyxy": [.1, .1, .2, .2]}] if model.source
                    else [{**b, "confidence": .8} for b in r["labels"]]} for r in data["splits"][split]]
        return {**validation, "split": split, "weights_sha256": p.sha256_file(weights), "records": records,
                **({"prediction_class_names": kwargs["prediction_class_names"]} if kwargs else {})}
    def analysis(*args):
        for name in ("operational_comparison.json", "sensitivity_analysis.json"):
            p.write_json(tmp_path / "results" / name, {"synthetic_test": True})
    monkeypatch.setitem(sys.modules, "ultralytics", SimpleNamespace(YOLO=FakeModel))
    import tools.windows_preflight
    monkeypatch.setattr(tools.windows_preflight, "check_cuda", lambda: {"synthetic_test": True})
    monkeypatch.setattr(p, "configure_ultralytics", lambda root: {})
    monkeypatch.setattr(p, "native_detection_metrics", native)
    monkeypatch.setattr(p, "cache_predictions", predictions)
    import analysis_results
    monkeypatch.setattr(analysis_results, "run_analysis", analysis)
    p.evaluate_frozen(config, tmp_path)
    assert calls == [("native", "test", False), ("predict", "test", False), ("predict", "test", True)]
    baseline = p.read_json(tmp_path / "results/model_baselines.json")
    assert baseline["raw_coco_prediction_count"] == 2
    assert baseline["models"]["coco_zero_shot"]["fixed_at_binary_threshold"]["predicted_instances"] == 0
    assert baseline["models"]["coco_zero_shot"]["native"]["overall"]["map50"] == 0
    assert p.read_json(tmp_path / "results/evaluation_provenance.json")["status"] == "complete"


def test_fresh_run_archives_results_and_checkpoint_without_touching_source(tmp_path):
    config, dataset = tiny_project(tmp_path)
    original = (dataset / "data.yaml").read_bytes()
    p.write_json(tmp_path / "results/frozen_policy.json", {"previous": True})
    checkpoint = tmp_path / "weights/best.pt"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"previous genuine weights")
    archive = p.archive_previous_run(tmp_path)
    assert (archive / "results/frozen_policy.json").is_file()
    assert (archive / "weights/best.pt").read_bytes() == b"previous genuine weights"
    assert not checkpoint.exists()
    assert not (tmp_path / "results").exists()
    assert (dataset / "data.yaml").read_bytes() == original


def test_best_epoch_uses_actual_checkpoint_vector_not_first_ap_maximum():
    rows = [{"epoch": "1", "time": "10", "metrics/mAP50-95(B)": "0.3", "metrics/precision(B)": "0.6"},
            {"epoch": "2", "time": "21", "metrics/mAP50-95(B)": "0.3", "metrics/precision(B)": "0.7"},
            {"epoch": "3", "time": "33", "metrics/mAP50-95(B)": "0.2", "metrics/precision(B)": "0.5"}]
    checkpoint = {"epoch": -1, "version": "8.4.173", "train_metrics": {
        "metrics/mAP50-95(B)": .30000001, "metrics/precision(B)": .70000001, "fitness": .30000001},
        "train_results": {"epoch": [1, 2, 3]}}
    summary = p.training_summary_data(checkpoint, rows)
    assert summary["best_epoch"] == 2
    assert summary["completed_epochs"] == 3
    assert summary["training_loop_seconds_to_final_logged_epoch"] == 33
    assert summary["best_epoch_source"] == "unique_saved_checkpoint_metric_vector_match_to_csv"


def test_identical_rounded_checkpoint_metrics_leave_best_epoch_unknown():
    rows = [{"epoch": str(i), "time": str(i * 10), "metrics/mAP50-95(B)": "0.3"} for i in (1, 2)]
    checkpoint = {"epoch": -1, "train_metrics": {"metrics/mAP50-95(B)": .3, "fitness": .3}}
    summary = p.training_summary_data(checkpoint, rows)
    assert summary["best_epoch"] is None
    assert summary["matching_metric_epoch_candidates"] == [1, 2]
    event = {"best_epoch": 2, "fitness": .3}
    assert p.training_summary_data(checkpoint, rows, selection=event)["best_epoch"] == 2


def test_unstripped_checkpoint_epoch_is_used_directly():
    rows = [{"epoch": str(i), "time": str(i * 10), "metrics/mAP50-95(B)": "0.3"} for i in (1, 2)]
    summary = p.training_summary_data({"epoch": 1, "train_metrics": {}}, rows)
    assert summary["best_epoch"] == 2
    assert summary["best_epoch_source"] == "actual_unstripped_checkpoint_epoch_plus_one"


def full_rerun_fixture(tmp_path):
    config, _, cache = ready_validation(tmp_path)
    config['model_study'] = {'candidate_id': 'candidate', 'previous_test_already_exposed': True}
    p.write_json(tmp_path / 'project_config.json', config)
    scenario = p.read_json(tmp_path / 'scenario.json')
    scenario['model_study_adoption'] = {'approved': True, 'candidate_id': 'candidate', 'weights_sha256': 'old-checkpoint'}
    p.write_json(tmp_path / 'scenario.json', scenario)
    cache['config_sha256'] = p.sha256_file(tmp_path / 'project_config.json')
    provenance = p.read_json(tmp_path / 'results/training_provenance.json')
    # Synthetic training artifacts are held aside to model the pre-training receipt.
    (tmp_path / 'results/training_provenance.json').unlink()
    (tmp_path / 'results/validation_predictions.json').unlink()
    p.record_full_run_authorization(config, tmp_path)
    p.write_json(tmp_path / 'results/validation_predictions.json', cache)
    provenance.update(config_sha256=cache['config_sha256'],
        validation_predictions_sha256=p.sha256_file(tmp_path / 'results/validation_predictions.json'),
        full_run_authorization_sha256=p.full_run_authorization_digest(config, tmp_path))
    p.write_json(tmp_path / 'results/training_provenance.json', provenance)
    return config


def test_approved_full_rerun_freezes_new_checkpoint_without_promoting_study_candidate(tmp_path):
    config = full_rerun_fixture(tmp_path)
    frozen = p.freeze_policy(config, tmp_path)
    assert frozen['execution_mode'] == 'approved_protocol_rerun'
    assert frozen['weights_sha256'] != 'old-checkpoint'
    assert frozen['test_previously_exposed'] is True
    p.assert_frozen_integrity(frozen, config, tmp_path)
    assert p.read_json(tmp_path / 'scenario.json')['model_study_adoption']['weights_sha256'] == 'old-checkpoint'


@pytest.mark.parametrize('changed', ['scenario', 'receipt', 'missing_receipt', 'unbound_training'])
def test_full_rerun_refuses_changed_or_unbound_authorization(tmp_path, changed):
    config = full_rerun_fixture(tmp_path)
    receipt_path = tmp_path / 'results/full_run_authorization.json'
    if changed == 'scenario':
        document = p.read_json(tmp_path / 'scenario.json')
        document['values']['false_negative_eur'] = 999
        p.write_json(tmp_path / 'scenario.json', document)
    elif changed == 'receipt':
        receipt = p.read_json(receipt_path)
        receipt['execution_id'] = 'changed'
        p.write_json(receipt_path, receipt)
    elif changed == 'missing_receipt':
        receipt_path.unlink()
    else:
        provenance_path = tmp_path / 'results/training_provenance.json'
        provenance = p.read_json(provenance_path)
        provenance.pop('full_run_authorization_sha256')
        p.write_json(provenance_path, provenance)
    with pytest.raises(ValueError, match='authorization'):
        p.freeze_policy(config, tmp_path)


def test_frozen_full_run_receipt_cannot_be_replaced(tmp_path):
    config = full_rerun_fixture(tmp_path)
    frozen = p.freeze_policy(config, tmp_path)
    receipt_path = tmp_path / 'results/full_run_authorization.json'
    receipt = p.read_json(receipt_path)
    receipt['execution_id'] = 'changed-after-freeze'
    p.write_json(receipt_path, receipt)
    with pytest.raises(ValueError, match='authorization changed'):
        p.assert_frozen_integrity(frozen, config, tmp_path)


def test_full_entrypoint_records_protocol_before_training_and_continues_to_evaluation(tmp_path, monkeypatch):
    import demo_assets
    config, _ = tiny_project(tmp_path)
    document = p.read_json(tmp_path / 'scenario.json')
    document['policy_rule_approved'] = True
    p.write_json(tmp_path / 'scenario.json', document)
    events = []
    def train(config, root):
        assert p.full_run_authorization_digest(config, root)
        events.append('train')
    def evaluate(config, root):
        assert events == ['train']
        events.append('evaluate')
    monkeypatch.setattr(p, 'train_validation', train)
    monkeypatch.setattr(p, 'evaluate_frozen', evaluate)
    def package(root):
        assert events == ['train', 'evaluate']
        events.append('package')
    monkeypatch.setattr(demo_assets, 'package_demo_assets', package)
    p.run_pipeline('full', tmp_path)
    assert events == ['train', 'evaluate', 'package']
