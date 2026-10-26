"""Local demonstration packaging never needs a model, key or test image."""
import json
from pathlib import Path

from PIL import Image
import pytest

import demo_assets as demo


def save(path, document):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document), encoding="utf-8")


@pytest.fixture
def demo_run(tmp_path):
    root = tmp_path
    raw = root / "raw"
    raw.mkdir()
    config = {"scenario_path": "scenario.json", "dataset": {"root": "raw"}}
    save(root / "project_config.json", config)
    save(root / "scenario.json", {"approved": True})
    (root / "weights").mkdir()
    (root / "weights/best.pt").write_bytes(b"synthetic fixture, never loaded")
    rows, records = [], []
    for index, score in enumerate((0., .005, .01, .05, .1, .2, .4, .6, .9)):
        filename = f"case-{index}.png"
        Image.new("RGB", (16, 16), (index * 20, 100, 150)).save(raw / filename)
        labels = [] if index % 2 else [{"class_id": 0, "xyxy": [.1, .1, .5, .5]}]
        row = {"image_id": filename, "image": filename, "source_id": f"source-{index}",
               "width": 16, "height": 16, "image_sha256": demo.digest(raw / filename), "labels": labels}
        rows.append(row)
        records.append({key: row[key] for key in ("image_id", "width", "height", "labels")})
        records[-1]["predictions"] = [] if score == 0 else [{"class_id": 0, "xyxy": [.1, .1, .5, .5], "confidence": score}]
    manifest = {"class_names": ["Finding"],
                "dataset": {"workspace": "fixture", "project": "synthetic", "version": 1},
                "splits": {"val": rows, "test": [{"image_id": "never-open-test.png"}]}}
    save(root / "results/dataset_manifest.json", manifest)
    bindings = {"manifest_sha256": demo.digest(root / "results/dataset_manifest.json"),
                "weights_sha256": demo.digest(root / "weights/best.pt"),
                "config_sha256": demo.digest(root / "project_config.json"),
                "scenario_sha256": demo.digest(root / "scenario.json")}
    cache = {"split": "val", "class_names": ["Finding"], "positive_class_ids": [0], "records": records,
             "manifest_sha256": bindings["manifest_sha256"], "weights_sha256": bindings["weights_sha256"]}
    save(root / "results/validation_predictions.json", cache)
    bindings["validation_predictions_sha256"] = demo.digest(root / "results/validation_predictions.json")
    save(root / "results/frozen_policy.json", {"selection_split": "val", **bindings,
         "selected": {"hybrid": {"lower": .02, "upper": .27}}})
    return root


def test_bundle_uses_validation_median_routes_and_original_bytes_without_raw_data_or_key(demo_run, monkeypatch):
    monkeypatch.delenv("ROBOFLOW_API_KEY", raising=False)
    bundle = demo.package_demo_assets(demo_run)
    assert [case["image_id"] for case in bundle["examples"]] == ["case-1.png", "case-4.png", "case-7.png"]
    assert [case["cached_score"] for case in bundle["examples"]] == [.005, .1, .6]
    assert all(case["split"] == "val" for case in bundle["examples"])
    for case in bundle["examples"]:
        assert Path(case["path"]).read_bytes() == (demo_run / "raw" / case["source_image"]).read_bytes()
    # Remove only this test's own generated inputs, leaving the saved bundle.
    for image in (demo_run / "raw").glob("*.png"):
        image.unlink()
    (demo_run / "raw").rmdir()
    assert not (demo_run / ".env").exists()
    assert len(demo.load_demo_assets(demo_run)["examples"]) == 3


def test_modified_raw_source_is_rejected_before_becoming_an_example(demo_run):
    (demo_run / "raw/case-1.png").write_bytes(b"replaced")
    with pytest.raises(ValueError, match="changed since the audit"):
        demo.package_demo_assets(demo_run)


def test_bundled_image_corruption_is_detected(demo_run):
    bundle = demo.package_demo_assets(demo_run)
    Path(bundle["examples"][0]["path"]).write_bytes(b"replaced")
    with pytest.raises(ValueError, match="corrupt or replaced"):
        demo.load_demo_assets(demo_run)


def test_fabricated_provenance_or_test_case_cannot_replace_validation_example(demo_run):
    demo.package_demo_assets(demo_run)
    path = demo_run / "assets/demo/manifest.json"
    saved = demo.read_json(path)
    saved["examples"][0].update(image_id="never-open-test.png", split="test")
    save(path, saved)
    with pytest.raises(ValueError, match="provenance or selection"):
        demo.load_demo_assets(demo_run)


def test_changed_frozen_policy_requires_regenerating_bundle(demo_run):
    demo.package_demo_assets(demo_run)
    path = demo_run / "results/frozen_policy.json"
    frozen = demo.read_json(path)
    frozen["selected"]["hybrid"]["upper"] = .5
    save(path, frozen)
    with pytest.raises(ValueError, match="provenance or selection"):
        demo.load_demo_assets(demo_run)


def test_nonvalidation_cache_cannot_supply_demonstration_images(demo_run):
    cache_path = demo_run / "results/validation_predictions.json"
    cache = demo.read_json(cache_path)
    cache["split"] = "test"
    save(cache_path, cache)
    path = demo_run / "results/frozen_policy.json"
    frozen = demo.read_json(path)
    frozen["validation_predictions_sha256"] = demo.digest(cache_path)
    save(path, frozen)
    with pytest.raises(ValueError, match="must come from validation"):
        demo.package_demo_assets(demo_run)


def test_source_path_cannot_escape_dataset_directory(demo_run):
    path = demo_run / "results/dataset_manifest.json"
    manifest = demo.read_json(path)
    manifest["splits"]["val"][1]["image"] = "../outside.png"
    save(path, manifest)
    cache_path = demo_run / "results/validation_predictions.json"
    cache = demo.read_json(cache_path)
    cache["manifest_sha256"] = demo.digest(path)
    save(cache_path, cache)
    freeze_path = demo_run / "results/frozen_policy.json"
    frozen = demo.read_json(freeze_path)
    frozen.update(manifest_sha256=demo.digest(path), validation_predictions_sha256=demo.digest(cache_path))
    save(freeze_path, frozen)
    with pytest.raises(ValueError, match="escapes"):
        demo.package_demo_assets(demo_run)
