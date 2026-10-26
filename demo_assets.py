"""Package and verify three local validation examples, without model inference.

Packaging needs the audited raw validation images once. Loading the bundle needs
only saved evaluation artifacts and the bundled image files, never a key/dataset.
The examples illustrate routes; they are not a representative performance sample.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parent
ROUTES = ("routine", "extra_check", "accelerated")
SELECTION = "One median cached-score validation image per frozen route; ties sorted by image_id; lower median. No filtering on correctness."
NOTICES = {
    "purpose": "Local educational examples selected to illustrate routes, not a random sample or a performance estimate.",
    "clinical": "Finding-presence labels are a review-priority proxy, not observed urgency. Every image still receives clinical assessment; no box does not establish health.",
    "annotations": "Published boxes represent one source-reader region per present class; localization is not exhaustive.",
    "licence": "The Roboflow export reports CC BY 4.0. Kaggle Competition Rules and the PhysioNet Credentialed Health Data License apply to different source releases; uploader authority is unverified. Keep this bundle local; no publication or redistribution is authorised.",
    "source": "VinBigData Chest X-ray Abnormalities Detection / VinDr-CXR, provided in this export by a Roboflow user.",
}


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def inside(base, relative):
    base = Path(base).resolve()
    path = (base / relative).resolve()
    require(path.is_relative_to(base), "Demo path escapes its declared directory.")
    return path


def _context(root):
    root = Path(root).resolve()
    paths = {"manifest_sha256": root / "results/dataset_manifest.json",
             "validation_predictions_sha256": root / "results/validation_predictions.json",
             "weights_sha256": root / "weights/best.pt", "config_sha256": root / "project_config.json"}
    config = read_json(paths["config_sha256"])
    paths["scenario_sha256"] = inside(root, config["scenario_path"])
    freeze_path = root / "results/frozen_policy.json"
    frozen = read_json(freeze_path)
    require(frozen.get("selection_split") == "val", "Demo requires a validation-frozen policy.")
    bindings = {key: digest(path) for key, path in paths.items()}
    require(all(frozen.get(key) == value for key, value in bindings.items()), "Demo source artifacts changed since the policy freeze.")
    bindings["frozen_policy_sha256"] = digest(freeze_path)
    manifest, cache = read_json(paths["manifest_sha256"]), read_json(paths["validation_predictions_sha256"])
    require(cache.get("split") == "val", "Demo examples must come from validation, not training or test.")
    require(cache.get("manifest_sha256") == bindings["manifest_sha256"], "Demo cache belongs to another manifest.")
    require(cache.get("weights_sha256") == bindings["weights_sha256"], "Demo cache belongs to another model.")
    require(cache.get("class_names") == manifest["class_names"], "Demo class names differ from the manifest.")
    rows = {row["image_id"]: row for row in manifest["splits"]["val"]}
    records = {row["image_id"]: row for row in cache["records"]}
    require(len(rows) == len(manifest["splits"]["val"]) and len(records) == len(cache["records"]), "Duplicate validation image IDs.")
    require(set(rows) == set(records), "Demo cache must cover exactly the validation images.")
    groups = {route: [] for route in ROUTES}
    lower, upper = (frozen["selected"]["hybrid"][key] for key in ("lower", "upper"))
    positive = set(cache["positive_class_ids"])
    for image_id, record in records.items():
        row = rows[image_id]
        require(all(record[key] == row[key] for key in ("width", "height", "labels")), "Demo cache labels/dimensions differ from the validation manifest.")
        score = max((box["confidence"] for box in record["predictions"] if box["class_id"] in positive), default=0.)
        route = ROUTES[0 if score < lower else 1 if score < upper else 2]
        groups[route].append((score, image_id))
    examples = []
    for route, candidates in groups.items():
        if not candidates:
            continue  # An absent route cannot be illustrated by inventing an image.
        score, image_id = sorted(candidates)[(len(candidates) - 1) // 2]
        row, record = rows[image_id], records[image_id]
        filename = route + Path(row["image"]).suffix.lower()
        examples.append({"id": route, "filename": filename, "split": "val", "image_id": image_id,
                         "source_id": row.get("source_id"), "source_image": row["image"],
                         "image_sha256": row["image_sha256"], "width": row["width"], "height": row["height"],
                         "labels": row["labels"], "predictions": record["predictions"],
                         "cached_score": score, "frozen_route": route})
    dataset = manifest["dataset"]
    source_url = (f"https://universe.roboflow.com/{dataset['workspace']}/{dataset['project']}"
                  f"/dataset/{dataset['version']}")
    metadata = {"schema_version": 1, "selection": SELECTION, "notices": NOTICES, "source_url": source_url,
                "dataset": {key: dataset[key] for key in ("workspace", "project", "version")},
                "class_names": cache["class_names"], "frozen_thresholds": {"lower": lower, "upper": upper},
                "bindings": bindings, "examples": examples}
    return metadata, inside(root, config["dataset"]["root"])


def package_demo_assets(root=ROOT, assets_dir="assets/demo"):
    root = Path(root).resolve()
    metadata, dataset_root = _context(root)
    destination = inside(root, assets_dir)
    destination.mkdir(parents=True, exist_ok=True)
    for example in metadata["examples"]:
        source = inside(dataset_root, example["source_image"])
        data = source.read_bytes()
        require(hashlib.sha256(data).hexdigest() == example["image_sha256"], "Raw demo source image changed since the audit.")
        with Image.open(source) as image:
            require(image.size == (example["width"], example["height"]), "Raw demo source dimensions differ.")
        inside(destination, example["filename"]).write_bytes(data)  # Original bytes; no transformation.
    metadata["packaged_at_utc"] = datetime.now(timezone.utc).isoformat()
    (destination / "manifest.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return load_demo_assets(root, assets_dir)


def load_demo_assets(root=ROOT, assets_dir="assets/demo"):
    """Return verified metadata plus absolute local paths; raw data is never read."""
    root = Path(root).resolve()
    expected, _ = _context(root)
    destination = inside(root, assets_dir)
    saved = read_json(destination / "manifest.json")
    require(all(saved.get(key) == value for key, value in expected.items()), "Bundled demo provenance or selection differs from the frozen validation artifacts.")
    for example in saved["examples"]:
        image_path = inside(destination, example["filename"])
        require(digest(image_path) == example["image_sha256"], "Bundled demo image is corrupt or replaced.")
        with Image.open(image_path) as image:
            require(image.size == (example["width"], example["height"]), "Bundled demo image dimensions differ.")
        example["path"] = str(image_path)
    return saved


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--assets-dir", default="assets/demo")
    arguments = parser.parse_args()
    bundle = package_demo_assets(arguments.root, arguments.assets_dir)
    print(json.dumps({"examples": len(bundle["examples"]), "source_url": bundle["source_url"],
                      "routes": [case["id"] for case in bundle["examples"]]}))
