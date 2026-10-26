"""Audited Windows training and a validation freeze before held-out evaluation.

ASSIGNMENT_PHASE=train_validation stops before test inference. After approval,
evaluate_frozen continues from the genuine saved weights without retraining.
The default full phase requires the selection rule to be approved in advance.
"""
from __future__ import annotations

import contextlib
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import math
import os
from pathlib import Path
import shutil
import time
import uuid

import numpy as np
import yaml

from tools.audit_dataset import IMAGE_SUFFIXES, image_fingerprints, parse_labels, source_stem

ROOT = Path(__file__).resolve().parent


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def json_safe(value):
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    return value


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(json_safe(value), indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def read_json(path):
    def reject_constant(value):
        raise ValueError(f"Nonfinite JSON constant: {value}")
    return json.loads(Path(path).read_text(encoding="utf-8-sig"), parse_constant=reject_constant)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value):
    return hashlib.sha256(json.dumps(json_safe(value), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def local_path(value, root=ROOT):
    path = Path(value)
    return path if path.is_absolute() else Path(root) / path


def load_config(root=ROOT):
    config = read_json(Path(root) / "project_config.json")
    dataset = config["dataset"]
    names = dataset["class_names"]
    if dataset.get("approved") is not True or len(names) < 2 or len(set(names)) != len(names):
        raise ValueError("An approved dataset with distinct class names is required.")
    if not dataset["positive_class_ids"] or any(type(i) is not int or not 0 <= i < len(names) for i in dataset["positive_class_ids"]):
        raise ValueError("Positive class IDs are invalid.")
    if config["training"]["device"] != 0:
        raise ValueError("This assignment requires CUDA device 0; no CPU fallback.")
    if Path(config["training"]["model"]).name not in {"yolo11n.pt", "yolo11s.pt"}:
        raise ValueError("A genuine pretrained YOLO11n or YOLO11s checkpoint is required.")
    inference = config["inference"]
    if not 0 < inference["conf"] <= .001 or not 0 < inference["iou"] <= 1 or inference["max_det"] <= 0:
        raise ValueError("Use the documented low prediction floor and valid NMS settings.")
    return config


def load_scenario(config, root=ROOT, *, require_policy=False):
    from triage import Scenario
    document = read_json(local_path(config["scenario_path"], root))
    if document.get("approved") is not True:
        raise ValueError("The substantive operating scenario must be approved.")
    if require_policy and document.get("policy_rule_approved") is not True:
        raise ValueError("Approve the validation policy before freezing or evaluating test images.")
    if document.get("selection_rule") != "validation_grid_minimum_with_capacity":
        raise ValueError("Unsupported policy-selection rule.")
    return document, Scenario(**document["values"])


def configure_ultralytics(root=ROOT):
    directory = Path(root) / ".ultralytics"
    directory.mkdir(parents=True, exist_ok=True)
    os.environ["YOLO_CONFIG_DIR"] = str(directory)
    os.environ["WANDB_DISABLED"] = "true"
    os.environ["COMET_MODE"] = "DISABLED"
    from ultralytics import settings
    disabled = {key: False for key in ("sync", "clearml", "comet", "dvc", "mlflow", "raytune", "tensorboard", "wandb") if key in settings}
    settings.update(disabled)
    return disabled


def ensure_dataset(config, root=ROOT):
    dataset = config["dataset"]
    destination = local_path(dataset["root"], root)
    if not (destination / "data.yaml").is_file():
        if destination.exists() and any(destination.iterdir()):
            raise RuntimeError("Incomplete dataset directory: inspect it rather than overwrite data.")
        from api_key import load_api_key
        from roboflow import Roboflow
        try:
            # The SDK can print signed URLs. Discard them, including on failure.
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                rf = Roboflow(api_key=load_api_key())
                rf.workspace(dataset["workspace"]).project(dataset["project"]).version(dataset["version"]).download(dataset["format"], location=str(destination))
        except Exception as exc:
            raise RuntimeError(f"Dataset download failed ({type(exc).__name__}); credential details suppressed.") from None
    metadata = yaml.safe_load((destination / "data.yaml").read_text(encoding="utf-8"))
    names = metadata["names"]
    if isinstance(names, dict):
        names = [names[i] for i in range(len(names))]
    if names != dataset["class_names"]:
        raise ValueError("Published dataset class order differs from the approved mapping.")
    rf = metadata.get("roboflow", {})
    if any(str(rf.get(k)) != str(dataset[k]) for k in ("workspace", "project", "version")):
        raise ValueError("Dataset identity does not match the exact approved Roboflow version.")
    return destination


def verify_approved_audit(config, root=ROOT):
    dataset = config["dataset"]
    audit = read_json(local_path(dataset["audit_path"], root))
    if audit["classes"] != dataset["class_names"]:
        raise ValueError("Audit class names disagree with the approved dataset.")
    for key in ("label_or_image_issues", "cross_split_exact_duplicate_groups", "cross_split_filename_related_groups"):
        if audit[key]:
            raise ValueError(f"Unresolved fatal audit condition: {key}")
    for split in ("train", "val", "test"):
        counts = audit["splits"][split]
        if counts["images"] <= 0 or counts["missing_label_files"] or counts["orphan_labels"]:
            raise ValueError(f"Incomplete audited {split} split.")
    if not dataset.get("accepted_limitations"):
        raise ValueError("Document accepted source and independence limitations.")
    return audit


def xywh_to_label(box):
    cls, x, y, width, height = box
    # The audit permits only 1e-5 edge roundoff; no source annotation is rewritten.
    return {"class_id": int(cls), "xyxy": np.clip([x - width / 2, y - height / 2, x + width / 2, y + height / 2], 0, 1).tolist()}


def split_directory(dataset_root, split):
    if split == "val":
        return dataset_root / ("valid" if (dataset_root / "valid").is_dir() else "val")
    return dataset_root / split


def build_manifest(config, dataset_root, root=ROOT):
    """Reconcile bytes, decoded pixels and labels with the approved read-only audit."""
    audit = verify_approved_audit(config, root)
    dataset = config["dataset"]
    with local_path(dataset["manifest_path"], root).open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    approved = {row["image"].replace("\\", "/"): row for row in rows}
    if len(approved) != len(rows) or len(rows) != sum(s["images"] for s in audit["splits"].values()):
        raise ValueError("Approved audit manifest has duplicate or missing images.")
    manifest = {"schema_version": 1, "created_at_utc": utc_now(),
                "dataset": {k: dataset[k] for k in ("workspace", "project", "version", "root")},
                "class_names": dataset["class_names"], "accepted_limitations": dataset["accepted_limitations"],
                "approved_audit_sha256": sha256_file(local_path(dataset["audit_path"], root)),
                "approved_manifest_sha256": sha256_file(local_path(dataset["manifest_path"], root)), "splits": {}}
    seen = set()
    for split in ("train", "val", "test"):
        directory = split_directory(dataset_root, split)
        images = sorted(p for p in (directory / "images").iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
        if len(images) != audit["splits"][split]["images"]:
            raise ValueError(f"Image count changed in {split}.")
        records = []
        for image in images:
            image_id = image.relative_to(dataset_root).as_posix()
            expected = approved.get(image_id)
            if expected is None or expected["split"] != split:
                raise ValueError(f"Image absent from its approved split: {image_id}")
            label = directory / "labels" / (image.stem + ".txt")
            boxes, errors = parse_labels(label, len(dataset["class_names"]))
            if errors or sorted(boxes) != sorted(json.loads(expected["labels"])):
                raise ValueError(f"Labels changed or became invalid: {image_id}")
            fingerprint = image_fingerprints(image)
            if fingerprint["pixel_hash"] != expected["pixel_hash"]:
                raise ValueError(f"Decoded image changed since the approved audit: {image_id}")
            records.append({"image_id": image_id, "image": image_id,
                            "label": label.relative_to(dataset_root).as_posix(), "source_id": source_stem(image.name),
                            "width": fingerprint["width"], "height": fingerprint["height"],
                            "image_sha256": sha256_file(image), "label_sha256": sha256_file(label),
                            "labels": [xywh_to_label(box) for box in boxes]})
            seen.add(image_id)
            if len(seen) % 2000 == 0:
                print(f"Verified {len(seen):,} published images against the approved audit.", flush=True)
        if sum(len(r["labels"]) for r in records) != audit["splits"][split]["objects"]:
            raise ValueError(f"Object count changed in {split}.")
        manifest["splits"][split] = records
    if seen != set(approved):
        raise ValueError("The current data no longer matches the complete approved manifest.")
    write_json(Path(root) / "results/dataset_manifest.json", manifest)
    return manifest


def verify_manifest_files(manifest, dataset_root):
    for split, records in manifest["splits"].items():
        directory = split_directory(dataset_root, split)
        present = {p.relative_to(dataset_root).as_posix() for p in (directory / "images").iterdir() if p.suffix.lower() in IMAGE_SUFFIXES}
        if present != {r["image_id"] for r in records}:
            raise ValueError(f"The {split} image membership changed after training.")
        for row in records:
            for file_key in ("image", "label"):
                if sha256_file(dataset_root / row[file_key]) != row[file_key + "_sha256"]:
                    raise ValueError(f"Dataset bytes changed after training: {row[file_key]}")


def runtime_yaml(config, dataset_root, root=ROOT):
    path = Path(root) / "results/runtime_data.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    content = {split: str((split_directory(dataset_root, split) / "images").resolve()) for split in ("train", "val", "test")}
    content.update(nc=len(config["dataset"]["class_names"]), names=config["dataset"]["class_names"])
    path.write_text(yaml.safe_dump(content, sort_keys=False), encoding="utf-8")
    return path


def inference_kwargs(config):
    return {**config["inference"], "device": 0, "batch": config["training"]["batch"],
            "rect": False, "half": False, "augment": False, "verbose": False}


def native_detection_metrics(model, config, data_yaml, split, root=ROOT, run_id="evaluation"):
    from ultralytics.utils.metrics import smooth
    metrics = model.val(data=str(data_yaml), split=split, plots=True,
                        project=str(Path(root) / "runs"), name=f"{run_id}_{split}_native", exist_ok=True,
                        workers=config["training"]["workers"], **inference_kwargs(config))
    box = metrics.box
    positions = {int(c): i for i, c in enumerate(box.ap_class_index)}
    per_class = []
    for class_id, name in enumerate(config["dataset"]["class_names"]):
        index = positions.get(class_id)
        per_class.append({"class_id": class_id, "class_name": name,
                          "ap50": None if index is None else float(box.ap50[index]),
                          "ap50_95": None if index is None else float(box.ap[index]),
                          "precision_at_max_f1": None if index is None else float(box.p[index]),
                          "recall_at_max_f1": None if index is None else float(box.r[index]),
                          "instances": int(metrics.nt_per_class[class_id]),
                          "images": int(metrics.nt_per_image[class_id])})
    curves = np.asarray(box.f1_curve)
    confidence = None if not curves.size else float(box.px[smooth(curves.mean(0), .1).argmax()])
    return {"overall": {"map50": float(box.map50), "map50_95": float(box.map),
                        "mean_precision_at_max_f1": float(box.mp), "mean_recall_at_max_f1": float(box.mr)},
            "per_class": per_class, "maximum_f1_confidence": confidence,
            "operating_point_note": "Native P/R use the split's maximum smoothed mean-F1 operating point at IoU 0.50. This is descriptive, not the validation-frozen image triage threshold. AP integrates the confidence ranking; reported recall is not a single sensitivity for clinical care.",
            "inference": config["inference"], "split": split,
            "save_dir": str(Path(root) / "runs" / f"{run_id}_{split}_native"),
            "annotation_note": "Published representative boxes, not exhaustive lesion annotations."}


def cache_predictions(model, config, manifest, split, weights_path, root=ROOT, *, prediction_class_names=None):
    from detection_metrics import remap_predictions, validate_cache
    dataset_root = local_path(config["dataset"]["root"], root)
    expected = {str((dataset_root / row["image"]).resolve()).casefold(): row for row in manifest["splits"][split]}
    records = {}
    # A directory stream loads bounded batches; a list of every image may eagerly decode them all.
    source = split_directory(dataset_root, split) / "images"
    for result in model.predict(source=str(source), stream=True, save=False, **inference_kwargs(config)):
        path_key = str(Path(result.path).resolve()).casefold()
        if path_key not in expected:
            raise ValueError("Inference returned an image outside the approved split.")
        row = expected[path_key]
        if row["image_id"] in records:
            raise ValueError("Inference returned the same image twice.")
        predictions = []
        if result.boxes is not None:
            for cls, confidence, xyxy in zip(result.boxes.cls.cpu().tolist(), result.boxes.conf.cpu().tolist(), result.boxes.xyxyn.cpu().tolist()):
                predictions.append({"class_id": int(cls), "confidence": float(confidence), "xyxy": np.clip(xyxy, 0, 1).tolist()})
        records[row["image_id"]] = {"image_id": row["image_id"], "width": row["width"], "height": row["height"],
                                    "labels": row["labels"], "predictions": predictions}
    ids = [row["image_id"] for row in manifest["splits"][split]]
    if set(records) != set(ids):
        raise ValueError("Inference omitted approved images; missing results cannot become negatives.")
    ordered = [records[i] for i in ids]
    if prediction_class_names is None:
        validate_cache(ordered, config["dataset"]["class_names"], expected_image_ids=ids)
    else:
        remap_predictions(ordered, prediction_class_names, config["dataset"]["class_names"], {}, expected_image_ids=ids)
    cache = {"schema_version": 1, "split": split, "class_names": config["dataset"]["class_names"],
             "positive_class_ids": config["dataset"]["positive_class_ids"], "inference": config["inference"],
             "weights_sha256": sha256_file(weights_path), "config_sha256": sha256_file(Path(root) / "project_config.json"),
             "manifest_sha256": sha256_file(Path(root) / "results/dataset_manifest.json"),
             "split_manifest_sha256": canonical_sha256(manifest["splits"][split]),
             "created_at_utc": utc_now(), "records": ordered}
    if prediction_class_names is not None:
        cache["prediction_class_names"] = prediction_class_names
    return cache


def training_arguments(config, data_yaml, root, run_id):
    arguments = {key: value for key, value in config["training"].items() if key != "model"}
    arguments.update(data=str(data_yaml), project=str(Path(root) / "runs"), name=run_id,
                     exist_ok=False, pretrained=True, save=True, plots=True, val=True, split="val",
                     amp=True, cache=False, resume=False)
    arguments.update({key: 0.0 for key in ("mosaic", "mixup", "cutmix", "copy_paste", "fliplr", "flipud", "degrees", "translate", "scale", "shear", "perspective", "hsv_h", "hsv_s", "bgr")})
    arguments.update(close_mosaic=0, hsv_v=.1)
    return arguments


def train_validation(config, root=ROOT):
    from tools.windows_preflight import check_cuda
    from detection_metrics import image_scores
    from triage import select_policies
    _, scenario = load_scenario(config, root)
    environment = check_cuda()
    disabled = configure_ultralytics(root)
    dataset_root = ensure_dataset(config, root)
    manifest = build_manifest(config, dataset_root, root)
    data_yaml = runtime_yaml(config, dataset_root, root)
    from ultralytics import YOLO, __version__
    source_weights = local_path(config["training"]["model"], root)
    source_weights.parent.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(source_weights))
    if not source_weights.is_file():
        raise RuntimeError("Pretrained checkpoint was not saved at the provenance path.")
    run_id = datetime.now(timezone.utc).strftime("yolo11_%Y%m%dT%H%M%SZ_") + uuid.uuid4().hex[:6]
    arguments = training_arguments(config, data_yaml, root, run_id)
    provenance = {"schema_version": 1, "run_id": run_id, "started_at_utc": utc_now(), "status": "training",
                  "environment": environment, "ultralytics_version": __version__, "disabled_tracking": disabled,
                  "pretrained_weights": str(source_weights), "pretrained_weights_sha256": sha256_file(source_weights),
                  "settings": arguments, "config_sha256": sha256_file(Path(root) / "project_config.json"),
                  "manifest_sha256": sha256_file(Path(root) / "results/dataset_manifest.json"),
                  "test_inference_performed": False}
    provenance_path = Path(root) / "results/training_provenance.json"
    authorization = full_run_authorization_digest(config, root)
    if authorization is not None:
        provenance["full_run_authorization_sha256"] = authorization
    write_json(provenance_path, provenance)
    def record_checkpoint_selection(trainer):
        if trainer.best_fitness == trainer.fitness:
            write_json(Path(root) / "results/checkpoint_selection.json", {
                "run_id": run_id, "best_epoch": int(trainer.epoch) + 1,
                "fitness": float(trainer.fitness), "recorded_at_utc": utc_now(),
                "source": "on_model_save callback after the actual best.pt write, including equal-fitness replacements"})
    model.add_callback("on_model_save", record_checkpoint_selection)
    train_call_start = time.perf_counter()
    model.train(**arguments)
    train_call_seconds = time.perf_counter() - train_call_start
    trained = Path(model.trainer.best)
    if not trained.is_file():
        raise RuntimeError("Training completed without a best checkpoint.")
    weights = Path(root) / "weights/best.pt"
    weights.parent.mkdir(parents=True, exist_ok=True)
    temporary = weights.with_suffix(".pt.tmp")
    shutil.copy2(trained, temporary)
    temporary.replace(weights)
    best = YOLO(str(weights))
    if [best.names[i] for i in range(len(best.names))] != config["dataset"]["class_names"]:
        raise ValueError("Saved model taxonomy differs from the approved dataset.")
    provenance.update(status="trained", weights_sha256=sha256_file(weights), training_save_dir=str(model.trainer.save_dir), trained_at_utc=utc_now())
    write_json(provenance_path, provenance)
    extract_training_summary(root, train_call_seconds=train_call_seconds)
    native = native_detection_metrics(best, config, data_yaml, "val", root, run_id)
    cache = cache_predictions(best, config, manifest, "val", weights, root)
    validation_path = Path(root) / "results/validation_predictions.json"
    write_json(validation_path, cache)
    write_json(Path(root) / "results/validation_metrics.json", {"schema_version": 1, "split": "val", "native": native,
               "weights_sha256": cache["weights_sha256"], "manifest_sha256": cache["manifest_sha256"]})
    _, labels, scores = image_scores(cache["records"], cache["class_names"], cache["positive_class_ids"])
    proposal = select_policies(labels, scores, scenario)
    write_json(Path(root) / "results/validation_policy_proposal.json", {"schema_version": 1, "status": "proposal_not_frozen",
               "selection_split": "val", "selected": proposal, "validation_predictions_sha256": sha256_file(validation_path)})
    provenance.update(status="validation_ready", validation_finished_at_utc=utc_now(), validation_predictions_sha256=sha256_file(validation_path))
    write_json(provenance_path, provenance)
    return cache


def run_pipeline(phase=None, root=ROOT):
    from tools.windows_preflight import require_windows
    phase = phase or os.environ.get("ASSIGNMENT_PHASE", "full")
    if phase not in {"full", "train_validation", "evaluate_frozen"}:
        raise ValueError("ASSIGNMENT_PHASE must be full, train_validation or evaluate_frozen.")
    require_windows()
    config = load_config(root)
    if phase == "full":
        load_scenario(config, root, require_policy=True)
    if phase in {"full", "train_validation"}:
        archive_previous_run(root)
        if phase == "full":
            record_full_run_authorization(config, root)
        train_validation(config, root)
    if phase in {"full", "evaluate_frozen"}:
        evaluate_frozen(config, root=root)
        # Saved examples are checkpoint-bound too; stale examples must never
        # prevent the app from starting after an otherwise successful rerun.
        from demo_assets import package_demo_assets
        package_demo_assets(root)


def record_full_run_authorization(config, root=ROOT):
    """Bind the already approved protocol before a fresh uninterrupted run.

    Starting the full pipeline reruns the approved experiment; it does not
    promote another study candidate or change the scenario or thresholds.
    Staged validation runs deliberately do not receive this receipt.
    """
    root = Path(root)
    load_scenario(config, root, require_policy=True)
    if any((root / "results" / name).exists() for name in
           ("training_provenance.json", "validation_predictions.json", "test_predictions.json")):
        raise ValueError("Full-run authorization must precede fresh training.")
    write_json(root / "results/full_run_authorization.json", {
        "schema_version": 1, "phase": "full", "execution_id": uuid.uuid4().hex,
        "created_at_utc": utc_now(),
        "config_sha256": sha256_file(root / "project_config.json"),
        "scenario_sha256": sha256_file(local_path(config["scenario_path"], root)),
        "scope": "Rerun the approved protocol and policy without a manual checkpoint-promotion step.",
        "test_policy_optimization": False,
    })


def full_run_authorization_digest(config, root=ROOT):
    root = Path(root)
    path = root / "results/full_run_authorization.json"
    if not path.exists():
        return None
    receipt = read_json(path)
    if receipt.get("schema_version") != 1 or receipt.get("phase") != "full":
        raise ValueError("Invalid full-run authorization.")
    for key, source in (("config_sha256", root / "project_config.json"),
                        ("scenario_sha256", local_path(config["scenario_path"], root))):
        if receipt.get(key) != sha256_file(source):
            raise ValueError(f"Full-run authorization changed: {key}")
    return sha256_file(path)


def validation_context(config, root=ROOT):
    """Bind validation results to their actual weights, complete data and settings."""
    from detection_metrics import validate_cache
    root = Path(root)
    manifest_path = root / "results/dataset_manifest.json"
    validation_path = root / "results/validation_predictions.json"
    manifest, cache = read_json(manifest_path), read_json(validation_path)
    provenance = read_json(root / "results/training_provenance.json")
    weights = root / "weights/best.pt"
    expected = {"weights_sha256": sha256_file(weights), "manifest_sha256": sha256_file(manifest_path),
                "config_sha256": sha256_file(root / "project_config.json")}
    if cache.get("split") != "val":
        raise ValueError("Policy selection requires the validation cache, never test predictions.")
    for key, digest in expected.items():
        if cache.get(key) != digest or provenance.get(key) != digest:
            raise ValueError(f"Validation provenance mismatch: {key}")
    if provenance.get("validation_predictions_sha256") != sha256_file(validation_path):
        raise ValueError("Validation predictions changed after they were recorded.")
    for key in ("class_names", "positive_class_ids"):
        if cache.get(key) != config["dataset"][key]:
            raise ValueError(f"Validation cache mismatch: {key}")
    if cache.get("inference") != config["inference"]:
        raise ValueError("Validation prediction settings changed.")
    memberships = [{r["image_id"] for r in manifest["splits"][split]} for split in ("train", "val", "test")]
    if any(memberships[i] & memberships[j] for i in range(3) for j in range(i + 1, 3)):
        raise ValueError("Dataset manifest split membership overlaps.")
    validate_cache(cache["records"], cache["class_names"], expected_image_ids=memberships[1])
    ground_truth = {r["image_id"]: (r["labels"], r["width"], r["height"]) for r in manifest["splits"]["val"]}
    if any((r["labels"], r["width"], r["height"]) != ground_truth[r["image_id"]] for r in cache["records"]):
        raise ValueError("Validation ground truth differs from the approved manifest.")
    return manifest, cache, weights


def approved_policy_from_validation(cache, document, scenario):
    from detection_metrics import image_scores
    from triage import evaluate_policy, select_policies
    if cache.get("split") != "val":
        raise ValueError("Thresholds may only be selected from validation predictions.")
    _, labels, scores = image_scores(cache["records"], cache["class_names"], cache["positive_class_ids"])
    selected = select_policies(labels, scores, scenario)
    adopted = document.get("adopted_thresholds")
    if adopted is not None:
        if set(adopted) != {"binary_threshold", "lower", "upper"}:
            raise ValueError("Explicit thresholds require binary_threshold, lower and upper.")
        selected["grid_optimum"] = {"binary": selected["binary"], "hybrid": selected["hybrid"]}
        selected["binary"] = evaluate_policy(labels, scores, adopted["binary_threshold"], adopted["binary_threshold"], scenario)
        selected["hybrid"] = evaluate_policy(labels, scores, adopted["lower"], adopted["upper"], scenario)
        if not selected["hybrid"]["within_capacity"]:
            raise ValueError("The adopted validation policy exceeds the approved review capacity.")
        selected["adoption_method"] = "explicit_user_adoption_after_validation"
    else:
        selected["adoption_method"] = "approved_validation_grid_minimum_with_capacity"
    return selected


def assert_frozen_integrity(frozen, config, root=ROOT):
    root = Path(root)
    paths = {"validation_predictions_sha256": root / "results/validation_predictions.json",
             "weights_sha256": root / "weights/best.pt", "config_sha256": root / "project_config.json",
             "scenario_sha256": local_path(config["scenario_path"], root),
             "manifest_sha256": root / "results/dataset_manifest.json"}
    if frozen.get("schema_version") != 1 or frozen.get("selection_split") != "val" or frozen.get("selected", {}).get("selection_split") != "val":
        raise ValueError("The frozen policy must document validation-only selection.")
    for key, path in paths.items():
        if frozen.get(key) != sha256_file(path):
            raise ValueError(f"Frozen policy binding changed: {key}")
    if frozen.get("full_run_authorization_sha256") is not None:
        if full_run_authorization_digest(config, root) != frozen["full_run_authorization_sha256"]:
            raise ValueError("Frozen full-run authorization changed.")


def freeze_policy(config, root=ROOT):
    """Write the approved validation decision before either model sees test images."""
    root = Path(root)
    document, scenario = load_scenario(config, root, require_policy=True)
    _, cache, weights = validation_context(config, root)
    previous_test_exposure = bool(config.get("model_study", {}).get("previous_test_already_exposed", False))
    authorization = full_run_authorization_digest(config, root)
    provenance = read_json(root / "results/training_provenance.json")
    if authorization is not None or provenance.get("full_run_authorization_sha256") is not None:
        if authorization is None or provenance.get("full_run_authorization_sha256") != authorization:
            raise ValueError("Training is not bound to the full-run authorization.")
    elif config.get("model_study"):
        adoption = document.get("model_study_adoption", {})
        if (adoption.get("approved") is not True
                or adoption.get("candidate_id") != config["model_study"]["candidate_id"]
                or adoption.get("weights_sha256") != sha256_file(weights)):
            raise ValueError("A model-study candidate needs recorded user adoption of its exact checkpoint and policy before evaluation.")
    path = root / "results/frozen_policy.json"
    if path.exists():
        frozen = read_json(path)
        assert_frozen_integrity(frozen, config, root)
        return frozen
    if any((root / "results" / name).exists() for name in ("test_predictions.json", "coco_predictions_raw.json", "evaluation_provenance.json")):
        raise RuntimeError("Test artifacts already exist without a valid freeze. Do not select a new policy after test access.")
    selected = approved_policy_from_validation(cache, document, scenario)
    frozen = {"schema_version": 1, "selection_split": "val", "selected": selected,
              "validation_predictions_sha256": sha256_file(root / "results/validation_predictions.json"),
              "weights_sha256": sha256_file(weights), "config_sha256": sha256_file(root / "project_config.json"),
              "scenario_sha256": sha256_file(local_path(config["scenario_path"], root)),
              "manifest_sha256": sha256_file(root / "results/dataset_manifest.json"),
              "frozen_at_utc": utc_now(), "test_inference_performed_before_freeze": False,
              "test_inference_declaration_scope": "This checkpoint's evaluation only; prior study exposure is recorded separately.",
              "test_previously_exposed": previous_test_exposure,
              "evaluation_scope": "Reused test partition after baseline inspection; not a new blind test." if previous_test_exposure else "Initial frozen-policy test evaluation.",
              "selection_note": "Validation labels and predictions only. Test AP/P/R operating points are descriptive and cannot change this policy."}
    if authorization is not None:
        frozen["full_run_authorization_sha256"] = authorization
        frozen["execution_mode"] = "approved_protocol_rerun"
    write_json(path, frozen)
    return frozen


def empty_prediction_ap(fixed):
    """AP is exactly zero with target GT and no target predictions; P is undefined."""
    if fixed["predicted_instances"]:
        raise ValueError("The empty-output AP identity does not apply to nonempty predictions.")
    return {"overall": {"map50": 0.0 if fixed["ground_truth_instances"] else None,
                        "map50_95": 0.0 if fixed["ground_truth_instances"] else None},
            "per_class": [{"class_id": r["class_id"], "class_name": r["class_name"],
                           "ap50": 0.0 if r["ground_truth_instances"] else None,
                           "ap50_95": 0.0 if r["ground_truth_instances"] else None,
                           "precision_at_max_f1": None,
                           "recall_at_max_f1": 0.0 if r["ground_truth_instances"] else None,
                           "instances": r["ground_truth_instances"]} for r in fixed["per_class"]],
            "operating_point_note": "Analytical empty-target-output result, not a native numeric-ID evaluation. With no target predictions, AP and recall are zero for represented classes; precision is undefined. There is no maximum-F1 confidence to interpret."}


def observed_binary_metrics(labels, scores, threshold):
    labels, predicted = np.asarray(labels), np.asarray(scores) >= threshold
    tn = int(((labels == 0) & ~predicted).sum())
    fp = int(((labels == 0) & predicted).sum())
    fn = int(((labels == 1) & ~predicted).sum())
    tp = int(((labels == 1) & predicted).sum())
    return {"confusion": [[tn, fp], [fn, tp]], "rows": ["annotated_negative", "annotated_positive"],
            "columns": ["routine", "accelerated"], "sensitivity": tp / (tp + fn) if tp + fn else None,
            "specificity": tn / (tn + fp) if tn + fp else None,
            "precision": tp / (tp + fp) if tp + fp else None, "threshold": threshold,
            "units": "Observed image counts; no assumed human decisions included"}


def evaluate_frozen(config, root=ROOT):
    """Evaluate saved weights only after an approved, digest-bound policy exists."""
    from tools.windows_preflight import check_cuda
    from detection_metrics import assert_same_image_ids, fixed_threshold_metrics, image_scores, remap_predictions
    from triage import evaluate_policy
    root = Path(root)
    _, scenario = load_scenario(config, root, require_policy=True)
    manifest, validation, weights = validation_context(config, root)
    dataset_root = ensure_dataset(config, root)
    verify_manifest_files(manifest, dataset_root)
    frozen = freeze_policy(config, root)
    assert_frozen_integrity(frozen, config, root)
    freeze_path = root / "results/frozen_policy.json"
    freeze_digest = sha256_file(freeze_path)
    # All operations above are read-only checks or validation-only decisions.
    # Neither native test validation nor either model's test prediction precedes this point.
    environment = check_cuda()
    configure_ultralytics(root)
    provenance_path = root / "results/training_provenance.json"
    training_provenance = read_json(provenance_path)
    evaluation_path = root / "results/evaluation_provenance.json"
    evaluation = {"schema_version": 1, "status": "evaluating_frozen_policy", "started_at_utc": utc_now(),
                  "frozen_policy_sha256": freeze_digest, "weights_sha256": sha256_file(weights),
                  "manifest_sha256": validation["manifest_sha256"], "config_sha256": validation["config_sha256"], "environment": environment,
                  "test_policy_optimization": False,
                  "test_previously_exposed": frozen.get("test_previously_exposed", False),
                  "evaluation_scope": frozen.get("evaluation_scope", "Initial frozen-policy test evaluation.")}
    write_json(evaluation_path, evaluation)
    from ultralytics import YOLO
    model = YOLO(str(weights))
    if [model.names[i] for i in range(len(model.names))] != config["dataset"]["class_names"]:
        raise ValueError("Finetuned model taxonomy changed.")
    run_id = training_provenance["run_id"]
    data_yaml = runtime_yaml(config, dataset_root, root)
    native = native_detection_metrics(model, config, data_yaml, "test", root, run_id)
    test = cache_predictions(model, config, manifest, "test", weights, root)
    test["frozen_policy_sha256"] = freeze_digest
    write_json(root / "results/test_predictions.json", test)
    expected_ids = [r["image_id"] for r in manifest["splits"]["test"]]
    threshold = frozen["selected"]["binary"]["lower"]
    fixed = fixed_threshold_metrics(test["records"], test["class_names"], threshold, expected_image_ids=expected_ids)
    test_metrics = {"schema_version": 1, "split": "test", "native": native, "fixed_at_binary_threshold": fixed,
                    "weights_sha256": test["weights_sha256"], "manifest_sha256": test["manifest_sha256"],
                    "frozen_policy_sha256": freeze_digest, "test_policy_optimization": False}
    write_json(root / "results/test_metrics.json", test_metrics)
    validation_metrics_path = root / "results/validation_metrics.json"
    validation_metrics = read_json(validation_metrics_path)
    validation_metrics["fixed_at_binary_threshold"] = fixed_threshold_metrics(validation["records"], validation["class_names"], threshold)
    validation_metrics["frozen_policy_sha256"] = freeze_digest
    write_json(validation_metrics_path, validation_metrics)

    source_weights = local_path(config["training"]["model"], root)
    if sha256_file(source_weights) != training_provenance["pretrained_weights_sha256"]:
        raise ValueError("The COCO baseline checkpoint differs from the actual training initialization.")
    coco = YOLO(str(source_weights))
    coco_names = [coco.names[i] for i in range(len(coco.names))]
    if len(coco_names) != 80:
        raise ValueError("The original pretrained baseline must use its 80-class COCO taxonomy.")
    if config.get("coco_semantic_mapping") != {}:
        raise ValueError("The approved clinical study has no semantically equivalent COCO classes; numeric IDs must not be reused.")
    raw = cache_predictions(coco, config, manifest, "test", source_weights, root, prediction_class_names=coco_names)
    raw.update(frozen_policy_sha256=freeze_digest, semantic_mapping={},
               mapping_note="Raw predictions use the COCO namespace. All target labels use the clinical namespace. There is no justified equivalent class, so the explicit clinical mapping is empty.")
    write_json(root / "results/coco_predictions_raw.json", raw)
    mapped = remap_predictions(raw["records"], coco_names, test["class_names"], {}, expected_image_ids=expected_ids)
    nothing = [{**r, "predictions": []} for r in test["records"]]
    assert_same_image_ids(test["records"], mapped, nothing)
    write_json(root / "results/coco_predictions_target.json", {**raw, "records": mapped,
               "prediction_class_names": test["class_names"], "source_prediction_class_names": coco_names})
    write_json(root / "results/predict_nothing_predictions.json", {**test, "records": nothing,
               "weights_sha256": None, "model": "predict_nothing", "definition": "Always no target boxes and routine image priority."})
    mapped_fixed = fixed_threshold_metrics(mapped, test["class_names"], threshold, expected_image_ids=expected_ids)
    nothing_fixed = fixed_threshold_metrics(nothing, test["class_names"], threshold, expected_image_ids=expected_ids)
    _, labels, scores = image_scores(test["records"], test["class_names"], test["positive_class_ids"])
    zero = np.zeros(len(labels))
    baselines = {"schema_version": 1, "split": "test", "images": len(expected_ids),
                 "same_image_ids": True, "semantic_mapping": {}, "frozen_policy_sha256": freeze_digest,
                 "target_taxonomy": test["class_names"], "raw_coco_prediction_count": sum(len(r["predictions"]) for r in raw["records"]),
                 "note": "The COCO baseline was genuinely run. Its empty clinical mapping and the predict-nothing baseline have identical target detections, but the COCO baseline incurs the hypothetical inference cost. Zero-shot COCO AP is not evaluated by overlapping numeric IDs.",
                 "models": {
                     "finetuned": {"name": "Finetuned YOLO11n", "native": native,
                                   "fixed_at_binary_threshold": fixed, "observed_image_level": observed_binary_metrics(labels, scores, threshold),
                                   "projected_model_only": evaluate_policy(labels, scores, threshold, threshold, scenario)},
                     "coco_zero_shot": {"name": "Pretrained COCO YOLO11n, explicit empty target mapping",
                                        "native": empty_prediction_ap(mapped_fixed), "fixed_at_binary_threshold": mapped_fixed,
                                        "observed_image_level": observed_binary_metrics(labels, zero, 1.001),
                                        "projected_no_target_output": evaluate_policy(labels, zero, 1.001, 1.001, scenario)},
                     "predict_nothing": {"name": "Predict nothing / routine queue for all",
                                         "native": empty_prediction_ap(nothing_fixed), "fixed_at_binary_threshold": nothing_fixed,
                                         "observed_image_level": observed_binary_metrics(labels, zero, 1.001),
                                         "projected_no_target_output": evaluate_policy(labels, zero, 1.001, 1.001, scenario, model_used=False)}}}
    write_json(root / "results/model_baselines.json", baselines)
    assert_frozen_integrity(frozen, config, root)
    from analysis_results import run_analysis
    run_analysis(root / "results/validation_predictions.json", root / "results/test_predictions.json",
                 freeze_path, local_path(config["scenario_path"], root), root / "results")
    artifact_names = ["test_predictions.json", "test_metrics.json", "validation_metrics.json", "coco_predictions_raw.json",
                      "coco_predictions_target.json", "predict_nothing_predictions.json", "model_baselines.json",
                      "operational_comparison.json", "sensitivity_analysis.json"]
    if (root / "results/training_summary.json").is_file():
        artifact_names.append("training_summary.json")
    evaluation.update(status="complete", finished_at_utc=utc_now(),
                      artifact_sha256={name: sha256_file(root / "results" / name) for name in artifact_names})
    write_json(evaluation_path, evaluation)
    training_provenance.update(status="complete", test_inference_performed=True, frozen_policy_sha256=freeze_digest,
                               evaluation_finished_at_utc=evaluation["finished_at_utc"])
    write_json(provenance_path, training_provenance)
    return test_metrics


def archive_previous_run(root=ROOT):
    """Keep old decisions/results/checkpoints before a deliberately fresh training run."""
    root = Path(root).resolve()
    results = (root / "results").resolve()
    markers = ("training_provenance.json", "frozen_policy.json", "test_predictions.json")
    if not any((results / name).exists() for name in markers):
        return None
    if results.parent != root:
        raise ValueError("Generated results resolve outside the intended project directory.")
    name = datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%SZ_") + uuid.uuid4().hex[:6]
    destination = (root / "history" / name).resolve()
    if not destination.is_relative_to(root) or destination == root:
        raise ValueError("The archive destination must stay inside the project.")
    weights = (root / "weights/best.pt").resolve()
    if weights.exists() and not weights.is_relative_to(root):
        raise ValueError("The checkpoint resolves outside the project.")
    destination.mkdir(parents=True, exist_ok=False)
    # All resolved recursive source/destination paths were checked before moving.
    shutil.move(str(results), str(destination / "results"))
    if weights.is_file():
        (destination / "weights").mkdir()
        shutil.move(str(weights), str(destination / "weights/best.pt"))
    for name in ("project_config.json", "scenario.json"):
        if (root / name).is_file():
            shutil.copy2(root / name, destination / ("at_archival_" + name))
    write_json(destination / "archive_note.json", {"archived_at_utc": utc_now(),
               "reason": "A new training run was explicitly invoked; previous generated results and checkpoint preserved.",
               "configuration_note": "at_archival files capture the invocation configuration; old training settings remain in the archived provenance.",
               "source_dataset_modified": False})
    print(f"Preserved prior generated results and checkpoint in {destination.relative_to(root)}.", flush=True)
    return destination


def training_summary_data(checkpoint, rows, *, selection=None):
    """Interpret factual checkpoint/CSV metadata; never guess an epoch by argmax."""
    if not rows:
        raise ValueError("Training results.csv contains no logged epochs.")
    epochs = [int(float(row["epoch"])) for row in rows]
    if any(float(row["epoch"]) != epoch for row, epoch in zip(rows, epochs)):
        raise ValueError("Training CSV has nonintegral epoch numbers.")
    contiguous = epochs == list(range(1, len(epochs) + 1))
    metrics = checkpoint.get("train_metrics") or {}
    columns = sorted(key for key in metrics if key in rows[0] and isinstance(metrics[key], (int, float))
                     and math.isfinite(metrics[key]) and key not in {"epoch", "time", "fitness"})
    # The installed writer uses %.6g. Match the entire available metric vector,
    # not a rounded AP maximum whose tied epochs may identify different weights.
    matches = [epoch for row, epoch in zip(rows, epochs)
               if columns and all(float(row[key]) == float(format(float(metrics[key]), ".6g")) for key in columns)]
    epoch_raw = checkpoint.get("epoch", -1)
    best_epoch, source = None, "unresolved"
    if isinstance(epoch_raw, int) and epoch_raw >= 0:
        best_epoch, source = epoch_raw + 1, "actual_unstripped_checkpoint_epoch_plus_one"
    elif selection is not None:
        selected_epoch = selection["best_epoch"]
        if selected_epoch not in epochs or selection["fitness"] != metrics.get("fitness"):
            raise ValueError("Saved checkpoint-selection event conflicts with checkpoint/CSV evidence.")
        best_epoch, source = selected_epoch, "actual_on_model_save_checkpoint_selection_event"
    elif len(matches) == 1:
        best_epoch, source = matches[0], "unique_saved_checkpoint_metric_vector_match_to_csv"
    if best_epoch is not None and best_epoch not in epochs:
        raise ValueError("Checkpoint epoch is absent from the training CSV.")
    final_time = float(rows[-1]["time"])
    if not math.isfinite(final_time) or final_time < 0:
        raise ValueError("Training CSV duration is invalid.")
    return {"schema_version": 1, "completed_epochs": len(epochs) if contiguous else None,
            "logged_epochs": len(epochs), "last_logged_epoch": epochs[-1], "csv_epoch_sequence_contiguous": contiguous,
            "best_epoch": best_epoch, "best_epoch_source": source, "matching_metric_columns": columns,
            "matching_metric_epoch_candidates": matches,
            "checkpoint_epoch_field": epoch_raw, "checkpoint_fitness": metrics.get("fitness"),
            "checkpoint_ultralytics_version": checkpoint.get("version"),
            "checkpoint_criterion": "Validation detector mAP50-95 only; Metric.fitness weights [0,0,0,1] over [P,R,mAP50,mAP50-95].",
            "criterion_source_verified_version": "8.4.173",
            "training_loop_seconds_to_final_logged_epoch": final_time,
            "training_loop_duration_definition": "Cumulative results.csv time from trainer train_time_start through the final logged epoch; includes per-epoch validation, excludes later final-best validation and artifact generation.",
            "best_epoch_note": "Finalization resets checkpoint epoch to -1 and replaces best.pt train_results with the complete training history. A unique full saved-metric vector match is used only when an actual checkpoint-selection event is unavailable. Ambiguity remains null, never first/last argmax."}


def extract_training_summary(root=ROOT, *, train_call_seconds=None):
    """Read trusted local weights on CPU; no model inference or CUDA work."""
    root = Path(root)
    provenance_path = root / "results/training_provenance.json"
    provenance = read_json(provenance_path)
    weights = root / "weights/best.pt"
    if provenance.get("weights_sha256") != sha256_file(weights):
        raise ValueError("Only the genuine checkpoint bound to training provenance may be summarized.")
    csv_path = Path(provenance["training_save_dir"]) / "results.csv"
    if not csv_path.is_absolute():
        csv_path = root / csv_path
    with csv_path.open(encoding="utf-8-sig", newline="") as stream:
        rows = [{k.strip(): v.strip() for k, v in row.items()} for row in csv.DictReader(stream)]
    configure_ultralytics(root)
    import torch
    # The path/digest are local training outputs, not an arbitrary downloaded pickle.
    checkpoint = torch.load(weights, map_location="cpu", weights_only=False)
    selection_path = root / "results/checkpoint_selection.json"
    selection = read_json(selection_path) if selection_path.is_file() else None
    if selection is not None and selection.get("run_id") != provenance["run_id"]:
        raise ValueError("Checkpoint selection belongs to another training run.")
    summary = training_summary_data(checkpoint, rows, selection=selection)
    summary.update(created_at_utc=utc_now(), run_id=provenance["run_id"],
                   requested_maximum_epochs=provenance["settings"]["epochs"],
                   patience=provenance["settings"]["patience"], weights_sha256=sha256_file(weights),
                   results_csv=str(csv_path), results_csv_sha256=sha256_file(csv_path),
                   training_provenance_sha256_at_extraction=sha256_file(provenance_path))
    if train_call_seconds is not None:
        summary["model_train_call_seconds"] = float(train_call_seconds)
        summary["model_train_call_duration_definition"] = "Measured perf_counter elapsed around model.train, including setup/final validation/plots; excludes pre-training audit and post-training prediction caches."
    if provenance.get("trained_at_utc") and provenance.get("started_at_utc"):
        summary["elapsed_provenance_start_to_saved_weights_seconds"] = (
            datetime.fromisoformat(provenance["trained_at_utc"]) - datetime.fromisoformat(provenance["started_at_utc"])).total_seconds()
    write_json(root / "results/training_summary.json", summary)
    return summary
