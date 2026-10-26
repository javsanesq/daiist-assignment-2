"""Fixed-operating-point object metrics and image scores from saved predictions.

Each cache is a nonempty list of records with ``image_id``, integer ``width`` /
``height``, ``labels`` and ``predictions``. A box is ``{"class_id": int,
"xyxy": [x1, y1, x2, y2]}``, with coordinates normalised to [0, 1]. Predictions
also have a finite ``confidence`` in [0, 1]. Empty lists are explicit negatives
or no predictions; missing records or fields must not stand in for them.

This module does not calculate AP, perform inference, or choose thresholds.
Native Ultralytics AP must be reported separately. These counts use inclusive
confidence and IoU boundaries, confidence-ordered greedy matching, and each
ground-truth box at most once. Ties retain the saved prediction/label order.
They are not presented as Ultralytics' maximum-F1 precision/recall summary.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from numbers import Integral, Real

import numpy as np


def _names(class_names):
    if isinstance(class_names, Mapping):
        keys = list(class_names)
        if any(isinstance(k, bool) or not isinstance(k, Integral) for k in keys):
            raise ValueError("Class-name mapping keys must be integer IDs.")
        if set(keys) != set(range(len(keys))):
            raise ValueError("Class IDs must be contiguous from zero.")
        names = [class_names[i] for i in range(len(keys))]
    elif isinstance(class_names, Sequence) and not isinstance(class_names, (str, bytes)):
        names = list(class_names)
    else:
        raise ValueError("Provide class names as a sequence or an integer-ID mapping.")
    if not names or any(not isinstance(n, str) or not n.strip() for n in names):
        raise ValueError("Class names must be nonempty strings.")
    if len(set(names)) != len(names):
        raise ValueError("Class names must be unique.")
    return names


def _id(value, count, description):
    if isinstance(value, bool) or not isinstance(value, Integral) or not 0 <= value < count:
        raise ValueError(f"{description} must be an integer class ID in [0, {count}).")
    return int(value)


def _number(value, description):
    if isinstance(value, bool) or not isinstance(value, Real) or not np.isfinite(value):
        raise ValueError(f"{description} must be finite and numeric.")
    return float(value)


def _box(annotation, count, *, prediction, description):
    if not isinstance(annotation, Mapping) or "class_id" not in annotation or "xyxy" not in annotation:
        raise ValueError(f"{description} needs class_id and xyxy.")
    class_id = _id(annotation["class_id"], count, f"{description} class_id")
    raw = annotation["xyxy"]
    if (not isinstance(raw, (list, tuple, np.ndarray))
            or (isinstance(raw, np.ndarray) and raw.ndim != 1) or len(raw) != 4):
        raise ValueError(f"{description} xyxy must contain four normalised coordinates.")
    xyxy = [_number(v, f"{description} coordinate") for v in raw]
    if not (0 <= xyxy[0] < xyxy[2] <= 1 and 0 <= xyxy[1] < xyxy[3] <= 1):
        raise ValueError(f"{description} needs an ordered, positive-area box within [0, 1].")
    result = {**annotation, "class_id": class_id, "xyxy": xyxy}
    if prediction:
        if "confidence" not in annotation:
            raise ValueError(f"{description} is missing confidence.")
        confidence = _number(annotation["confidence"], f"{description} confidence")
        if not 0 <= confidence <= 1:
            raise ValueError(f"{description} confidence must lie in [0, 1].")
        result["confidence"] = confidence
    return result


def _image_ids(records, description="Cache"):
    if not isinstance(records, Sequence) or isinstance(records, (str, bytes)) or not records:
        raise ValueError(f"{description} must be a nonempty sequence of image records.")
    ids = []
    for record in records:
        if not isinstance(record, Mapping):
            raise ValueError(f"{description} contains a non-record value.")
        image_id = record.get("image_id")
        if not isinstance(image_id, str) or not image_id.strip():
            raise ValueError(f"{description} image_id must be a nonempty string.")
        ids.append(image_id)
    if len(set(ids)) != len(ids):
        raise ValueError(f"{description} contains duplicate image IDs.")
    return ids


def assert_same_image_ids(*caches):
    """Reject duplicate/missing/extra IDs; differing cache order is permitted.

    IDs are an alignment check, not a proof that image pixels or patients match.
    The caller must also retain the dataset manifest and split provenance.
    """
    if len(caches) < 2:
        raise ValueError("At least two caches are required for a comparison.")
    reference = set(_image_ids(caches[0], "Reference cache"))
    for cache in caches[1:]:
        candidate = set(_image_ids(cache))
        if candidate != reference:
            raise ValueError("Compared caches must contain identical unique image IDs.")


def _validate(records, label_names, prediction_names, expected_image_ids=None):
    ids = _image_ids(records)
    if expected_image_ids is not None:
        if isinstance(expected_image_ids, (str, bytes)):
            raise ValueError("Expected image IDs must be a collection, not one string.")
        try:
            expected = list(expected_image_ids)
        except TypeError as exc:
            raise ValueError("Expected image IDs must be an iterable of strings.") from exc
        if any(not isinstance(i, str) or not i.strip() for i in expected):
            raise ValueError("Expected image IDs must be nonempty strings.")
        if len(set(expected)) != len(expected) or set(ids) != set(expected):
            raise ValueError("Cache must contain exactly the expected unique image IDs.")
    output = []
    for record in records:
        image_id = record["image_id"]
        clean = dict(record)
        for field in ("width", "height"):
            value = record.get(field)
            if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
                raise ValueError(f"{image_id} {field} must be a positive integer.")
            clean[field] = int(value)
        for field, names, prediction in (("labels", label_names, False),
                                         ("predictions", prediction_names, True)):
            if field not in record or not isinstance(record[field], (list, tuple)):
                raise ValueError(f"{image_id} needs an explicit {field} list (possibly empty).")
            clean[field] = [_box(box, len(names), prediction=prediction,
                                 description=f"{image_id} {field}[{i}]")
                            for i, box in enumerate(record[field])]
        output.append(clean)
    return output


def validate_cache(records, class_names, *, expected_image_ids=None):
    """Return validated copies, preserving extra metadata and input image order."""
    names = _names(class_names)
    return _validate(records, names, names, expected_image_ids)


def box_iou(left, right):
    """IoU of two valid xyxy boxes. Coordinates may be normalised or pixels."""
    a, b = np.asarray(left, dtype=float), np.asarray(right, dtype=float)
    if a.shape != (4,) or b.shape != (4,) or not np.isfinite([a, b]).all():
        raise ValueError("IoU requires two finite xyxy boxes.")
    if np.any(a[2:] <= a[:2]) or np.any(b[2:] <= b[:2]):
        raise ValueError("IoU boxes must have positive area.")
    overlap = np.maximum(0., np.minimum(a[2:], b[2:]) - np.maximum(a[:2], b[:2]))
    intersection = float(np.prod(overlap))
    return intersection / float(np.prod(a[2:] - a[:2]) + np.prod(b[2:] - b[:2]) - intersection)


def _match(labels, predictions, iou_threshold, *, class_aware):
    # Python's stable sort preserves cache order for confidence ties. max below
    # preserves the earliest eligible label when two IoUs are exactly equal.
    order = sorted(range(len(predictions)), key=lambda i: -predictions[i]["confidence"])
    available = set(range(len(labels)))
    matched, unmatched_predictions = [], []
    for pred_index in order:
        prediction = predictions[pred_index]
        candidates = [i for i in sorted(available)
                      if not class_aware or labels[i]["class_id"] == prediction["class_id"]]
        if not candidates:
            unmatched_predictions.append(pred_index)
            continue
        overlaps = [box_iou(labels[i]["xyxy"], prediction["xyxy"]) for i in candidates]
        best = int(np.argmax(overlaps))
        if overlaps[best] >= iou_threshold:
            label_index = candidates[best]
            available.remove(label_index)
            matched.append((label_index, pred_index))
        else:
            unmatched_predictions.append(pred_index)
    return matched, sorted(available), unmatched_predictions


def fixed_threshold_metrics(records, class_names, confidence_threshold, iou_threshold=.5,
                            *, expected_image_ids=None):
    """Return JSON-ready object counts and two distinctly labelled evaluations.

    Per-class precision/recall matches only the same class. The raw confusion
    matrix matches geometrically without requiring the same class, which exposes
    misclassification. Its rows are true classes and columns predicted classes;
    the last row/column represents unmatched background. Background/background
    is zero: object detection has no enumerable true-negative box population.
    Therefore matrix diagonals need not equal class-aware TP in crowded images.

    Precision is None when no retained box is predicted for a class. Recall is
    None when the ground truth has no instances of that class. No AP is returned.
    The 1.001 threshold is an explicit above-score sentinel for predict-nothing.
    """
    names = _names(class_names)
    threshold = _number(confidence_threshold, "Confidence threshold")
    iou = _number(iou_threshold, "IoU threshold")
    if not 0 <= threshold <= 1.001 or not 0 < iou <= 1:
        raise ValueError("Require confidence threshold in [0, 1.001] and IoU in (0, 1].")
    cache = _validate(records, names, names, expected_image_ids)
    count = len(names)
    truth_counts = np.zeros(count, dtype=int)
    pred_counts = np.zeros(count, dtype=int)
    true_positives = np.zeros(count, dtype=int)
    matrix = np.zeros((count + 1, count + 1), dtype=int)
    empty_truth = empty_predictions = 0
    for record in cache:
        labels = record["labels"]
        predictions = [p for p in record["predictions"] if p["confidence"] >= threshold]
        empty_truth += not labels
        empty_predictions += not predictions
        for box in labels:
            truth_counts[box["class_id"]] += 1
        for box in predictions:
            pred_counts[box["class_id"]] += 1
        matches, _, _ = _match(labels, predictions, iou, class_aware=True)
        for label_index, _ in matches:
            true_positives[labels[label_index]["class_id"]] += 1
        matches, missed, extra = _match(labels, predictions, iou, class_aware=False)
        for label_index, pred_index in matches:
            matrix[labels[label_index]["class_id"], predictions[pred_index]["class_id"]] += 1
        for label_index in missed:
            matrix[labels[label_index]["class_id"], count] += 1
        for pred_index in extra:
            matrix[count, predictions[pred_index]["class_id"]] += 1
    per_class = []
    for index, name in enumerate(names):
        tp, n_truth, n_pred = map(int, (true_positives[index], truth_counts[index], pred_counts[index]))
        per_class.append({"class_id": index, "class_name": name, "true_positives": tp,
                          "false_positives": n_pred - tp, "false_negatives": n_truth - tp,
                          "ground_truth_instances": n_truth, "predicted_instances": n_pred,
                          "precision": tp / n_pred if n_pred else None,
                          "recall": tp / n_truth if n_truth else None})
    tp, n_truth, n_pred = map(int, (true_positives.sum(), truth_counts.sum(), pred_counts.sum()))
    return {
        "level": "object", "confidence_threshold": threshold, "iou_threshold": iou,
        "threshold_rule": "confidence >= threshold; IoU >= threshold",
        "matching": "Score-ordered greedy; stable cache order for ties; no reuse of a ground-truth box",
        "images": len(cache), "images_with_no_ground_truth_boxes": int(empty_truth),
        "images_with_no_retained_predictions": int(empty_predictions),
        "ground_truth_instances": n_truth, "predicted_instances": n_pred,
        "per_class": per_class,
        "micro": {"true_positives": tp, "false_positives": n_pred - tp,
                  "false_negatives": n_truth - tp,
                  "precision": tp / n_pred if n_pred else None,
                  "recall": tp / n_truth if n_truth else None},
        "confusion": {"matrix": matrix.tolist(), "labels": names + ["background (unmatched)"],
                      "rows": "true", "columns": "predicted", "background_index": count,
                      "matching": "Class-agnostic geometric matching, separately from class-aware precision/recall",
                      "true_negative_boxes": None},
        "metric_note": "Fixed-threshold counts, not AP and not native maximum-F1 precision/recall. Empty denominators are null.",
    }


def image_scores(records, class_names, positive_class_ids, *, expected_image_ids=None):
    """Return (ordered image IDs, binary truth array, maximum confidence array).

    Every image is retained, including empty ground-truth or prediction lists.
    Positive truth means at least one annotated target box. No target prediction
    gives score zero. Confidence is an uncalibrated score, not a probability.
    """
    names = _names(class_names)
    try:
        positive = {_id(i, len(names), "Positive class ID") for i in positive_class_ids}
    except TypeError as exc:
        raise ValueError("Positive class IDs must be a nonempty iterable of integers.") from exc
    if not positive:
        raise ValueError("At least one positive target class is required.")
    cache = _validate(records, names, names, expected_image_ids)
    truth, scores = [], []
    for record in cache:
        truth.append(int(any(box["class_id"] in positive for box in record["labels"])))
        scores.append(max((box["confidence"] for box in record["predictions"]
                           if box["class_id"] in positive), default=0.))
    return [r["image_id"] for r in cache], np.asarray(truth, dtype=int), np.asarray(scores, dtype=float)


def remap_predictions(records, source_class_names, target_class_names, semantic_mapping,
                      *, expected_image_ids=None):
    """Map source predictions using an explicit {source_name: target_name} dict.

    Labels already use the target namespace. The caller is responsible for
    justifying semantic equivalence. Numeric IDs never establish equivalence;
    they are rejected as mapping keys/values. Unmapped predictions are omitted.
    For COCO versus unrelated medical classes, pass {}: this retains every image
    and target label while producing no target detections. This is a limitation
    of the baseline, not evidence of useful medical zero-shot capability.
    """
    source_names, target_names = _names(source_class_names), _names(target_class_names)
    if not isinstance(semantic_mapping, Mapping):
        raise ValueError("Semantic mapping must explicitly map source names to target names.")
    for source, target in semantic_mapping.items():
        if not isinstance(source, str) or not isinstance(target, str):
            raise ValueError("Map semantic class names, never numeric class IDs.")
        if source not in source_names or target not in target_names:
            raise ValueError("Semantic mapping contains an unknown source or target class name.")
    cache = _validate(records, target_names, source_names, expected_image_ids)
    target_ids = {name: i for i, name in enumerate(target_names)}
    for record in cache:
        mapped = []
        for prediction in record["predictions"]:
            source_name = source_names[prediction["class_id"]]
            if source_name in semantic_mapping:
                mapped.append({**prediction, "class_id": target_ids[semantic_mapping[source_name]]})
        record["predictions"] = mapped
    return cache
