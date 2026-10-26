from copy import deepcopy

import numpy as np
import pytest

from detection_metrics import (
    assert_same_image_ids, box_iou, fixed_threshold_metrics, image_scores,
    remap_predictions, validate_cache,
)


NAMES = ["negative_target", "positive_target", "absent_class"]
BOX = [0., 0., 1., 1.]


def annotation(class_id, box=BOX, confidence=None):
    result = {"class_id": class_id, "xyxy": list(box)}
    if confidence is not None:
        result["confidence"] = confidence
    return result


def record(image_id, labels=(), predictions=()):
    return {"image_id": image_id, "width": 640, "height": 480,
            "labels": list(labels), "predictions": list(predictions)}


def test_wrong_class_duplicates_and_empty_images_have_explicit_denominators():
    cache = [
        record("duplicate", [annotation(1)], [annotation(1, confidence=.9), annotation(1, confidence=.8)]),
        record("wrong_class", [annotation(1)], [annotation(0, confidence=.9)]),
        record("empty", [], [annotation(0, confidence=.7)]),
        record("miss", [annotation(0)], []),
        record("true_empty", [], []),
    ]
    metrics = fixed_threshold_metrics(cache, NAMES, .5)
    zero, positive, absent = metrics["per_class"]
    assert zero["ground_truth_instances"] == 1
    assert zero["predicted_instances"] == 2
    assert zero["precision"] == zero["recall"] == 0
    assert positive["true_positives"] == 1
    assert positive["false_positives"] == positive["false_negatives"] == 1
    assert positive["precision"] == positive["recall"] == .5
    assert absent["precision"] is None and absent["recall"] is None
    # Rows=true, columns=predicted. A wrong label is off-diagonal; a duplicate
    # prediction is in the background row. Empty images add no true-negative box.
    assert metrics["confusion"]["matrix"] == [
        [0, 0, 0, 1], [1, 1, 0, 0], [0, 0, 0, 0], [1, 1, 0, 0]]
    assert metrics["images"] == 5
    assert metrics["images_with_no_ground_truth_boxes"] == 2
    assert metrics["images_with_no_retained_predictions"] == 2
    assert metrics["micro"]["precision"] == .25
    assert metrics["micro"]["recall"] == pytest.approx(1 / 3)


def test_matching_is_score_ordered_not_a_global_optimal_assignment():
    left, right, middle = [0, 0, .75, 1], [.25, 0, 1, 1], [.125, 0, .875, 1]
    # The middle prediction has equal 5/7 IoU with both labels. The cache places
    # the lower-confidence prediction first to verify that confidence drives it.
    cache = [record("crowded", [annotation(1, left), annotation(1, right)],
                    [annotation(1, left, .7), annotation(1, middle, .9)])]
    result = fixed_threshold_metrics(cache, NAMES, .5, iou_threshold=.6)
    assert result["per_class"][1]["true_positives"] == 1
    assert result["per_class"][1]["false_positives"] == 1
    assert result["per_class"][1]["false_negatives"] == 1


def test_confidence_and_iou_boundaries_are_inclusive_and_sentinel_is_empty():
    cache = [record("boundary", [annotation(1)], [annotation(1, [0, 0, 1, .5], .5)])]
    assert box_iou(BOX, [0, 0, 1, .5]) == .5
    assert fixed_threshold_metrics(cache, NAMES, .5)["per_class"][1]["true_positives"] == 1
    assert fixed_threshold_metrics(cache, NAMES, .50001)["predicted_instances"] == 0
    sentinel = fixed_threshold_metrics(cache, NAMES, 1.001)
    assert sentinel["per_class"][1]["precision"] is None
    assert sentinel["per_class"][1]["recall"] == 0


def test_class_agnostic_matrix_is_not_falsely_equated_with_class_aware_counts():
    cache = [record("competing", [annotation(1)],
                    [annotation(0, confidence=.99), annotation(1, confidence=.7)])]
    result = fixed_threshold_metrics(cache, NAMES, .5)
    assert result["per_class"][1]["true_positives"] == 1
    assert result["confusion"]["matrix"][1][1] == 0
    assert result["confusion"]["matrix"][1][0] == 1
    assert result["confusion"]["matrix"][3][1] == 1


def test_image_scores_preserve_every_negative_and_use_only_chosen_classes():
    cache = [record("empty"), record("other", [annotation(0)], [annotation(0, confidence=1.)]),
             record("positive_missed", [annotation(1)]),
             record("positive_found", [annotation(1)],
                    [annotation(1, confidence=.12), annotation(1, confidence=.87)])]
    ids, truth, scores = image_scores(cache, NAMES, {1})
    assert ids == ["empty", "other", "positive_missed", "positive_found"]
    np.testing.assert_array_equal(truth, [0, 0, 1, 1])
    np.testing.assert_allclose(scores, [0, 0, 0, .87])
    with pytest.raises(ValueError, match="At least one"):
        image_scores(cache, NAMES, set())


def test_empty_semantic_mapping_never_maps_coco_id_zero_to_medical_id_zero():
    source = [record("test_image", [annotation(0)], [annotation(0, confidence=.99)]),
              record("negative_image")]
    mapped = remap_predictions(source, ["person", "bicycle"], NAMES, {},
                               expected_image_ids=["negative_image", "test_image"])
    assert mapped[0]["labels"] == source[0]["labels"]
    assert all(not row["predictions"] for row in mapped)
    assert source[0]["predictions"]  # Source cache was not changed.
    assert_same_image_ids(source, mapped)
    result = fixed_threshold_metrics(mapped, NAMES, .001)
    assert result["per_class"][0]["precision"] is None
    assert result["per_class"][0]["recall"] == 0
    with pytest.raises(ValueError, match="never numeric"):
        remap_predictions(source, ["person", "bicycle"], NAMES, {0: 0})


def test_explicit_semantic_mapping_uses_names_even_when_ids_differ():
    source = [record("a", [annotation(1)], [annotation(0, confidence=.8), annotation(1, confidence=.9)])]
    mapped = remap_predictions(source, ["lesion", "unrelated"], NAMES, {"lesion": "positive_target"})
    assert [p["class_id"] for p in mapped[0]["predictions"]] == [1]
    assert mapped[0]["predictions"][0]["confidence"] == .8
    with pytest.raises(ValueError, match="unknown"):
        remap_predictions(source, ["lesion", "unrelated"], NAMES, {"missing": "positive_target"})


def test_missing_or_repeated_images_are_never_silently_dropped_or_duplicated():
    reference = [record("a"), record("b")]
    assert_same_image_ids(reference, list(reversed(reference)))
    with pytest.raises(ValueError, match="identical"):
        assert_same_image_ids(reference, [record("a")])
    with pytest.raises(ValueError, match="duplicate"):
        assert_same_image_ids(reference, [record("a"), record("a")])
    with pytest.raises(ValueError, match="expected"):
        fixed_threshold_metrics(reference, NAMES, .5, expected_image_ids=["a", "c"])
    with pytest.raises(ValueError, match="expected"):
        validate_cache(reference, NAMES, expected_image_ids=["a", "a"])


@pytest.mark.parametrize("change", [
    lambda r: r.update(width=0),
    lambda r: r.update(height=640.5),
    lambda r: r.pop("labels"),
    lambda r: r.update(predictions=None),
    lambda r: r["labels"][0].update(class_id=3),
    lambda r: r["labels"][0].update(class_id=1.0),
    lambda r: r["labels"][0].update(class_id=True),
    lambda r: r["labels"][0].update(xyxy=[0, 0, float("nan"), 1]),
    lambda r: r["labels"][0].update(xyxy=[0, 0, 0, 1]),
    lambda r: r["labels"][0].update(xyxy=[0, 0, 1.1, 1]),
    lambda r: r["predictions"][0].update(confidence=float("inf")),
    lambda r: r["predictions"][0].update(confidence=-.1),
    lambda r: r["predictions"][0].pop("confidence"),
])
def test_invalid_cache_rejected_before_metrics(change):
    row = record("bad", [annotation(1)], [annotation(1, confidence=.8)])
    change(row)
    with pytest.raises(ValueError):
        fixed_threshold_metrics([row], NAMES, .5)


def test_validation_returns_independent_box_copies_and_checks_class_metadata():
    cache = [record("a", [annotation(1)], [annotation(1, confidence=.8)])]
    original = deepcopy(cache)
    clean = validate_cache(cache, {0: "a", 1: "b", 2: "c"})
    clean[0]["labels"][0]["xyxy"][0] = .2
    assert cache == original
    for names in ({1: "a"}, ["same", "same"], {"0": "a"}):
        with pytest.raises(ValueError):
            validate_cache(cache, names)
