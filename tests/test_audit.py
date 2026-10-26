from pathlib import Path

from PIL import Image
import pytest

import numpy as np

from tools.audit_dataset import audit, parse_labels, parse_labels_detailed, source_stem


def test_invalid_and_empty_boxes(tmp_path):
    label = tmp_path / "x.txt"
    label.write_text("")
    assert parse_labels(label, 2) == ([], [])
    label.write_text("0 .5 .5 .2 .3\n1 .9 .5 .4 .2\n2 .5 .5 .1 .1\n0 nan .5 .1 .1\n")
    boxes, issues = parse_labels(label, 2)
    assert len(boxes) == 1 and len(issues) == 3
    assert parse_labels(tmp_path / "missing.txt", 2)[1] == ["missing_label_file"]


def test_copy_name_is_only_a_source_heuristic():
    assert source_stem("2_3_-_Copy_(3)_jpg.rf.abcdef.jpg") == "2_3"
    assert source_stem("benign_12_png.rf.abcdef.jpg") != source_stem("malignant_12_png.rf.abcdef.jpg")


def test_polygon_is_native_detection_enclosure_without_rewriting(tmp_path):
    label = tmp_path / "triangle.txt"
    source = "1 0.123456789 0.2 0.8 0.2 0.4 0.9\n"
    label.write_text(source)
    boxes, issues, geometry = parse_labels_detailed(label, 2)
    assert not issues
    assert geometry["polygon_rows"] == geometry["valid_polygon_rows"] == 1
    assert geometry["box_rows"] == geometry["invalid_rows"] == 0
    # Verify against the installed library instead of a second hand-written
    # formula; float32 rounding must agree with the native loading path.
    from ultralytics.utils.ops import segments2boxes
    points = np.asarray([float(x) for x in source.split()[1:]], dtype=np.float32).reshape(-1, 2)
    np.testing.assert_array_equal(boxes[0][1:], segments2boxes([points])[0])
    assert boxes[0][0] == 1
    assert label.read_text() == source
    assert parse_labels(label, 2) == (boxes, issues)


@pytest.mark.parametrize("source", [
    "0 .1 .2 .5 .6 .8",              # Odd number of xy coordinates.
    "0 .2 .2 .2 .5 .2 .9",          # Zero enclosing width.
    "0 .2 .2 .5 .2 .9 .2",          # Zero enclosing height.
    "0 .1 .2 .8 .2 nan .9",         # Non-finite coordinate.
    "0 .1 .2 1.1 .2 .4 .9",         # Outside the normalised image.
    "2 .1 .2 .8 .2 .4 .9",          # Invalid class ID.
    "0.5 .1 .2 .8 .2 .4 .9",        # Fractional class ID.
])
def test_invalid_polygon_rows_report_syntax_or_geometry_not_clinical_validity(tmp_path, source):
    label = tmp_path / "bad.txt"
    label.write_text(source)
    boxes, issues, geometry = parse_labels_detailed(label, 2)
    assert not boxes
    assert issues == ["invalid_polygon_line_1"]
    assert geometry["invalid_rows"] == geometry["polygon_rows"] == 1


def test_mixed_geometry_within_file_is_flagged_as_native_loader_incompatible(tmp_path):
    label = tmp_path / "mixed.txt"
    label.write_text("0 .5 .5 .2 .2\n1 .1 .2 .8 .2 .4 .9\n")
    boxes, issues, geometry = parse_labels_detailed(label, 2)
    assert len(boxes) == 2
    assert issues == ["mixed_box_and_polygon_rows_unsupported_by_native_loader"]
    assert geometry["valid_box_rows"] == geometry["valid_polygon_rows"] == 1


def test_cross_split_pixels_and_counts(tmp_path):
    (tmp_path / "data.yaml").write_text("names: [benign, malignant]\n")
    for split, name, colour in (("train", "a", "red"), ("valid", "b", "blue"), ("test", "c", "red")):
        (tmp_path / split / "images").mkdir(parents=True)
        (tmp_path / split / "labels").mkdir()
        Image.new("RGB", (20, 20), colour).save(tmp_path / split / "images" / f"{name}.png")
        (tmp_path / split / "labels" / f"{name}.txt").write_text("1 .5 .5 .2 .4\n")
    report, rows = audit(tmp_path)
    assert len(rows) == 3
    assert report["cross_split_exact_duplicate_groups"] == 1
    assert report["splits"]["test"]["instances_per_class"] == {"benign": 0, "malignant": 1}
    assert report["box_area_fraction"]["test/malignant"]["median"] == pytest.approx(.08)


def test_audit_reports_polygon_enclosing_area_and_original_geometry_counts(tmp_path):
    (tmp_path / "data.yaml").write_text("names: [normal, abnormal]\n")
    for split in ("train", "valid", "test"):
        (tmp_path / split / "images").mkdir(parents=True)
        (tmp_path / split / "labels").mkdir()
        Image.new("RGB", (20, 20), "red").save(tmp_path / split / "images" / "a.png")
        (tmp_path / split / "labels" / "a.txt").write_text("1 .1 .2 .9 .2 .1 .8\n")
    report, rows = audit(tmp_path)
    assert not report["label_or_image_issues"]
    assert report["annotation_geometry"]["polygon_rows"] == 3
    assert report["annotation_geometry"]["valid_polygon_rows"] == 3
    assert report["splits"]["test"]["annotation_geometry"]["box_rows"] == 0
    assert report["splits"]["test"]["instances_per_class"] == {"normal": 0, "abnormal": 1}
    # Triangle's area is .24, but detection enclosure is .8*.6=.48.
    assert report["box_area_fraction"]["test/abnormal"]["median"] == pytest.approx(.48)
    assert "not polygon area" in report["detection_geometry"]["box_area_fraction_meaning"]
    assert report["detection_geometry"]["data_modified"] is False
    assert len(rows) == 3
