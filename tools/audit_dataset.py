"""Read-only audit of YOLO boxes/polygons for detection; never changes a split."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import re

import numpy as np
from PIL import Image, ImageOps
import yaml

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def source_stem(name: str) -> str:
    """Conservative filename heuristic, NOT a patient identifier."""
    stem = re.sub(r"\.rf\.[0-9a-f]+$", "", Path(name).stem, flags=re.I)
    stem = re.sub(r"_(jpg|jpeg|png|bmp)$", "", stem, flags=re.I)
    return re.sub(r"[_ -]*copy(?:[_ -]*\(?\d+\)?)?", "", stem, flags=re.I).lower()


def polygon_enclosing_box(points: np.ndarray) -> list[float]:
    """Use the native detection loader's float32 ``segments2boxes`` geometry.

    This produces an axis-aligned xywh enclosure for detection. It does not
    replace or rewrite the original polygon and does not measure polygon area.
    Callers validate the original normalised vertices before calling this helper.
    """
    points = np.asarray(points, dtype=np.float32)
    x, y = points.T
    xyxy = np.array([[x.min(), y.min(), x.max(), y.max()]], dtype=np.float32)
    xywh = np.empty_like(xyxy)
    xywh[:, 0] = (xyxy[:, 0] + xyxy[:, 2]) / 2
    xywh[:, 1] = (xyxy[:, 1] + xyxy[:, 3]) / 2
    xywh[:, 2:] = xyxy[:, 2:] - xyxy[:, :2]
    return xywh[0].tolist()


def parse_labels_detailed(path: Path, n_classes: int) -> tuple[list[list[float]], list[str], dict]:
    """Validate published box or polygon rows and report their original geometry.

    Five tokens mean ``class cx cy width height``. More than five are counted as
    polygon-format rows before validity is assessed; a valid polygon requires a
    class followed by at least three finite normalised xy pairs. Its detection
    representation is the enclosing box used by native Ultralytics. We check a
    positive enclosing width and height, not clinical annotation quality, polygon
    self-intersection or contour area. Mixed boxes/polygons in one file are
    reported because the installed native loader rejects that combination.
    """
    boxes, issues = [], []
    geometry = {"nonempty_rows": 0, "box_rows": 0, "polygon_rows": 0,
                "unsupported_rows": 0, "valid_box_rows": 0,
                "valid_polygon_rows": 0, "invalid_rows": 0}
    if not path.exists():
        return [], ["missing_label_file"], geometry
    for line_no, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        tokens = line.split()
        kind = "box" if len(tokens) == 5 else "polygon" if len(tokens) > 5 else "unsupported"
        geometry["nonempty_rows"] += 1
        geometry[f"{kind}_rows"] += 1
        try:
            row = [float(value) for value in tokens]
            if not row or not np.isfinite(row).all():
                raise ValueError
            cls = row[0]
            if cls != int(cls) or not 0 <= cls < n_classes:
                raise ValueError
            if kind == "polygon":
                if len(row) < 7 or (len(row) - 1) % 2:
                    raise ValueError
                points = np.asarray(row[1:], dtype=float).reshape(-1, 2)
                if np.any(points < 0) or np.any(points > 1):
                    raise ValueError
                row = [cls, *polygon_enclosing_box(points)]
            elif kind != "box":
                raise ValueError
            _, cx, cy, width, height = row
            if not (0 < width <= 1 and 0 < height <= 1 and 0 <= cx <= 1 and 0 <= cy <= 1):
                raise ValueError
            if min(cx - width / 2, cy - height / 2) < -1e-5:
                raise ValueError
            if max(cx + width / 2, cy + height / 2) > 1 + 1e-5:
                raise ValueError
            boxes.append(row)
            geometry[f"valid_{kind}_rows"] += 1
        except (ValueError, OverflowError):
            geometry["invalid_rows"] += 1
            issue_kind = "polygon" if kind == "polygon" else "box"
            issues.append(f"invalid_{issue_kind}_line_{line_no}")
    if geometry["box_rows"] and geometry["polygon_rows"]:
        issues.append("mixed_box_and_polygon_rows_unsupported_by_native_loader")
    return boxes, issues, geometry


def parse_labels(path: Path, n_classes: int) -> tuple[list[list[float]], list[str]]:
    """Backward-compatible interface returning detection boxes and audit issues."""
    boxes, issues, _ = parse_labels_detailed(path, n_classes)
    return boxes, issues


def image_fingerprints(path: Path) -> dict:
    with Image.open(path) as img:
        rgb = ImageOps.exif_transpose(img).convert("RGB")
        width, height = rgb.size
        pixel_hash = hashlib.sha256(f"{width},{height}:".encode() + rgb.tobytes()).hexdigest()
        small = np.asarray(rgb.convert("L").resize((9, 8), Image.Resampling.LANCZOS))
        bits = (small[:, 1:] > small[:, :-1]).reshape(-1)
        dhash = sum(int(bit) << i for i, bit in enumerate(bits))
    return {"width": width, "height": height, "pixel_hash": pixel_hash, "dhash": dhash}


def duplicate_groups(rows: list[dict], key: str) -> list[dict]:
    buckets = defaultdict(list)
    for row in rows:
        buckets[row[key]].append(row)
    result = []
    for value, group in buckets.items():
        if len(group) < 2:
            continue
        splits = sorted({r["split"] for r in group})
        result.append({"key": str(value), "images": [r["image"] for r in group],
                       "splits": splits, "cross_split": len(splits) > 1,
                       # A shared filename alone cannot establish label conflict.
                       "conflicting_labels": len({r["labels"] for r in group}) > 1 if key == "pixel_hash" else None,
                       "label_comparison": "identical_pixels" if key == "pixel_hash" else "not_evaluated_filename_only"})
    return result


def audit(root: Path) -> tuple[dict, list[dict]]:
    config = yaml.safe_load((root / "data.yaml").read_text())
    names = config["names"]
    if isinstance(names, dict):
        names = [names[i] for i in range(len(names))]
    rows, issues, stats = [], [], {}
    geometry_total = Counter()
    areas = defaultdict(list)
    for split, dirname in (("train", "train"), ("val", "valid"), ("test", "test")):
        directory = root / dirname
        if split == "val" and not directory.exists():
            directory = root / "val"
        images = sorted(p for p in (directory / "images").glob("*") if p.suffix.lower() in IMAGE_SUFFIXES)
        counts, image_counts, empty, missing = Counter(), Counter(), 0, 0
        split_geometry = Counter()
        for path in images:
            label_path = directory / "labels" / f"{path.stem}.txt"
            boxes, errors, geometry = parse_labels_detailed(label_path, len(names))
            split_geometry.update(geometry)
            geometry_total.update(geometry)
            if errors:
                issues.append({"image": str(path.relative_to(root)), "issues": errors})
            missing += not label_path.exists()
            empty += not boxes and not errors
            for cls, _, _, w, h in boxes:
                counts[names[int(cls)]] += 1
                areas[(split, names[int(cls)])].append(w * h)
            for cls in {int(b[0]) for b in boxes}:
                image_counts[names[cls]] += 1
            try:
                fingerprints = image_fingerprints(path)
            except Exception as exc:
                issues.append({"image": str(path.relative_to(root)), "issues": [f"unreadable_{type(exc).__name__}"]})
                continue
            rows.append({"image": str(path.relative_to(root)), "split": split,
                         "source_stem": source_stem(path.name), "objects": len(boxes),
                         "classes": ";".join(sorted({names[int(b[0])] for b in boxes})),
                         "labels": json.dumps(sorted(boxes)),
                         "annotation_geometry": json.dumps(geometry), **fingerprints})
        orphan_labels = sorted(p.name for p in (directory / "labels").glob("*.txt")
                               if p.stem not in {i.stem for i in images})
        stats[split] = {"images": len(images), "objects": sum(counts.values()),
                        "instances_per_class": {name: counts[name] for name in names},
                        "images_per_class": {name: image_counts[name] for name in names},
                        "confirmed_empty_images": empty, "missing_label_files": missing,
                        "annotation_geometry": dict(split_geometry),
                        "orphan_labels": orphan_labels}
        if not images:
            issues.append({"split": split, "issues": ["empty_split"]})
    pixels = duplicate_groups(rows, "pixel_hash")
    stems = duplicate_groups(rows, "source_stem")
    near = []
    for i, left in enumerate(rows):
        for right in rows[i + 1:]:
            if left["split"] == right["split"] or left["pixel_hash"] == right["pixel_hash"]:
                continue
            distance = (left["dhash"] ^ right["dhash"]).bit_count()
            if distance <= 3:
                near.append({"left": left["image"], "right": right["image"], "hamming_distance": distance})
    area_summary = {f"{split}/{name}": {"n": len(values), "q10": float(np.quantile(values, .1)),
                                      "median": float(np.median(values)), "q90": float(np.quantile(values, .9))}
                    for (split, name), values in areas.items()}
    report = {"dataset_root": str(root.resolve()), "classes": names, "splits": stats,
              "annotation_geometry": dict(geometry_total),
              "detection_geometry": {
                  "box_rows": "Published class/cx/cy/width/height rows",
                  "polygon_rows": "Published class plus xy vertices; detection uses native float32 axis-aligned enclosing boxes",
                  "box_area_fraction_meaning": "Bounding-box area divided by image area; polygon rows use enclosing-box area, not polygon area",
                  "data_modified": False},
              "box_area_fraction": area_summary, "label_or_image_issues": issues,
              "exact_pixel_duplicate_groups": pixels,
              "cross_split_exact_duplicate_groups": sum(g["cross_split"] for g in pixels),
              "filename_related_groups": stems,
              "cross_split_filename_related_groups": sum(g["cross_split"] for g in stems),
              "cross_split_near_duplicate_candidate_pairs": len(near),
              "near_duplicate_pairs": near,
              "near_duplicate_candidates_truncated": False,
              "near_duplicate_examples_max_200": near[:200],
              "interpretation_limits": [
                  "Filename groups and perceptual similarities are review candidates, not proof of identical patients.",
                  "Label conflicts are evaluated only for identical decoded pixels, not filename groups.",
                  "Label-conflict comparisons use detection boxes, including polygon enclosures; distinct contours with the same enclosure are not distinguished.",
                  "Polygon coordinates are accepted as published and represented by native detection-loader enclosures only for analysis. No source annotations are rewritten.",
                  "Unsupported annotation syntax in an earlier parser is not evidence of clinically incorrect labels.",
                  "All near-duplicate candidates are retained in near_duplicate_pairs; the legacy examples key shows at most 200.",
                  "Identical decoded pixels across splits constitute confirmed image leakage.",
                  "Patient independence cannot be established from these hashes or a filename alone.",
                  "Missing annotation files are flagged separately from explicit empty annotations.",
                  "The original images, labels and splits were not altered."]}
    return report, rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report, rows = audit(args.dataset)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
    if rows:
        with (args.output / "image_manifest.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    summary = {key: report[key] for key in ("classes", "splits", "cross_split_exact_duplicate_groups",
                                           "cross_split_filename_related_groups", "cross_split_near_duplicate_candidate_pairs")}
    summary["label_or_image_issue_count"] = len(report["label_or_image_issues"])
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
