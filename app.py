"""Local dashboard: saved weights/results; examples download only on request."""
from __future__ import annotations
import hashlib
import html
import json
import os
from pathlib import Path
import threading

ROOT = Path(__file__).resolve().parent
LOCAL_CONFIG = ROOT / ".ultralytics" / "Ultralytics"
LOCAL_CONFIG.mkdir(parents=True, exist_ok=True)
os.environ["YOLO_CONFIG_DIR"] = str(LOCAL_CONFIG.parent)
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"

import gradio as gr
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import plotly.graph_objects as go
import torch
from ultralytics import YOLO, settings
from detection_metrics import fixed_threshold_metrics, remap_predictions, validate_cache
from analysis_results import validate_inputs
from pipeline import assert_frozen_integrity, canonical_sha256, local_path
from triage import evaluate_policy, triage_actions, alternatives
from demo_assets import load_demo_assets

settings.update({k: False for k in ("sync", "clearml", "comet", "dvc", "mlflow", "raytune", "tensorboard", "wandb")})


def read_json(path):
    if not path.is_file():
        raise RuntimeError(f"Missing genuine artifact {path.name}. Complete training/evaluation first.")
    def reject_constant(value):
        raise ValueError(f"Nonfinite JSON constant in {path.name}: {value}")
    return json.loads(path.read_text(encoding="utf-8-sig"), parse_constant=reject_constant)


def digest(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


COMPLETE_ARTIFACTS = (
    "validation_metrics.json", "test_predictions.json", "test_metrics.json",
    "coco_predictions_raw.json", "coco_predictions_target.json",
    "predict_nothing_predictions.json", "model_baselines.json",
    "operational_comparison.json", "sensitivity_analysis.json",
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def check_cache_manifest(cache, split, manifest, config, frozen, *, source_names=None):
    """Check saved membership/labels only; never require raw images or a key."""
    expected = manifest["splits"][split]
    ids = [record["image_id"] for record in expected]
    require(cache.get("schema_version") == 1 and cache.get("split") == split,
            f"Invalid {split} prediction schema or split.")
    for key in ("class_names", "positive_class_ids"):
        require(cache.get(key) == config["dataset"][key], f"Prediction {key} differs from approved configuration.")
    require(cache.get("inference") == config["inference"], "Prediction inference settings differ from configuration.")
    for key in ("config_sha256", "manifest_sha256"):
        require(cache.get(key) == frozen[key], f"Prediction {key} differs from frozen evidence.")
    require(cache.get("split_manifest_sha256") == canonical_sha256(expected), "Prediction split manifest digest differs.")
    if source_names is None:
        records = validate_cache(cache["records"], cache["class_names"], expected_image_ids=ids)
    else:
        records = remap_predictions(cache["records"], source_names, cache["class_names"], {}, expected_image_ids=ids)
    truth = {row["image_id"]: (row["width"], row["height"], row["labels"]) for row in expected}
    require(all((row["width"], row["height"], row["labels"]) == truth[row["image_id"]] for row in records),
            "Prediction ground truth differs from saved approved manifest.")
    return records


def load_dashboard_artifacts(root):
    """Reject incomplete or mixed runs before loading any model or displaying results."""
    root = Path(root)
    config = read_json(root / "project_config.json")
    require(config["dataset"].get("approved") is True, "Dataset selection is not approved.")
    scenario_path = local_path(config["scenario_path"], root)
    scenario_document = read_json(scenario_path)
    require(scenario_document.get("policy_rule_approved") is True, "The final policy is not approved.")
    frozen = read_json(root / "results/frozen_policy.json")
    assert_frozen_integrity(frozen, config, root)
    freeze_digest = digest(root / "results/frozen_policy.json")
    evaluation = read_json(root / "results/evaluation_provenance.json")
    require(evaluation.get("schema_version") == 1 and evaluation.get("status") == "complete",
            "Evaluation is not complete; baseline results cannot yet be claimed.")
    require(evaluation.get("test_policy_optimization") is False, "Evaluation does not document a frozen test policy.")
    require(evaluation.get("frozen_policy_sha256") == freeze_digest, "Evaluation belongs to a different frozen policy.")
    for key in ("weights_sha256", "manifest_sha256", "config_sha256"):
        require(evaluation.get(key) == frozen[key], f"Evaluation {key} differs from the frozen run.")
    artifact_hashes = evaluation.get("artifact_sha256", {})
    for name in COMPLETE_ARTIFACTS:
        require(artifact_hashes.get(name) == digest(root / "results" / name), f"Completed artifact changed: {name}")
    manifest = read_json(root / "results/dataset_manifest.json")
    names = config["dataset"]["class_names"]
    require(manifest.get("schema_version") == 1 and manifest.get("class_names") == names,
            "Saved dataset manifest taxonomy differs from configuration.")
    require(all(str(manifest["dataset"].get(k)) == str(config["dataset"][k])
                for k in ("workspace", "project", "version")), "Saved dataset identity differs from configuration.")
    memberships = []
    for split in ("train", "val", "test"):
        rows = manifest["splits"][split]
        require(bool(rows), f"Saved {split} manifest is empty.")
        validate_cache([{**row, "predictions": []} for row in rows], names)
        memberships.append({row["image_id"] for row in rows})
    require(not any(memberships[i] & memberships[j] for i in range(3) for j in range(i + 1, 3)),
            "Saved manifest split membership overlaps.")
    validation = read_json(root / "results/validation_predictions.json")
    test = read_json(root / "results/test_predictions.json")
    scenario, selected, val_data, test_data = validate_inputs(
        validation, test, frozen, scenario_document,
        validation_sha256=digest(root / "results/validation_predictions.json"), scenario_sha256=digest(scenario_path))
    for cache, split in ((validation, "val"), (test, "test")):
        check_cache_manifest(cache, split, manifest, config, frozen)
    require(test.get("frozen_policy_sha256") == freeze_digest, "Test predictions belong to a different policy.")
    metrics = read_json(root / "results/test_metrics.json")
    require(metrics.get("schema_version") == 1 and metrics.get("split") == "test", "Invalid test metric identity.")
    for key in ("weights_sha256", "manifest_sha256"):
        require(metrics.get(key) == frozen[key], f"Test metrics {key} differs from frozen evidence.")
    require(metrics.get("frozen_policy_sha256") == freeze_digest, "Test metrics belong to a different policy.")
    fixed = fixed_threshold_metrics(test["records"], names, selected["binary"]["lower"], expected_image_ids=memberships[2])
    require(metrics.get("fixed_at_binary_threshold") == fixed, "Saved fixed-threshold metrics differ from test predictions.")
    native = metrics["native"]
    require(native.get("split") == "test" and native.get("inference") == config["inference"], "Native metrics have incompatible inference provenance.")
    require([r["class_id"] for r in native["per_class"]] == list(range(len(names)))
            and [r["class_name"] for r in native["per_class"]] == names, "Native metric taxonomy differs.")
    require([r["instances"] for r in native["per_class"]] == [r["ground_truth_instances"] for r in fixed["per_class"]],
            "Native metric class denominators differ from saved test labels.")
    baselines = read_json(root / "results/model_baselines.json")
    require(baselines.get("schema_version") == 1 and baselines.get("split") == "test"
            and baselines.get("images") == len(test_data[0]) and baselines.get("same_image_ids") is True,
            "Baseline evaluation does not cover the same test images.")
    require(baselines.get("frozen_policy_sha256") == freeze_digest and baselines.get("target_taxonomy") == names,
            "Baseline policy or taxonomy differs.")
    require(config.get("coco_semantic_mapping") == {} and baselines.get("semantic_mapping") == {},
            "This clinical study requires an explicit empty COCO mapping.")
    raw = read_json(root / "results/coco_predictions_raw.json")
    mapped = read_json(root / "results/coco_predictions_target.json")
    nothing = read_json(root / "results/predict_nothing_predictions.json")
    coco_names = raw.get("prediction_class_names", [])
    require(len(coco_names) == 80 and len(set(coco_names)) == 80, "Raw COCO prediction taxonomy is invalid.")
    remapped = check_cache_manifest(raw, "test", manifest, config, frozen, source_names=coco_names)
    for cache in (mapped, nothing):
        check_cache_manifest(cache, "test", manifest, config, frozen)
    for cache in (raw, mapped, nothing):
        require(cache.get("frozen_policy_sha256") == freeze_digest, "Baseline cache belongs to a different policy.")
    require(raw.get("semantic_mapping") == {} and mapped.get("semantic_mapping") == {}, "COCO cache mapping differs.")
    require(raw.get("weights_sha256") == mapped.get("weights_sha256") and isinstance(raw.get("weights_sha256"), str)
            and len(raw["weights_sha256"]) == 64, "COCO cache checkpoint provenance is missing or inconsistent.")
    require(mapped.get("prediction_class_names") == names and mapped.get("source_prediction_class_names") == coco_names,
            "Mapped COCO taxonomy provenance differs.")
    require({r["image_id"]: r for r in mapped["records"]} == {r["image_id"]: r for r in remapped},
            "COCO target predictions differ from the declared empty mapping.")
    require(nothing.get("weights_sha256") is None and nothing.get("model") == "predict_nothing"
            and all(not r["predictions"] for r in nothing["records"]), "Predict-nothing baseline is not empty.")
    require(baselines.get("raw_coco_prediction_count") == sum(len(r["predictions"]) for r in raw["records"]),
            "Raw COCO prediction count differs from the baseline summary.")
    models = baselines["models"]
    require(set(models) == {"finetuned", "coco_zero_shot", "predict_nothing"}, "Required baseline comparison is incomplete.")
    require(models["finetuned"]["native"] == native and models["finetuned"]["fixed_at_binary_threshold"] == fixed,
            "Finetuned baseline comparison differs from test metrics.")
    return {"config": config, "scenario": scenario, "freeze": frozen, "selected": selected,
            "validation": validation, "test": test, "metrics": metrics, "fixed": fixed,
            "baselines": baselines, "val_data": val_data, "test_data": test_data}


def heatmap(matrix, columns, title, expected=False):
    labels = [[f"{v:,.1f}" if expected else str(int(v)) for v in row] for row in matrix]
    fig = go.Figure(go.Heatmap(z=matrix, x=columns, y=["No listed finding", "Listed finding"],
        text=labels, texttemplate="%{text}", colorscale=[[0, "#eef2f8"], [1, "#315a8e"]],
        showscale=False, hovertemplate="%{y} → %{x}: %{text}<extra></extra>"))
    display_title = "Expected decisions / 1,000<br>After assumed human review" if expected else title.replace(" — ", "<br>")
    fig.update_layout(title=dict(text=display_title, font=dict(size=14)), height=330, margin=dict(l=115, r=15, t=72, b=65),
        font=dict(family="Arial", size=13, color="#17283f"), paper_bgcolor="white",
        yaxis=dict(autorange="reversed"))
    return fig


class Dashboard:
    def __init__(self, root=ROOT):
        self.root = Path(root)
        artifacts = load_dashboard_artifacts(self.root)
        for name in ("config", "scenario", "freeze", "selected", "validation", "test", "metrics", "fixed", "baselines"):
            setattr(self, name, artifacts[name])
        self.names, self.positive = self.validation["class_names"], self.validation["positive_class_ids"]
        self.ids, self.truth, self.scores = artifacts["val_data"]
        self.test_ids, self.test_truth, self.test_scores = artifacts["test_data"]
        self.device = 0 if torch.cuda.is_available() else "cpu"
        self.model = YOLO(str(self.root / "weights/best.pt"))
        if [self.model.names[i] for i in range(len(self.model.names))] != self.names:
            raise RuntimeError("Model and evaluation classes differ.")
        self.lock = threading.Lock()

    def curve(self, lower, upper):
        sweep, chosen = self.selected["binary_sweep"], self.selected["binary"]
        fig = go.Figure(go.Scatter(x=[r["lower"] for r in sweep], y=[r["cost_per_1000"] for r in sweep],
            mode="lines", name="Model only", line=dict(color="#315a8e", width=3)))
        fig.add_scatter(x=[chosen["lower"]], y=[chosen["cost_per_1000"]], mode="markers",
            marker=dict(size=12, color="#c15120"), name="Frozen model-only choice")
        fig.add_vline(x=float(lower), line_dash="dot", line_color="#168079", annotation_text="Current L")
        fig.add_vline(x=float(upper), line_dash="dash", line_color="#a76b18", annotation_text="Current U")
        fig.update_layout(title="Validation cost curve — model-only decisions", height=370,
            xaxis_title="Confidence threshold (1.001 means flag nothing)",
            yaxis_title="Hypothetical EUR per 1,000 images", template="plotly_white",
            margin=dict(l=75, r=25, t=70, b=65), font=dict(family="Arial", size=13),
            legend=dict(orientation="h", y=1.17))
        return fig

    def policy_view(self, lower, upper):
        r = evaluate_policy(self.truth, self.scores, float(lower), float(upper), self.scenario)
        entries = [("Expected cost / 1,000", f"€{r['cost_per_1000']:,.2f}"),
                   ("Extra human review", f"{r['review_fraction']:.1%}"),
                   ("Checks per day / capacity", f"{r['daily_reviews']:.1f} / {self.scenario.daily_capacity:.0f}"),
                   ("Staff hours per day", f"{r['daily_review_hours']:.2f}")]
        cards = "<div class='metric-row'>" + "".join(
            f"<div class='metric-card'><span>{label}</span><strong>{value}</strong></div>"
            for label, value in entries) + "</div>"
        state = "Within assumed capacity" if r["within_capacity"] else "Exceeds assumed capacity"
        note = (f"**{state}.** Needs {r['reviewers_needed']} person(s) at "
            f"{self.scenario.review_hours_per_person:g} assigned hours each. Costs and review share use "
            f"{self.scenario.prevalence:.0%} assumed prevalence. Final decisions assume reviewer sensitivity/"
            f"specificity of {self.scenario.reviewer_sensitivity:.0%}/{self.scenario.reviewer_specificity:.0%}; "
            "they are not measured human results.")
        observed = heatmap(r["observed_actions"], ["Routine", "Human check", "Accelerated"],
            f"Observed validation routing — {len(self.truth):,} images")
        expected = heatmap(r["expected_resolved_confusion_per_1000"], ["Routine", "Accelerated"],
            "Expected final decisions / 1,000 — after assumed review", True)
        return cards, note, observed, expected, self.curve(lower, upper)

    def lower_changed(self, lower, upper):
        lower, upper = float(lower), float(upper)
        # A delayed callback must not restore the other slider's captured value.
        adjusted_upper = lower if lower > upper else gr.skip()
        return adjusted_upper, *self.policy_view(lower, max(lower, upper))

    def upper_changed(self, lower, upper):
        lower, upper = float(lower), float(upper)
        adjusted_lower = upper if lower > upper else gr.skip()
        return adjusted_lower, *self.policy_view(min(lower, upper), upper)

    def predict_image(self, uploaded, lower, upper):
        if uploaded is None:
            raise gr.Error("Upload an image first.")
        if lower > upper:
            raise gr.Error("Lower threshold must not exceed upper threshold.")
        image, cfg = uploaded.convert("RGB"), self.validation["inference"]
        with self.lock:
            pred = self.model.predict(image, conf=cfg["conf"], iou=cfg["iou"], imgsz=cfg["imgsz"],
                max_det=cfg["max_det"], device=self.device, rect=False, half=False,
                augment=False, verbose=False, save=False)[0]
        all_boxes = [(int(b.cls.item()), float(b.conf.item()), b.xyxy[0].cpu().tolist()) for b in pred.boxes]
        score = max((s for c, s, _ in all_boxes if c in self.positive), default=0.)
        return self.render_prediction(image, all_boxes, score, lower, upper, "Live run · approved reference model")

    def prediction_card(self, score, lower, upper, source):
        action = int(triage_actions([score], lower, upper)[0])
        route_codes = ["ROUTINE CLINICAL QUEUE", "EXTRA HUMAN PRIORITY CHECK", "ACCELERATED CLINICAL REVIEW"]
        headings = ["Usual review queue", "An extra priority check", "Earlier clinical review"]
        details = ["Keep this image in the usual clinical review queue.",
                   "Ask a person to check whether this image should move earlier in the queue.",
                   "Move this image to the earlier clinical review queue in this simulation."]
        reasons = ["The model score is below the extra-check boundary.",
                   "The model score sits between the two review boundaries.",
                   "The model score meets the earlier-review boundary."]
        colours = ["routine", "review", "earlier"]
        selected = self.selected["hybrid"]
        reference = lower == selected["lower"] and upper == selected["upper"]
        policy = "Approved reference thresholds" if reference else "Exploratory thresholds"
        return (f'<section class="route-card {colours[action]}" data-route="{route_codes[action]}" '
            f'data-score="{score:.12g}" data-lower="{lower:.12g}" data-upper="{upper:.12g}">'
            f'<div class="eyebrow">Suggested next step</div><span class="route-dot" aria-hidden="true"></span>'
            f'<h2>{headings[action]}</h2><p class="route-action">{details[action]}</p>'
            f'<div class="route-reason">{reasons[action]}</div>'
            '<div class="score-line"><span>Model score</span>'
            f'<strong>{score:.4f}</strong></div><p class="score-explainer">'
            'This is a model confidence score, not a probability of disease.</p>'
            f'<div class="policy-stamp">{policy}<span>L {lower:.3f} · U {upper:.3f}</span></div>'
            f'<div class="case-source">{html.escape(source)}</div></section>')

    def render_prediction(self, image, all_boxes, score, lower, upper, source):
        """Shared rendering for cached examples and real upload inference."""
        output = image.copy()
        draw, rows = ImageDraw.Draw(output), []
        font_size = max(13, round(max(image.size) / 65))
        try:
            font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", font_size)
        except OSError:
            font = ImageFont.load_default()
        for cls, confidence, coords in all_boxes:
            if confidence < lower:
                continue
            colour = "#147d78" if confidence >= upper else "#b27620"
            draw.rectangle(coords, outline=colour, width=max(2, round(max(image.size) / 350)))
            label = f"{self.names[cls]} {confidence:.3f}"
            x, y = coords[:2]
            top = max(0, y - font_size - 5)
            draw.rectangle(draw.textbbox((x, top), label, font=font), fill=colour)
            draw.text((x, top), label, font=font, fill="white")
            rows.append([self.names[cls], round(confidence, 4), "Accelerated" if confidence >= upper else "Review band"])
        text = self.prediction_card(score, lower, upper, source)
        return output, text, pd.DataFrame(rows, columns=["Finding", "Confidence", "Box score band"])

    def select_example(self, choice, lower, upper):
        case = self.demo_examples.get(choice)
        if case is None:
            raise gr.Error("Choose one of the bundled validation examples.")
        with Image.open(case["path"]) as source:
            original = source.convert("RGB")
        width, height = original.size
        all_boxes = [(box["class_id"], box["confidence"],
                      [value * (width if index % 2 == 0 else height) for index, value in enumerate(box["xyxy"])])
                     for box in case["predictions"]]
        annotated, card, rows = self.render_prediction(original, all_boxes, case["cached_score"],
            float(lower), float(upper), "Saved validation prediction · approved reference model")
        annotation = ", ".join(self.names[i] for i in sorted({box["class_id"] for box in case["labels"]})) or "No listed finding"
        note = (f"**Published annotation:** {annotation}. This is a selected illustration of a route, not a random performance sample. "
                "Boxes show model suggestions; the source retains representative regions rather than every lesion.")
        return original, annotated, card, rows, note

    def fetch_examples(self, lower, upper):
        from download_examples import download_examples
        try:
            bundle = download_examples(self.root)
        except ValueError as error:
            raise gr.Error(str(error)) from None
        self.demo_examples = {case["id"]: case for case in bundle["examples"]}
        default = "extra_check" if "extra_check" in self.demo_examples else next(iter(self.demo_examples))
        choices = [(f"Example {index + 1}", route) for index, route in
                   enumerate(("routine", "extra_check", "accelerated")) if route in self.demo_examples]
        return (gr.update(choices=choices, value=default, visible=True), gr.update(visible=False),
                *self.select_example(default, lower, upper))

    def prepare_upload(self, filepath):
        if filepath is None:
            raise gr.Error("Choose an image file first.")
        with Image.open(filepath) as source:
            original = source.convert("RGB")
        card = ('<section class="route-card ready"><div class="eyebrow">Your image is ready</div>'
                '<h2>See its suggested route</h2><p class="route-action">Select “Run model on this image” below.</p>'
                '<p class="score-explainer">The model runs locally. This educational output does not provide a diagnosis.</p></section>')
        return original, original.copy(), card, pd.DataFrame(columns=["Finding", "Confidence", "Box score band"]), \
            "**Your uploaded image.** No published annotation is available for comparison. Every case still needs clinical assessment."

    def reset_thresholds(self):
        selected = self.selected["hybrid"]
        lower, upper = selected["lower"], selected["upper"]
        return lower, upper, *self.policy_view(lower, upper)

    def reference_workload(self):
        s = self.scenario
        choice = self.selected["hybrid"]
        result = evaluate_policy(self.test_truth, self.test_scores, choice["lower"], choice["upper"], s)
        fraction = min(100., 100 * result["daily_reviews"] / s.daily_capacity) if s.daily_capacity else 100.
        minutes = result["daily_review_hours"] * 60
        staff = "reviewer" if s.reviewers_available == 1 else "reviewers"
        each = "" if s.reviewers_available == 1 else " each"
        return (f'<section class="workload-panel"><div class="eyebrow">A hypothetical day · approved reference policy</div>'
            '<h2>What would the extra workload look like?</h2><div class="workload-flow">'
            f'<div><strong>{s.daily_images:,}</strong><span>X-rays arrive each day</span></div>'
            '<span class="flow-arrow" aria-hidden="true">→</span>'
            f'<div class="workload-emphasis"><strong>{result["daily_reviews"]:.1f}</strong><span>extra priority checks</span></div>'
            '<span class="flow-divider" aria-hidden="true">/</span>'
            f'<div><strong>{s.daily_capacity:g}</strong><span>checks available per day</span></div></div>'
            '<div class="capacity-track" role="img" '
            f'aria-label="{result["daily_reviews"]:.1f} expected extra checks out of {s.daily_capacity:g} available">'
            f'<span style="width:{fraction:.2f}%"></span></div>'
            f'<p class="capacity-caption">About {minutes:.0f} minutes of extra checking, from '
            f'{s.reviewers_available:g} {staff} with {s.review_hours_per_person:g} hours available{each}.</p>'
            '<div class="assumption-chips">'
            f'<span>{s.prevalence:.0%} assumed finding prevalence</span><span>{s.review_minutes:g} min / extra check</span>'
            f'<span>{s.reviewer_sensitivity:.0%} sensitivity · {s.reviewer_specificity:.0%} specificity assumed for reviewers</span></div>'
            f'<p class="small-note">A projection using {len(self.test_truth):,} held-out images, not an observed hospital day. '
            'All images still receive clinical assessment. Reviewer accuracy and workload are assumptions.</p></section>')

    def per_class_frame(self):
        native = {r["class_id"]: r for r in self.metrics["native"]["per_class"]}
        threshold = self.selected["binary"]["lower"]
        return pd.DataFrame([{"Finding": r["class_name"], "Test objects": r["ground_truth_instances"],
            "AP50": native[r["class_id"]]["ap50"], "mAP50–95": native[r["class_id"]]["ap50_95"],
            f"Precision @ {threshold:.3f}": r["precision"], f"Recall @ {threshold:.3f}": r["recall"]}
            for r in self.fixed["per_class"]]).round(4)

    def baseline_frame(self):
        rows = []
        for key in ("finetuned", "coco_zero_shot", "predict_nothing"):
            model = self.baselines["models"][key]
            native, observed = model["native"]["overall"], model["observed_image_level"]
            operating = model.get("projected_model_only", model.get("projected_no_target_output"))
            rows.append({"Model": model["name"], "Test images": self.baselines["images"],
                         "AP50": native["map50"], "mAP50–95": native["map50_95"],
                         "Image sensitivity": observed["sensitivity"], "Image specificity": observed["specificity"],
                         "Hypothetical EUR / 1,000": operating["cost_per_1000"]})
        return pd.DataFrame(rows).round(4)

    def object_confusion(self):
        confusion = self.fixed["confusion"]
        matrix = np.asarray(confusion["matrix"], dtype=float).copy()
        matrix[-1, -1] = np.nan
        labels = [[str(value) for value in row] for row in confusion["matrix"]]
        labels[-1][-1] = "N/A"
        fig = go.Figure(go.Heatmap(z=matrix, x=confusion["labels"], y=confusion["labels"],
            text=labels, texttemplate="%{text}",
            colorscale=[[0, "#eef2f8"], [1, "#315a8e"]],
            hovertemplate="True: %{y}<br>Predicted: %{x}<br>Count: %{text}<extra></extra>"))
        fig.add_annotation(x=confusion["labels"][-1], y=confusion["labels"][-1], text="N/A",
                           showarrow=False, font=dict(color="#526576", size=11))
        fig.update_layout(title=f"Raw object confusion — confidence ≥ {self.selected['binary']['lower']:.3f}, IoU ≥ 0.50",
            xaxis_title="Predicted class", yaxis_title="True class", yaxis=dict(autorange="reversed"),
            xaxis=dict(tickangle=-45), height=max(450, 32 * len(confusion["labels"]) + 220),
            margin=dict(l=160, r=30, t=70, b=170), font=dict(family="Arial", size=11), template="plotly_white")
        return fig

    def alternatives_frame(self):
        records = alternatives(self.test_truth, self.test_scores, self.selected, self.scenario)
        return pd.DataFrame([{"Policy": r["policy"], "EUR / 1,000": round(r["cost_per_1000"], 2),
            "Extra checks / day": round(r["daily_reviews"], 2), "Staff hours / day": round(r["daily_review_hours"], 2),
            "People needed": r["reviewers_needed"], "Within capacity": r["within_capacity"]} for r in records])

    def build(self):
        selected = self.selected["hybrid"]
        lower, upper = selected["lower"], selected["upper"]
        initial = self.policy_view(lower, upper)
        bundle_path = self.root / "assets/demo/manifest.json"
        bundle = load_demo_assets(self.root) if bundle_path.exists() else {"examples": []}
        self.demo_examples = {case["id"]: case for case in bundle["examples"]}
        default_case = "extra_check" if "extra_check" in self.demo_examples else next(iter(self.demo_examples), None)
        empty_rows = pd.DataFrame(columns=["Finding", "Confidence", "Box score band"])
        if default_case:
            original, preview, first_card, first_rows, first_note = self.select_example(default_case, lower, upper)
        else:
            original = preview = None
            first_card = ('<section class="route-card ready"><div class="eyebrow">Try the model</div>'
                          '<h2>Start with an X-ray</h2><p>Upload an image, then run the reference model.</p></section>')
            first_rows, first_note = empty_rows, "Upload an image to see its suggested review route."
        entry_hint = ("A saved example is ready below. Try another example or upload an image."
                      if default_case else "Upload an image, then run the model to see its suggested review route.")
        with gr.Blocks(title="X-ray review, explained", analytics_enabled=False) as demo:
            gr.HTML('<header class="masthead"><div class="wordmark"><span class="brand-icon" aria-hidden="true">+</span>'
                    'X-ray review, explained</div><span class="reference-pill">Assignment prototype</span></header>'
                    f'<section class="intro"><div class="eyebrow">An educational study · {len(self.names)} finding categories</div>'
                    '<h1>See how an X-ray is prioritised.</h1>'
                    '<p>Explore how a trained model helps a hypothetical radiology team organise its review queue. '
                    '<strong>Every image still receives clinical assessment.</strong></p>'
                    '<div class="scope-note">This study uses finding labels as a priority proxy. '
                    'It does not diagnose patients or establish clinical urgency.</div></section>'
                    '<div class="section-heading opening-step"><span class="section-number">01</span><div><h2>Start with one image</h2>'
                    f'<p>{entry_hint}</p></div></div>',
                    elem_id="opening-block")
            uploaded = gr.Image(value=original, type="pil", format="png", visible=False, buttons=[])
            with gr.Row(equal_height=False, elem_id="case-row"):
                with gr.Column(scale=7, min_width=330, elem_id="case-image-panel"):
                    annotated = gr.Image(value=preview, interactive=False, show_label=False,
                                         height=480, format="png", buttons=["fullscreen"], elem_id="case-preview")
                    gr.HTML('<p class="image-caption">Outlined regions are model suggestions, not confirmed diagnoses.</p>')
                with gr.Column(scale=5, min_width=300, elem_id="case-result-panel"):
                    description = gr.HTML(first_card, elem_id="case-result")
                    choices = [("Example 1", "routine"), ("Example 2", "extra_check"), ("Example 3", "accelerated")]
                    examples = gr.Radio(choices=[item for item in choices if item[1] in self.demo_examples],
                        value=default_case, label="Try a saved validation example", interactive=True,
                        visible=bool(self.demo_examples), elem_id="example-choice")
                    fetch_button = gr.Button("Load three examples from Roboflow", visible=not bool(self.demo_examples))
                    if not self.demo_examples:
                        gr.Markdown("Examples need internet and your Roboflow key from setup. Only three images are retrieved and kept locally. You can also upload your own image.")
                    with gr.Row():
                        upload_button = gr.UploadButton("Upload an image", file_types=["image"],
                            file_count="single", type="filepath", variant="secondary", min_width=130)
                        run_button = gr.Button("Run model on this image", variant="primary", min_width=180)
                    gr.HTML('<p class="local-note">Runs locally using the approved reference model. '
                            'Candidate experiments do not replace it automatically.</p>')
            with gr.Accordion("About this image and its suggested regions", open=False, elem_classes=["quiet-accordion"]):
                example_note = gr.Markdown(first_note)
                boxes = gr.Dataframe(first_rows, interactive=False, wrap=True,
                    label="Suggested regions above the extra-check boundary", max_height=260)
                gr.Markdown("A model score is an uncalibrated confidence value. No displayed region does not establish health. "
                            "An image may contain several listed findings; its route uses the largest model score.")
            gr.HTML(ROUTING_HTML)
            gr.HTML('<div class="section-heading"><span class="section-number">02</span><div><h2>See the workload</h2>'
                    '<p>A scenario estimate, with the approved policy held fixed.</p></div></div>')
            gr.HTML(self.reference_workload())
            with gr.Accordion("Understand the assumptions behind these numbers", open=False, elem_classes=["quiet-accordion"]):
                s = self.scenario
                gr.Markdown(f"The scenario assumes **{s.daily_images} images per day**, with a listed finding in "
                    f"**{s.prevalence:.0%}** of images. An extra priority check takes **{s.review_minutes:g} minutes** "
                    f"and costs **€{s.review_eur:g}**, at **€{s.reviewer_hourly_eur:g}/hour**. "
                    f"Reviewers are assumed to have **{s.reviewer_sensitivity:.0%} sensitivity** and "
                    f"**{s.reviewer_specificity:.0%} specificity**, including on uncertain images. Their accuracy was not measured.\n\n"
                    f"A missed priority carries a hypothetical **€{s.false_negative_eur:g}** penalty; "
                    f"an unnecessary priority carries **€{s.false_positive_eur:g}**. Model use adds "
                    f"**€{s.inference_eur_per_image:g}/image**. These are chosen scenario assumptions, not measured hospital costs. "
                    "Routine clinical assessment is shared across alternatives and excluded equally from incremental costs.\n\n"
                    "Capacity is an expected-workload limit, not a promise that every day will fit. "
                    "These calculations do not establish readiness for clinical deployment.")
            gr.HTML('<div class="section-heading"><span class="section-number">03</span><div><h2>Go deeper, if you want</h2>'
                    '<p>Explore the trade-off or inspect the evidence behind the study.</p></div></div>')
            with gr.Accordion("Explore the trade-off: extra checks, missed priorities and cost", open=False,
                              elem_classes=["explore-accordion"]):
                gr.Markdown("### Try different review boundaries\n"
                    "These controls recalculate the **validation** results. They do not change the approved reference policy "
                    "or its held-out test results. The next image run or example selection uses these settings; "
                    "the image card always records which boundaries produced its displayed result.")
                with gr.Row():
                    low = gr.Slider(0, 1.001, value=lower, step=.001, label="L · Start extra priority checks",
                                    elem_classes=["threshold-control"])
                    high = gr.Slider(0, 1.001, value=upper, step=.001, label="U · Move to earlier review",
                                     elem_classes=["threshold-control"])
                gr.Markdown("Below L: usual queue. Between L and U: extra priority check. At or above U: earlier review. "
                    "Crossing the controls brings them together. 1.001 is above all scores, so it can turn a route off; "
                    "it is not a probability. At L = U = 0, every image goes earlier, even without a detected region.")
                reset = gr.Button("Restore approved reference boundaries", variant="secondary", size="sm")
                cards, capacity = gr.HTML(initial[0]), gr.Markdown(initial[1])
                gr.Markdown("The line shows the model working alone at each single threshold. "
                            "The summary above shows the model plus human checks between L and U.")
                curve = gr.Plot(initial[4], show_label=False, elem_id="cost-curve")
                with gr.Accordion("See how validation images are routed", open=False):
                    with gr.Row():
                        observed = gr.Plot(initial[2], show_label=False)
                        expected = gr.Plot(initial[3], show_label=False)
                    gr.Markdown("Left: observed routing counts. Right: expected final decisions per 1,000 after applying "
                        "the hypothetical prevalence and reviewer accuracy. Decimal expected counts are not observed people.")
            with gr.Accordion("Inspect the study evidence and model limitations", open=False,
                              elem_classes=["evidence-accordion"]):
                native = self.metrics["native"]["overall"]
                exposure_note = ("This test partition was already inspected for the original model. This is a follow-up evaluation, not a new blind test. "
                                 if self.freeze.get("test_previously_exposed") else "")
                gr.Markdown(f"### Evaluation on {len(self.test_truth):,} held-out images\n"
                    + exposure_note + "The policy was selected using validation data and frozen before this model's test inference. "
                    "Test results describe this dataset and do not validate clinical urgency or patient independence.\n\n"
                    f"**Detection AP50: {native['map50']:.4f} · mAP50–95: {native['map50_95']:.4f}.** "
                    "These measure agreement with the published representative boxes. They are separate from image-level routing.")
                r = evaluate_policy(self.test_truth, self.test_scores, lower, upper, self.scenario)
                with gr.Accordion("Held-out image routing and operating alternatives", open=False):
                    gr.Plot(heatmap(r["observed_actions"], ["Usual queue", "Extra check", "Earlier review"],
                                    "Observed test routing — approved policy"), show_label=False)
                    gr.Dataframe(self.alternatives_frame(), interactive=False, wrap=True,
                                 label="Hypothetical operating alternatives", max_height=330)
                with gr.Accordion("Finding-by-finding detection results", open=False):
                    gr.Markdown(f"Precision and recall use the frozen model-only threshold "
                        f"**{self.selected['binary']['lower']:.3f}** and IoU ≥ 0.50. "
                        "AP integrates confidence rankings. Blank precision is undefined when there are no predictions.")
                    gr.Dataframe(self.per_class_frame(), interactive=False, wrap=True, max_height=600,
                                 label="Per-class object results")
                with gr.Accordion("Object confusion matrix", open=False):
                    gr.Plot(self.object_confusion(), show_label=False)
                    gr.Markdown("Rows are true classes; columns are predicted classes. Background marks unmatched boxes. "
                        "**Background/background is N/A** because true-negative boxes cannot be enumerated. "
                        "The underlying raw matrix retains its storage placeholder; it is not a measured zero. "
                        "This matrix matches geometry across classes; class precision/recall use class-aware matching.")
                with gr.Accordion("Compare the three model baselines", open=False):
                    gr.Dataframe(self.baseline_frame(), interactive=False, wrap=True, label="Same held-out images", max_height=300)
                    gr.Markdown("The original COCO model genuinely ran on these images. Its 80 categories have no "
                        "equivalent among the medical target classes, so the explicit semantic mapping is empty. "
                        "Its target outputs therefore match predict-nothing. Numeric class IDs are never treated as shared meanings. "
                        "This comparison does not demonstrate medical zero-shot ability.")
                with gr.Accordion("Dataset, provenance and limits of the demonstration", open=False):
                    gr.Markdown("\n".join("- " + text for text in self.config["dataset"]["accepted_limitations"]))
                    gr.Markdown("[VinBigData-derived chest X-rays · exact Roboflow version 3]"
                        "(https://universe.roboflow.com/vinbigdataxrayproject/chest-xray-symptom-detection/dataset/3). "
                        "The bundled examples are original validation-image bytes selected by median score within each route, "
                        "without filtering for correctness. Source IDs, image hashes and saved prediction provenance are verified locally.\n\n"
                        "The export reports CC BY 4.0, while upstream Kaggle and PhysioNet releases carry different conditions. "
                        "The uploader's licensing authority is unverified. These examples remain local; no redistribution is authorised. "
                        "The model has not been clinically validated or deployed as a medical service.")
            low.input(self.lower_changed, [low, high], [high, cards, capacity, observed, expected, curve], api_name="change_lower")
            high.input(self.upper_changed, [low, high], [low, cards, capacity, observed, expected, curve], api_name="change_upper")
            reset.click(self.reset_thresholds, [], [low, high, cards, capacity, observed, expected, curve], api_name="reset_thresholds")
            examples.input(self.select_example, [examples, low, high], [uploaded, annotated, description, boxes, example_note],
                           api_name="select_example")
            fetch_button.click(self.fetch_examples, [low, high],
                               [examples, fetch_button, uploaded, annotated, description, boxes, example_note],
                               api_name="load_examples")
            upload_button.upload(self.prepare_upload, [upload_button], [uploaded, annotated, description, boxes, example_note],
                                 api_name="prepare_upload")
            run_button.click(self.predict_image, [uploaded, low, high], [annotated, description, boxes], api_name="predict_image")
            gr.HTML('<footer class="page-footer"><span>X-ray review, explained</span>'
                    '<span>Educational demonstration · local reference model · every case receives clinical review</span></footer>')
        return demo


ROUTING_HTML = """
<section class="routing-panel" aria-label="Three possible routes, all leading to clinical assessment">
  <div class="routing-origin"><span class="origin-dot"></span>One image · one model score</div>
  <div class="route-branches">
    <div class="route-path routine"><span class="path-dot"></span><h3>Usual queue</h3><p>Clinical review in the usual queue.</p></div>
    <div class="route-path review"><span class="path-dot"></span><h3>Extra priority check</h3><p>A person checks whether to move it earlier.</p></div>
    <div class="route-path earlier"><span class="path-dot"></span><h3>Earlier review</h3><p>Move earlier in the clinical review queue.</p></div>
  </div>
  <div class="routing-destination">Every path leads to clinical assessment. No route means “healthy” or “cleared”.</div>
</section>
"""


CSS = """
:root {--ink:#182f42;--muted:#5b6d7c;--teal:#147d78;--amber:#a76c18;--line:#dfe7eb;--page:#f6f8f9;--white:#fff}
body,.gradio-container {background:var(--page)!important;color:var(--ink)!important}
.gradio-container {max-width:1180px!important;margin:0 auto!important;padding:8px 30px 0!important;font-family:Arial,sans-serif!important}
.gradio-container .prose,.gradio-container label {color:var(--ink)!important}
.gradio-container button {font-weight:600!important;border-radius:10px!important;min-height:42px}
.gradio-container button.primary {background:var(--teal)!important;border-color:var(--teal)!important;color:white!important}
.gradio-container button.secondary {background:white!important;border-color:#cbd8df!important;color:var(--ink)!important}
.gradio-container input[type=range] {accent-color:var(--teal)}
.threshold-control label,.threshold-control span,.threshold-control input[type=number] {color:#405969!important}
.gradio-container .header-cell {background:#e9f0f3!important;color:#182f42!important}
.gradio-container .virtual-row,.gradio-container .body-cell,.gradio-container .cell-wrap {color:#182f42!important}
.gradio-container .body-cell.cell-selected,.gradio-container .header-cell.focus {background:#dceeea!important;color:#182f42!important}
.gradio-container .virtual-row:hover {background:#dceeea!important}
#opening-block {padding:0!important;border:0!important;background:transparent!important}
.masthead {display:flex;align-items:center;justify-content:space-between;padding:4px 0 8px;border-bottom:1px solid var(--line);gap:15px}
.wordmark {display:flex;align-items:center;gap:10px;font-size:16px;font-weight:700;letter-spacing:-.2px}
.brand-icon {width:29px;height:29px;display:grid;place-items:center;background:#e1f0ed;color:var(--teal);font-size:25px;border-radius:8px;font-weight:400}
.reference-pill {font-size:11px;font-weight:600;letter-spacing:.3px;background:#e9eef3;color:#40566c;padding:8px 12px;border-radius:30px;white-space:nowrap}
.intro {padding:14px 0 6px;max-width:950px}
.eyebrow {font-size:11px;letter-spacing:1.4px;text-transform:uppercase;font-weight:700;color:#607486;line-height:1.7}
.intro h1 {font-size:36px;line-height:1.14;letter-spacing:-1.2px;margin:6px 0 8px;color:var(--ink);font-weight:700}
.intro>p {font-size:16px;line-height:1.55;color:#435967;margin:0;max-width:900px}
.intro strong {font-weight:600;color:var(--ink)}
.scope-note {font-size:13px;color:#657887;margin-top:7px;line-height:1.55}
.section-heading {display:flex;align-items:center;gap:14px;margin:22px 0 16px}
.opening-step {margin:12px 0 10px}
.section-number {font-size:12px;font-weight:700;letter-spacing:1px;color:var(--teal);border:1px solid #c9ddda;border-radius:50%;width:36px;height:36px;display:grid;place-items:center;flex-shrink:0}
.section-heading h2 {font-size:23px;letter-spacing:-.5px;line-height:1.3;margin:0 0 4px;color:var(--ink)}
.section-heading p {font-size:16px;line-height:1.5;color:var(--muted);margin:0}
#case-row {gap:24px!important;align-items:stretch}
#case-image-panel,#case-result-panel {background:transparent!important;border:0!important;gap:10px!important}
#case-preview {border-radius:18px!important;overflow:hidden;border:1px solid var(--line)!important;background:#101b22!important}
#case-preview img {object-fit:contain!important;background:#101b22}
.image-caption {font-size:11px;line-height:1.6;color:#657887;text-align:center;margin:2px 0}
.route-card {background:#fff;border:1px solid var(--line);border-radius:18px;padding:25px 27px 20px;position:relative;min-height:328px;border-top:4px solid #607488}
.route-card.review {border-top-color:#c38b37}.route-card.earlier {border-top-color:var(--teal)}
.route-card .route-dot {width:10px;height:10px;border-radius:50%;background:#607488;display:block;margin:20px 0 10px}
.route-card.review .route-dot {background:#c38b37}.route-card.earlier .route-dot {background:var(--teal)}
.route-card h2 {font-size:29px;line-height:1.17;letter-spacing:-.8px;margin:0 0 13px;color:var(--ink);font-weight:700;max-width:350px}
.route-action {font-size:16px;line-height:1.6;color:#3f5667;margin:0 0 13px}
.route-reason {font-size:13px;color:#697b87;line-height:1.6;margin-bottom:20px}
.score-line {display:flex;justify-content:space-between;align-items:baseline;gap:20px;border-top:1px solid #e7edef;padding-top:15px}
.score-line span {font-size:12px;color:#586d7c}.score-line strong {font-size:24px;letter-spacing:-.5px;color:var(--ink);font-variant-numeric:tabular-nums}
.score-explainer {font-size:13px;line-height:1.6;color:#647784;margin:5px 0 14px}
.policy-stamp {font-size:10px;font-weight:600;letter-spacing:.2px;display:flex;justify-content:space-between;gap:10px;color:#506779;background:#f1f5f6;padding:8px 10px;border-radius:6px}
.policy-stamp span {white-space:nowrap;font-weight:400}.case-source {font-size:10px;color:#788994;margin-top:10px;line-height:1.6}
#example-choice {background:transparent!important;border:0!important;padding:2px 0!important}
#example-choice .wrap {background:transparent!important}
#example-choice label {font-size:12px!important;background:#fff!important;color:#182f42!important;border-color:#cbd8df!important;box-shadow:none!important}
#example-choice label:hover {background:#eef5f4!important;border-color:#91bdb5!important}
#example-choice label:has(input:checked) {background:#e0efec!important;color:#134e4a!important;border-color:#147d78!important}
#example-choice label span {color:inherit!important}
#example-choice input {accent-color:var(--teal);background-color:white!important;border-color:#8ba5b3!important}
#example-choice input:checked {background-color:#147d78!important;border-color:#147d78!important}
.local-note {font-size:11px;line-height:1.65;color:#687e89;margin:1px 0 0}
.quiet-accordion,.explore-accordion,.evidence-accordion {border:1px solid var(--line)!important;border-radius:12px!important;background:#fff!important;margin:7px 0!important;box-shadow:none!important;padding:0!important;min-height:0!important;overflow:hidden!important}
.quiet-accordion>div:first-child,.explore-accordion>div:first-child,.evidence-accordion>div:first-child {color:var(--ink)!important}
.gradio-container button.label-wrap {display:flex!important;align-items:center!important;justify-content:space-between!important;box-sizing:border-box!important;width:100%!important;max-width:100%!important;min-width:0!important;min-height:46px!important;height:auto!important;padding:12px 16px!important;margin:0!important;text-align:left;background:#fff!important;color:#182f42!important;border-radius:0!important;gap:16px!important;overflow:hidden!important}
.gradio-container button.label-wrap>span:first-child {min-width:0;white-space:normal;line-height:1.5}
.gradio-container button.label-wrap>.icon {flex:none;font-size:10px;color:#58707f}
.quiet-accordion>[data-testid="accordion-content"],.explore-accordion>[data-testid="accordion-content"],.evidence-accordion>[data-testid="accordion-content"] {padding:16px;box-sizing:border-box;min-width:0;max-width:100%}
.gradio-container [data-testid="accordion-content"][style*="display: none"] {padding:0!important;height:0!important;overflow:hidden!important}
.routing-panel {margin:30px 0 12px;padding:24px 25px 20px;background:#fff;border:1px solid var(--line);border-radius:16px}
.routing-origin {text-align:center;font-size:12px;font-weight:600;color:#536b7c;margin-bottom:20px;display:flex;justify-content:center;gap:8px;align-items:center}
.origin-dot {display:inline-block;width:8px;height:8px;background:var(--ink);border-radius:50%}
.route-branches {display:grid;grid-template-columns:repeat(3,1fr);gap:24px;position:relative;padding-top:18px}
.route-branches:before {content:'';position:absolute;left:16.5%;right:16.5%;top:0;border-top:1px solid #b9cbd1}
.route-path {text-align:center;position:relative}.route-path:before {content:'';position:absolute;top:-18px;left:50%;height:17px;border-left:1px solid #b9cbd1}
.path-dot {display:block;width:9px;height:9px;border-radius:50%;margin:0 auto 10px;background:#607488}
.route-path.review .path-dot {background:#c38b37}.route-path.earlier .path-dot {background:var(--teal)}
.route-path h3 {font-size:14px;margin:0 0 5px;color:var(--ink);font-weight:600}.route-path p {font-size:12px;line-height:1.6;margin:0;color:#687d8a}
.routing-destination {text-align:center;margin-top:21px;padding-top:15px;border-top:1px solid #edf1f3;font-size:11px;line-height:1.7;color:#5d7482}
.workload-panel {background:#ecf4f3;border:1px solid #d1e4e0;border-radius:18px;padding:27px 32px 22px}
.workload-panel h2 {font-size:24px;line-height:1.3;letter-spacing:-.5px;margin:8px 0 26px;color:var(--ink)}
.workload-flow {display:grid;grid-template-columns:1fr 35px 1fr 25px 1fr;align-items:center;gap:12px}
.workload-flow>div {display:flex;flex-direction:column;gap:7px}.workload-flow strong {font-size:40px;font-weight:700;line-height:1.05;letter-spacing:-1.5px;color:var(--ink);font-variant-numeric:tabular-nums}
.workload-flow span {font-size:12px;color:#58727b;line-height:1.5}.workload-flow .workload-emphasis strong {color:var(--teal)}
.workload-flow .flow-arrow {font-size:24px;color:#6c9393}.workload-flow .flow-divider {font-size:32px;color:#aac5c3;font-weight:300}
.capacity-track {height:7px;background:#d6e6e3;border-radius:10px;margin-top:25px;overflow:hidden}
.capacity-track>span {height:100%;display:block;background:var(--teal);border-radius:10px}
.capacity-caption {font-size:12px;color:#4e6c75;margin:10px 0 20px;line-height:1.6}
.assumption-chips {display:flex;flex-wrap:wrap;gap:7px}.assumption-chips span {font-size:10px;padding:6px 9px;border:1px solid #d0e2de;border-radius:5px;background:#f7fbfa;color:#476873}
.small-note {font-size:11px;color:#637d85;line-height:1.7;margin:13px 0 0;max-width:850px}
.metric-row {display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:14px 0}
.metric-card {padding:17px;border:1px solid var(--line);border-radius:10px;background:#f4f7f8}
.metric-card span {display:block;font-size:11px;color:#58707f;margin-bottom:9px;line-height:1.5}
.metric-card strong {font-size:23px;color:var(--ink);line-height:1.25;letter-spacing:-.4px}
.page-footer {display:flex;justify-content:space-between;gap:20px;padding:27px 0;margin-top:24px;border-top:1px solid var(--line);color:#72848e;font-size:10px;line-height:1.7}
.page-footer>span:first-child {font-weight:600;color:#4d6574;white-space:nowrap}
@media(max-width:800px){.gradio-container {padding:8px 18px 0!important}.intro h1{font-size:33px}.intro{padding:12px 0 6px}.reference-pill{font-size:10px;padding:7px 9px}.wordmark{font-size:13px}.metric-row{grid-template-columns:repeat(2,1fr)}.workload-panel{padding:22px}.workload-flow strong{font-size:32px}.route-card{min-height:0}.route-branches{gap:12px}.route-path p{font-size:11px}.page-footer{flex-direction:column;gap:5px}}
@media(max-width:500px){.intro h1{font-size:29px}.workload-flow{gap:8px;grid-template-columns:1fr 18px 1fr 15px 1fr}.workload-flow strong{font-size:28px}.workload-flow span{font-size:10px}.route-card{padding:22px}.routing-panel{padding:20px 12px}.route-path h3{font-size:12px}}
"""


def dashboard_theme():
    """Keep the deliberately light interface coherent under either OS preference."""
    theme = gr.themes.Base(primary_hue="teal", secondary_hue="slate", neutral_hue="slate",
        font=["Arial", "sans-serif"], font_mono=["Consolas", "monospace"]).set(
        body_background_fill="#f6f8f9", body_text_color="#182f42", body_text_color_subdued="#5b6d7c",
        block_title_text_color="#405969", block_background_fill="#ffffff",
        background_fill_primary="#ffffff", background_fill_secondary="#f4f7f8",
        input_background_fill="#ffffff", block_label_text_color="#405969", border_color_primary="#dfe7eb",
        checkbox_label_background_fill="#ffffff", checkbox_label_background_fill_hover="#eef5f4",
        checkbox_label_background_fill_selected="#e0efec", checkbox_label_text_color="#182f42",
        checkbox_label_text_color_selected="#134e4a", table_text_color="#182f42",
        table_border_color="#d7e1e6", table_even_background_fill="#ffffff",
        table_odd_background_fill="#f2f6f8", table_row_focus="#dceeea",
        button_primary_background_fill="#147d78", button_primary_background_fill_hover="#106b66",
        button_primary_text_color="#ffffff", button_primary_text_color_hover="#ffffff")
    # Gradio exposes these supported theme tokens as attributes; .set is its public update API.
    # Include pairs whose original dark value was None, so every state shares the light palette.
    pairs = {key: getattr(theme, key[:-5]) for key in vars(theme)
             if key.endswith("_dark") and hasattr(theme, key[:-5])}
    # Some .set parameters have concrete defaults, so retain the paired light values too.
    light_values = {key[:-5]: value for key, value in pairs.items()}
    return theme.set(**light_values, **pairs)


def main():
    dashboard = Dashboard()
    demo = dashboard.build()
    demo.queue(default_concurrency_limit=1)
    demo.launch(server_name="127.0.0.1", server_port=int(os.environ.get("APP_PORT", "7860")),
        share=False, inbrowser=False, show_error=True, css=CSS, theme=dashboard_theme(),
        footer_links=[], blocked_paths=[str(ROOT / ".env"), str(ROOT / "datasets")])


if __name__ == "__main__":
    main()
