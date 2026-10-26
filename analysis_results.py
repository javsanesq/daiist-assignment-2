"""Cost/staffing evidence from cached predictions and a validation-frozen policy.

No training or inference occurs here. Test scores never select a threshold.
Reviewer outcomes and service costs are hypothetical expectations, not observed
clinical outcomes. Native object AP belongs to the separate model evaluation.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path

import numpy as np

from detection_metrics import image_scores
from triage import Scenario, alternatives, select_policies, threshold_grid, evaluate_policy, wilson_interval


POLICY_IDS = ("routine_all", "human_all", "model_only", "model_human")
ACTION_NAMES = ["routine", "human_priority_review", "accelerated_clinical_review"]
LIMITATIONS = [
    "All costs, operating prevalence, workload, review duration and reviewer accuracy are hypothetical assumptions.",
    "Priority-positive means at least one annotated target finding; it is not clinical urgency or a confirmed patient outcome.",
    "No finding means none of the target findings annotated, not an assurance that the patient is healthy.",
    "Score is the maximum target-box confidence and is uncalibrated; a score is not a disease probability.",
    "Reviewer sensitivity/specificity are assumed constant even on model-selected difficult cases; no reviewer study was performed.",
    "Wilson intervals are marginal image-level binomial intervals, conditional on the adopted threshold and observed class. They omit selection, label and patient-dependence uncertainty.",
    "Patient independence is unverified. Correlated images could make image-level intervals too narrow.",
    "No interval is attached to expected resolved confusion as if reviewer outcomes had been observed.",
    "Routine clinical assessment is common to all alternatives. Costs cover incremental prioritization and hypothetical priority-error penalties only.",
    "Sensitivity analyses are one-factor hypothetical projections, not validation of clinical utility or a search on test outcomes.",
]


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _digest(value, name):
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError(f"{name} must be a lowercase SHA256 digest.")
    return value


def validate_inputs(validation, test, frozen, scenario_document, *, validation_sha256, scenario_sha256):
    """Validate evidence linkage before examining test policy performance."""
    if scenario_document.get("approved") is not True:
        raise ValueError("The operating scenario must be explicitly approved.")
    scenario = Scenario(**scenario_document["values"])
    if frozen.get("schema_version") != 1 or frozen.get("selection_split") != "val":
        raise ValueError("Require a schema1 policy frozen on validation.")
    for field, actual in (("validation_predictions_sha256", validation_sha256), ("scenario_sha256", scenario_sha256)):
        if _digest(frozen.get(field), field) != actual:
            raise ValueError(f"Frozen {field} does not match the supplied file.")
    selected = frozen.get("selected", {})
    if selected.get("selection_split") != "val" or selected.get("scenario") != asdict(scenario):
        raise ValueError("Frozen selected policy must use validation and the approved scenario values.")
    for key in ("binary", "hybrid"):
        policy = selected.get(key, {})
        lower, upper = policy.get("lower"), policy.get("upper")
        if (isinstance(lower, bool) or isinstance(upper, bool)
                or not isinstance(lower, (int, float)) or not isinstance(upper, (int, float))
                or not np.isfinite([lower, upper]).all() or not 0 <= lower <= upper <= 1.001):
            raise ValueError(f"Frozen {key} thresholds are invalid.")
    if selected["binary"]["lower"] != selected["binary"]["upper"]:
        raise ValueError("The model-only binary policy cannot contain a review band.")
    if validation.get("split") not in ("val", "valid", "validation") or test.get("split") != "test":
        raise ValueError("Require separate validation and held-out test caches.")
    for field in ("class_names", "positive_class_ids", "inference"):
        if field not in validation or validation[field] != test.get(field):
            raise ValueError(f"Validation/test {field} must agree.")
    if not isinstance(validation["inference"], dict) or not validation["inference"]:
        raise ValueError("Inference settings must be explicitly recorded.")
    for field in ("weights_sha256", "manifest_sha256"):
        expected = _digest(frozen.get(field), field)
        if validation.get(field) != expected or test.get(field) != expected:
            raise ValueError(f"Frozen, validation and test {field} must agree.")
    val_ids, val_y, val_scores = image_scores(validation["records"], validation["class_names"], validation["positive_class_ids"])
    test_ids, test_y, test_scores = image_scores(test["records"], test["class_names"], test["positive_class_ids"])
    if set(val_ids) & set(test_ids):
        raise ValueError("Validation/test caches contain overlapping image IDs.")
    # evaluate_policy enforces both target classes and finite score ranges.
    evaluate_policy(val_y, val_scores, 0.5, 0.5, scenario)
    evaluate_policy(test_y, test_scores, 0.5, 0.5, scenario)
    return scenario, selected, (val_ids, val_y, val_scores), (test_ids, test_y, test_scores)


def observed_uncertainty(evaluation, *, direct_binary):
    """Uncertainty of observed routing only; no simulated reviewer trials."""
    counts = np.asarray(evaluation["observed_actions"], dtype=int)
    cells = []
    for target in (0, 1):
        total = int(counts[target].sum())
        for action in range(3):
            count = int(counts[target, action])
            cells.append({"target": target, "action": ACTION_NAMES[action], "count": count,
                          "class_total": total, "proportion": count / total,
                          "wilson_95": wilson_interval(count, total)})
    result = {"method": "Marginal Wilson score interval, z=1.96; image-level class-conditional routing",
              "routing": cells, "direct_binary": None,
              "expected_reviewer_outcome_intervals": None}
    if direct_binary:
        if counts[:, 1].sum():
            raise ValueError("A direct binary rule cannot include reviewed cases.")
        tn, fp, fn, tp = map(int, (counts[0, 0], counts[0, 2], counts[1, 0], counts[1, 2]))
        result["direct_binary"] = {
            "observed_confusion": [[tn, fp], [fn, tp]], "rows": ["negative", "positive"],
            "columns": ["routine", "accelerated"],
            "sensitivity": {"value": tp / (tp + fn), "wilson_95": wilson_interval(tp, tp + fn), "denominator": tp + fn},
            "specificity": {"value": tn / (tn + fp), "wilson_95": wilson_interval(tn, tn + fp), "denominator": tn + fp},
        }
    return result


def compare_policies(labels, scores, selected, scenario, *, uncertainty=True):
    """Evaluate four alternatives at supplied thresholds, with no selection."""
    results = alternatives(labels, scores, selected, scenario)
    for policy_id, result in zip(POLICY_IDS, results):
        result["policy_id"] = policy_id
        result["basis"] = "Observed image routing; hypothetical prevalence-weighted resolved outcomes and costs"
        result["uncertainty"] = observed_uncertainty(result, direct_binary=policy_id in ("routine_all", "model_only")) if uncertainty else None
        result["daily_review_capacity"] = float(scenario.daily_capacity)
        result["daily_backlog_if_no_extra_staff"] = max(0., result["daily_reviews"] - scenario.daily_capacity)
        result["capacity_utilization"] = result["daily_reviews"] / scenario.daily_capacity
        result["daily_cost"] = result["cost_per_1000"] * scenario.daily_images / 1000
    return results


def formula_evidence(scenario):
    """Worked numbers use assumptions only and must not be called observed."""
    p, n = scenario.prevalence, 1000
    human_fn = n * p * (1 - scenario.reviewer_sensitivity)
    human_fp = n * (1 - p) * (1 - scenario.reviewer_specificity)
    return {
        "units": {"cost": "EUR per 1000 images", "workload": "images per day", "review_time": "hours per day"},
        "review_cost": {"formula": "review_minutes / 60 * reviewer_hourly_eur", "substitution": f"{scenario.review_minutes} / 60 * {scenario.reviewer_hourly_eur}", "eur_per_review": scenario.review_eur},
        "capacity": {"formula": "reviewers_available * review_hours_per_person * 60 / review_minutes", "substitution": f"{scenario.reviewers_available} * {scenario.review_hours_per_person} * 60 / {scenario.review_minutes}", "reviews_per_day": scenario.daily_capacity, "maximum_review_fraction": min(1., scenario.daily_capacity / scenario.daily_images)},
        "expected_actions": "P(action|annotated class) * assumed class prevalence * 1000",
        "expected_fp": "direct_accelerated_negatives + reviewed_negatives * (1 - reviewer_specificity)",
        "expected_fn": "direct_routine_positives + reviewed_positives * (1 - reviewer_sensitivity)",
        "total_cost": "expected_FP * FP_penalty + expected_FN * FN_penalty + expected_reviews * review_cost + 1000 * inference_cost (model alternatives only)",
        "staffing": "daily_reviews = review_fraction * daily_images; daily_hours = daily_reviews * review_minutes / 60; people_needed = ceil(daily_hours / hours_per_person)",
        "hypothetical_routine_all": {"fn_per_1000": n * p, "fp_per_1000": 0., "cost_per_1000": n * p * scenario.false_negative_eur},
        "hypothetical_human_all": {"fn_per_1000": human_fn, "fp_per_1000": human_fp, "error_cost_per_1000": human_fn * scenario.false_negative_eur + human_fp * scenario.false_positive_eur, "review_cost_per_1000": n * scenario.review_eur, "cost_per_1000": human_fn * scenario.false_negative_eur + human_fp * scenario.false_positive_eur + n * scenario.review_eur, "daily_review_hours": scenario.daily_images * scenario.review_minutes / 60},
        "cost_scope": "Incremental prioritization only; shared routine clinical assessment is excluded consistently, not assumed free.",
    }


def scenario_variants(base):
    definitions = [
        ("prevalence", [.05, .10, .20], lambda s, v: replace(s, prevalence=v), "Assumed target prevalence"),
        ("fn_fp_ratio", [5., 20., 50.], lambda s, v: replace(s, false_negative_eur=25. * v, false_positive_eur=25.), "FN/FP penalty ratio (FP EUR25)"),
        ("reviewer_accuracy", [.80, .90, .95, 1.], lambda s, v: replace(s, reviewer_sensitivity=v, reviewer_specificity=v), "Assumed reviewer sensitivity and specificity"),
        ("review_minutes", [1., 2., 4.], lambda s, v: replace(s, review_minutes=v), "Minutes per priority review"),
        ("staff_hours", [1., 2., 4.], lambda s, v: replace(s, review_hours_per_person=v), "Review hours per available person per day"),
        ("inference_allowance", [0., .01, .05], lambda s, v: replace(s, inference_eur_per_image=v), "Inference allowance (EUR per image)"),
    ]
    for factor, values, mutate, label in definitions:
        for value in values:
            yield {"factor": factor, "value": value, "factor_label": label, "scenario": mutate(base, value)}


def sensitivity_analysis(val_labels, val_scores, test_labels, test_scores, selected, base, *, selector=select_policies):
    """Frozen-policy test projections and separate validation-only reselection."""
    selections, output = {}, []
    for variant in scenario_variants(base):
        scenario = variant.pop("scenario")
        key = json.dumps(asdict(scenario), sort_keys=True)
        if key not in selections:
            selections[key] = selector(val_labels, val_scores, scenario)
        reselected = selections[key]
        output.append({**variant, "scenario": asdict(scenario), "hypothetical": True,
                       "frozen_policy_test_projection": compare_policies(test_labels, test_scores, selected, scenario, uncertainty=False),
                       "validation_only_reselection": {"selection_split": "val", "test_evaluation": None,
                           "binary": reselected["binary"], "hybrid": reselected["hybrid"],
                           "note": "Thresholds chosen and evaluated on validation for sensitivity illustration only; no adoption or held-out retuning."}})
    return {"design": "One factor at a time; all other approved assumptions fixed", "test_optimization": False,
            "reselected_policy_test_evaluation": False, "scenarios": output}


def _write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _write_csv(path, rows, fields):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def create_plots(output_dir, comparison, sweep, sensitivity):
    """Small static charts, with expectation/observation distinctions in titles."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
                         "font.size": 10, "text.color": "black", "axes.labelcolor": "black",
                         "axes.titlecolor": "black", "axes.edgecolor": "black", "xtick.color": "black",
                         "ytick.color": "black", "axes.spines.top": False, "axes.spines.right": False,
                         "figure.facecolor": "white", "axes.facecolor": "white"})
    colors = ["#64748b", "#be7a34", "#287da8", "#684b91"]
    paths = []
    def save(fig, name):
        fig.savefig(output_dir / f"{name}.png", dpi=180, bbox_inches="tight")
        fig.savefig(output_dir / f"{name}.svg", bbox_inches="tight")
        plt.close(fig)
        paths.extend([f"{name}.png", f"{name}.svg"])

    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    ax.plot([r["lower"] for r in sweep], [r["cost_per_1000"] for r in sweep], color=colors[2])
    threshold = comparison["frozen_thresholds"]["binary"]["lower"]
    ax.axvline(threshold, color=colors[3], linestyle="--", label=f"Frozen threshold {threshold:g}")
    ax.set(xlabel="Model confidence threshold (1.001 = no positives)", ylabel="Expected EUR per 1,000 images", title="Validation binary cost sweep — hypothetical operating scenario")
    ax.legend(frameon=False)
    fig.tight_layout()
    save(fig, "validation_cost_sweep")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    labels = ["Routine\nall", "Human\nall", "Model\nonly", "Model +\nhuman"]
    rows = comparison["alternatives"]
    for ax, key, ylabel, title in ((axes[0], "cost_per_1000", "Expected EUR per 1,000 images", "Hypothetical incremental cost"), (axes[1], "daily_review_hours", "Expected priority-review hours / day", "Hypothetical staffing demand")):
        bars = ax.bar(labels, [r[key] for r in rows], color=colors)
        ax.bar_label(bars, fmt="%.1f", padding=3, fontsize=8, color="black")
        ax.set(ylabel=ylabel, title=title)
        ax.margins(y=.18)
    capacity_hours = comparison["scenario"]["reviewers_available"] * comparison["scenario"]["review_hours_per_person"]
    axes[1].axhline(capacity_hours, color="#b53a3a", linestyle="--", label="Assumed available review time")
    axes[1].legend(frameon=False, fontsize=8)
    fig.suptitle("Held-out test routing projected to the approved operating scenario", fontsize=11)
    fig.tight_layout()
    save(fig, "operational_comparison")

    factors = list(dict.fromkeys(r["factor"] for r in sensitivity["scenarios"]))
    nrows = (len(factors) + 2) // 3
    fig, axes = plt.subplots(nrows, 3, figsize=(12, 3.5 * nrows + .4), squeeze=False)
    short_labels = {"prevalence": "Assumed target prevalence", "fn_fp_ratio": "FN/FP penalty ratio (FP EUR25)",
                    "reviewer_accuracy": "Assumed reviewer sensitivity / specificity", "review_minutes": "Minutes per priority review",
                    "staff_hours": "Review hours per person / day", "inference_allowance": "Inference allowance (EUR / image)"}
    for ax, factor in zip(axes.flat, factors):
        rows = [r for r in sensitivity["scenarios"] if r["factor"] == factor]
        x = np.asarray([r["value"] for r in rows], dtype=float)
        for index, policy_id in enumerate(POLICY_IDS):
            ax.plot(x, [r["frozen_policy_test_projection"][index]["cost_per_1000"] for r in rows],
                    marker=["o", "s", "^", "D"][index], linestyle=["-", "--", "-.", ":"][index],
                    color=colors[index], label=labels[index].replace("\n", " "))
        ax.set_xticks(x, [f"{r['value']:g}" for r in rows])
        ax.set(xlabel=short_labels.get(factor, rows[0]["factor_label"]), ylabel="Expected EUR / 1,000")
    for ax in list(axes.flat)[len(factors):]:
        ax.axis("off")
    handles, legend_labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="lower center", ncols=4, frameon=False)
    fig.suptitle("Hypothetical sensitivity — original frozen policies, test routing\nEach panel uses its own vertical scale; one assumption varies at a time", fontsize=11)
    fig.tight_layout(rect=(0, .055, 1, .96))
    save(fig, "frozen_policy_sensitivity")

    fig, (ax, table_ax) = plt.subplots(2, 1, figsize=(8.5, 4.9), gridspec_kw={"height_ratios": [2.2, 1]})
    matrix = np.asarray(comparison["alternatives"][3]["observed_actions"])
    fractions = matrix / matrix.sum(axis=1, keepdims=True)
    left = np.zeros(2)
    action_colors = ["#e0e4e8", "#f0dfc5", "#c9dfec"]
    for action, color in enumerate(action_colors):
        ax.barh(["Annotated negative", "Annotated positive"], fractions[:, action], left=left,
                label=ACTION_NAMES[action].replace("_", " "), color=color, edgecolor="#555555", linewidth=.5)
        left += fractions[:, action]
    ax.set(xlabel="Observed fraction within annotated class", xlim=(0, 1), title="Frozen model + human policy — observed test routing")
    ax.invert_yaxis()
    table_ax.axis("off")
    # Every count, including zero and tiny segments, is shown in a separate table.
    cells = [[f"{int(matrix[row, action]):,} ({fractions[row, action]:.1%})" for action in range(3)] for row in range(2)]
    summary_table = table_ax.table(cellText=cells, rowLabels=["Annotated negative", "Annotated positive"],
        colLabels=["Routine", "Extra human check", "Accelerated review"], colColours=action_colors,
        cellLoc="center", rowLoc="left", loc="center", bbox=[.20, .10, .80, .80])
    summary_table.auto_set_font_size(False)
    summary_table.set_fontsize(9)
    for cell in summary_table.get_celld().values():
        cell.get_text().set_color("black")
        cell.set_edgecolor("#777777")
        cell.set_linewidth(.4)
    fig.tight_layout()
    save(fig, "observed_test_routing")
    return paths


def run_analysis(validation_path, test_path, frozen_path, scenario_path, output_dir, *, plots=True, selector=select_policies):
    paths = [Path(p) for p in (validation_path, test_path, frozen_path, scenario_path)]
    validation, test, frozen, scenario_doc = map(read_json, paths)
    scenario, selected, val_data, test_data = validate_inputs(validation, test, frozen, scenario_doc,
        validation_sha256=sha256_file(paths[0]), scenario_sha256=sha256_file(paths[3]))
    _, val_y, val_scores = val_data
    _, test_y, test_scores = test_data
    # Read thresholds from the frozen artifact. Never select using held-out data.
    comparison = {"schema_version": 1, "evaluation_split": "test", "hypothetical_operating_projection": True,
        "test_optimization": False, "scenario": asdict(scenario),
        "provenance": {"validation_predictions_sha256": sha256_file(paths[0]), "test_predictions_sha256": sha256_file(paths[1]), "frozen_policy_sha256": sha256_file(paths[2]), "scenario_sha256": sha256_file(paths[3]), "weights_sha256": frozen["weights_sha256"], "manifest_sha256": frozen["manifest_sha256"], "config_sha256": frozen.get("config_sha256"), "frozen_at_utc": frozen.get("frozen_at_utc"), "inference": validation["inference"]},
        "target": {"class_names": validation["class_names"], "positive_class_ids": validation["positive_class_ids"], "definition": "At least one annotated target box; priority proxy, not clinical urgency", "score": "Maximum target prediction confidence, or zero if no target prediction"},
        "sample": {"validation_images": len(val_y), "test_images": len(test_y), "test_annotated_positives": int(test_y.sum()), "test_annotated_negatives": int((test_y == 0).sum()), "observed_test_prevalence": float(test_y.mean()), "projected_operating_prevalence": scenario.prevalence},
        "matrix_conventions": {"observed_actions": {"rows": ["annotated_negative", "annotated_positive"], "columns": ACTION_NAMES, "units": "Observed image counts; reviewed outcomes unresolved"}, "expected_actions_per_1000": {"rows": ["annotated_negative", "annotated_positive"], "columns": ACTION_NAMES, "units": "Hypothetical expected counts per 1000 operating images"}, "expected_resolved_confusion_per_1000": {"rows": ["annotated_negative", "annotated_positive"], "columns": ["routine", "accelerated"], "units": "Hypothetical expected counts using assumed reviewer accuracy; not observed reviewer decisions"}},
        "frozen_thresholds": {k: {field: selected[k][field] for field in ("lower", "upper")} for k in ("binary", "hybrid")},
        "alternatives": compare_policies(test_y, test_scores, selected, scenario), "formulas_and_worked_numbers": formula_evidence(scenario), "limitations": LIMITATIONS}
    sweep = [evaluate_policy(val_y, val_scores, t, t, scenario) for t in threshold_grid()]
    sensitivity = sensitivity_analysis(val_y, val_scores, test_y, test_scores, selected, scenario, selector=selector)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    _write_json(output / "operational_comparison.json", comparison)
    _write_json(output / "sensitivity_analysis.json", sensitivity)
    _write_csv(output / "validation_binary_cost_sweep.csv", sweep, ["lower", "upper", "cost_per_1000", "error_cost_per_1000", "inference_cost_per_1000", "false_negatives_per_1000", "false_positives_per_1000", "sensitivity", "specificity"])
    _write_csv(output / "operational_comparison.csv", comparison["alternatives"], ["policy_id", "policy", "lower", "upper", "cost_per_1000", "error_cost_per_1000", "review_cost_per_1000", "inference_cost_per_1000", "false_negatives_per_1000", "false_positives_per_1000", "review_fraction", "daily_reviews", "daily_review_hours", "reviewers_needed", "within_capacity", "daily_backlog_if_no_extra_staff"])
    plot_files = create_plots(output, comparison, sweep, sensitivity) if plots else []
    _write_json(output / "analysis_artifacts.json", {"schema_version": 1, "files": ["operational_comparison.json", "operational_comparison.csv", "sensitivity_analysis.json", "validation_binary_cost_sweep.csv", *plot_files], "plots_generated": plots, "basis": "Frozen validation policy; held-out test evaluation; hypothetical operating projections"})
    return comparison


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation-cache", default="results/validation_predictions.json")
    parser.add_argument("--test-cache", default="results/test_predictions.json")
    parser.add_argument("--frozen-policy", default="results/frozen_policy.json")
    parser.add_argument("--scenario", default="scenario.json")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args(argv)
    result = run_analysis(args.validation_cache, args.test_cache, args.frozen_policy, args.scenario, args.output_dir, plots=not args.no_plots)
    print(f"Wrote frozen-policy analysis for {result['sample']['test_images']} held-out images to {args.output_dir}.")


if __name__ == "__main__":
    main()
