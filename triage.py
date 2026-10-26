"""Image-level triage and costs. No training. Defaults are unapproved toy assumptions."""
from __future__ import annotations
from dataclasses import asdict, dataclass
import math

import numpy as np


@dataclass(frozen=True)
class Scenario:
    prevalence: float = .10
    false_positive_eur: float = 25.
    false_negative_eur: float = 500.
    review_minutes: float = 2.
    reviewer_hourly_eur: float = 60.
    reviewer_sensitivity: float = .95
    reviewer_specificity: float = .95
    daily_images: int = 200
    reviewers_available: int = 1
    review_hours_per_person: float = 2.
    inference_eur_per_image: float = .01

    def __post_init__(self):
        if not 0 < self.prevalence < 1:
            raise ValueError("Prevalence must be strictly between zero and one.")
        for name in ("reviewer_sensitivity", "reviewer_specificity"):
            if not 0 <= getattr(self, name) <= 1:
                raise ValueError(f"{name} must be between zero and one.")
        for name in ("review_minutes", "reviewer_hourly_eur", "daily_images", "reviewers_available", "review_hours_per_person"):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive.")
        for name in ("false_positive_eur", "false_negative_eur", "inference_eur_per_image"):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f"{name} must be finite and nonnegative.")

    @property
    def review_eur(self):
        return self.review_minutes / 60 * self.reviewer_hourly_eur

    @property
    def daily_capacity(self):
        return self.reviewers_available * self.review_hours_per_person * 60 / self.review_minutes


def validate_cases(labels, scores):
    y, s = np.asarray(labels), np.asarray(scores, dtype=float)
    if y.ndim != 1 or s.shape != y.shape or not len(y):
        raise ValueError("Labels and scores must be nonempty, equally sized vectors.")
    if not np.isin(y, [0, 1]).all() or not np.isfinite(s).all() or not ((s >= 0) & (s <= 1)).all():
        raise ValueError("Binary labels and finite scores in [0,1] are required.")
    if len(np.unique(y)) != 2:
        raise ValueError("Both positive and negative images are required for prevalence weighting.")
    return y.astype(int), s


def triage_actions(scores, lower, upper):
    """0=routine, 1=human priority review, 2=accelerated clinical review."""
    if not 0 <= lower <= upper <= 1.001:
        raise ValueError("Require 0 <= lower <= upper <= 1.001.")
    s = np.asarray(scores)
    return np.where(s >= upper, 2, np.where(s < lower, 0, 1))


def evaluate_policy(labels, scores, lower, upper, scenario: Scenario, *, model_used=True):
    y, s = validate_cases(labels, scores)
    actions = triage_actions(s, lower, upper)
    observed = np.zeros((2, 3), dtype=int)
    np.add.at(observed, (y, actions), 1)
    conditional = observed / np.bincount(y, minlength=2)[:, None]
    population = conditional * np.array([1 - scenario.prevalence, scenario.prevalence])[:, None] * 1000
    tn = population[0, 0] + population[0, 1] * scenario.reviewer_specificity
    fp = population[0, 2] + population[0, 1] * (1 - scenario.reviewer_specificity)
    fn = population[1, 0] + population[1, 1] * (1 - scenario.reviewer_sensitivity)
    tp = population[1, 2] + population[1, 1] * scenario.reviewer_sensitivity
    review_per_1000 = float(population[:, 1].sum())
    error_eur = fp * scenario.false_positive_eur + fn * scenario.false_negative_eur
    review_eur = review_per_1000 * scenario.review_eur
    inference_eur = 1000 * scenario.inference_eur_per_image if model_used else 0.
    daily_review = review_per_1000 / 1000 * scenario.daily_images
    daily_hours = daily_review * scenario.review_minutes / 60
    return {"lower": float(lower), "upper": float(upper), "observed_actions": observed.tolist(),
            "expected_actions_per_1000": population.tolist(),
            "expected_resolved_confusion_per_1000": [[float(tn), float(fp)], [float(fn), float(tp)]],
            "cost_per_1000": float(error_eur + review_eur + inference_eur),
            "error_cost_per_1000": float(error_eur), "review_cost_per_1000": float(review_eur),
            "inference_cost_per_1000": float(inference_eur),
            "false_negatives_per_1000": float(fn), "false_positives_per_1000": float(fp),
            "review_fraction": review_per_1000 / 1000, "daily_reviews": daily_review,
            "daily_review_hours": daily_hours,
            "reviewers_needed": math.ceil(max(0., daily_hours / scenario.review_hours_per_person - 1e-12)),
            "within_capacity": daily_review <= scenario.daily_capacity + 1e-9,
            "sensitivity": float(tp / (tp + fn)), "specificity": float(tn / (tn + fp))}


def threshold_grid():
    # The final sentinel permits a true predict-nothing operating point.
    return np.r_[np.linspace(0, 1, 101), 1.001]


def select_policies(labels, scores, scenario: Scenario):
    """Use validation inputs only. Never choose thresholds from held-out test data."""
    grid = threshold_grid()
    binary = [evaluate_policy(labels, scores, t, t, scenario) for t in grid]
    order = lambda r: (r["cost_per_1000"], r["false_negatives_per_1000"], r["review_fraction"], r["lower"], r["upper"])
    best_binary = min(binary, key=order)
    feasible = []
    for i, lower in enumerate(grid):
        for upper in grid[i:]:
            result = evaluate_policy(labels, scores, lower, upper, scenario)
            if result["within_capacity"]:
                feasible.append(result)
    return {"binary": best_binary, "hybrid": min(feasible, key=order), "binary_sweep": binary,
            "scenario": asdict(scenario), "selection_split": "val", "grid_step": .01}


def alternatives(labels, scores, selected, scenario):
    points = [("Routine queue for all (no added triage)", 1.001, 1.001, False),
              ("Human priority review for all", 0., 1.001, False),
              ("Model only", selected["binary"]["lower"], selected["binary"]["upper"], True),
              ("Model + human review", selected["hybrid"]["lower"], selected["hybrid"]["upper"], True)]
    return [{"policy": name, **evaluate_policy(labels, scores, low, high, scenario, model_used=model_used)}
            for name, low, high, model_used in points]


def wilson_interval(successes, total, z=1.96):
    if not total:
        return [None, None]
    p = successes / total
    denominator = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    radius = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return [max(0., centre - radius), min(1., centre + radius)]
