from dataclasses import replace

import numpy as np
import pytest

from triage import Scenario, alternatives, evaluate_policy, select_policies, triage_actions, wilson_interval


def test_threshold_boundaries_and_no_review():
    assert triage_actions([0, .2, .5, .8, 1], .2, .8).tolist() == [0, 1, 1, 2, 2]
    assert triage_actions([.49, .5, .51], .5, .5).tolist() == [0, 2, 2]
    with pytest.raises(ValueError):
        triage_actions([.4], .8, .2)


def test_prevalence_weighted_cost_known_example():
    # Half of each class is wrong. Deployment prevalence differs from this balanced test.
    result = evaluate_policy([0, 0, 1, 1], [.2, .8, .2, .8], .5, .5, Scenario())
    assert result['false_positives_per_1000'] == pytest.approx(450)
    assert result['false_negatives_per_1000'] == pytest.approx(50)
    assert result['cost_per_1000'] == pytest.approx(450*25 + 50*500 + 10)
    assert np.sum(result['expected_resolved_confusion_per_1000']) == pytest.approx(1000)


def test_review_all_imperfect_human_has_residual_errors_and_no_model_cost():
    result = evaluate_policy([0, 1], [.3, .7], 0, 1.001, Scenario(), model_used=False)
    assert result['cost_per_1000'] == pytest.approx(5625)
    assert result['daily_reviews'] == pytest.approx(200)
    assert result['daily_review_hours'] == pytest.approx(200*2/60)
    assert result['reviewers_needed'] == 4  # Each only has two hours assigned to triage.
    assert not result['within_capacity']


def test_duplicate_case_sampling_does_not_change_expected_population_cost():
    scenario = Scenario()
    a = evaluate_policy([0, 1], [.2, .8], .1, .9, scenario)
    b = evaluate_policy([0, 0, 0, 1], [.2, .2, .2, .8], .1, .9, scenario)
    assert a['cost_per_1000'] == pytest.approx(b['cost_per_1000'])
    assert a['review_fraction'] == pytest.approx(b['review_fraction'])


def test_selection_respects_capacity_and_binary_can_be_hybrid():
    selection = select_policies([0, 0, 1, 1], [.1, .2, .8, .9], Scenario())
    assert selection['hybrid']['within_capacity']
    assert selection['hybrid']['cost_per_1000'] <= selection['binary']['cost_per_1000'] + 1e-9
    assert selection['selection_split'] == 'val'
    assert len(alternatives([0, 1], [.1, .9], selection, Scenario())) == 4


def test_invalid_cases_and_interval():
    with pytest.raises(ValueError):
        evaluate_policy([1, 1], [.2, .8], .2, .8, Scenario())
    with pytest.raises(ValueError):
        Scenario(false_negative_eur=-1)
    low, high = wilson_interval(10, 10)
    assert 0 < low < .8 and high == pytest.approx(1)
