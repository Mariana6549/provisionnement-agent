import pytest

from provisionnement.evaluation import (
    evaluate_faithfulness,
    evaluate_guard_detection,
    evaluate_reserve,
    relative_error,
)
from provisionnement.llm import TemplateExplainer
from provisionnement.tools import BRANCHES, compute_ldfs, estimate_reserve, load_triangle_data


@pytest.fixture(scope="module")
def results():
    out = []
    for lob in sorted(BRANCHES):
        data = load_triangle_data(lob)
        out.append(estimate_reserve(data, compute_ldfs(data)))
    return out


def test_relative_error_value():
    assert relative_error(105.0, 100.0) == pytest.approx(0.05)


def test_relative_error_rejects_zero_reference():
    with pytest.raises(ValueError):
        relative_error(1.0, 0.0)


def test_reserve_evaluation_is_tiny_on_branches_and_samples():
    table, summary = evaluate_reserve(n_companies=10, seed=0)
    assert summary["n_evaluated"] + summary["n_skipped"] == summary["n_requested"]
    assert summary["n_evaluated"] >= 6  # les 6 branches sont toujours évaluables
    assert summary["max_relative_error"] < 1e-9
    assert {"triangle", "agent_reserve", "reference_reserve", "relative_error"} <= set(table.columns)


def test_template_explainer_is_fully_faithful(results):
    report = evaluate_faithfulness(TemplateExplainer(), results)
    assert report["n_faithful"] == report["n_explanations"] == 6


def test_faithfulness_detects_a_lying_explainer(results):
    liar = lambda r: f"La provision est de {round(r.total_reserve) + 1}."  # noqa: E731
    report = evaluate_faithfulness(liar, results)
    assert report["n_faithful"] == 0
    assert len(report["unfaithful_examples"]) == 6


def test_guard_detects_every_injected_fake_number(results):
    report = evaluate_guard_detection(results, seed=0)
    assert report["detection_rate"] == 1.0
