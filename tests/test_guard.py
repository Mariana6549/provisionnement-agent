import pytest

from provisionnement.guard import allowed_numbers, find_unverified_numbers, format_facts
from provisionnement.llm import TemplateExplainer
from provisionnement.tools import compute_ldfs, estimate_reserve, load_triangle_data


@pytest.fixture(scope="module")
def result():
    data = load_triangle_data("wkcomp")
    return estimate_reserve(data, compute_ldfs(data))


def test_template_explanation_has_only_verified_numbers(result):
    assert find_unverified_numbers(TemplateExplainer()(result), result) == []


def test_invented_number_is_detected(result):
    text = f"La provision totale est de {round(result.total_reserve) + 12345}."
    assert find_unverified_numbers(text, result) == [str(round(result.total_reserve) + 12345)]


def test_exact_numbers_from_the_tools_are_accepted(result):
    text = f"La provision totale est de {round(result.total_reserve)}."
    assert find_unverified_numbers(text, result) == []


def test_rounded_number_is_rejected(result):
    """Le garde-fou est strict : 2.2 n'est pas 2.2012, donc il est signalé."""
    assert "2.2" in find_unverified_numbers("Le facteur vaut 2.2.", result)


def test_comma_decimal_is_understood(result):
    factor = f"{result.ldfs[0]:.4f}".replace(".", ",")
    assert find_unverified_numbers(f"Le facteur vaut {factor}.", result) == []


def test_text_without_numbers_is_accepted(result):
    assert find_unverified_numbers("Aucun chiffre ici.", result) == []


def test_facts_contain_every_allowed_number(result):
    facts = format_facts(result)
    for number in allowed_numbers(result):
        assert (f"{number:.0f}" in facts) or (f"{number}" in facts)
