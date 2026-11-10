from langgraph.checkpoint.memory import InMemorySaver

from provisionnement.graph import build_graph, resume_run, start_run
from provisionnement.llm import TemplateExplainer


def make_graph(explainer=None):
    return build_graph(explainer or TemplateExplainer(), InMemorySaver())


def test_agent_pauses_for_human_validation():
    graph = make_graph()
    output = start_run(graph, "wkcomp", "t-pause")
    assert "__interrupt__" in output  # le graphe attend l'actuaire
    state = graph.get_state({"configurable": {"thread_id": "t-pause"}})
    assert state.next == ("human_review",)
    assert "status" not in state.values  # aucune conclusion avant la validation


def test_interrupt_payload_carries_result_and_explanation():
    graph = make_graph()
    output = start_run(graph, "wkcomp", "t-payload")
    payload = output["__interrupt__"][0].value
    assert payload["result"]["branch"] == "wkcomp"
    assert payload["explanation"]
    assert payload["explanation_source"] == "llm"


def test_approval_concludes_as_validated():
    graph = make_graph()
    start_run(graph, "ppauto", "t-ok")
    final = resume_run(graph, "t-ok", approved=True, comment="OK")
    assert final["status"] == "validated"
    assert final["review"]["comment"] == "OK"


def test_rejection_concludes_as_rejected():
    graph = make_graph()
    start_run(graph, "ppauto", "t-ko")
    final = resume_run(graph, "t-ko", approved=False, comment="facteurs douteux")
    assert final["status"] == "rejected"


def test_invented_number_triggers_fallback_to_template():
    """Un LLM qui invente un chiffre est remplacé par le gabarit sûr."""

    def lying_explainer(result):
        return f"La provision est de {round(result.total_reserve) + 999}."

    graph = make_graph(lying_explainer)
    output = start_run(graph, "wkcomp", "t-lie")
    payload = output["__interrupt__"][0].value
    assert payload["explanation_source"] == "fallback_template"
    state = graph.get_state({"configurable": {"thread_id": "t-lie"}})
    assert state.values["unverified_numbers"]  # le chiffre inventé est conservé pour audit


def test_faithful_explainer_is_kept():
    def faithful_explainer(result):
        return f"La provision totale est de {round(result.total_reserve)}."

    graph = make_graph(faithful_explainer)
    output = start_run(graph, "wkcomp", "t-faithful")
    assert output["__interrupt__"][0].value["explanation_source"] == "llm"
