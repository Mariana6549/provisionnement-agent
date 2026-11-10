"""Le graphe LangGraph : l'enchaînement des étapes de l'agent.

START -> load -> factors -> reserve -> explain -> verify_numbers
      -> human_review (PAUSE : attend l'actuaire) -> finalize -> END

Honnêteté sur le vocabulaire : le parcours est fixe, c'est un workflow avec un
nœud LLM et une validation humaine, pas un agent qui choisit librement ses outils.
"""
from typing import TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from provisionnement.guard import find_unverified_numbers
from provisionnement.llm import Explainer, TemplateExplainer
from provisionnement.schemas import ReserveResult, TriangleData
from provisionnement.tools import compute_ldfs, estimate_reserve, load_triangle_data


class AgentState(TypedDict, total=False):
    """L'état partagé : chaque nœud lit ce dont il a besoin et ajoute son résultat."""

    branch: str
    measure: str
    triangle: dict  # TriangleData sérialisé
    ldfs: list[float]
    result: dict  # ReserveResult sérialisé
    explanation: str
    explanation_source: str  # 'llm' ou 'fallback_template'
    unverified_numbers: list[str]  # nombres du LLM absents des faits
    review: dict  # décision de l'humain
    status: str  # 'validated' ou 'rejected'


def build_graph(explainer: Explainer, checkpointer=None):
    """Construit et compile le graphe. L'explicateur est injecté (LLM ou gabarit)."""
    fallback = TemplateExplainer()

    def load_node(state: AgentState):
        measure = state.get("measure", "CumPaidLoss")
        data = load_triangle_data(state["branch"], measure)
        return {"measure": measure, "triangle": data.model_dump()}

    def factors_node(state: AgentState):
        data = TriangleData(**state["triangle"])
        return {"ldfs": compute_ldfs(data)}

    def reserve_node(state: AgentState):
        data = TriangleData(**state["triangle"])
        result = estimate_reserve(data, state["ldfs"])
        return {"result": result.model_dump()}

    def explain_node(state: AgentState):
        # Seul nœud où un LLM intervient. Il reçoit le résultat des outils, rien d'autre.
        result = ReserveResult(**state["result"])
        return {"explanation": explainer(result), "explanation_source": "llm"}

    def verify_numbers_node(state: AgentState):
        # Garde-fou : tout nombre cité doit venir des outils. Sinon, texte gabarit.
        result = ReserveResult(**state["result"])
        unverified = find_unverified_numbers(state["explanation"], result)
        if unverified:
            return {
                "unverified_numbers": unverified,
                "explanation": fallback(result),
                "explanation_source": "fallback_template",
            }
        return {"unverified_numbers": []}

    def human_review_node(state: AgentState):
        # interrupt() fige le graphe ici et renvoie ce dictionnaire à l'appelant.
        # Au rappel avec Command(resume=...), interrupt() renvoie la décision.
        decision = interrupt(
            {
                "question": "Validez-vous cette provision ?",
                "result": state["result"],
                "explanation": state["explanation"],
                "explanation_source": state["explanation_source"],
            }
        )
        return {"review": decision}

    def finalize_node(state: AgentState):
        approved = bool(state["review"].get("approved"))
        return {"status": "validated" if approved else "rejected"}

    graph = StateGraph(AgentState)
    graph.add_node("load", load_node)
    graph.add_node("factors", factors_node)
    graph.add_node("reserve", reserve_node)
    graph.add_node("explain", explain_node)
    graph.add_node("verify_numbers", verify_numbers_node)
    graph.add_node("human_review", human_review_node)
    graph.add_node("finalize", finalize_node)

    graph.add_edge(START, "load")
    graph.add_edge("load", "factors")
    graph.add_edge("factors", "reserve")
    graph.add_edge("reserve", "explain")
    graph.add_edge("explain", "verify_numbers")
    graph.add_edge("verify_numbers", "human_review")
    graph.add_edge("human_review", "finalize")
    graph.add_edge("finalize", END)

    # Le checkpointer sauvegarde l'état à chaque étape : c'est lui qui permet
    # d'interrompre puis de reprendre. InMemorySaver = en mémoire (perdu au redémarrage).
    return graph.compile(checkpointer=checkpointer or InMemorySaver())


def start_run(graph, branch: str, thread_id: str, measure: str = "CumPaidLoss") -> dict:
    """Lance l'agent : il s'arrête tout seul au nœud human_review."""
    config = {"configurable": {"thread_id": thread_id}}
    return graph.invoke({"branch": branch, "measure": measure}, config)


def resume_run(graph, thread_id: str, approved: bool, comment: str = "") -> dict:
    """Reprend l'agent avec la décision de l'actuaire."""
    config = {"configurable": {"thread_id": thread_id}}
    return graph.invoke(Command(resume={"approved": approved, "comment": comment}), config)
