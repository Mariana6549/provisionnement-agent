"""Les « explicateurs » : ils transforment un ReserveResult en texte français.

Deux versions interchangeables (même signature : ReserveResult -> str) :
- TemplateExplainer : un gabarit, sans LLM, 100 % déterministe (sert de filet
  de sécurité et de version de référence pour les tests) ;
- ClaudeExplainer : un vrai LLM, qui ne reçoit QUE le bloc de faits.
"""
import os
from typing import Callable

from provisionnement.guard import _amount, _ldf, format_facts
from provisionnement.schemas import ReserveResult

Explainer = Callable[[ReserveResult], str]

SYSTEM_PROMPT = (
    "Tu es un assistant d'aide au provisionnement pour des actuaires. "
    "Explique le résultat en français clair, pour quelqu'un qui n'est pas actuaire, "
    "en 6 à 10 phrases. Règles strictes : "
    "(1) n'utilise QUE les chiffres du bloc de faits, recopiés exactement, "
    "sans arrondir, sans séparateur de milliers ; "
    "(2) ne calcule aucun nouveau chiffre (pas de pourcentage, pas de somme, pas de moyenne) ; "
    "(3) n'écris aucun autre nombre que ceux du bloc de faits ; "
    "(4) rappelle que l'estimation suppose que le rythme de règlement passé se répète "
    "et qu'elle doit être validée par un actuaire."
)


class TemplateExplainer:
    """Explication par gabarit : aucun LLM, aucun chiffre hors des faits."""

    def __call__(self, result: ReserveResult) -> str:
        first, last = result.per_origin[0], result.per_origin[-1]
        biggest = max(result.per_origin, key=lambda o: o.reserve)
        return (
            f"Pour la branche {result.branch} (mesure {result.measure}), la provision a été "
            f"estimée avec la méthode Chain-Ladder. "
            f"Les facteurs de développement montrent comment les paiements cumulés grossissent "
            f"avec le temps : entre {result.ages[0]} et {result.ages[1]} mois, les montants sont "
            f"multipliés par {_ldf(result.ldfs[0])}. Les facteurs suivants se rapprochent de un, "
            f"ce qui signifie que les paiements se stabilisent. "
            f"Pour les sinistres survenus de {first.origin} à {last.origin}, "
            f"{_amount(result.total_paid_to_date)} ont déjà été payés. "
            f"Le coût final estimé est de {_amount(result.total_ultimate)}, ce qui donne une "
            f"provision totale estimée à {_amount(result.total_reserve)}. "
            f"L'année de survenance qui porte la plus forte provision est {biggest.origin}, "
            f"avec {_amount(biggest.reserve)}. "
            f"Cette estimation suppose que le rythme de règlement passé se répétera ; "
            f"elle doit être validée par un actuaire."
        )


class ClaudeExplainer:
    """Explication rédigée par Claude. Nécessite ANTHROPIC_API_KEY."""

    def __init__(self, model: str = "claude-sonnet-5-5"):
        # Import ici : les tests et le gabarit n'ont pas besoin de cette dépendance.
        from langchain_anthropic import ChatAnthropic

        self._llm = ChatAnthropic(model=model, temperature=0, max_tokens=800)

    def __call__(self, result: ReserveResult) -> str:
        reply = self._llm.invoke([("system", SYSTEM_PROMPT), ("human", format_facts(result))])
        content = reply.content
        if isinstance(content, list):  # la réponse peut être découpée en blocs
            content = "".join(b.get("text", "") for b in content if isinstance(b, dict))
        return content


def get_explainer(kind: str | None = None) -> Explainer:
    """kind = 'template' ou 'claude' ; par défaut, la variable EXPLAINER, sinon 'template'."""
    kind = (kind or os.environ.get("EXPLAINER", "template")).lower()
    if kind == "template":
        return TemplateExplainer()
    if kind == "claude":
        return ClaudeExplainer()
    raise ValueError(f"Explicateur inconnu : {kind!r} (choix : 'template', 'claude')")
