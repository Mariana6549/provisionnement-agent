"""Évaluation : l'agent est-il fidèle à la référence, et n'invente-t-il aucun chiffre ?

Trois mesures :
1. écart relatif entre la provision de l'agent (notre numpy) et celle de chainladder ;
2. fidélité des chiffres cités par un explicateur (part de textes sans nombre inventé) ;
3. détection du garde-fou : sur des textes où l'on injecte volontairement un faux
   chiffre, quelle part de fausses valeurs est repérée ?
"""
import random
import re

import chainladder as cl
import numpy as np
import pandas as pd

from provisionnement.guard import NUMBER, allowed_numbers, find_unverified_numbers
from provisionnement.llm import Explainer, TemplateExplainer
from provisionnement.schemas import ReserveResult
from provisionnement.tools import (
    BRANCHES,
    _clrd,
    compute_ldfs,
    estimate_reserve,
    load_triangle,
    triangle_to_data,
)


def relative_error(agent: float, reference: float) -> float:
    """|agent - référence| / |référence|. Impossible si la référence est nulle."""
    if reference == 0:
        raise ValueError("Référence nulle : écart relatif indéfini")
    return abs(agent - reference) / abs(reference)


def reference_reserve(triangle: cl.Triangle) -> float:
    """Provision de référence : chainladder, configuration par défaut.

    On calcule « coût final - dernier payé » année par année (chainladder nomme
    cette quantité `ibnr_`, mais sur un triangle de PAYÉS c'est la provision restante).
    Si la bibliothèque renvoie NaN pour une année, on refuse d'évaluer : un `.sum()`
    ignorerait cette année en silence et fausserait la comparaison.
    """
    model = cl.Chainladder().fit(triangle)
    ultimate = model.ultimate_.values.flatten()
    latest = triangle.latest_diagonal.values.flatten()
    if np.isnan(ultimate).any() or np.isnan(latest).any():
        raise ValueError("Référence incomplète : chainladder renvoie NaN pour au moins une année")
    return float((ultimate - latest).sum())


def agent_reserve(triangle: cl.Triangle, branch: str, measure: str) -> ReserveResult:
    """Provision calculée par NOS outils (numpy), pas par chainladder."""
    data = triangle_to_data(triangle, branch, measure)
    return estimate_reserve(data, compute_ldfs(data))


def _evaluate_one(name: str, triangle: cl.Triangle, branch: str, measure: str) -> dict:
    agent = agent_reserve(triangle, branch, measure).total_reserve
    reference = reference_reserve(triangle)
    if reference != reference:  # NaN
        raise ValueError("Référence non définie (NaN)")
    return {
        "triangle": name,
        "agent_reserve": agent,
        "reference_reserve": reference,
        "relative_error": relative_error(agent, reference),
    }


def evaluate_reserve(n_companies: int = 100, seed: int = 0, measure: str = "CumPaidLoss"):
    """Compare agent et référence sur les 6 branches + un échantillon d'assureurs.

    Renvoie (tableau détaillé, résumé). Les triangles non évaluables sont comptés
    et leur raison est conservée : on ne les cache pas.
    """
    rows, skipped = [], []

    for lob in sorted(BRANCHES):
        rows.append(_evaluate_one(f"branche:{lob}", load_triangle(lob, measure), lob, measure))

    clrd = _clrd()
    rng = random.Random(seed)
    for pos in rng.sample(range(len(clrd)), n_companies):
        name = f"{clrd.index.iloc[pos]['GRNAME']} | {clrd.index.iloc[pos]['LOB']}"
        try:
            triangle = clrd.iloc[pos][measure]
            rows.append(_evaluate_one(name, triangle, clrd.index.iloc[pos]["LOB"], measure))
        except Exception as error:  # triangle dégénéré : on garde la raison
            skipped.append({"triangle": name, "reason": f"{type(error).__name__}: {error}"})

    table = pd.DataFrame(rows)
    errors = table["relative_error"]
    summary = {
        "n_requested": 6 + n_companies,
        "n_evaluated": len(table),
        "n_skipped": len(skipped),
        "mean_relative_error": float(errors.mean()),
        "median_relative_error": float(errors.median()),
        "max_relative_error": float(errors.max()),
        "skipped": skipped,
    }
    return table, summary


def evaluate_faithfulness(explainer: Explainer, results: list[ReserveResult]) -> dict:
    """Part des explications dont TOUS les nombres figurent dans les faits."""
    faithful, examples, explanations = 0, [], []
    for result in results:
        text = explainer(result)
        unverified = find_unverified_numbers(text, result)
        # on garde chaque texte généré : c'est la trace qu'on pourra relire
        explanations.append({"branch": result.branch, "text": text, "unverified": unverified})
        if unverified:
            examples.append({"branch": result.branch, "unverified": unverified})
        else:
            faithful += 1
    return {
        "n_explanations": len(results),
        "n_faithful": faithful,
        "faithful_rate": faithful / len(results),
        "unfaithful_examples": examples,
        "explanations": explanations,
    }


def _format_like(token: str, value: float) -> str:
    """Écrit `value` avec le même nombre de décimales que `token`."""
    match = re.search(r"[.,](\d+)$", token)
    return f"{value:.{len(match.group(1))}f}" if match else str(round(value))


def evaluate_guard_detection(results: list[ReserveResult], seed: int = 0) -> dict:
    """Injecte un faux chiffre dans un texte correct et vérifie que le garde-fou le voit.

    Limite : un faux chiffre qui égale par hasard un chiffre autorisé n'est pas
    détectable ; on tire donc des valeurs qui n'en font pas partie.
    """
    rng = random.Random(seed)
    template = TemplateExplainer()
    detected = 0
    for result in results:
        text = template(result)
        allowed = allowed_numbers(result)
        match = rng.choice(list(NUMBER.finditer(text)))
        original = float(match.group().replace(",", "."))
        while True:
            fake = _format_like(match.group(), original * rng.uniform(1.05, 1.9))
            if float(fake) not in allowed:
                break
        corrupted = text[: match.start()] + fake + text[match.end() :]
        if find_unverified_numbers(corrupted, result):
            detected += 1
    return {
        "n_injections": len(results),
        "n_detected": detected,
        "detection_rate": detected / len(results),
    }
