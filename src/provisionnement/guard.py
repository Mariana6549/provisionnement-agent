"""Garde-fou « aucun chiffre inventé ».

Principe : on fabrique un bloc de FAITS à partir du résultat des outils.
Le LLM ne doit citer que ces chiffres, recopiés tels quels. On extrait ensuite
tous les nombres du texte généré et on vérifie que chacun figure dans les faits.

Limite assumée : ce test contrôle les CHIFFRES, pas le raisonnement. Un texte
peut citer de bons chiffres et en tirer une mauvaise conclusion.
"""
import re

from provisionnement.schemas import ReserveResult

# Un nombre : entier ou décimal, avec point ou virgule (ex. 2777813, 2.2012, 2,2012)
NUMBER = re.compile(r"\d+(?:[.,]\d+)?")

LDF_DECIMALS = 4


def _amount(x: float) -> int:
    """Montants affichés en entiers, sans séparateur de milliers."""
    return round(x)


def _ldf(x: float) -> float:
    return round(x, LDF_DECIMALS)


def allowed_numbers(result: ReserveResult) -> set[float]:
    """Tous les nombres que l'explication a le droit de citer."""
    allowed: set[float] = set()
    allowed.update(float(a) for a in result.ages)
    allowed.update(float(_ldf(f)) for f in result.ldfs)
    allowed.update(float(o.origin) for o in result.per_origin)
    for o in result.per_origin:
        allowed.update(float(_amount(v)) for v in (o.latest_paid, o.ultimate, o.reserve))
    allowed.update(
        float(_amount(v))
        for v in (result.total_paid_to_date, result.total_ultimate, result.total_reserve)
    )
    return allowed


def format_facts(result: ReserveResult) -> str:
    """Bloc de faits donné au LLM : la seule source de chiffres autorisée."""
    lines = [
        f"Branche : {result.branch}",
        f"Mesure : {result.measure}",
        "Facteurs de développement (de l'âge en mois a l'âge en mois suivant) :",
    ]
    for age_from, age_to, f in zip(result.ages, result.ages[1:], result.ldfs):
        lines.append(f"  de {age_from} à {age_to} : {_ldf(f)}")
    lines.append("Résultats par année de survenance (montants entiers) :")
    for o in result.per_origin:
        lines.append(
            f"  {o.origin} : payé à ce jour {_amount(o.latest_paid)}, "
            f"coût final estimé {_amount(o.ultimate)}, provision {_amount(o.reserve)}"
        )
    lines.append(f"Total payé à ce jour : {_amount(result.total_paid_to_date)}")
    lines.append(f"Total coût final estimé : {_amount(result.total_ultimate)}")
    lines.append(f"Provision totale estimée : {_amount(result.total_reserve)}")
    return "\n".join(lines)


def find_unverified_numbers(text: str, result: ReserveResult) -> list[str]:
    """Renvoie la liste des nombres du texte qui ne figurent pas dans les faits."""
    allowed = allowed_numbers(result)
    unverified = []
    for token in NUMBER.findall(text):
        value = float(token.replace(",", "."))
        if not any(abs(value - a) <= 1e-9 * max(1.0, abs(a)) for a in allowed):
            unverified.append(token)
    return unverified
