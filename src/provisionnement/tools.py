"""Outils déterministes de l'agent : du calcul, jamais de LLM.

Rôles :
- `chainladder` charge les données (clrd) ;
- NOTRE implémentation numpy calcule facteurs et provision ;
- `chainladder` sert ensuite de calcul de RÉFÉRENCE dans les tests et l'évaluation.
Si l'agent utilisait chainladder pour calculer, comparer l'agent à chainladder
donnerait 0 par construction et ne prouverait rien.
"""
from functools import lru_cache

import chainladder as cl
import numpy as np

from provisionnement.schemas import OriginReserve, ReserveResult, TriangleData

BRANCHES = {"comauto", "medmal", "othliab", "ppauto", "prodliab", "wkcomp"}


@lru_cache(maxsize=1)
def _clrd() -> cl.Triangle:
    """Charge clrd une seule fois (le chargement prend plusieurs secondes)."""
    return cl.load_sample("clrd")


def load_triangle(lob: str, measure: str = "CumPaidLoss") -> cl.Triangle:
    """Une branche, une mesure, tous les assureurs additionnés en UN triangle."""
    if lob not in BRANCHES:
        raise ValueError(f"Branche inconnue : {lob!r}. Choix : {sorted(BRANCHES)}")
    clrd = _clrd()
    return clrd[clrd["LOB"] == lob][measure].sum()


def triangle_to_data(triangle: cl.Triangle, branch: str, measure: str) -> TriangleData:
    """Convertit un triangle chainladder en TriangleData (listes Python simples)."""
    frame = triangle.to_frame(origin_as_datetime=False)
    array = frame.to_numpy(dtype=float)
    return TriangleData(
        branch=branch,
        measure=measure,
        # l'index est une période (ex. 1988) : .year donne l'entier
        origins=[int(getattr(year, "year", year)) for year in frame.index],
        ages=[int(age) for age in frame.columns],
        values=[[None if np.isnan(x) else float(x) for x in row] for row in array],
    )


def load_triangle_data(lob: str, measure: str = "CumPaidLoss") -> TriangleData:
    """Outil 1 de l'agent : charger le triangle."""
    return triangle_to_data(load_triangle(lob, measure), lob, measure)


def _to_array(data: TriangleData) -> np.ndarray:
    """None -> NaN pour pouvoir calculer avec numpy."""
    return np.array(data.values, dtype=float)


def validate_triangle(data: TriangleData) -> None:
    """Refuse les triangles qu'on ne sait pas traiter proprement (plutôt que de deviner).

    Un triangle propre est carré, a des montants positifs ou nuls, et chaque année i
    est observée exactement jusqu'à la colonne n-1-i (forme en escalier), sans trou.
    Constat sur clrd : près de la moitié des triangles d'assureurs ne respectent pas
    cette forme (trou dans la dernière diagonale, montants négatifs, colonnes à zéro).
    """
    array = _to_array(data)
    n_origins, n_ages = array.shape
    if n_origins != n_ages:
        raise ValueError("Triangle non évaluable : il doit être carré (autant d'années que d'âges)")
    if (array[~np.isnan(array)] < 0).any():
        raise ValueError("Triangle non évaluable : montant cumulé négatif")
    for i, row in enumerate(array):
        expected_last = n_ages - 1 - i  # dernière colonne qui doit être observée
        has_hole = np.isnan(row[: expected_last + 1]).any()
        has_future_data = (~np.isnan(row[expected_last + 1 :])).any()
        if has_hole or has_future_data:
            raise ValueError(
                f"Triangle non évaluable : année {data.origins[i]} avec trou ou forme inattendue"
            )


def compute_ldfs(data: TriangleData) -> list[float]:
    """Outil 2 : facteurs de développement (moyenne pondérée par les volumes).

    Pour le passage de la colonne j à j+1 :
        f_j = somme(col j+1) / somme(col j), sur les années où j ET j+1 sont observés.
    """
    validate_triangle(data)
    array = _to_array(data)
    factors = []
    for j in range(array.shape[1] - 1):
        both = ~np.isnan(array[:, j]) & ~np.isnan(array[:, j + 1])
        denominator = array[both, j].sum()
        if not both.any() or denominator == 0:
            raise ValueError(f"Triangle non évaluable : facteur impossible à l'âge {data.ages[j]}")
        factors.append(float(array[both, j + 1].sum() / denominator))
    return factors


def estimate_reserve(data: TriangleData, ldfs: list[float]) -> ReserveResult:
    """Outil 3 : provision Chain-Ladder.

    Pour chaque année : coût final = dernier montant observé x produit des
    facteurs restants ; provision = coût final - dernier montant observé.
    Pas de facteur de queue (on suppose que le développement s'arrête à la
    dernière colonne), comme dans la configuration par défaut de chainladder.
    """
    validate_triangle(data)
    array = _to_array(data)
    n_ages = array.shape[1]
    if len(ldfs) != n_ages - 1:
        raise ValueError("Nombre de facteurs incohérent avec le triangle")

    per_origin = []
    for i, row in enumerate(array):
        known = np.where(~np.isnan(row))[0]
        if known.size == 0:
            raise ValueError(f"Année {data.origins[i]} sans aucune donnée")
        last = int(known.max())  # indice de la dernière colonne observée
        latest = float(row[last])
        cdf = float(np.prod(ldfs[last:]))  # produit des facteurs restants (1.0 si aucun)
        ultimate = latest * cdf
        per_origin.append(
            OriginReserve(
                origin=data.origins[i],
                latest_paid=latest,
                ultimate=ultimate,
                reserve=ultimate - latest,
            )
        )

    return ReserveResult(
        branch=data.branch,
        measure=data.measure,
        ages=data.ages,
        ldfs=ldfs,
        per_origin=per_origin,
        total_paid_to_date=sum(o.latest_paid for o in per_origin),
        total_ultimate=sum(o.ultimate for o in per_origin),
        total_reserve=sum(o.reserve for o in per_origin),
    )
