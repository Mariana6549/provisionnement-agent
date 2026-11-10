"""Modèles Pydantic : la forme exacte des données échangées entre les étapes.

Pydantic valide les types à la création : si un outil renvoie un montant
absent ou d'un mauvais type, on le voit tout de suite, pas trois étapes plus loin.
"""
from pydantic import BaseModel


class TriangleData(BaseModel):
    """Un triangle sous une forme simple (listes), sérialisable en JSON."""

    branch: str
    measure: str
    origins: list[int]  # années de survenance (lignes)
    ages: list[int]  # âges en mois (colonnes)
    values: list[list[float | None]]  # None = futur, donc inconnu


class OriginReserve(BaseModel):
    """Résultat Chain-Ladder pour UNE année de survenance."""

    origin: int
    latest_paid: float  # dernier montant cumulé observé
    ultimate: float  # coût final estimé
    reserve: float  # provision = ultimate - latest_paid


class ReserveResult(BaseModel):
    """Résultat complet : c'est la seule source de chiffres de l'agent."""

    branch: str
    measure: str
    ages: list[int]
    ldfs: list[float]  # facteurs de développement (une valeur par passage d'âge)
    per_origin: list[OriginReserve]
    total_paid_to_date: float
    total_ultimate: float
    total_reserve: float
