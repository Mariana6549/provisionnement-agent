import chainladder as cl
import pytest

from provisionnement.schemas import TriangleData
from provisionnement.tools import (
    BRANCHES,
    _clrd,
    compute_ldfs,
    estimate_reserve,
    load_triangle,
    load_triangle_data,
    triangle_to_data,
)


def test_load_triangle_shape():
    tri = load_triangle("wkcomp")
    assert tri.shape == (1, 1, 10, 10)


def test_load_triangle_unknown_branch():
    with pytest.raises(ValueError, match="Branche inconnue"):
        load_triangle("inconnue")


def test_triangle_data_has_empty_lower_right_corner():
    data = load_triangle_data("wkcomp")
    assert data.origins == list(range(1988, 1998))
    assert data.ages == list(range(12, 121, 12))
    assert data.values[0][-1] is not None  # 1988 à 120 mois : observé
    assert data.values[-1][-1] is None  # 1997 à 120 mois : futur
    assert data.values[-1][0] is not None  # 1997 à 12 mois : observé


@pytest.mark.parametrize("lob", sorted(BRANCHES))
def test_ldfs_match_chainladder(lob):
    tri = load_triangle(lob)
    reference = cl.Development().fit(tri).ldf_.values.flatten()
    ours = compute_ldfs(load_triangle_data(lob))
    assert len(ours) == 9
    assert ours == pytest.approx(reference)


@pytest.mark.parametrize("lob", sorted(BRANCHES))
def test_reserve_matches_chainladder(lob):
    tri = load_triangle(lob)
    reference = float(cl.Chainladder().fit(tri).ibnr_.sum().sum())
    data = load_triangle_data(lob)
    result = estimate_reserve(data, compute_ldfs(data))
    assert result.total_reserve == pytest.approx(reference, rel=1e-9)


def test_reserve_is_ultimate_minus_paid():
    data = load_triangle_data("ppauto")
    result = estimate_reserve(data, compute_ldfs(data))
    for o in result.per_origin:
        assert o.reserve == pytest.approx(o.ultimate - o.latest_paid)
    assert result.total_reserve == pytest.approx(sum(o.reserve for o in result.per_origin))


def test_oldest_year_has_no_reserve_without_tail():
    """1988 est entièrement observée : aucun facteur à appliquer, provision nulle."""
    data = load_triangle_data("wkcomp")
    result = estimate_reserve(data, compute_ldfs(data))
    assert result.per_origin[0].reserve == pytest.approx(0.0)


def test_hand_computed_example():
    """Triangle fictif calculable à la main (celui du document LaTeX)."""
    data = TriangleData(
        branch="demo",
        measure="CumPaidLoss",
        origins=[2021, 2022, 2023],
        ages=[12, 24, 36],
        values=[[100.0, 150.0, 165.0], [110.0, 168.0, None], [120.0, None, None]],
    )
    ldfs = compute_ldfs(data)
    assert ldfs[0] == pytest.approx((150 + 168) / (100 + 110))
    assert ldfs[1] == pytest.approx(165 / 150)
    result = estimate_reserve(data, ldfs)
    assert result.per_origin[1].ultimate == pytest.approx(168 * 1.1)
    assert result.per_origin[2].ultimate == pytest.approx(120 * ((150 + 168) / 210) * 1.1)


def test_degenerate_triangle_is_rejected():
    data = TriangleData(
        branch="demo",
        measure="CumPaidLoss",
        origins=[2021, 2022],
        ages=[12, 24],
        values=[[0.0, 0.0], [0.0, None]],
    )
    with pytest.raises(ValueError, match="non évaluable"):
        compute_ldfs(data)


def test_triangle_with_hole_is_rejected():
    """Année 2022 sans valeur à 24 mois alors qu'elle devrait en avoir : trou."""
    data = TriangleData(
        branch="demo",
        measure="CumPaidLoss",
        origins=[2021, 2022, 2023],
        ages=[12, 24, 36],
        values=[[1.0, 2.0, 3.0], [1.0, None, None], [1.0, None, None]],
    )
    with pytest.raises(ValueError, match="trou"):
        compute_ldfs(data)


def test_negative_amount_is_rejected():
    data = TriangleData(
        branch="demo",
        measure="CumPaidLoss",
        origins=[2021, 2022, 2023],
        ages=[12, 24, 36],
        values=[[10.0, 20.0, 30.0], [10.0, -5.0, None], [10.0, None, None]],
    )
    with pytest.raises(ValueError, match="négatif"):
        compute_ldfs(data)


@pytest.mark.parametrize(
    "company, reason",
    [
        ("Farmers Home Mut Fire Ins Co", "trou"),  # année 1996 sans valeur à 24 mois
        ("Oswego County Mut Ins Co", "négatif"),  # année 1995 avec montants négatifs
    ],
)
def test_real_problem_triangles_are_rejected(company, reason):
    clrd = _clrd()
    index = clrd.index
    pos = int(((index["GRNAME"] == company) & (index["LOB"] == "othliab")).to_numpy().nonzero()[0][0])
    data = triangle_to_data(clrd.iloc[pos]["CumPaidLoss"], "othliab", "CumPaidLoss")
    with pytest.raises(ValueError, match=reason):
        compute_ldfs(data)


def test_inconsistent_ldf_count_is_rejected():
    data = load_triangle_data("wkcomp")
    with pytest.raises(ValueError, match="incohérent"):
        estimate_reserve(data, [1.5, 1.2])
