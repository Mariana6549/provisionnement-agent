"""Lance l'évaluation complète et écrit les résultats dans results/.

Exemples :
    python scripts/evaluate.py                          # gabarit, 100 assureurs
    python scripts/evaluate.py --n-companies 775        # tout le jeu de données
    python scripts/evaluate.py --explainer claude       # vrai LLM (ANTHROPIC_API_KEY requise)
"""
import argparse
import json
import os
import sys
from pathlib import Path

# Permet de lancer le script sans installer le paquet : on ajoute src/ au chemin Python.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from provisionnement.evaluation import (  # noqa: E402
    agent_reserve,
    evaluate_faithfulness,
    evaluate_guard_detection,
    evaluate_reserve,
)
from provisionnement.llm import get_explainer  # noqa: E402
from provisionnement.tools import BRANCHES, load_triangle  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--n-companies", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--explainer", choices=["template", "claude"], default="template")
    args = parser.parse_args()

    if args.explainer == "claude" and not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("ANTHROPIC_API_KEY n'est pas définie : impossible d'appeler Claude.")

    out = ROOT / "results"
    out.mkdir(exist_ok=True)

    # 1) écart agent / chainladder
    table, reserve_summary = evaluate_reserve(args.n_companies, args.seed)
    table.to_csv(out / "eval_reserve.csv", index=False)

    # 2) fidélité des chiffres cités (sur les 6 branches)
    results = [agent_reserve(load_triangle(lob), lob, "CumPaidLoss") for lob in sorted(BRANCHES)]
    faithfulness = evaluate_faithfulness(get_explainer(args.explainer), results)
    (out / f"explanations_{args.explainer}.json").write_text(
        json.dumps(faithfulness["explanations"], ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # 3) le garde-fou repère-t-il les faux chiffres injectés ?
    detection = evaluate_guard_detection(results, seed=args.seed)

    summary = {
        "explainer": args.explainer,
        "seed": args.seed,
        "reserve": reserve_summary,
        "faithfulness": {k: v for k, v in faithfulness.items() if k != "explanations"},
        "guard_detection": detection,
    }
    (out / "eval_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    r, f, d = reserve_summary, faithfulness, detection
    print(f"Triangles évalués : {r['n_evaluated']} sur {r['n_requested']} ({r['n_skipped']} non évaluables)")
    print(f"Écart relatif moyen : {r['mean_relative_error']:.3e}")
    print(f"Écart relatif médian : {r['median_relative_error']:.3e}")
    print(f"Écart relatif maximal : {r['max_relative_error']:.3e}")
    print(f"Explications sans chiffre inventé ({args.explainer}) : {f['n_faithful']} sur {f['n_explanations']}")
    print(f"Faux chiffres injectés détectés : {d['n_detected']} sur {d['n_injections']}")
    print(f"Détails écrits dans {out}")


if __name__ == "__main__":
    main()
