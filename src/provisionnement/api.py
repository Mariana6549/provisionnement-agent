"""API FastAPI : expose l'agent en deux temps, comme le graphe.

POST /runs                 -> lance l'agent, qui s'arrête en attendant la validation
POST /runs/{run_id}/review -> l'actuaire valide ou refuse, l'agent conclut
GET  /runs/{run_id}        -> consulte l'état d'un run
"""
import uuid
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from provisionnement.graph import build_graph, resume_run, start_run
from provisionnement.llm import Explainer, get_explainer
from provisionnement.schemas import ReserveResult

Branch = Literal["comauto", "medmal", "othliab", "ppauto", "prodliab", "wkcomp"]


class RunRequest(BaseModel):
    branch: Branch


class ReviewRequest(BaseModel):
    approved: bool
    comment: str = ""


class RunResponse(BaseModel):
    run_id: str
    status: Literal["awaiting_validation", "validated", "rejected"]
    result: ReserveResult
    explanation: str
    explanation_source: str
    unverified_numbers: list[str]


def create_app(explainer: Explainer | None = None) -> FastAPI:
    """Fabrique l'application. On injecte l'explicateur pour pouvoir tester sans LLM."""
    graph = build_graph(explainer or get_explainer())
    app = FastAPI(title="Agent d'aide au provisionnement")

    def snapshot(run_id: str) -> RunResponse:
        config = {"configurable": {"thread_id": run_id}}
        values = graph.get_state(config).values
        if not values:
            raise HTTPException(status_code=404, detail="Run inconnu")
        return RunResponse(
            run_id=run_id,
            status=values.get("status", "awaiting_validation"),
            result=values["result"],
            explanation=values["explanation"],
            explanation_source=values["explanation_source"],
            unverified_numbers=values.get("unverified_numbers", []),
        )

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.post("/runs", response_model=RunResponse, status_code=201)
    def create_run(request: RunRequest):
        run_id = str(uuid.uuid4())
        start_run(graph, request.branch, run_id)
        return snapshot(run_id)

    @app.get("/runs/{run_id}", response_model=RunResponse)
    def get_run(run_id: str):
        return snapshot(run_id)

    @app.post("/runs/{run_id}/review", response_model=RunResponse)
    def review_run(run_id: str, review: ReviewRequest):
        current = snapshot(run_id)  # lève 404 si inconnu
        if current.status != "awaiting_validation":
            raise HTTPException(status_code=409, detail="Ce run est déjà conclu")
        resume_run(graph, run_id, review.approved, review.comment)
        return snapshot(run_id)

    return app


app = create_app()
