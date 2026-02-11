"""Minimal FastAPI application for serving recommendations."""

from __future__ import annotations

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import PlainTextResponse

from shrekommender_system.services import RecommenderService
from shrekommender_system.serving.context import build_context

service = RecommenderService()


async def lifespan(_: FastAPI):
    yield
    service.shutdown()


app = FastAPI(title="Shrekommender System", lifespan=lifespan)


def get_service() -> RecommenderService:
    return service


@app.get("/recommend/{user_id}", response_class=PlainTextResponse)
def recommend(user_id: str, recommender: RecommenderService = Depends(get_service)) -> str:
    context = build_context(user_id)
    try:
        result = recommender.recommend(user_id, top_k=20, context=context)
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    movie_ids = [rec["movie_id"] for rec in result.recommendations[:20]]
    return ",".join(map(str, movie_ids))


@app.get("/health")
def health(recommender: RecommenderService = Depends(get_service)):
    return recommender.health()


def main() -> None:
    import uvicorn

    uvicorn.run("shrekommender_system.api:app", host="0.0.0.0", port=8082, reload=True)
