"""FastAPI charge le modèle une fois ; PostgreSQL reste géré par Next.js."""
from contextlib import asynccontextmanager
import logging
import os
from pathlib import Path
from threading import BoundedSemaphore
from time import perf_counter

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator
from threadpoolctl import threadpool_limits

from app.model import LABELS, ROOT, ModelBundle, has_content, normalize_claim

logger = logging.getLogger("zenassist.ml")


class TagRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    user_claim: str = Field(strict=True, min_length=1, max_length=100_000)

    @field_validator("user_claim")
    @classmethod
    def valid_content(cls, value):
        value = normalize_claim(value)
        if not has_content(value):
            raise ValueError("A complaint containing text is required.")
        return value


class TagResponse(BaseModel):
    tag: str = Field(json_schema_extra={"enum": LABELS})
    model: str
    modelVersion: str
    elapsedMs: float = Field(ge=0)


def create_app(model_dir: Path | None = None):
    directory = model_dir or Path(os.environ.get("MODEL_DIR", ROOT / "models"))

    @asynccontextmanager
    async def lifespan(app):
        with threadpool_limits(limits=2):
            app.state.bundle = ModelBundle.load(directory)
            app.state.capacity = BoundedSemaphore(2)
            yield
            app.state.bundle = None

    api = FastAPI(title="ZenAssist ML", version="0.1.0", lifespan=lifespan)

    @api.exception_handler(RequestValidationError)
    async def invalid_request(_request, _error):
        # FastAPI's default errors echo the input, which can contain personal data.
        return JSONResponse(status_code=422, content={"error": "Provide a non-empty user_claim string, up to 100000 characters, without extra fields.", "code": "INVALID_CLAIM"})

    @api.get("/health")
    def health(request: Request):
        bundle = request.app.state.bundle
        return {"status": "ready", "model": "logistic_regression", "modelVersion": bundle.metadata["modelVersion"], "categories": len(LABELS)}

    @api.post("/tags", response_model=TagResponse)
    def tags(claim: TagRequest, request: Request):
        if not request.app.state.capacity.acquire(blocking=False):
            raise HTTPException(status_code=503, detail="The model is busy. Please retry shortly.")
        try:
            started = perf_counter()
            bundle = request.app.state.bundle
            tag = bundle.predict(claim.user_claim)
            return {"tag": tag, "model": "logistic_regression", "modelVersion": bundle.metadata["modelVersion"], "elapsedMs": round((perf_counter() - started) * 1000, 3)}
        except Exception:
            logger.error("ML prediction failed; no complaint text logged.")
            raise HTTPException(status_code=500, detail="Classification failed. Please retry or assign a tag manually.") from None
        finally:
            request.app.state.capacity.release()

    return api


app = create_app()
