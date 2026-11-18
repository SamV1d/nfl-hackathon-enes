import os
from pathlib import Path


def cargar_env():
    f = Path(__file__).resolve().parent.parent / ".env"
    if f.exists():
        for linea in f.read_text(encoding="utf-8-sig").splitlines():
            linea = linea.strip()
            if linea and not linea.startswith("#") and "=" in linea:
                k, v = linea.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


cargar_env()

import logging  # noqa: E402
import sys  # noqa: E402

from fastapi import FastAPI, HTTPException, Request  # noqa: E402
from fastapi.exceptions import RequestValidationError  # noqa: E402
from fastapi.responses import JSONResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402

from . import nova, reports, schemas  # noqa: E402
from .config import PROJECT_ROOT, data_dir_from_env  # noqa: E402
from .loader import StartupError  # noqa: E402
from .ml.service import AuditUnavailable  # noqa: E402
from .ml.validation import run_audit  # noqa: E402
from .models import InvalidSituation, Situation  # noqa: E402
from .recommender import PredictionUnavailable, recommend  # noqa: E402
from .state import AppState, build_state  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("backend.main")

DEFAULT_STATIC_DIR = PROJECT_ROOT / "dashboard" / "dist"
MODEL_UNAVAILABLE = "model unavailable"
PREDICTION_UNAVAILABLE = "prediction unavailable"
REPORT_FAILED = "report could not be generated"


def _error(status: int, detail: str, **extra) -> JSONResponse:
    return JSONResponse(status_code=status, content={"detail": detail, **extra})


def _mount_static(app: FastAPI, static_dir: Path) -> None:
    """Último registro, para que las rutas /api/ y /health tengan prioridad."""
    if static_dir.is_dir():
        app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")


def _new_app() -> FastAPI:
    # La documentación automática vive bajo /api/ (no se permiten rutas fuera de /api/ y /health).
    app = FastAPI(title="Coach Pass Rush Assistant", docs_url="/api/docs",
                  openapi_url="/api/openapi.json", redoc_url=None,
                  swagger_ui_oauth2_redirect_url="/api/docs/oauth2-redirect")

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.exception_handler(RequestValidationError)
    async def validation_handler(request: Request, exc: RequestValidationError):
        fields = sorted({next((str(p) for p in e["loc"] if isinstance(p, str)
                               and p not in ("body", "query", "path")), "body")
                         for e in exc.errors()})
        return _error(422, "invalid input", fields=fields)

    return app


def _fallback_app(error: StartupError, static_dir: Path) -> FastAPI:
    """Arranque fallido: toda ruta /api/ responde 503 con los archivos faltantes."""
    app = _new_app()
    app.state.startup_error = error
    body = {"missing_files": error.missing}
    if error.file:
        body.update(file=error.file, category=error.category)

    @app.api_route("/api/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE"],
                   include_in_schema=False)
    def unavailable(path: str):
        return _error(503, str(error), **body)

    _mount_static(app, static_dir)
    return app


def create_app(data_dir: Path | None = None, static_dir: Path | None = None) -> FastAPI:
    static_dir = static_dir or DEFAULT_STATIC_DIR
    data_dir = data_dir or data_dir_from_env()
    try:
        state = build_state(data_dir)
    except StartupError as exc:
        log.error("Arranque fallido: %s", exc)
        return _fallback_app(exc, static_dir)

    app = _new_app()
    app.state.startup_error = None
    app.state.app_state = state

    def models_ready() -> AppState:
        if state.pressure_model is None or state.yards_model is None:
            raise HTTPException(503, MODEL_UNAVAILABLE)
        return state

    @app.get("/api/vocab", response_model=schemas.VocabOut)
    def vocab():
        models_ready()
        return {"offense_formations": state.vocab["offenseFormation"],
                "expected_coverages": state.vocab["expectedCoverage"]}

    @app.post("/api/recommend", response_model=schemas.RecommendationOut)
    def post_recommend(req: schemas.RecommendRequest):
        models_ready()
        s = Situation(req.down, req.yards_to_go, req.offense_formation, req.expected_coverage)
        try:
            rec = recommend(state, s)
        except InvalidSituation as exc:
            return _error(422, "invalid input", fields=exc.fields)
        except PredictionUnavailable:
            return _error(503, PREDICTION_UNAVAILABLE)
        state.store(rec)
        return rec.as_dict()

    @app.get("/api/metrics", response_model=schemas.MetricsOut)
    def metrics():
        models_ready()
        return {"pressure_model": state.pressure_model.metrics(),
                "yards_model": state.yards_model.metrics()}

    @app.get("/api/model-audit")
    def model_audit():
        """Comparación de modelos con validación temporal por semana (ver docs/ML.md)."""
        if state.plays is None:
            raise HTTPException(503, "model audit unavailable")
        try:
            return state.audit.get(lambda: run_audit(state.dataset, state.plays, state.games))
        except AuditUnavailable:
            raise HTTPException(503, "model audit unavailable") from None

    @app.get("/api/games", response_model=list[schemas.GameOut])
    def games():
        return reports.list_games(state.games)

    @app.get("/api/games/{game_id}/plays", response_model=schemas.ReplayOut)
    def game_plays(game_id: int):
        if game_id not in set(state.games["gameId"]):
            raise HTTPException(404, "game not found")
        return reports.replay_plays(state, game_id)

    @app.get("/api/teams", response_model=schemas.TeamsOut)
    def teams():
        return {"teams": state.teams}

    @app.get("/api/teams/{team}/report", response_model=schemas.TeamReportOut)
    def team_report(team: str):
        try:
            return state.team_stats.get(team) or reports.empty_report(team)
        except Exception:  # noqa: BLE001
            log.exception("fallo al generar el reporte")
            raise HTTPException(500, REPORT_FAILED) from None

    @app.get("/api/findings", response_model=schemas.FindingsOut)
    def findings():
        return state.findings

    @app.post("/api/recommendations/{rec_id}/explain", response_model=schemas.ExplainOut)
    def explain(rec_id: str):
        rec = state.lookup(rec_id)
        if rec is None:
            raise HTTPException(404, "recommendation not found")  # no se invoca Bedrock
        result = nova.explain(rec.as_dict())
        if not result.available:
            raise HTTPException(503, result.message)
        return {"explanation": result.text}

    _mount_static(app, static_dir)
    return app


app = create_app()


if __name__ == "__main__":
    # `python -m backend.main`: sale con código distinto de cero si el arranque falló.
    err = app.state.startup_error
    if err is not None:
        print(f"ERROR de arranque: {err}", file=sys.stderr)
        sys.exit(1)
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8000")))
