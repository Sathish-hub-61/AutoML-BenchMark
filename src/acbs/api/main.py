from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from acbs.api.routes import datasets, experiments, frameworks, health
from acbs.utils.paths import repo_root

app = FastAPI(title="Auto Classifier Benchmarking System", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(health.router)
app.include_router(datasets.router)
app.include_router(experiments.router)
app.include_router(frameworks.router)

# Plain-HTML frontend (web/), mounted last so API routes always win.
_WEB_DIR = repo_root() / "web"
if _WEB_DIR.is_dir():
    app.mount("/", StaticFiles(directory=str(_WEB_DIR), html=True), name="web")
