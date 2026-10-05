from __future__ import annotations

import threading
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from sqlalchemy.orm import Session

from acbs.api.deps import get_session
from acbs.api.schemas import DatasetUploadResponse
from acbs.data.loader import load_dataset, load_local, load_registry
from acbs.data.profiler import profile_dataset
from acbs.exceptions import DatasetError
from acbs.storage.repository import Repository
from acbs.utils.paths import repo_root, settings
from acbs.utils.timeout import file_hash

router = APIRouter(prefix="/datasets", tags=["datasets"])
_UPLOAD_LOCK = threading.Lock()
UPLOAD_DIR = repo_root() / "data" / "raw"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


@router.get("/registry")
def registry():
    return load_registry()


@router.get("")
def list_datasets(session: Session = Depends(get_session)):
    repo = Repository(session)
    return [
        {"id": d.id, "name": d.name, "source": d.source, "n_rows": d.n_rows, "n_features": d.n_features}
        for d in repo.list_datasets()
    ]


@router.get("/{dataset_id}")
def get_dataset(dataset_id: int, session: Session = Depends(get_session)):
    row = Repository(session).get_dataset(dataset_id)
    if not row:
        raise HTTPException(404, "dataset not found")
    return {
        "id": row.id,
        "name": row.name,
        "source": row.source,
        "profile": row.profile_json,
    }


@router.post("", response_model=DatasetUploadResponse)
async def upload_dataset(
    file: UploadFile = File(...),
    target: str | None = None,
    session: Session = Depends(get_session),
):
    raw = await file.read()
    max_bytes = int(settings.max_upload_mb) * 1024 * 1024
    if len(raw) > max_bytes:
        raise HTTPException(413, "file too large")
    name = Path(file.filename or "upload.csv").name.replace("..", "")
    dest = UPLOAD_DIR / name
    with _UPLOAD_LOCK:
        dest.write_bytes(raw)
    digest = file_hash(raw)
    repo = Repository(session)
    existing = repo.get_dataset_by_hash(digest)
    if existing:
        return DatasetUploadResponse(
            id=existing.id,
            name=existing.name,
            profile=existing.profile_json or {},
            warnings=(existing.profile_json or {}).get("warnings", []),
        )
    if target is None:
        raise HTTPException(400, "target query/form field is required for uploads")
    try:
        loaded = load_local(dest, target)
    except DatasetError as exc:
        raise HTTPException(400, str(exc)) from exc
    profile = profile_dataset(loaded.X, loaded.y)
    row = repo.create_dataset(
        name=name,
        source="upload",
        n_rows=profile.n_rows,
        n_features=profile.n_features,
        profile_json=profile.model_dump(),
        file_hash=digest,
    )
    return DatasetUploadResponse(id=row.id, name=row.name, profile=profile.model_dump(), warnings=profile.warnings)


@router.post("/from-registry/{name}")
def from_registry(name: str, session: Session = Depends(get_session)):
    try:
        loaded = load_dataset(name)
    except DatasetError as exc:
        raise HTTPException(400, str(exc)) from exc
    profile = profile_dataset(loaded.X, loaded.y)
    repo = Repository(session)
    row = repo.create_dataset(
        name=name,
        source=loaded.meta.get("source", "registry"),
        n_rows=profile.n_rows,
        n_features=profile.n_features,
        profile_json={**profile.model_dump(), "meta": loaded.meta},
        file_hash=None,
    )
    return {"id" : row.id, "name": name, "profile": profile.model_dump(), "warnings": profile.warnings, "target": loaded.meta.get("target_column")}
