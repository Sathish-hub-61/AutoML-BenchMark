param(
  [Parameter(Position = 0)]
  [ValidateSet("install", "test", "lint", "api", "ui")]
  [string]$Target = "test"
)
$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)
switch ($Target) {
  "install" { python -m pip install -e ".[dev]" }
  "test" { python -m pytest }
  "lint" { python -m ruff check src tests app }
  "api" { python -m uvicorn acbs.api.main:app --reload --port 8000 }
  "ui" { python -m streamlit run app/streamlit_app.py }
}
