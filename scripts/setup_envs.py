"""Setup isolated virtual environments for AutoML frameworks using uv."""

from __future__ import annotations

import argparse
import platform
import subprocess
import sys
from pathlib import Path

import yaml


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def get_python_exe(env_dir: Path) -> Path:
    if platform.system() == "Windows":
        return env_dir / "Scripts" / "python.exe"
    return env_dir / "bin" / "python"


def setup_env(framework_id: str, spec: dict, force: bool = False) -> dict:
    root = repo_root()
    env_dir = root / ".envs" / framework_id
    python_exe = get_python_exe(env_dir)
    if python_exe.exists() and not force:
        print(f"[+] {framework_id}: Environment already exists at {env_dir}")
        return {"status": "exists", "env_dir": str(env_dir), "python_exe": str(python_exe)}

    env_dir.parent.mkdir(parents=True, exist_ok=True)
    preferred_python = spec.get("preferred_python", "3.11")

    print(f"[*] Creating venv for {framework_id} (Python {preferred_python})...")
    # Try uv venv with preferred python
    res = subprocess.run(
        ["uv", "venv", str(env_dir), "--python", preferred_python],
        capture_output=True,
        text=True,
    )
    if res.returncode != 0:
        # Fall back to default uv python
        print(
            f"    Warning: uv venv --python {preferred_python} failed, falling back to current interpreter..."
        )
        res = subprocess.run(
            ["uv", "venv", str(env_dir)],
            capture_output=True,
            text=True,
        )
        if res.returncode != 0:
            return {"status": "failed", "stage": "venv_create", "error": res.stderr}

    # Install pip packages
    packages = spec.get("pip_packages", [])
    print(f"[*] Installing packages for {framework_id}: {packages}...")
    cmd = ["uv", "pip", "install", "--python", str(python_exe), *packages]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"[-] Installation failed for {framework_id}")
        return {
            "status": "failed",
            "stage": "pip_install",
            "error": res.stderr.strip() or res.stdout.strip(),
        }

    print(f"[+] Successfully configured environment for {framework_id}")
    return {"status": "ok", "env_dir": str(env_dir), "python_exe": str(python_exe)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build isolated AutoML environments using uv")
    parser.add_argument("--framework", help="Specific framework ID to setup (e.g., flaml)")
    parser.add_argument("--force", action="store_true", help="Force rebuild existing environment")
    args = parser.parse_args()

    root = repo_root()
    config_file = root / "envs" / "frameworks.yaml"
    with open(config_file, encoding="utf-8") as fh:
        config = yaml.safe_load(fh)

    frameworks = config.get("frameworks", {})
    if args.framework:
        if args.framework not in frameworks:
            print(f"Unknown framework: {args.framework}. Available: {list(frameworks)}")
            sys.exit(1)
        target = {args.framework: frameworks[args.framework]}
    else:
        target = frameworks

    summary = {}
    for fid, spec in target.items():
        summary[fid] = setup_env(fid, spec, force=args.force)

    print("\n=== Setup Summary ===")
    for fid, res in summary.items():
        print(f"{fid:15}: {res.get('status')} {res.get('reason', '')}")


if __name__ == "__main__":
    main()
