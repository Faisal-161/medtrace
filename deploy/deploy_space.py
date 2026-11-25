"""Deploy the Streamlit demo to a Hugging Face Space.

Usage:  HF_TOKEN=... python deploy/deploy_space.py [--space-name medtrace] [--dry-run]

The Space owner is read from the token, so no username is needed. The Space is created on the
first run and updated on every later run.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INCLUDE = ["medtrace", "eval", "data", "ui", "pyproject.toml", "LICENSE"]
SPACE_FILES = ROOT / "deploy" / "space"


def stage(target: Path) -> None:
    """Copy the app plus the Space's own Dockerfile and README (with Space metadata)."""
    ignore = shutil.ignore_patterns("__pycache__", "*.pyc", "results.json")
    for name in INCLUDE:
        src = ROOT / name
        if src.is_dir():
            shutil.copytree(src, target / name, ignore=ignore)
        else:
            shutil.copy2(src, target / name)
    shutil.copy2(SPACE_FILES / "Dockerfile", target / "Dockerfile")
    shutil.copy2(SPACE_FILES / "README.md", target / "README.md")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--space-name", default="medtrace")
    parser.add_argument("--dry-run", action="store_true", help="Stage files and list them; upload nothing")
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        target = Path(tmp)
        stage(target)
        if args.dry_run:
            for p in sorted(target.rglob("*")):
                if p.is_file():
                    print(p.relative_to(target))
            return 0

        token = os.environ.get("HF_TOKEN")
        if not token:
            print("HF_TOKEN is not set. Add it as a GitHub Actions secret named HF_TOKEN.", file=sys.stderr)
            return 1

        from huggingface_hub import HfApi

        api = HfApi(token=token)
        owner = api.whoami()["name"]
        repo_id = f"{owner}/{args.space_name}"
        api.create_repo(repo_id, repo_type="space", space_sdk="docker", exist_ok=True)
        api.upload_folder(
            repo_id=repo_id,
            repo_type="space",
            folder_path=str(target),
            commit_message=f"Deploy {os.environ.get('GITHUB_SHA', 'local')[:7]}",
            delete_patterns=["*"],  # mirror exactly: remove files no longer in the app
        )
        print(f"Deployed: https://huggingface.co/spaces/{repo_id}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
