"""Continuous deployment of the /predict API to a Hugging Face Docker Space.

    HF_TOKEN=... HF_SPACE=desbature/learner-risk-api python scripts/deploy_hf_space.py

Assembles a minimal build context (Dockerfile, Space card, requirements,
params.yaml, src/ and the registered model) and uploads it. The Space then
rebuilds itself. Run by the deploy-api job in .github/workflows/ci.yml after
every test job has passed on main.
"""
from __future__ import annotations

import os
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_PACKAGES = ["__init__.py", "config.py", "serving", "models", "dashboard", "features", "governance"]


def build_context(dest: Path) -> Path:
    for f in ["Dockerfile", "README.md"]:  # a Space reads both from its root
        shutil.copy(ROOT / "deploy" / "api" / f, dest / f)
    (dest / "deploy" / "api").mkdir(parents=True)
    shutil.copy(ROOT / "deploy" / "api" / "requirements.txt", dest / "deploy" / "api" / "requirements.txt")
    shutil.copy(ROOT / "params.yaml", dest / "params.yaml")
    (dest / "src").mkdir()
    for item in SRC_PACKAGES:
        src = ROOT / "src" / item
        if src.is_dir():
            shutil.copytree(src, dest / "src" / item, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        elif src.exists():
            shutil.copy(src, dest / "src" / item)
    shutil.copytree(ROOT / "models", dest / "models")
    return dest


def main() -> int:
    token, space = os.environ.get("HF_TOKEN"), os.environ.get("HF_SPACE", "desbature/learner-risk-api")
    if not token:
        print("HF_TOKEN not set; skipping deployment")
        return 0
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.create_repo(space, repo_type="space", space_sdk="docker", exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        ctx = build_context(Path(tmp))
        api.upload_folder(folder_path=str(ctx), repo_id=space, repo_type="space",
                          commit_message=f"Deploy {os.environ.get('GITHUB_SHA', 'local')[:7]} from GitHub Actions")
    print(f"deployed to https://huggingface.co/spaces/{space}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
