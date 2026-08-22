"""Build a citable, fixed-state archive of the analysis (N12 / Phase 5b).

Produces a single zip of everything needed to reproduce the study, plus a
manifest listing every file with its SHA-256, so an archived copy can be
verified against this repository at any later date.

The archive is refused if the working tree is dirty: an archive that does
not correspond to a commit cannot be cited, because there would be no way
to say what it contains.

Usage
-----
    python scripts/make_archive.py [--out DIR] [--allow-dirty]
"""

from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
import zipfile
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ckd.config import project_root  # noqa: E402

#: Directories and files that belong in a citable archive.
INCLUDE = (
    "src", "scripts", "tests", "config", "data/raw", "data/processed",
    "data/literature", "reports", "docs", "notebooks",
    "README.md", "PLAN.md", "WORKLOG.md", "REVIEWER_RESPONSE.md",
    "requirements.txt", "requirements.lock.txt", "pyproject.toml",
    ".gitattributes", ".gitignore",
)

EXCLUDE_PARTS = ("__pycache__", ".pytest_cache", ".ipynb_checkpoints")


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=project_root(), capture_output=True, text=True, check=False
    ).stdout.strip()


def collect(root: Path) -> list[Path]:
    files: list[Path] = []
    for entry in INCLUDE:
        path = root / entry
        if path.is_file():
            files.append(path)
        elif path.is_dir():
            files.extend(p for p in sorted(path.rglob("*")) if p.is_file())
    return [
        f for f in files
        if not any(part in EXCLUDE_PARTS for part in f.parts)
    ]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="dist")
    parser.add_argument("--allow-dirty", action="store_true",
                        help="archive anyway (the manifest will say so)")
    args = parser.parse_args()

    root = project_root()
    dirty = bool(git("status", "--porcelain"))
    if dirty and not args.allow_dirty:
        print("Working tree is dirty. An archive that does not correspond to a "
              "commit cannot be cited - commit first, or pass --allow-dirty.")
        return 1

    commit = git("rev-parse", "HEAD") or "unknown"
    describe = git("describe", "--tags", "--always") or "unknown"
    out_dir = root / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"ckd-benchmark-reanalysis-{date.today().isoformat()}-{describe}"
    zip_path = out_dir / f"{stem}.zip"

    files = collect(root)
    manifest_lines = [
        f"# Archive manifest: {stem}",
        f"# commit: {commit}",
        f"# describe: {describe}",
        f"# clean working tree: {not dirty}",
        f"# files: {len(files)}",
        "#",
        "# sha256  path",
    ]
    for path in files:
        manifest_lines.append(f"{sha256(path)}  {path.relative_to(root).as_posix()}")
    manifest = "\n".join(manifest_lines) + "\n"

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, arcname=f"{stem}/{path.relative_to(root).as_posix()}")
        archive.writestr(f"{stem}/MANIFEST.sha256", manifest)

    (out_dir / f"{stem}.MANIFEST.sha256").write_text(manifest, encoding="utf-8")
    size_mb = zip_path.stat().st_size / (1 << 20)
    print(f"wrote {zip_path} ({len(files)} files, {size_mb:.1f} MB)")
    print(f"wrote {out_dir / f'{stem}.MANIFEST.sha256'}")
    print(f"commit {commit}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
