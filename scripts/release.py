#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

_PYPROJECT_PATH = "pyproject.toml"
_DIST_DIR = "dist"

_TESTPYPI_PUBLISH_URL = "https://test.pypi.org/legacy/"
_TESTPYPI_CHECK_URL = "https://test.pypi.org/simple"
_PYPI_CHECK_URL = "https://pypi.org/simple"


class ReleaseError(RuntimeError):
    pass


@dataclass(frozen=True)
class ReleaseConfig:
    repository: str
    skip_tests: bool
    skip_build: bool
    skip_publish: bool
    require_tag: bool
    yes: bool
    dry_run: bool


def _run(cmd: list[str], *, cwd: Path, dry_run: bool) -> None:
    logging.info("$ %s", " ".join(cmd))
    if dry_run:
        return
    subprocess.run(cmd, cwd=str(cwd), check=True)


def _git_output(args: list[str], *, cwd: Path) -> str:
    out = subprocess.check_output(["git", *args], cwd=str(cwd))
    return out.decode("utf-8", errors="replace").strip()


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_pyproject_text(root: Path) -> str:
    path = root / _PYPROJECT_PATH
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError as e:
        raise ReleaseError(f"Missing {_PYPROJECT_PATH} at {path}") from e


def _read_project_version(pyproject_text: str) -> str:
    try:
        import tomllib  # pyright: ignore[reportMissingImports]
    except Exception as e:
        raise ReleaseError("Python 3.11+ required (tomllib unavailable)") from e

    data = tomllib.loads(pyproject_text)
    project = data.get("project")
    if not isinstance(project, dict):
        raise ReleaseError("Missing [project] table in pyproject.toml")
    version = project.get("version")
    if not isinstance(version, str) or not version:
        raise ReleaseError("Missing/invalid [project].version in pyproject.toml")
    return version


def _require_clean_git(root: Path) -> None:
    status = _git_output(["status", "--porcelain=v1"], cwd=root)
    if status:
        raise ReleaseError("Working tree is not clean. Commit or stash changes before releasing.")


def _require_head_tag(root: Path, expected_tag: str) -> None:
    try:
        tag = _git_output(["describe", "--tags", "--exact-match"], cwd=root)
    except subprocess.CalledProcessError as e:
        raise ReleaseError(f"Refusing to publish: HEAD is not exactly at tag {expected_tag}.") from e
    if tag != expected_tag:
        raise ReleaseError(f"Refusing to publish: expected tag {expected_tag}, got {tag}.")


def _clean_dist(root: Path) -> None:
    dist = root / _DIST_DIR
    dist.mkdir(exist_ok=True)
    for p in dist.iterdir():
        if p.is_file() and (p.name.endswith(".whl") or p.name.endswith(".tar.gz")):
            p.unlink()


def _require_uv_installed(root: Path) -> None:
    try:
        subprocess.run(["uv", "--version"], cwd=str(root), check=True, capture_output=True)
    except FileNotFoundError as e:
        raise ReleaseError("uv not found in PATH. Install uv first.") from e
    except subprocess.CalledProcessError as e:
        raise ReleaseError("uv is installed but failed to run `uv --version`.") from e


def _require_publish_token() -> None:
    if os.environ.get("UV_PUBLISH_TOKEN"):
        return
    raise ReleaseError("Missing UV_PUBLISH_TOKEN in environment.")


def _prompt_confirm(message: str, *, yes: bool) -> None:
    if yes:
        return
    reply = input(f"{message} [y/N] ").strip().lower()
    if reply not in {"y", "yes"}:
        raise ReleaseError("Aborted.")


def release(config: ReleaseConfig) -> None:
    root = _repo_root()
    _require_uv_installed(root)
    _require_clean_git(root)
    pyproject_text = _load_pyproject_text(root)
    version = _read_project_version(pyproject_text)
    tag = f"v{version}"

    if config.require_tag:
        _require_head_tag(root, tag)

    if not config.skip_tests:
        _run(
            ["uv", "run", "--extra", "dev", "--prerelease=allow", "-m", "pytest", "-m", "unit"],
            cwd=root,
            dry_run=config.dry_run,
        )

    if not config.skip_build:
        _clean_dist(root)
        _run(["uv", "build", "--out-dir", _DIST_DIR], cwd=root, dry_run=config.dry_run)

    if not config.skip_publish:
        _require_publish_token()
        if config.repository == "testpypi":
            _prompt_confirm(f"Publish {tag} to TestPyPI?", yes=config.yes)
            _run(
                [
                    "uv",
                    "publish",
                    "--publish-url",
                    _TESTPYPI_PUBLISH_URL,
                    "--check-url",
                    _TESTPYPI_CHECK_URL,
                    f"{_DIST_DIR}/*",
                ],
                cwd=root,
                dry_run=config.dry_run,
            )
        else:
            _prompt_confirm(f"Publish {tag} to PyPI?", yes=config.yes)
            _run(
                [
                    "uv",
                    "publish",
                    "--check-url",
                    _PYPI_CHECK_URL,
                    f"{_DIST_DIR}/*",
                ],
                cwd=root,
                dry_run=config.dry_run,
            )


def _parse_args(argv: list[str]) -> ReleaseConfig:
    parser = argparse.ArgumentParser(
        description="Release helper for this repo (tests, build, publish).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--repository",
        choices=["pypi", "testpypi"],
        default="pypi",
        help="Package index to publish to.",
    )
    parser.add_argument("--skip-tests", action="store_true", help="Skip pytest unit tests.")
    parser.add_argument("--skip-build", action="store_true", help="Skip building dist artifacts.")
    parser.add_argument("--skip-publish", action="store_true", help="Skip publishing to an index.")
    parser.add_argument(
        "--require-tag",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require that HEAD is exactly at tag v<version> (from pyproject.toml).",
    )
    parser.add_argument("-y", "--yes", action="store_true", help="Assume yes for prompts.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing.")
    args = parser.parse_args(argv)

    return ReleaseConfig(
        repository=args.repository,
        skip_tests=args.skip_tests,
        skip_build=args.skip_build,
        skip_publish=args.skip_publish,
        require_tag=args.require_tag,
        yes=args.yes,
        dry_run=args.dry_run,
    )


def main(argv: list[str]) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        release(_parse_args(argv))
    except ReleaseError as e:
        logging.error("release: %s", e)
        return 2
    except subprocess.CalledProcessError as e:
        logging.error("release: command failed (%s)", e)
        return e.returncode or 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
