"""Sanity checks for console script path resolution."""

from delta.cli.repo_root import project_dir, repo_root


def test_repo_root_contains_project() -> None:
    root = repo_root()
    assert (root / "project" / "main.py").is_file()


def test_project_dir_matches_root() -> None:
    assert project_dir() == repo_root() / "project"


def test_repo_root_is_absolute() -> None:
    assert repo_root().is_absolute()
