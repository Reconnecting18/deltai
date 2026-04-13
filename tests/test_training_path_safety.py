"""Unit tests for project.path_safety (CodeQL path-injection hardening helpers)."""

import os
import tempfile

import pytest
from path_safety import (
    require_path_under,
    resolve_under,
    safe_dataset_basename,
    safe_export_filename,
    safe_jsonl_basename,
)


def test_safe_dataset_basename_strips_unsafe_chars() -> None:
    assert safe_dataset_basename("deltai-racing") == "deltai-racing"
    with pytest.raises(ValueError):
        safe_dataset_basename("")


def test_safe_export_filename_allowlist() -> None:
    assert safe_export_filename("deltai-racing", "alpaca") == "deltai-racing_alpaca.json"
    with pytest.raises(ValueError):
        safe_export_filename("deltai-racing", "evilfmt")


def test_safe_jsonl_basename() -> None:
    assert safe_jsonl_basename("deltai-racing.jsonl") == "deltai-racing.jsonl"
    assert safe_jsonl_basename("../evil.jsonl") is None
    assert safe_jsonl_basename("bad/e.jsonl") is None


def test_resolve_under_rejects_escape() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = os.path.realpath(tmp)
        inner = os.path.join(root, "inner")
        os.makedirs(inner, exist_ok=True)
        good = resolve_under(root, "inner", "file.txt")
        assert good == os.path.join(inner, "file.txt")
        with pytest.raises(ValueError):
            resolve_under(root, "..", "etc", "passwd")


def test_require_path_under() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = os.path.realpath(tmp)
        sub = os.path.join(root, "ad")
        os.makedirs(sub, exist_ok=True)
        target = os.path.join(sub, "x.bin")
        with open(target, "wb"):
            pass
        assert require_path_under(target, root) == os.path.realpath(target)
        parent_file = os.path.join(os.path.dirname(root), "path_safety_probe.txt")
        with open(parent_file, "w", encoding="utf-8"):
            pass
        try:
            with pytest.raises(ValueError):
                require_path_under(parent_file, root)
        finally:
            os.remove(parent_file)
