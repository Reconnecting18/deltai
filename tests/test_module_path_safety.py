"""Regression tests for module file path hardening in HTTP handlers."""

import os

import deltai_app
import main


def _assert_module_resolution_safety(mod) -> None:
    path = mod._resolve_module_path("protocols")
    assert path is not None
    assert path.startswith(mod.MODULES_DIR + os.sep)
    assert mod._resolve_module_path("../etc/passwd") is None
    assert mod._resolve_module_path("missing-module") is None


def test_deltai_app_module_resolution_is_bounded() -> None:
    _assert_module_resolution_safety(deltai_app)


def test_main_module_resolution_is_bounded() -> None:
    _assert_module_resolution_safety(main)
