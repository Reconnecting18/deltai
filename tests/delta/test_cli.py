"""Smoke tests for delta.interfaces.cli (no running daemon required)."""

from unittest.mock import MagicMock, patch

import pytest

from delta import __version__
from delta.interfaces import cli


def test_run_version(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.run(["version"]) == 0
    assert capsys.readouterr().out.strip() == __version__


def test_run_paths(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.run(["paths"]) == 0
    out = capsys.readouterr().out
    assert "data_dir=" in out
    assert "daemon_socket_path=" in out
    assert "ipc_socket_path=" in out
    assert "ollama_url=" in out


def test_help_exits_zero() -> None:
    with pytest.raises(SystemExit) as exc:
        cli.run(["--help"])
    assert exc.value.code == 0


def test_subcommand_help() -> None:
    with pytest.raises(SystemExit) as exc:
        cli.run(["execute", "--help"])
    assert exc.value.code == 0


def test_execute_requires_query_when_tty(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    assert cli.run(["execute"]) == 2


def test_ipc_requires_query_when_tty(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    assert cli.run(["ipc"]) == 2


def test_daemon_http_client_uses_uds_transport() -> None:
    with patch.object(cli.httpx, "HTTPTransport", wraps=cli.httpx.HTTPTransport) as t:
        with patch.object(cli.httpx, "Client", MagicMock()):
            cli.daemon_http_client("/fake/daemon.sock")
    t.assert_called_once_with(uds="/fake/daemon.sock")
