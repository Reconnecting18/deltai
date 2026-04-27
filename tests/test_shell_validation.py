"""Tests for project.shell_validation (CWE-78 bash -c validation)."""

from __future__ import annotations

import pytest
from shell_validation import (
    is_bash_c_command_allowed,
    validated_bash_c_command,
    validated_remote_stdin_script,
)


def test_validated_bash_c_allows_pipe_and_blocks_substitution():
    assert validated_bash_c_command("ps aux | grep python").startswith("ps aux")
    with pytest.raises(ValueError, match="substitution"):
        validated_bash_c_command("echo $(whoami)")
    with pytest.raises(ValueError, match="backtick"):
        validated_bash_c_command("echo `id`")


def test_is_bash_c_command_allowed_matches_validation():
    assert is_bash_c_command_allowed("ls")
    assert not is_bash_c_command_allowed("sudo reboot")


def test_validated_remote_stdin_allows_multiline():
    body = "#!/usr/bin/env bash\necho hello\n"
    assert validated_remote_stdin_script(body).startswith("#!")
