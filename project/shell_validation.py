"""
Validate one-line shell strings before subprocess bash -c or SSH remote exec (CWE-78).

Remote multi-line scripts (bash -s) use length/null checks only — see run_remote_script callers.

Used as the CodeQL barrier for command-line injection on validated returns.
"""

from __future__ import annotations

# Best-effort substring blocklist (same intent as legacy tools.executor guards).
_BLOCKED_SUBSTRINGS = (
    # Filesystem destruction
    "rm -rf /",
    "rm -rf /*",
    "mkfs",
    "dd if=",
    # Privilege escalation
    "sudo ",
    "su -",
    "sudo -i",
    "chmod 777 /",
    "chown root",
    # User/group manipulation
    "useradd",
    "userdel",
    "usermod",
    "groupadd",
    "groupdel",
    "passwd ",
    # System shutdown/reboot
    "shutdown",
    "reboot",
    "halt",
    "poweroff",
    "init 0",
    "init 6",
    # Dangerous network ops
    "wget ",
    "curl ",
    "nc -",
    # Fork bomb and shell escape patterns
    ":(){ :|:& };:",
    "> /dev/sda",
)


def validated_bash_c_command(cmd: str, *, max_len: int = 8192) -> str:
    """
    Return the command unchanged if it passes structural and blocklist checks; else raise ValueError.

    For a **single** logical command passed to ``bash -c`` or as the remote argv tail of ``ssh``.
    Blocks obvious command substitution (CWE-78), newline injection, and the keyword blocklist.
    """
    if not isinstance(cmd, str):
        raise TypeError("command must be a string")
    s = cmd.strip()
    if not s:
        raise ValueError("empty command")
    if len(s) > max_len:
        raise ValueError(f"command too long (max {max_len} chars)")
    if "\x00" in s:
        raise ValueError("invalid character in command")
    if "\n" in s or "\r" in s:
        raise ValueError("embedded newlines not allowed")
    if "$(" in s:
        raise ValueError("command substitution $() is not allowed")
    if "`" in s:
        raise ValueError("backtick command substitution is not allowed")

    lower = s.lower()
    for blocked in _BLOCKED_SUBSTRINGS:
        if blocked in lower:
            raise ValueError(f"blocked command pattern: {blocked.strip()}")

    return s


def is_bash_c_command_allowed(cmd: str, *, max_len: int = 8192) -> bool:
    """Best-effort bool wrapper for callers that cannot raise."""
    try:
        validated_bash_c_command(cmd, max_len=max_len)
        return True
    except (TypeError, ValueError):
        return False


def validated_remote_stdin_script(body: str, *, max_len: int = 64000) -> str:
    """
    Validate body passed to ``ssh … bash -s`` (may contain newlines). Defense-in-depth only.
    """
    if not isinstance(body, str):
        raise TypeError("script must be a string")
    if len(body) > max_len:
        raise ValueError(f"script too long (max {max_len} chars)")
    if "\x00" in body:
        raise ValueError("invalid character in script")
    return body
