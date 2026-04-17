"""Block SSRF by validating HTTP(S) URLs before outbound fetches (CodeQL py/full-ssrf)."""

from __future__ import annotations

import ipaddress
import socket
from urllib.parse import urlparse


def _forbidden_ip(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    if ip.is_loopback or ip.is_private or ip.is_link_local or ip.is_multicast:
        return True
    if ip.is_reserved or ip.is_unspecified:
        return True
    if ip.version == 4 and ip == ipaddress.IPv4Address("169.254.169.254"):
        return True
    return False


def validate_http_url_for_fetch(url: str) -> None:
    """
    Raise ValueError if the URL must not be fetched (private / loopback / metadata, etc.).
    Call before httpx or other HTTP clients; use with request hooks so redirects are checked.
    """
    if not url or not isinstance(url, str):
        raise ValueError("invalid url")
    parsed = urlparse(url.strip())
    if parsed.scheme not in ("http", "https"):
        raise ValueError("only http(s) URLs are allowed")
    host = parsed.hostname
    if not host:
        raise ValueError("missing host")
    port = parsed.port
    if port is None:
        port = 443 if parsed.scheme == "https" else 80

    try:
        infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except OSError as e:
        raise ValueError("cannot resolve host") from e

    for _fam, _type, _proto, _canon, sockaddr in infos:
        addr_str = sockaddr[0]
        try:
            ip = ipaddress.ip_address(addr_str)
        except ValueError:
            continue
        if _forbidden_ip(ip):
            raise ValueError("host resolves to a disallowed address")
