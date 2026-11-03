"""Offline-Only network guard.

Two independent layers stop data from leaving the machine while Offline-Only
mode is on:

1. **HTTP layer** – every HTTP client the application creates goes through
   :func:`guarded_http_client`, whose request hook rejects URLs whose host is
   not loopback or explicitly allow-listed (e.g. the ``ollama`` container).
2. **Socket layer** – a Python audit hook (``sys.addaudithook``) inspects every
   ``socket.connect`` call made by *any* Python code in the process, including
   third-party libraries (MLflow, Hugging Face, ...). Connections to
   non-allowed IP addresses raise :class:`OfflineNetworkBlocked` before a
   packet is sent.

Limits (documented in docs/PRIVACY_SECURITY.md): the guard cannot see traffic
made by *other* processes (e.g. the Ollama server itself, the browser, or the
OS). Disable Ollama's own cloud features with ``OLLAMA_NO_CLOUD=1``.
"""

from __future__ import annotations

import ipaddress
import logging
import socket
import sys
import threading
import time
from collections import deque
from urllib.parse import urlsplit

import httpx

from app.core.errors import OfflineModeViolation

log = logging.getLogger("app.network_guard")


class OfflineNetworkBlocked(ConnectionRefusedError):
    """Raised by the socket-level guard. Subclasses ConnectionRefusedError so
    libraries treat it like an unreachable host instead of crashing."""


class NetworkGuard:
    RESOLVE_TTL_S = 60.0

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.offline_only: bool = True
        self.allowed_hosts: set[str] = {"localhost", "127.0.0.1", "::1"}
        self._allowed_ips: set[str] = set()
        self._resolved_at: float = 0.0
        self.audit_hook_installed = False
        self.blocked: deque[dict] = deque(maxlen=50)

    # ------------------------------------------------------------------ config
    def configure(self, *, offline_only: bool | None = None, allowed_hosts: list[str] | None = None) -> None:
        with self._lock:
            if offline_only is not None:
                self.offline_only = bool(offline_only)
            if allowed_hosts is not None:
                self.allowed_hosts = {h.strip().lower().strip("[]") for h in allowed_hosts if h.strip()} | {
                    "localhost",
                    "127.0.0.1",
                    "::1",
                }
                self._resolved_at = 0.0

    # ----------------------------------------------------------------- checks
    @staticmethod
    def _is_local_ip(value: str) -> bool:
        try:
            ip = ipaddress.ip_address(value.split("%", 1)[0])
        except ValueError:
            return False
        mapped = getattr(ip, "ipv4_mapped", None)
        if mapped is not None:
            ip = mapped
        return ip.is_loopback or ip.is_unspecified

    def _refresh_allowed_ips(self) -> set[str]:
        now = time.monotonic()
        if now - self._resolved_at < self.RESOLVE_TTL_S:
            return self._allowed_ips
        ips: set[str] = set()
        for host in list(self.allowed_hosts):
            try:
                ipaddress.ip_address(host)
                ips.add(host)
                continue
            except ValueError:
                pass
            try:
                for info in socket.getaddrinfo(host, None):
                    ips.add(str(info[4][0]).split("%", 1)[0])
            except OSError:
                continue
        self._allowed_ips = ips
        self._resolved_at = now
        return ips

    def is_host_allowed(self, host: str) -> bool:
        if not self.offline_only:
            return True
        host = (host or "").strip().lower().strip("[]")
        if not host:
            return False
        if host in self.allowed_hosts or self._is_local_ip(host):
            return True
        return host in self._refresh_allowed_ips()

    def check_url(self, url: str) -> None:
        host = urlsplit(str(url)).hostname or ""
        if not self.is_host_allowed(host):
            self._record(host, "http")
            raise OfflineModeViolation(
                f"Offline-Only mode blocked a network request to '{host}'.",
                hint="Disable Offline-Only mode in Settings if you intend to use a remote or cloud service.",
            )

    def _record(self, host: str, layer: str) -> None:
        self.blocked.append({"host": host, "layer": layer, "at": time.time()})
        log.warning("network request blocked by offline guard", extra={"host": host, "layer": layer})

    # ------------------------------------------------------------- audit hook
    def _audit(self, event: str, args: tuple) -> None:
        if event != "socket.connect" or not self.offline_only:
            return
        try:
            sock, address = args
            if sock.family not in (socket.AF_INET, socket.AF_INET6):
                return  # unix sockets etc. never leave the machine
            host = str(address[0])
        except Exception:  # noqa: BLE001  # pragma: no cover - never break the host program
            return
        if self.is_host_allowed(host):
            return
        self.blocked.append({"host": host, "layer": "socket", "at": time.time()})
        raise OfflineNetworkBlocked(f"Offline-Only mode blocked a socket connection to {host}")

    def install_audit_hook(self) -> None:
        if self.audit_hook_installed:
            return
        sys.addaudithook(self._audit)  # audit hooks cannot be removed; the hook reads live state
        self.audit_hook_installed = True

    def status(self) -> dict:
        return {
            "offline_only": self.offline_only,
            "allowed_hosts": sorted(self.allowed_hosts),
            "socket_guard_installed": self.audit_hook_installed,
            "recent_blocked": list(self.blocked)[-10:],
        }


guard = NetworkGuard()


def guarded_http_client(
    base_url: str,
    *,
    timeout: httpx.Timeout | float = 30.0,
    headers: dict[str, str] | None = None,
    transport: httpx.BaseTransport | None = None,
) -> httpx.Client:
    """Create an httpx client whose every request passes the offline guard."""

    def _hook(request: httpx.Request) -> None:
        guard.check_url(str(request.url))

    return httpx.Client(
        base_url=base_url,
        timeout=timeout,
        headers=headers,
        transport=transport,
        event_hooks={"request": [_hook]},
        trust_env=False,  # never route local traffic through system proxies
    )
