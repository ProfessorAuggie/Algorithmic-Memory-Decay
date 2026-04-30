"""Minimal WSGI entrypoint for Vercel Python builds.

This repository is primarily a simulation project, so the app simply exposes
basic status endpoints while keeping the simulation code unchanged.
"""

from __future__ import annotations

import json
from typing import Iterable, Tuple


def app(environ, start_response) -> Iterable[bytes]:
    """WSGI application object expected by Vercel.

    Returns a small JSON payload for the root path and a 404 for everything else.
    """
    path = environ.get("PATH_INFO", "/")

    if path == "/":
        payload = {
            "status": "ok",
            "project": "Algorithmic Memory Decay",
            "message": "Simulation repository entrypoint is live.",
        }
        body = json.dumps(payload).encode("utf-8")
        headers: list[Tuple[str, str]] = [
            ("Content-Type", "application/json; charset=utf-8"),
            ("Content-Length", str(len(body))),
        ]
        start_response("200 OK", headers)
        return [body]

    body = b"Not Found"
    headers = [
        ("Content-Type", "text/plain; charset=utf-8"),
        ("Content-Length", str(len(body))),
    ]
    start_response("404 Not Found", headers)
    return [body]
