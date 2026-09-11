from __future__ import annotations

from urllib.parse import urlparse

from google.auth.transport.requests import Request
from google.oauth2 import id_token


def audience_from_service_url(url: str) -> str:
    parsed = urlparse(url)
    if not parsed.scheme or not parsed.netloc:
        raise ValueError("RAG service URL must be absolute")
    return f"{parsed.scheme}://{parsed.netloc}"


def fetch_google_id_token(audience: str) -> str:
    if not audience.strip():
        raise ValueError("id token audience must be non-empty")
    return id_token.fetch_id_token(Request(), audience)
