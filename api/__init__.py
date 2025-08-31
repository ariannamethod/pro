"""REST API package."""
try:  # pragma: no cover - optional dependency
    from .app import app  # noqa: F401
except Exception:  # pragma: no cover
    app = None
