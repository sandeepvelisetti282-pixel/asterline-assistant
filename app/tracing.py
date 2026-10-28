# app/tracing.py
"""LangSmith tracing configuration.

Tracing is optional and requires:
  LANGSMITH_TRACING=true
  LANGSMITH_API_KEY=<your-key>
  LANGSMITH_PROJECT=asterline-assistant  (optional, defaults above)

When disabled (or when the SDK is not installed) this module provides a no-op
``traced`` decorator so that the rest of the application can remain unchanged.
"""

import os

# Must set env vars BEFORE importing LangSmith so it picks them up.
_TRACING_ENABLED = os.getenv("LANGSMITH_TRACING", "false").lower() == "true"
_API_KEY = os.getenv("LANGSMITH_API_KEY", "")
_PROJECT = os.getenv("LANGSMITH_PROJECT", "asterline-assistant")

# Propagate vars expected by LangChain / LangSmith integrations
if _TRACING_ENABLED and _API_KEY:
    os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
    os.environ.setdefault("LANGCHAIN_API_KEY", _API_KEY)
    os.environ.setdefault("LANGCHAIN_PROJECT", _PROJECT)

tracer = None  # exposed for callers that want to pass it explicitly


def _try_init_tracer():
    """Attempt to instantiate a LangChainTracer, returning None on failure."""
    global tracer
    try:
        from langchain_core.tracers.langchain import LangChainTracer  # noqa: PLC0415
        tracer = LangChainTracer(project_name=_PROJECT)
    except Exception as exc:
        # If the SDK doesn't support this API, degrade gracefully.
        import sys
        print(f"[tracing] LangChainTracer init failed: {exc}", file=sys.stderr)
        tracer = None


if _TRACING_ENABLED:
    if not _API_KEY:
        import sys
        print(
            "[tracing] LANGSMITH_TRACING=true but LANGSMITH_API_KEY is empty. "
            "Tracing will be disabled.",
            file=sys.stderr,
        )
    else:
        _try_init_tracer()


def traced(name: str):
    """Decorator that wraps a function with LangSmith tracing when enabled.

    Falls back to a no-op wrapper when tracing is disabled or unavailable.
    """
    try:
        from langsmith import traceable  # noqa: PLC0415
        return traceable(name=name)
    except Exception:
        def _noop(fn):
            return fn
        return _noop
