"""Presentation layer -- the FastAPI adapter.

Thin by rule: parse (pydantic/FastAPI), call one use case from
``ApplicationServices``, shape the response. No SQL, no threads, no
business rules, no ``os.environ``. The only clever code allowed here
is transport-specific (the SSE stream bridge, error mapping).
"""
