"""Domain layer -- entities, value objects, events, domain errors.

The innermost layer: it imports nothing from application,
infrastructure, presentation, or any third-party framework (no
FastAPI, no pydantic, no sqlite3). All state transitions and
invariants of the training business live here.
"""
