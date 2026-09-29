"""Ports -- the ABCs through which the application reaches the outside world.

Every port is owned by the application layer and implemented by
infrastructure. This is the dependency-inversion seam: use cases depend
on these abstractions, never on sqlite, subprocesses, or FastAPI.
"""
