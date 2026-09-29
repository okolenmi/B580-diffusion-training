"""Application layer -- use cases, ports (ABCs), DTOs.

Orchestrates the domain to fulfil delivery-independent scenarios.
Depends on the domain and on nothing else: infrastructure types, the
web framework, and SQL never appear here -- external capabilities are
reached through the ABCs in ``ports``.
"""
