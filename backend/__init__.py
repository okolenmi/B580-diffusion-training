"""Training backend -- the clean-room replacement for ``server/``.

Architecture: full enterprise layering (domain / application /
infrastructure / presentation) with a single composition root, ABC
ports, and zero import-time side effects. The design rationale, the
layering rules, and the milestone plan live in
``docs/design/backend/01-architecture.md``.

The old ``server/`` package stays untouched and in use until the
migration strategy (a future doc) is agreed; this package develops
in parallel on its own port and its own database file.
"""

__version__ = "0.1.0-dev"
