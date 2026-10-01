"""limits -- the numbers the API and its clients both have to agree on.

Every one of these was written twice: once in the use case that enforces
it and once in the route that documents the default, which is how
``MAX_ITEM_PAGE`` ended up in two files with the same value and
``DEFAULT_LIMIT`` in a third (docs 08 S-08). Use cases own the rule and
import from here; routes import the same constants instead of repeating
them, so a doc and a 422 can no longer disagree.
"""

# --- paging ---------------------------------------------------------------

MAX_PAGE_SIZE = 500
"""Ceiling for any list endpoint's ``limit``."""

DEFAULT_RUN_PAGE_SIZE = 50
DEFAULT_EXECUTION_PAGE_SIZE = 50

# --- run log tail ----------------------------------------------------------

MAX_LOG_LINES = 500
DEFAULT_LOG_LINES = 100

# --- graphs ---------------------------------------------------------------

MAX_GRAPH_DESCRIPTION = 1000
"""Ceiling on a saved graph's description."""