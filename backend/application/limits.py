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

DEFAULT_EXECUTION_PAGE_SIZE = 50

# --- dataset curation -----------------------------------------------------

DEFAULT_DATASET_ITEM_PAGE_SIZE = 500
"""Rows per page when a caller does not ask for a size.

Was: every row. A dataset is the one collection in this system whose
size is genuinely unbounded -- it grows by ingestion, not by user
action -- so "return it all" meant a request whose response grew until
it hit the point where the browser stopped rendering it (docs 07 F-14,
docs 08 Q10). The page is large enough that ordinary datasets are
still served whole, which is what the curation UI assumed; `total` and
`next_offset` let a client know when it is not.
"""

MAX_PREVIEW_BYTES = 32 * 1024 * 1024
"""Ceiling on a dataset file served as a preview.

Bytes, not pixels: an allowlist already guarantees the extension is an
image type, and a decoder's own limits are its business. The cap exists
so a 4 GB file named .png cannot be streamed to a browser.
"""

# --- uploads ---------------------------------------------------------------

MAX_UPLOAD_BYTES = 8 * 1024 * 1024 * 1024
"""Ceiling for one uploaded asset.

Generous on purpose: this is a trained .safetensors, which is routinely
multiple gigabytes, and refusing it would mean the user finds out after
the upload rather than before. The streaming upload (WP-03) is what
makes a ceiling like this affordable -- 8 GB is a disk write, not 8 GB
resident.
"""

# --- event stream ----------------------------------------------------------

EVENT_REPLAY_RING = 512
"""Lifecycle events kept for replay to a reconnecting client.

The answer to "a browser tab slept through `run_completed` and now
believes the run is still running". 512 covers a very long unattended
run at the supervisor's polling rate; past that a client is told
`resync_required` and refetches, which for a single-user loopback tool is
cheaper and always correct (docs 09 event contract).

Not a byte budget and not a promise of durability: it is a ring in memory,
lost on restart, and only lifecycle events are kept -- deltas are
coalesced per client anyway.
"""

SSE_QUEUE_MAX = 256
"""Frames buffered per subscriber before the slowest one loses frames.

The buffer holds *lifecycle* events, which are never dropped for
coalescing (see presentation/sse.py): a missed run_completed is a row
that stays `running` forever. Deltas coalesce instead, so this ceiling
only bites a client that has stopped reading altogether.
"""

SSE_HEARTBEAT_SECONDS = 15.0
"""Interval between keep-alive comments on an idle stream.

Sits under the typical 60s idle timeout of whatever sits in front, so a
stream that is genuinely quiet is not mistaken for a dead one.
"""

# --- graphs ---------------------------------------------------------------

MAX_GRAPH_NAME = 120
"""Ceiling on a saved graph's name."""

MAX_GRAPH_DESCRIPTION = 1000
"""Ceiling on a saved graph's description."""


# --- internal defaults, not part of the API contract ----------------------
#
# Kept here for one reason: the file is where someone looks when asking
# "what numbers does this system have?". These are not numbers the API
# and its clients have to agree on, and moving them here does not make
# them such -- it only makes them findable, which is a different and
# weaker claim.

DEFAULT_STOP_GRACE_SECONDS = 15.0
"""How long a trainer gets to exit on SIGTERM before SIGKILL.

Long enough for a checkpoint write to finish (which is the whole point
of stopping rather than killing), short enough that a user does not
think the button is broken.
"""