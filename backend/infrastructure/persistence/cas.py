"""The compare-and-swap, defined once.

Every terminal transition in this backend is the same statement: *change
this row's status only if it still holds a status I expected.* Whoever
changes a row wins; everyone else changed nothing. Both consequences
matter, and the second one is the one that is easy to lose:

* a row is moved exactly once, so two writers cannot both claim it;
* ``rowcount`` **is** the verdict, so the caller must not re-read to
  find out whether it won -- a re-read answers a different question, and
  answering "am I still running?" after losing a race is how a run gets
  reported as finished by a writer that never finished it.

Both aggregates that have a terminal state need this: graph executions
(guarded on one exact status, the one the writer read) and dataset tasks
(guarded on "any active status"). They are the same rule with a
different set of columns set, so they share the statement rather than
each writing their own.

What this deliberately does not do is announce anything. Only the
aggregates that buffer events go through
``application/lifecycle_writer.py``, which wraps this in
CAS-then-publish. Dataset tasks have no events yet, so they call this
directly -- which is a difference in what they can announce, not a second
definition of how the swap works.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence


def compare_and_swap_status(
    conn,
    *,
    table: str,
    row_id: int,
    from_statuses: Sequence[str],
    to_status: str,
    set_columns: Mapping[str, object] | None = None,
) -> bool:
    """Move one row from any of ``from_statuses`` to ``to_status``.

    Returns True iff this call was the one that moved it -- i.e. it is the
    winner, and the caller may announce. False means someone else got
    there first and this statement changed nothing.

    ``set_columns`` is anything besides the status -- a reason, a
    timestamp, a payload. Omitted for the many callers whose only change
    is the status itself.

    ``from_statuses`` is a set rather than one status because the two
    callers genuinely differ: a graph-execution writer CASes from the
    single status it read (claiming ``queued -> running`` must not
    succeed if the row is already ``running``), whereas a dataset task
    finalises from *any* active status, because ``pending`` and
    ``running`` are both something a task can be stopped out of.

    One statement on purpose. A read-then-write would leave a window
    between them, and that window is the entire thing this exists to
    close.
    """
    if not from_statuses:
        raise ValueError(
            "compare_and_swap_status needs at least one expected status; "
            "an unguarded UPDATE is a different operation and almost "
            "never what the caller meant"
        )
    columns = {"status": to_status, **(set_columns or {})}
    assignments = ", ".join(f"{name} = ?" for name in columns)
    placeholders = ", ".join("?" for _ in from_statuses)
    cursor = conn.execute(
        f"UPDATE {table} SET {assignments} WHERE id = ? AND status IN ({placeholders})",
        (*columns.values(), row_id, *from_statuses),
    )
    return cursor.rowcount > 0