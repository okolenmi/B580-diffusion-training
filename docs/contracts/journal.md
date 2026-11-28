# Journal (the system console)

The console every page shows is one list kept by the server, so a reload, another tab or a restart of the server loses nothing. Code:
`backend/application/ports/journal.py`, `backend/infrastructure/file_journal.py`, `journal_log_handler.py`, `backend/application/journal_listener.py`,
`backend/presentation/api/journal.py`; the page side is `frontend/js/console/` (`journal_client.js` keeps in step with the server, `console_view.js` draws,
`journal_model.js` filters) and the window around it is `frontend/js/shell.js`.

## Entry

`{seq, at, level, source, text}`: `seq` counts up from 1 and continues after a restart; `at` is seconds since the epoch; `level` is `info`, `success`,
`warn` or `error`; `source` is `server`, `page` or `run #N`; `text` is at most 4000 characters.

## What is written

| Source | What |
|---|---|
| `server` | `server started` at every start, the warnings and errors of the server's own log (`logging`, WARNING and above), `deleted N runs from the history` |
| `run #N` | `queued`, `started`, `finished`, `failed: <reason>`, `stopped`, from the run's lifecycle events; and what a node warned about during the run (a `warning` report: Inject Adapter, Trainer, Preview Image, ...), once, not again when a restarted server reads the run's event file from the start |
| `page` | the lines pages write (`Saved "x"`, `Could not ...`); the source is always `page`, a page cannot pass itself off as the server or a run |

Held in memory (newest 2000) and in `journal.jsonl` beside the database, cut back when it has grown to twice that; a torn last line is ignored when it is read back.

## REST

| Route | |
|---|---|
| `GET /api/v1/journal?after=N&limit=M` | `{entries, last}`: the newest `M` (default 300, at most 1000) entries numbered above `N`, oldest first, and the number of the newest entry there is. `after=0` is the history |
| `POST /api/v1/journal` | body `{level, text}`: 201, the entry as filed (source `page`); 422 `invalid_query` for an empty text |

503 `journal_unavailable` when the server was built without one.

## The page

The client asks for the history once, then for what came after the highest number it holds every 1.5 s (10 s while the tab is hidden), and at once after it
wrote a line. A line the server cannot take (it is down) is kept on the page, marked local, and the loss and the return of the server are each said once.
An `error` entry that is new to the page (not part of the history it loaded) opens a docked console; warnings and errors that arrive while it is docked are
counted on its button. The window can filter (everything, warnings and errors, errors only), copy the lines shown, and Clear, which hides what is shown now
(a number kept in the browser; the history stays on the server).
