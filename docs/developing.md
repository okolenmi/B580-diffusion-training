# Developing

Rules for changing this repository. They are short on purpose: a rule that can be checked gets followed. Where a script enforces one, it is named.
Layout and flow: `architecture.md`. What exists: `status.md`.

## The three questions before a change

1. Which layer owns this? (table below)
2. Is it a process choice (a node field), an environment fact (a graph setting) or a machine fact (app settings / `.env`)?
3. Is a wire the honest model, or is it an event (going backward: end of epoch, prompts changed, checkpoint due)?

## Where code goes

`python scripts/check_imports.py --list` prints the enforced table; `scripts/check_imports.py` fails the gate on a violation. In short:

| Package | May import | Never |
|---|---|---|
| `graph`, `families/api`, `families/registry` | stdlib (plus `graph` for the families) | torch, anything else in the repo |
| `families/general`, `families/<name>` | `families/api`, `graph`, torch; a family only itself | other families, nodes, engine, backend |
| `data`, `engine` | `graph`, `families/api`, `families/registry`; `engine` also `data` | nodes, backend, a concrete family |
| `nodes` | `graph`, `data`, `engine`, `families/api`, `families/registry`, `perf` | a concrete family, backend |
| `backend` | its own layers (`domain` ← `application` ← `infrastructure`, `presentation`); `graph`, `families/api` | torch at import time |
| `frontend` | itself; talks to `/api/v1` | anything Python by name |

Inside a package imports are relative, between packages absolute. ComfyUI is never imported by production code; the two parity scripts in
`accept/` run as separate processes and take our folders off `sys.path` first (also checked).

Where does this logic go? The first yes wins:

1. True only because the model is SDXL or Anima → `families/<name>/`.
2. True of several models but needs torch → `families/general/`.
3. The server needs it without torch (listing, detecting, validating) → `families/api` or a `fingerprint.py` beside the family.
4. About running a graph (order, events, completion) → `graph/`, never a node.
5. About drawing or editing a node → catalog metadata (`Group`, `Port`, `MODULES`) plus the generic editor code; never a node-specific branch in the frontend.
6. A step of training a person may want to replace → a node with one clear job; an optional extra of a node → an addon (`internals/addons.md`).
7. A computation every training step needs fast → `engine/`.
8. Anything else is a utility: put it where its only caller lives and promote it when a second caller appears. No `utils.py`.

A file that needs a section-header comment to separate two topics is two files.

## Writing a node

A node declares class data and does its work in `build`; `__init_subclass__` validates the declaration at import, so a broken node fails at import,
not at the first run. The smallest one is `nodes/primitive/values.py`:

```python
class FloatNode(Node):
    DISPLAY_NAME = "Number"
    CATEGORY = "Primitive"
    INPUTS = {"value": Port("value", float, required=False, default=0.0)}
    OUTPUTS = {"value": Port("value", float)}

    def build(self, **inputs):
        self.validate_inputs(inputs)
        return {"value": 0.0 if inputs.get("value") is None else inputs["value"]}
```

What a node may declare (`graph/node.py`): `INPUTS`/`OUTPUTS`, `POSSIBLE_*` (shape depends on resolve), `GROUPS`, `EVENTS_IN`/`EVENTS_OUT`,
`ADDON_SLOTS`, `MODULES` (what is worth watching), `READS_SETTINGS`, `SETTINGS`, `DYNAMIC`, `CATEGORY`. `resolve(params)` says what the node
will have once files are read (`contracts/resolve.md`); it reads headers only, with a size cap, never the weights.

- A wire's type is the Python class of the handle that travels on it; handles are real objects (`DenoiserPart`, `Conditioning`, ...).
- Ports and names are API: they are saved in graphs. Change them like any contract (below).
- Environment (device, precision, limits, seeds) comes from graph settings; declare them in `READS_SETTINGS`. Never write `"xpu"` or a dtype literal in a node.
- Nothing family-specific in a node: call the family through `families/api`.
- No module-level "current" state; state lives on an object with run lifetime. Graph logic is single-threaded; threads are for I/O and prefetch only.
- Memory: anything that holds more than a few MB on the device is accounted through the engine's residency manager so admission sees it.
  Never call `empty_cache` or `synchronize` from a node; `engine/runtime` owns them.
- Cancel-aware: check `context.should_cancel()` in long loops.
- Adding a family: a `families/<name>/` with `fingerprint.py` (decisive header keys), parts, conditioning, latents, objective, adapters, and its id in
  `families/registry.py`; cover detection and a toy pipeline in `tests/`.

## Errors and logging

- Fail early and name the thing: node uid, port, file, value. A message that cannot be acted on without a debugger is a bug.
- Catch the specific exception, add context, re-raise or turn it into a validation code. No catch-all to continue.
- Tell the person through a node's `diagnostics` or an info display, not `print`.
- No silent fallbacks: an unavailable option fails, or the fallback is chosen visibly.

## Comments and docs

| Put it in | What |
|---|---|
| the name or type | what a thing is |
| a short comment | why a non-obvious choice was made; one to three lines, true now |
| the docstring | the contract a caller must know; no history |
| `docs/decisions/NNNN-*.md` | a decision with a cost and a test that pins it |
| `docs/experiment-results.md` | measured numbers and incidents; code cites the entry, it does not copy it |
| `docs/status.md` | what is built, missing, waiting on hardware |
| git history | how it used to be |

No measurements and no history ("previously", "used to") in code comments; a comment that restates the next line is deleted. If a long
explanation is needed to understand the code, simplify the code. `scripts/check_comments.py` flags numbers with units and history phrases;
`scripts/check_doc_links.py` and `scripts/check_doc_refs.py` fail on dead links, citations and repo paths. Do not delete or rename an
`experiment-results.md` entry a comment cites.

## Tests

- Plain scripts with the `check()` helper of `tests/support.py` (no pytest in the trainer): `python -m tests.<name>`, grouped by what they protect
  (`tests/README.md`); backend suites in `backend/tests/`, frontend in `frontend/tests/` (`node --test`). `python run_tests.py` runs them all,
  `python run_tests.py trainer -- <name-part>` one. Run the group that matches what changed, not everything after a docs edit.
- Every file's docstring says its purpose. A bug fix gets the test that fails without it. Remove tests that guard nothing.
- Deterministic: seeded, no wall-clock assertions. Fixtures hold no weights; tiny random models are built in the test. Do not mock the layer under test.
- Everything must run on CPU. A behaviour that needs the card goes into `TESTING.md` with how to check it, and is marked `untested` in `manual/`.

## Changing a public contract

Port names, handle classes, event types, graph JSON keys, catalog format, setting keys, REST routes and database schemas are in `contracts/`.
Change the contract page first, update every user (`grep`; starters and frontend included) in the same commit, and bump `catalog_format` or the
graph `format` if the editor must tell. Saved graphs are not migrated; that is an owner decision, so a change may invalidate old files.

## Before you commit

1. Which layer owns it, and does `check_imports` pass?
2. Could it have been a setting, an event, an addon or a family method?
3. What do the starters, contracts and frontend need in the same change?
4. What happens on Stop, on failure and on a second run in the same process?
5. Where does its memory show up in admission control?
6. Is `status.md` still true?
