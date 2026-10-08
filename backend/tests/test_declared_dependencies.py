r"""Does the requirements manifest actually match what the code imports?

`requirements_manifest.py` exists because the answer to "what does this
project need" lived in three places that could not see each other. This
file is the check that keeps two of them from drifting again, and it
exists because they already had.

Two training-tier packages were missing while being hard top-level imports
of live modules:

* `tqdm` -- `from tqdm import tqdm` at the top of `manager/builder.py`,
  where it drives dataset ingest. Without it that module does not import,
  so the entire dataset builder is unavailable. It predates this work.
* `regex` -- the CLIP tokenizer's word splitter (`nodes/model/tokenizer.py`,
  design doc 12 section 7.3-C2). CLIP's pattern needs `\p{L}` and `\p{N}`,
  which the standard library's `re` does not have.

Neither appeared in `requirements.txt`, in the manifest, or in
`docs/setup.md`. The manifest's own docstring names this failure -- "a
package the trainer imports, which appears in no requirements file at
all" -- and then nothing checked for it, so for a while it was a
description rather than a guarantee.

**The check is an AST pass over the live tree**, not a grep. A grep finds
the words "tqdm" and "comfy" in docstrings and comments, and this project's
comments discuss both constantly; an AST pass finds only what is actually
imported. Every third-party module the running code imports has to be
either in the manifest or in the allowlist below, with a reason.

**What is deliberately excluded**, and why:

* `scripts/` -- analysis and probe tools, not the running project.
  `libcst`, `mutmut` (mutation testing) and `playwright` (UI probing) are
  development dependencies and belong in `requirements-dev.txt`, which is
  not this manifest's subject.
* `archive/` -- retired code, kept for reference and imported by nothing.
* test and smoke-test directories -- they run under the gate, not in
  production, and they are allowed to reach further than the code they
  check.

**The one direction between our own packages** (MEM-05 #5): `nodes/` is
the layer the backend is built on top of, so a node module importing
`backend/` inverts that and is refused here. `backend/` importing
`nodes/` is the whole design and stays legal. The rule is a directional
AST check over the `nodes/` tree -- absolute `backend` imports and
relative ones whose level resolves up to the top-level `backend` -- and
it reports the offending file and line, not just a count. A live smoke
test (`smoke_test_graph_memory.py`) reaches into `backend/` to test the
worker's own wiring helpers; that is why `nodes/smoke_tests` is excluded
above rather than this rule carving out exceptions.

Run: `python backend/tests/test_declared_dependencies.py`
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.application.ports.environment import IMPORT_NAMES  # noqa: E402
from backend.application.ports.requirements_manifest import (  # noqa: E402
    COMFY_ADDITIONS,
    FULL_INSTALL,
    REQUIREMENTS,
    TRAINING,
    by_tier,
)
from backend.tests.support import check, finish  # noqa: E402

REPO = Path(__file__).resolve().parents[2]

#: This project's own packages, which are not dependencies of each other in
#: any way an installer should be told about.
LOCAL = frozenset({
    "nodes", "backend", "server", "manager", "runs", "runs_hw_validation",
    "paths", "run_tests", "config_io", "config_model", "compute_update",
    "monitor_bus", "frozen_lora_sd",
    # `scripts` is this project's own measurement harness (hw_validate.py and
    # the sweep drivers), and the analysis scripts under "shapes diversity
    # problem 2" import it to reuse the harness rather than reimplement it.
    # Its own files are excluded from the walk below, but the import edge
    # into it is real, so it belongs here rather than in TRANSITIVE_BUT_DIRECT.
    "scripts",
})

#: Top-level directories whose contents are not the running project.
#:
#: `bad_LoRA_investigation` joins `scripts` here for the same reason: it holds
#: standalone diagnostic programs, nothing in the running tree imports them,
#: and they deliberately depend on *reference* implementations the trainer
#: itself must not acquire -- diffusers, transformers, and (for A2) kohya
#: sd-scripts. Those are oracles for checking this project, not inputs to
#: it; declaring them in requirements.txt would put them in COMFY_ADDITIONS
#: and pip-install an oracle into the trainer's own environment, which is the
#: wrong dependency direction. The trainer reaches its training deps only
#: through the manifest below.
EXCLUDED_PREFIXES = (
    "archive",
    "scripts",
    "bad_LoRA_investigation",
    "backend/tests",
    "nodes/smoke_tests",
    "manager/smoke_tests",
    "frontend",
)

#: Imported directly, declared only transitively. Each is present because a
#: package we *do* declare requires it, and the dependency is one every
#: version of that package makes -- but the code reaching for it directly is
#: still a fact, and recording it is cheaper than rediscovering it later.
#:
#: These are deliberately *not* in the manifest. Adding them to the required
#: tier would put them in `COMFY_ADDITIONS`, which is the list the installer
#: pip-installs into ComfyUI's virtualenv -- a venv this project does not
#: own. Trading a guaranteed transitive dependency for a new install into
#: someone else's environment is the wrong trade, so the import stays and
#: the reasoning lives here instead.
TRANSITIVE_BUT_DIRECT: dict[str, str] = {
    "pydantic": "fastapi requires it, in every version since 0.100",
    "pydantic_core": "a pydantic implementation detail, same reasoning",
    "starlette": "fastapi requires it; we import it directly in the ASGI "
                 "app's own plumbing",
}


def report(condition: bool, message: str, detail: str = "") -> None:
    """`check`, but with the measurement attached to a failure.

    `support.check` takes only a condition and a message, so a check that
    wants to say *what it measured* has to build the message itself. The
    detail is appended only on failure -- a passing line stays one line.
    """
    check(condition,
          message if condition or not detail else f"{message} -- {detail}")


def imported_distributions() -> dict[str, set[str]]:
    """Third-party distributions the live tree imports, and where.

    Distribution names rather than module names, because that is what a
    requirements file holds. `IMPORT_NAMES` is the project's own map for the
    pairs where the two differ.
    """
    reverse = {module: dist for dist, module in IMPORT_NAMES.items()}
    found: dict[str, set[str]] = {}

    for path in sorted(REPO.rglob("*.py")):
        rel = path.relative_to(REPO).as_posix()
        if rel.startswith(EXCLUDED_PREFIXES):
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError):
            continue

        modules: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules.update(a.name.split(".")[0] for a in node.names)
            elif isinstance(node, ast.ImportFrom):
                # level > 0 is a relative import: our own package.
                if node.level == 0 and node.module:
                    modules.add(node.module.split(".")[0])

        for module in modules:
            if module in sys.stdlib_module_names or module in LOCAL:
                continue
            found.setdefault(reverse.get(module, module), set()).add(rel)
    return found


print("-- what the live tree actually imports --")

imported = imported_distributions()
for distribution in sorted(imported):
    files = imported[distribution]
    print(f"  {distribution:18} {len(files):3} file(s)"
          f"  e.g. {sorted(files)[0]}")

report(bool(imported),
       f"the AST pass found third-party imports ({len(imported)} distinct "
       "distributions)",
       "none at all, so this check would be measuring nothing")

print("\n-- each one is declared, or is a recorded exception --")

declared = set(FULL_INSTALL)
undeclared = [(d, sorted(imported[d])) for d in sorted(imported)
              if d not in declared and d not in TRANSITIVE_BUT_DIRECT]

for distribution, reason in TRANSITIVE_BUT_DIRECT.items():
    if distribution in imported:
        print(f"  ALLOWED: {distribution} -- {reason}")

report(not undeclared,
       f"every third-party import the live tree makes is in the manifest "
       f"({len(declared)} declared, {len(imported)} imported)",
       f"{len(undeclared)} undeclared: "
       + "; ".join(f"{d} (e.g. {f[0]})" for d, f in undeclared[:4]))

print("\n-- the allowlist does not rot --")

stale = sorted(set(TRANSITIVE_BUT_DIRECT) - set(imported))
report(not stale,
       f"every allowed-but-transitive package is still really imported "
       f"({len(TRANSITIVE_BUT_DIRECT)} in the allowlist)",
       f"no longer imported, so the exception is dead weight: {stale}")

reasonless = sorted(d for d, why in TRANSITIVE_BUT_DIRECT.items()
                    if not why.strip())
report(not reasonless,
       "and every one of them says why",
       f"no reason given, which reads as a decision and records nothing: "
       f"{reasonless}")

print("\n-- the tier invariants still hold --")

training = [r.distribution for r in by_tier(TRAINING)]

report("regex" in training,
       "regex is a declared training requirement -- the tokenizer's \\p{L} "
       "has no standard-library equivalent",
       f"training tier: {training}")

report("tqdm" in training,
       "and so is tqdm, which manager/builder.py imports at top level",
       f"training tier: {training}")

not_flagged = [r.distribution for r in by_tier(TRAINING) if not r.never_install]
report(not not_flagged,
       "every training row is never_install, so the installer will not pip "
       "them into a virtualenv this project does not own",
       f"rows without the flag: {not_flagged}")

leaked = sorted(set(training) & set(COMFY_ADDITIONS))
report(not leaked,
       "and none of them reached COMFY_ADDITIONS",
       f"leaked into what gets installed into comfyi's venv: {leaked}")

print("\n-- requirements.txt and the required tier are the same list --")

from_txt = sorted(
    line.strip()
    for line in (REPO / "requirements.txt").read_text(encoding="utf-8").splitlines()
    if line.strip() and not line.lstrip().startswith("#")
)
from_manifest = sorted(r.distribution for r in REQUIREMENTS
                       if r.tier == "required")
report(from_txt == from_manifest,
       f"they agree ({len(from_txt)} packages)",
       f"requirements.txt has {from_txt} and the required tier has "
       f"{from_manifest}")

report(not (set(imported) - declared - set(TRANSITIVE_BUT_DIRECT)),
       "so no import is unaccounted for by either list")

print("\n-- and the one direction between our own packages that is forbidden --")


def _backend_target(node, package_parts: list[str]) -> str | None:
    """The top-level ``backend`` module one import node reaches, or None.

    Two shapes count, because both are ways to actually import it:
    ``import backend.x`` / ``from backend.x import y`` (absolute), and
    a relative import whose level climbs out of the importing file's own
    package to the repo root. For ``nodes/memory/graph_memory.py`` the
    package is ``nodes.memory``, so ``.`` is ``nodes.memory``, ``..`` is
    ``nodes`` and ``...`` is the top level -- only the last reaches
    ``backend``. Walking the package parts up by ``level - 1`` and
    reading the target off whatever is left is what keeps a
    ``from ..backend`` inside ``nodes/memory/`` (which is ``nodes.backend``,
    a different thing) from being flagged.
    """
    if isinstance(node, ast.Import):
        for alias in node.names:
            if alias.name == "backend" or alias.name.startswith("backend."):
                return alias.name
        return None
    if not isinstance(node, ast.ImportFrom):
        return None
    if node.level == 0:
        module = node.module or ""
        return module if module == "backend" or module.startswith("backend.") else None
    climbed = node.module or ""
    remaining = package_parts[: len(package_parts) - (node.level - 1)]
    resolved = ".".join(remaining + ([climbed] if climbed else []))
    return resolved if resolved == "backend" or resolved.startswith("backend.") else None


def backend_imports_under(prefix: str) -> list[tuple[str, int]]:
    """Every `backend` import reached from a module under ``prefix``.

    Returns ``(relative path, line number)`` for each hit, so a failure
    names the offending line instead of just the file.
    """
    hits: list[tuple[str, int]] = []
    for path in sorted(REPO.rglob("*.py")):
        rel = path.relative_to(REPO).as_posix()
        if not rel.startswith(prefix) or rel.startswith(EXCLUDED_PREFIXES):
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError):
            continue
        package_parts = rel.split("/")[:-1]
        for node in ast.walk(tree):
            if _backend_target(node, package_parts) is not None:
                hits.append((rel, node.lineno))
    return hits


layering_violations = backend_imports_under("nodes/")
report(not layering_violations,
       f"no node module imports backend ({len(layering_violations)} violation(s)) "
       f"-- nodes/ is the layer the backend runs on top of, never the other "
       f"way round (task rule 9 / ADR 0005 'Layering')",
       "reached backend from: "
       + "; ".join(f"{f}:{line}" for f, line in layering_violations[:6]))

# Guard against the check quietly measuring nothing: it must actually walk
# the node tree, or a future refactor that moves the files would turn this
# into a green that means nothing.
node_modules = [
    path for path in REPO.rglob("*.py")
    if path.relative_to(REPO).as_posix().startswith("nodes/")
    and not path.relative_to(REPO).as_posix().startswith(EXCLUDED_PREFIXES)
]
report(len(node_modules) > 100,
       f"and the pass actually walked the node tree ({len(node_modules)} modules)",
       f"only {len(node_modules)} modules found under nodes/ -- the rule "
       f"would pass by measuring nothing")

finish()
