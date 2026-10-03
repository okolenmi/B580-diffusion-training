"""The bootstrap: stdlib-only, no-op when it should be, and honest when not.

`backend/bootstrap.py` has one property that everything else follows from:
**it must run on the machine where the server cannot start.** So the first
thing checked here is that it imports nothing outside the standard library
and nothing from this project that imports a third-party package. That is
the failure mode that would make it useless exactly when it is needed, and
it is invisible to every other kind of test.

The rest is the behaviour a user sees, on the paths that are reachable
without uninstalling anything from the developer's venv: the no-op, the
"what is missing" report, the URL, the re-exec command, and the refusal to
guess when an install fails.

The full install path is exercised by `scripts/test_bootstrap_install.sh`,
which runs it in a throwaway venv with no site-packages -- because doing it
here would mean deleting fastapi from the venv this test suite is running
in.
"""

from __future__ import annotations

import ast
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend import first_run as bootstrap
from backend.tests.support import check, finish

HERE = Path(__file__).resolve().parent


# ==========================================================================
# Section A: the constraint everything else follows from
# ==========================================================================
print("-- stdlib only, or it cannot run where it is needed --")

tree = ast.parse((HERE.parent / "bootstrap.py").read_text(encoding="utf-8"))
imported: set[str] = set()
for node in ast.walk(tree):
    if isinstance(node, ast.Import):
        imported.update(alias.name.split(".")[0] for alias in node.names)
    elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
        imported.add(node.module.split(".")[0])

third_party = sorted(
    name for name in imported
    if name not in sys.stdlib_module_names
    # A relative import (`from .x import y`) has level > 0 and is not
    # collected above; `backend` itself is this package.
    and name not in {"backend", "__future__"}
)
check(not third_party,
      f"every import is stdlib -- a fastapi import here would break the "
      f"one machine this exists for (found {third_party})")

check(all(
    name in sys.stdlib_module_names or name == "__future__"
    for name in {n.split(".")[0] for n in imported if not n.startswith("_")}
), "and nothing sneaks in through a dotted name")

# The four server packages, and what makes them the four.
check(set(bootstrap.SERVER_PACKAGES)
      == {"fastapi", "uvicorn", "python-multipart", "tomli_w"},
      f"the list is requirements.txt exactly ({list(bootstrap.SERVER_PACKAGES)})")

check(not any("torch" in p for p in bootstrap.SERVER_PACKAGES),
      "and torch is not in it: the training stack is a multi-gigabyte, "
      "device-specific decision that belongs to the wizard "
      "(design doc 12 §5)")

# python-multipart is imported as `multipart`, and a check that got this
# wrong would report a package as missing on a machine that has it.
check(bootstrap._IMPORT_NAMES.get("python-multipart") == "multipart",
      f"python-multipart maps to the module it actually imports as "
      f"({bootstrap._IMPORT_NAMES})")

missing = bootstrap.missing_server_packages()
check(not missing,
      f"on this machine nothing is missing, which is why the no-op path "
      f"below is the one under test (got {missing})")

# ==========================================================================
# Section B: the no-op
# ==========================================================================
print("\n-- nothing to do, when nothing is missing --")

result = subprocess.run(
    [sys.executable, "-m", "backend.first_run", "--port", "8799"],
    capture_output=True, text=True, timeout=120,
    cwd=str(HERE.parent.parent),
)
check(result.returncode == 0,
      f"exits 0 when the packages are present (got {result.returncode})")
check(result.stdout.strip() == "" and result.stderr.strip() == "",
      f"and says nothing at all -- a bootstrap that reports on every start "
      f"is noise about a problem that does not exist (got "
      f"{result.stdout.strip()[:80]!r})")

# ==========================================================================
# Section C: reporting what is missing
# ==========================================================================
print("\n-- what is missing, and why --")

# A package that is genuinely absent, against the real reporting path.
reasons = bootstrap.missing_reasons(["definitely_not_installed_xyzzy"])
check(len(reasons) == 1 and "not installed" in reasons[0],
      f"an absent package is reported as absent ({reasons})")

# Present but broken is a different problem with a different fix, and
# collapsing the two into "not installed" sends the user down the wrong path.
broken = bootstrap.missing_reasons(["fastapi"])
check(not any("not installed" in r for r in broken),
      f"a package that imports fine is not called absent ({broken})")

# ==========================================================================
# Section D: the URL
# ==========================================================================
print("\n-- the link it prints --")

check(bootstrap.server_url(8766, "0.0.0.0") == "http://localhost:8766/setup",
      f"a wildcard bind still gets a link a browser can open "
      f"({bootstrap.server_url(8766, '0.0.0.0')})")
check(bootstrap.server_url(9000, "127.0.0.1") == "http://127.0.0.1:9000/setup",
      f"a specific host is used as given "
      f"({bootstrap.server_url(9000, '127.0.0.1')})")
check(bootstrap.server_url(8766, "::1") == "http://[::1]:8766/setup",
      f"an IPv6 literal is bracketed, or the link is malformed "
      f"({bootstrap.server_url(8766, '::1')})")

# ==========================================================================
# Section E: the hand-off
# ==========================================================================
print("\n-- what it re-execs --")

argv = bootstrap.next_argv(["--port", "9000"])
check(argv[0].endswith("run_server.sh") and Path(argv[0]).exists(),
      f"it re-execs run_server.sh, so .env and argument precedence are "
      f"applied once and in one place ({argv[0]})")
check(argv[1:] == ["--port", "9000"],
      f"and passes the user's own arguments through untouched ({argv[1:]})")

# The bug this file previously missed. `next_argv` above was always right;
# what was wrong was what got handed to it. `--port` is *known* to this
# module, so `parse_known_args` consumed it and the re-exec was left with
# an empty remainder -- the server bound the default 8766 while the printed
# link said 8799. Testing next_argv alone would have stayed green through
# that, so the assertion has to be on what main() forwards.
seen: list[list[str]] = []
saved_bootstrap = bootstrap.bootstrap


def _record(_python, _host, _port, forwarded):
    seen.append(list(forwarded))
    raise bootstrap.BootstrapError("stop here, this test only records")


bootstrap.bootstrap = _record  # type: ignore[assignment]
try:
    # Force the missing-package branch on a machine that has them, so the
    # code under test is reached without uninstalling anything.
    saved_missing = bootstrap.missing_server_packages
    bootstrap.missing_server_packages = lambda: ["fastapi"]  # type: ignore[assignment]
    bootstrap.main(["--port", "9000", "--host", "127.0.0.1", "--extra", "x"])
finally:
    bootstrap.bootstrap = saved_bootstrap  # type: ignore[assignment]
    bootstrap.missing_server_packages = saved_missing  # type: ignore[assignment]

check(bool(seen) and seen[0] == ["--port", "9000", "--host", "127.0.0.1", "--extra", "x"],
      f"--port and --host are forwarded even though this module parses them, "
      f"so the server binds the port the link names (got {seen[0] if seen else 'nothing forwarded'})")

# The same forwarding, reached the way `__main__` reaches it: no argument at
# all, so argv comes from sys.argv. Calling main() with a list -- which is
# what the check above does -- cannot see this, and the result was
# `TypeError: Value after * must be an iterable, not NoneType` on the way to
# a re-exec. Found by running the module; kept here because the install test
# that found it costs a minute and this costs nothing.
seen.clear()
saved_bootstrap = bootstrap.bootstrap
bootstrap.bootstrap = _record  # type: ignore[assignment]
saved_argv = sys.argv
try:
    saved_missing = bootstrap.missing_server_packages
    bootstrap.missing_server_packages = lambda: ["fastapi"]  # type: ignore[assignment]
    sys.argv = ["backend.first_run", "--port", "8799"]
    bootstrap.main()
finally:
    sys.argv = saved_argv
    bootstrap.bootstrap = saved_bootstrap  # type: ignore[assignment]
    bootstrap.missing_server_packages = saved_missing  # type: ignore[assignment]
check(bool(seen) and seen[0] == ["--port", "8799"],
      f"main() with no argv forwards sys.argv's, which is how __main__ "
      f"calls it (got {seen[0] if seen else 'nothing forwarded'})")

# And the check-only path must not install, forward, or restart anything.
seen.clear()
saved_bootstrap = bootstrap.bootstrap
bootstrap.bootstrap = _record  # type: ignore[assignment]
try:
    saved_missing = bootstrap.missing_server_packages
    bootstrap.missing_server_packages = lambda: ["fastapi"]  # type: ignore[assignment]
    code = bootstrap.main(["--check", "--port", "9000"])
finally:
    bootstrap.bootstrap = saved_bootstrap  # type: ignore[assignment]
    bootstrap.missing_server_packages = saved_missing  # type: ignore[assignment]
check(code == 1 and not seen,
      f"--check reports and stops without installing or restarting "
      f"(exit {code}, forwarded {seen})")

# The temporary venv is disposable by construction, not by a cleanup path.
location = bootstrap.venv_dir()
check(str(location).startswith(tempfile.gettempdir()),
      f"the venv goes under the system temp directory, so it cannot end up "
      f"in a git status ({location})")
check(location.name.startswith("distillation-bootstrap-"),
      f"and is named so a crashed one is identifiable ({location.name})")
check(str(os_pid := str(__import__("os").getpid())) in location.name,
      f"carrying the pid, so two bootstraps cannot collide ({location.name}, "
      f"pid {os_pid})")

# ==========================================================================
# Section F: failing without guessing
# ==========================================================================
print("\n-- a failed install stops, and changes nothing --")

# A pip that cannot run is the realistic failure, and the one where a
# bootstrap is most tempted to try something else.
#
# The stub raises FileNotFoundError because that is what subprocess.run
# raises for a missing executable. An earlier version raised a bespoke
# exception, which tested the stub rather than the code: the handler is
# `except FileNotFoundError`, so a bespoke exception correctly escaped --
# and the check then reported "raised instead of refusing" for a case that
# could never happen in production. Testing a handler means raising what
# the thing being handled raises.
saved_run = subprocess.run


def _missing_binary(*_a, **_k):
    raise FileNotFoundError(2, "No such file or directory")


subprocess.run = _missing_binary  # type: ignore[assignment]
try:
    bootstrap.install_packages(Path("/nonexistent/python"), ["fastapi"])
    refused = False
    detail = ""
except bootstrap.BootstrapError as exc:
    refused = True
    detail = str(exc)
finally:
    subprocess.run = saved_run  # type: ignore[assignment]

check(refused,
      f"an interpreter with no pip is refused with a sentence, not a "
      f"traceback ({detail[:70]})")
check("ensurepip" in detail or "by hand" in detail,
      f"and the sentence says what to do about it ({detail[:90]})")

# The refusal must not have invented an alternative target.
check("venv" not in detail.lower() or "temporary" not in detail.lower(),
      "and it did not quietly install somewhere else instead")

finish()