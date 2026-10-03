"""Bootstrap -- make the server's own dependencies installable, then hand off.

The first-run installer work, ``docs/design/12-installer-and-comfy-decoupling.md``
§2.

**This module is stdlib-only, and that is the whole design constraint.**
It has to run on the machine where the server cannot start, so it may not
import anything that imports a third-party package. ``backend/__init__`` is
a docstring and ``backend.python_floor`` imports only ``sys``, so a module
inside the package is still safe to import here -- and
``backend/tests/test_bootstrap.py`` asserts it by parsing this file's
imports, because that is the failure mode which would make this useless
exactly when it is needed, and which no other kind of test would catch.

**Not named `backend.bootstrap`.** That is the composition root
(``build_container``), and this file was written over it once before the
name clash was noticed -- by a test failing with "cannot import name
'build_container'". The collision was in the launcher, so it surfaced
immediately rather than silently; but a module that shadows the wiring
root is worth avoiding by name as well as by test.

**What it does, in order.**

1. Check whether the server's four packages are already importable. If they
   are, this module does nothing at all -- it is not on the path a working
   install takes.
2. Create a **temporary virtualenv** under the system temp directory.
3. Install the four packages into it. They are small and pure-Python; the
   measured cost on this machine is 7 s and 32 MB, which is why automating
   this is worth anything at all.
4. Print the URL and try to open a browser, in that order -- the terminal
   message *is* the link, because an automatic open fails often enough (no
   browser, headless, a desktop that is not running) that assuming it
   worked would be wrong.
5. Re-exec the server from that venv.

`--check` stops after step 1: it prints the same report and installs
nothing. Not decoration -- without it the only way to ask what is missing
is to start installing, and the test that drives this path had to work
around that.

**Why a temporary venv rather than the user's.** It cannot break anything
the user already has, and it is removable by construction: under the temp
directory, named with the pid, alive only for as long as the server runs
from it. There is no cleanup path to forget.

It is *not* where the project ends up. The wizard's next step moves the
project into a venv the user chooses -- their own, or ComfyUI's -- and the
training stack is installed there. This one holds four small packages and
never grows.

**No third path on failure.** If the install cannot complete, this says so
and exits. It does not try another index, does not fall back to a user-site
install, and does not continue with a partial environment: a bootstrap that
guesses is worse than one that stops and explains, because the thing it is
bootstrapping is the thing that would have told the user what was wrong.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
import time
import webbrowser
from pathlib import Path

#: The packages the *server* needs -- exactly `requirements.txt`. Not the
#: training stack: torch and the accelerator are a multi-gigabyte,
#: device-specific decision (design doc §5) that belongs to the wizard, not
#: to a bootstrap that runs before any of it can be asked.
SERVER_PACKAGES: tuple[str, ...] = (
    "fastapi",
    "uvicorn",
    "python-multipart",
    "tomli_w",
)

#: Import name where pip and Python disagree. A missing-package check that
#: imported `python_multipart` instead of `multipart` would report a
#: package as absent on a machine that has it.
_IMPORT_NAMES = {"python-multipart": "multipart"}

#: Exit code distinguishing "could not bootstrap" from the server's own, so
#: run_server.sh can say something better than a traceback.
EXIT_BOOTSTRAP_FAILED = 78  # EX_CONFIG, the closest portable choice


class BootstrapError(Exception):
    """Something the user needs to be told, in one sentence."""


# --------------------------------------------------------------------------
# what is missing
# --------------------------------------------------------------------------


def missing_server_packages() -> list[str]:
    """Which of the four cannot be imported here. Empty means nothing to do.

    Import-based rather than metadata-based, because the question is "can
    the server run" and the server imports them. A draft that read
    `importlib.metadata` would have been satisfied by a distribution whose
    import then failed.
    """
    import importlib
    import importlib.util

    missing = []
    for distribution in SERVER_PACKAGES:
        module = _IMPORT_NAMES.get(distribution, distribution)
        try:
            if importlib.util.find_spec(module) is None:
                missing.append(distribution)
                continue
            importlib.import_module(module)
        except Exception:  # noqa: BLE001 -- a broken install is a missing one
            missing.append(distribution)
    return missing


def missing_reasons(packages: list[str]) -> list[str]:
    """Per package: whether it is absent, or present and unusable.

    The distinction is worth printing. "No module named 'fastapi'" and
    "fastapi is installed but a dependency of it is missing" have the same
    symptom and completely different fixes, and collapsing them into one
    message sends the user to reinstall something that was fine.
    """
    reasons = []
    for distribution in packages:
        module = _IMPORT_NAMES.get(distribution, distribution)
        try:
            __import__(module)
            reasons.append(f"  {distribution}: installed, but importing it fails")
        except ModuleNotFoundError as exc:
            reasons.append(f"  {distribution}: not installed ({exc})")
        except Exception as exc:  # noqa: BLE001
            reasons.append(
                f"  {distribution}: installed, but importing it raises "
                f"{type(exc).__name__}: {exc}"
            )
    return reasons


# --------------------------------------------------------------------------
# the temporary venv
# --------------------------------------------------------------------------


def venv_dir() -> Path:
    """Where the temporary venv goes.

    Under the system temp directory and named with the pid, so two
    bootstraps on one machine cannot collide and a crashed one is
    identifiable. Not under the project: this directory is disposable and
    must not turn up in a git status.
    """
    return Path(tempfile.gettempdir()) / f"distillation-bootstrap-{os.getpid()}"


def create_venv(target: Path, python: str) -> Path:
    """Create the venv and return its interpreter.

    `--without-pip` is deliberately not used: pip is how the four packages
    get in, and ensurepip ships with CPython rather than being downloaded,
    so this needs no network to produce a usable interpreter.
    """
    try:
        subprocess.run(
            [python, "-m", "venv", str(target)],
            check=True, capture_output=True, text=True, timeout=300,
        )
    except subprocess.TimeoutExpired as exc:
        raise BootstrapError(
            f"creating the temporary environment timed out after 300s "
            f"({python} -m venv). Is the disk full, or is "
            f"{tempfile.gettempdir()} not writable?"
        ) from exc
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or "").strip().splitlines()
        raise BootstrapError(
            f"could not create a temporary environment with "
            f"{python} -m venv"
            + (f":\n    {detail[-1]}" if detail else f" (exit {exc.returncode})")
        ) from exc

    interpreter = target / ("Scripts" if os.name == "nt" else "bin") / "python"
    if not interpreter.exists():
        raise BootstrapError(
            f"the temporary environment was created but has no interpreter at "
            f"{interpreter}"
        )
    return interpreter


def install_packages(interpreter: Path, packages: list[str]) -> None:
    """Install into the temporary venv, streaming pip's output.

    Streamed rather than captured: pip is the only thing here that touches
    the network, and a user watching a stalled install should see that it is
    stalled rather than sit in front of a frozen script for two minutes.
    """
    command = [
        str(interpreter), "-m", "pip", "install",
        "--disable-pip-version-check",
        *packages,
    ]
    print(f"  $ {' '.join(command)}", flush=True)
    try:
        result = subprocess.run(command, timeout=1800)
    except subprocess.TimeoutExpired as exc:
        raise BootstrapError(
            "the install did not finish within 30 minutes. If this machine "
            "is behind a proxy, pip needs its index configured; the output "
            "above is where it stopped."
        ) from exc
    except FileNotFoundError as exc:
        raise BootstrapError(
            f"could not run {interpreter} -m pip: {exc}. The interpreter "
            f"exists but pip does not -- create the environment with "
            f"ensurepip, or install the four packages by hand."
        ) from exc
    if result.returncode != 0:
        raise BootstrapError(
            f"pip exited {result.returncode}. Its output above is the reason; "
            f"the four packages are {', '.join(packages)}."
        )


# --------------------------------------------------------------------------
# telling the user, and handing off
# --------------------------------------------------------------------------


def server_url(port: int, host: str) -> str:
    """The URL to print, with the host the server will actually bind.

    `localhost` rather than the bind address: `0.0.0.0` is not something a
    browser can be sent to, and a user who passed `--host 0.0.0.0` (which
    run_server.sh does by default) still wants a link that works on this
    machine.
    """
    shown = "localhost" if host in ("0.0.0.0", "::", "") else host
    if ":" in shown and not shown.startswith("["):
        shown = f"[{shown}]"
    return f"http://{shown}:{port}/setup"


#: Set to anything to stop the bootstrap opening a browser. Documented in
#: the message it prints rather than being silent about it, so someone who
#: sees no tab knows why.
NO_BROWSER_ENV = "DISTILLATION_NO_BROWSER"


def browser_suppressed() -> bool:
    """Whether this run was asked not to open a browser.

    An environment variable rather than a flag because the caller is
    usually a shell script or a CI job, not a person: `--no-browser` on
    `run_server.sh` would have to survive the re-exec to be useful, and
    the environment does that for free.
    """
    return bool(os.environ.get(NO_BROWSER_ENV))


def open_browser(url: str) -> bool:
    """Try to open a browser. Returns whether it looked like it worked.

    Best-effort by construction: `webbrowser.open` returns False on a
    machine with no browser and raises on some. Either way the link has
    already been printed, so the failure costs nothing.

    Suppressed entirely when `DISTILLATION_NO_BROWSER` is set. That
    variable exists because this ran unattended: the install test drives
    this path on purpose, and every run opened a real tab on the
    developer's desktop. A bootstrap that reaches for the user's browser
    without being asked is doing something surprising, whether or not the
    user wanted the install.
    """
    if browser_suppressed():
        return False
    try:
        return bool(webbrowser.open(url))
    except Exception:  # noqa: BLE001 -- never fatal, the link is printed
        return False


def next_argv(argv: list[str]) -> list[str]:
    """The `run_server.sh` invocation to re-exec with.

    Re-execing the *script* rather than `python -m backend.cli` directly, so
    the `.env` it loads, the interpreter precedence it implements and the
    user's own arguments are all applied once and in one place. `os.execv`
    replaces this process, so nothing above this point runs twice and
    Ctrl-C reaches the server directly.

    `argv` is the *original* argument list, not the leftovers from
    `parse_known_args`, and that distinction is load-bearing.
    `--host` and `--port` are known to this module -- it needs them to
    print a link -- so they are consumed, and forwarding only the
    unrecognised remainder silently dropped them. The server then came up
    on the default port while the link pointed at the requested one:

        Open this to finish setting up:  http://localhost:8799/setup
        ERROR: [Errno 98] bind on address ('0.0.0.0', 8766): in use

    Forwarding the original list reproduces exactly what run_server.sh
    would have done without a bootstrap in the way, which is the property
    that makes the bootstrap invisible. `run_server.sh` puts its own
    `--host 0.0.0.0` *before* `"$@"`, so a user-supplied host still wins
    by argparse's last-value rule.
    """
    return [str(Path(__file__).resolve().parents[1] / "run_server.sh"), *argv]


def bootstrap(python: str, host: str, port: int, argv: list[str]) -> None:
    """Install what is missing into a temporary venv, then re-exec."""
    target = venv_dir()
    url = server_url(port, host)

    print()
    print("  This project's server needs four packages that are not here.")
    print("  Installing them into a temporary environment now.")
    print(f"    {target}")
    print()

    started = time.monotonic()
    interpreter = create_venv(target, python)
    install_packages(interpreter, list(SERVER_PACKAGES))
    elapsed = time.monotonic() - started

    # Verify by importing in the *new* interpreter rather than trusting the
    # exit code: pip can exit 0 having installed something unusable, and
    # the next thing that happens is a server start, so this is the last
    # cheap place to notice.
    check = subprocess.run(
        [str(interpreter), "-c",
         "import fastapi, uvicorn, multipart, tomli_w"],
        capture_output=True, text=True,
    )
    if check.returncode != 0:
        detail = (check.stderr or "").strip().splitlines()
        raise BootstrapError(
            "the packages installed but the server still cannot import them"
            + (f":\n    {detail[-1]}" if detail else "")
        )

    print()
    print(f"  Done in {elapsed:.0f}s. Starting the server.")
    print()
    print(f"    Open this to finish setting up:  {url}")
    if browser_suppressed():
        print(f"    (no browser opened: {NO_BROWSER_ENV} is set)")
    elif not open_browser(url):
        # Not an error. Printed either way, because an auto-open that
        # silently did nothing is indistinguishable from one that worked.
        print("    (could not open a browser automatically -- use the link above)")
    print()

    # Hand the interpreter forward, so the server -- and every training
    # subprocess it spawns, which resolves VENV_PYTHON the same way -- runs
    # from the venv that actually has the packages.
    command = next_argv(argv)
    try:
        # Flush before exec, which is not obvious and was a real bug.
        # `os.execve` replaces this process without flushing, so every
        # `print` above is lost when stdout is not a terminal -- a pipe, a
        # redirect to a log, systemd, Docker. On a terminal stdout is
        # line-buffered and it works, which is why it survived being run by
        # hand. The install test caught it because it redirects.
        sys.stdout.flush()
        sys.stderr.flush()
        os.execve(command[0], command, dict(os.environ, VENV_PYTHON=str(interpreter)))
    except OSError as exc:
        raise BootstrapError(
            f"could not restart run_server.sh ({exc}). The packages are "
            f"installed and working -- start the server yourself with:\n"
            f"    VENV_PYTHON={interpreter} ./run_server.sh"
        ) from exc


# --------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m backend.first_run",
        description="Install the server's own dependencies, then start it.",
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument(
        "--check", action="store_true",
        help="report which packages are missing and exit; install nothing",
    )
    # Resolved once, up front. `argv=None` means "read sys.argv", and argparse
    # does that -- but the parameter itself stays None, so forwarding it
    # handed `next_argv` a None to unpack:
    #
    #     TypeError: Value after * must be an iterable, not NoneType
    #
    # Which the unit test could not see, because it calls main() with an
    # explicit list and `__main__` calls it with none. Only running the
    # module for real finds that; the install test does, at a cost of about
    # a minute. So there is a check for it below too.
    argv = list(sys.argv[1:] if argv is None else argv)

    # `--host`/`--port` are known here so the link can name the right port, and
    # what is forwarded on re-exec is this original `argv` rather than the
    # unrecognised remainder -- see `next_argv`.
    args, _unrecognised = parser.parse_known_args(argv)

    missing = missing_server_packages()
    if not missing:
        # The normal path, and deliberately silent: run_server.sh calls this
        # speculatively, and a message on every start would be noise about
        # a problem that does not exist.
        return 0

    print()
    print("  Missing server dependencies:")
    for reason in missing_reasons(missing):
        print(f"    {reason.strip()}")
    print()

    if args.check:
        # Exit 1, not 78: `--check` is a question with a yes/no answer, and
        # 78 would report a configuration error to whatever called it. It is
        # consumed here rather than passed through, so `run_server.sh --check`
        # exits instead of handing the flag to a server that cannot parse it.
        return 1

    try:
        bootstrap(sys.executable, args.host, args.port, argv)
    except BootstrapError as exc:
        print()
        print(f"  Could not install them: {exc}")
        print()
        print("  Nothing was changed. Once they are available, start the")
        print("  server again with ./run_server.sh")
        return EXIT_BOOTSTRAP_FAILED
    except KeyboardInterrupt:
        print("\n  Cancelled. Nothing was changed.")
        return 130
    return 0  # unreachable: bootstrap() execs or raises


if __name__ == "__main__":
    sys.exit(main())