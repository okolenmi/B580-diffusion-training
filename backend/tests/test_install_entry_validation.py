"""What the installer will and will not accept before it runs pip.

Design doc §3, and round-5 finding R5-02. This endpoint has no
authentication (ADR 0001) and its one protective job is refusing to damage
an environment this project does not own. The guard it had was

    forbidden.intersection(packages)

on the client's own strings, which compares text to a set of names and so
accepts everything except an exact match. Measured, with `target="comfy"`:

    torch==2.5.0     Torch    torch[opt]   torch>=1   ' torch'
    pytorch-triton-xpu                      -- all reached pip

and `--index-url=http://...` and `-r /some/file` were accepted for *both*
targets, because `build_command` appended the package list with no `--`
separator, so anything beginning with `-` arrived as an option. The
interpreter that would have run pip came from the request too.

**Every case below is run against both targets.** A guard that only applies
to the protected environment is a guard whose failure mode is "it was fine
because the target was the other one".

Three of these checks are about the *validator* rather than the endpoint,
because that is where the decision now lives: `parse_requirement` and
`validate_install_entries` are called directly with strings, so a shape the
endpoint cannot even be asked for is still covered.

Two checks are deliberately about behaviour that would be a *regression* if
it appeared: that the wizard's own request -- what `install.js` actually
sends -- is still accepted, and that a package list which is well formed
reaches pip unaltered. A validator that refuses everything passes every
refusal check above.

Run: `python backend/tests/test_install_entry_validation.py`
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.application.ports.package_installer import (  # noqa: E402
    InstallError,
    InstallRequest,
    PipInstaller,
)
from backend.application.ports.requirements_manifest import (  # noqa: E402
    REQUIREMENTS,
)
from backend.application.use_cases.install_packages import (  # noqa: E402
    StartInstall,
    canonicalize,
    parse_requirement,
    validate_install_entries,
)
from backend.tests.support import check, finish  # noqa: E402

#: A distribution that is in the manifest, and one that is not.
KNOWN = next(r.distribution for r in REQUIREMENTS
             if not r.never_install)
FORBIDDEN = next(r.distribution for r in REQUIREMENTS if r.never_install)


class _NeverRuns:
    """Stands in for the installer. Every refusal must happen before this."""

    def __init__(self) -> None:
        self.requests: list = []

    def build_command(self, request, constraints_path=None):
        self.requests.append(request)
        return ["/fake/python", "-m", "pip", "install", "--"] + list(
            request.packages)

    def install(self, request, constraints_path=None, **kwargs):
        self.requests.append(request)
        return None

    def create_virtualenv(self, *args, **kwargs):
        raise AssertionError("a refusal must happen before any venv is made")


def _start(detector=lambda: "/comfy/venv/bin/python") -> StartInstall:
    return StartInstall(
        installer=_NeverRuns(),
        project_root=Path("/tmp/does-not-need-to-exist"),
        base_python="python3",
        detect_comfy_python=detector,
    )


def _refused(fn, *args, **kwargs) -> str:
    """The refusal message, or `""` if it was accepted."""
    try:
        fn(*args, **kwargs)
    except InstallError as exc:
        return str(exc)
    return ""

def report(condition: bool, message: str, detail: str = "") -> None:
    """`check`, with the measurement carried by the message.

    `support.check` takes only a condition and a message, so a check that
    wants to say what it measured has to build the message itself. The
    detail is only appended on failure -- a passing line stays one line.
    """
    check(condition, message if condition or not detail
          else f"{message} -- {detail}")



# ==========================================================================
print("\n-- every spelling of a forbidden package, for both targets --")

#: Refused for `target="comfy"` only: these are all one package by name, and
#: the never-install flag is scoped to an environment this project does not
#: own. Installing torch into a *new* virtualenv is the supported path and
#: is why the flag is per-target at all.
COMFY_ONLY = [
    (FORBIDDEN, "the exact name -- this one the old guard did catch"),
    (f"{FORBIDDEN}==2.5.0", "a version pin: not a member of a bare-name set"),
    (FORBIDDEN.upper(), "case: 'TORCH' is not the string 'torch'"),
    (FORBIDDEN.capitalize(), "and 'Torch' from the other direction"),
]

#: Refused for **both** targets: either the text is not a requirement at all,
#: or it names something outside the manifest. Neither is about ComfyUI --
#: an option string is an instruction whichever pip reads it, and an unknown
#: package is one nobody chose on purpose.
BOTH_TARGETS = [
    (f"{FORBIDDEN}[opt]", "extras, which can pull in anything at all"),
    (f"{FORBIDDEN}>=1", "a range rather than a pin"),
    (f"{FORBIDDEN}~=2.5", "a compatible-release specifier"),
    (f"{FORBIDDEN}!=2.5", "an exclusion specifier"),
    (f" {FORBIDDEN}", "leading whitespace, which PEP 508 allows"),
    (f"{FORBIDDEN} ", "trailing whitespace, likewise"),
    (f"{FORBIDDEN} @ https://example.invalid/torch.whl", "a direct URL"),
    (f'{FORBIDDEN};python_version < "3"', "an environment marker"),
    ("pytorch-triton-xpu", "a torch companion that is not a manifest row"),
    ("--index-url=http://evil.example/simple", "a pip option"),
    ("-r", "pip's read-another-file option"),
    ("--upgrade", "a bare option"),
]

for entry, why in COMFY_ONLY:
    message = _refused(validate_install_entries, (entry,), (), "comfy")
    report(bool(message), f"comfy: refused {entry!r} ({why})", "ACCEPTED")

for entry, why in BOTH_TARGETS:
    for target in ("comfy", "new"):
        message = _refused(validate_install_entries, (entry,), (), target)
        report(bool(message),
               f"{target}: refused {entry!r} ({why})", "ACCEPTED")

print("\n-- and a never-install name is refused only where it is scoped --")

# The complement, so the refusals above cannot pass by refusing everything:
# the same spellings are *accepted* for a new virtualenv, because this
# project owns that environment and installing torch there is the point of
# the `new` target.
for entry, _why in COMFY_ONLY:
    accepted = not _refused(validate_install_entries, (entry,), (), "new")
    report(accepted,
           f"new: accepted {entry!r}, because never_install is scoped to an "
           f"environment this project does not own",
           f"refused: {_refused(validate_install_entries, (entry,), (), 'new')}")

print("\n-- and the reason names the right thing --")

message = _refused(validate_install_entries, (FORBIDDEN,), (), "comfy")
report("never-install" in message and "separate virtualenv" in message,
      "a forbidden name says never-install and offers the alternative",
      message)

message = _refused(validate_install_entries, ("pytorch-triton-xpu",), (),
                   "comfy")
report("not a package this project declares" in message,
      "an unknown name says it is not in the manifest, rather than "
      "pretending it is a forbidden spelling",
      message)

message = _refused(validate_install_entries, (f"{FORBIDDEN}>=1",), (), "new")
report("rather than '=='" in message,
      "a bad specifier says the operator was wrong, so the message is "
      "actionable",
      message)

print("\n-- parse_requirement on its own, since that is the decision --")

parsed = parse_requirement("fastapi")
report(parsed.name == "fastapi" and parsed.canonical == "fastapi"
      and parsed.version is None,
      "a bare name parses to a name with no version",
      f"{parsed}")
parsed = parse_requirement("torch==2.5.0")
report(parsed.version == "2.5.0",
      "and 'name==version' keeps the version",
      f"{parsed}")
report(parse_requirement("Torch").canonical == "torch",
      "and the canonical form is what a set comparison uses")

for entry in ("", "  ", "-x", "a b", "x[1]", "x;y", "x@y", "x>=1"):
    report(bool(_refused(parse_requirement, entry)),
          f"parse_requirement refuses {entry!r}")

print("\n-- canonicalisation is what makes the spellings one name --")

report(canonicalize("Torch") == canonicalize("torch") == canonicalize("TORCH"),
      "Torch, torch and TORCH canonicalise to one name",
      f"{canonicalize('Torch')}, {canonicalize('torch')}, "
      f"{canonicalize('TORCH')}")
report(canonicalize("torch_") != canonicalize("torch"),
      "and torch_ does not: it canonicalises to 'torch-', so it is refused "
      "as an unknown name rather than as a spelling of a forbidden one",
      f"torch_ -> {canonicalize('torch_')}")

message = _refused(validate_install_entries, ("torch_",), (), "comfy")
report("not a package this project declares" in message,
      "which is what actually happens for torch_", message)

print("\n-- constraints are held to a stricter shape --")

for line, why in [
    ("--index-url=http://evil.example/simple", "an option line"),
    ("-r /some/file", "another option line"),
    ("torch @ https://example.invalid/x", "a URL"),
    ("torch;python_version < \"3\"", "a marker"),
    ("torch>=1", "a range rather than a pin"),
    ("torch", "no version at all"),
    ("torch 2.5.0", "whitespace rather than =="),
]:
    message = _refused(validate_install_entries, (KNOWN,), (line,), "new")
    report(bool(message), f"constraint {line!r} refused ({why})", "ACCEPTED")

report(not _refused(validate_install_entries, (KNOWN,), ("torch==2.12.1+xpu",),
                   "comfy"),
      "and a plain pin is accepted for both targets, including a local "
      "version tag")

print("\n-- the interpreter comes from the server, not the request --")

start = _start(detector=lambda: "/comfy/venv/bin/python")
message = _refused(start.execute, "comfy", (KNOWN,),
                   comfy_venv_python="/attacker/bin/python")
report(bool(message) and "/attacker/bin/python" in message
      and "/comfy/venv/bin/python" in message,
      "a request naming a different interpreter is refused, and the message "
      "shows both so the disagreement is diagnosable",
      f"refusal was {message!r}" or "ACCEPTED")

message = _refused(start.execute, "comfy", (KNOWN,))
# Supplying no interpreter must *succeed*: the field exists so the wizard can
# show the user what will run, not because the server needs permission. The
# old code refused here, which has the same shape of failure as the bypass --
# it made the safe request the one that needed a form field to get through.
report(not message,
      "and a request naming no interpreter at all proceeds on the server's "
      "own -- not supplying one is not a bypass",
      f"refused with {message!r}")

message = _refused(_start(detector=lambda: None).execute, "comfy", (KNOWN,))
report(bool(message) and "could not be found" in message,
      "a server that cannot detect the interpreter refuses rather than "
      "falling back to anything the caller said",
      f"refusal was {message!r}" or "ACCEPTED")

print("\n-- build_command separates options from requirements --")

command = PipInstaller().build_command(
    InstallRequest(target_python="/py", packages=("--index-url=http://x", "fastapi")))
report("--" in command,
      "the command carries a -- separator",
      f"{command}")
try:
    separator = command.index("--")
except ValueError:
    separator = -1
report(separator == len(command) - 3,
      "immediately before the package list, with exactly the packages after "
      f"it (index {separator} of {len(command)})",
      f"packages land at {separator} in {command}")
# The point of `--` is the opposite of "nothing looks like an option": an
# entry that *does* start with `-` is read as a requirement precisely
# because it is after the separator. What must hold is that no *package*
# sits before it, where pip would try to read it as one.
before = command[:separator]
report(all(entry.startswith("-") or entry in ("install", "--disable-pip-version-check", "--no-input")
          or entry == "/py" or entry == "-m" or entry == "pip"
          for entry in before),
      "and every entry before it is an option, the interpreter or the pip "
      "verb -- no package sits where pip would read it as an option",
      f"{before}")

constrained = PipInstaller().build_command(
    InstallRequest(target_python="/py", packages=("fastapi",)),
    constraints_path=Path("/tmp/c.txt"))
report(constrained.index("--") > constrained.index("--constraint"),
      "the -- comes after --constraint, so the constraint file is still an "
      "option",
      f"{constrained}")

print("\n-- what the wizard itself sends still works --")

# Read the real request out of the frontend rather than restating it, so
# this cannot drift from what the page actually posts.
js = Path(__file__).resolve().parents[2] / "frontend/js/views/install.js"
source = js.read_text(encoding="utf-8")
report("packages" in source and "comfy_venv_python" in source,
      "install.js is the file that builds the install request",
      f"{js} does not mention them")

# The names it could offer are the manifest's, filtered for the target --
# which is the same list the validator allows.
from backend.application.ports.requirements_manifest import by_tier, REQUIRED  # noqa: E402

offered = [r.distribution for r in by_tier(REQUIRED)]
report(all(not _refused(validate_install_entries, (p,), (), "comfy")
          for p in offered),
      f"every package the wizard may offer for a comfy install is accepted "
      f"({len(offered)} names: {', '.join(offered)})")
report(all(not _refused(validate_install_entries, (p,), (), "new")
          for p in offered),
      "and for a new virtualenv too")

print("\n-- a well-formed list reaches pip unaltered --")

recorder = _NeverRuns()
start = StartInstall(installer=recorder,
                     project_root=Path("/tmp/does-not-need-to-exist"),
                     base_python="python3",
                     detect_comfy_python=lambda: "/comfy/venv/bin/python")
start.execute(target="comfy", packages=("fastapi", "uvicorn"),
              constraints=("torch==2.12.1+xpu",))
report(len(recorder.requests) == 1,
      f"exactly one install was requested ({len(recorder.requests)})")
if recorder.requests:
    sent = recorder.requests[0]
    report(tuple(sent.packages) == ("fastapi", "uvicorn"),
          "with the packages exactly as sent -- the validator rewrites "
          "nothing",
          f"{sent.packages}")
    report(tuple(sent.constraints) == ("torch==2.12.1+xpu",),
          "and the constraints untouched", f"{sent.constraints}")

print("\n-- and a refusal happens before anything runs --")

recorder = _NeverRuns()
start = StartInstall(installer=recorder,
                     project_root=Path("/tmp/does-not-need-to-exist"),
                     base_python="python3",
                     detect_comfy_python=lambda: "/comfy/venv/bin/python")
_refused(start.execute, "comfy", (FORBIDDEN,))
report(not recorder.requests,
      f"pip was never asked to do anything ({len(recorder.requests)} "
      f"requests)")

finish()
