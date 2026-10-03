"""ComfyUI's environment port -- a venv this process is not running in.

The installer wants to install four packages into ComfyUI's virtualenv, and
to do that safely it has to read two things about a venv it is not running
inside: what ComfyUI *declares* it needs (`requirements.txt`), and what is
*actually installed* in the venv.

**These are two sources because they answer two questions**, and they can
disagree -- which is the interesting case rather than an edge case. A user
upgrades one and not the other, edits the file by hand, or installs a
package directly; then the file is not a description of the venv and
trusting either alone gets it wrong in a different way. Design doc §3.

    declared + installed matches   known-safe, and it is what ComfyUI wants
    declared + installed differs   a conflict: ComfyUI's own declaration
                                   is already broken. Refuse, and say so --
                                   the user needs to know that before
                                   anything is written into that venv
    installed + not declared      *unknown*. Installed is a fact about a
                                   machine; a declaration is a contract the
                                   publisher made. An undeclared package
                                   has nothing vouching for it, so it gets
                                   an exact pin rather than a range

That third row is the user's correction to a first draft which treated
"installed" as evidence of a working setup. It is the absence of evidence,
and the safe reading of it is the strict one.

**Everything is a failure mode, never an exception.** Reading another
interpreter's installed packages means running a subprocess in it, and
that can fail in several ways that are all *answers*: no such venv, no
Python in it, a venv so old it has no `importlib.metadata`, a package whose
metadata is corrupt. All of them mean "we cannot prove this is safe", which
is a reason to refuse rather than a reason to guess.
"""

from __future__ import annotations

import subprocess
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass

#: How a distribution's name is normalised before comparing two sources.
#: `Pillow` and `pillow` are the same package, and requirements files are
#: written by hand, so they disagree about capitalisation and about
#: underscores constantly.
def normalise(name: str) -> str:
    return name.strip().lower().replace("_", "-")


@dataclass(frozen=True, slots=True)
class Declaration:
    """One line of ComfyUI's `requirements.txt`.

    `specifier` is the PEP 440 specifier text (`>=4.50.3`), empty when the
    line pins nothing (`torch`). `marker` is an environment marker when the
    line has one, and it is kept rather than discarded: a declaration that
    only applies on Windows says nothing about this machine, and treating it
    as a constraint here would refuse an install for a package that was
    never going to be installed on Linux.
    """

    name: str
    specifier: str = ""
    extras: tuple[str, ...] = ()
    marker: str | None = None

    @property
    def key(self) -> str:
        return normalise(self.name)


@dataclass(frozen=True, slots=True)
class ComfyEnvironmentInfo:
    """Both sources, read. Missing either is `None`, not an exception."""

    comfy_dir: str
    requirements_path: str | None = None
    declarations: tuple[Declaration, ...] = ()
    installed: tuple[tuple[str, str], ...] = ()
    #: The interpreter the inventory actually came from, so the report can
    #: name it. Set even on failure, because "which venv did you try" is
    #: the first question when the answer is no.
    venv_python: str | None = None
    #: Why a source could not be read, in the user's terms.
    requirements_error: str | None = None
    venv_error: str | None = None

    @property
    def declarations_by_key(self) -> dict[str, Declaration]:
        return {d.key: d for d in self.declarations}

    @property
    def installed_by_key(self) -> dict[str, str]:
        return {normalise(n): v for n, v in self.installed}


class ComfyEnvironment(ABC):
    """Read ComfyUI's declarations and its venv. Never writes."""

    @abstractmethod
    def read(self, comfy_dir: str, venv_python: str | None = None) -> ComfyEnvironmentInfo:
        """Both sources, for one directory.

        `venv_python` is an argument rather than only construction state
        because it is a *setting* the wizard sets and the user can change
        later. A port that captured the interpreter at wiring time would
        check a different venv than the one the machine is configured to
        use, and it would do so silently -- the answer would be about the
        right kind of thing and the wrong one.
        """
        raise NotImplementedError


@dataclass(slots=True)
class LocalComfyEnvironment(ComfyEnvironment):
    """The real one: the checkout on this machine and the venv beside it.

    `venv_python` is a *fallback*, used only when the caller passes nothing.
    When set it is usually the server's own interpreter, which is the right
    thing to inspect when this project is running inside ComfyUI's venv --
    that is, when there is no separate "other" venv to check.
    """

    venv_python: str | None = None
    timeout: float = 60.0

    #: Runs in ComfyUI's venv. Prints one JSON line of ``[name, version]``
    #: pairs; anything else is a failure rather than an answer.
    #:
    #: ``--no-user-site`` is not a flag here but a `-I`: a user-site install
    #: on this machine would otherwise be attributed to ComfyUI's venv, and
    #: then we would pin a package that belongs to neither.
    _INVENTORY_CHILD = """
import json
try:
    import importlib.metadata as md
except ImportError:
    print(json.dumps({"ok": False, "reason": "this interpreter has no "
                      "importlib.metadata (Python 3.7 or older)"}))
    raise SystemExit(0)
out = []
seen = set()
for dist in md.distributions():
    try:
        name = dist.metadata["Name"]
        version = dist.version
    except Exception:
        continue
    if not name:
        continue
    key = name.lower()
    if key in seen:
        continue
    seen.add(key)
    out.append([name, version or ""])
print(json.dumps({"ok": True, "packages": out}))
"""

    def read(self, comfy_dir: str, venv_python: str | None = None) -> ComfyEnvironmentInfo:
        declarations, path, req_error = self._read_requirements(comfy_dir)
        installed, venv_error = self._read_venv(venv_python or self.venv_python)
        return ComfyEnvironmentInfo(
            comfy_dir=comfy_dir,
            requirements_path=path,
            declarations=declarations,
            installed=installed,
            requirements_error=req_error,
            venv_error=venv_error,
            venv_python=venv_python or self.venv_python,
        )

    # -- source 1: the declaration file ---------------------------------

    def _read_requirements(
        self, comfy_dir: str
    ) -> tuple[tuple[Declaration, ...], str | None, str | None]:
        from pathlib import Path

        try:
            path = Path(comfy_dir) / "requirements.txt"
        except TypeError:
            return (), None, "the ComfyUI directory is not a usable path"

        if not path.is_file():
            return (), None, (
                f"there is no requirements.txt in {path.parent} -- so there "
                f"is nothing that declares what ComfyUI needs, and an "
                f"install cannot be shown to be safe"
            )

        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            return (), str(path), f"requirements.txt could not be read: {exc}"

        declarations: list[Declaration] = []
        seen: set[str] = set()
        unparsed: list[str] = []
        for raw in text.splitlines():
            line = raw.split("#")[0].strip()
            if not line or line.startswith("-"):
                continue
            declaration = self._parse_line(line)
            if declaration is None:
                unparsed.append(line)
                continue
            if declaration.key in seen:
                # A duplicate declaration in a requirements file is either a
                # mistake or two lines meant to intersect. pip takes the
                # union; taking the first silently would make the check
                # weaker than pip's own resolution without saying so.
                continue
            seen.add(declaration.key)
            declarations.append(declaration)

        error = None
        if unparsed:
            error = (
                f"{len(unparsed)} line(s) of {path.name} could not be parsed, "
                f"starting with {unparsed[0]!r} -- a declaration this check "
                f"cannot read is a declaration it cannot clear"
            )
        return tuple(declarations), str(path), error

    def _parse_line(self, line: str) -> Declaration | None:
        """One requirements line into a `Declaration`, or None.

        `packaging` does the parsing. It is in `requirements.txt` for
        exactly this: PEP 508 grammar has more corners than a regex
        survives, and a check built on a hand-rolled regex would quietly
        mis-read some line and report "safe" for it.
        """
        from packaging.requirements import InvalidRequirement, Requirement

        try:
            parsed = Requirement(line)
        except InvalidRequirement:
            return None
        return Declaration(
            name=parsed.name,
            specifier=str(parsed.specifier),
            extras=tuple(sorted(parsed.extras)),
            marker=str(parsed.marker) if parsed.marker else None,
        )

    # -- source 2: what is actually there -------------------------------

    def _read_venv(self, interpreter: str | None) -> tuple[tuple[tuple[str, str], ...], str | None]:
        import json

        if not interpreter:
            return (), (
                "ComfyUI's virtualenv interpreter is not known -- set "
                "VENV_PYTHON in .env, or on the settings page, so there is "
                "something to check"
            )

        try:
            result = subprocess.run(
                [interpreter, "-I", "-c", self._INVENTORY_CHILD],
                capture_output=True, text=True, timeout=self.timeout,
            )
        except subprocess.TimeoutExpired:
            return (), (
                f"reading {interpreter} did not finish in "
                f"{self.timeout:.0f}s"
            )
        except OSError as exc:
            return (), f"{interpreter} could not be run: {exc}"

        # Backwards scan for a JSON line, because this interpreter may print
        # a banner on stdout on the way (the same reason the device probe
        # scans rather than parsing the whole stream).
        for line in reversed(result.stdout.splitlines()):
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                payload = json.loads(line)
            except ValueError:
                continue
            if not isinstance(payload, dict) or "ok" not in payload:
                continue
            if not payload["ok"]:
                return (), payload.get("reason") or "the venv gave no answer"
            packages = payload.get("packages") or []
            return tuple(
                (str(name), str(version)) for name, version in packages
            ), None

        if result.returncode != 0:
            return (), (
                f"{interpreter} exited {result.returncode}: "
                f"{(result.stderr or '').strip()[-200:]}"
            )
        return (), f"{interpreter} printed no package list"

    def _read_venv_in_process(self) -> tuple[tuple[tuple[str, str], ...], str | None]:
        """For tests: read *this* interpreter instead of a subprocess."""
        import json

        result = subprocess.run(
            [sys.executable, "-I", "-c", self._INVENTORY_CHILD],
            capture_output=True, text=True, timeout=self.timeout,
        )
        payload = json.loads(result.stdout.splitlines()[-1])
        return (
            tuple((str(n), str(v)) for n, v in payload["packages"]), None,
        ) if payload.get("ok") else ((), payload.get("reason"))