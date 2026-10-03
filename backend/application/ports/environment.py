"""Environment port -- what this machine can actually do.

Round-3 follow-on to the first-run installer work
(`docs/design/11-first-run-and-installer.md`). The installer's whole value
is turning "the server started, so something must be fine" into an
answerable question, and that needs three facts the server does not
otherwise have:

* **which packages are present** -- and that has to be answerable *without*
  importing them, because the answer to "is torch installed" must not be
  "I imported it and used 1.5 GB";
* **what the accelerator can do** -- which needs torch, so it is a separate
  question from the first;
* **which interpreter a child would run under**, for the same reason.

Two ports rather than one, because they fail differently and the
difference is the point. `PackageInventory` is cheap, total, and never
raises -- a missing package is an answer, not a fault. `DeviceProbe`
imports torch, which can fail in several ways that are all *data* (not
installed, no such backend, no card), so it returns a result object rather
than raising.

`pip` is reached through `Installer` rather than here: what is installed
and what could be installed are different questions, and only the second
one has a subprocess in it.
"""

from __future__ import annotations

import subprocess
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

#: Distribution name -> the name code imports, where they differ. Keyed by
#: what `importlib.metadata` reports, because that is the only direction
#: the inventory ever needs: it looks up by distribution name.
#:
#: The reverse mapping below is derived from this one rather than written
#: out again, because a hand-maintained pair of tables is a table that
#: disagrees with itself. (It did, briefly: this map was originally keyed
#: the other way round, so `python-multipart` reported its import name as
#: "python-multipart" -- a package the code does not import under that
#: name, reported correctly by the check that was supposed to catch it
#: only because it compared against the same wrong map.)
IMPORT_NAMES: dict[str, str] = {
    "python-multipart": "multipart",
    "pillow": "PIL",
    "scikit-learn": "sklearn",
    "opencv-python": "cv2",
}


@dataclass(frozen=True, slots=True)
class InstalledPackage:
    """One distribution found in the running interpreter.

    `distribution` is the pip name (`python-multipart`) and `import_name`
    is what the code imports (`multipart`) -- they are not the same string,
    and guessing wrong is how a dependency check reports a package missing
    on a machine that has it.
    """

    distribution: str
    import_name: str
    version: str


class PackageInventory(ABC):
    """What is installed, read without importing anything."""

    @abstractmethod
    def installed(self) -> tuple[InstalledPackage, ...]:
        """Every distribution visible to the running interpreter.

        Total by contract: a package that cannot be read is absent from
        the answer rather than an exception, because "I could not tell"
        and "it is not there" lead to the same next step here -- install it
        -- and one of them being an error would mean a single unreadable
        distribution makes the whole installer unavailable.
        """
        raise NotImplementedError

    @abstractmethod
    def version_of(self, distribution: str) -> str | None:
        """One package's version, or None if it is not installed."""
        raise NotImplementedError

    def import_name_for(self, distribution: str) -> str:
        """The name code would import, given a distribution name."""
        return IMPORT_NAMES.get(distribution, distribution)

    def distribution_for(self, import_name: str) -> str:
        """The pip name, given the name code imports. Derived, not stored."""
        for dist, mod in IMPORT_NAMES.items():
            if mod == import_name:
                return dist
        return import_name


class MetadataPackageInventory(PackageInventory):
    """`importlib.metadata` -- reads dist-info, imports nothing.

    Verified on this machine that asking for torch's version does not put
    torch in `sys.modules`, which is the property the whole design rests
    on: the server must be able to report a missing torch without having
    loaded one.
    """

    def installed(self) -> tuple[InstalledPackage, ...]:
        import importlib.metadata as metadata

        found: list[InstalledPackage] = []
        seen: set[str] = set()
        for dist in metadata.distributions():
            name = dist.metadata["Name"]
            if not name or name.lower() in seen:
                continue
            seen.add(name.lower())
            found.append(
                InstalledPackage(
                    distribution=name,
                    import_name=self.import_name_for(name),
                    version=dist.version or "",
                )
            )
        return tuple(sorted(found, key=lambda p: p.distribution.lower()))

    def version_of(self, distribution: str) -> str | None:
        import importlib.metadata as metadata

        try:
            return metadata.version(distribution)
        except metadata.PackageNotFoundError:
            return None


@dataclass(frozen=True, slots=True)
class DeviceReport:
    """What the accelerator turned out to be.

    Every field is optional because every one of them can be unknown:
    torch may be absent, may lack the `xpu` attribute, may report no
    device, and the property query may raise on a driver this torch build
    does not understand. `present` is the one that decides whether a run
    can start; the rest is detail for the user who has just been told no.
    """

    present: bool
    backend: str | None = None
    name: str | None = None
    total_memory_mb: float | None = None
    #: Why it is not present, in the user's terms. None when it is.
    reason: str | None = None
    #: Set when the probe itself could not run, as distinct from running
    #: and finding nothing. The installer says different things about
    #: those two.
    detail: str | None = None


class DeviceProbe(ABC):
    """Ask the accelerator what it is. Imports torch, so it can fail."""

    @abstractmethod
    def report(self) -> DeviceReport:
        raise NotImplementedError


@dataclass(slots=True)
class TorchDeviceProbe(DeviceProbe):
    """The real probe: torch's own XPU or CUDA reporting.

    Runs in a subprocess by default. That is not defensiveness for its own
    sake -- importing torch in the server process costs seconds and
    gigabytes, and the installer's whole job is to be answerable *before*
    the user has committed to a multi-gigabyte download. Paying that cost
    in a throwaway process also means a torch build that crashes on import
    cannot take the server with it, which is the same reason the graph
    execution child exists.
    """

    backend: str = "xpu"
    timeout: float = 60.0
    _in_process: bool = field(default=False, repr=False)

    #: Runs in the child. Prints one JSON line; anything else is a probe
    #: failure rather than an answer.
    #:
    #: The backend is substituted in rather than read from argv: this same
    #: source is also exec'd in-process, where there is no argv[1] and the
    #: first version raised IndexError before asking the question at all.
    #:
    #: Substitution is a plain token replacement, not str.format: the
    #: source is dense with JSON and dict braces, so a format template
    #: would need every literal brace doubled and the first version raised
    #: KeyError('"ok"') on the first line it touched.
    _CHILD_TEMPLATE = """
import json, sys
try:
    import torch
except Exception as exc:
    print(json.dumps({"ok": False, "reason": "torch is not importable in this "
                         "interpreter (%s: %s)" % (type(exc).__name__, exc)}))
    raise SystemExit(0)
backend = __BACKEND__
module = getattr(torch, backend, None)
if module is None:
    print(json.dumps({"ok": False, "reason": "this torch build has no %r "
                         "backend" % backend}))
    raise SystemExit(0)
try:
    available = bool(module.is_available())
except Exception as exc:
    print(json.dumps({"ok": False, "reason": "%s.is_available() raised %s: %s"
                         % (backend, type(exc).__name__, exc)}))
    raise SystemExit(0)
if not available:
    print(json.dumps({"ok": False, "reason": "torch reports no %s device"
                         % backend}))
    raise SystemExit(0)
out = {"ok": True, "backend": backend, "name": None, "total_memory_mb": None}
try:
    props = module.get_device_properties(module.current_device())
    out["name"] = getattr(props, "name", None)
    total = getattr(props, "total_memory", None)
    if total:
        out["total_memory_mb"] = round(total / (1024 ** 2))
except Exception as exc:
    # A visible device whose properties cannot be read is still a visible
    # device. Reporting it as absent would tell the user their card is
    # missing when it is answering, just not about itself.
    out["detail"] = "device is present but its properties could not be read "\\
                    "(%s: %s)" % (type(exc).__name__, exc)
print(json.dumps(out))
"""

    @classmethod
    def _child_source(cls, backend: str) -> str:
        # __BACKEND__ is replaced with a repr, so the result is valid
        # Python whatever the backend string contains.
        return cls._CHILD_TEMPLATE.replace("__BACKEND__", repr(backend))

    def report(self) -> DeviceReport:
        if self._in_process:
            return self._report_here()
        try:
            result = subprocess.run(
                [sys.executable, "-c", self._child_source(self.backend)],
                capture_output=True, text=True, timeout=self.timeout,
            )
        except subprocess.TimeoutExpired:
            return DeviceReport(
                present=False,
                reason="the device probe did not finish in time",
                detail=f"importing torch and asking about {self.backend} "
                       f"took longer than {self.timeout:.0f}s",
            )
        except OSError as exc:
            return DeviceReport(
                present=False,
                reason="the device probe could not be started",
                detail=str(exc),
            )
        if result.returncode != 0:
            return DeviceReport(
                present=False,
                reason="the device probe failed",
                detail=(result.stderr or "").strip()[-400:] or
                        f"exit status {result.returncode}",
            )
        return self._parse(result.stdout)

    def _parse(self, output: str) -> DeviceReport:
        """The child's one JSON line, out of whatever else it printed.

        Scans backwards for a line that parses, because the XPU runtime
        prints a Rusticl/Mesa banner to stdout on this machine and a
        parser that expected the whole stream to be JSON would report a
        working card as a broken probe. The banner is also why this is
        shared by both paths rather than written twice.
        """
        import json

        for line in reversed(output.splitlines()):
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
                return DeviceReport(
                    present=False,
                    reason=payload.get("reason") or "no device",
                    detail=payload.get("detail"),
                )
            return DeviceReport(
                present=True,
                backend=payload.get("backend"),
                name=payload.get("name"),
                total_memory_mb=payload.get("total_memory_mb"),
                detail=payload.get("detail"),
            )
        return DeviceReport(
            present=False,
            reason="the device probe produced no answer",
            detail=output.strip()[-400:],
        )

    def _report_here(self) -> DeviceReport:
        """Same question, same interpreter, with stdout captured.

        For tests, and for a caller that has already paid the import cost.

        `print` inside `exec` writes to the *process's* stdout, so the
        answer has to be captured rather than read out of the exec scope --
        the first version of this looked for it in the scope and therefore
        never found it, reporting "the device probe raised
        IndexError" on a machine with a perfectly good B580 in it.
        """
        import contextlib
        import io

        buffer = io.StringIO()
        try:
            with contextlib.redirect_stdout(buffer):
                exec(  # noqa: S102 -- our own source, with the backend substituted
                    compile(self._child_source(self.backend),
                            "<device-probe>", "exec"),
                    {"__name__": "__probe__"},
                )
        except SystemExit:
            pass
        except Exception as exc:  # noqa: BLE001 -- reported, not raised
            return DeviceReport(
                present=False, reason="the device probe raised",
                detail=f"{type(exc).__name__}: {exc}",
            )
        return self._parse(buffer.getvalue())


@dataclass(slots=True)
class InterpreterInfo:
    """Which interpreter this is, and whether it is the one to use.

    `is_expected` is the floor check from `backend/python_floor.py`,
    reported rather than enforced: the installer's job is to *tell* the
    user, and the entry points already refuse to start. Having the
    installer able to say "your interpreter is too old, and here is what it
    is" is the difference between a wizard and a traceback.
    """

    executable: str
    version: str
    prefix: str
    is_virtualenv: bool
    is_expected: bool
    floor: str