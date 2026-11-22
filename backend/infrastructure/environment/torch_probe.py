"""The torch device probe: a throwaway subprocess by default, so the server never pays the torch import."""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

from ...application.ports.system_cards import SystemCards
from ...application.ports.environment import DeviceProbe, DeviceReport

def _last_json_object(output: str) -> dict | None:
    """The last line of `output` that parses as a JSON object, or None.

    Shared by every probe child rather than written per-probe, because all
    of them have the same problem and the same reason for it: on this
    machine the XPU runtime prints a Rusticl/Mesa banner to stdout before
    the answer, so a probe that parsed the whole stream would report a
    working card as a broken probe. Scanning backwards finds the answer and
    ignores the banner.

    Returns None rather than raising, because "no answer" is a result every
    caller has to handle anyway.
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
        if isinstance(payload, dict) and "ok" in payload:
            return payload
    return None


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
    #: Names the cards by what the system lists when torch is not installed yet.
    system_cards: SystemCards = field(default_factory=SystemCards, repr=False)

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

    #: Runs in the child for `devices()`. Separate from `_CHILD_TEMPLATE`
    #: because that one is built around `current_device()` and answers a
    #: yes/no question; this one enumerates and answers which.
    #:
    #: A per-device failure becomes one row carrying `ok: False` rather than
    #: an exception: a machine where device 1 answers and device 2 does not
    #: is a real configuration, and abandoning the whole answer would leave
    #: a user with no way to pick the card that works.
    _LIST_CHILD_TEMPLATE = """
import json
try:
    import torch
except Exception as exc:
    print(json.dumps({"ok": False, "torch_missing": isinstance(exc, ImportError), "reason": "torch is not importable in "
                         "this interpreter (%s: %s)" % (type(exc).__name__, exc)}))
    raise SystemExit(0)
backend = __BACKEND__
module = getattr(torch, backend, None)
if module is None:
    print(json.dumps({"ok": False, "reason": "this torch build has no %r "
                         "backend" % backend}))
    raise SystemExit(0)
try:
    count = int(module.device_count())
except Exception as exc:
    print(json.dumps({"ok": False, "reason": "%s.device_count() raised %s: %s"
                         % (backend, type(exc).__name__, exc)}))
    raise SystemExit(0)
if count <= 0:
    print(json.dumps({"ok": True, "devices": []}))
    raise SystemExit(0)
devices = []
for index in range(count):
    row = {"index": index, "ok": False}
    try:
        props = module.get_device_properties(index)
        row["name"] = getattr(props, "name", None)
        total = getattr(props, "total_memory", None)
        if total:
            row["total_memory_mb"] = round(total / (1024 ** 2))
        # The visible line, and it was missing: `ok` starts False and the
        # success path only ever filled in name and memory, so every device
        # came back absent. `report()` said present=True for the same card
        # at the same moment, which is what exposed it -- the two probes
        # disagreeing about one device is not a state either of them can
        # legitimately be in.
        row["ok"] = True
    except Exception as exc:
        row["error"] = "%s: %s" % (type(exc).__name__, exc)
    devices.append(row)
print(json.dumps({"ok": True, "devices": devices, "count": count}))
"""

    @property
    def enumerate_all(self) -> bool:
        return True

    @staticmethod
    def _python() -> str:
        """The interpreter the probe child runs under. When the environment the server started in has been deleted (a new one is being
        built over it) its python is gone; the base interpreter then answers "no torch" and the cards are listed from the system."""
        if Path(sys.executable).exists():
            return sys.executable
        return getattr(sys, "_base_executable", None) or sys.executable

    def devices_with_reason(self) -> tuple[tuple[DeviceReport, ...], str | None]:
        """The devices, by enumeration rather than by asking which is current."""
        if self._in_process:
            return self._list_here()

        try:
            result = subprocess.run(
                [self._python(), "-c",
                 self._LIST_CHILD_TEMPLATE.replace("__BACKEND__", repr(self.backend))],
                capture_output=True, text=True, timeout=self.timeout,
            )
        except subprocess.TimeoutExpired:
            return (), (
                f"listing {self.backend} devices did not finish in "
                f"{self.timeout:.0f}s"
            )
        except OSError as exc:
            return (), f"the device list could not be started: {exc}"

        payload = _last_json_object(result.stdout)
        if payload is None:
            if result.returncode != 0:
                return (), (
                    f"listing devices exited {result.returncode}: "
                    f"{(result.stderr or '').strip()[-200:]}"
                )
            return (), "the device list produced no answer"
        if not payload.get("ok"):
            return self._without_torch(payload)

        return tuple(
            DeviceReport(
                present=bool(row.get("ok")),
                backend=self.backend,
                name=row.get("name"),
                total_memory_mb=row.get("total_memory_mb"),
                reason=row.get("error"),
            )
            for row in payload.get("devices") or []
        ), None

    def _without_torch(self, payload: dict) -> tuple[tuple[DeviceReport, ...], str | None]:
        """Torch is not installed yet (the setup page asks before it is): list the cards by name from the system instead.

        The rows say so in ``detail``, because "present" here means the system sees an adapter, not that torch can use it."""
        reason = payload.get("reason") or "no devices"
        if not payload.get("torch_missing"):
            return (), reason
        found = self.system_cards.list()
        if not found:
            return (), reason
        note = "seen by the system; torch checks it after the install"
        return tuple(DeviceReport(present=True, backend=self.backend, name=card.name, detail=note) for card in found), None

    def _list_here(self) -> tuple[tuple[DeviceReport, ...], str | None]:
        """Same question, same interpreter, stdout captured. For tests."""
        import contextlib
        import io

        buffer = io.StringIO()
        try:
            with contextlib.redirect_stdout(buffer):
                exec(  # noqa: S102 -- our own source
                    compile(
                        self._LIST_CHILD_TEMPLATE.replace(
                            "__BACKEND__", repr(self.backend)
                        ),
                        "<device-list>", "exec",
                    ),
                    {"__name__": "__probe__"},
                )
        except SystemExit:
            pass
        except Exception as exc:  # noqa: BLE001 -- reported, not raised
            return (), f"the device list raised: {type(exc).__name__}: {exc}"
        return self._parse_list(buffer.getvalue())

    def _parse_list(self, output: str) -> tuple[tuple[DeviceReport, ...], str | None]:
        payload = _last_json_object(output)
        if payload is None:
            return (), f"the device list produced no answer: {output.strip()[-200:]}"
        if not payload.get("ok"):
            return self._without_torch(payload)
        return tuple(
            DeviceReport(
                present=bool(row.get("ok")),
                backend=self.backend,
                name=row.get("name"),
                total_memory_mb=row.get("total_memory_mb"),
                reason=row.get("error"),
            )
            for row in payload.get("devices") or []
        ), None

    def report(self) -> DeviceReport:
        if self._in_process:
            return self._report_here()
        try:
            result = subprocess.run(
                [self._python(), "-c", self._child_source(self.backend)],
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
        working card as a broken probe. See `_last_json_object`, which is
        the shared implementation of exactly that.
        """
        payload = _last_json_object(output)
        if payload is None:
            return DeviceReport(
                present=False,
                reason="the device probe produced no answer",
                detail=output.strip()[-400:],
            )
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
