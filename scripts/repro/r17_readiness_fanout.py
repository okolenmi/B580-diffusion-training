"""R4-01: concurrent readiness requests must not each import torch.

Run from the repo root:

    python scripts/repro/r17_readiness_fanout.py

Exits non-zero if the fan-out returns, so it is a regression guard rather
than a demonstration. That was the review's actual request -- "add a
real-subprocess variant that asserts a peak of 1" -- and the first version
of this file did not do it: it printed the peak and exited 0, so it would
have watched a regression happen without failing.

It also measured the wrong object. It built a bare ``TorchDeviceProbe``,
which is exactly the uncached thing the fix replaced, so after the fix it
still reported a peak of 8 and a reader could reasonably conclude the fix
had not worked. It had -- but not here. This version takes the probe the
container actually wires, and runs the bare one alongside as a control, so
the difference is visible rather than asserted.

What it asserts: the wired probe peaks at one torch-importing process, and
the 8 concurrent callers all get an answer.
"""
import os
import subprocess
import sys
import threading
import time

sys.path.insert(0, os.environ.get("REPO", "."))

CALLERS = 8


def _watch(stop, peak):
    """Sample how many torch-importing processes exist, continuously."""
    while not stop.is_set():
        out = subprocess.run(
            ["pgrep", "-fc", "import torch|torch"],
            capture_output=True, text=True,
        ).stdout.strip()
        try:
            peak["n"] = max(peak["n"], int(out))
        except ValueError:
            pass
        time.sleep(0.1)


def _hammer(probe, label):
    peak = {"n": 0}
    stop = threading.Event()
    watcher = threading.Thread(target=_watch, args=(stop, peak))
    watcher.start()
    began = time.monotonic()
    answers = []
    threads = [
        threading.Thread(target=lambda: answers.append(probe.report()))
        for _ in range(CALLERS)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    elapsed = time.monotonic() - began
    stop.set()
    watcher.join()
    print(f"  {label:34} {CALLERS} concurrent: {elapsed:.1f}s, "
          f"peak torch processes {peak['n']}, "
          f"{len(answers)} answered, device {answers[0].name!r}")
    return peak["n"], len(answers)


print("== the probe the server actually wires ==")
from backend.bootstrap import build_container  # noqa: E402
from backend.config import Settings  # noqa: E402
from pathlib import Path  # noqa: E402
import tempfile  # noqa: E402

root = Path(tempfile.mkdtemp(prefix="r17-"))
container = build_container(Settings(project_root=root, db_path=root / "b.db"))
wired = container.services.installer.device_probe
print(f"  type: {type(wired).__name__}")

wired_peak, answered = _hammer(wired, "wired probe")

print()
print("== control: the bare probe, which is what the fix replaced ==")
from backend.application.ports.environment import TorchDeviceProbe  # noqa: E402

bare_peak, _ = _hammer(TorchDeviceProbe(backend="xpu"), "bare TorchDeviceProbe")

print()
print(f"  wired peak {wired_peak}, bare peak {bare_peak}")

failures = []
if wired_peak > 1:
    failures.append(
        f"the wired probe peaked at {wired_peak} torch-importing processes; "
        f"single-flight is not holding"
    )
if answered != CALLERS:
    failures.append(f"only {answered} of {CALLERS} callers got an answer")
if bare_peak <= 1:
    # If the control does not fan out, this script is measuring nothing and
    # a green result here would be meaningless.
    failures.append(
        f"the bare probe peaked at {bare_peak}, so this script is not "
        f"reproducing the thing it is meant to guard"
    )

if failures:
    print()
    for line in failures:
        print(f"  FAIL  {line}")
    sys.exit(1)
print()
print("OK: one torch import answers every caller; the bare probe still "
      "fans out, so this file is measuring something.")