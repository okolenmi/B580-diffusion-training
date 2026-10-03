"""The install executor: the one thing in the installer that writes.

Everything else here looks. `CheckComfyConflicts` reads two files in
another interpreter, `CheckRequirements` reads dist-info, and both are
pure. This file covers the operation that can break something the user
already had, so it is weighted toward the properties that make that safe
rather than toward the happy path:

* **the command line is the safety argument.** `--constraint` is what makes
  "nothing already installed can change" true; `--user` or
  `--break-system-packages` would make it false, and either appearing is a
  bug that no other check would catch.
* **every thread escape lands.** A worker that raises without recording
  anything leaves the job `running` for ever, and the page then polls a
  spinner that never resolves. That is tested with a bare `RuntimeError`,
  not with the `InstallError` the port documents.
* **a second install is refused at the door**, with the running job's id,
  because two concurrent pips writing one virtualenv is not something to
  allow and diagnose afterwards.

`PipInstaller` itself is never run against a real index here; its command
construction is checked directly, since that is the part that encodes the
safety property and it is pure. `test_bootstrap_install.sh` covers the
real-network path for the preflight, which is the other thing that
installs.
"""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


from backend.application.ports.package_installer import (  # noqa: E402
    InstallError,
    InstallRequest,
    PackageInstaller,
    PipInstaller,
)
from backend.application.use_cases.install_packages import (  # noqa: E402
    FAILED,
    SUCCEEDED,
    GetInstall,
    InstallAlreadyRunning,
    JobNotFound,
    StartInstall,
)
from backend.tests.support import check, finish  # noqa: E402


class FakeInstaller(PackageInstaller):
    """Scripted, and records what it was asked to do.

    `gate` is a threading.Event so the concurrency test can hold an install
    open deliberately rather than by sleeping and hoping.
    """

    def __init__(self, outcome="ok", gate=None, lines=("installing", "done")):
        self.outcome = outcome
        self.gate = gate
        self.lines = lines
        self.requests: list[InstallRequest] = []

    def install(self, request, on_line=None):
        self.requests.append(request)
        if self.gate is not None:
            self.gate.wait(timeout=30)
        for line in self.lines:
            if on_line:
                on_line(line)
        if self.outcome == "fail":
            raise InstallError("pip exited 1. ResolutionImpossible: no match")
        if self.outcome == "explode":
            # Not InstallError. A worker thread must still finish.
            raise RuntimeError("something nobody planned for")
        if self.outcome == "hang":
            time.sleep(30)


def make(outcome="ok", gate=None, packages=("fastapi",)):
    installer = FakeInstaller(outcome=outcome, gate=gate)
    jobs: dict = {}
    start = StartInstall(
        installer=installer,
        project_root=Path("/tmp/does-not-need-to-exist"),
        base_python="python3",
        jobs=jobs,
    )
    return installer, start, GetInstall(jobs=jobs)


def wait_terminal(status, job_id, timeout=30.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        snap = status.execute(job_id)
        if snap["terminal"]:
            return snap
        time.sleep(0.02)
    return status.execute(job_id)


# ==========================================================================
print("-- a successful install --")

installer, start, status = make()
job = start.execute(target="new", packages=("fastapi", "packaging"),
                    constraints=("torch==2.12.1+xpu",))
snap = wait_terminal(status, job.id)

check(snap["state"] == SUCCEEDED,
      f"the job reaches a terminal state ({snap['state']}, {snap['error']!r})")
check(len(installer.requests) == 1,
      f"pip was run exactly once ({len(installer.requests)})")
check(installer.requests[0].packages == ("fastapi", "packaging"),
      f"with the packages asked for ({installer.requests[0].packages})")
check(installer.requests[0].constraints == ("torch==2.12.1+xpu",),
      f"and the pins handed through unchanged "
      f"({installer.requests[0].constraints})")
check("done" in snap["log"],
      f"and its output lines are in the job's log, for the page to show "
      f"({snap['log']})")
check(snap["error"] is None, f"with no error on success ({snap['error']!r})")

# ==========================================================================
print("\n-- it returns before the install finishes --")

# That is the whole reason a job exists. A 2.5 GB download cannot be waited
# on by an HTTP request.
gate = threading.Event()
installer, start, status = make(gate=gate)
began = time.monotonic()
job = start.execute(target="new", packages=("fastapi",))
returned = time.monotonic() - began
check(returned < 2.0,
      f"execute() returned in {returned:.2f}s while the install is held open "
      f"(a blocked request is the thing being avoided)")
check(status.execute(job.id)["state"] in ("queued", "running"),
      f"and the job is already pollable ({status.execute(job.id)['state']})")
gate.set()  # let it finish so the thread does not linger
check(wait_terminal(status, job.id)["state"] == SUCCEEDED,
      "and it still completes once released")

# ==========================================================================
print("\n-- every way it can fail, reported as a failure --")

installer, start, status = make(outcome="fail")
snap = wait_terminal(status, start.execute(
    target="new", packages=("fastapi",)).id)
check(snap["state"] == FAILED,
      f"a refused pip is FAILED, not left running ({snap['state']})")
check(snap["error"] and "ResolutionImpossible" in snap["error"],
      f"carrying pip's own reason ({snap['error']!r})")
check("installing" in snap["log"],
      f"and the log survives the failure, so there is something to read "
      f"({snap['log']})")

# The one that matters most: an *unexpected* exception must still finish the
# job. A worker thread that dies quietly leaves the page polling a spinner
# for ever, and nothing else in the system would notice.
installer, start, status = make(outcome="explode")
snap = wait_terminal(status, start.execute(target="new", packages=("fastapi",)).id)
check(snap["state"] == FAILED,
      f"an unexpected exception still reaches a terminal state "
      f"({snap['state']}) -- a worker must never leave a job 'running'")
check(snap["error"] and "RuntimeError" in snap["error"],
      f"and names what happened ({snap['error']!r})")

# ==========================================================================
print("\n-- refusals, before anything is written --")

installer, start, status = make()
try:
    start.execute(target="new", packages=())
    refused = "nothing"
except InstallError as exc:
    refused = str(exc)
check("nothing to install" in refused,
      f"an empty package list is refused, not installed as a no-op "
      f"({refused!r})")
check(not installer.requests,
      f"and pip was never run ({len(installer.requests)} requests)")

installer, start, status = make()
try:
    start.execute(target="somewhere-else", packages=("fastapi",))
    refused = "nothing"
except InstallError as exc:
    refused = str(exc)
check("not a target this project installs into" in refused,
      f"an unknown target is refused by name ({refused!r})")
check(not installer.requests,
      f"and pip was never run ({len(installer.requests)} requests)")

# The realistic first-run shape: the user picked "reuse ComfyUI's venv" but
# nothing has told us which interpreter that is.
installer, start, status = make()
try:
    start.execute(target="comfy", packages=("fastapi",), constraints=("x==1",))
    refused = "nothing"
except InstallError as exc:
    refused = str(exc)
check("not known" in refused and "Nothing was changed" in refused,
      f"reusing a venv we cannot identify is refused, and says so "
      f"({refused!r})")
check(not installer.requests,
      f"and pip was never run ({len(installer.requests)} requests)")

# ==========================================================================
print("\n-- one install at a time --")

gate = threading.Event()
installer, start, status = make(gate=gate)
first = start.execute(target="new", packages=("fastapi",))
try:
    start.execute(target="new", packages=("uvicorn",))
    refused = None
except InstallAlreadyRunning as exc:
    refused = str(exc)
check(refused == first.id,
      f"a second install is refused with the running job's id, so the page "
      f"can attach to it rather than start a rival ({refused!r} vs {first.id})")
gate.set()
wait_terminal(status, first.id)

# And after it finishes, another is allowed.
installer, start, status = make(gate=gate)
gate.set()  # already open, so this one completes
second = start.execute(target="new", packages=("fastapi",))
check(wait_terminal(status, second.id)["state"] == SUCCEEDED,
      "and a later install is allowed once the first is terminal")

# ==========================================================================
print("\n-- an unknown job is not a fabricated success --")

installer, start, status = make()
try:
    status.execute("never-existed")
    found = "no error"
except JobNotFound:
    found = "JobNotFound"
check(found == "JobNotFound",
      f"reading a job this process does not know raises rather than "
      f"inventing a state ({found})")

# ==========================================================================
print("\n-- the command line is the safety argument --")

pip = PipInstaller()
new_cmd = pip.build_command(InstallRequest(
    target_python="/tmp/x/bin/python", packages=("fastapi", "torch")))
check("--constraint" not in new_cmd,
      f"a fresh venv has nothing to protect, so no constraints file is "
      f"invented ({new_cmd})")

pins = PipInstaller.write_constraints(InstallRequest(
    target_python="/tmp/comfy/venv/bin/python",
    packages=("fastapi",), constraints=("torch==2.12.1+xpu", "anyio==4.14.0")))
# write_constraints creates a real directory, so this file has to remove it.
# An earlier version did not, and left eight of them in /tmp per run -- which
# is exactly what scripts/test_install_executor.sh checks for, so the leak
# was found by the production-path test rather than by this one.
_DIRS_TO_CLEAN: list = [pins[1]] if pins else []
comfy_cmd = pip.build_command(InstallRequest(
    target_python="/tmp/comfy/venv/bin/python", packages=("fastapi",)),
    constraints_path=pins[0] if pins else None)
check("--constraint" in comfy_cmd and str(pins[0]) in comfy_cmd,
      f"a reused venv always gets one, pointing at the pins "
      f"({comfy_cmd})")
check(pins[0].read_text(encoding="utf-8").split() ==
      ["torch==2.12.1+xpu", "anyio==4.14.0"],
      f"and the file contains exactly the pins, one per line "
      f"({pins[0].read_text(encoding='utf-8').split()})")
check(not Path("/home/okolenmi/Desktop/B580-diffusion-training").joinpath("torch==2.12.1+xpu").exists(),
      "and nothing was written into the project")
check(PipInstaller.write_constraints(InstallRequest(
    target_python="/x", packages=("a",))) is None,
    "no pins means no file, rather than an empty constraints file that pip "
    "would still have to read")

# build_command must not write: it is what the wizard shows before running.
before = sorted(p.name for p in Path("/tmp").glob("distillation-install-*"))
pip.build_command(InstallRequest(
    target_python="/x", packages=("a",)), constraints_path=Path("/nonexistent/c.txt"))
after = sorted(p.name for p in Path("/tmp").glob("distillation-install-*"))
check(before == after,
      f"build_command is pure -- inspecting it creates nothing "
      f"({len(before)} -> {len(after)})")

# These three would each defeat the entire safety argument.
for flag in ("--user", "--break-system-packages", "--ignore-installed"):
    check(flag not in comfy_cmd and flag not in new_cmd,
          f"{flag} never appears -- it would let pip change things outside "
          f"the named environment")

check("--no-input" in comfy_cmd,
      "and --no-input is always passed, so pip cannot stop for a question "
      "nobody is there to answer")

# The interpreter is named absolutely, never resolved here.
check(comfy_cmd[0] == "/tmp/comfy/venv/bin/python",
      f"the target interpreter is explicit ({comfy_cmd[0]})")

# ==========================================================================
print("\n-- a missing interpreter is refused, not half-attempted --")

pip = PipInstaller()
try:
    pip.install(InstallRequest(
        target_python="/nonexistent/python", packages=("fastapi",)))
    refused = None
except InstallError as exc:
    refused = str(exc)
check(refused and "does not exist" in refused,
      f"a target interpreter that is not there is refused with a sentence "
      f"({refused!r})")


# -- and nothing this file created is left behind --------------------------
#
# `write_constraints` makes a real directory, so this file has to remove it.
# An earlier version did not, and left eight per run in /tmp -- which is how
# `scripts/test_install_executor.sh` found it, because checking for stale
# constraint directories is the kind of thing only a real run notices.
import shutil  # noqa: E402

for _directory in _DIRS_TO_CLEAN:
    shutil.rmtree(_directory, ignore_errors=True)
check(not any(d.exists() for d in _DIRS_TO_CLEAN),
      f"and this test leaves no constraint directory in the temp dir "
      f"({[str(d) for d in _DIRS_TO_CLEAN]})")

finish()