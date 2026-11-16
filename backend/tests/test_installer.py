"""The installer: what an environment holds, what installing does to it, and when setup refuses.

Claims here are about machines that are not the one running the test, so the environment inspector and the pip runner are fakes
that can only be satisfied by doing the thing. Covers: inspecting a new/existing environment (missing packages, pins, the XPU torch
check), the install stages (torch from its own index first, everything installed pinned), the pip command line, the apply gate,
tokenizer files, and the HTTP surface.
"""

from __future__ import annotations

import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.application.ports.package_installer import InstallError, InstallRequest, PackageInstaller, PipInstaller
from backend.application.ports.requirements_manifest import REQUIREMENTS, TORCH_INDEX_URL
from backend.application.ports.settings_store import SettingsChanges
from backend.application.ports.tokenizer_files import check_content
from backend.application.ports.venv_inspection import VenvInventory, find_interpreter
from backend.presentation.app import create_app
from backend.tests.support import FakeTokenizerFiles, FakeVenvInspector, asgi_request, build_services, check, finish

NAMES = [r.distribution for r in REQUIREMENTS]
SERVER = [r.distribution for r in REQUIREMENTS if r.tier == "required"]


def inventory(**versions: str) -> VenvInventory:
    return VenvInventory("/env/bin/python", "3.13.1", tuple(versions.items()))


class RecordingInstaller(PackageInstaller):
    def __init__(self, fail: Exception | None = None) -> None:
        self.requests: list[InstallRequest] = []
        self.fail = fail

    def install(self, request, on_line=None):
        self.requests.append(request)
        if on_line:
            on_line("fake pip line")
        if self.fail:
            raise self.fail


def services_with(environments, installer=None, **kw):
    root = Path(tempfile.mkdtemp(prefix="backend-installer-"))
    services = build_services(project_root=root, venv_inspector=FakeVenvInspector(environments), **kw)
    if installer is not None:
        services.installer.install.installer = installer
    return services, root


def wait(services, job_id, seconds=10):
    end = time.time() + seconds
    while time.time() < end:
        snap = services.installer.install_status.execute(job_id)
        if snap["terminal"]:
            return snap
        time.sleep(0.02)
    raise AssertionError("install job did not finish")


# ==========================================================================
print("-- inspecting an environment: packages first, then problems --")

existing = {"/env": inventory(fastapi="0.139.0", numpy="2.1.0", torch="2.9.0+xpu", sympy="1.13")}
services, root = services_with(existing)
report = services.installer.inspect.for_target("existing", "/env")
check(report.usable and report.error is None, "an environment with XPU torch is usable")
check("torch" not in report.to_install and "numpy" not in report.to_install, "what is installed is not installed again")
check("uvicorn" in report.to_install and "tokenizers" in report.to_install, "what is missing is listed")
check("numpy==2.1.0" in report.constraints and "sympy==1.13" in report.constraints,
      "every installed package is pinned, including ones the project does not know")
check(report.download_mb > 0 and report.installed_count == 4, "the download size counts only what is missing")

bad = services_with({"/cpu": inventory(torch="2.9.0")})[0].installer.inspect.for_target("existing", "/cpu")
check(not bad.usable and "XPU" in bad.problems[0], "torch without the +xpu build is a problem, not a silent pass")

none = services.installer.inspect.for_target("existing", "/nowhere")
check(none.error and "no Python" in none.error and not none.usable, "a folder without an interpreter is an error")
empty = services.installer.inspect.for_target("existing", "  ")
check(empty.error and "choose" in empty.error, "an existing environment needs a folder")

fresh = services.installer.inspect.for_target("new", None)
check(fresh.kind == "new" and fresh.to_install == tuple(NAMES) and fresh.constraints == (),
      "a new environment is missing everything and protects nothing")
try:
    services.installer.inspect.for_target("comfy", None)
    check(False, "an unknown target is refused")
except ValueError:
    check(True, "an unknown target is refused")

# ==========================================================================
print("\n-- installing: torch from its own index first, the rest pinned --")

recorder = RecordingInstaller()
services, root = services_with({"/env": inventory(fastapi="0.139.0", numpy="2.1.0")}, recorder)
job = services.installer.install.execute("existing", "/env")
snap = wait(services, job.id)
check(snap["state"] == "succeeded" and snap["target_python"] == "/env/bin/python", f"the job succeeds ({snap['error']})")
check(len(recorder.requests) == 2, "two stages: torch, then everything else")
first, second = recorder.requests
check(first.packages == ("torch==2.12.1",) and first.index_url == TORCH_INDEX_URL, "the first stage is torch from the XPU index")
check(second.index_url is None and "torch" not in second.packages, "the second stage uses the default index")
check("numpy==2.1.0" in first.constraints and first.constraints == second.constraints, "both stages are pinned to what was already there")
check("fake pip line" in snap["log"], "pip output reaches the job log")
check(first.packages == ("torch==2.12.1",), "torch is pinned to the tested release by default")

recorder = RecordingInstaller()
services, root = services_with({"/env": inventory(fastapi="0.139.0")}, recorder)
wait(services, services.installer.install.execute("existing", "/env", torch="2.11.0").id)
check(recorder.requests[0].packages == ("torch==2.11.0",), "an exact version can be chosen")
recorder = RecordingInstaller()
services, root = services_with({"/env": inventory(fastapi="0.139.0")}, recorder)
wait(services, services.installer.install.execute("existing", "/env", torch="latest").id)
check(recorder.requests[0].packages == ("torch",), "latest leaves torch unpinned, so the XPU index decides")
for bad in ("2.12; rm", "--index-url=x", "torch==2"):
    try:
        services.installer.install.execute("existing", "/env", torch=bad)
        check(False, f"torch choice {bad!r} is refused")
    except InstallError:
        check(True, f"torch choice {bad!r} is refused")

recorder = RecordingInstaller()
services, root = services_with({"/env": inventory(**{n: "1.0+xpu" if n == "torch" else "1.0" for n in NAMES})}, recorder)
try:
    services.installer.install.execute("existing", "/env")
    check(False, "nothing to install is refused")
except InstallError as exc:
    check("nothing to install" in str(exc) and not recorder.requests, "an environment that is complete is refused before pip runs")

recorder = RecordingInstaller()
services, root = services_with({"/cpu": inventory(torch="2.9.0")}, recorder)
try:
    services.installer.install.execute("existing", "/cpu")
    check(False, "a wrong torch is refused")
except InstallError as exc:
    check("XPU" in str(exc) and not recorder.requests, "a CPU/CUDA torch is refused rather than piled onto")

services, root = services_with({"/env": inventory()}, RecordingInstaller(fail=RuntimeError("boom")))
snap = wait(services, services.installer.install.execute("existing", "/env").id)
check(snap["state"] == "failed" and "boom" in snap["error"], "any escape from the worker still ends the job as failed")

services, root = services_with({"/env": inventory()}, RecordingInstaller(fail=InstallError("pip exited 1")))
snap = wait(services, services.installer.install.execute("existing", "/env").id)
check(snap["state"] == "failed" and snap["error"] == "pip exited 1", "a pip failure is reported as the job's error")

# ==========================================================================
print("\n-- the pip command line --")

request = InstallRequest("/env/bin/python", ("torch",), ("numpy==2.1.0",), TORCH_INDEX_URL)
command = PipInstaller.build_command(request, Path("/tmp/c.txt"))
check("--user" not in command and "--break-system-packages" not in command, "never --user or --break-system-packages")
check(command[command.index("--") + 1:] == ["torch"] and command.index("--") > command.index("--constraint"), "packages come after `--`")
check(command[command.index("--index-url") + 1] == TORCH_INDEX_URL, "the accelerator index is passed as the only index")
uv = PipInstaller.build_command(request, None, uv="/usr/bin/uv")
check(uv[:3] == ["/usr/bin/uv", "pip", "install"] and "--python" in uv, "uv is used when the environment has no pip")
try:
    PipInstaller().install(InstallRequest("/nonexistent/python", ("fastapi",)))
    check(False, "a missing interpreter is refused")
except InstallError as exc:
    check("does not exist" in str(exc), "a missing interpreter is refused with a sentence")

here = Path(tempfile.mkdtemp(prefix="backend-venvdir-"))
check(find_interpreter(str(here)) is None, "a folder with no bin/python is not an environment")
check(find_interpreter(str(here / "notes.txt")) is None, "a missing path is not an interpreter")
(here / "notes.txt").write_text("x")
check(find_interpreter(str(here / "notes.txt")) is None, "a file that is not python is not an interpreter")

# ==========================================================================
print("\n-- setup state and the gate --")


class FreshMachine:
    """A settings store on a machine with no ComfyUI beside it: a path resolves only once something was stored for it.

    The suite's own services find a temporary ComfyUI tree, so the unconfigured state cannot be tested through them. Writes pass
    through to the real store, which keeps the apply-then-refuse sequence a test of the gate and not of a mock.
    """

    def __init__(self, inner) -> None:
        self._inner = inner

    def get(self, key, default=""):
        return self._inner.get(key, default)

    def read(self):
        from backend.application.ports.settings_store import SettingsView

        real = self._inner.read()
        chosen = bool(real.stored.get("models_dir"))
        return SettingsView(stored=dict(real.stored), resolved={
            key: (value if chosen or real.stored.get(key) else None) for key, value in real.resolved.items()})

    def update(self, changes):
        return self._inner.update(changes)


from backend.application.use_cases import ApplyInstallation  # noqa: E402
from backend.infrastructure.persistence.sqlite import SqliteDatabase  # noqa: E402
from backend.infrastructure.settings_store import SqliteSettingsStore  # noqa: E402

root = Path(tempfile.mkdtemp(prefix="backend-installer-fresh-"))
database = SqliteDatabase(root / "backend.db")
database.initialize()
apply = ApplyInstallation(FreshMachine(SqliteSettingsStore(database, root)), root)
state = apply.execute()
check(not state.configured and set(state.missing) == {"checkpoints_dir", "loras_dir"},
      f"a fresh machine needs only the model folders, never ComfyUI ({state.missing})")
check(state.default_models_dir == str(root / "models"), "the default models folder is ./models in the project")

models = root / "chosen" / "models"
apply.apply(SettingsChanges(models_dir=str(models)))
state = apply.execute()
check(state.configured and state.resolved_checkpoints_dir == str(models / "checkpoints"), "choosing a models folder finishes setup")
check((models / "checkpoints").is_dir() and (models / "loras").is_dir(), "the ComfyUI-compatible layout is created inside it")
try:
    apply.apply(SettingsChanges(models_dir=str(root / "other")))
    check(False, "a configured installation refuses the wizard")
except Exception as exc:  # noqa: BLE001
    check(getattr(exc, "code", "") == "installer_not_allowed", "a configured installation refuses the wizard (installer_not_allowed)")

# ==========================================================================
print("\n-- the project's own venv with the wrong torch, and moving the server into the chosen one --")

from backend.application.ports.server_restart import ServerRestarter  # noqa: E402


def own_venv(root: Path) -> str:
    interpreter = root / "venv" / ("Scripts" if sys.platform == "win32" else "bin") / ("python.exe" if sys.platform == "win32" else "python")
    interpreter.parent.mkdir(parents=True)
    interpreter.write_text("#!/bin/sh\n")
    interpreter.chmod(0o755)
    return str(interpreter)


root = Path(tempfile.mkdtemp(prefix="backend-installer-own-"))
own = own_venv(root)
recorder = RecordingInstaller()
inventory_own = VenvInventory(own, "3.13.1", (("torch", "2.15.0"), ("numpy", "2.1.0"), ("sympy", "1.13")))
services = build_services(project_root=root, venv_inspector=FakeVenvInspector({str(root / "venv"): inventory_own}))
services.installer.install.installer = recorder
report = services.installer.inspect.for_target("new", None)
check(report.usable and not report.problems, "a wrong torch in the project's own venv is not a blocker")
check(report.replaces == ("torch 2.15.0",) and "torch" in report.to_install and report.constraints == (),
      "it is replaced, and nothing is pinned against the replacement")
wait(services, services.installer.install.execute("new").id)
check(recorder.requests[0].reinstall and recorder.requests[0].index_url == TORCH_INDEX_URL, "the XPU torch is installed over it")
check(not recorder.requests[1].reinstall, "only the torch stage reinstalls")
check("--force-reinstall" in PipInstaller.build_command(recorder.requests[0]), "the reinstall reaches the pip command line")

other = services_with({"/cpu": inventory(torch="2.15.0")})[0].installer.inspect.for_target("existing", "/cpu")
check(not other.usable and other.replaces == (), "a user's own environment is still never modified")


class FakeRestarter(ServerRestarter):
    def __init__(self, running: str = "", serves: bool = True) -> None:
        self.running, self.serves, self.restarted = running, serves, []

    def running_in(self, interpreter):
        return interpreter == self.running

    def can_serve(self, interpreter):
        return self.serves

    def restart_into(self, interpreter):
        self.restarted.append(interpreter)


for running, serves, expected, why in ((own, True, False, "already running there"), ("", False, False, "it cannot run the server"),
                                       ("", True, True, "a different environment that can")):
    root = Path(tempfile.mkdtemp(prefix="backend-installer-restart-"))
    chosen = own_venv(root)
    database = SqliteDatabase(root / "backend.db")
    database.initialize()
    restarter = FakeRestarter(chosen if running else "", serves)
    result = ApplyInstallation(FreshMachine(SqliteSettingsStore(database, root)), root, restarter).apply(
        SettingsChanges(models_dir=str(root / "m"), venv_python=chosen))
    check(result.restarting is expected and bool(restarter.restarted) is expected, f"restart into the chosen environment: {expected} ({why})")

root = Path(tempfile.mkdtemp(prefix="backend-installer-restart-"))
database = SqliteDatabase(root / "backend.db")
database.initialize()
restarter = FakeRestarter()
check(not ApplyInstallation(FreshMachine(SqliteSettingsStore(database, root)), root, restarter).apply(
    SettingsChanges(models_dir=str(root / "m"))).restarting, "no environment chosen: the server stays where it is")

from backend import first_run  # noqa: E402

check(first_run.same_environment("/a/venv/bin/python", "/a/venv/bin/python3") and not first_run.same_environment("/a/venv/bin/python", "/b/venv/bin/python"),
      "two interpreters of one virtualenv are one environment, without resolving the link to the base python")

import os  # noqa: E402

root = Path(tempfile.mkdtemp(prefix="backend-installer-stored-"))
chosen = own_venv(root)
database = SqliteDatabase(root / "backend.db")
database.initialize()
SqliteSettingsStore(database, root).update(SettingsChanges(venv_python=chosen))
saved_env = {key: os.environ.pop(key, None) for key in ("VENV_PYTHON", "BACKEND_DB_PATH")}
try:
    os.environ["BACKEND_DB_PATH"] = str(root / "backend.db")
    check(first_run.stored_interpreter() == chosen, "the launcher reads the stored interpreter without the server's own code")
    check(first_run.preferred_interpreter() == chosen, "a stored environment other than the running one is preferred at start")
    os.environ["VENV_PYTHON"] = "/explicit/python"
    check(first_run.preferred_interpreter() is None, "an explicit VENV_PYTHON still wins")
finally:
    for key, value in saved_env.items():
        os.environ.pop(key, None)
        if value is not None:
            os.environ[key] = value

# ==========================================================================
print("\n-- tokenizer files --")

check(check_content("vocab.json", b"<html>error</html>") is not None, "an error page is not a vocabulary")
check(check_content("merges.txt", b"a b\n" * 20000) is None, "a long merges file passes")
check(check_content("tokenizer.json", b'{"model": {}}') is None and check_content("tokenizer.json", b"{}") is not None,
      "a tokenizer definition needs a model")

# ==========================================================================
print("\n-- the HTTP surface --")

files = FakeTokenizerFiles()
services, root = services_with({"/env": inventory(torch="2.9.0+xpu")}, RecordingInstaller(), tokenizer_files=files)
http_db = SqliteDatabase(root / "backend-http.db")
http_db.initialize()
fresh_store = FreshMachine(SqliteSettingsStore(http_db, root))
services.installer.apply._settings = fresh_store
app = create_app(services)

status, _, body = asgi_request(app, "/api/v1/installer/state")
check(status == 200 and body["configured"] is False and body["default_models_dir"].endswith("models"), "GET state on a fresh machine")
check(body["installing"] is False and body["install_id"] is None, "no install yet: nothing to lock or resume")
status, _, body = asgi_request(app, "/api/v1/installer/manifest")
check(status == 200 and len(body["requirements"]) == len(REQUIREMENTS), "GET manifest has one row per requirement")
status, _, body = asgi_request(app, "/api/v1/installer/environment", method="POST", json_body={"target": "existing", "venv_dir": "/env"})
check(status == 200 and body["usable"] and "torch" not in body["to_install"], "POST environment reports the missing packages")
status, _, body = asgi_request(app, "/api/v1/installer/environment", method="POST", json_body={"target": "bogus"})
check(status == 422, f"an unknown target is rejected by the schema (got {status})")
status, _, body = asgi_request(app, "/api/v1/installer/environment/verify", method="POST", json_body={"target": "existing", "venv_dir": "/env"})
check(status == 200 and body["present"] and body["torch_version"].endswith("+xpu"), "POST verify answers from the environment's own torch")
status, _, body = asgi_request(app, "/api/v1/installer/tokenizers")
check(status == 200 and {row["key"] for row in body} == {"clip", "qwen", "t5"} and not any(r["complete"] for r in body), "GET tokenizers lists the sets")
status, _, body = asgi_request(app, "/api/v1/installer/tokenizers/t5", method="POST")
check(status == 200 and body["complete"] and files.downloads == ["t5"], "POST tokenizers/t5 fetches that set")
status, _, body = asgi_request(app, "/api/v1/installer/install", method="POST", json_body={"target": "existing", "venv_dir": "/env"})
check(status == 200 and body["id"] and body["packages"], "POST install returns a job with its packages")
job_id = body["id"]
status, _, body = asgi_request(app, "/api/v1/installer/state")
check(body["install_id"] == job_id, "state names the latest install, so a reloaded page can resume it")
wait(services, job_id)
status, _, body = asgi_request(app, "/api/v1/installer/state")
check(body["installing"] is False and body["install_id"] == job_id, "a finished install no longer locks the app")
status, _, body = asgi_request(app, "/api/v1/installer/install/nope")
check(status == 404 and body["error"]["code"] == "install_job_not_found", "an unknown job is a 404 with a code")
status, _, body = asgi_request(app, "/api/v1/installer/apply", method="POST", json_body={"models_dir": str(root / "m")})
check(status == 200 and body["state"]["configured"], "POST apply finishes setup")
status, _, body = asgi_request(app, "/api/v1/installer/apply", method="POST", json_body={"models_dir": str(root / "n")})
check(status == 409 and body["error"]["code"] == "installer_not_allowed", "a second apply is 409")
status, _, body = asgi_request(app, "/api/v1/installer/install", method="POST", json_body={"target": "existing", "venv_dir": "/env"})
check(status == 409, "installing on a configured machine is refused")

finish()
