"""The installer surface: what it reports, and when it refuses.

Round-3 follow-on to `docs/design/11-first-run-and-installer.md`. The
document's §1 asks for a machine-readable requirements list and its §5
asks for the wizard's checks; this file is the backend half of both, and
it is deliberately the *hardest* half to get wrong, because every claim it
makes is about a machine that is not the one running the test.

Three properties, and each is checked against a stub that can only be
satisfied by doing the thing:

**The readiness report is true about the machine it ran on.** The
inventory is stubbed with a machine that has the server's packages and not
the training stack, and the device probe is a spy that fails the test if it
is asked at all. The second half matters: a probe that imports torch costs
1.7 s on this hardware, and calling it to be told "no" anyway is a cost
with no information in it.

**A wizard write is refused once configured.** Not because writing is
dangerous in itself -- `POST /settings` exists and does the same thing --
but because a wizard left open in a tab on a working machine should not be
able to re-point the model directories out from under a run. The refusal
is 409, not 403: the request was well-formed and the state does not allow
it now.

**The manifest is data, and the two lists it names are different.** torch
is in the full install and must never be in the ComfyUI additions, because
installing torch into a venv another application pins is exactly the
failure the constraints-file design exists to make unreachable.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.application.ports.environment import (
    DeviceReport,
    InstalledPackage,
)
from backend.application.ports.requirements_manifest import REQUIREMENTS
from backend.application.ports.settings_store import SettingsChanges
from backend.application.use_cases import (
    ApplyInstallation,
    CheckRequirements,
    DescribeRequirements,
)
from backend.bootstrap import build_container
from backend.config import Settings
from backend.presentation.app import create_app
from backend.tests.support import (
    FakeDeviceProbe,
    asgi_request,
    check,
    finish,
)

ROOT = Path(tempfile.mkdtemp(prefix="backend-installer-"))


class StubInventory:
    """Reports exactly the versions it is handed, and imports nothing.

    A stub rather than the real inventory on purpose: this file's claims
    are about a *hypothetical* machine, and reading the real one's metadata
    would quietly make every assertion below a statement about the machine
    running the tests instead.
    """

    def __init__(self, versions: dict[str, str]) -> None:
        self._versions = versions

    def installed(self):
        return tuple(
            InstalledPackage(distribution=name, import_name=name, version=version)
            for name, version in sorted(self._versions.items())
        )

    def version_of(self, distribution: str) -> str | None:
        return self._versions.get(distribution)

    def import_name_for(self, distribution: str) -> str:
        from backend.application.ports.environment import IMPORT_NAMES

        return IMPORT_NAMES.get(distribution, distribution)

    def distribution_for(self, import_name: str) -> str:
        for dist, mod in __import__(
            "backend.application.ports.environment", fromlist=["IMPORT_NAMES"]
        ).IMPORT_NAMES.items():
            if mod == import_name:
                return dist
        return import_name


class NeverProbe:
    """A device probe that fails the test if it is asked anything."""

    def __init__(self) -> None:
        self.calls = 0

    def report(self) -> DeviceReport:
        self.calls += 1
        raise AssertionError(
            "the device was probed on a machine already known not to be ready"
        )


#: A machine that can serve the API and cannot train. The realistic shape
#: of "someone installed requirements.txt and stopped".
#:
#: `packaging` is in this list because requirements.txt lists it and the
#: manifest now agrees -- and it was *missed* in both when the conflict
#: check added it. That omission was silent in a nasty way: a package the
#: server needs but the manifest does not know about is invisible to the
#: wizard, so the install would have run pip without it.
SERVER_ONLY = {
    "fastapi": "0.139.0",
    "uvicorn": "0.49.0",
    "python-multipart": "0.0.32",
    "tomli_w": "1.2.0",
    "packaging": "26.3",
}

#: The same machine with the training stack, and a card.
FULL = {
    **SERVER_ONLY,
    "torch": "2.12.1+xpu",
    "numpy": "2.5.0rc1",
    "safetensors": "0.8.0",
    "pillow": "12.2.0",
}

PRESENT_CARD = DeviceReport(
    present=True, backend="xpu",
    name="Intel(R) Arc(TM) B580 Graphics", total_memory_mb=12216,
)

# ==========================================================================
# Section A: the readiness report is true about the machine it ran on
# ==========================================================================
print("-- readiness: a machine that can serve but not train --")

never = NeverProbe()
report = CheckRequirements(StubInventory(SERVER_ONLY), never).execute()

check(not report.ready,
      "a machine with no training stack is not ready, whatever else is true")
check({row.name for row in report.blocking_missing}
      == {"torch", "numpy", "safetensors", "pillow"},
      f"and the four missing training packages are named "
      f"(got {[r.name for r in report.blocking_missing]})")
check(never.calls == 0,
      f"the device was never asked, because the answer was already no "
      f"(asked {never.calls}x)")
check(not report.device_checked,
      "and the report says the device was not checked, rather than "
      "reporting a card it never looked for")
check(report.device_reason is None,
      f"'not checked' is not 'no card': device_reason stays None "
      f"(got {report.device_reason!r}) -- a UI rendering this as the cause "
      f"would tell a user with a B580 that their graphics card is missing")

check(all(row.satisfied for row in report.packages
          if row.name in SERVER_ONLY),
      "and the four server packages it does have are reported satisfied")

ready = CheckRequirements(StubInventory(FULL), FakeDeviceProbe()).execute()
check(ready.ready,
      f"the same machine with torch, numpy, safetensors and pillow is ready "
      f"(missing {[r.name for r in ready.missing]})")
check(ready.device_checked and ready.device_present,
      "and now the device *is* asked, and reported present")
check(ready.device_name == "Intel(R) Arc(TM) B580 Graphics"
      and ready.device_total_memory_mb == 12216,
      f"with the card's own name and size, not a guess "
      f"(got {ready.device_name!r}, {ready.device_total_memory_mb} MB)")

no_card = CheckRequirements(
    StubInventory(FULL),
    FakeDeviceProbe(DeviceReport(present=False, reason="torch reports no xpu device")),
).execute()
check(not no_card.ready,
      "everything installed and no card is still not ready -- the packages "
      "answer 'can it serve', never 'can it train'")
check(no_card.device_reason == "torch reports no xpu device",
      f"and the reason is carried through verbatim ({no_card.device_reason!r})")

# A missing *server* package is blocking by definition.
missing_server = dict(FULL)
del missing_server["uvicorn"]
rows = {r.name: r for r in CheckRequirements(
    StubInventory(missing_server), FakeDeviceProbe()).execute().packages}
check(rows["uvicorn"].blocking and not rows["uvicorn"].satisfied,
      "a missing server package is blocking: the server cannot start")

# ==========================================================================
# Section B: the manifest is data, and the two lists differ
# ==========================================================================
print("\n-- the manifest, and the split that makes an install safe --")

manifest = DescribeRequirements().execute()
names = {r["name"] for r in manifest["requirements"]}
# Every package requirements.txt names, not a hand-copied four: the copy is
# what let `packaging` be missed in both places at once.
_req = sorted(
    line.strip()
    for line in (Path(__file__).resolve().parents[2] / "requirements.txt")
    .read_text(encoding="utf-8").splitlines()
    if line.strip() and not line.startswith(("#", "-"))
)
check(set(_req) <= names,
      f"every package requirements.txt names is in the manifest "
      f"(file {_req}, missing {sorted(set(_req) - names)})")
check({"torch", "numpy", "safetensors", "pillow"} <= names,
      "and so are the four the trainer imports, which appear in no "
      "requirements file at all -- the gap this surface exists to show")

tiers = {r["name"]: r["tier"] for r in manifest["requirements"]}
check(tiers["fastapi"] == "required" and tiers["torch"] == "training",
      f"the server's own packages are 'required' and the training stack is "
      f"'training' (got {tiers['fastapi']!r}, {tiers['torch']!r})")
check(all(r["why"] for r in manifest["requirements"]),
      "every row carries a reason, so a missing package can be explained "
      "in the user's terms rather than by name alone")

check("torch" in manifest["full_install"],
      "torch is in the full install list (a new venv needs it)")
check("torch" not in manifest["comfy_additions"],
      f"and NOT in the additions list -- installing torch into a venv "
      f"another application pins is the failure the constraints file "
      f"exists to prevent (got {manifest['comfy_additions']})")
check(set(manifest["comfy_additions"]) < set(manifest["full_install"]),
      "the additions list is a strict subset of the full one")
check(manifest["full_install_approx_mb"] > 2000,
      f"the full install is sized in MB, because the venv choice is a disk "
      f"decision (got {manifest['full_install_approx_mb']})")

never_install = {r["name"] for r in manifest["requirements"]
                 if r["never_install"]}
check("torch" in never_install,
      f"and torch is marked never-install-into-a-foreign-venv "
      f"(marked: {sorted(never_install)})")

# ==========================================================================
# Section C: first-run state, and the write that is refused after it
# ==========================================================================
print("\n-- first run: state, one apply, and the refusal after --")


class UnresolvableStore:
    """A settings store whose paths resolve to nothing, on purpose.

    The real store resolves `comfy_dir` by asking the repo's own `paths`
    module, which on a configured machine succeeds -- so a test for the
    *unconfigured* state cannot be written against it without depending on
    where the repository happens to live and what `.env` says. This states
    the state directly: every path setting resolves to nothing.

    Writes pass through to the real store, which is what makes the "apply
    then refuse" sequence below a test of the gate rather than of a mock.
    """

    def __init__(self, inner) -> None:
        self._inner = inner

    def get(self, key, default=""):
        return self._inner.get(key, default)

    def read(self):
        from backend.application.ports.settings_store import SettingsView

        real = self._inner.read()
        return SettingsView(
            stored=dict(real.stored),
            resolved={key: None for key in real.resolved},
        )

    def update(self, changes):
        return self._inner.update(changes)


def _store(project_root: Path):
    from backend.infrastructure.persistence.sqlite import SqliteDatabase
    from backend.infrastructure.settings_store import SqliteSettingsStore

    db = SqliteDatabase(project_root / "backend.db")
    db.initialize()
    return SqliteSettingsStore(db, project_root)


def _comfy_tree(root: Path) -> Path:
    comfy = root / "ComfyUI"
    (comfy / "models" / "checkpoints").mkdir(parents=True)
    (comfy / "models" / "loras").mkdir(parents=True)
    return comfy


# -- the unconfigured state, and applying to it ---------------------------
fresh_root = Path(tempfile.mkdtemp(prefix="backend-installer-fresh-"))
fresh_comfy = _comfy_tree(fresh_root)
fresh_store = _store(fresh_root)
unconfigured = ApplyInstallation(UnresolvableStore(fresh_store))

state = unconfigured.execute()
check(not state.configured,
      f"a machine that resolves no path is unconfigured, which is the state "
      f"the wizard exists for (got configured={state.configured})")
check(set(state.missing) == {"comfy_dir", "checkpoints_dir", "loras_dir"},
      f"and all three are named, so the wizard can show what is left to do "
      f"rather than a yes/no (got {list(state.missing)})")

view = unconfigured.apply(SettingsChanges(
    comfy_dir=str(fresh_comfy),
    checkpoints_dir=str(fresh_comfy / "models" / "checkpoints"),
    loras_dir=str(fresh_comfy / "models" / "loras"),
))
check(view.stored["comfy_dir"] == str(fresh_comfy),
      f"the apply persists the paths it was given (stored comfy="
      f"{view.stored['comfy_dir']!r})")

# The gate is a predicate, not a one-shot flag: it refuses again.
#
# Over the *unresolvable* wrapper, because that is the only store that can
# be configured while still reporting nothing resolved -- and without that,
# the second write is rejected by the settings store's own validation
# (the directory does not exist) before the gate is ever consulted. Which
# error comes back is the whole point of the check: `settings_invalid`
# would mean the user is told their path is wrong when what actually
# happened is that the wizard is closed.
unconfigured_now_resolving = ApplyInstallation(fresh_store)
try:
    unconfigured_now_resolving.apply(
        SettingsChanges(comfy_dir=str(fresh_comfy / "other"))
    )
    code = None
except Exception as exc:  # noqa: BLE001 -- the code is what is asserted
    code = getattr(exc, "code", None)
check(code == "installer_not_allowed",
      f"a second wizard write on the same, now-configured store is refused "
      f"by the gate rather than by path validation (got {code!r})")

# -- the HTTP surface -------------------------------------------------------
root = Path(tempfile.mkdtemp(prefix="backend-installer-api-"))
comfy = _comfy_tree(root)
container = build_container(
    Settings(project_root=root, db_path=root / "backend.db")
)
app = create_app(container.services)

status, _, body = asgi_request(app, "/api/v1/installer/state")
check(status == 200, f"GET installer/state 200 (got {status})")
check(body["configured"] is True,
      f"a machine whose ComfyUI auto-detects reports configured, so the "
      f"wizard is not offered (got {body})")

status, _, body = asgi_request(app, "/api/v1/installer/manifest")
check(status == 200 and len(body["requirements"]) == len(REQUIREMENTS),
      f"GET installer/manifest 200 with one row per manifest entry "
      f"(got {status}, {len(body.get('requirements', []))} rows, "
      f"{len(REQUIREMENTS)} expected)")

status, _, body = asgi_request(app, "/api/v1/installer/readiness")
check(status == 200 and "ready" in body,
      f"GET installer/readiness 200 with a verdict (got {status})")
check(isinstance(body.get("device_checked"), bool),
      "and device_checked is always present as a boolean -- a client must "
      "not have to tell 'absent' from 'not asked' by omission")
check(isinstance(body.get("packages"), list)
      and all({"name", "tier", "why", "installed", "blocking"} <= set(row)
              for row in body["packages"]),
      "and every package row carries the fields a wizard renders without "
      "a second request")

status, _, body = asgi_request(app, "/api/v1/installer/apply", method="POST",
                              json_body={"checkpoints_dir": str(comfy)})
check(status == 409,
      f"POST installer/apply on a configured machine is 409, not 200 "
      f"(got {status})")
check(body.get("error", {}).get("code") == "installer_not_allowed",
      f"with the documented code, so a client can tell a refused wizard "
      f"from a rejected path (got {body.get('error', {}).get('code')!r})")

# The refusal must not have written anything.
status, _, after = asgi_request(app, "/api/v1/settings")
check(after["stored"]["checkpoints_dir"] != str(comfy),
      f"and the refused write left the stored value untouched (got "
      f"{after['stored']['checkpoints_dir']!r})")

# An unknown route under the prefix is a 404, not a 500.
status, _, body = asgi_request(app, "/api/v1/installer/nonsense")
check(status == 404, f"an unknown installer path is 404 (got {status})")

# ==========================================================================
print("\n-- the conflicts endpoint: three outcomes over HTTP --")

# A fake port, because a *conflict* on a real machine is a bug in someone's
# environment and a test that waits for one is a test that never runs. The
# classification itself is covered in test_comfy_conflicts.py, against both
# a fake and the real checkout; this section is about the wire format and
# about which arguments the route passes.
from backend.application.ports.comfy_environment import (  # noqa: E402
    ComfyEnvironment,
    ComfyEnvironmentInfo,
    Declaration,
)
from backend.application.ports.package_installer import (  # noqa: E402
    PackageInstaller,
)
from backend.application.use_cases.check_comfy_conflicts import (  # noqa: E402
    CheckComfyConflicts,
)
from backend.application.use_cases.install_packages import (  # noqa: E402
    GetInstall,
    StartInstall,
)


def _with_conflicts(base, check):
    """Rebuild container -> services -> installer with a replaced use case.

    Three frozen dataclasses, so a test cannot just assign the field --
    `FrozenInstanceError` is the first thing that happens otherwise.
    `dataclasses.replace` is the intended way through, and rebuilding the
    app from the new services means the route under test is reached through
    the same wiring production uses.
    """
    import dataclasses

    installer = dataclasses.replace(base.services.installer, conflicts=check)
    services = dataclasses.replace(base.services, installer=installer)
    return dataclasses.replace(base, services=services), create_app(services)


def _with_install(base, start):
    """Rebuild the installer services with a real StartInstall.

    Same reason as _with_conflicts: three frozen dataclasses, and the
    install services must share one job dict or a job cannot be polled.
    """
    import dataclasses

    installer = dataclasses.replace(
        base.services.installer, install=start,
        install_status=GetInstall(jobs=start.jobs),
    )
    services = dataclasses.replace(base.services, installer=installer)
    return dataclasses.replace(base, services=services), create_app(services)


class _StubComfyEnvironment(ComfyEnvironment):
    def __init__(self, declarations=(), installed=()):
        self.declarations = tuple(
            d if isinstance(d, Declaration) else Declaration(*d) for d in declarations
        )
        self.installed = tuple(installed)
        self.seen = []

    def read(self, comfy_dir, venv_python=None):
        self.seen.append((comfy_dir, venv_python))
        return ComfyEnvironmentInfo(
            comfy_dir=comfy_dir,
            requirements_path="/stub/requirements.txt",
            declarations=self.declarations,
            installed=self.installed,
            venv_python=venv_python,
        )


stub = _StubComfyEnvironment(
    [("torch", ""), ("transformers", ">=4.50.3")],
    [("torch", "2.12.1"), ("transformers", "4.44.0"), ("anyio", "4.14.0")],
)
container, app = _with_conflicts(container, CheckComfyConflicts(environment=stub))

status, _, body = asgi_request(app, "/api/v1/installer/conflicts")
check(status == 200, f"GET installer/conflicts 200 (got {status})")

check(body["safe"] is False and body["checked"] is True,
      f"a declared-and-violated package makes it unsafe but *checked* "
      f"(safe={body['safe']}, checked={body['checked']})")
check(body["refusal_reason"] and "transformers" in body["refusal_reason"],
      f"and the refusal names the package and both versions, so the user "
      f"can act on it ({body['refusal_reason']!r})")

check(body["counts"] == {"total": 3, "safe": 1, "conflict": 1, "unknown": 1},
      f"the three outcomes are counted separately, not folded into a "
      f"boolean ({body['counts']})")

outcomes = {row["name"]: row["outcome"] for row in body["findings"]}
check(outcomes == {"transformers": "conflict", "anyio": "unknown", "torch": "safe"},
      f"and each row carries its own outcome, so the client does not "
      f"re-derive it ({outcomes})")

check([r["name"] for r in body["findings"]][0] == "transformers",
      f"conflicts lead, because a refusal must open with its reason "
      f"({[r['name'] for r in body['findings']]})")

check(body["constraints"] == ["anyio==4.14.0", "torch==2.12.1", "transformers==4.44.0"],
      f"the exact pins are sent, so 'nothing already installed can change' "
      f"is visible rather than asserted ({body['constraints']})")

check(all({"name", "outcome", "description"} <= set(row) for row in body["findings"])
      and all("description" in a for a in body["additions"]),
      "every row carries a sentence, so the wizard does not write prose "
      "that can drift from the rule")

# The *stored* interpreter, not the resolved one. They differ exactly when
# nothing is configured, which is every first-run machine: `resolved` then
# ends in a bare "python" -- the right answer for a training subprocess,
# and the wrong one here, because it is this server's interpreter and the
# question is about ComfyUI's venv. Passing it read 87 packages where the
# real answer is 185, with nothing in the response to say whose venv it was.
view = container.services.settings.read.execute()
asgi_request(app, "/api/v1/installer/conflicts")
check(stub.seen[-1][1] == (view.stored.get("venv_python") or None),
      f"the interpreter passed to the port is the configured one, and None "
      f"when there is none so the port can derive it from the checkout "
      f"(got {stub.seen[-1][1]!r}, stored {view.stored.get('venv_python')!r}, "
      f"resolved {view.resolved.get('venv_python')!r})")

# comfy_dir is overridable per call, so the wizard can check a directory the
# user has just typed. It must reach the port unchanged.
asgi_request(app, "/api/v1/installer/conflicts?comfy_dir=/somewhere/typed")
check(stub.seen[-1][0] == "/somewhere/typed",
      f"comfy_dir from the query reaches the port "
      f"({stub.seen[-1][0]})")

# And with no query it comes from settings, not from the request.
asgi_request(app, "/api/v1/installer/conflicts")
configured_dir = container.services.settings.read.execute().resolved.get("comfy_dir")
check(stub.seen[-1][0] == (configured_dir or ""),
      f"with no query it uses the configured ComfyUI directory "
      f"({stub.seen[-1][0]!r} vs {configured_dir!r})")

# An unreadable source is 200 with safe=false -- not a 404, and not 500.
stub2 = _StubComfyEnvironment()
stub2.read = lambda comfy_dir, venv_python=None: ComfyEnvironmentInfo(
    comfy_dir=comfy_dir, requirements_error="there is no requirements.txt",
)
container, app = _with_conflicts(container, CheckComfyConflicts(environment=stub2))
status, _, body = asgi_request(app, "/api/v1/installer/conflicts")
check(status == 200 and body["safe"] is False and body["checked"] is False,
      f"an unreadable source is a 200 saying unchecked-and-unsafe, so the "
      f"client can render the reason (status={status}, safe={body['safe']}, "
      f"checked={body['checked']})")
check(body["refusal_reason"] and "Cannot check" in body["refusal_reason"],
      f"with a reason that says the check did not run "
      f"({body['refusal_reason']!r})")

# ==========================================================================
print("\n-- what may be installed into ComfyUI's venv --")

# This is the split the whole conflict check exists to protect, and it is
# per-row rather than per-tier. A first version of the wizard filtered on
# `tier != "comfy_provided"` -- which no requirement actually uses, so the
# filter matched nothing and the page offered to install torch into
# ComfyUI's virtualenv. That is the single operation the whole design
# refuses, so the invariant is asserted here rather than left to the JS.
from backend.application.ports.requirements_manifest import (  # noqa: E402
    COMFY_ADDITIONS,
)

# The manifest and requirements.txt must agree on the server's own
# packages. They drifted: requirements.txt gained `packaging` for the
# conflict check and the manifest kept saying four. The manifest is what
# the wizard renders *and* what the install acts on, so a package the
# server needs and the manifest does not know about is invisible to the
# only screen that can install it.
req_file = sorted(
    line.strip()
    for line in (Path(__file__).resolve().parents[2] / "requirements.txt")
    .read_text(encoding="utf-8").splitlines()
    if line.strip() and not line.startswith(("#", "-"))
)
required_tier = sorted(r.distribution for r in REQUIREMENTS if r.tier == "required")
check(req_file == required_tier,
      f"requirements.txt and the manifest's required tier are the same list "
      f"({req_file} vs {required_tier})")

check("torch" not in COMFY_ADDITIONS and "numpy" not in COMFY_ADDITIONS,
      f"the accelerator stack is never in what may go into ComfyUI's venv "
      f"({list(COMFY_ADDITIONS)})")
check(all(r.never_install for r in REQUIREMENTS if r.distribution
          in ("torch", "numpy", "safetensors", "pillow")),
      "and each of those rows says so on itself, per row")
check(not any(r.tier == "comfy_provided" for r in REQUIREMENTS),
      "no requirement uses the comfy_provided tier, which is why filtering "
      "on the tier rather than the flag matched nothing")

# A fresh, *unconfigured* container. The one above has already had its
# paths applied by the first-run section, so it is `configured: true` and
# the install gate refuses it -- correctly. Reusing it here would have
# tested the gate twice and the install not at all.
install_root = Path(tempfile.mkdtemp(prefix="backend-installer-unconf-"))
install_container = build_container(
    Settings(project_root=install_root, db_path=install_root / "backend.db")
)

# An empty project_root is not enough: `paths` is already imported by the
# time this runs, and it resolves the *developer's* ComfyUI, so every
# container in this process reports configured: true. That is the same trap
# `test_path_tiers.py` works around, in the other direction, by copying
# paths.py into a scratch tree.
#
# Forcing `get_comfy_dir` to raise is the honest way to get an unconfigured
# machine in-process -- it is exactly the condition a fresh install has, and
# it is restored below so nothing after this section sees it.
import paths as _paths_module  # noqa: E402

_saved_get_comfy_dir = _paths_module.get_comfy_dir


def _no_comfyui(*_a, **_k):
    raise RuntimeError("Cannot find ComfyUI directory.")


_paths_module.get_comfy_dir = _no_comfyui
check(install_container.services.installer.apply.execute().configured is False,
      "a machine with no ComfyUI is unconfigured, which is the only state "
      "the install gate allows")

# The endpoint must not install into a venv it has not been told about.
class _RecordingInstaller(PackageInstaller):
    def install(self, request, on_line=None):
        recorded["request"] = request
        if on_line:
            on_line("ok")


recorded: dict = {}
install_container, install_app = _with_install(
    install_container,
    StartInstall(installer=_RecordingInstaller(),
                 project_root=install_root, base_python=sys.executable,
                 jobs={}),
)
# torch is never-install, so this is refused on the package list alone --
# *before* the venv is even considered. That ordering matters: the check is
# about ComfyUI's environment, and it must not depend on the caller having
# told us which interpreter that is.
status, _, body = asgi_request(install_app, "/api/v1/installer/install",
                               method="POST",
                               json_body={"target": "comfy",
                                          "packages": ["torch"],
                                          "comfy_venv_python": sys.executable})
check(status == 400,
      f"a request to install torch into ComfyUI's venv is refused even with "
      f"a valid interpreter (got {status})")
message = str((body.get("error") or body.get("detail") or {}).get("message", ""))
check("never-install" in message and "separate virtualenv" in message,
      f"and says why, and what to do instead ({message!r})")

# And with no interpreter either, it is still refused -- for the other reason.
status, _, body = asgi_request(install_app, "/api/v1/installer/install",
                               method="POST",
                               json_body={"target": "comfy",
                                          "packages": ["fastapi"]})
check(status == 400 and "not known" in str(body),
      f"an unknown ComfyUI venv is refused too, rather than guessing one "
      f"(got {status}: {body})")

# With one, it proceeds -- and installs exactly what it was told, so the
# caller's list is never second-guessed into something else.
recorded.clear()
install_container, install_app = _with_install(
    install_container,
    StartInstall(installer=_RecordingInstaller(),
                 project_root=install_root, base_python=sys.executable,
                 jobs={}),
)
status, _, body = asgi_request(install_app, "/api/v1/installer/install",
                               method="POST",
                               json_body={"target": "comfy",
                                          "packages": ["fastapi"],
                                          "constraints": ["torch==2.12.1+xpu"],
                                          "comfy_venv_python": sys.executable})
check(status == 200, f"and proceeds when the venv is known (got {status})")
check(recorded.get("request") is not None
      and recorded["request"].packages == ("fastapi",),
      f"installing exactly the packages it was sent, unaltered "
      f"({recorded.get('request') and recorded['request'].packages})")
check(recorded["request"].constraints == ("torch==2.12.1+xpu",),
      f"with the pins intact ({recorded['request'].constraints})")
check(recorded["request"].target_python == sys.executable,
      f"into the interpreter it was told to, named absolutely "
      f"({recorded['request'].target_python})")

# An unknown job is a 404 with a code, never a fabricated success.
status, _, body = asgi_request(install_app, "/api/v1/installer/install/never-existed")
check(status == 404,
      f"an unknown job id is 404, not a fabricated state (got {status})")
check(body.get("error", {}).get("code") == "install_job_not_found",
      f"with a documented code, so the client can tell it from a lost job "
      f"({body.get('error', {}).get('code')!r})")

# And an empty package list is refused rather than run as a no-op.
status, _, body = asgi_request(install_app, "/api/v1/installer/install",
                               method="POST",
                               json_body={"target": "new", "packages": []})
check(status == 400 and "nothing to install" in str(body).lower(),
      f"an empty install is refused with a sentence (got {status}: {body})")

_paths_module.get_comfy_dir = _saved_get_comfy_dir
check(install_container.services.installer.apply.execute().configured is True,
      "and the forced 'no ComfyUI' is restored, so nothing later in this "
      "process is a statement about a fiction")

finish()
