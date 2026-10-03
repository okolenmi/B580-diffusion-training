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
SERVER_ONLY = {
    "fastapi": "0.139.0",
    "uvicorn": "0.49.0",
    "python-multipart": "0.0.32",
    "tomli_w": "1.2.0",
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
check({"fastapi", "uvicorn", "python-multipart", "tomli_w"} <= names,
      f"the server's four requirements are in the manifest ({sorted(names)})")
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
check(status == 200 and len(body["requirements"]) == 8,
      f"GET installer/manifest 200 with 8 rows (got {status}, "
      f"{len(body.get('requirements', []))})")

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

finish()