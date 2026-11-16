"""InspectEnvironment: which of the project's packages an environment has, and what installing the rest would have to respect.

For an existing environment every installed package is pinned in the install, so pip can add what is missing but cannot move
what is already there. For a new environment nothing exists yet and everything is missing. Never writes.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

from packaging.utils import canonicalize_name as canonicalize

from ..ports.requirements_manifest import FULL_INSTALL, REQUIREMENTS, TIERS, total_approx_mb
from ..ports.venv_inspection import VenvDevice, VenvInspector, find_interpreter


@dataclass(frozen=True, slots=True)
class PackageState:
    name: str
    tier: str
    why: str
    approx_mb: int | None
    installed_version: str | None


@dataclass(frozen=True, slots=True)
class EnvironmentReport:
    kind: str                      # "new" or "existing"
    venv_dir: str
    interpreter: str | None
    python_version: str | None
    error: str | None
    packages: tuple[PackageState, ...]
    constraints: tuple[str, ...]
    installed_count: int
    #: Reasons the environment cannot serve a run even after the missing packages are added.
    problems: tuple[str, ...]
    #: Installed packages the install will replace, as "name version". Only ever in the project's own environment.
    replaces: tuple[str, ...] = ()

    @property
    def to_install(self) -> tuple[str, ...]:
        return tuple(p.name for p in self.packages if p.installed_version is None)

    @property
    def download_mb(self) -> int:
        return sum(p.approx_mb or 0 for p in self.packages if p.installed_version is None)

    @property
    def usable(self) -> bool:
        return self.error is None and not self.problems


def is_xpu_build(version: str) -> bool:
    return "+xpu" in version


class InspectEnvironment:
    def __init__(self, inspector: VenvInspector, project_root: Path) -> None:
        self._inspector = inspector
        self._project_root = project_root

    @property
    def new_venv_dir(self) -> Path:
        return self._project_root / "venv"

    def for_target(self, target: str, venv_dir: str | None) -> EnvironmentReport:
        """The report for ``"new"`` (the project's own venv folder) or ``"existing"`` (a folder the user names)."""
        if target == "new":
            return self.plan_new()
        if target == "existing":
            if not (venv_dir or "").strip():
                return EnvironmentReport("existing", "", None, None, "choose the folder of the virtualenv to use", self._states({}), (), 0, ())
            return self.inspect_existing(venv_dir.strip())
        raise ValueError(f"'{target}' is not an install target; use 'new' or 'existing'")

    def plan_new(self) -> EnvironmentReport:
        """A fresh environment inside the project.

        A venv already in that folder is the project's own, so it is reused, and a torch in it that is not the XPU build is replaced
        rather than treated as a blocker: unlike a user's environment, nothing there needs protecting from it.
        """
        target = self.new_venv_dir
        if find_interpreter(str(target)):
            report = replace(self.inspect_existing(str(target)), kind="new")
            wrong = self._wrong_torch(report)
            if wrong is None:
                return report
            version = wrong.installed_version
            packages = tuple(replace(p, installed_version=None) if p is wrong else p for p in report.packages)
            return replace(report, packages=packages, constraints=(), problems=(), replaces=(f"torch {version}",))
        problems = ()
        if target.exists() and any(target.iterdir()):
            problems = (f"{target} exists and is not a virtualenv; move it aside or choose an existing environment.",)
        return EnvironmentReport("new", str(target), None, None, None, self._states({}), (), 0, problems)

    def verify_device(self, venv_dir: str) -> VenvDevice:
        return self._inspector.device(venv_dir)

    def inspect_existing(self, venv_dir: str) -> EnvironmentReport:
        inventory = self._inspector.inspect(venv_dir)
        installed = {canonicalize(name): version for name, version in inventory.packages}
        problems = []
        torch = installed.get("torch")
        if torch and not is_xpu_build(torch):
            problems.append(f"torch {torch} in this environment is not the Intel XPU build, so the card cannot be used; "
                            "use a new environment, or install torch from the XPU index there first")
        return EnvironmentReport(
            "existing", venv_dir, inventory.interpreter, inventory.python_version, inventory.error,
            self._states(installed),
            tuple(f"{canonicalize(name)}=={version}" for name, version in inventory.packages if version),
            len(inventory.packages), tuple(problems),
        )

    @staticmethod
    def _wrong_torch(report: EnvironmentReport) -> PackageState | None:
        for package in report.packages:
            if canonicalize(package.name) == "torch" and package.installed_version and not is_xpu_build(package.installed_version):
                return package
        return None

    @staticmethod
    def _states(installed: dict[str, str]) -> tuple[PackageState, ...]:
        return tuple(
            PackageState(r.distribution, r.tier, r.why, r.approx_mb, installed.get(canonicalize(r.distribution)))
            for r in REQUIREMENTS
        )



class DescribeRequirements:
    """The manifest as data, for ``GET /manifest``."""

    def execute(self) -> dict[str, object]:
        return {
            "requirements": [{"name": r.distribution, "tier": r.tier, "why": r.why, "approx_mb": r.approx_mb} for r in REQUIREMENTS],
            "tiers": list(TIERS),
            "full_install": list(FULL_INSTALL),
            "full_install_approx_mb": total_approx_mb(),
        }


class TokenizerSetup:
    """Which tokenizer file sets are in place, and fetching the missing ones."""

    def __init__(self, files) -> None:
        self._files = files

    def status(self) -> list[dict[str, object]]:
        from ..ports.tokenizer_files import SETS

        rows = []
        for item in SETS:
            present = self._files.present(item.key)
            rows.append({"key": item.key, "label": item.label, "needed_for": item.needed_for,
                         "files": list(item.files), "present": list(present), "complete": len(present) == len(item.files)})
        return rows

    def download(self, key: str) -> dict[str, object]:
        self._files.download(key)
        return next(row for row in self.status() if row["key"] == key)
