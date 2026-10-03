"""CheckComfyConflicts -- may we install into ComfyUI's virtualenv?

Design doc §3, screen 2. The wizard asks "reuse ComfyUI's venv?" and the
answer has to be computed *before* anything is written, because the answer
is allowed to be no.

**Three outcomes, not two.** The first draft of this had two -- matching or
clashing -- which silently treated "installed but nobody declared it" as
fine. It is not: an installed package is a fact about a machine, a
declaration is a contract its publisher made, and an undeclared package has
nothing vouching for it. So it is reported separately, as *unknown*, and it
is the row that gets the strictest treatment. Treating unknown as safe is
how a soft install quietly becomes the hard one.

**A conflict is the check working, not the check failing.** It means
ComfyUI's own `requirements.txt` and its own venv already disagree. We did
not cause it and we cannot fix it: correcting it means changing a
declaration or changing a version, and both are the user's ComfyUI. So the
report names it, refuses to proceed, and points at the other option. There
is no third path and no resolver.

**The constraints file pins every pre-existing package to its exact
installed version**, not just the declared ones. That is one sentence to
say and one property to have: *nothing already in ComfyUI's venv can change
because of this.* A weaker rule -- constrain only what is declared -- lets
pip move an undeclared package as collateral, which is precisely the case
that has nothing vouching for it.

**Pure.** It reads two sources through one port and returns a report. It
never writes, never runs pip, and never resolves. The only side effect in
this area is `ApplyInstallation`, and it is gated.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..ports.comfy_environment import ComfyEnvironment, normalise
from ..ports.requirements_manifest import COMFY_ADDITIONS

#: The three ways a package can stand in relation to ComfyUI's declaration.
#: Named as constants because the wizard renders three different treatments
#: and a typo in a string here is a typo in the UI.
SAFE = "safe"
CONFLICT = "conflict"
UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class Finding:
    """One package in ComfyUI's venv, checked against ComfyUI's own file."""

    name: str
    installed_version: str | None
    declared_specifier: str | None
    outcome: str

    @property
    def safe(self) -> bool:
        return self.outcome in (SAFE, UNKNOWN)

    @property
    def declared(self) -> bool:
        """Whether ComfyUI's file mentions this package on *this* machine.

        False for `unknown`, which is the point: undeclared and
        declared-but-unconstrained are different facts, and the wizard
        renders them differently.
        """
        return self.declared_specifier is not None

    def describe(self) -> str:
        """The sentence the wizard shows. Says which of the three it is."""
        if self.outcome == CONFLICT:
            return (
                f"{self.name} {self.installed_version} is installed, but "
                f"ComfyUI requires {self.declared_specifier}. ComfyUI's own "
                f"environment is already inconsistent; installing into it "
                f"would not be the thing that broke it, but it would not be "
                f"safe either."
            )
        if self.outcome == UNKNOWN:
            return (
                f"{self.name} {self.installed_version} is installed but "
                f"ComfyUI's requirements.txt does not mention it. Nothing "
                f"declares it, so it will be pinned to exactly this version "
                f"and nothing may change it."
            )
        return (
            f"{self.name} {self.installed_version} satisfies "
            f"{self.declared_specifier or 'any version'}."
        )


@dataclass(frozen=True, slots=True)
class AdditionFinding:
    """One of our four, against ComfyUI's declarations.

    `declared_specifier` distinguishes three states, which is why it is not
    a boolean:

    ``None``
        ComfyUI does not mention it. Adding it cannot break a declaration,
        because there is none to break.
    ``""``
        ComfyUI declares it with no version -- a bare ``torch`` line. Any
        version satisfies that, so it blocks nothing either. Treating this
        as a conflict would refuse a correct install over a line that pins
        no version, which is the common case for the accelerator stack.
    ``">=1.0"``
        ComfyUI pins it, so adding our own version means either breaking
        their declaration or failing to satisfy ours. Refuse.
    """

    name: str
    declared_specifier: str | None

    @property
    def blocked(self) -> bool:
        return bool(self.declared_specifier)

    @property
    def declared(self) -> bool:
        return self.declared_specifier is not None

    def describe(self) -> str:
        if not self.declared:
            return (
                f"{self.name} is not declared by ComfyUI, so adding it "
                f"cannot break a declaration."
            )
        if not self.declared_specifier:
            return (
                f"{self.name} is declared by ComfyUI with no version "
                f"constraint, so any version satisfies it."
            )
        return (
            f"This project also needs {self.name}, and ComfyUI requires "
            f"{self.declared_specifier}. Installing ours would change a "
            f"version ComfyUI declares."
        )


@dataclass(frozen=True, slots=True)
class ConflictReport:
    """Everything screen 2 needs to decide, and to explain the decision."""

    comfy_dir: str
    findings: tuple[Finding, ...]
    additions: tuple[AdditionFinding, ...]
    constraints: tuple[str, ...]
    requirements_path: str | None
    venv_python: str | None = None
    #: Set when a source could not be read at all. Its presence means the
    #: check could not run, which is reported as unsafe -- "we could not
    #: check" and "we checked and it is fine" must never look alike.
    requirements_error: str | None = None
    venv_error: str | None = None

    @property
    def conflicts(self) -> tuple[Finding, ...]:
        return tuple(f for f in self.findings if f.outcome == CONFLICT)

    @property
    def unknowns(self) -> tuple[Finding, ...]:
        return tuple(f for f in self.findings if f.outcome == UNKNOWN)

    @property
    def blocked_additions(self) -> tuple[AdditionFinding, ...]:
        return tuple(a for a in self.additions if a.blocked)

    @property
    def checked(self) -> bool:
        """Whether both sources were actually read."""
        return not self.requirements_error and not self.venv_error

    @property
    def safe(self) -> bool:
        """May we install into this venv?

        False on: any conflict, any of our four already declared, and any
        source that could not be read. All three are refusals with a
        different explanation, and the wizard shows which.
        """
        return (
            self.checked
            and not self.conflicts
            and not self.blocked_additions
        )

    def refusal_reason(self) -> str | None:
        """Why not, in one sentence, or None when it is safe."""
        if not self.checked:
            parts = [e for e in (self.requirements_error, self.venv_error) if e]
            return "Cannot check whether this is safe: " + " ".join(parts)
        if self.conflicts:
            first = self.conflicts[0]
            return (
                f"{first.name} {first.installed_version} is installed but "
                f"ComfyUI requires {first.declared_specifier}. ComfyUI's own "
                f"environment is already inconsistent, and we will not change "
                f"a version ComfyUI declares."
            )
        if self.blocked_additions:
            # Not `first` again: the earlier one is a Finding and this is an
            # AdditionFinding. mypy caught the reuse, which is the only thing
            # it has ever caught that mattered here.
            addition = self.blocked_additions[0]
            return (
                f"{addition.name} is needed by this project and is declared by "
                f"ComfyUI as {addition.declared_specifier}. Adding our own "
                f"version would change a version ComfyUI declares."
            )
        return None


class CheckComfyConflicts:
    """Read both sources and classify. Never writes."""

    def __init__(self, environment: ComfyEnvironment) -> None:
        self._environment = environment

    def execute(
        self, comfy_dir: str, venv_python: str | None = None
    ) -> ConflictReport:
        info = self._environment.read(comfy_dir, venv_python)
        declarations = info.declarations_by_key
        installed = info.installed_by_key

        findings: list[Finding] = []
        for name, version in sorted(installed.items()):
            declaration = declarations.get(name)
            findings.append(
                self._classify(name, version, declaration)
            )

        additions = tuple(
            AdditionFinding(
                name=addition,
                declared_specifier=(
                    declarations[normalise(addition)].specifier
                    if normalise(addition) in declarations
                    else None
                ),
            )
            for addition in COMFY_ADDITIONS
        )

        return ConflictReport(
            comfy_dir=comfy_dir,
            findings=tuple(findings),
            additions=additions,
            # Every pre-existing package, declared or not, pinned exactly.
            # See the module docstring: this is the whole safety property in
            # one line, and a shorter rule lets pip move undeclared packages
            # as collateral.
            constraints=tuple(
                f"{name}=={version}" for name, version in sorted(installed.items())
            ),
            requirements_path=info.requirements_path,
            venv_python=info.venv_python or getattr(self._environment, "venv_python", None),
            requirements_error=info.requirements_error,
            venv_error=info.venv_error,
        )

    def _classify(self, name: str, version: str, declaration) -> Finding:
        """Decide which of the three this is.

        The ordering matters: *declared* is asked before *satisfied*, because
        a package that is installed and undeclared is `unknown` and must not
        be recorded as safe by default.
        """
        if declaration is None:
            return Finding(
                name=name,
                installed_version=version,
                declared_specifier=None,
                outcome=UNKNOWN,
            )

        # An environment marker that does not apply here is not a
        # declaration about this machine. Treating it as one would refuse an
        # install over a Windows-only line.
        if not self._marker_applies(declaration.marker):
            return Finding(
                name=name,
                installed_version=version,
                declared_specifier=None,
                outcome=UNKNOWN,
            )

        specifier = declaration.specifier
        if not specifier:
            # `torch` with no version -- declared, and anything satisfies it.
            return Finding(
                name=name,
                installed_version=version,
                declared_specifier="",
                outcome=SAFE,
            )

        satisfied = self._satisfies(version, specifier)
        return Finding(
            name=name,
            installed_version=version,
            declared_specifier=specifier,
            outcome=SAFE if satisfied else CONFLICT,
        )

    @staticmethod
    def _marker_applies(marker: str | None) -> bool:
        """Whether a requirements-file environment marker holds here.

        Evaluated in the *server's* interpreter, which is a second
        approximation and the honest one to state: the marker describes the
        machine installing the package, and on a machine where the two
        differ, the installer's answer is the wrong one. Such a machine is
        not a configuration this project supports, and the report says which
        lines it could not evaluate rather than silently ignoring them.
        """
        if not marker:
            return True
        from packaging.requirements import Requirement

        try:
            parsed = Requirement(f"distillation-probe; {marker}").marker
            # `.evaluate()`, not `bool()`. An earlier version wrote
            # `bool(marker)`, and `Marker` defines no `__bool__` -- so that
            # was `True` for *every* marker, and the branch below was
            # unreachable. A Windows-only line was therefore treated as a
            # declaration about this Linux machine. Caught by the test that
            # feeds it `sys_platform == "win32"`.
            #
            # A marker that parsed to nothing is treated as not applying,
            # same as one that raised: neither is a declaration this check
            # can stand behind.
            return parsed is not None and bool(parsed.evaluate())
        except Exception:  # noqa: BLE001 -- an unreadable marker is not a pass
            return False

    @staticmethod
    def _satisfies(version: str, specifier: str) -> bool:
        """PEP 440 containment, via `packaging`.

        **`prereleases=True` is deliberate and is a deviation from the strict
        reading.** On this machine numpy `2.5.0rc1` is installed against a
        declared `>=1.25.0`, and PEP 440 says a prerelease does not satisfy a
        `>=` range. Reporting that as a conflict would refuse a correct
        install and tell the user their ComfyUI is broken when it is
        running. The version *is* above the floor; only the prerelease rule
        objects, and that rule exists to stop an installer from *choosing* a
        prerelease, which is not what this check is doing -- it is reading a
        version someone already chose.

        An unparseable installed version is *not* treated as satisfying: the
        answer to "is this version acceptable" is no when the version cannot
        be read, because saying yes would let an unknown into a check whose
        entire job is refusing what it cannot vouch for.
        """
        from packaging.specifiers import InvalidSpecifier, SpecifierSet
        from packaging.version import InvalidVersion, Version

        try:
            return SpecifierSet(specifier).contains(Version(version), prereleases=True)
        except (InvalidSpecifier, InvalidVersion):
            return False