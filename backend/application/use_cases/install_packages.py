"""StartInstall / GetInstall -- the first thing that writes anything.

Design doc §3, and the part the wizard was built for. Everything before
this in the installer *looks*; this *acts*, and it is the only operation in
the project that can break something the user already had.

**Why a job and not a request that blocks.** A full install is a ~2.5 GB
download. Holding an HTTP request open for ten minutes would time out the
proxy, the browser, or both, and a half-finished pip with no way to see how
far it got is the worst outcome available. So the request returns a job id
immediately and the page polls. Job id plus polling rather than SSE: the
same shape the graph-execution endpoints already use in this codebase, it
survives a page reload, and every state is fetchable by id after the fact.

**Process-local, deliberately.** Jobs live in this process's memory. A
restart drops them, and a job that cannot be found afterwards is reported
as `interrupted` rather than as success -- a wizard that shows a green tick
because its memory was cleared is worse than one that admits it lost track.
The installation itself is durable (pip's own writes), so only the
*reporting* is lost, and it is durable on disk in the environment itself:
the readiness report answers "is it installed" from dist-info regardless
of what happened to any job.

**The order the wizard uses.** Target and GPU are chosen on screen 2,
because they decide *what is installed and where*. Model paths come after,
on screen 3, because they do not affect the install at all. Applying the
paths makes the machine `configured`, which is what closes the wizard -- so
the install has to happen before it, and it does.
"""

from __future__ import annotations

import logging
import threading
import uuid
from collections.abc import Callable
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

from ..errors import ApplicationError
from ..ports.package_installer import (
    InstallError,
    InstallRequest,
    PackageInstaller,
    create_virtualenv,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ParsedRequirement:
    """One accepted install entry: a name, and optionally one `==` version."""

    text: str
    name: str
    canonical: str
    version: str | None


def canonicalize(name: str) -> str:
    from packaging.utils import canonicalize_name
    return canonicalize_name(name)


def parse_requirement(text: str, *, require_version: bool = False
                      ) -> ParsedRequirement:
    """Parse `name` or `name==version`. Anything else is an `InstallError`.

    **Why this exists at all.** The guard in `StartInstall.execute` used to
    be `forbidden.intersection(packages)` on bare strings, which compares
    the client's text to a set of names. That accepts everything except an
    exact match, so all of these reached pip with `target="comfy"` -- a venv
    this project does not own, and the one operation the whole design
    refuses:

        torch==2.5.0   Torch   torch[opt]   torch>=1   ' torch'
        pytorch-triton-xpu

    A string set membership test cannot see that `Torch` is `torch`. Parsing
    can, and `packaging` -- already a server dependency, for the conflict
    check's version comparisons -- is the right parser rather than a
    hand-rolled regex, because PEP 508's name grammar has edge cases a regex
    gets wrong.

    `packaging` is not sufficient on its own, and the cases it accepts that
    this must not were measured rather than assumed:

    * `' torch'` and `'torch '` -- PEP 508 allows surrounding whitespace, so
      without an explicit check the text handed to pip would not be the text
      that was validated.
    * `torch[opt]` -- extras, which can pull in anything at all.
    * `torch @ https://...` -- a URL, which makes pip fetch from wherever.
    * `torch;python_version<"3"` -- an environment marker.
    * `torch>=1`, `torch~=2.5`, `torch!=2.5`, `torch===2.5` -- specifiers
      other than a pin, which select a version nobody reviewed.

    Names are compared through `canonicalize_name`, so `Torch`, `TORCH` and
    `torch` are one name. `torch_` canonicalises to `torch-`, which is *not*
    in the manifest, so it is refused as an unknown name rather than as a
    spelling of a forbidden one -- refused either way, but the message says
    which, and both reasons are tested.
    """
    from packaging.requirements import InvalidRequirement, Requirement

    raw = text.strip()
    if not raw:
        raise InstallError(
            "an empty entry is not a package name. Nothing was changed.")
    if any(character.isspace() for character in text):
        raise InstallError(
            f"'{text}' contains whitespace. Only 'name' or 'name==version' is "
            f"accepted, with nothing around it. Nothing was changed.")
    if raw.startswith("-"):
        raise InstallError(
            f"'{text}' starts with '-', which makes it a pip option rather "
            f"than a package. Nothing was changed.")

    try:
        requirement = Requirement(raw)
    except InvalidRequirement as exc:
        raise InstallError(
            f"'{text}' is not a valid requirement ({exc}). Only 'name' or "
            f"'name==version' is accepted. Nothing was changed.") from exc

    if requirement.url:
        raise InstallError(
            f"'{text}' names a URL, which would make pip fetch from there. "
            f"Only 'name' or 'name==version' is accepted. Nothing was changed.")
    if requirement.extras:
        raise InstallError(
            f"'{text}' requests extras "
            f"({', '.join(sorted(requirement.extras))}), which can pull in "
            f"anything at all. Only 'name' or 'name==version' is accepted. "
            f"Nothing was changed.")
    if requirement.marker is not None:
        raise InstallError(
            f"'{text}' carries an environment marker ({requirement.marker}), "
            f"so what it installs would depend on where it runs. Only 'name' "
            f"or 'name==version' is accepted. Nothing was changed.")

    operators = sorted({str(spec.operator) for spec in requirement.specifier})
    if any(operator != "==" for operator in operators):
        raise InstallError(
            f"'{text}' pins with {', '.join(operators)} rather than '=='. An "
            f"exact version is what gets reviewed; a range selects one nobody "
            f"looked at. Nothing was changed.")

    versions = [spec.version for spec in requirement.specifier]
    if require_version and not versions:
        raise InstallError(
            f"'{text}' has no version, and this position requires an exact "
            f"pin of the form 'name==version'. Nothing was changed.")

    return ParsedRequirement(
        text=text,
        name=requirement.name,
        canonical=canonicalize(requirement.name),
        version=versions[0] if versions else None,
    )


def validate_install_entries(packages, constraints, target: str) -> None:
    """Refuse anything the manifest does not vouch for, before anything runs.

    Both checks apply to **both** targets, and the allowlist runs first.
    Neither is really about ComfyUI: an entry that is not in the manifest is
    an entry nobody chose on purpose, and a new virtualenv is still an
    environment this project would then be responsible for.

    `pytorch-triton-xpu` needs no manifest row of its own to be refused
    here. It is a torch companion, so it belongs on the `never_install` set
    if it is ever added -- and until then the allowlist refuses it first,
    which is the stronger position.

    Constraints are held to a stricter shape than packages: exactly
    `name==version`. They are written into a file pip reads, and pip honours
    option lines inside such a file, so an entry that is not a pin is an
    entry that could be an instruction.
    """
    from ..ports.requirements_manifest import REQUIREMENTS

    known = {canonicalize(r.distribution) for r in REQUIREMENTS}
    forbidden = {canonicalize(r.distribution)
                 for r in REQUIREMENTS if r.never_install}

    for entry in packages:
        requirement = parse_requirement(entry)
        if requirement.canonical not in known:
            raise InstallError(
                f"'{entry}' is not a package this project declares it needs. "
                f"This installer only installs the project's own manifest, "
                f"because an entry nobody chose is an entry nobody reviewed. "
                f"Known packages: {', '.join(sorted(known))}. Nothing was "
                f"changed.")
        if target == "comfy" and requirement.canonical in forbidden:
            raise InstallError(
                f"{requirement.name} is marked never-install: ComfyUI declares "
                f"and pins it, and installing into that environment would "
                f"change a version it owns. Use a separate virtualenv, where "
                f"this project owns the environment and may install "
                f"anything. Nothing was changed.")

    for line in constraints:
        parse_requirement(line, require_version=True)

#: Job states. `interrupted` is not a failure of the install -- pip may well
#: have finished -- it is a failure of the *report*, and the distinction is
#: kept because "we lost track" and "it broke" call for different words.
QUEUED = "queued"
RUNNING = "running"
SUCCEEDED = "succeeded"
FAILED = "failed"
INTERRUPTED = "interrupted"

TERMINAL = frozenset({SUCCEEDED, FAILED, INTERRUPTED})


@dataclass
class InstallJob:
    """One install, and the log lines it produced.

    Mutated from the worker thread and read from request threads, so every
    field a reader touches is written under `_lock`. The lock is not
    defensive: this is the one place in the installer with two threads,
    and a half-updated job read over HTTP is a plausible and invisible
    corruption.
    """

    id: str
    target_label: str
    packages: tuple[str, ...]
    state: str = QUEUED
    #: The exact pins, and the exact command. Both are in the report
    #: because "we will not change anything already installed" is a claim
    #: about a specific file, and a claim is worth more when the file is
    #: readable.
    constraints: tuple[str, ...] = ()
    command: tuple[str, ...] = ()
    log: deque = field(default_factory=lambda: deque(maxlen=400))
    error: str | None = None
    #: Where packages went, once they have. Needed because "the install
    #: succeeded" does not tell a later server start which interpreter to
    #: use -- that has to be persisted by the wizard's apply step.
    target_python: str | None = None
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "id": self.id,
                "state": self.state,
                "target_label": self.target_label,
                "packages": list(self.packages),
                "constraints": list(self.constraints),
                "command": list(self.command),
                "log": list(self.log),
                "error": self.error,
                "target_python": self.target_python,
                "terminal": self.state in TERMINAL,
            }

    def line(self, text: str) -> None:
        """Append one log line. Called from the worker thread."""
        with self._lock:
            self.log.append(text)

    def set_state(self, state: str, error: str | None = None) -> None:
        with self._lock:
            self.state = state
            if error is not None:
                self.error = error


class JobNotFound(Exception):
    """No such job. Not an error the user caused."""


class InstallJobNotFound(ApplicationError):
    """A job id this process does not know. 404, with a code.

    Its own class rather than `JobNotFound` because the use case raises the
    latter and the route turns it into this one: the port-level exception
    says "not found" and the HTTP one says "not found, with this code, and
    here is what to check instead". Wording that belongs to a user
    response should not be baked into the thing a use case raises.
    """

    code = "install_job_not_found"
    status_code = 404

    def __init__(self, job_id: str) -> None:
        super().__init__(
            f"install job {job_id} is not known to this server. It either "
            f"never existed or was lost when the server restarted -- check "
            f"the readiness report for whether the packages are installed."
        )


class InstallAlreadyRunning(Exception):
    """A second install was asked for while one is in flight.

    Two concurrent pips writing the same virtualenv is not a thing to
    allow and diagnose afterwards, so it is refused at the door with the
    id of the one already running.
    """


@dataclass(slots=True)
class StartInstall:
    """Begin an install and return its job immediately."""

    installer: PackageInstaller
    #: Where a new virtualenv for this project goes. Injected rather than
    #: computed so the wizard and the settings page cannot disagree about
    #: where it is.
    project_root: Path
    #: The interpreter used to *create* a venv. The preflight's, i.e. the
    #: one running the server -- which is the only one known to work.
    base_python: str = ""
    jobs: dict[str, InstallJob] = field(default_factory=dict)
    #: ComfyUI's venv interpreter, as this server finds it.
    #:
    #: A callable rather than a string so it is asked at install time and not
    #: captured when the server started: the wizard's whole job is to *find*
    #: that interpreter, and a value wired in at boot would report on
    #: whichever venv happened to be configured at that moment. `None` means
    #: "this construction cannot detect it", and a `comfy` install then
    #: refuses rather than falling back to the client's value.
    detect_comfy_python: Callable[[], str | None] | None = None
    _active: str | None = None
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    # -- the two outcomes of choosing a target --------------------------

    def plan(self, target: str) -> tuple[str, str]:
        """``(target_python, target_label)`` for a target name.

        Pure: no directory is created and no pip is run here, so the wizard
        can show the plan before anything happens. That separation is why
        `create_virtualenv` lives in the worker and not here.

        The empty interpreter for `comfy` is deliberate and is resolved by
        the caller, which is the only place that knows it: a target is a
        name here, not a path.
        """
        if target == "new":
            return (str(self.project_root / "venv"),
                    "a new virtualenv for this project")
        if target == "comfy":
            return ("", "ComfyUI's virtualenv")
        raise InstallError(
            f"'{target}' is not a target this project installs into. "
            f"Choose a new virtualenv, or ComfyUI's."
        )

    def execute(
        self,
        target: str,
        packages: tuple[str, ...],
        constraints: tuple[str, ...] = (),
        comfy_venv_python: str | None = None,
    ) -> InstallJob:
        """Queue an install. Returns without waiting for it."""
        if not packages:
            raise InstallError(
                "there is nothing to install -- every package this project "
                "needs is already present."
            )

        target_python, label = self.plan(target)

        # Enforced here, not only in the wizard. The filter lives in two
        # places on purpose: the page needs it to render the right list, and
        # the server needs it because this is a browser-driven endpoint
        # with no authentication (ADR 0001) whose whole purpose is to
        # protect a venv this project does not own. A filter that lives
        # only in the client is a request away from being wrong.
        #
        # This is the bug that made it worth adding: a first version of the
        # wizard filtered on `tier != "comfy_provided"`, a tier no
        # requirement uses, so it filtered nothing -- and the request it
        # then sent was accepted, asking to install torch into ComfyUI's
        # virtualenv. The one operation the entire design refuses.
        #
        # It was then a bare set intersection on the client's own text, which
        # `torch==2.5.0`, `Torch`, `torch[opt]`, `torch>=1` and `' torch'`
        # all walked straight past. `validate_install_entries` parses instead.
        validate_install_entries(packages, constraints, target)

        if target == "comfy":
            # The interpreter that runs pip is the server's answer, never the
            # request's. `comfy_venv_python` used to be taken from the client
            # and used verbatim, which made a form field the only thing
            # between this endpoint and an arbitrary interpreter. It stays in
            # the schema because the wizard shows the user which interpreter
            # will run -- but only as something to check the server against,
            # and a disagreement is a refusal rather than a preference.
            detected = (self.detect_comfy_python()
                        if self.detect_comfy_python else None)
            if not detected:
                raise InstallError(
                    "ComfyUI's virtualenv could not be found, so this project "
                    "cannot tell which interpreter it would install into. "
                    "Refusing rather than guessing: 'we cannot show this is "
                    "safe' is the honest answer for an environment it cannot "
                    "identify. Nothing was changed."
                )
            if comfy_venv_python and Path(comfy_venv_python) != Path(detected):
                raise InstallError(
                    f"the request names ComfyUI's interpreter as "
                    f"{comfy_venv_python}, but the server finds it at "
                    f"{detected}. This project installs into the interpreter it "
                    f"detected, not one a caller supplied. Nothing was "
                    f"changed."
                )
            target_python = detected

        job = InstallJob(
            id=uuid.uuid4().hex[:12],
            target_label=label,
            packages=packages,
            constraints=constraints,
        )
        with self._lock:
            if self._active is not None:
                running = self.jobs.get(self._active)
                if running and running.state not in TERMINAL:
                    raise InstallAlreadyRunning(running.id)
            self.jobs[job.id] = job
            self._active = job.id

        thread = threading.Thread(
            target=self._run, args=(job, target, target_python),
            name=f"install-{job.id}", daemon=True,
        )
        thread.start()
        return job

    # -- the worker ----------------------------------------------------

    def _run(self, job: InstallJob, target: str, target_python: str) -> None:
        """Everything that writes happens here, on one thread.

        A thread rather than a subprocess because the work is pip plus a
        venv creation -- both already subprocesses. A daemon so an
        in-flight install cannot keep the server from exiting; a killed
        daemon leaves pip's own writes behind, which is why the readiness
        report is the source of truth afterwards rather than this job.
        """
        job.set_state(RUNNING)
        try:
            if target == "new":
                job.line(
                    f"Creating a virtualenv at {self.project_root / 'venv'}"
                )
                target_python = create_virtualenv(
                    self.base_python or "python3",
                    self.project_root / "venv",
                )
                job.line(f"Created {target_python}")

            request = InstallRequest(
                target_python=target_python,
                packages=job.packages,
                constraints=job.constraints,
                target_label=job.target_label,
            )
            self.installer.install(request, on_line=job.line)
            job.target_python = target_python
            job.set_state(SUCCEEDED)
        except InstallError as exc:
            job.set_state(FAILED, str(exc))
        except Exception as exc:  # noqa: BLE001 -- a job must always finish
            # A worker thread that raises without recording anything leaves
            # the job `running` for ever, and the page polls a spinner that
            # never resolves. So every escape lands here.
            logger.exception("install job %s failed unexpectedly", job.id)
            job.set_state(
                FAILED,
                f"the install failed unexpectedly: "
                f"{type(exc).__name__}: {exc}",
            )
        finally:
            with self._lock:
                if self._active == job.id:
                    self._active = None


@dataclass(slots=True)
class GetInstall:
    """Read a job by id. Cheap, and safe to poll."""

    jobs: dict[str, InstallJob]

    def execute(self, job_id: str) -> dict:
        job = self.jobs.get(job_id)
        if job is None:
            raise JobNotFound(job_id)
        return job.snapshot()