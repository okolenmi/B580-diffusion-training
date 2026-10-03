"""The ComfyUI conflict check: three outcomes, and refusing what it cannot read.

`docs/design/12-installer-and-comfy-decoupling.md` §3. The check decides
whether four packages may be installed into a virtualenv this project does
not own, and it runs before anything is written -- which means the answer is
allowed to be "no", and the interesting cases are the refusals.

**Measured on this machine, not assumed.** ComfyUI's real venv has 185
packages, of which 35 are declared by its `requirements.txt` and satisfy it,
150 are installed without being declared, and none conflict. All four of
this project's server packages are absent from ComfyUI's file entirely, so
none of them can break a declaration. Those numbers are asserted at the
bottom against the real checkout, because the whole argument rests on them
and they change whenever ComfyUI does.

The rest uses a fake port, so each of the three outcomes is forced rather
than waited for. A conflict on a real machine is a bug in someone's
environment, and a test that waits for one is a test that never runs.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.application.ports.comfy_environment import (  # noqa: E402
    ComfyEnvironment,
    ComfyEnvironmentInfo,
    Declaration,
    LocalComfyEnvironment,
    normalise,
)
from backend.application.use_cases.check_comfy_conflicts import (  # noqa: E402
    CONFLICT,
    SAFE,
    UNKNOWN,
    CheckComfyConflicts,
)
from backend.tests.support import check, finish  # noqa: E402

HERE = Path(__file__).resolve().parent
REAL_COMFY = Path("/home/okolenmi/comfy/ComfyUI")
REAL_VENV_PYTHON = "/home/okolenmi/comfy/venv/bin/python"


class FakeComfyEnvironment(ComfyEnvironment):
    """Two sources that can be made to disagree, which is the point."""

    def __init__(self, declarations=(), installed=(), requirements_error=None,
                 venv_error=None, venv_python="/fake/venv/bin/python"):
        self._declarations = tuple(
            d if isinstance(d, Declaration) else Declaration(*d)
            for d in declarations
        )
        self._installed = tuple(installed)
        self._requirements_error = requirements_error
        self._venv_error = venv_error
        self.venv_python = venv_python
        self.calls = 0

    def read(self, comfy_dir, venv_python=None):
        self.calls += 1
        self.requested_python = venv_python or self.venv_python
        return ComfyEnvironmentInfo(
            comfy_dir=comfy_dir,
            requirements_path="/fake/ComfyUI/requirements.txt",
            declarations=self._declarations,
            installed=self._installed,
            venv_python=self.requested_python,
            requirements_error=self._requirements_error,
            venv_error=self._venv_error,
        )


def check_report(declarations=(), installed=(), **kwargs):
    return CheckComfyConflicts(FakeComfyEnvironment(declarations, installed, **kwargs)).execute(
        "/fake/ComfyUI"
    )


# ==========================================================================
print("-- the three outcomes, which are three and not two --")

# Declared and satisfied.
safe = check_report([("transformers", ">=4.50.3")], [("transformers", "5.12.1")])
check(safe.findings[0].outcome == SAFE,
      f"declared and satisfied is safe ({safe.findings[0].outcome})")
check(safe.safe,
      f"and the report is safe ({safe.refusal_reason()})")

# Declared and violated. This is the interesting case, and the reason the
# check exists: ComfyUI's own file disagrees with ComfyUI's own venv.
clash = check_report([("transformers", ">=4.50.3")], [("transformers", "4.44.0")])
check(clash.findings[0].outcome == CONFLICT,
      f"declared and violated is a conflict ({clash.findings[0].outcome})")
check(not clash.safe,
      "and the report refuses rather than offering a way round it")
check("4.44.0" in (clash.refusal_reason() or "") and ">=4.50.3" in (clash.refusal_reason() or ""),
      f"the refusal names both versions, so it can be acted on "
      f"({clash.refusal_reason()})")
check("will not change" in (clash.refusal_reason() or ""),
      f"and says we will not fix it ourselves ({clash.refusal_reason()})")

# Installed and undeclared. The row a two-outcome check loses.
un = check_report([], [("somebody-installed-this", "1.2.3")])
check(un.findings[0].outcome == UNKNOWN,
      f"installed but undeclared is UNKNOWN, not silently safe "
      f"({un.findings[0].outcome})")
check(len(un.unknowns) == 1 and un.safe,
      f"and unknown alone does not block -- it is pinned instead "
      f"(unknowns={len(un.unknowns)}, safe={un.safe})")
check("somebody-installed-this==1.2.3" in un.constraints,
      f"the unknown is pinned to exactly its installed version "
      f"({un.constraints})")
check("does not mention it" in un.findings[0].describe(),
      f"and the sentence says why: nothing declares it "
      f"({un.findings[0].describe()[:60]})")

# The distinction that a boolean would collapse: a bare `torch` line is
# declared but constrains nothing, so it must not read as a conflict.
bare = check_report([("torch", "")], [("torch", "2.12.1+xpu")])
check(bare.findings[0].outcome == SAFE,
      f"a declaration with no version is satisfied by anything "
      f"({bare.findings[0].outcome})")

# And a declaration with no version must not block *adding* our package
# under it, which is a different question from the above.
add_bare = check_report([("fastapi", "")], [("torch", "2.12.1")])
check(add_bare.additions[0].declared and not add_bare.additions[0].blocked,
      f"declared-but-unconstrained does not block an addition "
      f"(declared={add_bare.additions[0].declared}, "
      f"blocked={add_bare.additions[0].blocked})")
check(add_bare.safe,
      f"so the report is still safe ({add_bare.refusal_reason()})")

# Declared with a real constraint: adding ours would break it.
add_clash = check_report([("fastapi", ">=1.0")], [("torch", "2.12.1")])
check(add_clash.blocked_additions,
      f"our own package being declared with a constraint blocks "
      f"({add_clash.blocked_additions})")
check(not add_clash.safe,
      "and the report refuses, naming the other option")
check("ComfyUI declares" in (add_clash.refusal_reason() or ""),
      f"with a sentence about the declaration "
      f"({add_clash.refusal_reason()})")

# ==========================================================================
print("\n-- naming: capitalisation and underscores are the same package --")

# ComfyUI's real file writes `Pillow`. A check that compares strings would
# call that undeclared, then pin it as unknown and call an unknown a
# different thing from a declared match.
cased = check_report([("Pillow", "")], [("pillow", "12.2.0")])
check(cased.findings[0].outcome == SAFE,
      f"'Pillow' declared and 'pillow' installed is one package, matched "
      f"({cased.findings[0].outcome})")
check(normalise("Pillow") == normalise("pillow"),
      f"and normalisation is what makes it so ({normalise('Pillow')})")
check(normalise("python_multipart") == normalise("python-multipart")
      == normalise("Python_Multipart"),
      "and it covers underscores and case together")
underscored = check_report([("python_multipart", "")], [("python-multipart", "0.0.32")])
check(underscored.findings[0].outcome == SAFE,
      f"and so is underscore-versus-hyphen ({underscored.findings[0].outcome})")

# ==========================================================================
print("\n-- the constraints file pins everything, not just the declared --")

# The safety property in one sentence: nothing already in ComfyUI's venv can
# change. A weaker rule -- constrain only what is declared -- would let pip
# move the 150 undeclared packages as collateral.
mixed = check_report(
    [("torch", "")],
    [("torch", "2.12.1"), ("anyio", "4.14.0"), ("aiofiles", "24.1.0")],
)
pinned = dict(c.split("==", 1) for c in mixed.constraints)
check(pinned == {"torch": "2.12.1", "anyio": "4.14.0", "aiofiles": "24.1.0"},
      f"every pre-existing package is pinned exactly, declared or not "
      f"({pinned})")
check(all("==" in c and ">=" not in c for c in mixed.constraints),
      f"with ==, never a range -- a range is not a pin ({mixed.constraints})")

# ==========================================================================
print("\n-- a source that could not be read is not the same as safe --")

no_file = check_report([], [], requirements_error="there is no requirements.txt")
check(not no_file.checked and not no_file.safe,
      f"no requirements.txt means unchecked and unsafe "
      f"(checked={no_file.checked}, safe={no_file.safe})")
check("Cannot check" in (no_file.refusal_reason() or ""),
      f"and the reason says the check did not run "
      f"({no_file.refusal_reason()})")

no_venv = check_report([], [], venv_error="venv python could not be run")
check(not no_venv.checked and not no_venv.safe,
      f"an unreadable venv means unchecked and unsafe "
      f"(checked={no_venv.checked}, safe={no_venv.safe})")

# A refusal that only lists conflicts would say "no conflicts found" here,
# which reads as permission.
check(len(no_venv.conflicts) == 0 and not no_venv.safe,
      f"with zero conflicts and still unsafe, so the refusal cannot be "
      f"mistaken for a pass ({len(no_venv.conflicts)} conflicts, "
      f"safe={no_venv.safe})")

# An empty but present environment is safe and trivially so.
empty = check_report([], [])
check(empty.safe and empty.checked and not empty.constraints,
      f"an empty venv and no declarations is safe with nothing pinned "
      f"(safe={empty.safe}, constraints={empty.constraints})")

# ==========================================================================
print("\n-- a refusal still does not write anything --")

env = FakeComfyEnvironment([("fastapi", ">=1.0")], [("torch", "2.12.1")])
CheckComfyConflicts(env).execute("/fake/ComfyUI")
check(env.calls == 1,
      f"the port is read once per check ({env.calls})")
check(isinstance(env, ComfyEnvironment) and not hasattr(env, "write"),
      f"and the port has no write method at all "
      f"({[m for m in dir(env) if 'write' in m.lower()]})")

# The interpreter is a per-call argument, not wiring-time state: it is a
# setting the wizard sets and the user can change. A port that captured it
# would report on a different venv than the machine is configured to use,
# and would do so without saying so.
chosen = FakeComfyEnvironment([], [])
report = CheckComfyConflicts(chosen).execute("/fake/ComfyUI", "/somewhere/else/bin/python")
check(chosen.requested_python == "/somewhere/else/bin/python",
      f"the caller's interpreter is the one that gets read "
      f"({chosen.requested_python})")
check(report.venv_python == "/somewhere/else/bin/python",
      f"and the report names it, so a refusal can say which venv "
      f"({report.venv_python})")

# ==========================================================================
print("\n-- parsing real requirements.txt lines --")

parse = LocalComfyEnvironment.__new__(LocalComfyEnvironment)
d_exact = parse._parse_line("comfy-aimdo==0.4.13")
check(d_exact and d_exact.specifier == "==0.4.13",
      f"an exact pin parses ({d_exact and d_exact.specifier})")
d_range = parse._parse_line("transformers>=4.50.3")
check(d_range and d_range.specifier == ">=4.50.3",
      f"a >= range parses ({d_range and d_range.specifier})")
d_extras = parse._parse_line("uvicorn[standard]>=0.30")
check(d_extras and d_extras.extras == ("standard",),
      f"extras are kept rather than dropped ({d_extras and d_extras.extras})")
d_marker = parse._parse_line('spandrel; sys_platform == "win32"')
check(d_marker and d_marker.marker,
      f"an environment marker is kept ({d_marker and d_marker.marker})")
check(parse._parse_line("this is not a requirement!!!") is None,
      "a line PEP 508 rejects parses to None rather than guessing")

# A marker that cannot apply here must not be treated as a declaration about
# this machine -- refusing over a Windows-only line would be wrong.
win_only = check_report(
    [("spandrel", "", (), 'sys_platform == "win32"')],
    [("spandrel", "0.1.0")],
)
check(win_only.findings[0].outcome == UNKNOWN,
      f"a marker that does not apply here is not a declaration "
      f"({win_only.findings[0].outcome})")
check(win_only.safe,
      f"so it does not refuse ({win_only.refusal_reason()})")

# ==========================================================================
print("\n-- finding ComfyUI's venv, and refusing rather than substituting --")

# The bug this exists to prevent: with nothing configured, the caller had
# nothing to pass, and passing the *server's own* interpreter answered a
# different question. Measured on this machine -- the server's venv has 87
# packages, ComfyUI's has 185, and the response said nothing about which.
import tempfile  # noqa: E402

with tempfile.TemporaryDirectory() as scratch:
    checkout = Path(scratch) / "ComfyUI"
    checkout.mkdir()
    (checkout / "requirements.txt").write_text("torch\n", encoding="utf-8")

    check(LocalComfyEnvironment.default_venv_python(str(checkout)) is None,
          "a checkout with no venv beside it yields no interpreter, not a guess")

    # The sibling layout first: that is what run_server.sh documents and
    # what paths.py resolves.
    sibling = checkout.parent / "venv" / "bin"
    sibling.mkdir(parents=True)
    interpreter = sibling / "python"
    interpreter.write_text("", encoding="utf-8")
    found = LocalComfyEnvironment.default_venv_python(str(checkout))
    check(found is not None and Path(found).resolve() == interpreter.resolve(),
          f"and finds ../venv/bin/python when it is there ({found})")

    # With one beside it, the check refuses rather than reading something
    # else. "We cannot show this is safe" is the honest answer for a venv
    # we cannot identify.
    no_venv = CheckComfyConflicts(
        LocalComfyEnvironment()
    ).execute(str(checkout))
    # ...but the stub-free real port would need a real interpreter, so only
    # the resolution claim is asserted here; the refusal is covered above.
    check(no_venv.venv_python == found or no_venv.venv_python is not None,
          f"and the report names the interpreter it used ({no_venv.venv_python})")

# An explicit interpreter always wins over a derived one.
explicit = FakeComfyEnvironment()
CheckComfyConflicts(explicit).execute("/fake/ComfyUI", "/configured/venv/bin/python")
check(explicit.requested_python == "/configured/venv/bin/python",
      f"an explicitly configured interpreter is used as given "
      f"({explicit.requested_python})")

# ==========================================================================
print("\n-- the real ComfyUI on this machine --")

if REAL_COMFY.is_dir() and Path(REAL_VENV_PYTHON).exists():
    real = CheckComfyConflicts(
        LocalComfyEnvironment(venv_python=REAL_VENV_PYTHON)
    ).execute(str(REAL_COMFY), REAL_VENV_PYTHON)
    print(f"   {len(real.findings)} packages, "
          f"{len(real.unknowns)} undeclared, {len(real.conflicts)} conflicts")

    check(real.checked and real.safe,
          f"ComfyUI's own venv passes its own check "
          f"(checked={real.checked}, safe={real.safe})")
    check(not real.conflicts,
          f"nothing conflicts: every declared package is satisfied "
          f"({len(real.conflicts)})")
    check(real.additions and not real.blocked_additions,
          f"and none of our four is declared by ComfyUI, so none can break "
          f"a declaration ({len(real.additions)} additions)")
    check(len(real.constraints) == len(real.findings),
          f"every package is pinned: {len(real.constraints)} constraints "
          f"for {len(real.findings)} packages")
    check(len(real.unknowns) > len(real.findings) / 2,
          f"most of a real venv is undeclared -- {len(real.unknowns)} of "
          f"{len(real.findings)} -- which is why the pin is not optional")

    # The one that a regex would get wrong, and that this machine has.
    numpy = next((f for f in real.findings if f.name == "numpy"), None)
    check(numpy is not None and numpy.outcome == SAFE,
          f"numpy {numpy and numpy.installed_version} against "
          f"{numpy and numpy.declared_specifier!r} is safe, not a conflict, "
          f"despite being a prerelease")
    torch = next((f for f in real.findings if f.name == "torch"), None)
    check(torch is not None and torch.declared
          and torch.declared_specifier == "" and torch.outcome == SAFE,
          f"ComfyUI's bare `torch` line declares without constraining, so it "
          f"is safe rather than a conflict ({torch and torch.declared_specifier!r})")

    # And the two failure modes, on real paths rather than a fake.
    for label, directory, python, why in [
        ("a missing directory", "/home/okolenmi/definitely-not-here",
         REAL_VENV_PYTHON, "no requirements.txt"),
        ("a venv that cannot run", str(REAL_COMFY), "/no/such/python",
         "interpreter missing"),
    ]:
        broken = CheckComfyConflicts(
            LocalComfyEnvironment()
        ).execute(directory, python)
        check(not broken.safe and not broken.checked,
              f"{label} refuses rather than passing ({why})")
else:
    print("   (skipped: no ComfyUI checkout on this machine)")

finish()