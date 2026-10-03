"""What this project needs, as data rather than prose.

`docs/design/11-first-run-and-installer.md` §1 asks for a
machine-readable list, and the reason it matters is that the answer
currently lives in three places that cannot see each other:
`requirements.txt` (four lines, the server's own imports), the source
(what the trainer imports, which is four *more* packages that appear in no
requirements file at all), and `docs/setup.md` (which names neither set).

Measured on this machine, by reading the source rather than by guessing:

    server   fastapi, uvicorn, python-multipart, tomli_w
             (requirements.txt, exactly)
    training torch, numpy, pillow, safetensors
             (imported by nodes/ and manager/; in NO requirements file,
             because they are expected to come from ComfyUI's venv)

That gap is the thing the installer exists to close: a user who installs
`requirements.txt` into a fresh venv gets a server that starts and a
training stack that is silently absent.

**The four tiers**, which are the design's section 1 list:

``required``
    The server does not start without it. It is running, so by the time
    anything asks, these are all present -- which is exactly why their
    absence is not worth a row in the wizard's blocking list.
``training``
    The UI works and a run does not. This is the tier that matters: it is
    what makes the difference between "the server is up" and "you can
    train", and nothing currently reports on it.
``optional``
    A capability that degrades quietly. Absence is reported and never
    blocks.
``comfy_provided``
    torch and the accelerator stack. Not installed into a venv this
    project owns -- they are exactly what ComfyUI already has and pins,
    and listing them anywhere an installer would act is how "ComfyUI broke
    and the installer did it" happens.

Sizes are on the rows because the venv decision is a *disk* decision
(§2): "torch is missing" is a fact, "torch is missing and it is 2.5 GB" is
what lets someone choose the new-venv option knowingly.
"""

from __future__ import annotations

from dataclasses import dataclass

#: The four tiers, as the design states them.
REQUIRED = "required"
TRAINING = "training"
OPTIONAL = "optional"
COMFY_PROVIDED = "comfy_provided"

TIERS: tuple[str, ...] = (REQUIRED, TRAINING, OPTIONAL, COMFY_PROVIDED)

#: Approximate download size in megabytes, where it is knowable and stable
#: enough to plan a disk decision with. None where it is not -- and None is
#: honest here, where a made-up number would be worse than none.
_MB = 1000


@dataclass(frozen=True, slots=True)
class Requirement:
    """One thing this project needs.

    `distribution` is the pip name, which is also the name
    `importlib.metadata` reports, so presence is one dict lookup.
    `import_name` is only here for the error message: "python-multipart is
    missing" and "multipart is missing" name the same package and only one
    of them is what the user would type.
    """

    distribution: str
    tier: str
    why: str
    approx_mb: int | None = None
    #: Do not install this into a venv this project does not own.
    #:
    #: Set on the `training` rows -- torch, numpy, safetensors, pillow --
    #: and *not* on `comfy_provided`, which this field's docstring used to
    #: claim and which no requirement actually uses. The flag is the
    #: load-bearing one: it keeps the accelerator stack out of an install
    #: aimed at ComfyUI's environment. It is per-row rather than per-tier
    #: because a package can be ours to install in a venv we own and not
    #: ours to install in one we do not.
    #:
    #: Reading it as a tier is how the wizard first came to offer to
    #: install torch into ComfyUI's virtualenv: filtering on
    #: `tier != "comfy_provided"` matched nothing, because that tier is
    #: empty.
    never_install: bool = False

    @property
    def import_name(self) -> str:
        from .environment import IMPORT_NAMES

        return IMPORT_NAMES.get(self.distribution, self.distribution)


#: The manifest. Order is the order a wizard should ask about it: what the
#: server needs, then what training needs, then what is a nice-to-have.
REQUIREMENTS: tuple[Requirement, ...] = (
    # -- the server ------------------------------------------------------
    Requirement("fastapi", REQUIRED, "the API framework the server is built on"),
    Requirement("uvicorn", REQUIRED, "the ASGI server that serves it"),
    Requirement("python-multipart", REQUIRED,
                "multipart body parsing, for the upload endpoints"),
    Requirement("tomli_w", REQUIRED, "writing training config TOML back out"),
    # Added with the ComfyUI conflict check, and *missed* here when it was:
    # requirements.txt gained a fifth line and the manifest kept saying
    # four. The manifest is what the wizard renders and what the install
    # acts on, so a package the server needs and the manifest does not know
    # about is invisible to the only screen that can install it -- and the
    # install would then run pip without it. A test asserts the two agree,
    # because the disagreement is silent in both directions.
    Requirement("packaging", REQUIRED,
                "PEP 440 version comparison, for checking ComfyUI's "
                "declarations against what is installed"),
    # -- training --------------------------------------------------------
    Requirement("torch", TRAINING,
                "every training step runs through it; without it no run starts",
                approx_mb=2500, never_install=True),
    Requirement("numpy", TRAINING, "array maths throughout the trainer and nodes",
                approx_mb=20, never_install=True),
    Requirement("safetensors", TRAINING, "the only checkpoint format read or written",
                approx_mb=3, never_install=True),
    Requirement("pillow", TRAINING, "image loading for dataset items",
                approx_mb=4, never_install=True),
)

#: What a fresh venv for this project alone would have to install. The
#: full list -- the design's §2a "requirements.txt (full)".
FULL_INSTALL: tuple[str, ...] = tuple(
    r.distribution for r in REQUIREMENTS
)

#: What may be installed *into ComfyUI's venv*: the server's own packages
#: and nothing else. torch and the accelerator stack are excluded on
#: purpose -- they are what ComfyUI already pins, and listing them here is
#: an invitation for pip to "fix" a version it believes is wrong, which is
#: the failure the whole constraints-file design exists to prevent.
COMFY_ADDITIONS: tuple[str, ...] = tuple(
    r.distribution for r in REQUIREMENTS if r.tier == REQUIRED
)


def by_tier(tier: str) -> tuple[Requirement, ...]:
    """Every requirement in one tier."""
    return tuple(r for r in REQUIREMENTS if r.tier == tier)


def total_approx_mb() -> int:
    """Rough download size of a full install, for the disk decision."""
    return sum(r.approx_mb or 0 for r in REQUIREMENTS)