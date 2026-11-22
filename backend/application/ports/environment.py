"""Environment port -- what this machine can actually do.

Two questions that fail differently, so two contracts. `PackageInventory` is cheap, total and never raises: it answers
"which packages are present" without importing them, because "is torch installed" must not cost a torch import.
`DeviceProbe` needs torch, which can fail in several ways that are all data (not installed, no such backend, no card),
so it returns a `DeviceReport` rather than raising. `pip` is reached through the package installer port: what is
installed and what could be installed are different questions, and only the second has a subprocess in it.

The implementations are in `backend.infrastructure.environment`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

#: Distribution name -> the name code imports, where they differ. Keyed by
#: what `importlib.metadata` reports, because that is the only direction
#: the inventory ever needs: it looks up by distribution name.
#:
#: The reverse mapping is derived from this one rather than written out again: two hand-kept tables drift apart.
IMPORT_NAMES: dict[str, str] = {
    "python-multipart": "multipart",
    "pillow": "PIL",
    "scikit-learn": "sklearn",
    "opencv-python": "cv2",
    "xpu-kernels": "xpu_kernels",
}


@dataclass(frozen=True, slots=True)
class InstalledPackage:
    """One distribution found in the running interpreter.

    `distribution` is the pip name (`python-multipart`) and `import_name`
    is what the code imports (`multipart`) -- they are not the same string,
    and guessing wrong is how a dependency check reports a package missing
    on a machine that has it.
    """

    distribution: str
    import_name: str
    version: str


class PackageInventory(ABC):
    """What is installed, read without importing anything."""

    @abstractmethod
    def installed(self) -> tuple[InstalledPackage, ...]:
        """Every distribution visible to the running interpreter.

        Total by contract: a package that cannot be read is absent from
        the answer rather than an exception, because "I could not tell"
        and "it is not there" lead to the same next step here -- install it
        -- and one of them being an error would mean a single unreadable
        distribution makes the whole installer unavailable.
        """
        raise NotImplementedError

    @abstractmethod
    def version_of(self, distribution: str) -> str | None:
        """One package's version, or None if it is not installed."""
        raise NotImplementedError

    def import_name_for(self, distribution: str) -> str:
        """The name code would import, given a distribution name."""
        return IMPORT_NAMES.get(distribution, distribution)

    def distribution_for(self, import_name: str) -> str:
        """The pip name, given the name code imports. Derived, not stored."""
        for dist, mod in IMPORT_NAMES.items():
            if mod == import_name:
                return dist
        return import_name


@dataclass(frozen=True, slots=True)
class DeviceReport:
    """What the accelerator turned out to be.

    Every field is optional because every one of them can be unknown:
    torch may be absent, may lack the `xpu` attribute, may report no
    device, and the property query may raise on a driver this torch build
    does not understand. `present` is the one that decides whether a run
    can start; the rest is detail for the user who has just been told no.
    """

    present: bool
    backend: str | None = None
    name: str | None = None
    total_memory_mb: float | None = None
    #: Why it is not present, in the user's terms. None when it is.
    reason: str | None = None
    #: Set when the probe itself could not run, as distinct from running
    #: and finding nothing. The installer says different things about
    #: those two.
    detail: str | None = None


class DeviceProbe(ABC):
    """Ask the accelerator what it is. Imports torch, so it can fail."""

    @abstractmethod
    def report(self) -> DeviceReport:
        raise NotImplementedError

    def devices(self) -> tuple[DeviceReport, ...]:
        """Every device, not just the current one.

        Separate from `report()` because they are asked for different
        reasons and at different costs. `report()` answers "can a run start
        here", which is one boolean and is asked on every render.
        This answers "which card should the torch wheel be built for", which
        is only meaningful on a machine with more than one, and which has to
        enumerate before it can answer.

        The base implementation returns the single current device, so a port
        that only knows how to report one still works -- and a caller cannot
        mistake the default for "this machine has exactly one card", because
        `enumerate_all` says whether it actually looked.
        """
        rows, _ = self.devices_with_reason()
        return rows

    def devices_with_reason(self) -> tuple[tuple[DeviceReport, ...], str | None]:
        """The devices, *and* why there are none if there are none.

        On the ABC rather than only on the real probe, because a caller
        that cannot get both has to choose between inventing a reason and
        guessing from an empty list -- and the caller here renders the two
        differently ("no cards" versus "we could not look"). `TorchDeviceProbe`
        overrides this with a real enumeration.

        The default reports the current device, which is all a port that
        does not enumerate can honestly claim, and carries `report()`'s own
        reason when there is nothing.
        """
        single = self.report()
        if single.present:
            return (single,), None
        return (), single.reason

    @property
    def enumerate_all(self) -> bool:
        """Whether `devices()` really looked, rather than falling back.

        Sent to the client so a one-card answer and an unchecked one do not
        render identically. The distinction is the same one
        `ReadinessReport.device_checked` exists for.
        """
        return False
