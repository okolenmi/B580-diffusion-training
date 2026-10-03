"""CheckRequirements -- is this machine able to run a training step?

The design's Phase A, built as a use case rather than a script because
the installer's first screen is exactly this answer, and two answers from
two implementations would be two answers to drift.

**Pure and total.** It reads `importlib.metadata` through the
`PackageInventory` port and asks a fake device probe; it never imports a
package, never runs pip, and never touches the filesystem. That is what
makes it safe to call on every page load of a wizard, and it is why the
inventory port exists: the alternative -- "try importing torch and see" --
would cost 1.5 GB and seconds to answer a question about a machine that may
not have torch at all.

The device is probed separately and last, because it is the only part that
costs anything. A missing training stack is reported without ever asking
the hardware anything, since the answer to "will this train" is already no.

**Ready is not "everything installed".** It is: the server's own packages
present, the training stack present, and a device visible. Anything less
is reported with the specific rows that are missing, so the wizard can
render the list rather than a verdict.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..ports.environment import DeviceProbe, PackageInventory
from ..ports.requirements_manifest import (
    COMFY_ADDITIONS,
    COMFY_PROVIDED,
    FULL_INSTALL,
    OPTIONAL,
    REQUIRED,
    REQUIREMENTS,
    TIERS,
    TRAINING,
    Requirement,
    total_approx_mb,
)


@dataclass(frozen=True, slots=True)
class RequirementStatus:
    """One manifest row, checked.

    `installed_version` is None when absent, which is different from an
    empty string for a distribution that reports no version -- the first
    means "not there" and the second means "there, and it claims nothing",
    and the wizard says different things.
    """

    requirement: Requirement
    installed_version: str | None

    @property
    def name(self) -> str:
        return self.requirement.distribution

    @property
    def satisfied(self) -> bool:
        return self.installed_version is not None

    @property
    def blocking(self) -> bool:
        """Would this stop a training run from starting?"""
        return not self.satisfied and self.requirement.tier in (REQUIRED, TRAINING)


@dataclass(frozen=True, slots=True)
class ReadinessReport:
    """The whole answer: per-package rows, the device, and one verdict."""

    packages: tuple[RequirementStatus, ...]
    device_present: bool
    device_name: str | None
    device_total_memory_mb: float | None
    #: Why the device is not usable, in the user's terms. None when it is.
    device_reason: str | None
    #: A note about the probe itself, as distinct from what it found.
    device_detail: str | None
    #: Whether the device was actually asked. False when the packages
    #: already answered the question, and the difference matters: a UI
    #: that renders "no device" for a machine nobody asked would be
    #: reporting a fact it does not have.
    device_checked: bool = True

    @property
    def ready(self) -> bool:
        """A training run can start on this machine, right now."""
        return (
            self.device_present
            and not [row for row in self.packages if row.blocking]
        )

    @property
    def missing(self) -> tuple[RequirementStatus, ...]:
        return tuple(row for row in self.packages if not row.satisfied)

    @property
    def blocking_missing(self) -> tuple[RequirementStatus, ...]:
        return tuple(row for row in self.packages if row.blocking)

    @property
    def optional_missing(self) -> tuple[RequirementStatus, ...]:
        return tuple(
            row for row in self.missing
            if row.requirement.tier in (OPTIONAL, COMFY_PROVIDED)
        )

    @property
    def approx_download_mb(self) -> int:
        """What a full install would fetch, counting only what is absent.

        Missing-and-installable only: a package ComfyUI already provides is
        never fetched by this project's installer, so including it would
        overstate both the size and the risk.
        """
        return sum(
            row.requirement.approx_mb or 0
            for row in self.missing
            if not row.requirement.never_install
        )


class DescribeRequirements:
    """The manifest as data, for the wizard and for `GET /manifest`.

    Separate from `CheckRequirements` because the two answer different
    questions for different reasons: this one is a constant the server
    already knows and is asked by a page that only wants to *render* what
    is needed, while the other probes the machine. A wizard that fetched
    the manifest over HTTP on every render would be asking the network to
    be told something the server has in a module.

    It is a use case rather than a route reading the module directly
    because a handler that imports a constant is a handler whose answer
    cannot be substituted in a test, and every other response here is
    mapped from a DTO for the same reason.
    """

    def execute(self) -> dict[str, object]:
        return {
            "requirements": [
                {
                    "name": requirement.distribution,
                    "tier": requirement.tier,
                    "why": requirement.why,
                    "approx_mb": requirement.approx_mb,
                    "never_install": requirement.never_install,
                }
                for requirement in REQUIREMENTS
            ],
            "tiers": list(TIERS),
            # The two lists the design's §2a splits on. Named here because
            # the distinction is the whole safety argument: torch is in the
            # full list and must never be in the additions list.
            "full_install": list(FULL_INSTALL),
            "comfy_additions": list(COMFY_ADDITIONS),
            "full_install_approx_mb": total_approx_mb(),
        }


class CheckRequirements:
    def __init__(
        self,
        inventory: PackageInventory,
        device: DeviceProbe,
    ) -> None:
        self._inventory = inventory
        self._device = device

    def execute(self) -> ReadinessReport:
        rows = tuple(
            RequirementStatus(
                requirement=requirement,
                installed_version=self._inventory.version_of(
                    requirement.distribution
                ),
            )
            for requirement in REQUIREMENTS
        )
        blocking = [row for row in rows if row.blocking]

        if blocking:
            # Short-circuit: a machine missing torch does not need to be
            # asked whether it has a graphics card. Importing torch for
            # that costs seconds and gigabytes, and the answer cannot
            # change the verdict -- `ready` is already false and only the
            # rows above explain why.
            #
            # This is checked before the probe rather than after because a
            # probe that is never asked is worth more than a probe that
            # answers: the wizard calls this on every render, and "1.7s,
            # every time, to be told no" is a cost the user pays for a
            # fact already in hand.
            # `device_reason` stays None and `device_checked` False. The
            # first version put "not checked: ..." in `device_reason`,
            # which reads as the *cause* of an absent card -- so a UI
            # rendering "not ready: {reason}" would tell a user with a
            # perfectly good B580 that their graphics card is missing.
            # "Not known" and "known to be absent" are different facts and
            # the report has to be able to say which.
            return ReadinessReport(
                packages=rows,
                device_present=False,
                device_name=None,
                device_total_memory_mb=None,
                device_reason=None,
                device_detail=None,
                device_checked=False,
            )

        report = self._device.report()
        return ReadinessReport(
            packages=rows,
            device_present=report.present,
            device_name=report.name,
            device_total_memory_mb=report.total_memory_mb,
            device_reason=report.reason,
            device_detail=report.detail,
        )



def full_install_approx_mb() -> int:
    """The size of a from-scratch install, for the venv choice screen."""
    return total_approx_mb()