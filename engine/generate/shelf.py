"""Denoisers a generation run keeps on the device after sampling, for as long as the VRAM budget allows.

The sampler leaves its denoiser on the card: a second sampler with the same model reuses it, and the decode starts without a
reload. Nothing is moved unless the budget needs the room (a load that does not fit, a decode that runs out of memory); then
the denoiser goes to the host first and comes back when a sampler wants it again.
"""

from __future__ import annotations

import gc
from collections.abc import Callable
from typing import Any

from engine.devices import Accelerator
from engine.runtime.card import MB
from engine.runtime.memory import ResidencyManager
from engine.runtime.residents import ModulesResident


class DenoiserShelf:
    """Without a residency manager (a runtime-less run) nothing is kept: each denoiser is dropped when its sampler is done."""

    def __init__(self, residency: ResidencyManager | None, device: str) -> None:
        self._residency = residency
        self._device = device
        self._held: dict[Any, tuple[str, Any, ModulesResident]] = {}

    def acquire(self, key: Any, load: Callable[[], Any], prepare: Callable[[Any], None]) -> Any:
        """The denoiser for ``key``, on the device. ``load`` reads it onto the host, ``prepare`` (baking adapters in, eval mode) runs
        once it is on the device. A denoiser held before is reused, brought back from the host when it was moved off."""
        entry = self._held.get(key)
        if entry is not None:
            name, denoiser, resident = entry
            if resident.footprint_bytes() == 0:
                self._make_room(resident.size_bytes() / MB, "bringing the denoiser back")
                self._residency.restore([name])
            return denoiser
        denoiser = load()
        resident = ModulesResident([denoiser.module()], denoiser.to, self._device)
        self._make_room(resident.size_bytes() / MB, "loading the denoiser")
        denoiser.to(self._device)
        prepare(denoiser)
        if self._residency is not None:
            name = f"sampler denoiser {len(self._held) + 1}"
            self._residency.claim(name, resident, rank=0, purpose="the denoiser of a Sampler, kept for the next sampler and the decode; moves to the host when the budget needs the room")
            self._held[key] = (name, denoiser, resident)
        return denoiser

    def done(self, key: Any, denoiser: Any) -> None:
        """Its sampler finished (or failed): a denoiser the shelf does not keep goes now."""
        if key in self._held:
            return
        del denoiser
        gc.collect()
        Accelerator(self._device).empty_cache()

    def yield_room(self) -> bool:
        """Move every held denoiser to the host (a decode ran out of memory); False when nothing was on the device."""
        moved = False
        for _name, _denoiser, resident in self._held.values():
            if resident.footprint_bytes():
                resident.offload()
                moved = True
        if moved:
            self._residency.give_back("denoisers moved to the host for the decode")
        return moved

    def _make_room(self, mb: float, why: str) -> None:
        if self._residency is not None:
            self._residency.ensure_started()
            self._residency.make_room(mb, why)
