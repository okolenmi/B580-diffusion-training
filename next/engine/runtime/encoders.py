"""Text encoders that sit on the host and visit the device only to encode."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator

from ..devices import Accelerator
from .memory import ResidencyManager
from .residents import ModulesResident


class OnDemandKit:
    """A conditioning kit whose encoders are on the device only inside a ``session``.

    The cache makes encoding rare (once per prompt), so the encoders spend almost the whole run on the host and
    their memory goes to training. ``hold`` keeps them on the device instead, for runs whose prompts do not fit
    the cache and would otherwise move them every step.
    """

    def __init__(self, kit: Any, resident: ModulesResident, accelerator: Accelerator, residency: ResidencyManager | None = None):
        self._kit, self._resident, self._accelerator, self._residency = kit, resident, accelerator, residency
        self.hold = False
        self._depth = 0
        self._on_device = False

    def __getattr__(self, name: str) -> Any:
        return getattr(self._kit, name)

    @contextmanager
    def session(self) -> Iterator[None]:
        """Encodes inside share one visit; the encoders move to the device at the first encode, so a session whose
        prompts all come from a cache never loads them."""
        self._depth += 1
        try:
            yield
        finally:
            self._depth -= 1
            if self._depth == 0 and not self.hold:
                self.release()

    def _ensure_on_device(self) -> None:
        if not self._on_device:
            if self._residency is not None:
                self._residency.make_room(self._resident.size_bytes() / 2**20, "load the text encoders")
            self._resident.reload()
            self._on_device = True

    def release(self) -> None:
        if self._on_device:
            self._resident.offload()
            self._on_device = False
            self._accelerator.empty_cache()

    def encode_text(self, prompts):
        with self.session():
            self._ensure_on_device()
            return self._kit.encode_text(prompts)
