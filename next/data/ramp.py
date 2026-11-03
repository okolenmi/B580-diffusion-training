"""A smaller batch for the first batches of a run."""

from __future__ import annotations


class BatchRamp:
    """The first ``batches`` batches of a run hold at most ``size`` samples; every later batch has the Batcher's own size.

    A small batch makes each update follow one picture closely (more variety between updates); a large one averages several
    pictures (steadier, more detail). Starting small and ending large gets the first for the direction of the run and the second for
    its finish. ``skip`` is how far into the run training already is, so a continued run does not start small again."""

    def __init__(self, size: int, batches: int, skip: int = 0) -> None:
        if size < 1 or batches < 1 or skip < 0:
            raise ValueError("a batch ramp needs a size of at least 1, at least 1 batch and a skip of 0 or more")
        self.size, self.batches, self.skip = size, batches, skip

    def left(self, before: int) -> int:
        """Batches still to run small when ``before`` batches of the run have been handed out."""
        return max(self.batches - self.skip - before, 0)
