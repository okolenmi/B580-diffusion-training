"""Step profiler — accumulates per-phase wall time."""

import time

from tqdm import tqdm


class StepTimer:
    """Accumulates per-phase wall time and prints a report every N steps."""
    def __init__(self, report_every=100):
        self.report_every = report_every
        self.phases = {}
        self.counts = {}
        self._t0 = {}
        self._step = 0

    def start(self, name):
        self._t0[name] = time.perf_counter()

    def stop(self, name):
        if name in self._t0:
            dt = time.perf_counter() - self._t0.pop(name)
            self.phases[name] = self.phases.get(name, 0.0) + dt
            self.counts[name] = self.counts.get(name, 0) + 1

    def tick(self):
        self._step += 1
        if self._step % self.report_every == 0:
            self.report()

    def report(self):
        if not self.phases:
            return
        total = sum(self.phases.values())
        print(f"\n  ── Timing report (last {self.report_every} steps) ──")
        for name, t in self.phases.items():
            # avg should reflect contribution to the total training step time.
            # Previously this was t / counts[name], which was misleading for
            # things like optimizer steps that don't happen every step.
            avg = t / self.report_every * 1000
            pct = t / total * 100 if total > 0 else 0
            print(f"    {name:<22}: {avg:6.1f} ms/step  ({pct:4.1f}%)")
        print(f"    {'total':<22}: {total/self.report_every*1000:6.1f} ms/step")
        print("")
        self.phases.clear()
        self.counts.clear()
