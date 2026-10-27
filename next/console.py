"""What a command-line run prints: step lines, memory frames, task progress and one line per finished node."""

from __future__ import annotations

import json
import time

from .engine.devices import Accelerator
from .engine.runtime.sampler import MemorySampler


class Console:
    """Prints step metrics and memory frames; the rest is quiet unless ``verbose``."""

    def __init__(self, verbose: bool):
        self.verbose = verbose
        self.sampler: MemorySampler | None = None
        self.shapes_seen: set[tuple] = set()
        self._labels: list[str] = []
        self._progress_at = 0.0

    def report(self, stream: str, frame: dict) -> None:
        if stream == "memory":
            self._memory(frame)
        elif frame.get("name") == "step_finished":
            self._step(frame["summary"])
            if self.sampler is not None:
                self.sampler.label = f"step {frame['summary']['step'] + 1}"
        elif stream == "task":
            self._task(frame)
        elif stream == "buckets":
            print(f"data: {frame['samples']} samples, {frame['shapes_in']} latent shapes -> {frame['bucket_shapes']} padded shapes "
                  f"({frame['batch_shapes']} with short batches) in {frame['batches']} batches; padding mean {frame['padding_mean']:.0%}, "
                  f"worst {frame['padding_worst']:.0%}; {frame['short_batches']} short batches, {frame['packed']} samples packed up, {frame['dropped']} dropped", flush=True)
        elif stream == "captions":
            print(f"captions: {frame['summary']}", *(f"  failed: {e}" for e in frame["errors"]), sep="\n", flush=True)
        elif stream in ("quantization", "perf"):
            print(frame["message"], flush=True)
        elif stream == "graph":
            print(f"{frame.get('message') or 'graph capture is off: ' + str(frame.get('why'))}", flush=True)
        elif stream == "warning" or self.verbose:
            print(f"[{stream}] {json.dumps(frame, default=str)}", flush=True)

    def _task(self, frame: dict) -> None:
        """Named stretches of work: their titles label the memory log, and long ones print their progress."""
        title, state = frame["title"], frame["state"]
        count = f" {frame['done']}/{frame['total']}" if frame["total"] else ""
        if state == "started":
            self._labels.append(self.sampler.label if self.sampler else "")
            if self.sampler is not None:
                self.sampler.label = title
            print(f"{title}{' (' + str(frame['total']) + ')' if frame['total'] else ''} ...", flush=True)
        elif state == "progress":
            now = time.monotonic()
            if now - self._progress_at >= 10.0:
                self._progress_at = now
                print(f"  {title}:{count}", flush=True)
        else:
            if self.sampler is not None and self._labels:
                self.sampler.label = self._labels.pop()
            if frame["seconds"] >= 1.0:
                print(f"  {title}: {frame['seconds']:.1f} s{'' if frame['ok'] else ' (failed)'}", flush=True)

    def _step(self, summary: dict) -> None:
        m = summary["metrics"]
        shape = (m.get("batch"), m.get("latent_h"), m.get("latent_w"))
        fresh = shape not in self.shapes_seen
        self.shapes_seen.add(shape)
        where = f"  [{shape[0]}x{shape[1]}x{shape[2]}{' new' if fresh else ''}]" if None not in shape else ""
        memory = f"  device {m['device_mb']:.0f} MB (live {m['live_mb']:.0f}, step peak {m['step_peak_mb']:.0f}, cached {m['reserved_mb']:.0f})" \
            if "device_mb" in m else ""
        draws = "  draws " + "/".join(f"{m[k]:.2f}" for k in ("draw_t_low", "draw_t_mid", "draw_t_high")) if "draw_t_low" in m else ""
        graphs = f"  graphs {m['graphs']} ({m['graph_replays']} replays)" if "graphs" in m else ""
        weights = "  weights " + "/".join(f"{m[k]:.2f}" for k in ("weight_t_low", "weight_t_mid", "weight_t_high") if k in m) \
            if "weight_t_low" in m or "weight_t_mid" in m or "weight_t_high" in m else ""
        epoch = f"  epoch {m['epoch'] + 1}{' end' if m.get('epoch_end') else ''}" if "epoch" in m else ""
        print(f"step {summary['step']}/{summary['total']}  loss {m['loss']:.4f}  {m['step_ms']:.0f} ms{epoch}{where}{graphs}{weights}{draws}{memory}", flush=True)
        if "probe_worst_rel" in m:
            rel = "/".join(f"{m[k]:.3f}" for k in ("probe_rel_t_low", "probe_rel_t_mid", "probe_rel_t_high") if k in m)
            print(f"  probe: adapter/base loss by third {rel}, worst {m['probe_worst_rel']:.3f} (above 1: worse than the base model)", flush=True)
            gc = sorted(k for k in m if k.startswith("gc_"))
            if gc:
                print("  gradient alignment: " + " ".join(f"{k[3:]}={m[k]:.3f}" for k in gc), flush=True)

    @staticmethod
    def _memory(frame: dict) -> None:
        if frame["event"] == "breakdown":
            retries = "" if frame.get("alloc_retries") is None else f", allocator cache flushes {frame['alloc_retries']:.0f} in {frame['segments']:.0f} segments"
            print(f"memory (MB) at step {frame['step']}: needs {frame['need_mb']}, peak live {frame['peak_mb']}, card {frame['used_mb']}{retries}", flush=True)
            for name, mb in frame["parts"].items():
                print(f"  {mb:8.0f}  {name}", flush=True)
        elif frame["event"] == "claims":
            print("memory claims (MB):", flush=True)
            for row in frame["rows"]:
                print(f"  {row['mb']:8.0f}  {row['name']:<22} {row['purpose']}", flush=True)
        elif frame["event"] == "limit":
            print(f"memory budget: {frame['limit_mb']} MB of the card's {frame['card_mb']} MB, {frame['in_use_mb']} MB in use now, "
                  f"margin {frame['margin_mb']} MB; allocator config: {frame['allocator_config']}", flush=True)
        elif frame["event"] == "allocator_cap":
            ours = "" if frame.get("ours_outside_mb") is None else f" (this process {frame['ours_outside_mb']}, other programs {frame['other_programs_mb']})"
            print(f"memory budget: the allocator may hold {frame['cap_mb']} MB = {frame['limit_mb']} limit - {frame['outside_mb']} outside it{ours}"
                  f" - {frame['margin_mb']} margin ({frame['why']})", flush=True)
        else:
            print(f"memory: {json.dumps(frame, default=str)}", flush=True)


class NodeTrace:
    """One line per finished node: how long it took and where the card's memory stood, plus the sampler's label."""

    def __init__(self, doc, accelerator: Accelerator, sampler: MemorySampler | None):
        self.names = {spec.uid: spec.class_name for spec in doc.nodes}
        self.accelerator, self.sampler = accelerator, sampler
        self.t0 = time.monotonic()

    def node_state(self, uid: str, state: str, detail: dict) -> None:
        if self.sampler is not None:
            self.sampler.label = f"{self.names.get(uid, uid)} {state}"

    def node_done(self, result) -> None:
        stats = self.accelerator.memory()
        memory = (f"  allocated {stats['allocated']:.0f}, reserved {stats['reserved']:.0f}, device {stats['device_used']:.0f} MB"
                  if "device_used" in stats and "allocated" in stats and "reserved" in stats else "")
        print(f"[{time.monotonic() - self.t0:6.1f}s] {self.names.get(result.uid, result.uid)}: {result.duration_ms / 1000:.1f}s{memory}", flush=True)

    def run(self, state: str, error: str | None) -> None:
        pass
