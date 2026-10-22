"""Rules that turn a report's numbers into named issues, each with its evidence and the lever that applies."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Finding:
    title: str
    severity_ms: float   # recoverable milliseconds per step the rule can point to (0 when it is a warning about the data)
    evidence: str
    lever: str


class Rule:
    """``check`` returns findings from a plain dict report (the JSON form), so rules also run on saved reports."""

    def check(self, report: dict) -> list[Finding]:
        raise NotImplementedError


def _share(report: dict, category: str) -> float:
    total = sum(c["device_ms"] for c in report["categories"]) or 1.0
    return sum(c["device_ms"] for c in report["categories"] if c["category"] == category) / total


class LaunchBound(Rule):
    THRESHOLD = 0.6

    def check(self, report: dict) -> list[Finding]:
        b = report["budget"]
        if b.get("device_is_host") or b.get("utilisation") is None or b["utilisation"] >= self.THRESHOLD:
            return []
        gap = b["wall_ms"] - b["device_ms"]
        tried = b.get("replayed_ms") is not None
        lever = ("graph replay is already running for some steps; compare replayed and eager step times above"
                 if tried else "graph replay (trainer.graph=true) removes the launch cost if the capture can use the same kernels")
        return [Finding("The step is launch bound", gap,
                        f"the device is busy {b['utilisation']:.0%} of the step ({b['device_ms']:.0f} of {b['wall_ms']:.0f} ms) over "
                        f"{b['launches']:.0f} launches and {b['ops']:.0f} operators per step, {b['host_us_per_launch']:.0f} us of host time per launch", lever)]


class AttentionAsMath(Rule):
    def check(self, report: dict) -> list[Finding]:
        names = [(k["name"].lower(), k["device_ms"]) for k in report["kernels"] if k["category"] == "attention"]
        if not names:
            return []
        flash = sum(ms for n, ms in names if any(t in n for t in ("flash", "sdpa", "sdp_")))
        softmax = sum(ms for n, ms in names if "softmax" in n)
        if flash > 0 or softmax <= 0:
            return []
        share = _share(report, "attention")
        return [Finding("Attention runs as math (matmul, softmax, matmul)", softmax,
                        f"softmax kernels are present and no flash kernel is; attention is {share:.0%} of device time",
                        "a fused attention kernel that can be captured (rework/73), or run eagerly with the default (flash) kernel")]


class CastCopyShare(Rule):
    THRESHOLD = 0.08

    def check(self, report: dict) -> list[Finding]:
        share = _share(report, "copy")
        if share < self.THRESHOLD:
            return []
        ms = sum(c["device_ms"] for c in report["categories"] if c["category"] == "copy")
        return [Finding("Dtype and copy kernels take a large share", ms, f"{share:.0%} of device time is casts and copies",
                        "find the modules issuing `.to(dtype)` round trips (autocast boundaries, adapter branches) and keep one dtype through them")]


class Int8Overhead(Rule):
    THRESHOLD = 0.10

    def check(self, report: dict) -> list[Finding]:
        share = _share(report, "int8")
        if share < self.THRESHOLD:
            return []
        ms = sum(c["device_ms"] for c in report["categories"] if c["category"] == "int8")
        gemm = sum(c["device_ms"] for c in report["categories"] if c["category"] == "gemm")
        return [Finding("The int8 path's own kernels are a large share", ms,
                        f"rotate, quantize and dequantize kernels are {share:.0%} of device time against {gemm:.0f} ms of matmuls",
                        "int8_grad=int8 (no weight dequantize in backward), or a fused rotate+quantize in the kernels project; native weights if memory allows")]


class ElementwiseTail(Rule):
    THRESHOLD = 0.25

    def check(self, report: dict) -> list[Finding]:
        share = sum(_share(report, c) for c in ("elementwise", "norm", "copy", "memory", "reduce"))
        if share < self.THRESHOLD:
            return []
        ms = sum(c["device_ms"] for c in report["categories"] if c["category"] in ("elementwise", "norm", "copy", "memory", "reduce"))
        calls = sum(c["calls"] for c in report["categories"] if c["category"] in ("elementwise", "norm", "copy", "memory", "reduce"))
        return [Finding("Many small memory-bound kernels", ms * 0.5,
                        f"{share:.0%} of device time ({calls:.0f} calls per step) is elementwise, normalisation, copy, memory and reduction work",
                        "fusion (fewer, larger kernels): norm + modulation, activation + multiply, cast + matmul input; half the time is the most a fusion usually recovers")]


class RecomputeShare(Rule):
    THRESHOLD = 0.15

    def check(self, report: dict) -> list[Finding]:
        mods = report.get("modules") or []
        fwd = sum(m["fwd_ms"] for m in mods)
        re_ = sum(m["recompute_ms"] for m in mods)
        total = fwd + re_ + sum(m["bwd_ms"] for m in mods)
        if not total or re_ / total < self.THRESHOLD:
            return []
        return [Finding("Recomputing activations is a large part of the step", re_,
                        f"{re_:.0f} ms per step ({re_ / total:.0%} of module time) is forward work done a second time by checkpointing",
                        "keep more blocks un-checkpointed (trainer.checkpoint_by=measured with tune_checkpointing, attention_fraction) if memory allows")]


class NewShapes(Rule):
    def check(self, report: dict) -> list[Finding]:
        out = report["budget"].get("outliers") or []
        if not out:
            return []
        return [Finding("Some steps in the window were much slower than the rest", 0.0,
                        f"{len(out)} step(s) took over twice the median: " + ", ".join(f"step {o['step']} {o['shape']} {o['wall_ms']:.0f} ms" for o in out[:4]),
                        "a latent shape seen for the first time costs kernel and primitive creation (seconds); fewer distinct shapes (buckets.max_shapes) and a longer warm-up")]


class SlowModules(Rule):
    TOP = 5

    def check(self, report: dict) -> list[Finding]:
        groups = [g for g in (report.get("groups") or []) if g["recoverable_ms"] > 0]
        if not groups:
            return []
        top = sorted(groups, key=lambda g: -g["recoverable_ms"])[: self.TOP]
        total = sum(g["recoverable_ms"] for g in groups)
        return [Finding(f"Slowest pieces by recoverable time: {g['kind']} {g['role']} ({g['region']})", g["recoverable_ms"],
                        f"{g['total_ms']:.1f} ms per step against an ideal {g['ideal_ms']:.1f} ms ({g['efficiency']:.0%} efficient, {g['bound']}-bound), "
                        f"{g['count']} layers; all rated pieces together leave {total:.0f} ms",
                        "see the module table for the individual layers") for g in top]


class Coverage(Rule):
    THRESHOLD = 0.8

    def check(self, report: dict) -> list[Finding]:
        r = report.get("reconciliation") or {}
        if not r or r.get("module_ms") is None:
            return []
        out = []
        if r["device_ms"] and not (0.6 <= r["module_ms"] / r["device_ms"] <= 1.2):
            out.append(Finding("Module times do not add up to the device time", 0.0,
                               f"modules account for {r['module_ms']:.0f} ms of {r['device_ms']:.0f} ms device time per eager step",
                               "the difference is work outside the hooked modules (loss, optimizer, data movement) or host waits; see the phase and category tables"))
        if r["module_ms"] and r["unrated_ms"] / r["module_ms"] > 1 - self.THRESHOLD:
            out.append(Finding("Much of the module time has no cost model", r["unrated_ms"], f"{r['unrated_ms']:.0f} of {r['module_ms']:.0f} ms is in modules the cost model does not know",
                               "extend `perf/cost.py` for those kinds"))
        return out


RULES: tuple[Rule, ...] = (LaunchBound(), AttentionAsMath(), CastCopyShare(), Int8Overhead(), ElementwiseTail(), RecomputeShare(), NewShapes(), SlowModules(), Coverage())


def findings_for(report: dict, rules: tuple[Rule, ...] = RULES) -> list[Finding]:
    found = [f for rule in rules for f in rule.check(report)]
    return sorted(found, key=lambda f: -f.severity_ms)
