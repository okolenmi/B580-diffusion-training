"""What changed between two reports: budget, categories and piece groups, largest growth first."""

from __future__ import annotations

BUDGET_KEYS = ("wall_ms", "eager_ms", "replayed_ms", "device_ms", "launches", "utilisation", "peak_mb")


def _delta(a, b):
    return None if a is None or b is None else b - a


class Comparison:
    def __init__(self, baseline: dict, other: dict, baseline_name: str = "baseline", other_name: str = "other"):
        self.a, self.b, self.names = baseline, other, (baseline_name, other_name)

    def budget(self) -> list[dict]:
        return [{"key": k, "baseline": self.a["budget"].get(k), "other": self.b["budget"].get(k),
                 "delta": _delta(self.a["budget"].get(k), self.b["budget"].get(k))} for k in BUDGET_KEYS]

    def categories(self) -> list[dict]:
        old = {c["category"]: c for c in self.a["categories"]}
        new = {c["category"]: c for c in self.b["categories"]}
        rows = [{"category": name, "baseline_ms": old.get(name, {}).get("device_ms", 0.0), "other_ms": new.get(name, {}).get("device_ms", 0.0),
                 "baseline_calls": old.get(name, {}).get("calls", 0.0), "other_calls": new.get(name, {}).get("calls", 0.0)} for name in {*old, *new}]
        for r in rows:
            r["delta_ms"] = r["other_ms"] - r["baseline_ms"]
        return sorted(rows, key=lambda r: -abs(r["delta_ms"]))

    def groups(self) -> list[dict]:
        def key(g):
            return (g["kind"], g["role"], g["region"])
        old, new = {key(g): g for g in self.a["groups"]}, {key(g): g for g in self.b["groups"]}
        rows = []
        for k in {*old, *new}:
            before, after = old.get(k), new.get(k)
            rows.append({"group": " ".join(k), "baseline_ms": before["total_ms"] if before else 0.0, "other_ms": after["total_ms"] if after else 0.0,
                         "delta_ms": (after["total_ms"] if after else 0.0) - (before["total_ms"] if before else 0.0),
                         "only_in": None if before and after else (self.names[0] if before else self.names[1])})
        return sorted(rows, key=lambda r: -abs(r["delta_ms"]))

    def build(self) -> dict:
        return {"names": list(self.names), "budget": self.budget(), "categories": self.categories(), "groups": self.groups()}

    def markdown(self, top: int = 12) -> str:
        a, b = self.names
        out = [f"## {b} against {a}", "", "| step | " + a + " | " + b + " | change |", "|---|---:|---:|---:|"]
        for r in self.budget():
            if r["baseline"] is None and r["other"] is None:
                continue
            spec = ".0%" if r["key"] == "utilisation" else ".0f"
            fmt = lambda v: "-" if v is None else format(v, spec)  # noqa: E731
            change = "-" if r["delta"] is None else (f"{r['delta'] * 100:+.0f} points" if r["key"] == "utilisation" else f"{r['delta']:+.0f}")
            out.append(f"| {r['key']} | {fmt(r['baseline'])} | {fmt(r['other'])} | {change} |")
        out += ["", "Device time by category (per step):", "", f"| category | {a} ms | {b} ms | change ms | calls {a} to {b} |", "|---|---:|---:|---:|---|"]
        out += [f"| {r['category']} | {r['baseline_ms']:.1f} | {r['other_ms']:.1f} | {r['delta_ms']:+.1f} | {r['baseline_calls']:.0f} to {r['other_calls']:.0f} |"
                for r in self.categories() if abs(r["delta_ms"]) >= 0.05]
        out += ["", "Piece groups that changed most:", "", f"| group | {a} ms | {b} ms | change ms |", "|---|---:|---:|---:|"]
        out += [f"| {r['group']}{' (only in ' + r['only_in'] + ')' if r['only_in'] else ''} | {r['baseline_ms']:.1f} | {r['other_ms']:.1f} | {r['delta_ms']:+.1f} |"
                for r in self.groups()[:top]]
        return "\n".join(out) + "\n"
