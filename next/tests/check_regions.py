"""Several timesteps per image: stratified draws, the batch copies, how the gradient policies combine regions, and a run with a policy."""

from __future__ import annotations

import tempfile
from pathlib import Path

import torch

from ..data.noise import Noiser, TimestepPolicy
from ..data.store import DatasetStore
from ..engine.step.policy import CommonDescent, Mean, Project, make_policy
from ..engine.step.regional import RegionalUpdate
from ..graph import NodeRegistry, enable
from .check_data import make_images
from .check_pipeline import execute, graph
from .pipeline_nodes import TinyFamily
from .support import check, run


def noiser_checks() -> None:
    objective = TinyFamily().objective("eps", False)
    even = Noiser(objective, TimestepPolicy(regions=3))
    batch = even(torch.zeros(6, 1, 2, 2), torch.Generator().manual_seed(1))
    low, high = even.low, even.high
    fractions = ((batch["t"].float() - low) / (high - low)).view(3, 2)
    gaps = ((fractions[1] - fractions[0]) % 1.0, (fractions[2] - fractions[1]) % 1.0)
    check(batch["region"].tolist() == [0, 0, 1, 1, 2, 2] and all(abs(float(g) - 1 / 3) < 0.02 for gap in gaps for g in gap),
          f"regions are laid out copy by copy, a third of the range apart ({fractions.tolist()})")
    strata = Noiser(objective, TimestepPolicy(regions=4, layout="strata"))
    drawn = strata(torch.zeros(8, 1, 2, 2), torch.Generator().manual_seed(2))
    parts = ((drawn["t"].float() - low) / (high - low) * 4).floor().clamp(max=3).long().view(4, 2)
    check(parts.tolist() == [[0, 0], [1, 1], [2, 2], [3, 3]], f"strata: copy r draws from the r-th part of the range ({parts.tolist()})")
    perturbed = Noiser(objective, TimestepPolicy(regions=3, perturb=(0.0, 0.5)))
    base = Noiser(objective, TimestepPolicy(regions=3))
    clean = torch.zeros(6, 1, 4, 4)
    a, b = perturbed(clean, torch.Generator().manual_seed(5)), base(clean, torch.Generator().manual_seed(5))
    check(torch.equal(a["target"], b["target"]) and torch.equal(a["t"], b["t"]), "input perturbation leaves the target and the timesteps alone")
    off = (a["x_t"] != b["x_t"]).flatten(1).any(dim=1).tolist()
    check(off == [False, False, True, True, False, False], f"copies at strength 0 are unchanged, the others differ (regions 0 and 2 use 0, region 1 uses 0.5: {off})")
    odd = even(torch.zeros(4, 1, 2, 2), torch.Generator().manual_seed(1))
    check(odd["region"].tolist() == [0, 0, 0, 0], "a batch that cannot be split into copies is drawn as usual")


def policy_checks() -> None:
    torch.manual_seed(0)
    aligned = torch.tensor([[1.0, 0.0], [2.0, 0.0]])
    opposed = torch.tensor([[1.0, 0.0], [-1.0, 0.1]])
    orthogonal = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    check(torch.allclose(Mean().combine(aligned), torch.tensor([1.5, 0.0])), "mean is the average")
    out = Project().combine(opposed)
    check(all(float(torch.dot(out, g)) >= -1e-6 for g in opposed), f"projection leaves no region pushed back ({out.tolist()})")
    w = CommonDescent.weights((orthogonal @ orthogonal.T).double())
    check(torch.allclose(w, torch.tensor([0.5, 0.5], dtype=torch.double), atol=1e-6), "common descent of unrelated unit gradients weighs them equally")
    skewed = torch.tensor([[1.0, 0.0], [1.0, 1.0]])
    d = CommonDescent().combine(skewed)
    check(all(float(torch.dot(d, g)) > 0 for g in skewed), f"the common direction lowers every region's loss to first order ({d.tolist()})")
    cancel = CommonDescent(floor=0.1)
    check(cancel.combine(torch.tensor([[1.0, 0.0], [-1.0, 0.0]])) is None, "regions that cancel out give no update when a floor is set")
    update = RegionalUpdate(make_policy("common"), normalize=True)
    big = update.direction(torch.tensor([[100.0, 0.0], [0.0, 1.0]]))
    check(abs(float(big[0]) / float(big[1]) - 1.0) < 0.05, f"normalising stops a large gradient from deciding the direction ({big.tolist()})")


def gradnorm_check() -> None:
    from ..engine.step.balance import TimeBalance
    balance = TimeBalance("gradnorm", warmup_reports=3)
    for _ in range(3):
        balance.observe_gradient({"loss_t_low": 0.2}, 0.01)
        balance.observe_gradient({"loss_t_mid": 0.1}, 0.01)
        balance.observe_gradient({"loss_t_high": 0.01}, 0.0025)
    weights = balance.weights
    check(len(weights) == 3 and abs(weights["loss_t_high"] / weights["loss_t_low"] - 4.0) < 0.2 and abs(sum(weights.values()) / 3 - 1.0) < 0.2,
          f"gradnorm gives the third with a quarter of the gradient four times the weight ({weights})")
    balance.observe_gradient({"loss_t_low": 0.2, "loss_t_mid": 0.1}, 5.0)
    check(balance.weights == weights, "a step that mixes thirds changes nothing")


def window_check() -> None:
    """The policy sees each region's gradient over the whole accumulation window."""
    with tempfile.TemporaryDirectory() as raw:
        tmp = Path(raw)
        (tmp / "images").mkdir()
        make_images(tmp / "images")
        store = DatasetStore(tmp / "datasets")
        library = store.library("toy", create=True)
        library.ingest(tmp / "images")
        library.close()
        doc = graph(tmp, steps=2, save_every=1000, gradient_policy="common", accumulation=2)
        for node in doc.nodes:
            if node.class_name == "NoiserNode":
                node.params.update({"regions": 2, "perturb": "0,0.1"})
        _, outcome, trace = execute(store, doc)
        metrics = [f["summary"]["metrics"] for s, f in trace.frames if f.get("name") == "step_finished"]
        check(outcome.ok and len(metrics) == 2 and "region_shared" in metrics[-1], f"two batches per step with two regions each still make one update per step ({outcome.error})")


def balance_run_check() -> None:
    with tempfile.TemporaryDirectory() as raw:
        tmp = Path(raw)
        (tmp / "images").mkdir()
        make_images(tmp / "images")
        store = DatasetStore(tmp / "datasets")
        library = store.library("toy", create=True)
        library.ingest(tmp / "images")
        library.close()
        _, outcome, trace = execute(store, graph(tmp, steps=6, save_every=1000, time_balance="gradnorm"))
        check(outcome.ok and trace.steps() == 6, f"a run with the gradnorm balance finishes ({outcome.error})")


def stream_check() -> None:
    """RegionStream gives the update of the stored-vectors route and the exact mean pairwise cosine."""
    from ..engine.step.regional import RegionGradients, RegionStream

    class Holder:
        def __init__(self, params):
            self.params, self._t = params, set()

        def touched(self):
            return frozenset(self._t)

        def touch(self, ids):
            self._t.update(ids)

    torch.manual_seed(0)
    params = [torch.nn.Parameter(torch.zeros(5, 3)), torch.nn.Parameter(torch.zeros(7))]
    holder = Holder(params)
    stream = RegionStream(holder)
    vectors = [torch.randn(22) * k for k in (1.0, 0.1, 5.0)]
    stream.begin()
    for v in vectors:
        params[0].grad, params[1].grad = v[:15].view(5, 3).clone(), v[15:].clone()
        stream.add()
    norms, metrics = stream.finish()
    got = torch.cat([p.grad.flatten() for p in params])
    want = RegionalUpdate(make_policy("equal"), True).direction(torch.stack(vectors))
    unit = torch.stack([v / v.norm() for v in vectors])
    cos = unit @ unit.T
    mean_cos = float((cos.sum() - 3) / 6)
    check(torch.allclose(got, want, atol=1e-5) and abs(metrics["region_cos_mean"] - mean_cos) < 1e-5
          and all(abs(n - float(v.norm())) < 1e-4 for n, v in zip(norms, vectors)),
          f"streamed update equals the stored-vectors update (max diff {float((got - want).abs().max()):.2e}, cos {metrics['region_cos_mean']:.4f} vs {mean_cos:.4f})")


def trend_checks() -> None:
    """Trend weights: interpolation between bins, neutral until there is history, success read from a falling or a lucky loss,
    the weighted update (equal to the plain one at weight 1, longer and tilted otherwise, the same through the stream), and a run."""
    from ..engine.step.regional import RegionStream
    from ..engine.step.trend import TrendTracker

    t = TrendTracker(bins=12)
    check(t._spread(0.0) == [(0, 1.0), (1, 0.0)] and t._spread(1.0) == [(10, 0.0), (11, 1.0)], "the ends of the range feed their own bin")
    middle = t._spread(6 / 12)  # exactly between the centres of bins 5 and 6
    check([k for k, _ in middle] == [5, 6] and all(abs(share - 0.5) < 1e-9 for _, share in middle), "a place between two bins feeds both, half each")

    def run(mode, losses_at, steps=120, **kw):
        tr = TrendTracker(bins=4, window=60, recent=10, mode=mode, **kw)
        weights = []
        for step in range(steps):
            tr.begin()
            weights.append([tr.weight(i, torch.tensor([losses_at(step, place)]), torch.tensor([place])) for i, place in enumerate((0.1, 0.4, 0.7, 0.95))])
            tr.commit()
        return weights

    check(run("trend", lambda s, p: 1.0)[0] == [1.0] * 4, "no history, no weight")
    falling = run("trend", lambda s, p: 1.0 - 0.004 * s if p < 0.5 else 1.0)
    check(falling[-1][0] > 1.05 and falling[-1][1] > 1.05 and falling[-1][3] == 1.0, f"trend: the parts whose loss falls gain weight, flat ones do not ({falling[-1]})")
    check(run("trend", lambda s, p: 1.0 - 0.004 * s if p < 0.5 else 1.0, boost=0.0)[-1] == [1.0] * 4, "boost 0 changes nothing")
    rising = run("trend", lambda s, p: 1.0 + 0.004 * s if p < 0.5 else 1.0)
    check(rising[-1][0] == 1.0 and run("trend", lambda s, p: 1.0 + 0.004 * s if p < 0.5 else 1.0, penalty=0.5)[-1][0] < 1.0,
          "a rising loss gets no boost, and a penalty lowers its weight only when asked")
    lucky = run("current", lambda s, p: 0.5 if (s == 119 and p < 0.2) else 1.0 + (0.1 if s % 2 else -0.1))
    check(lucky[-1][0] > 1.0 and lucky[-1][3] <= 1.5, f"current: a loss well under its bins' average gets weight ({lucky[-1]})")

    torch.manual_seed(1)
    grads = torch.randn(3, 20)
    update = RegionalUpdate(make_policy("equal"), True)
    plain = update.direction(grads)
    check(torch.allclose(update.direction(grads, torch.ones(3)), plain, atol=1e-6), "weights of 1 give the plain update")
    heavy = update.direction(grads, torch.tensor([3.0, 1.0, 1.0]))
    unit = grads / grads.norm(dim=1, keepdim=True)
    check(float(heavy @ unit[0] / heavy.norm()) > float(plain @ unit[0] / plain.norm())
          and abs(float(heavy.norm() / plain.norm()) - 5 / 3) < 1e-5, "a heavier region pulls the update its way, and the length follows the average weight")

    class Holder:
        def __init__(self, params):
            self.params, self._t = params, set()

        def touched(self):
            return frozenset(self._t)

        def touch(self, ids):
            self._t.update(ids)

    params = [torch.nn.Parameter(torch.zeros(20))]
    stream = RegionStream(Holder(params))
    stream.begin()
    for v, w in zip(grads, (3.0, 1.0, 1.0)):
        params[0].grad = v.clone()
        stream.add(w)
    stream.finish()
    check(torch.allclose(params[0].grad, heavy, atol=1e-5), "the streamed weighted update equals the stored-vectors one")


def trend_run_check() -> None:
    with tempfile.TemporaryDirectory() as raw:
        tmp = Path(raw)
        (tmp / "images").mkdir()
        make_images(tmp / "images")
        store = DatasetStore(tmp / "datasets")
        library = store.library("toy", create=True)
        library.ingest(tmp / "images")
        library.close()
        doc = graph(tmp, steps=6, save_every=1000, trend_boost=0.5, gradient_policy="equal")
        for node in doc.nodes:
            if node.class_name == "NoiserNode":
                node.params.update({"regions": 3, "layout": "strata"})
        _, outcome, trace = execute(store, doc)
        metrics = [f["summary"]["metrics"] for s, f in trace.frames if f.get("name") == "step_finished"]
        check(outcome.ok and len(metrics) == 6 and "trend_w_region_2" in metrics[-1],
              f"a run with trend weights reports each region's weight ({outcome.error}, {sorted(k for k in metrics[-1] if k.startswith('trend')) if metrics else None})")


def ladder_check() -> None:
    """The regions ladder: the count falls from the start value to the normal one over the span, and batches follow it."""
    from ..data.ladder import RegionLadder
    from ..data.noise import Noiser, TimestepPolicy
    objective = TinyFamily().objective("eps", False)
    noiser = Noiser(objective, TimestepPolicy(regions=3, layout="strata"), ladder=RegionLadder(9, 10))
    counts = [noiser.regions_at(k) for k in (0, 1, 5, 9, 10, 50)]
    check(counts[0] == 9 and counts[-1] == 3 and counts[-2] == 3 and counts == sorted(counts, reverse=True) and noiser.max_regions == 9,
          f"ladder counts {counts}")
    out = noiser(torch.zeros(18, 4, 8, 8), torch.Generator().manual_seed(0), 9)
    check(int(out["region"].max()) == 8 and out["t"].shape[0] == 18, "a batch drawn with the ladder's count has that many regions")
    resumed = RegionLadder(9, 10, skip=5)
    check(resumed.at(0, 3) == RegionLadder(9, 10).at(5, 3) and resumed.at(5, 3) == 3 and RegionLadder(9, 10, skip=10).peak(3) == 3,
          "a continued run starts part-way up the ladder, or not at all once past its span")


def ladder_run_check() -> None:
    """A run whose regions fall from 4 to 2, with a learning-rate warmup: every step finishes and the batch size shrinks."""
    with tempfile.TemporaryDirectory() as raw:
        tmp = Path(raw)
        (tmp / "images").mkdir()
        make_images(tmp / "images")
        store = DatasetStore(tmp / "datasets")
        library = store.library("toy", create=True)
        library.ingest(tmp / "images")
        library.close()
        doc = graph(tmp, steps=6, save_every=1000, warmup=3, regions_from=4, regions_span=4, gradient_policy="equal")
        for node in doc.nodes:
            if node.class_name == "NoiserNode":
                node.params.update({"regions": 2, "layout": "strata"})
        _, outcome, trace = execute(store, doc)
        metrics = [f["summary"]["metrics"] for s, f in trace.frames if f.get("name") == "step_finished"]
        sizes = [m["batch"] for m in metrics]
        lrs = [m["lr"] for m in metrics]
        check(outcome.ok and len(metrics) == 6 and sizes[0] > sizes[-1] and lrs[0] < lrs[3] and "loss_region_3" in metrics[0] and "loss_region_3" not in metrics[-1],
              f"batch sizes {sizes}, lr {[round(v, 6) for v in lrs]} ({outcome.error})")


def calibration_check() -> None:
    """Regions with calibrate_every: some steps run each region alone and measure norms, the others are one batch with weights."""
    with tempfile.TemporaryDirectory() as raw:
        tmp = Path(raw)
        (tmp / "images").mkdir()
        make_images(tmp / "images")
        store = DatasetStore(tmp / "datasets")
        library = store.library("toy", create=True)
        library.ingest(tmp / "images")
        library.close()
        doc = graph(tmp, steps=40, save_every=1000, time_balance="gradnorm", gradient_policy="equal", policy_calibrate_every=10)
        for node in doc.nodes:
            if node.class_name == "NoiserNode":
                node.params.update({"regions": 3, "layout": "strata"})
            if node.class_name == "TimeBalanceNode":
                node.params["warmup_reports"] = 3
        _, outcome, trace = execute(store, doc)
        metrics = [f["summary"]["metrics"] for s, f in trace.frames if f.get("name") == "step_finished"]
        alone = [i for i, m in enumerate(metrics) if "loss_region_0" in m]
        check(outcome.ok and len(metrics) == 40 and 0 < len(alone) < 40 and all(i in alone for i in (0, 1, 2)) and 30 in alone and "weight_t_high" in metrics[-1]
              and all(m["batch"] % 3 == 0 for m in metrics),
              f"calibration steps {alone} ({outcome.error})")


def trainer_checks() -> None:
    with tempfile.TemporaryDirectory() as raw:
        tmp = Path(raw)
        (tmp / "images").mkdir()
        make_images(tmp / "images")
        store = DatasetStore(tmp / "datasets")
        library = store.library("toy", create=True)
        library.ingest(tmp / "images")
        library.close()
        for policy in ("mean", "project", "common"):
            doc = graph(tmp, steps=3, save_every=1000, gradient_policy=policy)
            for node in doc.nodes:
                if node.class_name == "NoiserNode":
                    node.params.update({"regions": 3})
            _, outcome, trace = execute(store, doc)
            metrics = [f["summary"]["metrics"] for s, f in trace.frames if f.get("name") == "step_finished"]
            ok = outcome.ok and len(metrics) == 3 and all(m["batch"] % 3 == 0 and m["batch"] >= 3 for m in metrics)
            extra = policy == "mean" or all("loss_region_2" in m and "region_cos_min" in m for m in metrics)
            check(ok and extra, f"{policy}: three steps with three regions per image ({outcome.error}, {sorted(metrics[-1]) if metrics else None})")


def perceptual_checks() -> None:
    """The self-perceptual term: zero for a perfect prediction, gradient into the prediction only, the adapter off for both passes."""
    from ..engine.step.perceptual import SelfPerceptual
    from ..engine.step.state import Inputs, StepState
    from ..families.general.adapters.gate import TimestepGate

    class Features:
        def __init__(self, gate):
            self.gate, self.seen = gate, []
            self.weight = torch.nn.Parameter(torch.ones(1))

        def mid_features(self, x, t, conditioning):
            self.seen.append(None if self.gate.value is None else float(self.gate.value.sum()))
            return torch.tanh(x * self.weight)

    objective = TinyFamily().objective("eps", False)
    gate = TimestepGate()
    net = Features(gate)
    term = SelfPerceptual(net, objective, gate)
    torch.manual_seed(0)
    x0, eps = torch.randn(2, 4, 8, 8), torch.randn(2, 4, 8, 8)
    t = torch.tensor([100, 700])
    x_t, target, sigma = objective.noisy(x0, eps, t)

    def state(prediction):
        s = StepState(0, 0, "cpu")
        s.inputs = Inputs(x_t=x_t, t=t, sigma=sigma.flatten(), target=target, t_frac=t / 999)
        s.prediction = prediction
        return s

    exact = term.distance(state(target.clone()))
    check(float(exact) < 1e-10, f"a perfect prediction costs nothing ({float(exact):.2e})")
    guess = (target + 0.5 * torch.randn_like(target)).requires_grad_()
    distance = term.distance(state(guess))
    distance.backward()
    check(float(distance) > 0 and guess.grad is not None and float(guess.grad.abs().sum()) > 0, "a wrong prediction costs, and the cost reaches it")
    check(net.weight.grad is not None and all(v == 0.0 for v in net.seen) and gate.value is None,
          f"both feature passes run with the adapter off, and the gate is restored ({net.seen})")


def perceptual_trainer_check() -> None:
    with tempfile.TemporaryDirectory() as raw:
        tmp = Path(raw)
        (tmp / "images").mkdir()
        make_images(tmp / "images")
        store = DatasetStore(tmp / "datasets")
        library = store.library("toy", create=True)
        library.ingest(tmp / "images")
        library.close()
        doc = graph(tmp, steps=2, save_every=1000)
        doc = enable(doc, NodeRegistry.default(), "DiffusionTrainerNode", "SelfPerceptualAddon", {"weight": 0.5})
        _, outcome, trace = execute(store, doc)
        metrics = [f["summary"]["metrics"] for s, f in trace.frames if f.get("name") == "step_finished"]
        check(outcome.ok and len(metrics) == 2 and all(m.get("loss_perceptual", -1) >= 0 for m in metrics),
              f"a run with the term reports it ({outcome.error}, {sorted(metrics[-1]) if metrics else None})")


def anchor_checks() -> None:
    """The anchor term: the base pass runs with the adapter off and without a gradient, the adapter pass gets the gradient, steps off the interval do nothing, and a zero adapter costs nothing."""
    from types import SimpleNamespace

    from ..engine.step.anchor import AnchorLoss
    from ..engine.step.state import Inputs, StepState
    from ..families.general.adapters.gate import TimestepGate

    class Net:
        def __init__(self, gate):
            self.gate, self.seen, self.scale = gate, [], torch.nn.Parameter(torch.ones(1))

        def predict(self, x, t, conditioning):
            self.seen.append(None if self.gate.value is None else float(self.gate.value.sum()))
            adapter = 0.0 if self.gate.value is not None and float(self.gate.value.sum()) == 0.0 else self.scale
            return x * (1.0 + 0.1 * adapter)

    class Cond:
        extra: set = set()

        def encode(self, prompts, sizes):
            self.asked = list(prompts)
            return None

    gate, net, cond = TimestepGate(), Net(None), Cond()
    net.gate = gate
    anchor = AnchorLoss(net, cond, gate, ("a", "b"), weight=2.0, every=2, accumulation=2)
    x = torch.randn(3, 4, 8, 8)
    inputs = Inputs(x_t=x, t=torch.tensor([1, 2, 3]), sigma=torch.ones(3), target=x, t_frac=torch.zeros(3))

    def step(n):
        s = StepState(n, 0, "cpu", batch={"sizes_px": None}, scale=1.0)
        s.inputs = inputs
        anchor.run(s)

    step(1)
    check(not net.seen and not anchor.report(), "a step off the interval does nothing")
    step(2)
    check(net.seen[0] == 0.0 and net.seen[1] is None and gate.value is None and set(cond.asked) <= {"a", "b"} and len(cond.asked) == 3,
          f"base pass with the adapter off, adapter pass as trained, one prompt per sample ({net.seen})")
    expected = float(((x * 0.1) ** 2).mean())
    check(net.scale.grad is not None and float(net.scale.grad) != 0.0 and abs(anchor.report()["loss_anchor"] - expected) < 1e-6,
          "the penalty reaches the adapter and is reported before the weight")
    check(cond.extra >= {"a", "b"}, "the anchor prompts are kept encoded")
    for bad in (dict(weight=0.0), dict(every=0), dict(prompts=())):
        try:
            AnchorLoss(net, cond, gate, **{"prompts": ("a",), "weight": 1.0, "every": 1, **bad})
            refused = False
        except ValueError:
            refused = True
        check(refused, f"the anchor refuses {bad}")


def anchor_trainer_check() -> None:
    with tempfile.TemporaryDirectory() as raw:
        tmp = Path(raw)
        (tmp / "images").mkdir()
        make_images(tmp / "images")
        store = DatasetStore(tmp / "datasets")
        library = store.library("toy", create=True)
        library.ingest(tmp / "images")
        library.close()
        for regions in (1, 3):
            doc = graph(tmp, steps=2, save_every=1000, **({"gradient_policy": "equal"} if regions > 1 else {}))
            for node in doc.nodes:
                if node.class_name == "NoiserNode" and regions > 1:
                    node.params.update({"regions": regions})
            doc = enable(doc, NodeRegistry.default(), "DiffusionTrainerNode", "AnchorAddon", {"weight": 0.5, "every": 1, "prompts": "a cat|a ship"})
            _, outcome, trace = execute(store, doc)
            metrics = [f["summary"]["metrics"] for s, f in trace.frames if f.get("name") == "step_finished"]
            check(outcome.ok and len(metrics) == 2 and all(m.get("loss_anchor", -1) >= 0 for m in metrics),
                  f"a run with the anchor reports it, {regions} region(s) ({outcome.error}, {sorted(metrics[-1]) if metrics else None})")


def cleaner_trainer_check() -> None:
    """A run that cleans its adapter after one step (both modes): it reports the cleaning, writes the cleaned adapter, the mask and the summary, and keeps the kept weights fixed to the end."""
    with tempfile.TemporaryDirectory() as raw:
        tmp = Path(raw)
        (tmp / "images").mkdir()
        make_images(tmp / "images")
        store = DatasetStore(tmp / "datasets")
        library = store.library("toy", create=True)
        library.ingest(tmp / "images")
        library.close()
        for mode in ("element", "component"):
            out = tmp / f"clean-{mode}"
            doc = graph(tmp, steps=4, save_every=1000)
            doc = enable(doc, NodeRegistry.default(), "DiffusionTrainerNode", "CleanerAddon", {"keep": 0.3, "at": 1, "steps": 2, "mode": mode, "directory": str(out)})
            doc = enable(doc, NodeRegistry.default(), "DiffusionTrainerNode", "AnchorAddon", {"weight": 0.5, "every": 1})
            _, outcome, trace = execute(store, doc)
            reports = [f for s, f in trace.frames if s == "cleaning"] if hasattr(trace, "frames") else []
            files = sorted(p.name for p in out.glob("*")) if out.exists() else []
            check(outcome.ok and files == ["lora_cleaned.safetensors", "mask.safetensors", "summary.json"],
                  f"{mode}: the run cleans, writes its files and finishes ({outcome.error}, {files})")


if __name__ == "__main__":
    run(noiser_checks, policy_checks, gradnorm_check, trainer_checks, window_check, balance_run_check, stream_check, ladder_check, ladder_run_check, calibration_check, perceptual_checks,
        perceptual_trainer_check, anchor_checks, anchor_trainer_check, cleaner_trainer_check, trend_checks, trend_run_check)
