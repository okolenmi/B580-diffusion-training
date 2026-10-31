"""Several timesteps per image: stratified draws, the batch copies, how the gradient policies combine regions, and a run with a policy."""

from __future__ import annotations

import tempfile
from pathlib import Path

import torch

from ..data.noise import Noiser, TimestepPolicy
from ..data.store import DatasetStore
from ..engine.step.policy import CommonDescent, Mean, Project, make_policy
from ..engine.step.regional import RegionalUpdate
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
        doc = graph(tmp, steps=2, save_every=1000)
        for node in doc.nodes:
            if node.class_name == "NoiserNode":
                node.params.update({"regions": 2, "perturb": "0,0.1"})
            if node.class_name == "DiffusionTrainerNode":
                node.params.update({"gradient_policy": "common", "accumulation": 2})
        _, outcome, trace = execute(store, doc)
        metrics = [f["summary"]["metrics"] for s, f in trace.frames if f.get("name") == "step_finished"]
        check(outcome.ok and len(metrics) == 2 and "region_shared" in metrics[-1], f"two batches per step with two regions each still make one update per step ({outcome.error})")


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
            overrides = {"noiser": {"regions": 3}, "trainer": {"gradient_policy": policy}}
            doc = graph(tmp, steps=3, save_every=1000)
            for node in doc.nodes:
                key = {"NoiserNode": "noiser", "DiffusionTrainerNode": "trainer"}.get(node.class_name)
                if key:
                    node.params.update(overrides[key])
            _, outcome, trace = execute(store, doc)
            metrics = [f["summary"]["metrics"] for s, f in trace.frames if f.get("name") == "step_finished"]
            ok = outcome.ok and len(metrics) == 3 and all(m["batch"] % 3 == 0 and m["batch"] >= 3 for m in metrics)
            extra = policy == "mean" or all("loss_region_2" in m and "region_cos_min" in m for m in metrics)
            check(ok and extra, f"{policy}: three steps with three regions per image ({outcome.error}, {sorted(metrics[-1]) if metrics else None})")


if __name__ == "__main__":
    run(noiser_checks, policy_checks, trainer_checks, window_check)
