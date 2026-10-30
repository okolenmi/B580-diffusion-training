"""The lab: samplers recover what a perfect denoiser knows, a grid renders deterministically, every image carries what made it.

    python -m next.tests.check_lab
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import torch

from ..lab.metrics import Change, Detail
from ..lab.render import Grid, ImageStore, Recipe, Renderer
from ..lab.sampling import SAMPLERS, GuidedDenoiser, NoiseTable, SigmaSchedule, initial_noise
from .support import check, run


class ToyBackend:
    """A network that is a smooth function of its input and of the strength, and a decoder that paints the latent."""

    def __init__(self):
        self.stages: list[str] = []

    def stage(self, name):
        self.stages.append(name)

    def condition(self, prompt, size):
        return torch.tensor(float(sum(map(ord, prompt)) % 7))

    def predict(self, x_in, t, conditioning, strength):
        return x_in * (1 + 0.1 * strength) + 0.01 * conditioning * torch.sin(x_in)

    def encode_image(self, pixels):
        return torch.nn.functional.avg_pool2d(pixels.float() / 127.5 - 1, 8)[:, :1].repeat(1, 4, 1, 1)

    def decode(self, latents):
        pixels = torch.tanh(latents[:, :3]).add(1).mul(127.5).to(torch.uint8)
        return torch.nn.functional.interpolate(pixels.float(), scale_factor=8).to(torch.uint8)


def samplers() -> None:
    table = NoiseTable()
    check(abs(table.sigma_max - 14.6146) < 1e-3 and abs(table.sigma_min - 0.02917) < 1e-4, "the sigma table spans the SDXL range")
    t = torch.tensor([0, 10, 500, 999])
    check(bool((table.timestep(table.sigma(t.float())) == t).all()), "timestep and sigma are inverse at the integer steps")
    for name in SigmaSchedule.NAMES:
        sigmas = SigmaSchedule(name, table).sigmas(12)
        check(len(sigmas) == 13 and sigmas[-1] == 0 and bool((sigmas[:-1] > sigmas[1:]).all()) and abs(float(sigmas[0]) - table.sigma_max) < 1e-3, f"{name}: 12 steps decrease from the largest sigma to 0")
    target = torch.randn(1, 4, 8, 8, generator=torch.Generator().manual_seed(1))
    sigmas = SigmaSchedule("karras", table).sigmas(10)
    for name, sampler in SAMPLERS.items():
        noise = initial_noise(5, (1, 4, 8, 8), float(sigmas[0]), table)
        out = sampler().run(lambda x, sigma: target.clone(), noise, sigmas)
        check(float((out - target).abs().max()) < 1e-4, f"{name} returns the clean sample a perfect denoiser knows")
    x = torch.randn(1, 4, 8, 8)
    eps = torch.randn_like(x)
    sigma = torch.tensor(3.0)
    denoised = GuidedDenoiser(lambda x_in, t, positive: eps, table, 5.0)(x, sigma)
    check(torch.allclose(denoised, x - eps * 3.0, atol=1e-6), "guidance with equal predictions leaves the prediction alone")
    mixed = GuidedDenoiser(lambda x_in, t, positive: eps if positive else torch.zeros_like(eps), table, 2.0)(x, sigma)
    check(torch.allclose(mixed, x - 2.0 * eps * 3.0, atol=1e-6), "guidance scales the difference between the prompts")


def rendering() -> None:
    backend = ToyBackend()
    grid = Grid(prompts=("a fox", "a house"), seeds=(1, 2), strengths=(0.0, 1.0), steps=6, width=64, height=64)
    recipes = grid.recipes()
    check(len(recipes) == 8 and len({r.stem for r in recipes}) == 8, "a grid crosses prompts, seeds and strengths with distinct file names")
    first = Renderer(backend).render(recipes, {"arm": "x"})
    again = Renderer(ToyBackend()).render(recipes, {"arm": "x"})
    check(all(torch.equal(a.pixels, b.pixels) for a, b in zip(first, again)), "the same recipe renders the same pixels")
    check(backend.stages == ["encode", "sample", "decode"], "the network and the decoder are staged one after the other")
    pick = lambda s, x: next(i for i in first if i.recipe.seed == s and i.recipe.strength == x and i.recipe.prompt == "a fox")  # noqa: E731
    check(not torch.equal(pick(1, 0.0).pixels, pick(1, 1.0).pixels) and not torch.equal(pick(1, 1.0).pixels, pick(2, 1.0).pixels), "strength and seed change the image")
    with tempfile.TemporaryDirectory() as tmp:
        store = ImageStore(Path(tmp) / "arm")
        paths = [store.save(image) for image in first]
        meta = ImageStore.read_meta(paths[3])
        check(meta["arm"] == "x" and meta["recipe"] == {**first[3].recipe.__dict__}, "the PNG carries the recipe and the arm")
        check(len(__import__("json").loads((Path(tmp) / "arm" / "index.json").read_text())) == 8, "the index lists every image")
    check(Recipe("a").stem == Recipe("a").stem and Recipe("a").stem != Recipe("b").stem, "file names do not depend on the process")


def texture(size: int = 256) -> torch.Tensor:
    """A natural-looking image: noise whose power falls with frequency, as (3, H, W) uint8."""
    gen = torch.Generator().manual_seed(0)
    freq = torch.fft.fftfreq(size)
    radius = (freq[:, None] ** 2 + freq[None, :] ** 2).sqrt().clamp(min=1.0 / size)
    layers = []
    for _ in range(3):
        field = torch.fft.ifft2(torch.fft.fft2(torch.randn(size, size, generator=gen)) / radius).real
        layers.append((field - field.min()) / (field.max() - field.min()))
    return (torch.stack(layers) * 255).to(torch.uint8)


def tiled_decoding() -> None:
    from ..lab.tiling import TiledDecoder

    def decode(latents):
        """Position-wise, so tiles must reassemble to exactly what one pass gives."""
        up = torch.nn.functional.interpolate(latents[:, :3].float(), scale_factor=8, mode="nearest")
        return (up * 40 + 128).clamp(0, 255).to(torch.uint8)

    latents = torch.randn(1, 4, 150, 90, generator=torch.Generator().manual_seed(2))
    whole = decode(latents)
    tiled = TiledDecoder(decode, tile=64, overlap=16)(latents)
    check(tiled.shape == whole.shape and int((tiled.int() - whole.int()).abs().max()) <= 1, "overlapping tiles reassemble to the whole decode (position-wise decoder)")
    small = latents[:, :, :40, :40]
    check(torch.equal(TiledDecoder(decode)(small), decode(small)), "a latent smaller than a tile is decoded in one piece")


def detail_metrics() -> None:
    sharp = texture()
    blurred = torch.nn.functional.avg_pool2d(sharp.float()[None], 5, stride=1, padding=2)[0].to(torch.uint8)
    a, b = Detail.of(sharp), Detail.of(blurred)
    check(b.sharpness < a.sharpness and b.fine_share < a.fine_share and b.slope < a.slope, "a blurred image has less sharpness, less fine power and a steeper spectrum")
    check(-4.0 < a.slope < 0.0, f"the texture's spectral slope is natural-looking ({a.slope:.2f})")
    change = Change.between(blurred, sharp)
    check(change.top_band_ratio < 0.5 and change.sharpness_ratio < 0.5 and change.fine_to_coarse > 1.0, "against its sharp original, the blur loses fine power and the change is mostly fine")
    same = Change.between(sharp, sharp)
    check(same.coarse == 0 and same.fine == 0 and abs(same.top_band_ratio - 1) < 1e-6, "an image does not differ from itself")
    shift = (sharp.float() * 0.8 + 20).to(torch.uint8)
    cc = Change.between(shift, sharp)
    check(cc.coarse > 0 and cc.fine_to_coarse < change.fine_to_coarse, "a tone change moves the coarse part more than a blur does")


def timestep_loss() -> None:
    from ..families.general.schedules.ddpm import DiscreteSchedule
    from ..lab.timestep_loss import TimestepLoss

    class Backend(ToyBackend):
        """Knows the clean latent, so it predicts the noise exactly without the adapter and 0.5 off with it at full strength."""

        def __init__(self, latent):
            super().__init__()
            self.latent = latent

        def predict(self, x_in, t, conditioning, strength):
            sigma = DiscreteSchedule().alpha_sigma(t.long())[1].view(-1, 1, 1, 1)
            return (x_in * (sigma ** 2 + 1).sqrt() - self.latent) / sigma + strength * 0.5

    latent = torch.randn(1, 4, 8, 8, generator=torch.Generator().manual_seed(3))
    backend = Backend(latent)
    result = TimestepLoss(backend, timesteps=(100, 500), draws=2).run([latent])
    check(all(abs(base) < 1e-6 and abs(lora - 0.25) < 1e-6 for base, lora in result.values()), "the loss is the squared error of the predicted noise, per timestep, with and without the adapter")
    check(backend.stages == ["encode", "sample"], "the prompts are encoded before the network runs")


def experiment_run() -> None:
    """Arms train (a stand-in process), render with and without an adapter, and the report names every arm, with sheets and a zip."""
    import json

    from PIL import Image
    from safetensors.torch import save_file

    from ..lab.experiment import Experiment
    from ..lab.report import Report
    from ..lab.runner import Lab, Training

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        for folder in ("train", "holdout"):
            (root / folder).mkdir()
            for i in range(3):
                Image.fromarray(texture(128).permute(1, 2, 0).numpy()).save(root / folder / f"{i}.png")
        spec = {"name": "demo", "models": "m", "datasets": "d", "dataset": "set", "checkpoint": "c.safetensors", "output": str(root / "out"), "device": "cpu",
                "images": {"train": str(root / "train"), "holdout": str(root / "holdout"), "count": 2, "size": 64}, "timesteps": [100, 500],
                "grid": {"prompts": ["", "a fox"], "seeds": [1], "strengths": [0.5, 1.0], "steps": 4, "width": 64, "height": 64},
                "arms": [{"name": "full"}], "product": [[{"name": "lr1", "args": ["--lr", "1e-4"]}, {"name": "lr2", "args": ["--lr", "2e-4"]}], [{"name": "b1", "args": ["--batch", "1"]}]]}
        experiment = Experiment.from_dict(spec)
        check([a.name for a in experiment.arms] == ["full", "lr1+b1", "lr2+b1"] and experiment.arm("lr2+b1").args == ("--lr", "2e-4", "--batch", "1"), "arms and products expand to named variants with their arguments")

        def trainer(exp, arm, folder):
            (folder / "train").mkdir(parents=True)
            path = folder / "train" / "lora_step000010.safetensors"
            save_file({"x": torch.zeros(1)}, str(path), metadata={"step": "10", "rank": "4", "alpha": "4", "scheme": "kohya"})
            (folder / "train.log").write_text("log")
            result = Training(str(path), 10, 1.0, [1.0, 0.5], [{"step": 10, "low": 0.9, "mid": 1.0, "high": 1.1, "worst": 1.1}], 0)
            (folder / "train.json").write_text(json.dumps(result.to_dict()))
            return result

        said: list[str] = []
        lab = Lab(experiment, lambda lora: ToyBackend(), said.append, trainer)
        lab.train(list(experiment.arms))
        lab.render(list(experiment.arms))
        done = len(said)
        lab.train(list(experiment.arms))
        check(all("trained already" in line for line in said[done:]), "an arm that is trained is not trained again")
        check(len(list((experiment.root / "reference" / "images").glob("*.png"))) == 2 and len(list((experiment.root / "full" / "images").glob("*.png"))) == 4, "the reference and each arm have their images")
        meta = ImageStore.read_meta(next((experiment.root / "lr1+b1" / "images").glob("*.png")))
        check(meta["arm"] == "lr1+b1" and meta["arm_args"] == ["--lr", "1e-4", "--batch", "1"] and meta["adapter"]["step"] == "10" and len(meta["adapter"]["sha1"]) == 40, "an image names its arm, the arguments and the adapter file")
        losses = json.loads((experiment.root / "full" / "timestep_loss.json").read_text())
        check(set(losses) == {"train", "holdout"} and set(losses["train"]) == {"100", "500"}, "the timestep loss is kept per split and timestep")
        report = Report(experiment)
        text = report.write().read_text()
        check(all(name in text for name in ("full", "lr1+b1", "lr2+b1", "reference (no adapter)", "Loss by timestep", "Probe at the end")), "the report has the arms, the detail and loss tables and the probe")
        check(len(list((experiment.root / "sheets").glob("*.png"))) >= 4, "contact sheets are made per prompt, seed and strength")
        archive = report.pack()
        check(archive.exists() and archive.stat().st_size < 25e6, "the share archive is made and small")


if __name__ == "__main__":
    run(samplers, rendering, tiled_decoding, detail_metrics, timestep_loss, experiment_run)
