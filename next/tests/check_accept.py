"""Acceptance lab: regions, the block recorder, the quantization probe (a planted fault is found where it was planted), fixed denoiser inputs.

    python -m next.tests.check_accept
"""

from __future__ import annotations

import torch
from torch import nn

from ..accept.fixtures import DenoiserInputs
from ..accept.quantization import QuantizationProbe, relative_error
from ..accept.recorder import BlockRecorder
from ..accept.regions import Region, Regions
from ..families.general.ldm_unet.unet import UNetModel
from ..families.general.quantize.backends import TorchBackend
from ..families.general.quantize.int8 import Int8Quantizer
from ..families.general.quantize.nf4 import Nf4Quantizer
from ..families.general.quantize.nf4_backends import TorchNf4Backend
from ..families.sdxl.denoiser import SDXLDenoiser
from .check_sdxl import TINY_UNET
from .support import check, run


def tiny_denoiser() -> SDXLDenoiser:
    torch.manual_seed(0)
    return SDXLDenoiser(UNetModel(**TINY_UNET), dtype=torch.float32)


def tiny_inputs() -> DenoiserInputs:
    return DenoiserInputs(resolution=128, timesteps=[100, 700], device="cpu", dtype=torch.float32, context_dim=48, adm_dim=40, tokens=5)


def regions():
    r = Regions()
    check(r.of("input_blocks.4.1.transformer_blocks.0.attn1.to_q") == Region("down", 4) and r.of("middle_block.1.proj_in") == Region("mid", None)
          and r.of("output_blocks.2.0.in_layers.2") == Region("up", 2) and r.of("time_embed.0") == Region("embed", None) and r.of("out.2") == Region("out", None),
          "paths map to their regions")
    check(Regions(coarse=True).of("input_blocks.4.0.x").name == "down" and r.of("input_blocks.4.0.x").name == "down.4", "coarse regions drop the block index")
    names = [x.name for x in r.order({Region("up", 1), Region("down", 3), Region("mid", None), Region("down", 1), Region("embed", None)})]
    check(names == ["embed", "down.1", "down.3", "mid", "up.1"], f"regions order from the input to the output ({names})")


def recorder():
    den, inputs = tiny_denoiser(), tiny_inputs()
    with BlockRecorder(den.module()) as rec:
        out, grad = inputs.run(den)
    rec = rec.recording
    check(rec.outputs and set(rec.grads) <= set(rec.outputs) and len(rec.grads) == len(rec.outputs), f"every block's output and incoming gradient are kept ({len(rec.outputs)} blocks)")
    check(all(t.dtype == torch.float32 and t.device.type == "cpu" for t in rec.outputs.values()), "recordings live on the host in fp32")
    check(not any(m._forward_hooks for m in den.module().modules()), "the hooks are gone after the block")
    again = tiny_inputs().run(den)
    check(torch.equal(again[0], out) and torch.equal(again[1], grad), "the fixed inputs give the same output and gradient every time")
    with BlockRecorder(den.module(), limit_mb=0.01) as small:
        inputs.run(den)
    check(small.recording.dropped > 0 and small.recording.megabytes <= 0.01 + 1e-9, "the memory limit drops blocks and counts them")


def nf4_probe():
    den, inputs = tiny_denoiser(), tiny_inputs()
    quantizer = Nf4Quantizer(TorchNf4Backend(), min_params=0)
    probe = QuantizationProbe()
    clean = probe.run(den.module(), quantizer, lambda _: inputs.run(den), precision="nf4")
    check(clean.report.layers > 0 and clean.weights and all(0 < e < 0.2 for e in clean.weights.values()), "every converted layer has a plausible weight error")
    check(0 < clean.final["output"] < 0.5 and 0 < clean.final["input_grad"] < 1.0, f"the whole network moved a little ({clean.final})")
    groups = {r.region.split(".")[0] for r in clean.regions}
    check({"down", "mid", "up"} <= groups, f"errors are reported by region ({sorted(r.region for r in clean.regions)[:6]} ...)")
    text = clean.markdown()
    check("## By region" in text and "weight error" in text and clean.to_json()["regions"], "the report has the region table and the worst layers")

    fault = "input_blocks.5.1.transformer_blocks.0.ff.net.2"

    def plant(model: nn.Module):
        report = quantizer.run(model)
        layer = dict(model.named_modules())[fault]
        layer.weight_absmax.mul_(8.0)
        return report

    den2 = tiny_denoiser()
    planted = probe.run(den2.module(), Nf4Quantizer(TorchNf4Backend(), min_params=0), lambda _: inputs.run(den2), precision="nf4", convert=plant)
    before = [p for p in clean.modules if p.startswith(("input_blocks.1", "input_blocks.2", "input_blocks.3", "input_blocks.4"))]
    check(before and all(abs(planted.modules[p]["output"] - clean.modules[p]["output"]) < 1e-7 for p in before), "blocks upstream of the fault are unaffected in the forward pass")
    hit = "input_blocks.5.1"
    check(planted.modules[hit]["output"] > 3 * clean.modules[hit]["output"], "the block holding the fault shows a much larger output error")
    check(planted.final["output"] > clean.final["output"], "and so does the whole network")


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.a, self.b = nn.Linear(256, 256, bias=False), nn.Linear(256, 256, bias=False)

    def forward(self, x):
        return x + self.b(torch.relu(self.a(x)))


def int8_probe():
    torch.manual_seed(0)
    model = nn.Sequential(Block(), Block())
    x = torch.randn(8, 256)
    dy = torch.randn(8, 256)

    def run_model(m):
        inp = x.clone().requires_grad_(True)
        out = m(inp)
        out.backward(dy)
        return out, inp.grad

    result = QuantizationProbe(select=lambda m: isinstance(m, Block)).run(model, Int8Quantizer(TorchBackend(), min_params=0), run_model, precision="int8")
    check(result.report.layers == 4 and set(result.modules) == {"0", "1"} and all(0 < e < 0.05 for e in result.weights.values()), "int8 layers convert and are measured")
    check(0 < result.final["output"] < 0.1 and relative_error(torch.ones(3), torch.ones(3)) == 0 and relative_error(torch.zeros(2), torch.zeros(2)) != relative_error(torch.zeros(2), torch.zeros(2)),
          "relative error is zero for equal tensors and undefined against a zero reference")


if __name__ == "__main__":
    run(regions, recorder, nf4_probe, int8_probe)
