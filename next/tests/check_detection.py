"""Self-checks for model detection and Load Model, on synthetic safetensors files (headers only).

    python -m next.tests.check_detection

Key layouts for SDXL follow the formats the old loader read. The Anima, Qwen3 and Wan-style VAE
layouts are written from recollection, so these checks prove the code agrees with itself; they
do not prove it agrees with real files. Real headers belong in a later phase.
"""

from __future__ import annotations

import json
import os
import struct
import tempfile
from pathlib import Path

from ..families.api.header import HeaderCache, ModelFileError, read_header
from ..families.api.library import ModelLibrary
from ..families.api.parts import ClipGPart, ClipLPart, DenoiserPart, ModelPart, VAEPart
from ..families.registry import detect_file, list_families
from ..graph import Environment, ExecutionContext, GraphDoc, GraphRunner, NodeRegistry, NodeSpec, resolve_node
from ..nodes.model.load_model import LoadModelNode
from .support import check, raises, run

DTYPE_BYTES = {"F16": 2, "BF16": 2, "F32": 4}


def write_safetensors(path: Path, tensors: dict[str, tuple], dtype: str = "F16", metadata: dict | None = None) -> None:
    """Header plus a sparse body: shapes are real, the data is zeros nobody reads."""
    header: dict = {} if metadata is None else {"__metadata__": metadata}
    offset = 0
    for name, shape in tensors.items():
        size = DTYPE_BYTES[dtype]
        for dim in shape:
            size *= dim
        header[name] = {"dtype": dtype, "shape": list(shape), "data_offsets": [offset, offset + size]}
        offset += size
    blob = json.dumps(header).encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(blob)))
        f.write(blob)
        f.truncate(8 + len(blob) + offset)


def sdxl_unet(prefix="model.diffusion_model."):
    return {f"{prefix}input_blocks.0.0.weight": (32, 4, 3, 3), f"{prefix}label_emb.0.0.weight": (16, 2816),
            f"{prefix}input_blocks.4.1.transformer_blocks.0.attn2.to_k.weight": (8, 2048),
            f"{prefix}input_blocks.1.0.in_layers.0.weight": (4,)}


def sd15_unet(prefix="model.diffusion_model."):
    return {f"{prefix}input_blocks.0.0.weight": (32, 4, 3, 3),
            f"{prefix}input_blocks.1.1.transformer_blocks.0.attn2.to_k.weight": (8, 768)}


def clip_l(base="conditioner.embedders.0.transformer."):
    return {f"{base}text_model.embeddings.token_embedding.weight": (8, 768),
            f"{base}text_model.encoder.layers.0.mlp.fc1.weight": (4, 4),
            f"{base}text_model.encoder.layers.1.mlp.fc1.weight": (4, 4)}


def clip_g(base="conditioner.embedders.1.model."):
    return {f"{base}token_embedding.weight": (8, 1280), f"{base}transformer.resblocks.0.attn.in_proj_weight": (4, 4),
            f"{base}text_projection": (4, 4)}


def vae(base="first_stage_model.", z=4):
    return {f"{base}encoder.down.0.block.0.conv1.weight": (4, 4, 3, 3), f"{base}quant_conv.weight": (2 * z, 2 * z, 1, 1),
            f"{base}decoder.conv_in.weight": (4, z, 3, 3)}


def anima_dit(prefix="net."):
    return {f"{prefix}blocks.0.mlp.layer1.weight": (8, 4), f"{prefix}blocks.1.mlp.layer1.weight": (8, 4),
            f"{prefix}x_embedder.proj.1.weight": (64, 68),
            f"{prefix}llm_adapter.blocks.0.cross_attn.q_proj.weight": (4, 4),
            f"{prefix}llm_adapter.blocks.1.cross_attn.q_proj.weight": (4, 4)}


def qwen3(base="model."):
    return {f"{base}embed_tokens.weight": (10, 1024), f"{base}layers.0.self_attn.q_norm.weight": (4,),
            f"{base}layers.1.self_attn.q_norm.weight": (4,), f"{base}layers.0.mlp.up_proj.weight": (4, 4)}


def wan_vae(z=16):
    return {"encoder.conv1.weight": (4, 3, 3, 3, 3), "decoder.upsamples.0.residual.0.gamma": (4,),
            "conv1.weight": (2 * z, 2 * z, 1, 1, 1), "conv2.weight": (z, z, 1, 1, 1)}


def lora():
    return {"lora_unet_input_blocks_1.lora_down.weight": (4, 8), "lora_unet_input_blocks_1.lora_up.weight": (8, 4),
            "lora_unet_input_blocks_1.alpha": ()}


class Fixture:
    def __init__(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.root = root
        folders = {s: [root / s] for s in ("checkpoints", "diffusion_models", "text_encoders", "vae", "loras")}
        self.library = ModelLibrary(folders, HeaderCache(capacity=8))
        ck, dm, te, vv = (root / s for s in ("checkpoints", "diffusion_models", "text_encoders", "vae"))
        write_safetensors(ck / "sdxl_full.safetensors", {**sdxl_unet(), **clip_l(), **clip_g(), **vae()}, "BF16")
        write_safetensors(ck / "sub" / "sdxl_nested.safetensors", {**sdxl_unet(), **clip_l(), **clip_g(), **vae()})
        write_safetensors(dm / "sdxl_unet.safetensors", sdxl_unet(prefix=""))
        write_safetensors(dm / "sd15.safetensors", sd15_unet())
        write_safetensors(dm / "anima.safetensors", anima_dit())
        write_safetensors(dm / "anima_model_prefix.safetensors", anima_dit("model.diffusion_model."))
        write_safetensors(dm / "cosmos_plain.safetensors", {k: v for k, v in anima_dit().items() if "llm_adapter" not in k})
        write_safetensors(te / "clip_l.safetensors", clip_l(base="") | {})
        write_safetensors(te / "clip_g.safetensors", clip_g(base=""))
        write_safetensors(te / "qwen3.safetensors", qwen3())
        write_safetensors(vv / "sdxl_vae.safetensors", vae(base=""))
        write_safetensors(vv / "qwen_image_vae.safetensors", wan_vae())
        write_safetensors(root / "loras" / "style.safetensors", lora())
        write_safetensors(ck / "junk.safetensors", {"foo.bar": (2, 2)})
        write_safetensors(ck / "both.safetensors", {**sdxl_unet(), **anima_dit(prefix="model.diffusion_model.")})
        (ck / "fake.ckpt").write_bytes(b"\x80\x04pickle")
        (ck / "garbage.safetensors").write_bytes(struct.pack("<Q", 5) + b"{not}")
        (ck / "huge.safetensors").write_bytes(struct.pack("<Q", 200 * 1024 * 1024) + b"{")
        (ck / "short.safetensors").write_bytes(b"abc")
        trunc = ck / "truncated.safetensors"
        write_safetensors(trunc, sdxl_unet())
        with open(trunc, "r+b") as f:
            f.truncate(os.path.getsize(trunc) - 4)
        outside = root / "outside.safetensors"
        write_safetensors(outside, sdxl_unet())
        try:
            os.symlink(outside, ck / "escape.safetensors")
        except OSError:
            pass

    def detect(self, scope, name, family="auto"):
        return detect_file(self.library, scope, name, family)


def roles(det):
    return [p.role for p in det.parts]


def sdxl_cases(fx):
    full = fx.detect("checkpoints", "sdxl_full.safetensors")
    check(full.ok and full.family_id == "sdxl", f"SDXL checkpoint is ready ({full.message})")
    check(roles(full) == ["denoiser", "clip_l", "clip_g", "vae"], f"four parts in order: {roles(full)}")
    check(full.label_of("denoiser") == "UNet", "SDXL's denoiser is labelled UNet")
    check(full.parts[0].dtype == "BF16" and full.file_size > 0, "dtype and size are reported")
    check("sdxl" in full.summary().lower() and "UNet" in full.summary(), f"summary reads well: {full.summary()}")
    nested = fx.detect("checkpoints", "sub/sdxl_nested.safetensors")
    check(nested.ok and roles(nested) == roles(full), "subfolders work")
    unet = fx.detect("diffusion_models", "sdxl_unet.safetensors")
    check(unet.ok and roles(unet) == ["denoiser"] and unet.parts[0].prefix == "", "bare UNet file: one output, empty prefix")
    check(roles(fx.detect("text_encoders", "clip_l.safetensors")) == ["clip_l"], "CLIP-L file alone")
    check(roles(fx.detect("text_encoders", "clip_g.safetensors")) == ["clip_g"], "CLIP-G file alone")
    v = fx.detect("vae", "sdxl_vae.safetensors")
    check(v.ok and roles(v) == ["vae"] and v.family_id == "sdxl", f"SDXL VAE file alone ({v.message})")
    check(fx.detect("checkpoints", "sdxl_full.safetensors").parts == full.parts, "detection is repeatable")
    bad = fx.detect("diffusion_models", "sd15.safetensors")
    check(bad.status == "error" and "768" in bad.message, f"SD 1.5 is refused by cross-attention width: {bad.message}")
    check(fx.detect("diffusion_models", "sdxl_unet.safetensors", "anima").status == "error", "forcing the wrong family fails")


def anima_cases(fx):
    a = fx.detect("diffusion_models", "anima.safetensors")
    check(a.ok and a.family_id == "anima", f"Anima DiT ({a.message})")
    check(roles(a) == ["denoiser", "adapter"], f"DiT file exposes denoiser and adapter: {roles(a)}")
    check(a.label_of("denoiser") == "DiT", "Anima's denoiser is labelled DiT")
    check(a.parts[0].config["blocks"] == 2 and a.parts[1].config["blocks"] == 2, "block counts are read")
    check(a.parts[0].key_count == 3 and a.parts[1].key_count == 2, "the adapter's keys are not counted in the DiT")
    check(fx.detect("diffusion_models", "anima_model_prefix.safetensors").parts[0].prefix == "model.diffusion_model.",
          "alternate DiT prefix")
    c = fx.detect("diffusion_models", "cosmos_plain.safetensors")
    check(c.status == "error" and "adapter" in c.message.lower(), f"Cosmos without the adapter is not Anima: {c.message}")
    q = fx.detect("text_encoders", "qwen3.safetensors")
    check(q.ok and roles(q) == ["qwen3"] and q.parts[0].config["layers"] == 2, f"Qwen3 ({q.message})")
    w = fx.detect("vae", "qwen_image_vae.safetensors")
    check(w.ok and w.family_id == "anima" and roles(w) == ["vae"], f"16-channel VAE belongs to Anima ({w.message})")
    check(fx.detect("vae", "sdxl_vae.safetensors", "anima").status == "error", "a 4-channel VAE is not Anima's")


def failure_cases(fx):
    l = fx.detect("loras", "style.safetensors")
    check(l.status == "error" and l.error_kind == "adapter" and "Load Adapter" in l.message, f"LoRA: {l.message}")
    j = fx.detect("checkpoints", "junk.safetensors")
    check(j.status == "error" and j.report, f"unknown file carries a key report: {j.report}")
    both = fx.detect("checkpoints", "both.safetensors")
    check(both.status == "ready" and both.family_id in ("sdxl", "anima"), "mixed file resolves to the better-scored family")
    for name, kind in (("fake.ckpt", "pickle_refused"), ("garbage.safetensors", "bad_header"),
                       ("huge.safetensors", "header_too_large"), ("short.safetensors", "bad_header"),
                       ("truncated.safetensors", "truncated"), ("missing.safetensors", "not_found"),
                       ("../outside.safetensors", "outside_scope"), ("/etc/passwd", "outside_scope"),
                       ("sub/../../outside.safetensors", "outside_scope"), ("", "no_file")):
        d = fx.detect("checkpoints", name)
        check(d.status == "error" and d.error_kind == kind, f"{name!r} -> {kind} (got {d.error_kind}: {d.message})")
    if (fx.root / "checkpoints" / "escape.safetensors").is_symlink():
        d = fx.detect("checkpoints", "escape.safetensors")
        check(d.status == "error" and d.error_kind == "outside_scope", "a symlink that leaves the folder is refused")
        check("escape.safetensors" not in fx.library.list_files("checkpoints"), "and is not listed")
    check(fx.detect("nowhere", "x.safetensors").error_kind == "bad_scope", "unknown scope")
    raises(ModelFileError, lambda: read_header(fx.root / "checkpoints" / "fake.ckpt"), "pickle refused at the reader", "pickle")
    names = fx.library.list_files("checkpoints")
    check("sub/sdxl_nested.safetensors" in names and "fake.ckpt" not in names, f"listing: .safetensors only, with subfolders")


def cache_cases(fx):
    cache = fx.library.cache
    path = fx.library.resolve("diffusion_models", "sdxl_unet.safetensors")
    first = cache.get(path)
    check(cache.get(path) is first, "second read is a cache hit")
    write_safetensors(path, sdxl_unet(prefix="") | {"extra.weight": (2,)})
    second = cache.get(path)
    check(second is not first and "extra.weight" in second, "a changed file invalidates the entry")
    for i in range(12):
        write_safetensors(fx.root / "vae" / f"v{i}.safetensors", vae(base=""))
        fx.detect("vae", f"v{i}.safetensors")
    check(len(cache) <= 8, f"the cache is bounded ({len(cache)})")


def node_cases(fx):
    env = Environment({"models": fx.library})
    res = LoadModelNode.resolve({"scope": "checkpoints", "file": "sdxl_full.safetensors"}, env)
    check(res.state == "ready" and list(res.outputs) == ["denoiser", "clip_l", "clip_g", "vae"], "resolve: four outputs")
    check(res.outputs["denoiser"].type is DenoiserPart and res.outputs["clip_l"].type is ClipLPart, "outputs are typed per role")
    check(res.outputs["denoiser"].label == "UNet" and "SDXL" in res.info, f"labels and info: {res.info}")
    one = LoadModelNode.resolve({"scope": "vae", "file": "sdxl_vae.safetensors"}, env)
    check(list(one.outputs) == ["vae"], "a VAE file gives one output")
    empty = LoadModelNode.resolve({}, env)
    check(empty.state == "ready" and not empty.outputs and empty.info == "choose a file", "no file: ready, no outputs")
    check(LoadModelNode.resolve({"file": "x"}, None).state == "error", "no library: error, not a crash")
    err = LoadModelNode.resolve({"scope": "checkpoints", "file": "../outside.safetensors"}, env)
    check(err.state == "error" and err.focus == "file", "bad path focuses the file field")
    check(LoadModelNode.resolve({"scope": "nope", "file": "x"}, env).focus == "scope", "bad scope focuses scope")
    check(LoadModelNode.resolve({"scope": "loras", "file": "style.safetensors"}, env).state == "error", "LoRA is an error here")

    registry = NodeRegistry(packages=("next.nodes",))
    classes, errors = registry.load()
    check("LoadModelNode" in classes and not errors, f"registry finds the node ({errors})")
    via = resolve_node(registry, "LoadModelNode", {"scope": "diffusion_models", "file": "anima.safetensors"}, env)
    check(via["state"] == "ready" and [p["name"] for p in via["outputs"]] == ["denoiser", "adapter"], "served through resolve_node")
    check(via["outputs"][0]["label"] == "DiT", "the served label follows the family")

    def one_node(params):
        return GraphDoc((NodeSpec("n_aaaaaa", "LoadModelNode", params),), (), (), {})

    runner = GraphRunner(registry, env=env)
    outcome = runner.run(one_node({"scope": "checkpoints", "file": "sdxl_full.safetensors", "dtype": "bf16"}))
    check(outcome.ok, f"a run loads the handles ({outcome.error})")
    check(outcome.results and outcome.results[0].ok, "the node reports success")
    out = LoadModelNode(ExecutionContext(env=env)).build(scope="checkpoints", file="sdxl_full.safetensors", family="auto",
                                                          dtype="bf16")
    check(isinstance(out.get("vae"), VAEPart) and isinstance(out.get("clip_g"), ClipGPart), "outputs are the declared part types")
    check(all(isinstance(v, ModelPart) and v.dtype == "bf16" for v in out.values()), "requested dtype travels with the handle")
    check(out["denoiser"].source.path.endswith("sdxl_full.safetensors") and out["denoiser"].source.prefix == "model.diffusion_model.",
          "handle records file and prefix")
    bad = runner.run(one_node({"scope": "checkpoints", "file": "junk.safetensors"}))
    check(not bad.ok and "graph invalid" in bad.error, f"a graph with an unresolvable file is refused before running ({bad.error})")
    raises(ValueError, lambda: LoadModelNode(ExecutionContext(env=env)).build(
        scope="checkpoints", file="junk.safetensors", family="auto", dtype="auto"), "build refuses what resolve refused")
    check(list_families() == [{"id": "sdxl", "label": "SDXL"}, {"id": "anima", "label": "Anima"}], "families are listed")


def run_all():
    fx = Fixture()
    cases = []
    for fn in (sdxl_cases, anima_cases, failure_cases, cache_cases, node_cases):
        def case(fn=fn):
            fn(fx)
        case.__name__ = fn.__name__
        cases.append(case)
    try:
        run(*cases)
    finally:
        fx.tmp.cleanup()


if __name__ == "__main__":
    run_all()
