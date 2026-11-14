"""Auto-caption: the loop, the cleanup, the node -- with a stub captioner.

No caption model runs here (none is reachable in the environment this was
written in), so the `transformers` backend is only checked for being
registered and failing clearly when the package is missing. Everything around
the model is real: a real metadata.db, real image files, real sqlite writes.

Pins: only empty captions are filled (a re-run resumes, nothing the user wrote
is replaced) and `overwrite` replaces all; the original image is preferred and
the VAE preview is the fallback; missing images, model failures and empty
outputs are counted, not fatal; prefix/suffix/lowercase; limit; a cancel keeps
the work done; a stub backend registered by name is reachable through the
node; the captions then reach the loader.
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
from PIL import Image

from manager.captioning import (CAPTIONER_BACKENDS, Captioner, caption_dataset,
                                clean_caption, make_captioner, register_captioner)
from manager.db import (_connect, add_shard, add_source, add_trajectory,
                        init_local_db)
from manager.loader import ManagedDatasetLoader
from manager.storage import ShardWriter

ManagedDatasetLoader._pin_batch = staticmethod(lambda batch: batch)


def check(condition, message):
    if not condition:
        raise AssertionError(message)


class Stub(Captioner):
    def __init__(self, **_):
        self.seen = []
        self.closed = False

    def caption(self, image):
        self.seen.append(image.size)
        r, g, b = image.getpixel((0, 0))
        if (r, g, b) == (255, 0, 0):
            return "A red square. "
        if (r, g, b) == (0, 255, 0):
            raise RuntimeError("model blew up")
        if (r, g, b) == (0, 0, 255):
            return "   "
        return "  Grey   thing,"

    def close(self):
        self.closed = True


def make(tmp: Path, prompts, colors, missing=(), previews=False):
    """A dataset of len(prompts) images; image i has colors[i]; `missing`
    indexes have their original deleted (preview kept when previews=True)."""
    images = tmp / "imgs"; images.mkdir()
    ds = tmp / "ds"; (ds / "shards").mkdir(parents=True); (ds / "previews").mkdir()
    db = ds / "metadata.db"
    init_local_db(db)
    src = add_source(db, "s", "real", config={"image_dir": str(images)})
    writer = ShardWriter(ds / "shards" / "x0.safetensors")
    ids = [writer.add_image_latent(torch.randn(1, 4, 8, 8)) for _ in prompts]
    count, size = writer.write()
    shard = add_shard(db, "shards/x0.safetensors", count, size)
    for i, (p, c) in enumerate(zip(prompts, colors)):
        name = f"img{i}.png"
        if i not in missing:
            Image.new("RGB", (16, 16), c).save(images / name)
        prev = None
        if previews:
            prev = f"previews/p{i}.webp"
            Image.new("RGB", (8, 8), c).save(ds / prev, "PNG")
        add_trajectory(db, src, shard, ids[i], 1, i, p, preview_path=prev,
                       source_path=name, latent_h=8, latent_w=8)
    return ds


def prompts_of(ds):
    with _connect(ds / "metadata.db") as conn:
        return [r["prompt"] for r in conn.execute("SELECT prompt FROM trajectories ORDER BY id")]


GREY, RED, GREEN, BLUE = (128, 128, 128), (255, 0, 0), (0, 255, 0), (0, 0, 255)


def check_fills_only_empty_and_resumes():
    print("[only empty captions are filled; user-written kept; re-run is a no-op]")
    with tempfile.TemporaryDirectory() as tmp:
        ds = make(Path(tmp), ["", "mine", ""], [RED, RED, GREY])
        stub = Stub()
        rep = caption_dataset(ds, stub)
        check(prompts_of(ds) == ["A red square", "mine", "Grey thing"], prompts_of(ds))
        check((rep.captioned, rep.skipped_has_caption, rep.total) == (2, 1, 3), rep)
        check(len(stub.seen) == 2, "the kept caption's image was captioned anyway")
        again = caption_dataset(ds, Stub())
        check(again.captioned == 0 and again.skipped_has_caption == 3, again)
    print("    PASS")


def check_overwrite_replaces_everything():
    print("[overwrite=True replaces all captions]")
    with tempfile.TemporaryDirectory() as tmp:
        ds = make(Path(tmp), ["old", "older"], [RED, GREY])
        caption_dataset(ds, Stub(), overwrite=True)
        check(prompts_of(ds) == ["A red square", "Grey thing"], prompts_of(ds))
    print("    PASS")


def check_problems_are_counted_not_fatal():
    print("[model failure / empty output / missing image: counted, run continues]")
    with tempfile.TemporaryDirectory() as tmp:
        ds = make(Path(tmp), ["", "", "", ""], [RED, GREEN, BLUE, GREY], missing={3})
        rep = caption_dataset(ds, Stub(), allow_preview=False)
        check(prompts_of(ds) == ["A red square", "", "", ""], prompts_of(ds))
        check((rep.captioned, rep.failed, rep.empty_output, rep.missing_image) == (1, 1, 1, 1), rep)
        check("model blew up" in rep.errors[0], rep.errors)
        check("1 captioned of 4" in rep.summary(), rep.summary())
    print("    PASS")


def check_preview_is_the_fallback():
    print("[original gone: the VAE preview is captioned (and counted); allow_preview off skips]")
    with tempfile.TemporaryDirectory() as tmp:
        ds = make(Path(tmp), ["", ""], [RED, GREY], missing={0}, previews=True)
        stub = Stub()
        rep = caption_dataset(ds, stub)
        check(rep.used_preview == 1 and rep.captioned == 2, rep)
        check(stub.seen[0] == (8, 8) and stub.seen[1] == (16, 16), stub.seen)
    with tempfile.TemporaryDirectory() as tmp:
        ds = make(Path(tmp), [""], [RED], missing={0}, previews=True)
        rep = caption_dataset(ds, Stub(), allow_preview=False)
        check(rep.missing_image == 1 and rep.captioned == 0, rep)
    print("    PASS")


def check_text_cleanup():
    print("[clean_caption: whitespace, trailing divider, case, prefix/suffix, empty stays empty]")
    check(clean_caption("  A  dog,  ") == "A dog", clean_caption("  A  dog,  "))
    check(clean_caption("A Dog.", lowercase=True) == "a dog", "lowercase")
    check(clean_caption("a dog", prefix="sks style", suffix="photo") == "sks style, a dog, photo", "affixes")
    check(clean_caption("", prefix="sks") == "" and clean_caption("  ", suffix="x") == "", "empty")
    check(clean_caption("A Dog", prefix="SKS", lowercase=True) == "SKS, a dog", "prefix keeps case")
    print("    PASS")


def check_limit_and_cancel_keep_the_work():
    print("[limit caps captions; cancel stops between images and keeps what was written]")
    with tempfile.TemporaryDirectory() as tmp:
        ds = make(Path(tmp), ["", "", "", ""], [GREY] * 4)
        rep = caption_dataset(ds, Stub(), limit=2)
        check(rep.captioned == 2 and prompts_of(ds).count("") == 2, (rep, prompts_of(ds)))
    with tempfile.TemporaryDirectory() as tmp:
        ds = make(Path(tmp), ["", "", "", ""], [GREY] * 4)
        calls = {"n": 0}
        def cancel():
            calls["n"] += 1
            return calls["n"] > 2          # allow two images, then stop
        rep = caption_dataset(ds, Stub(), should_cancel=cancel, commit_every=100)
        check(rep.cancelled and rep.captioned == 2, rep)
        check(prompts_of(ds).count("Grey thing") == 2, "cancelled work was not committed")
    print("    PASS")


def check_registry_and_node():
    print("[backend by name reaches the node; model closed after; captions reach the loader]")
    check("transformers" in CAPTIONER_BACKENDS, "transformers backend not registered")
    try:
        make_captioner("nope")
    except ValueError:
        pass
    else:
        check(False, "unknown backend accepted")
    made = []
    def factory(**kw):
        made.append(Stub()); return made[-1]
    register_captioner("stub", factory)
    try:
        from nodes.components.layout import ProjectLayout
        from nodes.dataset.auto_caption import AutoCaptionNode
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            make_ds_root = root / "datasets"; make_ds_root.mkdir()
            ds = make(make_ds_root, ["", ""], [RED, GREY])   # -> datasets/ds
            layout = ProjectLayout(comfy_dir=root, checkpoints_dir=root / "c",
                                   loras_dir=root / "l", datasets_dir=make_ds_root,
                                   runs_dir=root / "r")
            out = AutoCaptionNode().build(dataset_root=Path("ds"), backend="stub",
                                          project_layout=layout, prefix="sks")
            check(out["dataset_root"] == Path("ds"), out)
            check("2 captioned of 2" in out["summary"], out)
            check(made and made[0].closed, "captioner not closed")
            check(prompts_of(ds) == ["sks, A red square", "sks, Grey thing"], prompts_of(ds))
            got = sorted(b["prompt"] for b in ManagedDatasetLoader(ds, batch_size=1, shuffle=False))
            check(got == ["sks, A red square", "sks, Grey thing"], got)
    finally:
        CAPTIONER_BACKENDS.pop("stub", None)
    print("    PASS")


def check_transformers_backend_fails_clearly_without_the_package():
    print("[transformers backend: clear error when the package is absent]")
    try:
        import transformers  # noqa: F401
    except ImportError:
        try:
            make_captioner("transformers")
        except RuntimeError as exc:
            check("pip install transformers" in str(exc), exc)
            print("    PASS")
            return
        check(False, "no error without transformers")
    print("    SKIP (transformers is installed; the model itself is not exercised here)")


def main():
    check_fills_only_empty_and_resumes()
    check_overwrite_replaces_everything()
    check_problems_are_counted_not_fatal()
    check_preview_is_the_fallback()
    check_text_cleanup()
    check_limit_and_cancel_keep_the_work()
    check_registry_and_node()
    check_transformers_backend_fails_clearly_without_the_package()
    print("\nSMOKE TEST: ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
