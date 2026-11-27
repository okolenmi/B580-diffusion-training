"""Image generation (TESTING 97, 105): the SDXL and Anima image starters on tiny CPU stand-ins, driven from the apps page and the API: the
picture on the page and what its file carries, batch seeds, the adapter stack, Stop while sampling, Anima's size and sampler rules.

    python -m tests.browser.check_generation
"""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

import numpy as np
from PIL import Image

from tests.support import check
from . import fixtures
from .actions import back_to_app, card, card_text, create_app, open_app_graph, open_tab, wait_card
from .check_runs import executions, idle, stop_all, training_app
from .server import ApiError, TestServer
from .session import Chromium


def pngs(server, folder: str) -> dict[str, Path]:
    return {p.name: p for p in sorted((server.home / folder).glob("*.png"))}


def pixels(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB")).astype(np.int16)


def generate(server, graph: dict, timeout: float = 90.0) -> dict:
    """Run a graph through the API and wait for it; returns the finished execution row."""
    before = {e["execution_id"] for e in executions(server)}
    for attempt in range(20):  # the card is released a moment after the last run reports its end
        try:
            server.api("POST", "/api/v1/graphs/run", {"graph": graph, "mode": "parallel"})
            break
        except ApiError as error:
            if error.status != 409 or attempt == 19:
                raise
            print(f"  (the card was still held by the run before; waiting, attempt {attempt + 1})")
            time.sleep(0.5)
    idle(server, timeout)
    return next(e for e in executions(server) if e["execution_id"] not in before)


def seed_of(name: str) -> int:
    return int(name.split("_s")[1].split("_")[0])


def picture_on_the_page(server, browser):
    server.api("PUT", "/api/v1/graphs/library/img", fixtures.image_graph(server.home, steps=4, prompt="a test", seed=3, out="page-images"))
    page = browser.new_page()
    page.goto("/apps")
    create_app(page, "imgapp", start="A copy of img (7 nodes)")
    open_app_graph(page, "imgapp")
    back_to_app(page, "imgapp")
    page.page.click("#btn-run")
    check(wait_card(page, "Run status", "FINISHED", 90), "the image app runs to the end")
    check(wait_card(page, "Images", "seed 3 · 64×64", 20), f"the Images card names the seed and size ({card_text(page, 'Images')[:40]!r})")
    image = page.page.locator("#board img").first
    check(image.evaluate("i => i.naturalWidth") == 64, "and shows the picture itself")
    files = pngs(server, "page-images")
    check(len(files) == 1, f"one PNG was written ({list(files)})")
    info = Image.open(next(iter(files.values()))).info
    recipe = json.loads(info.get("lab", "{}")).get("recipe", {})
    check(recipe.get("seed") == 3 and recipe.get("steps") == 4 and recipe.get("prompt") == "a test", f"the file carries its recipe ({recipe})")
    check("Steps: 4" in info.get("parameters", "") and "Seed: 3" in info.get("parameters", ""), "and the text other tools read")
    page.page.reload()
    page.page.wait_for_selector("#btn-run")
    page.page.wait_for_timeout(1500)
    check(page.page.locator("#board img").count() >= 1, "the picture is still there after a reload")
    with page.page.context.expect_page() as popup:
        page.page.locator("#board img").first.click()
    check("/artifacts/" in popup.value.url, "a click opens the picture in full size")


def a_changed_setting_shows_a_new_picture(server, browser):
    """Same prompt and seed, another setting (a model change does the same through the file name): the card must not keep showing the first picture."""
    server.api("PUT", "/api/v1/graphs/library/img", fixtures.image_graph(server.home, steps=3, prompt="a test", seed=3, out="again-images"))
    page = browser.new_page()
    create_app(page, "again", start="A copy of img (7 nodes)")
    open_app_graph(page, "again")
    back_to_app(page, "again")
    source = lambda: page.page.evaluate("() => document.querySelector('#board img')?.src || ''")  # noqa: E731
    page.page.click("#btn-run")
    page.page.wait_for_function("() => { const i = document.querySelector('#board img'); return !!i && i.naturalWidth > 0; }", timeout=90000)
    first = source()
    idle(server)
    box = card(page, "Guidance").locator("input").first
    box.fill("2.5")
    box.press("Tab")
    page.page.wait_for_timeout(700)
    page.page.click("#btn-run")
    try:
        page.page.wait_for_function("(old) => { const i = document.querySelector('#board img'); return !!i && i.src !== old && i.naturalWidth > 0; }", arg=first, timeout=90000)
        changed = True
    except Exception:  # noqa: BLE001
        changed = False
    check(changed, "after another guidance value the card shows the new picture, not the one before")
    idle(server)
    names = sorted(p.name for p in (server.home / "again-images").glob("*.png"))
    check(len(names) >= 1, f"the output folder holds the pictures ({len(names)})")


def batch_and_seeds(server, browser):
    generate(server, fixtures.image_graph(server.home, steps=3, seed=5, batch=4, out="batch"))
    batch = pngs(server, "batch")
    check(sorted(seed_of(n) for n in batch) == [5, 6, 7, 8], f"a batch of 4 from seed 5 makes seeds 5-8 ({sorted(seed_of(n) for n in batch)})")
    generate(server, fixtures.image_graph(server.home, steps=3, seed=7, batch=1, out="single"))
    single = next(iter(pngs(server, "single").values()))
    same = next(p for n, p in batch.items() if seed_of(n) == 7)
    check(np.array_equal(pixels(single), pixels(same)), "the third image of the batch equals a single run with that seed")


def adapter_stack(server, browser):
    page = training_app(server, browser, "adapter_source", steps=4, save_every=4)
    page.page.click("#btn-run")
    check(wait_card(page, "Run status", "FINISHED", 90), "a tiny adapter is trained to use")
    source = server.wait_for("the adapter file", lambda: next((server.home / "out-adapter_source").rglob("toy_step000004.safetensors"), None), 30)
    shutil.copy(source, server.comfy / "models" / "loras" / "toy.safetensors")
    common = {"steps": 3, "seed": 2}
    none = generate(server, fixtures.image_graph(server.home, out="a-none", **common))
    zero = generate(server, fixtures.image_graph(server.home, out="a-zero", adapters=(("toy.safetensors", 0.0),), **common))
    off = generate(server, fixtures.image_graph(server.home, out="a-off", adapters=({"file": "toy.safetensors", "strength": 1.0, "on": False},), **common))
    full = generate(server, fixtures.image_graph(server.home, out="a-full", adapters=(("toy.safetensors", 1.0),), **common))
    check(all(e["status"] == "finished" for e in (none, zero, off, full)), f"all four runs finish ({[(e['status'], str(e.get('error'))[:200]) for e in (none, zero, off, full) if e['status'] != 'finished']})")
    one = lambda folder: pixels(next(iter(pngs(server, folder).values())))  # noqa: E731
    check(np.array_equal(one("a-none"), one("a-zero")), "an adapter at strength 0 gives the picture without it")
    check(np.array_equal(one("a-none"), one("a-off")), "an adapter switched off gives the picture without it")
    check(not np.array_equal(one("a-none"), one("a-full")), "an adapter at strength 1 changes the picture")


def stop_while_sampling(server, browser):
    server.api("PUT", "/api/v1/graphs/library/slow", fixtures.image_graph(server.home, steps=200, width=256, height=256, batch=4, out="slow-images"))
    page = browser.new_page()
    page.goto("/apps")
    create_app(page, "slowapp", start="A copy of slow (7 nodes)")
    open_app_graph(page, "slowapp")
    back_to_app(page, "slowapp")
    page.page.click("#btn-run")
    check(wait_card(page, "Sampling", " / 200", 60), "sampling is under way")
    page.page.wait_for_timeout(1500)
    page.page.click("#btn-stop")
    check(wait_card(page, "Run status", "STOPPED", 60), "Stop ends sampling")
    check(not pngs(server, "slow-images"), "and writes no picture")
    idle(server)
    generate(server, fixtures.image_graph(server.home, steps=3, out="after-stop"))
    check(len(pngs(server, "after-stop")) == 1, "the next generation works")


def anima_rules(server, browser):
    good = generate(server, fixtures.anima_graph(server.home, steps=3))
    check(good["status"] == "finished" and len(pngs(server, "anima-images")) == 1, f"Anima generates ({good['status']}: {good.get('error')})")
    wrong = generate(server, fixtures.anima_graph(server.home, steps=3, width=72, height=64, out="anima-bad"))
    check(wrong["status"] == "error" and "16" in str(wrong.get("error")), f"a width that is not a multiple of 16 is refused in words ({str(wrong.get('error'))[:90]})")
    check(not pngs(server, "anima-bad"), "and nothing is written")


def main():
    with TestServer() as server, Chromium(server.url) as browser:
        for part in (picture_on_the_page, a_changed_setting_shows_a_new_picture, batch_and_seeds, adapter_stack, stop_while_sampling, anima_rules):
            print(f"\n{part.__name__}")
            try:
                part(server, browser)
            except Exception as error:  # noqa: BLE001
                check(False, f"{part.__name__} stopped: {type(error).__name__}: {str(error)[:300]}")
            stop_all(server)
        problems = browser.problems()
        print()
        check(not problems, "no script errors or server errors on any page" + (f": {problems[:4]}" if problems else ""))


if __name__ == "__main__":
    from tests.support import finish
    main()
    finish()
