"""Pictures on nodes, node size and the Sampling Preview node (TESTING 107), on the tiny image starter in the graph editor of an app. A real server and
Chromium; no model is needed.

    python -m tests.browser.check_editor
"""

from __future__ import annotations

from PIL import Image

from tests.support import check
from . import fixtures
from .actions import back_to_app, create_app, node_count, open_app_graph
from .check_runs import idle, stop_all
from .server import ROOT, TestServer
from .session import Chromium

PREVIEW_UID = "n_hhhhhh"


def with_sampling_preview(graph: dict, every: int = 2) -> dict:
    graph["nodes"].append({"uid": PREVIEW_UID, "class_name": "SamplingPreviewNode", "params": {"every": every}})
    graph["edges"] += [{"from_node": "n_aaaaaa", "from_port": "vae", "to_node": PREVIEW_UID, "to_port": "vae"},
                       {"from_node": PREVIEW_UID, "from_port": "preview", "to_node": "n_dddddd", "to_port": "preview"}]
    return graph


def app_in_editor(server, browser, name: str, graph: dict):
    server.api("PUT", f"/api/v1/graphs/library/{name}", graph)
    page = browser.new_page()
    create_app(page, name, start=f"A copy of {name} ({len(graph['nodes'])} nodes)")
    open_app_graph(page, name)
    page.page.wait_for_selector("#g2-canvas .g2-node")
    page.page.wait_for_timeout(500)
    return page


def run_in_editor(server, page) -> None:
    page.page.click("#btn-run")
    server.wait_for("the run to start", lambda: any(e.get("status") in ("running", "finished") for e in __import__("tests.browser.check_runs", fromlist=["x"]).executions(server)), 60)
    idle(server)


def box(page, selector: str) -> dict:
    return page.page.evaluate("(s) => { const r = document.querySelector(s).getBoundingClientRect(); return {w: r.width, h: r.height, x: r.x, y: r.y}; }", selector)


def drag_corner(page, grip, dx: int, dy: int) -> None:
    box_ = grip.bounding_box()
    x, y = box_["x"] + box_["width"] / 2, box_["y"] + box_["height"] / 2
    page.page.mouse.move(x, y)
    page.page.mouse.down()
    page.page.mouse.move(x + dx, y + dy, steps=8)
    page.page.mouse.up()
    page.page.wait_for_timeout(300)


def preview_card(page):
    return page.page.locator(".g2-node", has=page.page.locator(".g2-preview")).first


def picture_and_size(server, browser):
    page = app_in_editor(server, browser, "pic", fixtures.image_graph(server.home, steps=3, out="pic-out"))
    run_in_editor(server, page)
    page.page.wait_for_function("() => { const i = document.querySelector('.g2-node .g2-preview img'); return !!i && i.naturalWidth > 0; }", timeout=20000)
    check(True, "Preview Image draws the picture on the canvas itself")
    node = preview_card(page)
    uid = node.get_attribute("data-uid") or node.get_attribute("data-node") or ""
    before = node.bounding_box()
    grip = node.locator(".g2-grip")
    node.hover()
    drag_corner(page, grip, 120, 90)
    after = node.bounding_box()
    check(after["width"] > before["width"] + 60 and after["height"] > before["height"] + 40, f"dragging the corner widens and heightens a card with a picture ({before['width']:.0f}x{before['height']:.0f} -> {after['width']:.0f}x{after['height']:.0f})")
    other = page.page.locator(".g2-node", has=page.page.locator(".g2-head", has_text="Sampler")).first
    o_before = other.bounding_box()
    other.hover()
    drag_corner(page, other.locator(".g2-grip"), 100, 100)
    o_after = other.bounding_box()
    check(o_after["width"] > o_before["width"] + 50 and abs(o_after["height"] - o_before["height"]) < 3, f"another node changes its width only ({o_before['width']:.0f}x{o_before['height']:.0f} -> {o_after['width']:.0f}x{o_after['height']:.0f})")
    page.page.click("#app-save")
    page.page.wait_for_function("() => !document.querySelector('#app-bar-title').innerText.includes('unsaved')")
    layout = stored_layout(server, "pic")
    picture_uid = next(uid for uid, lay in layout.items() if "h" in lay)
    check("w" in layout[picture_uid] and layout["n_dddddd"].get("w", 0) > 300 and "h" not in layout["n_dddddd"],
          f"the sizes are saved with the graph: a height only for the picture card ({layout[picture_uid]}, sampler {layout['n_dddddd']})")
    page.goto("/graph?app=pic", settle_ms=1500)
    page.page.wait_for_selector(".g2-node .g2-preview")
    sampler = page.page.locator(".g2-node", has=page.page.locator(".g2-head", has_text="Sampler")).first.bounding_box()
    shown = preview_card(page).bounding_box()
    ratio = shown["width"] / sampler["width"]
    expected = layout[picture_uid]["w"] / layout["n_dddddd"]["w"]
    check(abs(ratio - expected) < 0.05 * expected, f"after a reload the cards are drawn at their saved sizes (width ratio {ratio:.2f}, saved {expected:.2f})")
    preview_card(page).hover()
    w0 = preview_card(page).bounding_box()["width"]
    preview_card(page).locator(".g2-grip").dblclick()
    page.page.wait_for_timeout(300)
    check(preview_card(page).bounding_box()["width"] < w0 - 20, "the card is drawn at its default size again")
    page.page.click("#app-save")
    page.page.wait_for_function("() => !document.querySelector('#app-bar-title').innerText.includes('unsaved')")
    check(not {"w", "h"} & set(stored_layout(server, "pic")[picture_uid]), "a double click on the corner gives back the default size (nothing is stored)")


def stored_layout(server, name: str) -> dict:
    doc = server.api("GET", f"/api/v1/apps/{name}")["doc"]
    return {n["uid"]: {k: v for k, v in n.get("layout", {}).items() if k in ("w", "h")} for n in doc["graph"]["nodes"]}


def pixels(path):
    with Image.open(path) as image:
        return image.convert("RGB").tobytes()


def sampling_preview(server, browser):
    plain = fixtures.image_graph(server.home, steps=10, out="sp-plain")
    previewed = with_sampling_preview(fixtures.image_graph(server.home, steps=10, out="sp-with"))
    before = set(p.name for p in (ROOT / "artifacts").rglob("*") if p.is_file()) if (ROOT / "artifacts").exists() else set()
    page = app_in_editor(server, browser, "sp", previewed)
    check(node_count(page) == 8, "the Sampling Preview node is wired into the app's graph")
    run_in_editor(server, page)
    card = page.page.locator(".g2-node", has=page.page.locator(".g2-head", has_text="Sampling Preview")).first
    page.page.wait_for_function("() => { const i = [...document.querySelectorAll('.g2-node')].find(n => n.textContent.includes('Sampling Preview'))?.querySelector('.g2-preview img'); return !!i && i.naturalWidth > 0; }", timeout=20000)
    check(True, "the Sampling Preview card fills with a picture during the run")
    frames = sorted(p.name for p in (ROOT / "artifacts").rglob("*") if p.is_file() and p.name not in before and PREVIEW_UID in str(p))
    check(len(frames) == 5, f"a frame every 2 steps of 10 and the last one: 5 pictures ({len(frames)})")
    server.api("POST", "/api/v1/graphs/run", {"graph": plain, "mode": "auto"})
    idle(server)
    a = sorted((server.home / "sp-with").rglob("*.png"))
    b = sorted((server.home / "sp-plain").rglob("*.png"))
    check(len(a) == 1 and len(b) == 1 and pixels(a[0]) == pixels(b[0]), "the final picture is the same with and without the preview node")


def main() -> None:
    with TestServer() as server, Chromium(server.url) as browser:
        for part in (picture_and_size, sampling_preview):
            print(f"\n{part.__name__}")
            try:
                part(server, browser)
            except Exception as error:  # noqa: BLE001
                check(False, f"{part.__name__} stopped: {type(error).__name__}: {str(error)[:400]}")
            stop_all(server)
        problems = [p for p in browser.problems() if "409" not in p]
        print()
        check(not problems, "no script errors or server errors on any page" + (f": {problems[:4]}" if problems else ""))


if __name__ == "__main__":
    from tests.support import finish
    main()
    finish()
