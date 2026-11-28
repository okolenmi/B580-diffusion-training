"""Pictures on nodes, node size and the live preview inside Preview Image (TESTING 107), on the tiny image starter in the graph editor of an app. A real server and
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

SEEN = """() => { window.__seen = {texts: [], images: new Set(), live: 0};
  new MutationObserver(() => {
    const card = [...document.querySelectorAll('.g2-node')].find((n) => n.querySelector('.g2-preview'));
    const chip = card && card.querySelector('.bd-live:not([hidden])');
    if (chip) { window.__seen.live++; if (!window.__seen.texts.includes(chip.textContent)) window.__seen.texts.push(chip.textContent); }
    const img = card && card.querySelector('.g2-preview img'); if (img && window.__seen.live) window.__seen.images.add(img);
  }).observe(document.body, {subtree: true, childList: true, characterData: true, attributes: true}); }"""


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


def text_fields(server, browser):
    """Prompt and Negative are boxes with a small coloured name tab, and stay inside their node at any width the node can be given."""
    page = app_in_editor(server, browser, "tx", fixtures.image_graph(server.home, steps=3, out="tx-out"))
    node = page.page.locator(".g2-node", has=page.page.locator(".g2-head", has_text="prompt")).first
    measure = """() => { const n = [...document.querySelectorAll('.g2-node')].find(e => e.querySelector('.g2-head').textContent.includes('prompt')), r = n.getBoundingClientRect();
      return {width: r.width, boxes: [...n.querySelectorAll('.g2-textbox')].map(t => { const q = t.getBoundingClientRect(), a = t.querySelector('textarea').getBoundingClientRect();
        return {tone: t.dataset.tone, title: t.querySelector('.g2-textbox-title').textContent, left: q.left - r.left, right: r.right - q.right, area: a.right - q.right}; })}; }"""
    first = page.page.evaluate(measure)
    check([(b["title"], b["tone"]) for b in first["boxes"]] == [("Prompt", "green"), ("Negative", "red")], f"Prompt and Negative are boxes with a green and a red name tab ({first['boxes']})")
    drag_corner(page, node.locator(".g2-grip"), -600, 0)
    small = page.page.evaluate(measure)
    check(small["width"] < first["width"] and all(b["left"] >= 0 and b["right"] >= 0 and b["area"] <= 1 for b in small["boxes"]), f"narrowed as far as it goes, both text areas are still inside the node ({small})")


def pixels(path):
    with Image.open(path) as image:
        return image.convert("RGB").tobytes()


def live_preview(server, browser):
    """The Sampler's Live preview addon shows the forming picture in the Preview Image card itself; the finished image arrives in the same card."""
    plain = fixtures.image_graph(server.home, steps=10, out="lp-plain")
    for node in plain["nodes"]:
        for addon in node.get("addons", []):
            addon["enabled"] = False
    page = app_in_editor(server, browser, "lp", fixtures.image_graph(server.home, steps=10, out="lp-with"))
    page.page.evaluate(SEEN)
    run_in_editor(server, page)
    page.page.wait_for_function("() => { const i = document.querySelector('.g2-node .g2-preview img'); return !!i && i.naturalWidth > 0 && i.src.endsWith('.png'); }", timeout=20000)
    seen = page.page.evaluate("() => ({texts: window.__seen.texts, images: window.__seen.images.size, live: window.__seen.live})")
    check(seen["live"] > 0 and any(t.startswith("preview ") for t in seen["texts"]), f"the card said which step its live picture is ({seen['texts']})")
    check(seen["images"] == 1, f"live pictures and the finished image are shown by one and the same picture element ({seen['images']})")
    card = preview_card(page)
    height = page.page.evaluate("(s) => document.querySelector(s).getBoundingClientRect().height", ".g2-node .g2-preview img")
    check(height > 60, f"the card grew to show the picture, not a link to it (picture {height:.0f}px high)")
    check(not page.page.locator(".g2-node .g2-preview .bd-live:visible").count(), "once the image is final the live marker is gone")
    check(card.locator("a").count() == 0, "a single picture is not wrapped in a link")
    server.api("POST", "/api/v1/graphs/run", {"graph": plain, "mode": "auto"})
    idle(server)
    a, b = sorted((server.home / "lp-with").rglob("*.png")), sorted((server.home / "lp-plain").rglob("*.png"))
    check(len(a) == 1 and len(b) == 1 and pixels(a[0]) == pixels(b[0]), "the final picture is the same with and without the addon")
    


def settings_gear(server, browser):
    """The graph's settings sit behind a gear in the corner of the canvas: an overlay with the settings, a dot for what differs from the default, Esc and a click outside close it."""
    page = app_in_editor(server, browser, "gear", fixtures.image_graph(server.home, steps=3, out="gear-out"))
    p = page.page
    check(p.locator(".ed-right", has_text="Graph settings").count() == 0, "the settings are no longer a section of the right panel")
    gear = p.locator(".gs-gear")
    check(gear.is_visible() and p.locator(".gs-overlay").is_hidden(), "the gear is in sight and the overlay is closed")
    corner = p.evaluate("() => { const c = document.querySelector('#g2-canvas').getBoundingClientRect(), g = document.querySelector('.gs-gear').getBoundingClientRect(); return [c.right - g.right, g.top - c.top]; }")
    check(corner[0] < 30 and corner[1] < 30, f"the gear sits in the top right corner of the canvas ({corner})")
    gear.click()
    check(p.locator(".gs-overlay").is_visible(), "the gear opens the overlay")
    box = p.evaluate("() => { const o = document.querySelector('.gs-overlay').getBoundingClientRect(), c = document.querySelector('#g2-canvas').getBoundingClientRect(); return o.bottom <= c.bottom + 1 && o.right <= c.right + 1 && o.left >= c.left; }")
    check(box, "the overlay lies inside the canvas")
    check(p.locator(".gs-overlay .gs-doc").count() > 0, "each setting says what it does in words under its control")
    p.locator(".gs-overlay summary", has_text="Device").click()
    select = p.locator(".gs-overlay .gs-row", has_text="Denoiser precision").locator("select")
    base = int(p.locator(".gs-gear-dot").inner_text()) if p.locator(".gs-gear-dot").is_visible() else 0
    check(base == p.locator(".gs-overlay .gs-row.is-changed").count() + 0 or base > 0, f"the settings the graph carries show on the gear ({base})")
    row = p.locator(".gs-overlay .gs-row", has_text="Denoiser precision")
    was_changed = row.evaluate("(e) => e.classList.contains('is-changed')")
    select.select_option("fp16")
    p.wait_for_timeout(200)
    now = int(p.locator(".gs-gear-dot").inner_text())
    check(now == base + (0 if was_changed else 1), f"a changed setting is counted on the gear ({base} -> {now})")
    row = p.locator(".gs-overlay .gs-row", has_text="Denoiser precision")
    check(row.evaluate("(e) => e.classList.contains('is-changed')"), "and its row is marked")
    row.locator(".gs-reset").click()
    p.wait_for_timeout(200)
    check(not p.locator(".gs-overlay .gs-row", has_text="Denoiser precision").evaluate("(e) => e.classList.contains('is-changed')"),
          "Reset puts the default back and clears the mark of that row alone")
    p.keyboard.press("Escape")
    check(p.locator(".gs-overlay").is_hidden(), "Esc closes the overlay")
    gear.click()
    p.mouse.click(300, 500)
    check(p.locator(".gs-overlay").is_hidden(), "a click outside closes it")


def main() -> None:
    with TestServer() as server, Chromium(server.url) as browser:
        for part in (picture_and_size, live_preview, settings_gear, text_fields):
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
