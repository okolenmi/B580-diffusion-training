#!/usr/bin/env python3
"""Probe two reported UI bugs against a live server (read-only, no mutation):

1. System console: does a native resize stick after the window is moved,
   or does it snap back to the previous size?
2. Dataset item checkboxes: measure every .ds-item-check box on screen --
   are some of them collapsed/thin?

Usage: pw-python scripts/probe_ui_bugs.py [base_url]   (default 127.0.0.1:8766)
Diagnostics only; prints measurements, exits 0.
"""

import json
import sys

from playwright.sync_api import sync_playwright

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8766"

CONSOLE_JS = """() => {
  const w = document.getElementById('fconsole');
  if (!w || w.hidden) return null;
  const r = w.getBoundingClientRect();
  return { x: r.x, y: r.y, w: r.width, h: r.height,
           inlineW: w.style.width, inlineH: w.style.height,
           stored: localStorage.getItem('shell.console.v1') };
}"""


def drag(page, x1, y1, x2, y2, steps=12):
    page.mouse.move(x1, y1)
    page.mouse.down()
    for i in range(1, steps + 1):
        page.mouse.move(x1 + (x2 - x1) * i / steps, y1 + (y2 - y1) * i / steps)
    page.mouse.up()


def probe_console(page):
    print("== console resize/move ==")
    page.goto(f"{BASE}/")
    page.wait_for_selector("#fconsole")
    # reset to defaults so the probe is deterministic
    page.evaluate("localStorage.removeItem('shell.console.v1')")
    page.reload()
    page.wait_for_selector("#fconsole")
    base = page.evaluate(CONSOLE_JS)
    print("initial:      ", json.dumps(base))

    # A) resize FIRST (control) via native grip (bottom-right corner)
    r = page.evaluate(CONSOLE_JS)
    drag(page, r["x"] + r["w"] - 6, r["y"] + r["h"] - 6,
         r["x"] + r["w"] + 120, r["y"] + r["h"] + 80)
    after_resize = page.evaluate(CONSOLE_JS)
    print("after resize: ", json.dumps(after_resize))

    # B) now MOVE by the header, then re-measure
    r = after_resize
    hx, hy = r["x"] + r["w"] / 2, r["y"] + 14
    drag(page, hx, hy, hx - 150, hy - 60)
    after_move = page.evaluate(CONSOLE_JS)
    print("after move:   ", json.dumps(after_move))
    # moving must never change the window's size (border-box ratchet bug)
    same = (after_move["w"], after_move["h"]) == (r["w"], r["h"])
    print("move kept size:", same, "OK" if same
          else f"*** BUG: shrank {r['w']}x{r['h']} -> "
               f"{after_move['w']}x{after_move['h']} ***")

    # C) resize AGAIN after the move -- the reported bug
    r = after_move
    drag(page, r["x"] + r["w"] - 6, r["y"] + r["h"] - 6,
         r["x"] + r["w"] + 100, r["y"] + r["h"] + 60)
    after_resize2 = page.evaluate(CONSOLE_JS)
    print("resize#2:     ", json.dumps(after_resize2))
    grew = (after_resize2["w"], after_resize2["h"]) != \
           (after_move["w"], after_move["h"])
    print("resize#2 changed size:", grew,
          "OK" if grew else "*** BUG: resize ignored ***")


def probe_checkboxes(page):
    print("== dataset item checkboxes ==")
    page.goto(f"{BASE}/datasets")
    page.wait_for_selector(".ds-card")
    # open the first dataset that has items (real data, read-only)
    cards = page.query_selector_all(".ds-card")
    print(f"{len(cards)} dataset cards")
    for card in cards:
        card.click()
        page.wait_for_timeout(700)
        if page.query_selector("#items-grid .ds-item"):
            break

    # browse mode must expose NO selection UI
    n_browse = page.locator("#items-grid .ds-item-check").count()
    sel_hidden = page.locator("#select-all-label").is_hidden()
    print(f"browse: {n_browse} checkboxes, select-all hidden={sel_hidden}")
    print("browse invariant:",
          "OK" if n_browse == 0 and sel_hidden else "*** BUG ***")

    # edit mode: measure every checkbox
    page.click(".seg[data-mode='edit']")
    page.wait_for_timeout(300)
    checks = page.query_selector_all(".ds-item-check")
    if not checks:
        print("no item checkboxes found in edit mode")
        return
    sizes = []
    for c in checks[:40]:
        b = c.bounding_box()
        if b:
            sizes.append((round(b["width"], 1), round(b["height"], 1)))
    print(f"{len(sizes)} checkboxes; distinct (w,h): "
          f"{sorted(set(sizes))}")
    thin = [s for s in sizes if s[0] < 14 or s[1] < 14]
    print("thin boxes:", thin if thin else "none",
          "*** BUG" if thin else "OK")
    # bulk bar must NOT be up (nothing selected yet)
    print("bulk bar hidden in fresh edit mode:",
          page.locator("#bulk-bar").is_hidden())


with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page(viewport={"width": 1600, "height": 950})
    probe_console(page)
    probe_checkboxes(page)
    browser.close()
print("PROBE DONE")
