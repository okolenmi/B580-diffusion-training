"""The pages around the editor (TESTING 96, 102.2, 102.3): start page, Settings, Help, the dataset picture grid, and the console that opens on an error.
A real server and Chromium; the card is the fake one of the tiny profile, so only what does not depend on real hardware is judged here.

    python -m tests.browser.check_pages
"""

from __future__ import annotations

from tests.support import check
from . import fixtures
from .actions import open_app_graph, create_app, back_to_app
from .check_runs import executions, idle, stop_all
from .server import TestServer
from .session import Chromium

ROUTES = ("/", "/settings", "/help", "/datasets", "/apps", "/graph")


def every_page_loads(server, browser):
    page = browser.new_page()
    for route in ROUTES:
        before = len(page.problems)
        page.goto(route, settle_ms=900)
        check(len(page.problems) == before and len(page.text("body")) > 80, f"{route} loads without script or server errors ({page.problems[before:][:2]})")


def start_page(server, browser):
    page = browser.new_page()
    page.goto("/", settle_ms=1500)
    check(all(word in page.text("body") for word in ("SERVER", "GRAPHICS CARD", "PYTORCH", "NODE LIBRARY")), "four tiles are drawn")
    check("All modules loaded" in page.text("body"), "the node library tile says all modules loaded")
    check("1 of 5 done" in page.text("body") and "No runs yet" in page.text("body"), "Getting started counts what is missing (only the dataset exists); no runs are listed yet")
    check("12.0 GB in total" in page.text("body"), "the card memory panel shows the card's total")
    dataset = DATASET[0]
    (server.comfy / "models" / "checkpoints" / "stub.safetensors").write_bytes(b"x" * 64)
    page.goto("/", settle_ms=1500)
    check("2 of 5 done" in page.text("body"), f"a checkpoint and a dataset tick two steps ({[l for l in page.text('body').splitlines() if ' of 5' in l]})")
    server.api("POST", "/api/v1/graphs/run", {"graph": fixtures.lora_graph(server.home, dataset, steps=4, save_every=100, out="start-run"), "mode": "auto"})
    idle(server)
    page.goto("/", settle_ms=1500)
    body = page.text("body")
    check("No runs yet" not in body and "Recent runs" in body, "a finished run is listed under Recent runs")
    check("FINISHED" in body.upper(), "and carries its status")


def settings_page(server, browser):
    page = browser.new_page()
    page.goto("/settings", settle_ms=1500)
    check("same environment" in page.text("body"), "Python environment: the server's own interpreter reads 'same environment'")
    page.page.locator("details summary").first.click()
    page.page.fill('input[aria-label="ComfyUI folder"]', str(server.home / "no" / "such" / "folder"))
    page.page.locator("button:visible", has_text="Save").first.click()
    page.page.wait_for_timeout(1200)
    bad = page.page.locator("input.is-bad")
    check(bad.count() == 1 and bad.first.get_attribute("aria-label") == "ComfyUI folder" and page.page.locator(".field-error:not([hidden])").count() == 1,
          "a refused folder is marked on its own field, and only that one")
    page.page.locator("button:visible:not([disabled])", has_text="Revert").first.click()
    wipe = page.page.locator("button", has_text="Delete run history")
    runs = [e for e in executions(server)]
    check(not wipe.is_disabled() or not runs, "Delete run history is available with finished runs")
    if runs:
        wipe.click()
        check("Really delete" in page.text("body"), "the first click only asks 'Really delete N runs?'")
        page.page.locator("button", has_text="Really delete").click()
        check(page.says("Deleted", 6000) and not executions(server), "the second click deletes the history")
    dataset = DATASET[0]
    server.api("POST", "/api/v1/graphs/run", {"graph": fixtures.lora_graph(server.home, dataset, steps=150, save_every=1000, out="long"), "mode": "auto"})
    server.wait_for("a running run", lambda: any(e.get("status") == "running" for e in executions(server)), 60)
    page.goto("/settings", settle_ms=1500)
    check(page.page.locator("button", has_text="Delete run history").is_disabled() and "still active" in page.text("body"), "Delete run history is disabled while a run is active")
    stop_all(server)


def help_page(server, browser):
    page = browser.new_page()
    page.goto("/help", settle_ms=1200)
    before = page.page.locator("#help-root *").count()
    page.page.fill("#help-search", "memory")
    page.page.wait_for_timeout(500)
    marks = page.page.locator("mark.help-hit").count()
    check(marks > 3 and "found" in page.text("#help-count"), f"searching 'memory' marks the matches in place ({marks} marks, '{page.text('#help-count')}')")
    check(page.page.locator("#help-root *").count() >= before, "nothing is removed from the page while searching")
    page.page.press("#help-search", "Enter")
    first = page.page.evaluate("() => document.querySelector('.is-current-hit')?.id || document.querySelector('.is-current-hit')?.textContent.slice(0, 30)")
    page.page.press("#help-search", "Enter")
    second = page.page.evaluate("() => document.querySelector('.is-current-hit')?.id || document.querySelector('.is-current-hit')?.textContent.slice(0, 30)")
    check(first and second and first != second, f"Enter jumps from one match to the next ({first!r} -> {second!r})")
    page.page.fill("#help-search", "tokenize")
    page.page.wait_for_timeout(400)
    check(page.page.locator("mark.help-hit").count() >= 1, "'tokenize' finds its entry")
    page.page.fill("#help-search", "")
    page.page.wait_for_timeout(300)
    check(page.page.locator("mark.help-hit").count() == 0, "clearing the search removes the marks")
    page.goto("/help#nodes-red", settle_ms=1200)
    open_entry = page.page.evaluate("() => { const e = document.getElementById('nodes-red'); if (!e) return null; const r = e.getBoundingClientRect(); return {open: e.open ?? true, top: r.top, inView: r.top >= 0 && r.top < innerHeight}; }")
    check(open_entry and open_entry["open"] and open_entry["inView"], f"/help#nodes-red opens that entry and brings it into view ({open_entry})")
    for section, label in (("datasets", "Datasets"), ("graph", "The graph editor"), ("memory", "Memory")):
        delta = page.page.evaluate("(id) => { const r = document.getElementById(id).getBoundingClientRect(); return r.top + Math.min(r.height / 2, 150) - innerHeight * 0.4; }", section)
        page.page.mouse.move(700, 400)
        page.page.mouse.wheel(0, delta)  # a real wheel turn: the menu follows the reader, not a script's scroll
        page.page.wait_for_timeout(800)
        current = page.page.evaluate("() => [...document.querySelectorAll('#help-nav .is-current')].map(n => n.textContent.trim())")
        check(len(current) == 1 and label in current[0], f"the sidebar marks '{label}' while it is in the middle of the screen ({current})")
    page.page.click('#help-nav a[href="#graph"]')
    page.page.wait_for_timeout(900)
    top = page.page.evaluate("() => document.getElementById('graph').getBoundingClientRect().top")
    check(0 <= top < 200, f"a click on an entry puts its title at the top ({top:.0f}px)")


def datasets_grid(server, browser, dataset):
    page = browser.new_page()
    page.goto("/datasets", settle_ms=1000)
    page.page.click(f'a.ds-card[href$="{dataset}"]')
    page.page.wait_for_selector(".tile", timeout=10000)
    page.page.wait_for_timeout(500)
    tiles = page.page.locator(".tile")
    check(tiles.count() == 6, f"the dataset opens with its six pictures ({tiles.count()})")
    tiles.nth(0).click(position={"x": 40, "y": 40})
    check(page.page.locator(".tile.on").count() == 1 and page.page.locator(".tile.ticked").count() == 0, "a plain click outlines one picture and ticks nothing")
    tiles.nth(1).locator(".tile-check").click()
    on = page.page.locator(".tile.on")
    check(on.count() == 1 and page.page.locator(".tile.ticked").count() == 1 and "1" in (page.page.get_attribute(".tile.on", "data-id") or "1"), "a plain click then another picture's corner box ticks only the second")
    page.page.locator(".tile-check").first.click()  # reset to one ticked
    tiles.nth(2).click(position={"x": 40, "y": 40})
    tiles.nth(3).click(position={"x": 40, "y": 40}, modifiers=["Control"])
    check(page.page.locator(".tile.on").count() == 2 and page.page.locator(".tile.ticked").count() == 2, "a plain click then Ctrl-click gives both, ticked")


def console_on_error(server, browser, dataset):
    broken = fixtures.lora_graph(server.home, dataset, out="broken")
    broken["edges"] = []
    server.api("PUT", "/api/v1/graphs/library/broken", broken)
    page = browser.new_page()
    create_app(page, "broken", start="A copy of broken (15 nodes)")
    open_app_graph(page, "broken")
    page.page.wait_for_timeout(1200)
    check(page.visible("#fconsole-fab") and not page.visible("#fconsole"), "the console starts docked")
    page.page.click("#btn-run")
    check(page.wait_text("#fconsole", "Graph invalid") and page.visible("#fconsole"), "a docked console opens by itself on the error line")
    page.page.click("#fconsole-min")
    page.page.wait_for_timeout(300)
    check(page.visible("#fconsole-fab"), "and can be docked again")
    page.page.click("#fconsole-fab")
    page.page.wait_for_timeout(300)
    page.page.click("#btn-run")
    page.page.wait_for_timeout(1500)
    check(page.visible("#fconsole"), "an open console stays as it is on the next error")


def console_is_kept(server, browser):
    """The console is the server's journal: a line stays after a reload, shows on other pages, and a run that failed says why without anyone being on its page."""
    page = browser.new_page()
    page.goto("/settings", settle_ms=1200)
    server.api("POST", "/api/v1/journal", {"level": "warn", "text": "Disk almost full (kept-line)"})
    page.page.wait_for_function("() => document.querySelector('#fconsole-fab')?.dataset.unread === '1'", timeout=10000)
    check(True, "a warning that arrives while the console is docked is counted on its button")
    page.page.click("#fconsole-fab")
    check(page.wait_text("#fconsole", "kept-line"), "and is in the console when it is opened")
    other = browser.new_page()
    other.goto("/datasets", settle_ms=1500)
    other.page.click("#fconsole-fab")
    check(other.wait_text("#fconsole", "kept-line"), "another page, opened later, shows the same line (nothing was lost to the reload)")
    graph = fixtures.image_graph(server.home, steps=3, width=68, out="doomed")  # a size the sampler refuses when the run is under way
    server.api("POST", "/api/v1/graphs/run", {"graph": graph, "mode": "auto"})
    idle(server)
    told = other.wait_text("#fconsole", "multiple of 8", 30)
    check(told, "a failed run is told in the console with its reason, on a page that did not start it" + ("" if told else f" (the journal holds {[e['text'] for e in server.api('GET', '/api/v1/journal?after=0')['entries']][-6:]}, the page shows {other.text('#console-output')[-300:]!r})"))
    other.page.select_option("#fconsole-filter", "errors")
    shown = other.page.locator("#console-output .console-line").all_inner_texts()
    check(shown and not any("kept-line" in t or "server started" in t for t in shown) and any("multiple of 8" in t for t in shown), f"the filter keeps only errors ({shown})")
    other.page.click("#fconsole-clear")
    check("Nothing to show" in other.text("#console-output"), "Clear hides what is shown")
    other.goto("/", settle_ms=1500)
    if other.visible("#fconsole-fab"):
        other.page.click("#fconsole-fab")
    check("Nothing to show" in other.text("#console-output"), "and stays cleared after a reload")


DATASET: list[str] = []


def main() -> None:
    with TestServer() as server, Chromium(server.url) as browser:
        DATASET.append(fixtures.toy_dataset(server.home))
        for part in (every_page_loads, start_page, settings_page, help_page, lambda s, b: datasets_grid(s, b, DATASET[0]), lambda s, b: console_on_error(s, b, DATASET[0]), console_is_kept):
            name = getattr(part, "__name__", "part")
            print(f"\n{name if name != '<lambda>' else 'datasets_grid / console_on_error'}")
            try:
                part(server, browser)
            except Exception as error:  # noqa: BLE001
                check(False, f"{name} stopped: {type(error).__name__}: {str(error)[:300]}")
            stop_all(server)
        problems = [p for p in browser.problems() if not any(code in p for code in ("409", "422", "400 (Bad"))]
        print()
        check(not problems, "no script errors or server errors on any page" + (f": {problems[:4]}" if problems else ""))


if __name__ == "__main__":
    from tests.support import finish
    main()
    finish()
