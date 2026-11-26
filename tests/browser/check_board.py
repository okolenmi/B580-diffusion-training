"""The apps page editor (TESTING 98) and its field cards (104): sections, headings, tabs, Place in, Undo, Delete, saving and using the page, Phone view,
and the File card of Load Model as a drop-down of the checkpoints folder. A real server and Chromium; no model is needed.

    python -m tests.browser.check_board
"""

from __future__ import annotations

from tests.support import check
from . import fixtures
from .actions import card, create_app, open_tab, wait_card
from .check_runs import idle, stop_all
from .server import TestServer
from .session import Chromium


def block_titles(page, tab: str | None = None) -> list[str]:
    if tab:
        open_tab(page, tab)
    return page.page.evaluate("() => [...document.querySelectorAll('#board article.bd-block .bd-title')].map(e => e.textContent.trim())")


def editor_steps(server, browser):
    server.api("PUT", "/api/v1/graphs/library/img", fixtures.image_graph(server.home, steps=3, out="board-out"))
    page = browser.new_page()
    create_app(page, "board", start="A copy of img (7 nodes)")
    page.page.wait_for_timeout(1200)
    page.page.click("#mode-edit")
    page.page.wait_for_timeout(500)
    start = len(block_titles(page, "Run"))
    page.page.click("#add-section")
    page.page.wait_for_timeout(300)
    page.page.click("#add-heading")
    page.page.wait_for_timeout(300)
    check(len(block_titles(page)) == start + 2, "a Section and a heading are added to the page")
    page.page.click(".bd-tab-tool:text('+')")
    page.page.wait_for_timeout(400)
    tabs = page.page.evaluate("() => [...document.querySelectorAll('.bd-tab')].map(e => e.textContent.trim())")
    check(len(tabs) == 4, f"a tab is added with + ({tabs})")
    new_tab = tabs[-1]
    open_tab(page, "Run")
    card(page, "Images").locator(".bd-title").click()
    page.page.wait_for_timeout(300)
    page.page.locator("#inspector select", has=page.page.locator("option", has_text=f"Page › {new_tab}")).select_option(label=f"Page › {new_tab}")
    page.page.wait_for_timeout(400)
    check("Images" not in block_titles(page, "Run") and "Images" in block_titles(page, new_tab), "Place in moves a card to another tab")
    page.page.keyboard.press("Control+z")
    page.page.wait_for_timeout(400)
    open_tab(page, "Run")
    check("Images" in block_titles(page), "Undo (Ctrl+Z) brings the card back")
    before = len(block_titles(page, "Run"))
    page.page.locator("article.bd-block", has_text="Section").last.locator(".bd-title").click()
    page.page.keyboard.press("Delete")
    page.page.wait_for_timeout(300)
    check(len(block_titles(page)) == before - 1, "Delete removes the selected card")
    open_tab(page, "Run")
    card(page, "Images").locator(".bd-title").click()
    page.page.locator("#inspector select", has=page.page.locator("option", has_text=f"Page › {new_tab}")).select_option(label=f"Page › {new_tab}")
    page.page.wait_for_timeout(400)
    page.page.click("#btn-save")
    page.page.wait_for_function("() => !document.querySelector('#btn-save').classList.contains('is-dirty')")
    page.goto("/apps/board", settle_ms=1500)
    page.page.click("#mode-use")
    check(page.visible("#board") and "Images" in block_titles(page, new_tab), "after a reload the page keeps its tabs and the moved card")
    check("Section" not in " ".join(block_titles(page, "Run")), "and the section that was deleted stays deleted")
    page.page.click("#btn-run")
    open_tab(page, new_tab)
    page.page.wait_for_function("() => { const i = document.querySelector('#board img'); return !!i && i.naturalWidth > 0; }", timeout=60000)
    check(True, "a run fills the picture on its own tab")
    idle(server)
    page.page.click("#btn-phone")
    check(page.page.get_attribute("#board", "class").find("bd-phone") >= 0 and page.page.get_attribute("#btn-phone", "aria-pressed") == "true", "Phone view turns on and stays on")
    page.page.click("#btn-phone")


def file_card(server, browser):
    """The Load Model of the real starter (not the stand-in) lists the checkpoints folder."""
    import starters
    for name in ("alpha.safetensors", "beta.safetensors"):
        (server.comfy / "models" / "checkpoints" / name).write_bytes(b"x" * 64)
    doc = starters.sdxl_image(model="alpha.safetensors", output=str(server.home / "file-out"), steps=4, width=64, height=64).as_dict()
    server.api("PUT", "/api/v1/graphs/library/real", doc)
    page = browser.new_page()
    create_app(page, "files", start=f"A copy of real ({len(doc['nodes'])} nodes)")
    page.page.wait_for_timeout(1500)
    file_box = card(page, "File")
    select = file_box.locator("select")
    check(select.count() == 1, "the File card of Load Model is a drop-down")
    options = select.locator("option").all_inner_texts()
    check(any("alpha" in o for o in options) and any("beta" in o for o in options), f"it lists the checkpoints of the models folder ({options})")
    check(card(page, "Prompt").locator("textarea").count() == 1, "the Prompt card holds a text area")
    check(card(page, "Seed").locator("input").count() >= 1, "a number card holds its value in a box")


def main() -> None:
    with TestServer() as server, Chromium(server.url) as browser:
        for part in (editor_steps, file_card):
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
