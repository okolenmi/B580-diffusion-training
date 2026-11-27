"""Steps of a browser check that several checks share: creating an app, moving between the app page and its graph, adding a node."""

from __future__ import annotations

import re
import time


def create_app(page, name: str, start: str = "An empty app") -> None:
    page.goto("/apps")
    page.page.select_option("#new-graph", label=start)
    page.page.fill("#new-name", name)
    page.page.click("#btn-new")
    wait_url(page, f"/apps/{name}")
    page.page.wait_for_selector("#btn-graph")


def wait_url(page, fragment: str, timeout: float = 15000) -> None:
    page.page.wait_for_function("f => location.href.includes(f)", arg=fragment, timeout=timeout)


def add_number(page) -> None:
    page.page.wait_for_selector("#palette .ed-palette-item")
    page.page.fill("#palette-search", "Number")
    page.page.wait_for_function("() => document.querySelectorAll('#palette .ed-palette-item').length === 1")
    page.page.locator("#palette .ed-palette-item", has_text="Number").first.click()
    page.page.wait_for_timeout(300)


def node_count(page) -> int:
    return page.page.locator("#g2-canvas .g2-node").count()


def edit_mode(page) -> None:
    """The author's buttons (Edit graph, Save, Delete) exist in Edit mode only."""
    page.page.wait_for_selector("#mode-edit")
    if not page.page.locator("#btn-graph").is_visible():
        page.page.click("#mode-edit")
        page.page.wait_for_selector("#btn-graph", state="visible")


def open_app_graph(page, name: str) -> None:
    """From the app's page into its graph in the editor (the Save bar is up when this returns)."""
    edit_mode(page)
    page.page.wait_for_function("() => !document.querySelector('#btn-graph').disabled")
    for attempt in range(3):  # a click that lands while the page is still drawing itself is lost
        page.page.click("#btn-graph")
        try:
            wait_url(page, f"/graph?app={name}", timeout=8000)
            break
        except Exception:  # noqa: BLE001
            if attempt == 2:
                raise
    page.wait_text("#app-bar-title", "version")


def back_to_app(page, name: str) -> None:
    page.page.click("#app-back")
    wait_url(page, f"/apps/{name}")
    page.page.wait_for_selector("#btn-run")
    page.page.wait_for_timeout(800)  # the page settles its cards and the run button's state after it loads



def card(page, title: str):
    """The app-page card whose title is exactly ``title``."""
    return page.page.locator("article.bd-block").filter(has=page.page.locator(".bd-title", has_text=re.compile(f"^{re.escape(title)}$"))).first


def card_text(page, title: str) -> str:
    try:
        return card(page, title).inner_text(timeout=2000)
    except Exception:  # noqa: BLE001 - a card on another tab or not drawn yet reads as empty
        return ""


def wait_card(page, title: str, fragment: str, timeout: float = 60.0) -> bool:
    """True once the card's text holds ``fragment`` (polled, so a card on another tab or a late redraw is fine)."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if fragment in card_text(page, title):
            return True
        page.page.wait_for_timeout(250)
    return False


def open_tab(page, title: str) -> None:
    page.page.locator(".bd-tab", has_text=title).first.click()
    page.page.wait_for_timeout(400)
