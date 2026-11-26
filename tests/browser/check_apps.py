"""Apps own their graph (TESTING 106): empty apps, the graph editor's app session with its Save bar, copies that stay independent of the library,
and the page's cards when the graph changes. A real server and a real Chromium; no model is needed (the graphs hold plain number nodes).

    python -m tests.browser.check_apps
"""

from __future__ import annotations

from tests.support import check, run
from .server import ApiError, TestServer
from .actions import add_number, create_app, node_count, open_app_graph, wait_url
from .session import Chromium

EMPTY_PAGE = {"root": {"columns": 12, "tabs": [{"id": "main", "title": "Main", "blocks": []}]}}


def number_graph(count: int = 2) -> dict:
    nodes = [{"uid": f"n_{chr(97 + i) * 6}", "class_name": "FloatNode", "params": {"value": float(i)}} for i in range(count)]
    return {"format": 3, "nodes": nodes, "edges": []}


def empty_app_and_editor_session(server, browser):
    page = browser.new_page()
    page.goto("/graph")
    add_number(page)
    check(node_count(page) == 1, "the free canvas takes a node")
    create_app(page, "alpha")
    check(page.page.locator("#btn-run").is_disabled(), "a new app cannot run before it is saved")
    page.page.click("#btn-graph")
    wait_url(page, "/graph?app=alpha")
    check(page.wait_text("#app-bar-title", "version 1") and page.visible("#app-bar"), "Edit graph opens the app's graph with its bar (version 1)")
    check(node_count(page) == 0, "the canvas holds the app's own (empty) graph, not the free one")
    check(not page.visible("#library-section"), "the library and import/export are hidden in an app session")
    page.page.click("#app-back")
    wait_url(page, "/apps/alpha")
    page.page.wait_for_selector("#btn-run")
    page.page.wait_for_timeout(1500)
    check(page.page.locator("#btn-run").is_disabled() and "no nodes yet" in (page.page.get_attribute("#btn-run", "title") or ""), "an app with an empty graph cannot run, and says why")
    page.page.click("#btn-graph")
    wait_url(page, "/graph?app=alpha")
    page.wait_text("#app-bar-title", "version 1")
    add_number(page)
    check(page.wait_text("#app-bar-title", "unsaved changes"), "a change marks the bar unsaved")
    page.page.click("#app-save")
    check(page.wait_text("#app-bar-title", "version 2") and "unsaved" not in page.text("#app-bar-title"), "Save writes the next version and clears the mark")
    add_number(page)
    page.page.keyboard.press("Control+s")
    check(page.wait_text("#app-bar-title", "version 3"), "Ctrl+S saves too")
    stored = server.api("GET", "/api/v1/apps/alpha")
    check(stored["version"] == 3 and stored["node_count"] == 2, f"the server holds both nodes in version 3 (got v{stored['version']}, {stored['node_count']} nodes)")
    dialogs = []
    page.page.on("dialog", lambda d: (dialogs.append(d.type), d.dismiss()))
    add_number(page)
    page.page.evaluate("() => window.dispatchEvent(new Event('beforeunload', {cancelable: true}))")
    prevented = page.page.evaluate("() => { const e = new Event('beforeunload', {cancelable: true}); window.dispatchEvent(e); return e.defaultPrevented; }")
    check(prevented, "leaving with unsaved changes is held back by the browser's own question")
    page.page.click("#app-save")
    page.wait_text("#app-bar-title", "version 4")
    page.page.click("#app-back")
    wait_url(page, "/apps/alpha")
    page.goto("/graph")
    check(node_count(page) == 1, "the free canvas waited untouched with its one node")
    links = page.page.locator("#app-links a")
    check(links.count() == 1 and links.first.get_attribute("href") == "/graph?app=alpha", "the editor's left panel links to the app's graph")
    return page


def copies_stay_independent(server, browser):
    server.api("PUT", "/api/v1/graphs/library/source", number_graph(2))
    page = browser.new_page()
    page.goto("/apps")
    options = page.page.locator("#new-graph option").all_inner_texts()
    check(any("source" in o for o in options), f"the library graph is offered as a start ({options})")
    create_app(page, "beta", start=next(o for o in options if "source" in o))
    page.page.click("#btn-graph")
    wait_url(page, "/graph?app=beta")
    check(server.api("GET", "/api/v1/apps/beta")["node_count"] == 2, "the app starts with a copy of the graph (2 nodes)")
    page.page.wait_for_selector("#g2-canvas .g2-node")
    check(node_count(page) == 2, "the copy opens in the editor")
    add_number(page)
    page.page.click("#app-save")
    page.wait_text("#app-bar-title", "version 2")
    library = server.api("GET", "/api/v1/graphs/library/source")
    check(len(library.get("nodes") or library.get("graph", {}).get("nodes", [])) == 2, "changing the app's graph leaves the library graph alone")
    server.api("PUT", "/api/v1/graphs/library/source", number_graph(1))
    check(server.api("GET", "/api/v1/apps/beta")["node_count"] == 3, "changing the library graph leaves the app alone")


def cards_lose_their_target(server, browser):
    page = browser.new_page()
    create_app(page, "gamma", start="An empty app")
    page.page.click("#btn-graph")
    wait_url(page, "/graph?app=gamma")
    add_number(page)
    page.page.click("#app-save")
    page.wait_text("#app-bar-title", "version 2")
    stored = server.api("GET", "/api/v1/apps/gamma")
    uid = stored["doc"]["graph"]["nodes"][0]["uid"]
    card = {"id": "c1", "x": 0, "y": 0, "w": 6, "h": 2, "collapsed": False, "title": None, "options": {}, "ref": {"scope": "node", "kind": "field", "name": "value", "node": uid}}
    doc = {**stored["doc"], "root": {"columns": 12, "tabs": [{"id": "main", "title": "Main", "blocks": [card]}]}}
    server.api("PUT", "/api/v1/apps/gamma", {"description": "", "doc": doc})
    page.goto("/graph?app=gamma")
    page.page.wait_for_selector("#g2-canvas .g2-node")
    page.page.locator("#g2-canvas .g2-node .g2-head").first.click()
    page.page.evaluate("() => document.querySelector('#g2-canvas').dispatchEvent(new KeyboardEvent('keydown', {key: 'Delete', bubbles: true}))")
    page.page.wait_for_timeout(300)
    check(node_count(page) == 0, "the node is removed in the app's editor")
    page.page.click("#app-save")
    check(page.says("1 card on its page lost their target"), "Save says how many cards lost their target")
    page.page.click("#app-back")
    wait_url(page, "/apps/gamma")
    page.page.wait_for_selector("#btn-run")
    page.page.wait_for_timeout(1500)
    check(page.page.locator("#board .bd-orphan").count() == 1, "and the app page shows that card as one that lost its target")


def app_without_positions(server, browser):
    graph = number_graph(3)
    for n in graph["nodes"]:
        n.pop("layout", None)
    server.api("PUT", "/api/v1/apps/legacy", {"description": "", "doc": {**EMPTY_PAGE, "graph": graph}})
    page = browser.new_page()
    page.goto("/graph?app=legacy", settle_ms=1500)
    boxes = page.page.eval_on_selector_all("#g2-canvas .g2-node", "els => els.map(e => { const r = e.getBoundingClientRect(); return [r.left, r.top, r.right, r.bottom]; })")
    overlap = any(a is not b and a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3] for a in boxes for b in boxes)
    check(len(boxes) == 3 and not overlap, f"an app saved without positions is arranged on arrival, no two nodes overlap ({len(boxes)} nodes)")


def main():
    with TestServer() as server, Chromium(server.url) as browser:
        for part in (empty_app_and_editor_session, copies_stay_independent, cards_lose_their_target, app_without_positions):
            print(f"\n{part.__name__}")
            try:
                part(server, browser)
            except Exception as error:  # noqa: BLE001 - the report names the part; the rest still run
                check(False, f"{part.__name__} stopped: {type(error).__name__}: {str(error)[:300]}")
        problems = browser.problems()
        print()
        check(not problems, "no script errors, failed requests or server errors on any page" + (f": {problems[:4]}" if problems else ""))


if __name__ == "__main__":
    from tests.support import finish
    main()
    finish()
