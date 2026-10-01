"""Visual smoke for the main pages: idle state, mocked running state,
interactions, the config editor (schema form + raw), run detail views,
a regression pass over monitor/graph (they share style.css), the
dataset manager against the real library (incl. the M8f card preview
thumbs and the item ⋮ context menu), and the M8d shell (icon rail,
floating console, help/settings).

Not auto-discovered by run_all.py (no ``test_`` prefix): it needs a
LIVE backend and the Playwright venv.

    # terminal 1 (scratch DB, port 8766):
    #   BACKEND_DB_PATH=/tmp/opencode/smoke.db \
    #     /home/okolenmi/comfy/venv/bin/python -m backend.cli --port 8766
    # terminal 2:
    ~/.venvs/pw/bin/python backend/tests/visual_smoke.py

BASE and SMOKE_OUT override the backend URL and screenshot directory.
Exits non-zero on any failed check. See docs/design/backend/06.
"""
import json
import os
import re
import shutil
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

from playwright.sync_api import sync_playwright

BASE = os.environ.get("BASE", "http://127.0.0.1:8766")
OUT = Path(os.environ.get("SMOKE_OUT", "/tmp/opencode/training_smoke"))
OUT.mkdir(parents=True, exist_ok=True)

failures: list[str] = []
console_noise: list[str] = []


def check(cond, label):
    print(("  ok  " if cond else "  FAIL") + f"  {label}")
    if not cond:
        failures.append(label)


def hook_console(page, tag):
    page.on(
        "console",
        lambda m: console_noise.append(
            f"[{tag}] {m.type}: {m.text} @ {m.location.get('url', '')}"
        )
        if m.type in ("error", "warning")
        else None,
    )
    page.on("pageerror", lambda e: console_noise.append(f"[{tag}] pageerror: {e}"))


def is_expected_noise(text):
    # Browser network-log lines for deliberate contract 404s: the idle
    # probe of /runs/active (no_active_run) and scenario E1's probe of
    # a run id that never existed (run_not_found) -- both render honest
    # states in-app; the browser still logs the resource status.
    if "404" not in text:
        return False
    return "runs/active" in text or "runs/9999" in text


def main():
    now = datetime.now(timezone.utc)
    iso = lambda dt: dt.isoformat()

    active_run = {
        "id": 13,
        "status": "running",
        "config_path": "configs/distill.toml",
        "mode": "lora",
        "phase": "training",
        "total_steps": 1000,
        "done_steps": 412,
        "current_loss": 0.03412,
        "avg_loss": 0.03998,
        "cache_done": 3,
        "cache_total": 10,
        "pid": 4242,
        "exit_code": None,
        "error": None,
        "log_path": "/tmp/runs/13.log",
        "created_at": iso(now - timedelta(seconds=2530)),
        "updated_at": iso(now - timedelta(seconds=5)),
        "started_at": iso(now - timedelta(seconds=2520)),
        "finished_at": None,
    }
    history = [
        active_run,
        {
            "id": 12, "status": "completed", "config_path": "configs/distill.toml",
            "mode": "lora", "phase": None, "total_steps": 1000, "done_steps": 1000,
            "current_loss": 0.021, "avg_loss": 0.0289, "cache_done": None,
            "cache_total": None, "pid": None, "exit_code": 0, "error": None,
            "log_path": "/tmp/runs/12.log",
            "created_at": iso(now - timedelta(hours=5)),
            "updated_at": iso(now - timedelta(hours=3)),
            "started_at": iso(now - timedelta(hours=5)),
            "finished_at": iso(now - timedelta(hours=3)),
        },
        {
            "id": 11, "status": "failed", "config_path": "configs/experiments/long.toml",
            "mode": "full", "phase": None, "total_steps": 0, "done_steps": 417,
            "current_loss": None, "avg_loss": 0.0512, "cache_done": None,
            "cache_total": None, "pid": None, "exit_code": 1, "error": "CUDA OOM",
            "log_path": "/tmp/runs/11.log",
            "created_at": iso(now - timedelta(days=2)),
            "updated_at": iso(now - timedelta(days=2, seconds=-1800)),
            "started_at": iso(now - timedelta(days=2)),
            "finished_at": iso(now - timedelta(days=2, seconds=-1800)),
        },
    ]
    log_text = "\n".join(f"[train] step {i} loss {0.05 - i * 0.001:.4f}" for i in range(40))

    with sync_playwright() as p:
        browser = p.chromium.launch()

        # ---------- A: idle, against the real backend ----------
        print("== A: idle state (real backend) ==")
        ctx = browser.new_context(viewport={"width": 1600, "height": 1000})
        page = ctx.new_page()
        hook_console(page, "A")
        page.goto(f"{BASE}/", wait_until="networkidle")

        check(page.locator("#hero-idle").is_visible(), "idle hero visible")
        check(page.locator("#hero-run").is_hidden(), "running hero hidden")
        check(page.locator("#btn-stop").is_hidden(), "stop button hidden when idle")
        check(page.locator("#btn-kill").is_hidden(), "kill button hidden when idle")
        check(page.locator("#status-badge").inner_text().strip().upper() == "IDLE", "badge = Idle")
        check(page.locator(".page-topbar h1").inner_text() == "System tracker",
              "topbar title (renamed from Training)")
        check(page.locator(".util-strip").is_visible(), "monitor hand-off strip visible")
        check(
            page.locator("table.runs-table th").count() == 6,
            "history table has 6 column headers",
        )
        check(page.locator(".log-pane").is_visible(), "log pane visible")
        # placeholder must actually render (disabled+unselected renders blank)
        sel = page.evaluate(
            """() => { const s = document.getElementById('start-from');"""
            """ return {i: s.selectedIndex, t: s.options[0] ? s.options[0].label : ''}; }"""
        )
        check(sel["i"] == 0 and "config path" in sel["t"],
              f"start-from placeholder renders ({sel})")
        check(page.locator("#btn-wipe").is_disabled(), "wipe disabled with no history")

        # form guard: empty config -> inline error, no request
        page.click("#btn-start")
        check(
            page.locator("#start-error").is_visible()
            and "Config path is required" in page.locator("#start-error").inner_text(),
            "inline start error on empty config",
        )
        page.fill("#cfg-path", "configs/distill.toml")
        check(
            page.locator("#start-error").is_hidden(),
            "error clears when typing",
        )

        page.screenshot(path=str(OUT / "idle.png"), full_page=True)
        check((OUT / "idle.png").stat().st_size > 20000, "idle screenshot captured")
        ctx.close()

        # ---------- B: running + history, API mocked ----------
        print("== B: running state (API mocked) ==")
        ctx = browser.new_context(viewport={"width": 1600, "height": 1000})
        page = ctx.new_page()
        hook_console(page, "B")
        dialogs: list[str] = []
        page.on("dialog", lambda d: (dialogs.append(d.message), d.accept()))

        page.route(
            re.compile(r"/api/v1/runs/active"),
            lambda r: r.fulfill(status=200, content_type="application/json",
                                body=json.dumps(active_run)),
        )
        page.route(
            re.compile(r"/api/v1/runs\?limit=20"),
            lambda r: r.fulfill(status=200, content_type="application/json",
                                body=json.dumps({"runs": history, "count": len(history)})),
        )
        page.route(
            re.compile(r"/api/v1/runs/\d+/log"),
            lambda r: r.fulfill(status=200, content_type="application/json",
                                body=json.dumps({"log": log_text, "lines": 40})),
        )
        page.goto(f"{BASE}/", wait_until="networkidle")

        check(page.locator("#hero-run").is_visible(), "running hero visible")
        check(page.locator("#hero-idle").is_hidden(), "idle hero hidden")
        check(page.locator("#btn-stop").is_visible(), "stop button visible")
        check(page.locator("#btn-kill").is_visible(), "kill button visible")
        badge = page.locator("#status-badge").inner_text().strip().upper()
        check(badge == "RUNNING · TRAINING", f"badge shows phase ({badge!r})")
        check(page.locator("#run-id").inner_text() == "#13", "run id in hero")
        meta = page.locator("#run-meta").inner_text()
        check("configs/distill.toml" in meta and "lora" in meta, f"meta = config · mode ({meta!r})")
        ptext = page.locator("#progress-text").inner_text()
        check(ptext == "412 / 1000 · 41%", f"progress label ({ptext!r})")
        fill_w = page.evaluate("document.getElementById('progress-fill').style.width")
        check(fill_w == "41.2%", f"progress fill width ({fill_w})")
        check(page.locator("#cache-progress-wrap").evaluate("e => getComputedStyle(e).display") != "none",
              "cache sub-bar shown while caching")
        check("cache 3/10" in page.locator("#cache-text").text_content(), "cache label")
        loss = page.locator("#metric-loss").inner_text()
        check(loss == "0.03412", f"loss metric ({loss})")

        # elapsed ticker: wait >1s, text must advance
        elapsed1 = page.locator("#run-elapsed").inner_text()
        page.wait_for_timeout(1600)
        elapsed2 = page.locator("#run-elapsed").inner_text()
        check(elapsed1 != elapsed2, f"elapsed ticker advances ({elapsed1} -> {elapsed2})")

        # history: 3 rows + status chips + selection drives log
        check(page.locator("#history-list tr").count() == 3, "3 history rows")
        check(page.locator("#btn-wipe").is_enabled(), "wipe enabled with history")
        check(page.locator("#history-list .status-completed").count() == 1, "completed chip")
        check(page.locator("#history-list .status-failed").count() == 1, "failed chip")
        check(page.locator("#history-list .status-running").count() == 1, "running chip")
        # boot auto-shows the active run's log (id 13)
        check(page.locator("#log-title").inner_text() == "#13", "active run log shown on boot")
        check("step 39" in page.locator("#log-body").inner_text(), "log body filled")
        # click the completed run's row
        page.click("#history-list tr[data-run-id='12']")
        page.wait_for_timeout(300)
        check(page.locator("#log-title").inner_text() == "#12", "row click switches log")
        check(
            page.locator("#history-list tr[data-run-id='12']").evaluate(
                "e => e.classList.contains('selected')"
            ),
            "clicked row marked selected",
        )
        # hand-off: the quick log links to the full /run/{id} detail
        check(page.locator("#log-open").is_visible(), "'open full' link shown with a selection")
        check(page.locator("#log-open").get_attribute("href") == "/run/12",
              "open-full href follows selection")

        page.screenshot(path=str(OUT / "running.png"), full_page=True)
        check((OUT / "running.png").stat().st_size > 20000, "running screenshot captured")

        # wipe -> confirm dialog captured (handled in code, never blocks)
        page.click("#btn-wipe")
        page.wait_for_timeout(300)
        check(
            any("Delete ALL run history" in m for m in dialogs),
            f"wipe confirm captured ({dialogs})",
        )
        ctx.close()

        # ---------- B2: diverged run + resync-on-open (docs 07 F-03/F-09) ----------
        print("== B2: diverged values + resync on stream open ==")
        diverged = dict(
            active_run, current_loss=None, avg_loss=None,
            nonfinite={"current_loss": "nan", "avg_loss": "inf"},
        )
        active_hits: list[str] = []

        def serve_active(route):
            active_hits.append(route.request.url)
            route.fulfill(status=200, content_type="application/json",
                          body=json.dumps(diverged))

        ctx = browser.new_context(viewport={"width": 1600, "height": 1000})
        page = ctx.new_page()
        hook_console(page, "B2")
        page.route(re.compile(r"/api/v1/runs/active"), serve_active)
        page.route(
            re.compile(r"/api/v1/runs\?limit=20"),
            lambda r: r.fulfill(status=200, content_type="application/json",
                                body=json.dumps({"runs": [diverged, history[1]], "count": 2})),
        )
        page.route(
            re.compile(r"/api/v1/runs/\d+/log"),
            lambda r: r.fulfill(status=200, content_type="application/json",
                                body=json.dumps({"log": log_text, "lines": 40})),
        )
        page.goto(f"{BASE}/", wait_until="networkidle")
        page.wait_for_timeout(600)

        loss_text = page.locator("#metric-loss").inner_text()
        check(
            "NaN" in loss_text and "diverged" in loss_text,
            f"diverged loss is loud, not an em dash ({loss_text!r})",
        )
        check(
            page.locator("#metric-loss").evaluate("e => e.classList.contains('value-bad')"),
            "diverged loss carries the loud class",
        )
        avg_text = page.locator("#metric-avg").inner_text()
        check("diverged" in avg_text, f"diverged avg says so too ({avg_text!r})")
        cell = page.locator("#history-list tr[data-run-id='13'] .col-num").nth(1)
        check("diverged" in cell.inner_text(), f"history cell keeps the marker ({cell.inner_text()!r})")
        check(cell.evaluate("e => e.classList.contains('value-bad')"), "history cell styled loud")

        # /events has no replay: the page refetches the DB on every open.
        check(
            len(active_hits) >= 2,
            f"active run refetched when the stream opened ({len(active_hits)} calls)",
        )
        page.screenshot(path=str(OUT / "diverged.png"), full_page=True)
        check((OUT / "diverged.png").stat().st_size > 20000, "diverged screenshot captured")
        ctx.close()

        # ---------- C: monitor + graph regression (shared CSS surgery) ----------
        for name, path, ready_sel in (
            ("monitor", "/monitor/mon-test", "canvas"),
            ("graph", "/graph", ".ed-palette-item"),
        ):
            print(f"== C: {name} regression ==")
            ctx = browser.new_context(viewport={"width": 1600, "height": 1000})
            page = ctx.new_page()
            hook_console(page, name)
            page.goto(f"{BASE}{path}", wait_until="networkidle")
            # graph's palette items live in collapsed <details> -> attached only
            state = "visible" if name == "monitor" else "attached"
            page.wait_for_selector(ready_sel, state=state, timeout=30000)
            check(page.locator(ready_sel).first.is_visible() if state == "visible"
                  else page.locator(ready_sel).count() > 0,
                  f"{name}: core element renders")
            # the shell (rail + floating console) is global, these pages too
            check(page.locator(".rail").is_visible(), f"{name}: icon rail present")
            check(page.locator("#fconsole").count() == 1,
                  f"{name}: floating console mounted")
            page.screenshot(path=str(OUT / f"{name}.png"), full_page=True)

            if name == "graph":
                # layout contract (graph visibility pass): no page-local
                # console (notes ride the floating one), the search fits
                # its rail, rails collapse to a true 0 track, and the
                # executions drawer starts collapsed
                check(page.locator("#ed-log").count() == 0,
                      "graph: page-local console removed")
                fits = page.evaluate(
                    """() => {
                        const i = document.getElementById('palette-search')
                                      .getBoundingClientRect();
                        const a = document.querySelector('.ed-palette')
                                      .getBoundingClientRect();
                        return i.width > 0 && i.right <= a.right + 0.5
                            && i.left >= a.left - 0.5;
                    }""")
                check(fits, "graph: search box fits the palette rail")
                page.click("#btn-validate")
                check(page.locator("#console-output .console-line",
                                   has_text="Canvas is empty.").count() > 0,
                      "graph: notes land in the floating console")
                check(page.locator("#exec-section")
                          .get_attribute("open") is None,
                      "graph: executions drawer collapsed by default")
                page.click("#exec-section > summary")
                check(page.locator("#exec-list").is_visible(),
                      "graph: executions drawer expands")
                page.click("#exec-section > summary")
                for btn_sel, rail_sel in (
                    ("#btn-toggle-left", ".ed-palette"),
                    ("#btn-toggle-right", ".ed-right"),
                ):
                    page.click(btn_sel)
                    page.wait_for_timeout(250)  # grid transition settles
                    box = page.locator(rail_sel).bounding_box()
                    check(box is not None and box["width"] == 0,
                          f"graph: {rail_sel} rail collapses to 0")
                    page.click(btn_sel)
                    page.wait_for_timeout(250)
                    box = page.locator(rail_sel).bounding_box()
                    check(box is not None and box["width"] > 100,
                          f"graph: {rail_sel} rail restores")
                # canvas keeps the room it gained (toolbar + collapsed
                # drawer are the only chrome above/below it)
                grow = page.evaluate(
                    """() => {
                        const c = document.getElementById('graph-canvas')
                                      .getBoundingClientRect();
                        return c.height / window.innerHeight;
                    }""")
                check(grow > 0.85,
                      f"graph: canvas fills most of the viewport ({grow:.0%})")

                # infinite plane (canvas redesign): the viewport never
                # scrolls, a dashed circle marks the center of the plane,
                # and pan comes from wheel + background drags
                st = page.evaluate(
                    """() => {
                        const v = document.getElementById('graph-canvas');
                        const cs = getComputedStyle(v);
                        return { ox: cs.overflowX, oy: cs.overflowY,
                                 sb: (v.offsetWidth - v.clientWidth)
                                   + (v.offsetHeight - v.clientHeight) };
                    }""")
                check(st["ox"] == "hidden" and st["oy"] == "hidden"
                          and st["sb"] == 0,
                      f"graph: viewport never scrolls ({st['sb']}px chrome)")
                origin = page.evaluate(
                    """() => {
                        const o = document.getElementById('ed-origin')
                                      .getBoundingClientRect();
                        const v = document.getElementById('graph-canvas')
                                      .getBoundingClientRect();
                        return Math.hypot(
                            (o.left + o.width/2) - (v.left + v.width/2),
                            (o.top + o.height/2) - (v.top + v.height/2));
                    }""")
                check(origin <= 2,
                      f"graph: center circle sits in the middle ({origin:.1f}px)")
                cbox = page.locator("#graph-canvas").bounding_box()
                mid = (cbox["x"] + cbox["width"] / 2,
                       cbox["y"] + cbox["height"] / 2)
                get_pan = lambda: page.evaluate(
                    "document.getElementById('canvas-inner').style.transform")
                page.mouse.move(*mid)
                t0 = get_pan()
                page.mouse.wheel(80, 40)
                page.wait_for_timeout(80)
                t1 = get_pan()
                check(t0 != t1, f"graph: wheel pans the plane ({t1})")
                page.mouse.down()
                page.mouse.move(mid[0] - 80, mid[1] - 60, steps=6)
                page.mouse.up()
                check(t1 != get_pan(), "graph: background drag pans the plane")

                # drops cascade side by side (stacked nodes make wire
                # starts ambiguous) and land inside the viewport
                page.click(".ed-domain summary >> nth=0")
                page.click(".ed-palette-item >> nth=0")
                page.wait_for_timeout(120)
                page.click(".ed-palette-item >> nth=1")
                page.wait_for_timeout(220)
                pos = page.evaluate(
                    "[...document.querySelectorAll('.gnode')]"
                    ".map(n => [n.style.left, n.style.top])")
                check(len(pos) == 2 and pos[0] != pos[1],
                      f"graph: drops cascade instead of stacking ({pos})")
                in_view = page.evaluate(
                    """() => {
                        const v = document.getElementById('graph-canvas')
                                      .getBoundingClientRect();
                        return [...document.querySelectorAll('.gnode')]
                            .every(n => {
                                const r = n.getBoundingClientRect();
                                return r.left >= v.left - 1
                                    && r.top >= v.top - 1
                                    && r.right <= v.right + 1
                                    && r.bottom <= v.bottom + 1;
                            });
                    }""")
                check(in_view, "graph: drops land inside the viewport")
                # wiring must survive the transformed plane
                a = page.evaluate(
                    """() => {
                        const p = document.querySelectorAll('.gnode')[0]
                                      .querySelector('.gport.out');
                        const r = p.getBoundingClientRect();
                        return { x: r.x + r.width/2, y: r.y + r.height/2 };
                    }""")
                t = page.evaluate(
                    """() => {
                        const p = document.querySelectorAll('.gnode')[1]
                                      .querySelector('.gport.in');
                        const r = p.getBoundingClientRect();
                        return { x: r.x + r.width/2, y: r.y + r.height/2 };
                    }""")
                page.mouse.move(a["x"], a["y"])
                page.mouse.down()
                page.mouse.move(t["x"], t["y"], steps=8)
                page.mouse.up()
                page.wait_for_timeout(250)
                check(page.locator("path.edge").count() >= 1,
                      "graph: wire connects across the transformed plane")
                # the plane extends past the origin (no non-negative clamp)
                hb = page.locator(".gnode-head").first.bounding_box()
                page.mouse.move(hb["x"] + 40, hb["y"] + 8)
                page.mouse.down()
                page.mouse.move(hb["x"] - 500, hb["y"] - 400, steps=6)
                page.mouse.up()
                lefts = page.evaluate(
                    "[...document.querySelectorAll('.gnode')]"
                    ".map(n => n.style.left)")
                check(any(int(p.replace("px", "")) < 0 for p in lefts),
                      f"graph: plane extends past the origin ({lefts})")

                # ---- editable node bodies: values are set ON the node,
                # pickers read the server's folders, and diagnostics come
                # back live (docs 06 §3) ----
                def show_node(sel):
                    """Pan the plane until sel's center sits at the canvas
                    center (drops and drags land anywhere on the plane)."""
                    for _ in range(3):
                        b = page.locator(sel).first.bounding_box()
                        if b is None:
                            break
                        cb = page.locator("#graph-canvas").bounding_box()
                        mx = (b["x"] + b["width"] / 2) - (cb["x"] + cb["width"] / 2)
                        my = (b["y"] + b["height"] / 2) - (cb["y"] + cb["height"] / 2)
                        if abs(mx) <= 4 and abs(my) <= 4:
                            break
                        page.mouse.move(cb["x"] + cb["width"] / 2,
                                        cb["y"] + cb["height"] / 2)
                        page.mouse.wheel(mx, my)
                        page.wait_for_timeout(80)
                    return page.locator(sel).first.bounding_box()

                def add_via_search(query):
                    page.fill("#palette-search", query)
                    page.wait_for_timeout(60)
                    items = page.locator(".ed-palette-item:visible")
                    check(items.count() == 1,
                          f"graph: palette search '{query}' narrows to one item")
                    items.first.click()
                    page.wait_for_timeout(150)
                    page.fill("#palette-search", "")

                # n1 (the first drop) IS the Managed Dataset Source:
                # checkboxes, star/red-socket required state, visible_when
                # gating and the server-fed dataset picker -- all on the card
                show_node('.gnode[data-id="n1"]')
                n1 = '.gnode[data-id="n1"] '
                check(page.locator(n1 + "input[type=checkbox]").count() == 3,
                      "graph: bool params render as checkboxes on the node")
                check(page.locator(
                    n1 + '.gport-row[data-port="dataset_root"] .gport.req-unmet'
                ).count() == 1,
                      "graph: required-unconnected socket reads red")
                star = page.locator(
                    n1 + '.gport-row[data-port="dataset_root"] .gport-label'
                ).inner_text()
                check("*" in star, f"graph: required input marked with a star ({star!r})")
                opt = page.locator(
                    n1 + '.gport-row[data-port="set_identifier"] .gport-label'
                ).inner_text()
                check("*" not in opt, f"graph: optional input unmarked ({opt!r})")
                check(page.locator(
                    n1 + '.gport-row[data-port="set_identifier"] .gport.req-unmet'
                ).count() == 0,
                      "graph: optional socket not flagged as unmet")
                check(page.locator(n1 + '.gport-row[data-port="t_values"]').is_hidden(),
                      "graph: visible_when hides the gated row on the node")
                # wired sockets read filled (the wire n1->n2 exists by now)
                check(page.locator(
                    '.gnode[data-id="n2"] .gport.in.connected'
                ).count() >= 1,
                      "graph: wired socket reads filled")

                box1 = page.locator(
                    n1 + '.gport-row[data-port="shuffle"] input[type=checkbox]')
                box1.click()
                page.wait_for_timeout(120)
                check(page.locator(
                    n1 + '.gport-row[data-port="shuffle"] input[type=checkbox]'
                ).is_checked(),
                      "graph: clicking a node checkbox commits the value")
                check(page.locator(
                    '#inspector .param-row:has-text("shuffle") input[type=checkbox]'
                ).is_checked(),
                      "graph: inspector mirrors the node's checkbox")

                page.wait_for_timeout(400)  # /assets/dataset catalog arrives
                ds = n1 + '.gport-row[data-port="dataset_root"] select'
                opts = page.locator(ds + " option").count()
                check(opts >= 3, f"graph: dataset picker fed from the server ({opts} options)")
                page.select_option(ds, "1024 aes")
                page.wait_for_timeout(120)
                check(page.locator(
                    '#inspector .param-row:has-text("dataset_root") select'
                ).input_value() == "1024 aes",
                      "graph: inspector mirrors the node's picker value")
                up = page.locator(n1 + '.gport-row[data-port="dataset_root"] .pupload')
                check(up.count() == 1 and up.is_hidden(),
                      "graph: dataset picker offers no upload (catalog-only kind)")

                # Resources Controller: live diagnostics -- nothing for
                # empty params (never fabricated), then the server's ERROR
                # line under checkpoint_path once a missing file is set
                add_via_search("resources controller")
                page.wait_for_selector(
                    '.gnode[data-id="n3"] .gport-row[data-port="checkpoint_path"]')
                page.wait_for_timeout(800)  # debounce + POST round-trip
                check(page.locator('.gnode[data-id="n3"] .gdiag').count() == 0,
                      "graph: empty diagnostics render nothing (no fabrication)")
                show_node('.gnode[data-id="n3"]')
                check(page.locator(
                    '.gnode[data-id="n3"] input[type=checkbox]').count() == 2,
                      "graph: widget_only checkboxes render on the node")
                check(page.locator(
                    '.gnode[data-id="n3"] .gport-row[data-port="continue_training"] .gport'
                ).count() == 0,
                      "graph: widget_only inputs expose no socket")
                check(page.locator(
                    '.gnode[data-id="n3"] .gport-row[data-port="continue_lora_path"]'
                ).is_hidden(),
                      "graph: continue_lora_path gated until continue_training")
                page.evaluate(
                    """() => {
                        const row = document.querySelector(
                            '.gnode[data-id="n3"] .gport-row[data-port="checkpoint_path"]');
                        const sel = row.querySelector('select');
                        const o = document.createElement('option');
                        o.value = 'smoke-missing.safetensors';
                        o.textContent = 'smoke-missing.safetensors (probe)';
                        sel.appendChild(o);
                        sel.value = o.value;
                        sel.dispatchEvent(new Event('change', { bubbles: true }));
                    }""")
                page.wait_for_selector(
                    '.gnode[data-id="n3"] .gport-row[data-port="checkpoint_path"] .gdiag',
                    timeout=6000)
                dtext = page.locator(
                    '.gnode[data-id="n3"] .gport-row[data-port="checkpoint_path"] .gdiag'
                ).inner_text()
                check("ERROR" in dtext,
                      f"graph: live diagnostics update the node ({dtext[:52]!r})")
                # the same toggle reveals the gated picker (+ its upload)
                page.locator(
                    '.gnode[data-id="n3"] .gport-row[data-port="continue_training"] input'
                ).click()
                page.wait_for_timeout(150)
                check(page.locator(
                    '.gnode[data-id="n3"] .gport-row[data-port="continue_lora_path"]'
                ).is_visible(),
                      "graph: checking continue_training reveals its path picker")
                lora_up = page.locator(
                    '.gnode[data-id="n3"] .gport-row[data-port="continue_lora_path"] .pupload'
                )
                page.wait_for_selector(
                    '.gnode[data-id="n3"] .gport-row[data-port="continue_lora_path"] .pupload',
                    state="visible", timeout=5000)
                check(lora_up.is_visible(),
                      "graph: lora picker offers upload from the server folder")

                # Save-As (lora_output): a typed target, not a picker
                add_via_search("checkpoint saver")
                sav = '.gnode[data-id="n4"] .gport-row[data-port="relative_path"] '
                page.wait_for_selector(sav + 'input[type="text"]')
                check(page.locator(sav + 'input[type="text"]').count() == 1,
                      "graph: Save-As path is typed on the node")
                check(page.locator(
                    '.gnode[data-id="n4"] .gport-row[data-port="model"] .gport.req-unmet'
                ).count() == 1,
                      "graph: required pure handle reads red too")
                page.wait_for_selector(sav + ".pupload", state="visible", timeout=5000)
                check(page.locator(sav + ".pupload").is_visible(),
                      "graph: Save-As offers upload into the lora folder")

                # widgets never make the canvas scroll (infinite plane)
                scrolled = page.evaluate(
                    """() => {
                        const v = document.getElementById('graph-canvas');
                        return v.scrollTop || v.scrollLeft;
                    }""")
                check(scrolled == 0,
                      f"graph: canvas stays unscrolled with widgets ({scrolled}px)")
                page.screenshot(path=str(OUT / "graph_nodes.png"), full_page=True)
            ctx.close()

        # ---------- D: config editor (loads read-only; the save step
        # writes a throwaway copy so repo config files stay untouched) ----
        repo_root = Path(__file__).resolve().parents[2]
        smoke_cfg = OUT / "cfg_smoke.toml"
        shutil.copy(repo_root / "runs/hw_validation/legacy_check.toml", smoke_cfg)

        print("== D: config editor ==")
        ctx = browser.new_context(viewport={"width": 1600, "height": 1000})
        page = ctx.new_page()
        hook_console(page, "config")
        page.goto(f"{BASE}/config", wait_until="networkidle")

        check(page.locator("#form-state").is_visible(), "empty state before load")
        check(page.locator("#form-wrap").is_hidden(), "form hidden before load")

        page.fill("#cfg-path", "runs/hw_validation/legacy_check.toml")
        page.click("#btn-load")
        page.wait_for_selector("#config-form .cfg-field", timeout=15000)

        n_fields = page.locator(".cfg-field").count()
        n_groups = page.locator(".cfg-group").count()
        check(n_fields > 40, f"form renders {n_fields} fields")
        check(n_groups >= 6, f"{n_groups} group sections ({n_groups})")
        check(page.locator('[data-id="start_from"]').count() == 0,
              "launch-only option start_from excluded from the editor")
        check(page.locator('[data-id="reset_optimizer"]').count() == 0,
              "launch-only option reset_optimizer excluded from the editor")

        # visible_when: LoRA fields appear only while method == lora
        page.select_option("#f-tuning-method", "lora")
        check(page.locator('[data-id="tuning.rank"]').is_visible(),
              "rank visible when method = lora")
        page.select_option("#f-tuning-method", "distillation")
        check(page.locator('[data-id="tuning.rank"]').is_hidden(),
              "rank hidden when method = distillation")

        # dirty tracking: an edit enables Save + shows the chip; Revert clears
        check(page.locator("#dirty-chip").is_visible(), "dirty chip after edits")
        check(page.locator("#btn-save-form").is_enabled(), "save enabled when dirty")
        page.click("#btn-revert")
        page.wait_for_timeout(300)
        check(page.locator("#dirty-chip").is_hidden(), "chip cleared by revert")
        check(page.locator("#btn-save-form").is_disabled(), "save disabled after revert")

        # raw tab carries the real file content
        page.click("#tab-raw")
        raw_len = len(page.locator("#raw-editor").input_value())
        check(raw_len > 50, f"raw buffer holds the file ({raw_len} chars)")
        check(page.locator("#panel-form").is_hidden(), "form panel hidden on raw tab")
        page.click("#tab-form")
        check(page.locator("#panel-raw").is_hidden(), "raw panel hidden on form tab")

        # E2E save against a throwaway copy (absolute path) -- never the
        # repo's own config files.
        page.fill("#cfg-path", str(smoke_cfg))
        page.click("#btn-load")
        # loadConfig() ends with a console line AFTER it re-renders.
        # Editing before that lands races the reload: the form silently
        # reverts, dirty.clear() runs, and saveForm early-returns on an
        # empty dirty set (both downstream checks then fail on the old
        # value). Wait for the line -- the field itself already exists
        # from the previous load, so it proves nothing about this one.
        page.wait_for_function(
            """p => [...document.querySelectorAll('#console-output .console-line')]"""
            """ .some(l => l.textContent.includes('Loaded ' + p))""",
            arg=str(smoke_cfg), timeout=15000)
        page.wait_for_selector("#f-common-steps", timeout=15000)
        cur_raw = page.input_value("#f-common-steps").strip()
        new_steps = 1200 if cur_raw == "1100" else 1100
        page.fill("#f-common-steps", str(new_steps))
        check(page.locator("#dirty-chip").is_visible(), "dirty chip after steps edit")
        page.click("#btn-save-form")
        page.wait_for_timeout(600)
        check(page.locator("#btn-save-form").is_disabled(),
              "save disabled again after successful save")
        check(page.input_value("#f-common-steps") == str(new_steps),
              "form keeps the saved value")
        page.click("#tab-raw")
        content = page.locator("#raw-editor").input_value()
        check(re.search(rf"\b{new_steps}\b", content) is not None,
              "raw buffer refreshed from the written file")
        page.click("#tab-form")

        page.screenshot(path=str(OUT / "config.png"), full_page=True)
        check((OUT / "config.png").stat().st_size > 20000, "config screenshot captured")
        ctx.close()

        # ---------- E: run detail (/run/{id}) ----------
        print("== E: run detail ==")

        # E1: honest 404 against the real backend (scratch DB, no runs)
        ctx = browser.new_context(viewport={"width": 1600, "height": 1000})
        page = ctx.new_page()
        hook_console(page, "E404")
        page.goto(f"{BASE}/run/9999", wait_until="networkidle")
        check(page.locator("#run-grid").is_hidden(), "404: details stay hidden")
        check("not found" in page.locator("#run-state").inner_text().lower(),
              "404: honest not-found state")
        ctx.close()

        # E2: completed run mocked -- every field renders, log tails
        ctx = browser.new_context(viewport={"width": 1600, "height": 1000})
        page = ctx.new_page()
        hook_console(page, "E")
        run12 = next(r for r in history if r["id"] == 12)
        run11 = next(r for r in history if r["id"] == 11)
        page.route(
            re.compile(r"/api/v1/runs/12/log"),
            lambda r: r.fulfill(status=200, content_type="application/json",
                                body=json.dumps({"log": log_text, "lines": 40})),
        )
        page.route(
            re.compile(r"/api/v1/runs/12$"),
            lambda r: r.fulfill(status=200, content_type="application/json",
                                body=json.dumps(run12)),
        )
        page.goto(f"{BASE}/run/12", wait_until="networkidle")
        page.wait_for_selector("#run-grid", state="visible", timeout=15000)

        check(page.locator("#run-title").inner_text() == "Run #12", "title = Run #12")
        badge_cls = page.locator("#status-badge").get_attribute("class")
        check("status-completed" in badge_cls, f"completed badge class ({badge_cls})")
        n_rows = page.locator("#run-details dt").count()
        check(n_rows == 16, f"16 detail rows ({n_rows})")
        ddetails = page.locator("#run-details").inner_text()
        check("1000 / 1000" in ddetails, "steps rendered")
        # row order is pinned by render(): Exit code is index 9
        check(page.locator("#run-details dd").nth(9).inner_text().strip() == "0",
              "exit code 0 rendered")
        check("ago" in ddetails, "relative time in absolute timestamps")
        # Error row is index 10; a clean run renders "—" with class mono
        # (the error class only applies when run.error is truthy)
        check(page.locator("#run-details dd").nth(10).inner_text() == "—",
              "error empty (—) for a clean run")
        check("step 39" in page.locator("#log-body").inner_text(), "log tail filled")
        # .log-pane is shared chrome in style.css -- if it drifts back to a
        # page stylesheet, run detail loses line structure silently
        ws = page.locator("#log-body").evaluate("e => getComputedStyle(e).whiteSpace")
        check(ws == "pre-wrap", f"log pane keeps line structure ({ws})")
        page.screenshot(path=str(OUT / "run.png"), full_page=True)
        check((OUT / "run.png").stat().st_size > 20000, "run screenshot captured")

        # E3: failed run -- error renders in red, exit code shown
        page.route(
            re.compile(r"/api/v1/runs/11/log"),
            lambda r: r.fulfill(status=200, content_type="application/json",
                                body=json.dumps({"log": log_text, "lines": 40})),
        )
        page.route(
            re.compile(r"/api/v1/runs/11$"),
            lambda r: r.fulfill(status=200, content_type="application/json",
                                body=json.dumps(run11)),
        )
        page.goto(f"{BASE}/run/11", wait_until="networkidle")
        page.wait_for_selector("#run-grid", state="visible", timeout=15000)
        err_dd = page.locator("#run-details dd.error")
        check(err_dd.count() == 1 and err_dd.inner_text() == "CUDA OOM",
              "failed run shows its error text with the error class")
        err_color = err_dd.evaluate("e => getComputedStyle(e).color")
        norm_color = page.locator("#run-details dd.mono").first.evaluate(
            "e => getComputedStyle(e).color")
        check(err_color != norm_color, f"error rendered in red ({err_color})")
        ctx.close()

        # ---------- F: dataset manager (/datasets, real backend) ----------
        print("== F: dataset manager ==")
        ctx = browser.new_context(viewport={"width": 1600, "height": 1000})
        page = ctx.new_page()
        hook_console(page, "F")
        dialogs: list[str] = []
        page.on("dialog", lambda d: (dialogs.append(d.message), d.accept()))

        page.goto(f"{BASE}/datasets", wait_until="networkidle")
        page.wait_for_selector("#ds-grid", state="visible", timeout=15000)
        n0 = page.locator(".ds-card").count()
        check(n0 >= 3, f"library list renders the real datasets ({n0})")

        # card preview thumbs (M8f): one per card, real images resolve
        check(page.locator(".ds-card-thumb").count() == n0,
              "every dataset card renders a preview thumb")
        try:
            page.wait_for_function(
                "() => [...document.querySelectorAll('.ds-card-thumb img')]"
                "  .some(i => i.complete && i.naturalWidth > 0)",
                timeout=15000)
            check(True, "a real card's preview image loads via /files/{path}")
        except Exception as exc:  # noqa: BLE001 -- report, don't kill the run
            check(False, f"a real card's preview image loads ({type(exc).__name__})")

        # the M8d rail hands off to the dataset manager
        page.goto(f"{BASE}/", wait_until="networkidle")
        check(page.locator("a.rail-item[href='/datasets']").count() == 1,
              "rail links to /datasets")
        page.goto(f"{BASE}/datasets", wait_until="networkidle")
        page.wait_for_selector("#ds-grid", state="visible", timeout=15000)

        # create guard: empty name refuses inline, no request fired
        page.click("#btn-new-dataset")
        check(page.locator("#create-card").is_visible(), "create form opens")
        page.click("#btn-create")
        check(page.locator("#create-error").is_visible()
              and "required" in page.locator("#create-error").inner_text(),
              "empty name refused inline")
        check(page.locator(".ds-card").count() == n0,
              "refused create added no dataset")

        # create a throwaway dataset and inspect its honest empty detail
        page.fill("#new-name", "m8c-smoke-ds")
        page.fill("#new-desc", "throwaway -- deleted at the end of the smoke")
        page.click("#btn-create")
        page.wait_for_function(
            "n => document.querySelectorAll('.ds-card').length === n",
            arg=n0 + 1, timeout=10000)
        check(True, "create adds the dataset card")

        page.click("a.ds-card-name[href='/datasets/m8c-smoke-ds']")
        page.wait_for_selector("#detail-wrap", state="visible", timeout=15000)
        check(page.locator("#ds-title").inner_text() == "m8c-smoke-ds",
              "detail title follows the route")
        check(page.locator(".ds-stat").count() == 7, "seven stat chips render")
        first_stat = page.locator(".ds-stat b").first.inner_text()
        check(first_stat == "0",
              f"fresh dataset reports 0 items ({first_stat})")
        check(page.locator("#items-state").is_visible(),
              "items tab honest empty state")
        page.click("#tab-sets")
        check(page.locator("#sets-state").is_visible(),
              "sets tab honest empty state")
        page.click("#tab-tasks")
        check(page.locator("#btn-add-data-tab").is_visible(),
              "add-data entry point renders on the Tasks tab")
        check(page.locator("#add-dialog").is_hidden(),
              "add dialog starts closed")
        check(page.locator("#tasks-state").is_visible(),
              "tasks tab honest empty state")
        page.screenshot(path=str(OUT / "datasets_empty.png"), full_page=True)
        check((OUT / "datasets_empty.png").stat().st_size > 20000,
              "empty-detail screenshot captured")

        # a real curated dataset, read-only: stats, preview bytes, filters, sets
        page.click("#btn-back")
        page.wait_for_selector("#ds-grid", state="visible", timeout=15000)
        page.click("a.ds-card-name[href='/datasets/1024%20aes']")
        page.wait_for_selector("#detail-wrap", state="visible", timeout=15000)
        check(page.locator("#ds-title").inner_text() == "1024 aes",
              "space-in-name dataset opens (URL-encoded route)")
        check("201" in page.locator("#ds-stats").inner_text(),
              "real item count in stat chips")

        page.wait_for_selector("#items-grid .ds-item", timeout=15000)
        n_items = page.locator("#items-grid .ds-item").count()
        check(n_items == 201, f"all items render ({n_items})")
        # lazy thumbnails only fetch near the viewport -- bring row one in
        page.locator("#items-grid .ds-item").first.scroll_into_view_if_needed()
        try:
            page.wait_for_function(
                """() => { const i = document.querySelector('#items-grid .ds-thumb img');"""
                """ return !!i && i.complete && i.naturalWidth > 0; }""",
                timeout=15000)
            check(True, "preview image loads via /datasets/{name}/files/{path}")
        except Exception as exc:  # noqa: BLE001 -- report, don't kill the run
            check(False, f"preview image loads ({type(exc).__name__})")

        # -- M8f: item context menu (menus open, the option NEVER clicked --
        # this dataset is real, so any PUT would mutate user data)
        preview_puts: list[str] = []
        page.on("request", lambda r: preview_puts.append(r.url)
                if r.method == "PUT" and "/preview" in r.url else None)

        def open_item_menu(item_id: int):
            card = page.locator(
                f'#items-grid .ds-item[data-item-id="{item_id}"]'
            )
            card.locator(".ds-item-menu").click()
            page.wait_for_selector("#item-menu", state="visible", timeout=5000)
            return page.locator("#item-menu [role='menuitem']")

        detail_json = page.evaluate(
            "async () => (await fetch('/api/v1/datasets/1024%20aes')).json()")
        items_json = page.evaluate(
            "async () => (await fetch('/api/v1/datasets/1024%20aes/items')).json()")
        current = detail_json.get("preview_path")
        with_preview = [i for i in items_json["items"] if i.get("preview_path")]
        current_item = next(
            (i for i in with_preview if i["preview_path"] == current), None)
        other_item = next(
            (i for i in with_preview if i["preview_path"] != current), None)
        check(current is not None and current_item is not None and other_item is not None,
              "detail exposes a resolved preview with honest candidates "
              f"({current!r})")

        trigger = page.locator("#items-grid .ds-item .ds-item-menu").first
        check(trigger.is_visible(), "item ⋮ trigger renders on the thumb")

        if current_item is not None:
            opts = open_item_menu(current_item["id"])
            check(opts.count() == 1
                  and opts.inner_text().strip() == "Set as dataset preview",
                  "context menu offers exactly one option")
            check(opts.is_disabled(),
                  "the item already fronting the card is honestly disabled")
            page.screenshot(path=str(OUT / "datasets_menu.png"))
            check((OUT / "datasets_menu.png").stat().st_size > 15000,
                  "context-menu screenshot captured")
            page.keyboard.press("Escape")
            check(page.locator("#item-menu").is_hidden(),
                  "Escape closes the menu")

        if other_item is not None:
            opts = open_item_menu(other_item["id"])
            check(not opts.is_disabled(),
                  "another item's option renders enabled (not clicked)")
            page.locator("#ds-stats").click()  # outside click
            check(page.locator("#item-menu").is_hidden(),
                  "outside click closes the menu")

        check(preview_puts == [],
              "no PUT /preview fired -- the real dataset stays read-only")

        # pending is honestly empty on a fully curated dataset
        page.click(".seg[data-filter='pending']")
        page.wait_for_function(
            "n => document.querySelectorAll('#items-grid .ds-item').length !== n",
            arg=n_items, timeout=10000)
        check(page.locator("#items-state").is_visible(),
              "pending filter shows the honest empty state")
        page.click(".seg[data-filter='used']")
        page.wait_for_function(
            "n => document.querySelectorAll('#items-grid .ds-item').length === n",
            arg=n_items, timeout=10000)
        check(True, "used filter brings the items back")

        page.click("#tab-sets")
        check(page.locator(".ds-set").count() >= 1, "training sets list rows")
        page.click("#tab-items")
        # viewport shot: 201 cards would be a 17k-pixel full-page image
        page.screenshot(path=str(OUT / "datasets.png"))
        check((OUT / "datasets.png").stat().st_size > 20000,
              "datasets screenshot captured")

        # cleanup: delete the throwaway (confirmation dialog auto-accepted)
        page.click("#btn-back")
        page.wait_for_selector("#ds-grid", state="visible", timeout=15000)
        card = page.locator(".ds-card").filter(has_text="m8c-smoke-ds")
        empty_thumb = card.locator(".ds-card-thumb")
        check(empty_thumb.locator("img").count() == 0
              and empty_thumb.locator(".no-preview").is_visible(),
              "empty dataset card shows the honest NO PREVIEW thumb")
        card.locator("button", has_text="Delete").click()
        page.wait_for_function(
            "n => document.querySelectorAll('.ds-card').length === n",
            arg=n0, timeout=10000)
        check(page.locator(".ds-card").filter(has_text="m8c-smoke-ds").count() == 0,
              "throwaway dataset deleted (cleanup)")
        check(len(dialogs) == 1 and "m8c-smoke-ds" in dialogs[0],
              "delete asked for confirmation first")
        ctx.close()

        # ---------- G: shell -- rail + floating console + help/settings ----
        print("== G: shell (rail, console, help, settings) ==")
        ctx = browser.new_context(viewport={"width": 1600, "height": 1000})
        page = ctx.new_page()
        hook_console(page, "G")

        page.goto(f"{BASE}/", wait_until="networkidle")

        # rail: inventory, honest disabled slot, active state, hover tips
        check(page.locator(".rail").is_visible(), "icon rail visible")
        n_rail = page.locator(".rail-item").count()
        check(n_rail == 6, f"rail has 6 items (5 main + settings) ({n_rail})")
        soon = page.locator(".rail-item.rail-soon")
        check(soon.count() == 1 and soon.get_attribute("aria-disabled") == "true",
              "workflows slot present and honestly disabled")
        check(page.locator("a.rail-item[href='/']").get_attribute("aria-current")
              == "page", "tracker marked active on /")
        page.hover("a.rail-item[href='/graph']")
        # is_visible() ignores opacity -- assert the tip actually faded in
        try:
            page.wait_for_function(
                """() => { const t = document.querySelector(
                    "a.rail-item[href='/graph'] .rail-tip");"""
                """ return !!t && parseFloat(getComputedStyle(t).opacity) > 0.9; }""",
                timeout=3000)
            check(True, "rail tooltip actually shown on hover (opacity > 0.9)")
        except Exception as exc:  # noqa: BLE001 -- report, don't kill the run
            check(False, f"rail tooltip shown on hover ({type(exc).__name__})")
        check(page.locator(".rail-logo").get_attribute("href") == "/",
              "logo links home")

        # floating console: mounted, resizable, minimizes to a FAB, persists
        check(page.locator("#fconsole").is_visible(), "floating console visible")
        check(page.locator("#console-output .console-line").first.is_visible(),
              "console seeded with its first line")
        resize_mode = page.locator("#fconsole").evaluate(
            "e => getComputedStyle(e).resize")
        check(resize_mode == "both", f"console is natively resizable ({resize_mode})")
        page.click("#fconsole-min")
        check(page.locator("#fconsole").is_hidden(), "console minimizes")
        check(page.locator("#fconsole-fab").is_visible(),
              "FAB appears in the bottom-right corner")
        page.reload(wait_until="networkidle")
        check(page.locator("#fconsole-fab").is_visible(),
              "minimized state persists across a reload")
        page.click("#fconsole-fab")
        check(page.locator("#fconsole").is_visible(),
              "clicking the FAB restores the console")

        # rail navigation: tracker -> datasets, active state follows
        page.click("a.rail-item[href='/datasets']")
        page.wait_for_selector("#ds-grid", state="visible", timeout=15000)
        check(page.url.rstrip("/") == f"{BASE}/datasets",
              f"rail link navigates to /datasets ({page.url})")
        check(page.locator("a.rail-item[href='/datasets']").get_attribute(
            "aria-current") == "page", "destination rail item marked active")
        # the shell is global: console still mounted away from home
        check(page.locator("#fconsole").is_visible(),
              "console follows you to another page")

        # help: structured placeholder, honest stubs, one factual block
        page.click("a.rail-item[href='/help']")
        page.wait_for_selector("#help-where", state="visible", timeout=15000)
        check(page.locator(".page-topbar h1").inner_text() == "Help",
              "help page title")
        n_help = page.locator(".help-card").count()
        check(n_help == 6, f"six stub sections on help ({n_help})")
        check(page.locator(".help-todo").count() == n_help,
              "every stub honestly marked 'To be written.'")
        check(page.locator(".help-map-row").count() == 6,
              "where-things-live lists six destinations")
        page.screenshot(path=str(OUT / "help.png"), full_page=True)
        check((OUT / "help.png").stat().st_size > 20000,
              "help screenshot captured")

        # settings: design theme first, light honestly disabled
        page.click("a.rail-item[href='/settings']")
        page.wait_for_selector("#theme-group", state="visible", timeout=15000)
        check(page.locator(".page-topbar h1").inner_text() == "Settings",
              "settings page title")
        check(page.locator("input[name='theme'][value='dark']").is_checked(),
              "design theme: dark is the current selection")
        check(page.locator("input[name='theme'][value='light']").is_disabled(),
              "light honestly disabled (planned)")
        check(page.locator(".settings-link[href='/config']").count() == 1,
              "settings links the config editor (kept reachable)")
        page.screenshot(path=str(OUT / "settings.png"), full_page=True)
        check((OUT / "settings.png").stat().st_size > 15000,
              "settings screenshot captured")

        # the money shot: rail + floating console over the main page
        page.goto(f"{BASE}/", wait_until="networkidle")
        page.wait_for_selector("#hero-idle", state="visible", timeout=15000)
        page.screenshot(path=str(OUT / "shell.png"), full_page=True)
        check((OUT / "shell.png").stat().st_size > 20000,
              "shell screenshot captured")
        ctx.close()

        # ---------- H: dataset add-data dialog + edit modes (M8e) ------
        print("== H: add data dialog + item edit modes ==")
        ctx = browser.new_context(viewport={"width": 1600, "height": 1000})
        page = ctx.new_page()
        hook_console(page, "H")
        confirm_msgs: list[str] = []
        page.on("dialog", lambda d: (confirm_msgs.append(d.message), d.accept()))
        task_posts: list[str] = []
        page.on("request", lambda r: task_posts.append(r.url)
                if r.method == "POST" and "/tasks" in r.url else None)

        page.goto(f"{BASE}/datasets", wait_until="networkidle")
        page.wait_for_selector("#ds-grid .ds-card", timeout=15000)

        # card entry point -> dialog, generate tab by default
        card = page.locator(".ds-card").filter(has_text="1024 aes")
        card.locator("button", has_text="Add data").click()
        page.wait_for_selector("#add-dialog", state="visible", timeout=5000)
        check(page.locator("#add-ds-name").inner_text() == "1024 aes",
              "add dialog targets the card's dataset")
        check(page.locator("#add-panel-generate").is_visible()
              and page.locator("#add-panel-import").is_hidden(),
              "generate panel is the default tab")
        n_gen = page.locator(
            "#add-prompts, #add-cfg-min, #add-steps-min, #add-t-mode,"
            " #add-batch, #add-conditions, #add-samples, #add-latent,"
            " #add-model-type").count()
        check(n_gen == 9, f"generate form exposes its option set ({n_gen})")
        check("10 images" in page.locator("#add-total").inner_text(),
              "total preview computed from conditions x samples")
        check("512" in page.locator("#add-latent-px").inner_text(),
              "latent size renders its pixel equivalent")

        # client-side validation refuses before any request leaves
        page.click("#btn-add-start")
        check(page.locator("#add-error").is_visible()
              and page.locator("#add-dialog").is_visible(),
              "empty checkpoint refused locally (dialog stays open)")
        check(len(task_posts) == 0, "no task POST left the browser")

        # import tab: the resizing option set
        page.click("#add-tab-import")
        check(page.locator("#add-panel-import").is_visible()
              and page.locator("#add-panel-generate").is_hidden(),
              "import panel swaps in")
        check(len(page.locator("#add-resize-desc").inner_text()) > 20,
              "resize mode carries an honest description")
        check(page.locator("#add-max-aspect-row").is_hidden(),
              "max aspect ratio hidden when it does not apply")
        page.select_option("#add-resize-mode", "fit")
        check(page.locator("#add-max-aspect-row").is_visible(),
              "max aspect ratio shown for the splitting mode")
        n_imp = page.locator(
            "#add-image-dir, #add-recursive, #add-resize-mode,"
            " #add-import-latent, #add-import-model-type, #add-import-neg,"
            " #add-import-seed").count()
        check(n_imp == 7, f"import form exposes its option set ({n_imp})")
        page.click("#add-close")
        check(page.locator("#add-dialog").is_hidden(), "add dialog closes")

        # detail: edit mode + advanced editor (read-only: nothing saved)
        page.click("a.ds-card-name[href='/datasets/1024%20aes']")
        page.wait_for_selector("#items-grid .ds-item", timeout=15000)
        page.click(".seg[data-mode='edit']")
        check(page.locator("#items-grid").evaluate(
            "e => e.classList.contains('edit-mode')"),
            "edit mode marks the grid")
        check(page.locator(".item-edit-hint").first.is_visible(),
              "cards show the edit affordance")

        page.locator("#items-grid .ds-item .ds-item-body").first.click()
        page.wait_for_selector("#item-dialog", state="visible", timeout=5000)
        item_id = page.locator("#item-ed-id").inner_text()
        check(item_id.startswith("#"), f"editor bound to an item ({item_id})")
        check(page.locator("#item-ed-save").is_disabled(),
              "pristine editor refuses an empty save")
        check("item" in page.locator("#item-ed-meta").inner_text(),
              "editor shows read-only metadata")
        pos = page.locator("#item-ed-pos").inner_text()
        check("of" in pos, f"editor positions the walk ({pos})")

        page.fill("#item-ed-prompt", "smoke edit -- must be reverted")
        check(page.locator("#item-ed-save").is_enabled(),
              "edited field marks the editor dirty")
        page.click("#item-ed-revert")
        check(page.locator("#item-ed-save").is_disabled(),
              "revert restores the snapshot")
        page.click("#item-ed-next")
        check(page.locator("#item-ed-id").inner_text() != item_id,
              "next moves to the following item")
        page.click("#item-ed-close")
        check(page.locator("#item-dialog").is_hidden(),
              "editor closes from the walk")

        # browse mode: the prompt reaches the same editor
        page.click(".seg[data-mode='browse']")
        page.locator("#items-grid .item-prompt").first.click()
        page.wait_for_selector("#item-dialog", state="visible", timeout=5000)
        check(page.locator("#item-ed-save").is_disabled(),
              "prompt click opens the same pristine editor")
        page.click("#item-ed-close")
        check(page.locator("#item-dialog").is_hidden(),
              "editor closes from a prompt click")

        # browse mode exposes NO selection UI: checkboxes used to be
        # clickable in browse and raised multi-edit while the toolbar
        # still said "browse"
        check(page.locator("#items-grid .ds-item-check").count() == 0,
              "browse mode renders no item checkboxes")
        check(page.locator("#select-all-label").is_hidden(),
              "browse mode hides select-all")

        # multi-select -> multi-edit panel (edit mode; never applied here)
        page.click(".seg[data-mode='edit']")
        check(page.locator("#items-grid .ds-item-check").count() > 0,
              "edit mode brings the checkboxes back")
        page.locator("#items-grid .ds-item-check").nth(0).check()
        page.locator("#items-grid .ds-item-check").nth(1).check()
        check(page.locator("#bulk-bar").is_visible(),
              "multi-edit panel rises with a selection")
        n_bulk = page.locator("#bulk-field .seg").count()
        check(n_bulk == 4, f"multi-edit offers four fields ({n_bulk})")
        check("2 selected" in page.locator("#bulk-count").inner_text(),
              "selection count follows the checkboxes")
        page.click(".seg[data-field='cfg']")
        check(page.locator("#bulk-row-cfg").is_visible()
              and page.locator("#bulk-row-text").is_hidden(),
              "cfg value row swaps in")
        page.click(".seg[data-field='type']")
        check(page.locator("#bulk-row-type").is_visible(),
              "verdict row swaps in")
        check("2" in page.locator("#bulk-apply-n").inner_text(),
              "apply button counts the selection")
        check(task_posts == [], "still no task POST (read-only smoke)")

        # browse mid-selection: the panel hides, the selection survives
        page.click(".seg[data-mode='browse']")
        check(page.locator("#bulk-bar").is_hidden(),
              "browse hides multi-edit mid-selection")
        check(page.locator("#items-grid .ds-item-check").count() == 0,
              "browse still renders no checkboxes")
        page.click(".seg[data-mode='edit']")
        check(page.locator("#bulk-bar").is_visible()
              and "2 selected" in page.locator("#bulk-count").inner_text(),
              "selection survives the mode round-trip")

        page.locator("#bulk-bar").scroll_into_view_if_needed()
        page.screenshot(path=str(OUT / "datasets_edit.png"))
        check((OUT / "datasets_edit.png").stat().st_size > 20000,
              "edit-mode screenshot captured")
        page.click("#btn-clear-sel")
        check(page.locator("#bulk-bar").is_hidden(),
              "clearing the selection hides the panel")
        check(confirm_msgs == [],
              "no stray confirmation asked on the read-only walk")
        ctx.close()

        browser.close()

    unexpected = [n for n in console_noise if not is_expected_noise(n)]
    print("\nconsole (unexpected):")
    for n in unexpected:
        print("   ", n)
    check(not unexpected, "console clean across all pages")

    if failures:
        print(f"\nFAILED: {len(failures)} check(s)")
        for f in failures:
            print("  -", f)
        sys.exit(1)
    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    main()
