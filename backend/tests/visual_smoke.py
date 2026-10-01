"""Visual smoke for the main pages: idle state, mocked running state,
interactions, the config editor (schema form + raw), run detail views,
and a regression pass over monitor/graph (they share style.css).

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
        check(page.locator(".page-topbar h1").inner_text() == "Training", "topbar title")
        check(page.locator(".sidebar-tool").is_visible(), "monitor tool group visible")
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
            page.screenshot(path=str(OUT / f"{name}.png"), full_page=True)
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

        # the M8c nav hand-off from the main page
        page.goto(f"{BASE}/", wait_until="networkidle")
        check(page.locator("a.nav-item[href='/datasets']").count() == 1,
              "sidebar links to /datasets")
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
        check(page.locator("#task-image-dir").is_visible(), "task form renders")
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
        try:
            page.wait_for_function(
                """() => { const i = document.querySelector('#items-grid .ds-thumb img');"""
                """ return !!i && i.complete && i.naturalWidth > 0; }""",
                timeout=15000)
            check(True, "preview image loads via /datasets/{name}/files/{path}")
        except Exception as exc:  # noqa: BLE001 -- report, don't kill the run
            check(False, f"preview image loads ({type(exc).__name__})")

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
        card.locator("button", has_text="Delete").click()
        page.wait_for_function(
            "n => document.querySelectorAll('.ds-card').length === n",
            arg=n0, timeout=10000)
        check(page.locator(".ds-card").filter(has_text="m8c-smoke-ds").count() == 0,
              "throwaway dataset deleted (cleanup)")
        check(len(dialogs) == 1 and "m8c-smoke-ds" in dialogs[0],
              "delete asked for confirmation first")
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
