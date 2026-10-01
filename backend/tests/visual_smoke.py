"""Training page visual smoke: idle state, mocked running state,
interactions, and a regression pass over monitor/graph (they share
style.css with the training page).

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
    # Browser network-log line for the contract 404 when idle.
    return "404" in text and "runs/active" in text


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
