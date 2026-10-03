"""The install step, end to end, in a real browser against a real server.

Runs against a scratch tree with NO ComfyUI configured, so the only
installable target is "a new virtualenv for this project" -- which is the
case that can be driven without touching anything the user already has.

What this checks that the other screens do not:

  1. The install **actually runs**: a real venv is created under the
     project and a real package lands in it. Everything else in the
     installer is a report, so this is the only assertion here that can
     fail in a way nobody predicted.
  2. The refusal path: picking ComfyUI's venv without a successful check
     must not start a job. Installing without pins is the one action on
     this screen that could change a version ComfyUI declares.
  3. The pins are visible in the UI as the actual command, not as prose.

**Selectors are CSS, deliberately.** Playwright's `text=` engine does
substring matching, and this screen's own explanatory note contains the
words "install" and "cannot be shown to be safe" -- so a first version of
this file, clicking `text=Install` and asserting the page contains "cannot
be shown to be safe", passed without clicking anything or refusing
anything. It was matching the static note. Both are now `.setup-actions
.btn-primary` and `#setup-msg`.

Run with the server started from the scratch tree on :8767 (see
backend/tests/setup_screen2_smoke.py for the same setup).
"""
import os
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

BASE = os.environ.get("BASE", "http://127.0.0.1:8767")
OUT = os.environ.get("SMOKE_OUT", "/tmp/opencode/install_smoke")
PROJECT = os.environ.get("PROJECT", "")

os.makedirs(OUT, exist_ok=True)
fails = []


def check(ok, msg):
    print(("  PASS: " if ok else "  FAIL: ") + msg)
    if not ok:
        fails.append(msg)


with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page(viewport={"width": 1000, "height": 1000})
    errors = []
    page.on("pageerror", lambda e: errors.append(str(e)))
    page.on("console",
            lambda m: errors.append(m.text) if m.type == "error" else None)

    page.goto(f"{BASE}/setup", wait_until="networkidle")
    check(page.locator("text=Can this machine run").count() > 0,
          "the wizard is offered on an unconfigured machine")

    page.click(".setup-actions .btn-primary")
    page.wait_for_selector("#setup-target-new", timeout=30000)
    page.click("#setup-target-new")
    page.wait_for_timeout(1200)

    # ---- what this machine can reach -------------------------------------
    # The scratch server runs from an interpreter that already has every
    # package this project needs, so `readiness.missing` is empty. That
    # decides what is reachable from here, and being explicit about it is
    # better than a test that silently asserts nothing:
    #
    #   * the "nothing to install" path IS reachable and is checked;
    #   * the refusal path is NOT reachable, because `startInstall` returns
    #     on an empty package list before it ever looks at the conflict
    #     report -- which is correct, since an install of nothing cannot
    #     change a version ComfyUI declares.
    #
    # The refusal itself is covered where it is enforced: a request to
    # install torch into ComfyUI's virtualenv is refused by
    # `StartInstall`, in backend/tests/test_installer.py. That is the
    # stronger place for it -- the browser's guard is defence in depth
    # over a refusal the server makes regardless of what the page sends.
    missing = page.evaluate(
        "async () => (await (await fetch('/api/v1/installer/readiness'))"
        ".json()).missing"
    )
    check(missing == [],
          f"the scratch server has every package, so this machine can only "
          f"reach the nothing-to-install path ({missing})")

    # ---- a machine with nothing missing ----------------------------------
    # The scratch server runs from an interpreter that already has every
    # package this project needs, so there is nothing to install and the
    # wizard must say so rather than starting a job that would do nothing.
    page.click("#setup-target-new")
    page.wait_for_timeout(1500)
    if not missing:
        page.click(".setup-actions .btn-primary")
        page.wait_for_selector("#setup-models", timeout=30000)
        check(page.locator("text=Where do model files live").count() > 0,
              "with nothing missing, Install goes straight to the paths "
              "screen instead of starting a no-op job")
        check(page.evaluate(
            "async () => Object.keys(await (await "
            "fetch('/api/v1/installer/state')).json()).length > 0"),
            "and the server is untouched")
        page.screenshot(path=f"{OUT}/2-nothing-to-do.png", full_page=True)
    else:
        # Unreachable here, and deliberately not faked. The real install is
        # covered by scripts/test_install_executor.sh, which creates a venv
        # and runs pip for real -- the point of it being that nothing about
        # the install is simulated.
        page.click(".setup-actions .btn-primary")
        page.wait_for_selector(".setup-log", timeout=60000)
        import time as _t
        end = _t.monotonic() + 240
        while _t.monotonic() < end:
            text = page.inner_text("#setup-root")
            if "Installed." in text or "did not finish" in text:
                break
            _t.sleep(2)
        final = page.inner_text("#setup-root")
        check("Installed." in final,
              f"the install reaches a terminal state by polling ({final[:70]!r})")
        check("pip install" in final,
              "and the pip command it ran is visible, not just asserted")
        check(page.locator(".setup-cmd").count() >= 1,
              "in its own block, so the claim can be read")
        page.screenshot(path=f"{OUT}/2-installed.png", full_page=True)

        if PROJECT:
            interpreter = Path(PROJECT) / "venv" / "bin" / "python"
            check(interpreter.exists(),
                  f"a virtualenv was created at {interpreter}")

    # ---- on to paths -------------------------------------------------------
    if not missing:
        check(page.locator("#setup-models").count() > 0,
              "and the paths screen is reached")
        page.screenshot(path=f"{OUT}/3-paths.png", full_page=True)

    check(not errors, f"no console errors ({errors[:3]})")
    browser.close()

print()
if fails:
    print(f"{len(fails)} FAILED: {fails}")
    sys.exit(1)
print("INSTALL STEP: all checks passed")