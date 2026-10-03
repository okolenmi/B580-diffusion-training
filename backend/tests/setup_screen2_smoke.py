"""Screen 2 in a real browser, against a real server.

Run against a scratch tree with NO ComfyUI configured -- which is the
machine the wizard exists for, and the one that catches the interesting
case. Two things are checked that a happy-path screenshot would not:

  1. With nothing configured, the conflict check must REFUSE and say it
     could not check. On this machine it renders "Could not check" and
     offers the separate environment. A wizard that rendered "no
     conflicts" here would be inviting an install into a directory it has
     not found -- which is the one wrong answer available on that screen.

  2. Once comfy_dir points at a real ComfyUI, the same panel must render
     the real numbers: 185 packages, 150 undeclared, all pinned exactly.

Both are driven through the UI where possible, and the API where the UI
has no control (there is no comfy_dir field on screen 2 -- it belongs to
screen 3, and screen 2 reads what screen 1/3 already resolved).
"""
import os
import sys

from playwright.sync_api import sync_playwright

BASE = os.environ.get("BASE", "http://127.0.0.1:8767")
OUT = os.environ.get("SMOKE_OUT", "/tmp/opencode/s2shots")
COMFY = os.environ.get("COMFY", "/home/okolenmi/comfy/ComfyUI")

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
          "screen 1 rendered on an unconfigured machine")

    page.click("text=Continue")
    page.wait_for_selector(".setup-options", timeout=30000)
    page.wait_for_selector("text=B580", timeout=30000)

    body = page.inner_text("body")
    check("2.5 GB" in body,
          f"the new-venv option is sized in GB, which is the point of the "
          f"screen ({[w for w in body.split() if 'GB' in w][:3]})")
    # toLocaleString is locale-dependent and this browser renders 12216
    # with a non-breaking space, not a comma. Normalised rather than matched
    # literally: the assertion is that the machine's own number is shown,
    # not that this browser chose a particular separator.
    flattened = body.replace(" ", " ").replace(" ", " ")
    check("12 216 MB" in flattened or "12,216 MB" in flattened,
          f"the real card's VRAM is shown, read from the machine "
          f"({[w for w in flattened.split() if 'MB' in w][:3]})")
    check("Not supported yet" in body,
          "CUDA is below a divider, under its own heading")

    radios = page.locator("input[type=radio]")
    names = [radios.nth(i).get_attribute("name") for i in range(radios.count())]
    values = [radios.nth(i).get_attribute("value") for i in range(radios.count())]
    check("cuda" not in [v.lower() for v in values if v],
          f"CUDA is never a selectable value ({values}) -- a disabled radio "
          f"beside a live one is an invitation to install 3 GB of unusable torch")
    check(all(v in ("new", "comfy") for v in values if v),
          f"and the only targets are the two real ones ({values})")

    page.screenshot(path=f"{OUT}/2-target.png", full_page=True)

    # ---- the refusal, on a machine with no ComfyUI configured ----------
    page.click("#setup-target-comfy")
    page.wait_for_selector(".setup-conflict", timeout=40000)
    panel = page.inner_text(".setup-conflict")
    print(f"    unconfigured panel: {panel[:120]!r}")
    check("Could not check" in panel,
          f"with no ComfyUI path given, the panel refuses rather than "
          f"reporting no conflicts ({panel[:60]!r})")
    check("cannot be shown to be safe" in panel,
          "and the refusal says an install cannot be shown to be safe")
    check("separate environment" in panel,
          "and points at the other option, which is the only way forward")
    check(page.locator(".setup-conflict.is-refused").count() == 1,
          "rendered with the refusal styling, not the passing styling")
    page.screenshot(path=f"{OUT}/3-unchecked.png", full_page=True)

    # ---- the same panel, with a real ComfyUI path typed in --------------
    # Typed on screen 2 rather than configured through the API. Setting
    # comfy_dir in settings makes the server report `configured: true` and
    # the wizard steps aside entirely -- which is how the original version
    # of this screen came to offer an option that could never be selected
    # on a first-run machine.
    page.fill("#setup-comfy-dir", COMFY)
    page.dispatch_event("#setup-comfy-dir", "change")
    page.wait_for_timeout(3000)
    real = page.inner_text(".setup-conflict")
    print(f"    typed-path panel: {real[:150]!r}")
    check("No conflicts" in real,
          "with a real ComfyUI path the panel reports the measured result")
    # 185 is ComfyUI's *venv*, not this server's. The count is the evidence
    # that the right interpreter was read: the scratch server's own
    # interpreter has 87, and reading that instead was the bug this
    # assertion exists to keep fixed.
    check("185" in real,
          f"including the real package count read from ComfyUI's venv, not "
          f"the server's own ({real.split(chr(10))[0]!r})")
    check("150" in real,
          "and how many were installed but undeclared -- the number that "
          "justifies pinning everything")
    check("pinned" in real,
          "and that every package is pinned to its exact installed version")
    check(page.locator(".setup-conflict.is-safe").count() == 1,
          "and with the passing styling, so the two are distinguishable")
    page.screenshot(path=f"{OUT}/4-comfy-ok.png", full_page=True)

    # ---- screen 3 carries the path forward, not asking for it again -----
    page.click("text=Continue")
    page.wait_for_selector("#setup-models", timeout=30000)
    screen3 = page.inner_text("body")
    check(COMFY in screen3,
          "screen 3 shows the ComfyUI path given on screen 2, rather than "
          "asking for it a second time")
    check(page.locator("#setup-comfy").count() == 0,
          "and has no second ComfyUI field that could disagree with it")
    page.screenshot(path=f"{OUT}/5-paths.png", full_page=True)
    page.click("text=Back")
    page.wait_for_selector(".setup-options", timeout=30000)
    check(page.locator("#setup-target-comfy").is_checked(),
          "going Back returns to screen 2 with the choice still made")

    # ---- back to the new venv, and the panel must go away ---------------
    page.click("#setup-target-new")
    page.wait_for_timeout(800)
    check(page.locator(".setup-conflict").count() == 0,
          "switching back to a separate venv removes the ComfyUI panel -- it "
          "is about that option, not about the machine")

    check(not errors, f"no console errors ({errors[:3]})")
    browser.close()

print()
if fails:
    print(f"{len(fails)} FAILED: {fails}")
    sys.exit(1)
print("SCREEN 2: all checks passed")