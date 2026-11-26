"""Training through the apps page, end to end (TESTING 96.1, 99, 100, 101): the SDXL LoRA starter on tiny CPU stand-ins runs in a real graph child
process under a real server, driven from a real page. Status line, progress, files, Save now, history and Run again, queue, Stop, and the host
memory ceiling. No model and no card are needed.

    python -m tests.browser.check_runs
"""

from __future__ import annotations

import re
import time
from pathlib import Path

from tests.support import check
from . import fixtures
from .actions import back_to_app, card, card_text, create_app, open_app_graph, open_tab, wait_card
from .server import ApiError, TestServer
from .session import Chromium


def training_app(server, browser, name: str, **graph_options):
    """An app made from a saved copy of the tiny LoRA graph, opened on its Run tab."""
    dataset = fixtures.toy_dataset(server.home)
    server.api("PUT", f"/api/v1/graphs/library/{name}", fixtures.lora_graph(server.home, dataset, out=f"out-{name}", **graph_options))
    page = browser.new_page()
    page.goto("/apps")
    create_app(page, name, start=f"A copy of {name} (15 nodes)")
    open_app_graph(page, name)
    back_to_app(page, name)
    return page


def saved_files(server, app: str) -> list[Path]:
    """The adapter files an app's saver has written (every app writes to a folder of its own)."""
    return sorted((server.home / f"out-{app}").rglob("toy_step*.safetensors"))


def executions(server) -> list[dict]:
    answer = server.api("GET", "/api/v1/graphs/executions")
    return answer.get("executions", answer) if isinstance(answer, dict) else answer


def stop_all(server) -> None:
    """Whatever a failed part left running or waiting must not block the next part."""
    for e in executions(server):
        if e.get("status") in ("running", "planned", "queued"):
            try:
                server.api("POST", f"/api/v1/graphs/executions/{e['execution_id']}/stop")
            except ApiError:
                pass
    idle(server)


def idle(server, timeout: float = 90.0) -> None:
    server.wait_for("no active run", lambda: not [e for e in executions(server) if e.get("status") in ("running", "planned", "queued", "stopping")], timeout)


def status_and_files(server, browser):
    page = training_app(server, browser, "short", steps=6, save_every=3)
    check(not page.page.locator("#btn-run").is_disabled(), "the copied training graph can run")
    page.page.click("#btn-run")
    check(wait_card(page, "Run status", "FINISHED", 90), "Run status reaches FINISHED")
    check("Finished at step 6 of 6" in card_text(page, "Status"), f"the trainer's Status line ends 'Finished at step 6 of 6' ({card_text(page, 'Status')[:60]!r})")
    check("6 / 6" in card_text(page, "Progress"), "Progress shows 6 / 6")
    check("steps done" in card_text(page, "Result") and "last loss" in card_text(page, "Result"), "Result keeps the step count and the last loss")
    files = saved_files(server, "short")
    check(len(files) >= 2 and all(len(f.stem.rsplit("_step", 1)[1]) == 6 for f in files), f"adapter files carry the six-digit step ({[f.name for f in files]})")
    check(files[-1].name in card_text(page, "Last file") and "files written" in card_text(page, "Saved"), "Last file and Saved name the newest adapter and count the files")
    return page


def charts_after_a_run(server, browser):
    """Every chart on the Charts tab has a sane x range: seconds since the run began, never an epoch or a negative number."""
    page = browser.new_page()
    page.goto("/apps/short")
    page.page.wait_for_selector("#btn-run")
    page.page.wait_for_timeout(800)
    open_tab(page, "Charts")
    page.page.wait_for_timeout(1500)
    ranges = page.page.evaluate("() => [...document.querySelectorAll('#board article.bd-block')].map(a => [a.querySelector('.bd-title')?.textContent.trim(), (a.innerText.match(/\\bt (-?[\\d.]+)\\u2013(-?[\\d.]+)/) || [])[2]]).filter(r => r[1] !== undefined)")
    check(len(ranges) >= 1, f"a chart drawn against time is on the page ({[r[0] for r in ranges]})")
    check(all(0 <= float(r[1]) < 3600 for r in ranges), f"their x ranges are seconds of the run ({[(r[0], r[1]) for r in ranges]})")


def run_history_and_run_again(server, browser):
    page = browser.new_page()
    page.goto("/apps/short")
    page.page.wait_for_selector("#btn-run")
    page.page.wait_for_timeout(800)
    seed = card(page, "Seed").locator("input").first
    original = seed.input_value()
    seed.fill("7")
    seed.dispatch_event("change")
    page.page.click("#btn-run")
    check(wait_card(page, "Run status", "#2", 90) and wait_card(page, "Run status", "FINISHED", 90), "a second run with another seed finishes")
    open_tab(page, "Runs")
    check(page.page.locator("button", has_text="Run again").count() >= 2, "Runs lists both runs, each with Run again")
    page.page.locator("button", has_text="Run again").last.click()
    open_tab(page, "Run")
    page.page.wait_for_timeout(500)
    check(card(page, "Seed").locator("input").first.input_value() == original, f"Run again on the first run puts its values back (seed {original})")
    check(not [e for e in executions(server) if e.get("status") == "running"], "and starts nothing")


def save_now_and_stop(server, browser):
    page = training_app(server, browser, "long", steps=400, save_every=100000)
    idle(server)
    page.page.wait_for_function("() => !document.querySelector('#btn-run').disabled", timeout=20000)
    page.page.click("#btn-run")
    check(wait_card(page, "Progress", " / 400", 60), "a long run is under way")
    page.page.wait_for_timeout(1500)
    before = {f.name for f in saved_files(server, "long")}
    button = card(page, "Save now").locator(".bd-action button")
    button.click()
    check(wait_card(page, "Save now", "Done", 30), "Save now goes Sent, then Done")
    server.wait_for("the adapter file", lambda: {f.name for f in saved_files(server, "long")} - before, 30)
    first = {f.name for f in saved_files(server, "long")}
    page.page.wait_for_timeout(2500)  # two presses inside one step would write the one file twice
    button.click()
    check(wait_card(page, "Save now", "Sent", 10) or wait_card(page, "Save now", "Done", 10), "pressed again it answers again")
    server.wait_for("the second adapter file", lambda: {f.name for f in saved_files(server, "long")} - first, 30)
    names = [f.name for f in saved_files(server, "long")]
    check(len(set(names)) >= 2 and all("_step" in n for n in names), f"two presses, two files at two steps, while the run goes on ({names})")
    check(wait_card(page, "Run status", "RUNNING", 5) or "step" in card_text(page, "Status"), "the run was still going")
    return page


def queue_and_stop(server, browser, page):
    """Continues on the running 'long' app: queue a second run, stop the first, the second starts by itself and the page follows it."""
    first_id = int(re.search(r"#(\d+)", card_text(page, "Run status")).group(1))
    page.page.click("#btn-queue")
    open_tab(page, "Runs")
    check(wait_card(page, "Waiting runs", "#", 15), f"Add to queue lists a waiting run ({card_text(page, 'Waiting runs')[:60]!r})")
    open_tab(page, "Run")
    check(page.page.locator("#btn-stop").is_visible(), "Stop is offered while the run goes")
    page.page.click("#btn-stop")
    check(server.wait_for("the first run to stop", lambda: [e for e in executions(server) if e["execution_id"] == first_id and e["status"] == "stopped"], 60), "Stop ends the run")
    check(wait_card(page, "Run status", f"#{first_id + 1} ", 60) and "RUNNING" in card_text(page, "Run status"), f"the waiting run starts by itself and the page follows it without a reload ({card_text(page, 'Run status')!r}; runs {[(e['execution_id'], e['status']) for e in executions(server)][:4]})")
    page.page.click("#btn-stop")
    idle(server)
    open_tab(page, "Runs")
    check("Nothing is waiting" in card_text(page, "Waiting runs"), "nothing is left waiting")
    kinds = [e.get("status") for e in executions(server)]
    check(kinds.count("stopped") >= 2, f"both stopped runs are in the history as stopped ({kinds})")


def host_memory_ceiling(server, browser):
    graph = server.api("GET", "/api/v1/apps/short")["doc"]["graph"]
    before = len(executions(server))
    tight = {**graph, "settings": {**graph.get("settings", {}), "memory.ram_max_mb": 300}}
    server.api("PUT", "/api/v1/apps/short/graph", {"graph": tight})
    page = browser.new_page()
    page.goto("/apps/short")
    page.page.wait_for_selector("#btn-run")
    page.page.wait_for_timeout(800)
    page.page.click("#btn-run")
    server.wait_for("the run to start", lambda: executions(server)[0].get("status") != "finished" or len(executions(server)) > before, 20)
    idle(server, 90)
    last = executions(server)[0]
    check(last.get("status") == "error" and "host memory" in str(last.get("error", "")), f"a ceiling below what the run holds ends it as an error ({last.get('status')}: {str(last.get('error'))[:90]})")
    huge = {**graph, "settings": {**graph.get("settings", {}), "memory.ram_max_mb": 10_000_000}}
    try:
        server.api("POST", "/api/v1/graphs/run", {"graph": huge, "mode": "parallel"})
        check(False, "a ceiling above the machine's RAM is refused at the start")
    except ApiError as error:
        check(error.status == 409 and "host memory" in str(error.body), f"a ceiling above the machine's RAM is refused at the start ({error.status})")
    server.api("PUT", "/api/v1/apps/short/graph", {"graph": graph})


def main():
    with TestServer() as server, Chromium(server.url) as browser:
        running = None
        for part in (status_and_files, charts_after_a_run, run_history_and_run_again, save_now_and_stop, host_memory_ceiling):
            print(f"\n{part.__name__}")
            try:
                running = part(server, browser) or running
                if part is save_now_and_stop:
                    print("\nqueue_and_stop")
                    queue_and_stop(server, browser, running)
            except Exception as error:  # noqa: BLE001
                check(False, f"{part.__name__} stopped: {type(error).__name__}: {str(error)[:300]}")
            if part is save_now_and_stop:
                stop_all(server)
        problems = [p for p in browser.problems() if "409" not in p]
        print()
        check(not problems, "no script errors or server errors on any page" + (f": {problems[:4]}" if problems else ""))


if __name__ == "__main__":
    from tests.support import finish
    main()
    finish()
