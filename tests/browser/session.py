"""One headless Chromium for a browser check: pages that remember every script error, failed request and 5xx answer they saw."""

from __future__ import annotations

from playwright.sync_api import Page, sync_playwright

IGNORED_CONSOLE = ("Failed to load resource: the server responded with a status of 404",)  # a missing favicon or probe is not a page fault


class WatchedPage:
    """A Playwright page plus what went wrong on it. ``problems`` lists page errors, console errors and failed or 5xx requests."""

    def __init__(self, page: Page, base: str):
        self.page, self.base, self.problems = page, base, []
        page.on("pageerror", lambda e: self.problems.append(f"script error: {str(e)[:240]}"))
        page.on("console", lambda m: self._console(m))
        page.on("requestfailed", lambda r: self.problems.append(f"request failed: {r.url}") if "/stream" not in r.url and "/events" not in r.url else None)
        page.on("response", lambda r: self.problems.append(f"{r.status} on {r.url}") if r.status >= 500 else None)

    def _console(self, message) -> None:
        if message.type == "error" and not any(text in message.text for text in IGNORED_CONSOLE):
            self.problems.append(f"console error: {message.text[:240]}")

    def goto(self, path: str, settle_ms: int = 600) -> None:
        self.page.goto(self.base + path)
        self.page.wait_for_load_state("networkidle")
        self.page.wait_for_timeout(settle_ms)

    def text(self, selector: str) -> str:
        return self.page.inner_text(selector)

    def visible(self, selector: str) -> bool:
        return self.page.locator(selector).first.is_visible() if self.page.locator(selector).count() else False

    def wait_text(self, selector: str, fragment: str, timeout: float = 15000) -> bool:
        """True once the element's text contains ``fragment`` (False after ``timeout`` ms)."""
        try:
            self.page.wait_for_function("([s, f]) => { const e = document.querySelector(s); return !!e && e.innerText.includes(f); }",
                                        arg=[selector, fragment], timeout=timeout)
            return True
        except Exception:  # noqa: BLE001 - a timeout is the answer "no"
            return False

    def says(self, fragment: str, timeout: float = 8000) -> bool:
        """True when some text on the page contains ``fragment`` within ``timeout`` ms (notes and toasts come and go)."""
        try:
            self.page.wait_for_function("(f) => document.body.textContent.includes(f)", arg=fragment, timeout=timeout)
            return True
        except Exception:  # noqa: BLE001
            return False


class Chromium:
    """``with Chromium(server.url) as browser:`` then ``browser.new_page()``."""

    def __init__(self, base: str, width: int = 1300, height: int = 850):
        self.base, self.size = base, {"width": width, "height": height}
        self._playwright = self._browser = None
        self.pages: list[WatchedPage] = []

    def __enter__(self) -> Chromium:
        self._playwright = sync_playwright().start()
        self._browser = self._playwright.chromium.launch()
        return self

    def new_page(self, **options) -> WatchedPage:
        context = self._browser.new_context(viewport=self.size, **options)
        page = WatchedPage(context.new_page(), self.base)
        self.pages.append(page)
        return page

    def problems(self) -> list[str]:
        return [p for page in self.pages for p in page.problems]

    def __exit__(self, *exc) -> None:
        self._browser.close()
        self._playwright.stop()
