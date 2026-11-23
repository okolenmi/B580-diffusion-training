/* ---------------------------------------------------------------------------
   views/help.js -- Help: search that marks matches where they stand, a menu that follows the reading position, and
   troubleshooting entries that open when linked to.

   Search never removes text: blocks (`.help-item`) that match have their words marked, the others fade, entries that match open,
   and the menu shows which sections have matches. Enter jumps to the next match. The page stays plain HTML, so the text is readable
   without this script.
   --------------------------------------------------------------------------- */

import { matches, queryWords, splitHits } from "../lib/help_search.js";
import { followScroll } from "../lib/side_nav.js";

class HelpSearch {
  constructor(input, counter, root, nav) {
    this.input = input;
    this.counter = counter;
    this.root = root;
    this.nav = nav;
    this.sections = [...root.querySelectorAll(".help-section")];
    this.hits = [];
    this.at = -1;
    this.wasOpen = new Map(); // entries as the reader left them, restored when the search is cleared
    input.addEventListener("input", () => this.run());
    input.addEventListener("keydown", (e) => this.key(e));
  }

  run() {
    const query = this.input.value;
    const words = queryWords(query);
    this.clearMarks();
    this.hits = [];
    this.at = -1;
    for (const section of this.sections) {
      const title = section.querySelector(".panel-title")?.textContent || "";
      let any = false;
      for (const item of section.querySelectorAll(".help-item")) {
        // The section's title counts as part of each block's text: "memory batch" finds the batch advice in the memory section.
        const hit = words.length > 0 && matches(`${title} ${item.textContent || ""}`, query);
        item.classList.toggle("is-dim", words.length > 0 && !hit);
        if (hit) { this.mark(item, words); this.hits.push(item); any = true; }
        if (item.tagName === "DETAILS") this.openFor(item, words.length > 0, hit);
      }
      section.classList.toggle("is-dim", words.length > 0 && !any);
      const link = this.nav.querySelector(`a[href="#${section.id}"]`);
      link?.classList.toggle("has-match", any);
    }
    this.counter.textContent = words.length ? (this.hits.length ? `${this.hits.length} found · Enter for next` : "no match") : "";
    this.counter.classList.toggle("is-none", words.length > 0 && !this.hits.length);
  }

  /** An entry that matches is open; when the search ends each goes back to how the reader had it. */
  openFor(entry, searching, hit) {
    if (searching) {
      if (!this.wasOpen.has(entry)) this.wasOpen.set(entry, entry.open);
      entry.open = hit || this.wasOpen.get(entry);
    } else if (this.wasOpen.has(entry)) {
      entry.open = this.wasOpen.get(entry);
      this.wasOpen.delete(entry);
    }
  }

  mark(item, words) {
    const walker = document.createTreeWalker(item, NodeFilter.SHOW_TEXT);
    const nodes = [];
    for (let n = walker.nextNode(); n; n = walker.nextNode()) nodes.push(n);
    for (const node of nodes) {
      const runs = splitHits(node.nodeValue, words);
      if (!runs.some((r) => r.hit)) continue;
      const fragment = document.createDocumentFragment();
      for (const run of runs) {
        if (!run.hit) { fragment.append(run.text); continue; }
        const m = document.createElement("mark");
        m.className = "help-hit";
        m.textContent = run.text;
        fragment.append(m);
      }
      node.replaceWith(fragment);
    }
  }

  clearMarks() {
    for (const m of this.root.querySelectorAll("mark.help-hit")) m.replaceWith(document.createTextNode(m.textContent));
    this.root.normalize();
  }

  key(e) {
    if (e.key === "Escape") { this.input.value = ""; this.run(); return; }
    if (e.key !== "Enter" || !this.hits.length) return;
    e.preventDefault();
    this.at = (this.at + (e.shiftKey ? -1 : 1) + this.hits.length) % this.hits.length;
    const item = this.hits[this.at];
    this.root.querySelectorAll(".is-current-hit").forEach((n) => n.classList.remove("is-current-hit"));
    item.classList.add("is-current-hit");
    item.scrollIntoView({ block: "center", behavior: "smooth" });
  }
}

/* A link such as /help#nodes-red opens that entry and scrolls to it. */
function openTarget() {
  const target = window.location.hash ? document.getElementById(window.location.hash.slice(1)) : null;
  if (!target) return;
  if (target.tagName === "DETAILS") target.open = true;
  target.scrollIntoView();
}

const nav = document.getElementById("help-nav");
new HelpSearch(document.getElementById("help-search"), document.getElementById("help-count"), document.getElementById("help-root"), nav);
window.addEventListener("hashchange", openTarget);
followScroll(nav, [...nav.querySelectorAll("a")].map((a) => a.getAttribute("href").slice(1)));
openTarget();
