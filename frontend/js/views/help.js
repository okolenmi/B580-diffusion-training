/* ---------------------------------------------------------------------------
   views/help.js -- Help: search, a menu that follows the scroll, and troubleshooting entries that open when linked to.

   Search hides the blocks (`.help-item`) that do not match and the sections left empty; the page stays plain HTML, so the text is
   readable without this script.
   --------------------------------------------------------------------------- */

import { matches } from "../lib/help_search.js";
import { followScroll } from "../lib/side_nav.js";

const search = document.getElementById("help-search");
const empty = document.getElementById("help-empty");
const nav = document.getElementById("help-nav");
const sections = [...document.querySelectorAll(".help-section")];

function filter() {
  const query = search.value;
  let shown = 0;
  for (const section of sections) {
    const items = [...section.querySelectorAll(".help-item")];
    const title = section.querySelector(".panel-title")?.textContent || "";
    let any = false;
    for (const item of items) {
      // The section's title counts as part of each block's text: "memory batch" finds the batch advice in the memory section.
      const keep = matches(`${title} ${item.textContent || ""}`, query);
      item.hidden = !keep;
      any ||= keep;
      // A search that lands inside a troubleshooting entry shows its answer.
      if (item.tagName === "DETAILS" && query.trim()) item.open = keep;
    }
    section.hidden = Boolean(query.trim()) && !any;
    if (!section.hidden) shown += 1;
  }
  empty.hidden = shown > 0;
  nav.querySelectorAll("a").forEach((link) => {
    const target = document.getElementById(link.getAttribute("href").slice(1));
    link.hidden = Boolean(target && target.hidden);
  });
}

/* A link such as /help#nodes-red opens that entry and scrolls to it. */
function openTarget() {
  const target = window.location.hash ? document.getElementById(window.location.hash.slice(1)) : null;
  if (!target) return;
  if (target.tagName === "DETAILS") target.open = true;
  target.scrollIntoView();
}

search.addEventListener("input", filter);
window.addEventListener("hashchange", openTarget);
followScroll(nav, [...nav.querySelectorAll("a")].map((a) => a.getAttribute("href").slice(1)));
openTarget();
