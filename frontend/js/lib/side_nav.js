// @ts-check
/* ---------------------------------------------------------------------------
   lib/side_nav.js -- a side menu of anchors that follows the scroll position (Settings and Help share it).

   The current entry is the section under a line a little above the middle of the visible part, so a section that only just shows at
   the edge is not preferred over the one being read. A click on an entry makes it current until the reader scrolls on their own.
   --------------------------------------------------------------------------- */

/** How far down the visible part the reading line is (0 top, 1 bottom). */
export const READING_LINE = 0.4;

/**
 * The section being read.
 * @param {{id: string, top: number, bottom: number}[]} boxes  the sections in page order, positions in the same units as `line`
 * @param {number} line      the reading line
 * @param {{atTop: boolean, atBottom: boolean}} edge  whether the scroll is at its start or its end (the first / last section then)
 * @returns {string|null}
 */
export function sectionAt(boxes, line, edge) {
  if (!boxes.length) return null;
  if (edge.atTop) return boxes[0].id;
  if (edge.atBottom) return boxes[boxes.length - 1].id;
  const under = boxes.find((b) => b.top <= line && line < b.bottom);
  if (under) return under.id;
  let best = boxes[0];
  let gap = Infinity;
  for (const b of boxes) {
    const d = line < b.top ? b.top - line : line - b.bottom;
    if (d < gap) { best = b; gap = d; }
  }
  return best.id;
}

/** The nearest ancestor that scrolls on its own, or null when the page itself does. */
function scroller(/** @type {HTMLElement} */ node) {
  for (let n = node.parentElement; n; n = n.parentElement) {
    const overflow = getComputedStyle(n).overflowY;
    if ((overflow === "auto" || overflow === "scroll") && n.scrollHeight > n.clientHeight) return n;
  }
  return null;
}

/**
 * @param {HTMLElement} nav  holds `<a href="#id">` links
 * @param {string[]} ids     the section ids, in page order
 */
export function followScroll(nav, ids) {
  const links = new Map(ids.map((id) => [id, nav.querySelector(`a[href="#${id}"]`)]));
  const mark = (/** @type {string|null} */ id) => links.forEach((link, key) => link && link.classList.toggle("is-current", key === id));
  const fromUrl = (window.location.hash || "").slice(1);
  let pinned = links.has(fromUrl) ? fromUrl : null;
  mark(pinned || ids[0]);

  const update = () => {
    if (pinned) return;
    const nodes = ids.map((id) => document.getElementById(id)).filter((n) => n && !n.hidden);
    if (!nodes.length) return;
    const host = scroller(/** @type {HTMLElement} */ (nodes[0]));
    const view = host ? host.getBoundingClientRect() : { top: 0, bottom: window.innerHeight };
    const top = host ? host.scrollTop : window.scrollY;
    const room = host ? host.scrollHeight - host.clientHeight : document.documentElement.scrollHeight - window.innerHeight;
    const boxes = nodes.map((n) => { const r = /** @type {HTMLElement} */ (n).getBoundingClientRect(); return { id: n.id, top: r.top, bottom: r.bottom }; });
    const line = view.top + (view.bottom - view.top) * READING_LINE;
    mark(sectionAt(boxes, line, { atTop: top <= 2, atBottom: room > 2 && top >= room - 2 }));
  };

  let frame = 0;
  const later = () => { if (!frame) frame = requestAnimationFrame(() => { frame = 0; update(); }); };
  window.addEventListener("scroll", later, { capture: true, passive: true });
  window.addEventListener("resize", later);
  // Scrolling by hand (not the jump a click makes) hands the menu back to the reading line.
  const release = () => { pinned = null; later(); };
  for (const type of ["wheel", "touchmove", "keydown"]) window.addEventListener(type, release, { passive: true });
  nav.addEventListener("click", (e) => {
    const link = /** @type {HTMLElement} */ (e.target).closest("a[href^='#']");
    const id = link && link.getAttribute("href").slice(1);
    if (id && links.has(id)) { pinned = id; mark(id); }
  });
  window.addEventListener("hashchange", () => {
    const id = window.location.hash.slice(1);
    if (links.has(id)) { pinned = id; mark(id); }
  });
  later();
}
