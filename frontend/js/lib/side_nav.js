// @ts-check
/* ---------------------------------------------------------------------------
   lib/side_nav.js -- a side menu of anchors that follows the scroll position (Settings and Help share it).
   --------------------------------------------------------------------------- */

/**
 * Mark the link of the section nearest the top of the view as current; a section the URL names is current at once.
 * @param {HTMLElement} nav  holds `<a href="#id">` links
 * @param {string[]} ids     the section ids, in page order
 */
export function followScroll(nav, ids) {
  const links = new Map(ids.map((id) => [id, nav.querySelector(`a[href="#${id}"]`)]));
  const mark = (id) => links.forEach((link, key) => link && link.classList.toggle("is-current", key === id));
  const fromUrl = (window.location.hash || "").slice(1);
  mark(links.has(fromUrl) ? fromUrl : ids[0]);
  const visible = new Set();
  const observer = new IntersectionObserver((entries) => {
    for (const entry of entries) (entry.isIntersecting ? visible.add(entry.target.id) : visible.delete(entry.target.id));
    const first = ids.find((id) => visible.has(id));
    if (first) mark(first);
  }, { rootMargin: "0px 0px -65% 0px" });
  ids.forEach((id) => { const node = document.getElementById(id); if (node) observer.observe(node); });
}
