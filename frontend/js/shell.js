/* ---------------------------------------------------------------------------
   shell.js -- mounts the application shell on EVERY page:
     * the icon rail (logo + Graph Editor / Dataset manager / Pre-built
       workflows (soon) / System tracker / Help, Settings pinned at the
       bottom) -- icon-only, names ride along as hover tips;
     * the floating system console: a single resizable + draggable window
       that follows you across tabs, minimizable into a FAB at the bottom
       right corner. Its geometry and minimized state persist in
       localStorage, so all tabs agree on it.

   Contract with pages:
     * load this module FIRST (before the page's view module), so
       #console-output exists by the time anything logs. Views keep
       writing into #console-output / .console-line -- unchanged.
     * body gets the `has-rail` class; layout offset is pure CSS
       (shell.css), pages never hardcode rail dimensions.
     * active rail item is derived from location.pathname -- no page
       passes anything.

   No imports: this file must run standalone on every page.
--------------------------------------------------------------------------- */

/* ---- icons (inline SVG: no font, no sprite fetch, no build step) -------- */

const svg = (body) =>
  `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${body}</svg>`;

const ICONS = {
  bolt: `<svg viewBox="0 0 24 24" fill="currentColor" aria-hidden="true"><path d="M13.2 2 5 13.6h5.1L9.4 22l8.6-11.8h-5.3L13.2 2Z"/></svg>`,
  graph: svg(`<circle cx="6" cy="6.5" r="2.4"/><circle cx="17.8" cy="7.5" r="2.4"/><circle cx="12" cy="17.8" r="2.4"/><path d="M8.3 7 10.4 15.6M16.2 9.4 13.6 16.1M8.4 6.7h7"/>`),
  layers: svg(`<path d="M12 3.2 3.4 7.4 12 11.6l8.6-4.2L12 3.2Z"/><path d="M3.4 12.1 12 16.3l8.6-4.2"/><path d="M3.4 16.7 12 20.9l8.6-4.2"/>`),
  template: svg(`<rect x="3.5" y="3.5" width="17" height="17" rx="2.5"/><path d="M3.5 9.2h17M9.6 9.2v11.3"/>`),
  pulse: svg(`<path d="M3 12h3.8l2.3-5.6 3.9 11.2 2.3-5.6H21"/>`),
  help: svg(`<circle cx="12" cy="12" r="8.6"/><path d="M9.7 9.6a2.4 2.4 0 1 1 3.5 2.1c-.8.5-1.2 1-1.2 1.9"/><path d="M12 16.8h.01" stroke-width="2.4"/>`),
  sliders: svg(`<path d="M4 7.5h9M19.5 7.5h.5M4 16.5h3.5M13.5 16.5H20"/><circle cx="16" cy="7.5" r="2.4"/><circle cx="10" cy="16.5" r="2.4"/>`),
  terminal: svg(`<rect x="3" y="4.6" width="18" height="14.8" rx="2.5"/><path d="m7.6 10.2 2.4 2.1-2.4 2.1M13 14.4h3.6"/>`),
};

/* ---- rail --------------------------------------------------------------- */

/* Order is the product spec (M8d): main tabs top, Settings at the end. */
const MAIN_ITEMS = [
  { id: "graph", href: "/graph", label: "Graph Editor", icon: ICONS.graph },
  { id: "datasets", href: "/datasets", label: "Dataset manager", icon: ICONS.layers },
  { id: "workflows", href: null, label: "Pre-built workflows", icon: ICONS.template },
  { id: "tracker", href: "/", label: "System tracker", icon: ICONS.pulse },
  { id: "help", href: "/help", label: "Help", icon: ICONS.help },
];
const BOTTOM_ITEMS = [
  { id: "settings", href: "/settings", label: "Settings", icon: ICONS.sliders },
];

function activeRailId() {
  const path = location.pathname;
  if (path === "/") return "tracker";
  if (path.startsWith("/graph")) return "graph";
  if (path.startsWith("/datasets")) return "datasets";
  if (path.startsWith("/help")) return "help";
  if (path.startsWith("/settings")) return "settings";
  return null; // sub-pages (run detail, monitor, config) have no rail entry
}

function railItem(item, activeId) {
  const active = item.id === activeId;
  const tip = `<span class="rail-tip">${item.label}</span>`;
  if (!item.href) {
    // future feature: present, inert, and honest when clicked
    return `<button class="rail-item rail-soon" data-rail="${item.id}" type="button"
              aria-disabled="true" aria-label="${item.label} (coming soon)">
              ${item.icon}${tip}</button>`;
  }
  const cls = active ? "rail-item active" : "rail-item";
  const cur = active ? ` aria-current="page"` : "";
  return `<a class="${cls}" data-rail="${item.id}" href="${item.href}"
            aria-label="${item.label}"${cur}>${item.icon}${tip}</a>`;
}

function mountRail() {
  const activeId = activeRailId();
  const main = MAIN_ITEMS.map((it) => railItem(it, activeId)).join("\n");
  const bottom = BOTTOM_ITEMS.map((it) => railItem(it, activeId)).join("\n");
  document.body.insertAdjacentHTML(
    "afterbegin",
    `<nav class="rail" aria-label="Main navigation">
       <a class="rail-logo" href="/" aria-label="Distillation -- home">${ICONS.bolt}</a>
       <div class="rail-group">${main}</div>
       <div class="rail-group rail-bottom">${bottom}</div>
     </nav>`,
  );
  document.body.classList.add("has-rail");

  document.querySelector(".rail-soon")?.addEventListener("click", () => {
    logShell("Pre-built workflows: coming soon (future feature).", "warn");
  });
}

/* ---- floating console ---------------------------------------------------- */

const STORE_KEY = "shell.console.v1";

function loadGeometry() {
  try {
    const raw = localStorage.getItem(STORE_KEY);
    if (raw) return JSON.parse(raw);
  } catch (err) {
    // Corrupted or unavailable storage: fall back to defaults. Benign,
    // but said once so a browser denying localStorage (private mode,
    // blocked cookies) is not indistinguishable from "no saved geometry".
    console.warn("shell: stored geometry unreadable, using defaults", err);
  }
  return null;
}

function saveGeometry(state) {
  try {
    localStorage.setItem(STORE_KEY, JSON.stringify(state));
  } catch (err) {
    // Private mode or a full quota: geometry is a convenience, not data,
    // so this must not interrupt anything. Logged because the symptom
    // otherwise appears later and elsewhere ("my layout never sticks")
    // with nothing pointing at the cause.
    console.warn("shell: could not persist geometry", err);
  }
}

function logShell(message, kind = "info") {
  const out = document.getElementById("console-output");
  if (!out) return;
  const line = document.createElement("div");
  line.className = `console-line ${kind}`;
  line.textContent = message;
  out.appendChild(line);
  out.scrollTop = out.scrollHeight;
}

function mountConsole() {
  document.body.insertAdjacentHTML(
    "beforeend",
    `<div class="fconsole" id="fconsole" role="log" aria-label="System console">
       <div class="fconsole-head" id="fconsole-head">
         <span class="fconsole-title">System console</span>
         <button class="fconsole-btn" id="fconsole-min" type="button"
                 aria-label="Minimize console">–</button>
       </div>
       <div class="console-lines" id="console-output">
         <div class="console-line info">Ready.</div>
       </div>
     </div>
     <button class="fconsole-fab" id="fconsole-fab" type="button"
             aria-label="Show system console" hidden>${ICONS.terminal}</button>`,
  );

  const win = document.getElementById("fconsole");
  const head = document.getElementById("fconsole-head");
  const minBtn = document.getElementById("fconsole-min");
  const fab = document.getElementById("fconsole-fab");

  const state = Object.assign(
    { w: 440, h: 250, x: null, y: null, min: false },
    loadGeometry() || {},
  );

  const clamp = () => {
    const vw = window.innerWidth;
    const vh = window.innerHeight;
    state.w = Math.min(Math.max(state.w, 260), Math.max(vw - 80, 260));
    state.h = Math.min(Math.max(state.h, 130), Math.max(vh - 60, 130));
    if (state.x === null || state.y === null) {
      // default: bottom-right, clear of the rail
      state.x = vw - state.w - 16;
      state.y = vh - state.h - 16;
    }
    state.x = Math.min(Math.max(state.x, 0), Math.max(vw - state.w, 0));
    state.y = Math.min(Math.max(state.y, 0), Math.max(vh - state.h, 0));
  };

  const apply = () => {
    win.style.left = `${state.x}px`;
    win.style.top = `${state.y}px`;
    win.style.width = `${state.w}px`;
    win.style.height = `${state.h}px`;
    win.hidden = state.min;
    fab.hidden = !state.min;
  };

  clamp();
  apply();

  const persist = () => saveGeometry(state);

  /* minimize <-> FAB: the console never disappears, it docks */
  minBtn.addEventListener("click", () => {
    state.min = true;
    apply();
    persist();
  });
  fab.addEventListener("click", () => {
    state.min = false;
    apply();
    persist();
    document.getElementById("console-output").scrollTop = 1e9;
  });

  /* drag by the header (the minimize button stays clickable).
     Start from the element's LIVE position, not persisted state: stored
     geometry can go stale after a native resize, and starting from a stale
     state made the window jump on the first move. */
  head.addEventListener("pointerdown", (ev) => {
    if (ev.target.closest("button")) return;
    const grabX = ev.clientX;
    const grabY = ev.clientY;
    const startX = win.offsetLeft;
    const startY = win.offsetTop;
    head.setPointerCapture(ev.pointerId);
    const move = (e2) => {
      state.x = startX + (e2.clientX - grabX);
      state.y = startY + (e2.clientY - grabY);
      clamp();
      apply();
    };
    const up = () => {
      head.removeEventListener("pointermove", move);
      head.removeEventListener("pointerup", up);
      persist();
    };
    head.addEventListener("pointermove", move);
    head.addEventListener("pointerup", up);
  });

  /* native CSS resize (resize: both) -- keep state in sync.
     Read BORDER-BOX metrics (offset*): apply() writes border-box
     width/height, so reading clientWidth (which excludes the border) and
     writing it back shrank the window by 2px on every observer/move
     round-trip -- the window visibly shrank each time it was dragged.
     No clamping here: clamping stored x without moving the element made
     the next move() snap; the drag path clamps on its own. */
  new ResizeObserver(() => {
    if (state.min) return;
    const w = win.offsetWidth;
    const h = win.offsetHeight;
    if (w && h && (w !== state.w || h !== state.h)) {
      state.w = w;
      state.h = h;
      persist();
    }
  }).observe(win);

  /* keep a moved/resized window reachable after viewport changes */
  window.addEventListener("resize", () => {
    clamp();
    apply();
  });
}

/* ---- boot ---------------------------------------------------------------- */

mountRail();
mountConsole();
