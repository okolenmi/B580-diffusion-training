/* Keeps what is typed on an app's page, so that opening the page again shows it. Using an app is not publishing it: the values go to
   their own endpoint (docs/contracts/apps.md, "Using an app") and never make a version; Save is for the author's layout.

   A change is written a moment after the last one; closing the page writes what is waiting at once (a request that is allowed to outlive it). */

import { api } from "../api.js";

const enc = encodeURIComponent;

export class ValueKeeper {
  /** `values()` answers the page's current values in the shape of a run's `values`; `enabled()` is false until the app exists on the server. */
  constructor({ name, values, enabled, delay = 700, onError = () => {} }) {
    this.name = name;
    this.values = values;
    this.enabled = enabled;
    this.delay = delay;
    this.onError = onError;
    this.timer = 0;
    this.waiting = false;
    this.last = null; // what the server holds, to skip a write that would change nothing
    addEventListener("pagehide", () => this.flush({ leaving: true }));
  }

  /** The value of a field changed. */
  touch() {
    this.waiting = true;
    clearTimeout(this.timer);
    this.timer = setTimeout(() => this.flush(), this.delay);
  }

  /** What the server already holds (from loading the app), so the first write is not a repeat of it. */
  known(values) { this.last = JSON.stringify(values || {}); }

  async flush({ leaving = false } = {}) {
    clearTimeout(this.timer);
    if (!this.waiting || !this.enabled()) return;
    this.waiting = false;
    const body = JSON.stringify({ values: this.values() });
    const sent = JSON.stringify(JSON.parse(body).values);
    if (sent === this.last) return;
    try {
      if (leaving) {
        fetch(`/api/v1/apps/${enc(this.name)}/values`, { method: "PUT", headers: { "Content-Type": "application/json" }, body, keepalive: true });
      } else {
        await api(`/apps/${enc(this.name)}/values`, { method: "PUT", body: JSON.parse(body) });
      }
      this.last = sent;
    } catch (err) {
      this.waiting = true;
      this.onError(err);
    }
  }
}
