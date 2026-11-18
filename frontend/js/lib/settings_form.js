// @ts-check
/* ---------------------------------------------------------------------------
   lib/settings_form.js -- one group of path settings: its inputs, "unsaved" state, and the save call.

   An empty input means "detect it" (the server clears the override), so an input is changed when its trimmed text differs from the
   stored override -- not from what is in use, which is shown as the placeholder.
   --------------------------------------------------------------------------- */

import { api, ApiError } from "../api.js";
import { h } from "./dom.js";
import { errText } from "./errors.js";

/**
 * Keys whose text differs from the stored override.
 * @param {Record<string, string>} stored
 * @param {Record<string, string>} typed
 * @returns {string[]}
 */
export function changedKeys(stored, typed) {
  return Object.keys(typed).filter((key) => typed[key].trim() !== (stored[key] || ""));
}

/** The sentence for a refused save: the server names the field it refused, which beats the generic code. */
export function refusal(err) {
  const details = err instanceof ApiError && err.details && typeof err.details === "object" ? Object.values(err.details).join(" ") : "";
  return details || errText(err);
}

export class SettingsForm {
  /**
   * @param {{key: string, label: string, hint: string}[]} fields
   * @param {(view: any) => void} onSaved  called with the fresh settings view after a successful save
   */
  constructor(fields, onSaved = () => {}) {
    this.fields = fields;
    this.onSaved = onSaved;
    this.stored = /** @type {Record<string, string>} */ ({});
    this.inputs = /** @type {Record<string, HTMLInputElement>} */ ({});
    this.errors = /** @type {Record<string, HTMLElement>} */ ({});
    this.status = h("span", { class: "say", role: "status" });
    this.saveButton = h("button", { class: "btn btn-primary", type: "button", disabled: true, onclick: () => this.save() }, "Save");
    this.revertButton = h("button", { class: "btn btn-secondary", type: "button", disabled: true, onclick: () => this.revert() }, "Revert");
  }

  /** The labelled input for one key. */
  field(key) {
    const spec = this.fields.find((f) => f.key === key);
    if (!spec) throw new Error(`no such setting field: ${key}`);
    const input = /** @type {HTMLInputElement} */ (h("input", {
      class: "settings-input field-input", type: "text", spellcheck: false, autocomplete: "off", "aria-label": spec.label,
      oninput: () => this.touch(),
      onkeydown: (e) => { if (e.key === "Enter") this.save(); },
    }));
    this.inputs[key] = input;
    this.errors[key] = h("span", { class: "field-error", hidden: true });
    return h("label", { class: "field" }, h("span", { class: "field-label" }, spec.label), input,
      h("span", { class: "field-hint" }, spec.hint), this.errors[key]);
  }

  typed() {
    return Object.fromEntries(Object.entries(this.inputs).map(([key, input]) => [key, input.value]));
  }

  get dirty() {
    return changedKeys(this.stored, this.typed()).length > 0;
  }

  touch() {
    this.saveButton.disabled = this.revertButton.disabled = !this.dirty;
    this.say(this.dirty ? "Unsaved changes." : "");
    for (const [key, input] of Object.entries(this.inputs)) { input.classList.remove("is-bad"); this.errors[key].hidden = true; }
  }

  /** @param {{stored: Record<string, string>, resolved: Record<string, string|null>}} view */
  fill(view) {
    this.stored = view.stored;
    for (const [key, input] of Object.entries(this.inputs)) {
      input.value = view.stored[key] || "";
      input.placeholder = view.resolved[key] ? `in use: ${view.resolved[key]}` : "not set";
    }
    this.saveButton.disabled = this.revertButton.disabled = true;
  }

  revert() {
    for (const [key, input] of Object.entries(this.inputs)) input.value = this.stored[key] || "";
    this.touch();
  }

  say(text, kind = "") {
    this.status.textContent = text;
    this.status.className = `say${kind ? ` say-${kind}` : ""}`;
  }

  async save() {
    const keys = changedKeys(this.stored, this.typed());
    if (!keys.length) return;
    const typed = this.typed();
    try {
      const view = await api("/settings", { method: "POST", body: Object.fromEntries(keys.map((key) => [key, typed[key].trim()])) });
      this.fill(view);
      this.say("Saved.", "ok");
      this.onSaved(view);
    } catch (err) {
      const text = refusal(err);
      // The server's sentence quotes the refused value (or names the field); mark that input, else report in the footer.
      const hit = this.fields.find((f) => keys.includes(f.key) && typed[f.key].trim() && (text.includes(typed[f.key].trim()) || text.toLowerCase().includes(f.label.toLowerCase())));
      if (hit) { this.inputs[hit.key].classList.add("is-bad"); this.errors[hit.key].textContent = text; this.errors[hit.key].hidden = false; this.say(""); }
      else this.say(text, "bad");
    }
  }
}
