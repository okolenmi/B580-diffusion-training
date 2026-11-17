/* ---------------------------------------------------------------------------
   views/settings.js -- the folders and the Python environment, editable after setup.

   An empty field means "detect it": the placeholder then shows what is in use now. Values are validated by the server, which names
   the field that was refused.
   --------------------------------------------------------------------------- */

import { api, ApiError } from "../api.js";
import { h, mount } from "../lib/dom.js";
import { errText } from "../lib/errors.js";
import { logError } from "../lib/log.js";

const FIELDS = [
  ["models_dir", "Models folder", "Checkpoints are read from <folder>/checkpoints and trained LoRAs are saved to <folder>/loras. A ComfyUI `models` folder works as it is."],
  ["checkpoints_dir", "Checkpoints folder", "Only when checkpoints live somewhere other than <models folder>/checkpoints."],
  ["loras_dir", "LoRA folder", "Only when LoRAs should go somewhere other than <models folder>/loras."],
  ["venv_python", "Python interpreter", "The python file of the virtualenv that trains (venv/bin/python, venv\\Scripts\\python.exe on Windows). Training runs use it at once; the server itself moves into it the next time it starts."],
];

const inputs = {};
const note = h("p", { class: "settings-note", role: "status" });

function say(text, bad = false) {
  note.textContent = text;
  note.className = `settings-note ${bad ? "settings-bad" : "text-dim"}`;
}

function fill(view) {
  for (const [key] of FIELDS) {
    inputs[key].value = view.stored[key] || "";
    inputs[key].placeholder = view.resolved[key] ? `in use: ${view.resolved[key]}` : "not set";
  }
}

async function save() {
  const body = Object.fromEntries(FIELDS.map(([key]) => [key, inputs[key].value.trim()]));
  try {
    fill(await api("/settings", { method: "POST", body }));
    say("Saved.");
  } catch (err) {
    const details = err instanceof ApiError && err.details && typeof err.details === "object" ? Object.values(err.details).join(" ") : "";
    say(details || errText(err), true);
    logError("settings: save failed", err);
  }
}

function card() {
  return h("section", { class: "card settings-card", id: "folders-group" },
    h("div", { class: "card-header" }, h("span", {}, "Folders and environment")),
    FIELDS.map(([key, label, hint]) => {
      inputs[key] = h("input", { class: "settings-input", type: "text", spellcheck: false, autocomplete: "off", "aria-label": label,
        onkeydown: (e) => { if (e.key === "Enter") save(); } });
      return h("label", { class: "settings-field" },
        h("span", { class: "settings-field-label" }, label), inputs[key], h("span", { class: "settings-field-hint text-dim" }, hint));
    }),
    h("div", { class: "settings-actions" }, h("button", { class: "btn btn-primary", type: "button", onclick: save }, "Save"), note));
}

async function boot() {
  mount(document.getElementById("settings-folders"), card());
  fill(await api("/settings"));
}

boot().catch((err) => {
  logError("settings: could not load", err);
  say(errText(err), true);
});
