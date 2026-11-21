/* ---------------------------------------------------------------------------
   lib/picker_options.js -- the names a picker control offers, for the graph editor and for the Apps page alike.
   --------------------------------------------------------------------------- */

import { api } from "../api.js";

/** What a picker offers: model files of the node's `scope`, or the image libraries the nodes can open. */
const PICKER_TTL_MS = 5000;
const pickerAnswers = new Map();

export function pickerOptions(spec, values) {
  const route = spec.picker === "dataset" ? "/catalog/libraries" : spec.picker === "model_file" ? `/catalog/models/${encodeURIComponent(spec.ui?.picker_scope || spec.scope || values.scope || "checkpoints")}` : null;
  if (!route) return Promise.resolve([]);
  const kept = pickerAnswers.get(route);
  if (kept && Date.now() - kept.at < PICKER_TTL_MS) return kept.answer;
  const answer = api(route).then((reply) => (spec.picker === "dataset" ? reply.names || [] : reply.files || []));
  pickerAnswers.set(route, { at: Date.now(), answer });
  answer.catch(() => pickerAnswers.delete(route));
  return answer;
}
