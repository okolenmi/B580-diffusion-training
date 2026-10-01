/* ---------------------------------------------------------------------------
   lib/value.js -- rendering a measurement that is not a number.

   A diverged trainer writes NaN/Inf; the backend ships those as `null`
   plus a `nonfinite` map on the very object that carried them (docs 07
   F-03). The honest rendering is therefore NOT an em dash -- that would
   claim nothing was ever measured when in fact the run measured garbage.
   One shared definition so the dashboard, the history table and the
   monitor page all say the same thing.

   `value-bad` is styled per module (training.css / monitor.css) so each
   stylesheet keeps ownership of its own components.
   --------------------------------------------------------------------------- */

const LABELS = { nan: "NaN", inf: "∞", "-inf": "−∞" };

/** Text for a non-finite marker ("nan" | "inf" | "-inf") from the wire. */
export function nonfiniteText(kind) {
  return `${LABELS[kind] || "NaN"} · diverged`;
}

/**
 * Write a measured value into an element.
 * @param {HTMLElement} node
 * @param {string} text  the number already formatted
 * @param {boolean} bad  true when the wire marked it non-finite
 */
export function setValue(node, text, bad = false) {
  node.textContent = text;
  node.classList.toggle("value-bad", bad);
}

/**
 * Render a measured field from a wire object that may carry a
 * `nonfinite` marker: the marker wins over the value.
 * @param {HTMLElement} node
 * @param {object} source  the DTO/event (has the field and maybe `nonfinite`)
 * @param {string} field   e.g. "current_loss"
 * @param {(v: number|null) => string} format  number -> text
 */
export function renderMeasured(node, source, field, format) {
  const kind = source && source.nonfinite ? source.nonfinite[field] : undefined;
  if (kind) {
    setValue(node, nonfiniteText(kind), true);
    node.title = "the trainer reported a non-finite value here (training diverged)";
    return;
  }
  setValue(node, format(source ? source[field] : null), false);
  node.removeAttribute("title");
}