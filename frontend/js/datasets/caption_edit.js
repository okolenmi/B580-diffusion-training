/* datasets/caption_edit.js -- what a bulk caption edit will do to one caption, for the preview line. The server does the
   real edit (next/data/library.py edit_captions); this mirrors it so the preview matches, and the test pins both. */

export const EDITS = [
  ["prepend", "Add at the start"],
  ["append", "Add at the end"],
  ["replace", "Find and replace"],
  ["set", "Set to"],
  ["clear", "Clear"],
];

const joined = (a, b) => `${a}${a && b ? ", " : ""}${b}`;

export function applyEdit(old, mode, text, find = "") {
  switch (mode) {
    case "set": return text;
    case "clear": return "";
    case "replace": return find ? old.split(find).join(text) : old;
    case "prepend": return old.startsWith(text) ? old : joined(text, old);
    case "append": return old.endsWith(text) ? old : joined(old, text);
    default: throw new Error(`unknown caption edit ${mode}`);
  }
}
