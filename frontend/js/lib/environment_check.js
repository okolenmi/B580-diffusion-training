// @ts-check
/* ---------------------------------------------------------------------------
   lib/environment_check.js -- is the chosen Python environment usable, asked the way Setup asks it.

   Shared by the start page and Settings so the two cannot disagree about the same machine. The check imports PyTorch in the
   environment, so it takes seconds and is only ever run on request or once per page load.
   --------------------------------------------------------------------------- */

import { api } from "../api.js";
import { errText } from "./errors.js";

/**
 * The virtualenv folder an interpreter belongs to, or null for a bare command (`python`) that names no folder.
 * @param {string|null|undefined} python
 */
export function venvFolder(python) {
  if (!python) return null;
  const match = /^(.*)[\\/](?:bin|Scripts)[\\/][^\\/]+$/.exec(python);
  return match && match[1] ? match[1] : null;
}

/**
 * @param {string|null|undefined} python  the interpreter in use (resolved)
 * @returns {Promise<{python: string|null, folder: string|null, present: boolean|null, name: string|null, totalMemoryMb: number|null,
 *   torchVersion: string|null, reason: string|null, serverInEnvironment: boolean|null}>}
 */
export async function checkEnvironment(python) {
  const folder = venvFolder(python);
  const blank = { python: python || null, folder, present: null, name: null, totalMemoryMb: null, torchVersion: null, reason: null, serverInEnvironment: null };
  if (!folder) return { ...blank, reason: "No virtualenv is chosen yet." };
  try {
    const v = await api("/installer/environment/verify", { method: "POST", body: { target: "existing", venv_dir: folder } });
    return { ...blank, present: v.present, name: v.name, totalMemoryMb: v.total_memory_mb, torchVersion: v.torch_version, reason: v.reason, serverInEnvironment: v.server_in_environment };
  } catch (err) {
    return { ...blank, reason: errText(err) };
  }
}
