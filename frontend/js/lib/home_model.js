// @ts-check
/* ---------------------------------------------------------------------------
   lib/home_model.js -- what the start page says, derived from what the server answered.

   Pure: the view fetches, this decides. Each fetch can fail on its own, so every input may be null ("could not be read") and the
   result says "unknown" for it instead of guessing in either direction.
   --------------------------------------------------------------------------- */

/**
 * Whether the node modules loaded. A module that failed to import takes its nodes with it, and the graph editor shows them red.
 * @param {{load_errors?: {module: string, error: string}[]} | null} catalog
 * @returns {{state: "ok"|"bad"|"unknown", failed: number, reason: string|null}}
 */
export function nodeHealth(catalog) {
  if (!catalog) return { state: "unknown", failed: 0, reason: null };
  const errors = catalog.load_errors || [];
  if (!errors.length) return { state: "ok", failed: 0, reason: null };
  // One cause usually explains all of them (a missing package); name it once rather than sixteen times.
  const first = errors[0].error;
  const same = errors.every((e) => e.error === first);
  return { state: "bad", failed: errors.length, reason: same ? first : `${first} (and other errors)` };
}

/**
 * The card memory as the admission ledger knows it. Nulls stay null: "not measured yet" is not "zero".
 * @param {{total_mb: number|null, capacity_mb?: number, held_mb?: number, free_mb?: number, holders?: Record<string, {mb: number}>} | null | undefined} memory
 */
export function memoryView(memory) {
  if (!memory || memory.total_mb == null) {
    return { known: false, totalMb: null, heldMb: 0, freeMb: null, holders: holdersOf(memory) };
  }
  return {
    known: true,
    totalMb: memory.total_mb,
    heldMb: memory.held_mb ?? 0,
    freeMb: memory.free_mb ?? null,
    holders: holdersOf(memory),
  };
}

function holdersOf(memory) {
  return Object.entries((memory && memory.holders) || {}).map(([owner, claim]) => ({ owner, mb: claim.mb }));
}

/**
 * The first-run list. An item is done only when the server said so; an unreadable answer leaves it open rather than ticked.
 * @param {{
 *   environmentReady: boolean|null, checkpoints: number|null, datasets: number|null,
 *   promptFiles: {complete: boolean}[]|null, runs: number|null
 * }} facts
 */
export function checklist(facts) {
  const promptsDone = facts.promptFiles ? facts.promptFiles.some((set) => set.complete) : false;
  return [
    {
      id: "environment", done: facts.environmentReady === true, title: "PyTorch sees your graphics card",
      detail: "Training needs the Intel XPU build of PyTorch in the environment the server runs in.",
      href: "/settings#environment", action: "Check the environment",
    },
    {
      id: "checkpoint", done: (facts.checkpoints ?? 0) > 0, title: "A checkpoint is in the models folder",
      detail: "Copy an SDXL .safetensors file into <models folder>/checkpoints.",
      href: "/settings#folders", action: "See the folders",
    },
    {
      id: "dataset", done: (facts.datasets ?? 0) > 0, title: "A dataset is imported",
      detail: "Point a dataset at the folder with your pictures and write captions.",
      href: "/datasets", action: "Open datasets",
    },
    {
      id: "prompts", done: promptsDone, title: "Prompt files are downloaded",
      detail: "The vocabulary files the prompt encoders read.",
      href: "/settings#environment", action: "Download them",
    },
    {
      id: "run", done: (facts.runs ?? 0) > 0, title: "A first training run",
      detail: "Quick workflow in the graph editor builds the graph; a few hundred steps show that everything works.",
      href: "/graph", action: "Open the editor",
    },
  ];
}

/** Items still open, in order; the first is what to do next. */
export const remaining = (items) => items.filter((item) => !item.done);
