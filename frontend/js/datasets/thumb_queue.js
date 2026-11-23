/* datasets/thumb_queue.js -- which thumbnails to fetch next. Pure (no DOM): the mosaic says which pictures are near the
   visible part, and this hands out at most `limit` at a time, nearest to the centre of the view first, and remembers
   what loaded and what failed so a failed one is retried a few times, with growing waits, and then given up on. */

export class ThumbQueue {
  constructor({ limit = 6, retries = 3, wait = 400 } = {}) {
    this.limit = limit;
    this.retries = retries;
    this.wait = wait;
    this.wanted = new Set();
    this.active = new Set();
    this.done = new Set();
    this.failures = new Map();
  }

  /** The picture came near the view. Ignored when it is already loading or loaded. */
  want(id) {
    if (!this.active.has(id) && !this.done.has(id)) this.wanted.add(id);
  }

  /** It left the neighbourhood before its request started. */
  unwant(id) { this.wanted.delete(id); }

  /** The nearest wanted picture to `centre` (`at(id)` is a picture's position), now active; null when there is none or
      the limit is reached. */
  next(at, centre) {
    if (this.active.size >= this.limit) return null;
    let best = null;
    let bestDistance = Infinity;
    for (const id of this.wanted) {
      const distance = Math.abs(at(id) - centre);
      if (distance < bestDistance) { best = id; bestDistance = distance; }
    }
    if (best === null) return null;
    this.wanted.delete(best);
    this.active.add(best);
    return best;
  }

  loaded(id) {
    this.active.delete(id);
    this.done.add(id);
    this.failures.delete(id);
  }

  /** A request failed: the wait in ms before trying again, or null once the retries are used up. */
  failed(id) {
    this.active.delete(id);
    const n = (this.failures.get(id) || 0) + 1;
    this.failures.set(id, n);
    return n > this.retries ? null : this.wait * 2 ** (n - 1);
  }

  retry(id) { this.wanted.add(id); }

  forget(id) {
    this.wanted.delete(id);
    this.active.delete(id);
  }
}
