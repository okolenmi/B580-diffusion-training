/* ---------------------------------------------------------------------------
   board/picture_swap.js -- one <img> that follows a stream of pictures without flicker or wasted work.

   A live preview can deliver pictures faster than they load. `show(url)` keeps only the newest request: while one picture loads
   the others replace each other in a single waiting slot, so a burst costs one load of the last picture, not one per frame.
   A picture is put on the element only after it has loaded and decoded, so the old one stays until the new one can be painted whole
   (no blank flash, no half-decoded frame), and a picture that fails to load leaves the old one in place.

   `load` is injectable (frontend/tests/picture_swap.test.mjs); the default decodes off-screen through an `Image`.
   --------------------------------------------------------------------------- */

const decodeOffscreen = async (url) => {
  const probe = new Image();
  probe.src = url;
  await probe.decode();
};

export class PictureSwap {
  /** @param {{src: string}} img @param {{load?: (url: string) => Promise<void>}} [options] */
  constructor(img, { load = decodeOffscreen } = {}) {
    this.img = img;
    this._load = load;
    this._waiting = null;
    this._busy = false;
    this._gone = false;
    this.shown = null;
  }

  /** Ask for `url` to be on the element as soon as it can be. The same url again is free. */
  show(url) {
    if (this._gone || url === this.shown || url === this._waiting) return;
    this._waiting = url;
    if (!this._busy) this._pump();
  }

  async _pump() {
    this._busy = true;
    while (this._waiting !== null && !this._gone) {
      const url = this._waiting;
      this._waiting = null;
      try {
        await this._load(url);
        if (this._gone) break;
        this.img.src = url;
        this.shown = url;
      } catch { /* the old picture stays; a newer request may still come */ }
    }
    this._busy = false;
  }

  destroy() {
    this._gone = true;
    this._waiting = null;
  }
}
