# Contract: datasets (image libraries) API

A dataset is a folder `datasets/<name>/` holding `library.sqlite` (images by content hash with their size, path and captions,
and named sets) and `caches/<vae>/` (latent caches, built by runs). Images stay where they are on disk. The old trajectory
datasets (`metadata.db`, shards) are gone from the page and the API.

All routes are under `/api/v1/datasets`; errors use the one envelope.

| Route | Purpose |
|---|---|
| `GET /` | `{datasets: [{name, images, bytes, sets: {name: count}, captioned: {channel: count}, orientation: {portrait, square, landscape}, covers: [image id x4], caches: [{vae, files, bytes}]}]}` |
| `POST /` `{name}` | create an empty dataset (201; 409 exists; 422 unusable name) |
| `DELETE /{name}` | remove the library and its caches, never the image files |
| `GET /{name}/images?set=&q=&missing=&offset=&limit=` | `{total, items: [{id, path, width, height, bytes, captions: {text, tags, desc}, sets: [names besides all]}]}`, newest first; `q` matches captions and paths; `missing=<channel>` keeps only images without a caption there (422 for an unknown channel) |
| `GET /{name}/image-ids?set=&q=&missing=` | `{ids}` every id the same filters select, so "select all" reaches images that are not loaded |
| `GET /{name}/report?channel=` | what to check before a run: `{images, sizes, distinct_sizes, smallest_side, empty_captions, repeated_captions, distinct_captions, tags, distinct_tags}` |
| `GET /{name}/images/{id}/file?thumb=N` | the picture or a WebP thumbnail; by id only, never by path |
| `POST /{name}/captions` | `{ids|null, set, query, channel, mode: set|prepend|append|replace|clear, text, find}`; `ids: null` = everything the set and query select; prepend/append skip an image that already has the text there |
| `POST /{name}/images/remove` | `{ids, set|null}` out of a set, or out of the dataset when `set` is null |
| `POST /{name}/sets` | `{name, ids}` add images to a set (created when new) |
| `DELETE /{name}/sets/{set}` | delete a set (not `all`) |
| `DELETE /{name}/caches/{vae}` | delete one latent cache |
| `POST /{name}/imports` | `{folder, recursive, set}` read a folder in the background (202; one at a time per dataset); sidecar `name.txt`, `name.tags.txt`, `name.desc.txt` become captions |
| `GET /{name}/imports` | jobs of this server run, newest first: `{id, state: running|done|failed, done, total, added, known, skipped, error}` |

The node pickers list datasets through `GET /api/v1/catalog/libraries` (names that have a `library.sqlite`).
Not built yet: generating images with a teacher checkpoint into a dataset; uploading images from the browser.

## The page (`frontend/js/datasets/`)

`/datasets` shows cards (a mosaic of the newest pictures, caption completeness, sets and caches); the dashed first card creates one.
`/datasets/{name}` is a workbench:

- **Mosaic** (`mosaic.js`), two layouts chosen in the toolbar and remembered in the browser. *Balanced* (`packing.js`, default) packs blocks of six pictures, grouped by shape, into a tree of side-by-side and stacked splits chosen so that every picture gets about the same area, then stacks blocks into columns; tall and wide pictures end up the same visual size. *Rows* (`layout.js`) is a justified-row grid where wide pictures come out bigger. Both keep every picture's exact proportions. The layout is recomputed after a resize settles (120 ms), tiles are absolutely positioned, and pictures of the last unfinished block wait until more pages arrive. Tile size slider; infinite scroll.
- **Selection** (`selection.js`): click, Ctrl/Cmd-click toggles, Shift-click ranges, `Select all` takes every id of the view (loaded or not). Keys: arrows move to the geometrically nearest tile in that direction (the layout is uneven; the viewer and ranges follow the displayed order, which Balanced reorders by shape), Space/Enter opens the full-size viewer, Esc clears, Ctrl+A selects all, Delete removes after a confirmation.
- **Inspector** beside the mosaic: *Selection* tab (one image: preview, caption editor that saves by itself on idle, blur, image change and channel change, sets; several: bulk caption edit with a preview of the result on the first image, add to a set, take out of the current set, remove) and *Library* tab (shapes, caption completeness with "show missing", the report: repeated and empty captions, sizes, most used tags that search on click, sets, latent caches).
- **Set tabs** above the mosaic; dragging the selection onto a set tab adds it to that set.
- **Import** dialog with progress; a pill in the page header keeps the progress visible after the dialog is closed.

`LibraryState` is the one object the panels share (view, loaded images, selection, operations); panels listen to its events and never call the API for the dataset themselves.
