# Contract: datasets (image libraries) API

A dataset is a folder `datasets/<name>/` holding `library.sqlite` (images by content hash with their size, path and captions,
and named sets) and `caches/<vae>/` (latent caches, built by runs). Images stay where they are on disk. The old trajectory
datasets (`metadata.db`, shards) are gone from the page and the API.

All routes are under `/api/v1/datasets`; errors use the one envelope.

| Route | Purpose |
|---|---|
| `GET /` | `{datasets: [{name, images, sets: {name: count}, captioned: {channel: count}, caches: [{vae, files, bytes}]}]}` |
| `POST /` `{name}` | create an empty dataset (201; 409 exists; 422 unusable name) |
| `DELETE /{name}` | remove the library and its caches, never the image files |
| `GET /{name}/images?set=&q=&offset=&limit=` | `{total, items: [{id, path, width, height, bytes, captions: {text, tags, desc}}]}`, newest first; `q` matches captions and paths |
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
