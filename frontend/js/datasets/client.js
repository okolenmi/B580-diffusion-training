/* datasets/client.js -- the Datasets page's calls to /api/v1/datasets, in one place. */

import { api } from "../api.js";

const base = (name) => `/datasets/${encodeURIComponent(name)}`;
const filters = ({ set = "all", q = "", missing = "" }) =>
  `set=${encodeURIComponent(set)}&q=${encodeURIComponent(q)}&missing=${encodeURIComponent(missing)}`;

export const datasets = {
  list: () => api("/datasets").then((r) => r.datasets),
  create: (name) => api("/datasets", { method: "POST", body: { name } }),
  remove: (name) => api(base(name), { method: "DELETE" }),
  images: (name, view, { offset = 0, limit = 80 } = {}) => api(`${base(name)}/images?${filters(view)}&offset=${offset}&limit=${limit}`),
  ids: (name, view) => api(`${base(name)}/image-ids?${filters(view)}`).then((r) => r.ids),
  report: (name, channel) => api(`${base(name)}/report?channel=${channel}`),
  imageUrl: (name, id, thumb) => `/api/v1${base(name)}/images/${id}/file${thumb ? `?thumb=${thumb}` : ""}`,
  editCaptions: (name, body) => api(`${base(name)}/captions`, { method: "POST", body }),
  removeImages: (name, ids, set = null) => api(`${base(name)}/images/remove`, { method: "POST", body: { ids, set } }),
  addToSet: (name, setName, ids) => api(`${base(name)}/sets`, { method: "POST", body: { name: setName, ids } }),
  deleteSet: (name, setName) => api(`${base(name)}/sets/${encodeURIComponent(setName)}`, { method: "DELETE" }),
  deleteCache: (name, vae) => api(`${base(name)}/caches/${encodeURIComponent(vae)}`, { method: "DELETE" }),
  startImport: (name, folder, recursive, set) => api(`${base(name)}/imports`, { method: "POST", body: { folder, recursive, set } }),
  imports: (name) => api(`${base(name)}/imports`).then((r) => r.imports),
};

export const CHANNELS = ["text", "tags", "desc"];

export const fmtBytes = (n) => (n < 1e3 ? `${n} B` : n >= 1e9 ? `${(n / 1e9).toFixed(1)} GB` : n >= 1e6 ? `${(n / 1e6).toFixed(1)} MB` : `${Math.max(1, Math.round(n / 1e3))} kB`);
export const plural = (n, one, many = `${one}s`) => `${n.toLocaleString("en")} ${n === 1 ? one : many}`;
