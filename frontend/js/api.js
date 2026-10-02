/* ---------------------------------------------------------------------------
   api.js -- the ONE place fetch happens (03-migration-strategy.md §2).

   Every call goes through the backend's /api/v1 surface and decodes the
   single error envelope {"error": {code, message, details}} into a
   rejected ApiError, so views never hand-roll response parsing and the
   code/status mapping lives here exactly once.
   --------------------------------------------------------------------------- */

const BASE = "/api/v1";

export class ApiError extends Error {
  constructor(code, message, status, details) {
    super(message);
    this.name = "ApiError";
    this.code = code;      // machine code: "run_already_active", "validation_error", ...
    this.status = status;   // HTTP status
    this.details = details; // optional: graph issues, pydantic field errors
  }
}

/**
 * @param {string} path  path under /api/v1 (e.g. "/runs?limit=20")
 * @param {object} [opts] {method, body} -- body objects are JSON-encoded
 *                        unless opts.rawBody is given (bytes/string sent as-is).
 * @returns {Promise<any>} parsed JSON body (or null for empty responses)
 * @throws {ApiError} on any non-2xx, envelope decoded when present
 */
export async function api(path, opts = {}) {
  const { method = "GET", body, rawBody, headers = {} } = opts;
  const init = { method, headers: { ...headers } };
  if (rawBody !== undefined) {
    init.body = rawBody;
  } else if (body !== undefined) {
    init.headers["Content-Type"] = "application/json";
    init.body = JSON.stringify(body);
  }
  const res = await fetch(BASE + path, init);
  const text = await res.text();
  let data = null;
  if (text) {
    try {
      data = JSON.parse(text);
    } catch {
      // A non-JSON body should not happen on /api/v1; if it does, the
      // server's own error text is lost and the caller gets the generic
      // message below. Worth a line in the console so that is visible
      // rather than inferred from a vague UI message.
      console.warn("api: non-JSON response body", res.status, url);
      data = null;
    }
  }
  if (!res.ok) {
    const err = data && data.error ? data.error : null;
    throw new ApiError(
      err && err.code ? err.code : `http_${res.status}`,
      err && err.message ? err.message : `request failed (${res.status})`,
      res.status,
      err ? err.details : undefined
    );
  }
  return data;
}

/**
 * Subscribe to an SSE endpoint. Returns the source so the caller can
 * close it (page teardown).
 *
 * EventSource reconnects on its own and the server never replays: a
 * frame published while the client was away is gone for good. So
 * `onOpen` fires on EVERY (re)connect and is where a subscriber
 * refetches its authoritative state (docs 07 F-09); `onMessage` is for
 * live patches only. The caller owns message routing (see monitor.js /
 * views/dashboard.js), including what to do with an unparsable frame --
 * it must be surfaced, never dropped quietly.
 */
export function sse(path, { onMessage, onOpen, onError } = {}) {
  const source = new EventSource(BASE + path);
  if (onOpen) source.onopen = onOpen;
  if (onError) source.onerror = onError;
  if (onMessage) source.onmessage = onMessage;
  return source;
}
