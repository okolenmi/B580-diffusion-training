# 0001 — No authentication; a Host/Origin guard instead

## Context

This is a single-user tool for one workstation. The training backend starts
GPU jobs, writes model files, and edits configuration. The obvious question
is whether to put authentication in front of it.

The threat that actually exists is not a determined attacker with an
account. It is **a web page the user happens to have open**. The server
listens on loopback; the browser can reach it from any origin the user
visits; and a page that can POST to `127.0.0.1:8766` can start a training
run, discard curated dataset items, or rewrite the config. That is a
drive-by from any tab, including ones the user opened on purpose.

Auth proper would be the wrong shape for that: it implies accounts, a login
flow, and a secret, on a loopback service that only one person uses.

## Decision

No authentication. Instead, `backend/presentation/security.py` refuses
requests whose `Host` is not an answer this server gives, and refuses
state-changing requests whose `Origin` is neither the server's own origin
nor an explicitly allowed one.

* `GET`/`HEAD` are not origin-checked. A hostile page can read with them,
  and reads are scoped to a loopback deployment — the threat model stops at
  *writing*, not at a name appearing in a response.
* The guard sits **above** the router as pure ASGI middleware, so a refusal
  is a real response rather than an exception that would be caught by the
  error middleware and reported as a 500.
* `DEFAULT_HOSTS` is loopback only. `BACKEND_ALLOWED_HOSTS` and
  `BACKEND_ALLOWED_ORIGINS` exist for serving the UI from somewhere else.

`host_name()` trims and lowercases the header **before** splitting off a
port, and that order is deliberate: trimming afterwards would let
`Host: evil<U+0085>:80` normalise to `evil` and match an allowlisted host.
A header the guard does not fully recognise is refused, not tidied.

## Consequences

Accepted, and worth stating rather than discovering:

* **A non-browser client on the network is not stopped.** Anything that can
  open a socket can send `Host: 127.0.0.1:8766` and it will be served.
  DNS rebinding is what the Host check is actually for; a determined
  remote attacker who can already reach the port is out of scope.
* **Serving the UI remotely requires configuring both variables**, and
  forgetting `BACKEND_ALLOWED_ORIGINS` fails as *refused writes*, not as a
  clear startup error. That has cost time before.
* The guard is the whole security boundary. There is no second layer, so a
  bug in it is a security bug.

## What pins it

* `backend/tests/test_api.py` — refused Host, refused cross-origin write,
  allowed same-origin write, and the loopback defaults.
* `backend/tests/test_property_boundaries.py` — `host_name` is asserted
  against its *whole* contract on arbitrary input, including the
  two-colon case that must be returned unchanged rather than split, and
  random headers must never raise.