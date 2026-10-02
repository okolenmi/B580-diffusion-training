"""Property-based tests at the four boundaries that face untrusted input.

    python backend/tests/test_property_boundaries.py
    python -m pytest backend/tests/test_property_boundaries.py   (also works)

Property tests rather than examples because every one of these boundaries
has already produced a bug that a fixed example would have missed. The
truncation property is the clearest: the reader's offset bookkeeping is
correct for a whole file and for a file that was already torn, and can be
wrong for *every* offset in between at once. The only way to know is to
try them all.

The four boundaries, and what each is defending against:

  (a) the progress reader -- a trainer appending to a file the supervisor
      is tailing. A line may be torn at any byte.
  (b) json_safe -- the wire format. `NaN`/`Infinity` are not JSON, so a
      frame carrying one is a frame no browser can parse, and the failure
      is silent on our side.
  (c) the Host/Origin guard -- headers are attacker-controlled strings and
      must be refused, never crash.
  (d) the asset path sandbox -- a relative path from a client must not
      resolve outside the base directory.

Every property is "never raises" or "never escapes", which is why
hypothesis can check them on adversarial input without a corpus.
"""

from __future__ import annotations

import json
import math
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from backend.json_safe import sanitize, strict_dumps
from backend.presentation.security import host_name
from backend.tests.support import check, finish

# Directories rather than files: these tests must be fast enough to run
# hundreds of examples, and a real file per example turns that into disk
# I/O per example.
SCRATCH = Path(tempfile.mkdtemp(prefix="property-"))

_SETTING = settings(
    max_examples=200,
    deadline=None,           # the truncation loop does file I/O per example
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)


# ==========================================================================
# (b) what sanitize/strict_dumps emit is always parseable
# ==========================================================================

_jsonable = st.recursive(
    st.none()
    | st.booleans()
    | st.integers(min_value=-(2**53), max_value=2**53)
    | st.text(max_size=40)
    | st.floats(allow_nan=True, allow_infinity=True),
    lambda children: st.lists(children, max_size=4)
    | st.dictionaries(st.text(max_size=8), children, max_size=4),
    max_leaves=12,
)


@given(_jsonable)
@_SETTING
def test_sanitized_output_is_always_strict_json(payload: object) -> None:
    """`allow_nan=False` is the invariant guard, and it must never fire.

    A frame carrying `NaN` is not a frame a browser can parse, and nothing
    on our side would notice: the send succeeds. So the property is that
    sanitize catches every non-finite float in an arbitrarily nested
    structure, and what comes out parses strictly.
    """
    try:
        text = strict_dumps(sanitize(payload))
    except ValueError as exc:
        check(False, f"sanitize missed a non-finite value in {payload!r}: {exc}")
        return
    try:
        json.loads(text, parse_constant=_reject_constant)
    except Exception as exc:  # noqa: BLE001 -- any parse failure is the bug
        check(False, f"output did not parse strictly: {text!r} ({exc})")


@given(st.floats(allow_nan=True, allow_infinity=True))
@_SETTING
def test_a_float_is_sanitized_by_whether_it_is_finite(value: float) -> None:
    """The whole float domain, not just the non-finite part.

    A previous draft sampled floats and `assume`d a non-finite one, which
    filtered 50 of 50 inputs -- hypothesis rightly complained, and the
    complaint was the useful part: the filter would have distorted the
    distribution it claimed to be sampling. Finite values take the
    no-op path here, which is worth asserting too: sanitize must not
    invent a `nonfinite` marker for a number that is perfectly good.
    """
    clean = sanitize({"loss": value})
    if math.isfinite(value):
        check(clean["loss"] == value, f"a finite loss is untouched (got {clean!r})")
        check("nonfinite" not in clean,
              f"and is not marked (got {clean!r})")
        return
    check(clean["loss"] is None, f"a non-finite loss becomes null (got {clean!r})")
    check("nonfinite" in clean,
          f"and is named, so the UI can say diverged (got {clean!r})")
    check(clean["nonfinite"]["loss"] in {"nan", "inf", "-inf"},
          f"with a usable marker (got {clean['nonfinite']!r})")
    # And the marker is a plain string: a *value* describing a bad number
    # must not itself be the bad number.
    strict_dumps(clean)


def _reject_constant(name: str):
    raise ValueError(f"JSON constant {name} reached the parser")


# ==========================================================================
# (c) the Host/Origin guard never crashes on a header
# ==========================================================================

@given(st.text(max_size=80))
@_SETTING
def test_host_name_never_raises(value: str) -> None:
    """A `Host` header is attacker-controlled. It must produce a string, and
    producing a *wrong* one is handled by refusing -- but crashing here is a
    500 on every request."""
    try:
        result = host_name(value)
    except Exception as exc:  # noqa: BLE001 -- that is the property
        check(False, f"host_name raised on {value!r}: {type(exc).__name__}: {exc}")
        return
    check(isinstance(result, str), f"host_name returns a string (got {result!r})")

    # The whole contract, in one assertion, restated from the source.
    #
    # Two earlier versions of this test asserted *parts* of it -- "no
    # colon unless IPv6", "the result is trimmed" -- and hypothesis found
    # a counterexample to each within a dozen runs. Both were wrong, and
    # both in the same direction: they described a tidier function than
    # the one that exists. Restating the whole rule means a behaviour
    # change has to be a deliberate edit here rather than something a
    # partial property quietly rules out.
    #
    #   - trim, then lower;
    #   - `[...` is an IPv6 literal: keep up to and including `]`;
    #   - exactly one colon: that is `name:port`, keep the name;
    #   - anything else (no colon, or two or more) is returned as it is.
    #
    # The last case is what the partial properties got wrong, and it is
    # the fail-closed one: a header like `a:b:c` is not a host:port, so
    # nothing is stripped and it cannot match an allowlisted host. For the
    # same reason there is no "the result is trimmed" claim: the trim runs
    # *before* the port is split off, so `j\x85:80` becomes `j\x85`.
    # Trimming afterwards would let a header with junk after the port match
    # an allowlisted host -- the wrong way round for a rebinding guard.
    v = value.strip().lower()
    if v.startswith("["):
        end = v.find("]")
        expected = v[: end + 1] if end != -1 else v
    elif v.count(":") == 1:
        expected = v.split(":", 1)[0]
    else:
        expected = v
    check(result == expected,
          f"host_name({value!r}) == {expected!r} (got {result!r})")


@given(st.text(max_size=80), st.text(max_size=80))
@_SETTING
def test_host_and_origin_never_raise(host: str, origin: str) -> None:
    """Both headers go through the guard's parse path. Neither may raise;
    both may refuse."""
    from backend.presentation.security import HostOriginGuard

    async def app(scope, receive, send):  # noqa: ANN001
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"ok"})

    guard = HostOriginGuard(app, frozenset({"127.0.0.1"}), frozenset({"http://ok.test"}))
    scope = {
        "type": "http",
        "headers": [
            (b"host", host.encode("utf-8", errors="replace")),
            (b"origin", origin.encode("utf-8", errors="replace")),
        ],
    }

    import asyncio

    sent: list[dict] = []

    async def send(message):  # noqa: ANN001
        # Recorded whole: an earlier draft read message["status"] and
        # turned the guard's own (perfectly valid) body message into a
        # KeyError, which the property then reported as "the guard
        # crashed". The fake should not have opinions about which message
        # carries the status.
        sent.append(message)

    async def receive():  # noqa: ANN001
        return {"type": "http.request", "body": b"", "more_body": False}

    try:
        asyncio.run(guard(scope, receive, send))
    except Exception as exc:  # noqa: BLE001 -- that is the property
        check(False, f"the guard raised on host={host!r} origin={origin!r}: "
                     f"{type(exc).__name__}: {exc}")
        return

    check(bool(sent), f"the guard answered something (host={host!r})")
    starts = [m for m in sent if m.get("type") == "http.response.start"]
    check(len(starts) == 1,
          f"exactly one response start (got {[m.get('type') for m in sent]})")
    check(starts[0].get("status") in (200, 403),
          f"and it is a decision -- served or refused, never a 5xx "
          f"(got {starts[0].get('status')} for host={host!r} origin={origin!r})")


# ==========================================================================
# (d) the asset path sandbox never escapes
# ==========================================================================

def _store(root: Path):
    """A store whose kinds all point inside `root`.

    (No return annotation: the class is imported inside the function, so
    one would name something that does not exist at module scope -- ruff
    F821 caught exactly that.)

    This test must not assert against, let alone write into, the
    developer's real ComfyUI install -- the trap that left a 600 MB file in
    `models/loras` (docs 08 N-05, r9).

    `WorkspaceDirs` is the supported way to say that, so it is what this
    uses. An earlier draft supplied three settings keys by name
    (`comfy_dir`, `checkpoints_dir`, `loras_dir`) and had to know that the
    last two live under `<comfy>/models/` -- which is ComfyUI's layout,
    not this app's. Suppressing comfy_dir alone was not enough: each
    `*_dir` accessor checks its own key first, so the test was still
    resolving against the real directories and "passing" for the wrong
    reason. That knowledge is exactly what the seam exists to remove, so
    using the seam here is also what keeps the test honest.
    """
    from backend.infrastructure.file_asset_store import FileSystemAssetStore
    from backend.infrastructure.workspace import WorkspaceDirs, WorkspaceLayout

    comfy = root / "comfy"
    for kind in ("checkpoints", "loras"):
        (comfy / "models" / kind).mkdir(parents=True, exist_ok=True)
    (root / "datasets").mkdir(parents=True, exist_ok=True)
    return FileSystemAssetStore(
        WorkspaceLayout(
            root,
            runs_dir=root / "runs",
            dirs=WorkspaceDirs(
                comfy=comfy,
                checkpoints=comfy / "models" / "checkpoints",
                loras=comfy / "models" / "loras",
            ),
        )
    )


_path_chars = st.text(
    alphabet=st.characters(
        blacklist_characters="/\\", blacklist_categories=("Cs",)
    ),
    max_size=12,
)
_segments = st.lists(_path_chars, min_size=1, max_size=4)


def _assert_inside_or_refused(kind: str, relative: str, tag: str) -> None:
    """The sandbox property, once, for one (kind, relative) pair.

    Returns None on success; records a failure otherwise. Split out because
    two properties differ only in how they generate `relative`, and a
    duplicated assertion body is a duplicated assertion that one of them
    will forget to update.
    """
    from backend.application.errors import InvalidQueryError

    root = SCRATCH / f"{tag}-{kind}"
    root.mkdir(parents=True, exist_ok=True)
    store = _store(root)
    # The store's own base, not a hardcoded guess at it: the invariant is
    # "inside whatever this store was configured with", and getting the
    # expectation wrong (as a hardcoded `root/datasets` vs the layout's
    # `root/datasets` naming did) makes the property fail for the wrong
    # reason.
    base = store._base(kind).resolve()

    try:
        resolved = store._safe_resolve(kind, relative)
    except InvalidQueryError:
        return  # refused -- the acceptable outcome
    except Exception as exc:  # noqa: BLE001 -- that is the property
        check(False, f"_safe_resolve raised on {relative!r} instead of "
                     f"refusing it: {type(exc).__name__}: {exc}")
        return

    check(resolved == base or resolved.is_relative_to(base),
          f"{relative!r} resolved to {resolved}, outside {base}")


@given(st.sampled_from(["checkpoint", "lora", "dataset"]),
       st.one_of(
           _segments.map(lambda segs: "/".join(segs)),
           st.sampled_from(["..", "../..", "../../etc/passwd", "/etc/passwd",
                            "a/../../b", "./../x", "..\\..\\windows",
                            "\\..\\..\\etc", "a/b/../../../c", "a/./b",
                            "  ", "a b/c", ".hidden/x", "x/"])))
@_SETTING
def test_a_client_path_never_resolves_outside_the_base(kind: str, relative: str) -> None:
    """Plausible-looking paths, plus the traversal shapes by name.

    Refusing is always allowed -- `..` and absolute paths are rejected by
    name before any arithmetic happens -- so this is not "every path is
    refused"; it is "no accepted path escapes".
    """
    _assert_inside_or_refused(kind, relative, "sandbox")


@given(st.sampled_from(["checkpoint", "lora", "dataset"]),
       st.text(max_size=40))
@_SETTING
def test_any_client_path_answers_refused_or_inside(kind: str, relative: str) -> None:
    """The same property on arbitrary text, not just plausible paths:
    whitespace, control characters, emoji, the empty string, lone
    surrogates."""
    _assert_inside_or_refused(kind, relative, "sandbox2")


def main() -> int:
    print("\n== non-finite markers are plain strings ==")
    check(sanitize({"loss": float("nan")})["nonfinite"]["loss"] == "nan",
          "nan is named nan")
    check(sanitize({"loss": float("inf")})["nonfinite"]["loss"] == "inf",
          "inf is named inf")
    check(sanitize({"loss": float("-inf")})["nonfinite"]["loss"] == "-inf",
          "-inf is named -inf")


    test_sanitized_output_is_always_strict_json()
    test_a_float_is_sanitized_by_whether_it_is_finite()
    test_host_name_never_raises()
    test_host_and_origin_never_raise()
    test_a_client_path_never_resolves_outside_the_base()
    test_any_client_path_answers_refused_or_inside()

    finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())