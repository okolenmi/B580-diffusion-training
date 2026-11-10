"""The Datasets API over image libraries: create/delete, folder import with progress, paging and search, caption edits,
sets, thumbnails and the error envelope. Real composition root, real files. Run directly: python backend/tests/test_api_libraries.py
"""

from __future__ import annotations

import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from PIL import Image

from backend.bootstrap import build_container
from backend.config import Settings
from backend.presentation.app import create_app
from backend.tests.support import asgi_request, check, finish


def main() -> None:
    root = Path(tempfile.mkdtemp())
    pics = root / "pics"
    pics.mkdir()
    for n in range(4):
        Image.new("RGB", (90 + n, 70), (n * 50, 20, 20)).save(pics / f"p{n}.png")
        (pics / f"p{n}.txt").write_text(f"a cat, n{n}")
    app = create_app(build_container(Settings(project_root=root, db_path=root / "b.db")).services)

    def call(path: str, method: str = "GET", body: dict | None = None):
        status, _, payload = asgi_request(app, "/api/v1/datasets" + path, method=method, json_body=body)
        return status, payload

    status, made = call("", "POST", {"name": "mine"})
    check(status == 201 and made["images"] == 0 and made["sets"] == {"all": 0}, "a dataset is created empty")
    check(call("", "POST", {"name": "mine"})[0] == 409, "the same name twice is a conflict")
    check(call("", "POST", {"name": "../x"})[0] == 422, "a name that is not a plain folder name is refused")
    check(call("/ghost/images")[0] == 404, "an unknown dataset is a 404")

    status, job = call("/mine/imports", "POST", {"folder": str(pics)})
    check(status == 202 and job["state"] in ("running", "done"), "an import starts in the background")
    for _ in range(50):
        latest = call("/mine/imports")[1]["imports"][0]
        if latest["state"] != "running":
            break
        time.sleep(0.1)
    check(latest["state"] == "done" and latest["added"] == 4 and latest["total"] == 4, "the import finishes and reports what it added")
    check(call("/mine/imports", "POST", {"folder": str(root / "nowhere")})[0] == 422, "a folder that does not exist is refused")

    status, page = call("/mine/images?limit=2")
    check(status == 200 and page["total"] == 4 and len(page["items"]) == 2, "images come in pages with a total")
    check(call("/mine/images?q=n2")[1]["total"] == 1, "search matches captions")
    image_id = page["items"][0]["id"]
    status, _, body = asgi_request(app, f"/api/v1/datasets/mine/images/{image_id}/file?thumb=32")
    check(status == 200, "a thumbnail is served by image id")
    check(asgi_request(app, "/api/v1/datasets/mine/images/deadbeef/file")[0] == 404, "an unknown image id is a 404, never a path")

    check(call("/mine/captions", "POST", {"mode": "prepend", "text": "trigger"}) == (200, {"changed": 4}), "a caption edit without ids applies to everything selected by set and search")
    check(call("/mine/captions", "POST", {"mode": "nonsense"})[0] == 422, "an unknown edit kind is refused")
    check(call("/mine/sets", "POST", {"name": "pick", "ids": [image_id]})[1]["members"] == 1, "a set is made from chosen images")
    summary = call("")[1]["datasets"][0]
    check(summary["sets"] == {"all": 4, "pick": 1} and summary["captioned"]["text"] == 4, "the list carries counts per set and per caption channel")
    check(call("/mine/sets/all", "DELETE")[0] == 422, "the all set cannot be deleted")
    check(call("/mine/sets/pick", "DELETE")[0] == 204, "a set can be deleted")
    check(call("/mine/images/remove", "POST", {"ids": [image_id]})[1]["removed"] == 1 and call("/mine/images")[1]["total"] == 3, "an image can be removed from the dataset")
    check(call("/mine", "DELETE")[0] == 204 and call("")[1]["datasets"] == [] and (pics / "p0.png").is_file(), "deleting a dataset keeps the image files")
    finish()


if __name__ == "__main__":
    main()
