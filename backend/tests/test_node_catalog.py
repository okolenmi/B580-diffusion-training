"""Node catalog endpoints over the new node tree (rework/contracts/catalog.md and resolve.md).

Real composition root: the catalog is the document ``next/graph`` builds, and a resolve answers with the
shape the node declares. Run directly: python backend/tests/test_node_catalog.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.bootstrap import build_container
from backend.config import Settings
from backend.presentation.app import create_app
from backend.tests.support import asgi_request, check, finish

root = Path(tempfile.mkdtemp(prefix="backend-node-catalog-"))
app = create_app(build_container(Settings(project_root=root, db_path=root / "backend.db")).services)

status, _, doc = asgi_request(app, "/api/v1/catalog")
check(status == 200 and doc["catalog_format"] == 1, "the catalog is served in the contract's format")
names = {node["class_name"] for node in doc["nodes"]}
check({"LoadModelNode", "DiffusionTrainerNode", "InjectAdapterNode"} <= names, f"the nodes of the new tree are listed ({len(names)} classes)")
check(doc["load_errors"] == [], f"no module failed to load ({doc['load_errors']})")
check(any(f["id"] == "sdxl" for f in doc["families"]) and doc["settings"], "families and settings are part of the catalog")
trainer = next(n for n in doc["nodes"] if n["class_name"] == "DiffusionTrainerNode")
check(all("type_mro" in p and "group" in p for p in trainer["inputs"]) and trainer["groups"], "ports carry the type chain and their group")

status, _, again = asgi_request(app, "/api/v1/catalog")
check(status == 200 and again == doc, "a second request returns the same document")
status, _, fresh = asgi_request(app, "/api/v1/catalog?refresh=true")
check(status == 200 and {n["class_name"] for n in fresh["nodes"]} == names, "a refresh scans again and finds the same classes")

status, _, shape = asgi_request(app, "/api/v1/graphs/nodes/LoadModelNode/resolve", method="POST", json_body={"params": {"scope": "checkpoints", "file": "nothing.safetensors"}})
check(status == 200 and shape["state"] in ("error", "needs_choice", "ready") and "inputs" in shape and "outputs" in shape,
      f"a resolve answers with a state and a shape ({shape.get('state')}: {shape.get('message')})")
status, _, shape = asgi_request(app, "/api/v1/graphs/nodes/InjectAdapterNode/resolve", method="POST", json_body={"params": {}})
check(status == 200 and shape["state"] == "ready", "a node with a static shape resolves to ready")
status, _, body = asgi_request(app, "/api/v1/graphs/nodes/NoSuchNode/resolve", method="POST", json_body={"params": {}})
check(status == 404 and body["error"]["code"] == "node_class_not_found", "an unknown class is a 404 in the one error envelope")

finish()
