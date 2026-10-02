import sys, tempfile
from pathlib import Path
sys.path.insert(0,__import__("os").environ.get("REPO","."))
from backend.tests.support import build_services, asgi_request
from backend.presentation.app import create_app
from nodes.config_io import write_config, read_config
from nodes.config_model import TrainingConfig

with tempfile.TemporaryDirectory() as tmp:
    tmp=Path(tmp); proj=tmp/"proj"; (proj/"configs").mkdir(parents=True)
    cfg = proj/"configs"/"t.toml"
    write_config(cfg, TrainingConfig())          # a valid baseline file
    original = cfg.read_text()
    edited = "# MY NOTES: lr tuned on 2026-09-28, don't touch\n" + original.replace("\n[", "\n# section comment\n[",1) + "\n[experimental]\nmy_flag = true\n"
    svc = build_services(project_root=proj, runs_dir=tmp/"runs"); app = create_app(svc)
    st,_,b = asgi_request(app,"/api/v1/config/raw",method="PUT",json_body={"path":"configs/t.toml","content":edited})
    print("PUT /config/raw ->", st, b if st!=200 and st!=204 else "")
    after = cfg.read_text()
    print("saved text == what the user sent :", after == edited)
    print("user comment survived            :", "MY NOTES" in after)
    print("unknown [experimental] survived  :", "experimental" in after)
