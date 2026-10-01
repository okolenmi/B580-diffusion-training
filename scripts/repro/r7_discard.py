import sys, json, sqlite3, tempfile
from pathlib import Path
sys.path.insert(0,__import__("os").environ.get("REPO","."))
import torch
from backend.infrastructure.dataset_library import SqliteDatasetLibrary
from backend.infrastructure.workspace import WorkspaceLayout
from manager.db import add_shard, add_source
from manager.storage import ShardWriter

with tempfile.TemporaryDirectory() as tmp:
    tmp=Path(tmp); layout=WorkspaceLayout(tmp, runs_dir=tmp/"runs", settings_kv=lambda k,d: str(tmp/"datasets") if k=="datasets_dir" else d)
    lib=SqliteDatasetLibrary(layout); lib.create("ds")
    root=tmp/"datasets"/"ds"; db=root/"metadata.db"
    src=add_source(db,"s","real")
    ids=[]
    for i in range(2):                                   # two shards, one trajectory each
        sf=root/"shards"/f"s{i}.safetensors"; w=ShardWriter(sf); idx=w.add_image_latent(torch.randn(1,4,8,8)); c,sz=w.write()
        sid=add_shard(db,str(sf.relative_to(root)),c,sz)
        conn=sqlite3.connect(db); conn.execute("INSERT INTO trajectories (source_id,shard_id,shard_index,sample_count,seed,prompt,latent_h,latent_w) VALUES (?,?,?,?,?,?,?,?)",(src,sid,idx,1,i,"p",8,8)); conn.commit(); conn.close()
    n_items=len(lib.list_items("ds")); files_before=sorted(p.name for p in (root/"shards").glob("*.safetensors"))
    # the 2nd file removal fails (permissions / file in use / antivirus on Windows)
    real=Path.unlink; calls={"n":0}
    def flaky(self,*a,**k):
        if self.suffix==".safetensors":
            calls["n"]+=1
            if calls["n"]==2: raise PermissionError("locked")
        return real(self,*a,**k)
    Path.unlink=flaky
    try: lib.discard("ds",[i.id for i in lib.list_items("ds")])
    except Exception as e: print("discard raised:",type(e).__name__)
    finally: Path.unlink=real
    print("trajectory rows after failed discard:",len(lib.list_items("ds")),"(rolled back, was",n_items,")")
    print("shard files before:",files_before,"| after:",sorted(p.name for p in (root/"shards").glob("*.safetensors")))
    print("=> dataset rows reference a shard file that no longer exists" if len(lib.list_items("ds"))==n_items and len(list((root/"shards").glob("*.safetensors")))<2 else "not reproduced")
