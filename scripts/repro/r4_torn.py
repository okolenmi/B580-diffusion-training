import sys, json, tempfile
from pathlib import Path
sys.path.insert(0,__import__("os").environ.get("REPO","."))
from backend.infrastructure.jsonl_progress_source import JsonlProgressSource
with tempfile.TemporaryDirectory() as tmp:
    p = Path(tmp)/"p.jsonl"; src = JsonlProgressSource()
    full = json.dumps({"phase":"step","step":42,"total":100,"loss":0.25,"avg":0.3,"lr":1e-4})+"\n"
    # the trainer's write is observed half-way (no newline yet)
    p.write_text(json.dumps({"phase":"step","step":41,"total":100,"loss":0.2})+"\n" + full[:30])
    a = src.read_new(p)
    print("read 1 (torn tail):", [(s.step) for s in a])
    with open(p,"a") as f: f.write(full[30:])         # rest of the line arrives
    b = src.read_new(p)
    print("read 2 (rest arrives):", [(s.step) for s in b], "  <- step 42 is gone for good" if not b else "")
    # NaN loss passes straight through
    with open(p,"a") as f: f.write('{"phase":"step","step":43,"total":100,"loss":NaN,"avg":NaN,"lr":1e-4}\n')
    c = src.read_new(p); print("NaN line parsed as loss =", c[0].loss)
