"""N4-01: every GET /api/v1/installer/readiness spawns its own torch subprocess; nothing caches or limits them."""
import os, sys, threading, time, subprocess
sys.path.insert(0, os.environ.get("REPO","."))
import backend.application.ports.environment as E
probe_cls = [E.TorchDeviceProbe]
print("probe classes:", [c.__name__ for c in probe_cls])
P = probe_cls[0]
p = P()
peak = {"n": 0}; stop = threading.Event()
def watch():
    while not stop.is_set():
        out = subprocess.run(["pgrep","-fc","import torch|torch"],capture_output=True,text=True).stdout.strip()
        try: peak["n"] = max(peak["n"], int(out))
        except ValueError: pass
        time.sleep(0.1)
w = threading.Thread(target=watch); w.start()
fn = p.report
t0=time.time(); ths=[threading.Thread(target=fn) for _ in range(8)]
[t.start() for t in ths]; [t.join() for t in ths]; dt=time.time()-t0
stop.set(); w.join()
print(f"8 concurrent probes finished in {dt:.1f}s; peak simultaneous torch-importing processes seen: {peak['n']}")
