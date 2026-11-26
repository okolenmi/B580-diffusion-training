"""A real backend server in a child process, for browser tests: its own database, scratch folders and a fake ComfyUI models folder.

It runs the tiny CPU profile (``tiny_profile/sitecustomize.py``), so graphs that name ``TinyModelSource`` really run, in a graph child process,
in seconds. Nothing of the developer's own server data is touched; the dataset store of the project (``datasets/``, not tracked) is the one
shared place, so datasets made here carry a ``zz_browser_`` prefix and are removed on stop.
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROFILE = Path(__file__).resolve().parent / "tiny_profile"
DATASET_PREFIX = "zz_browser_"


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class ApiError(RuntimeError):
    def __init__(self, status: int, body):
        super().__init__(f"{status}: {body}")
        self.status, self.body = status, body


class TestServer:
    """``with TestServer() as server:`` gives ``server.url`` and ``server.api(method, path, body)``; ``restart()`` keeps the database."""

    __test__ = False  # not a pytest class

    def __init__(self, *, startup_seconds: float = 90.0):
        self.startup_seconds = startup_seconds
        self.home = Path(tempfile.mkdtemp(prefix="browser-test-"))
        self.comfy = self.home / "comfy"
        for sub in ("checkpoints", "loras", "vae", "diffusion_models", "text_encoders"):
            (self.comfy / "models" / sub).mkdir(parents=True)
        (self.comfy / "custom_nodes").mkdir()
        self.port = free_port()
        self.process: subprocess.Popen | None = None
        self.log_path = self.home / "server.log"
        self._artifacts_before: set[str] = set()

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def environment(self) -> dict:
        env = dict(os.environ)
        env.update({
            "COMFY_DIR": str(self.comfy), "B580_TINY_PROFILE": "1", "BACKEND_DB_PATH": str(self.home / "backend.db"),
            "PYTHONPATH": os.pathsep.join([str(PROFILE), str(ROOT), env.get("PYTHONPATH", "")]).rstrip(os.pathsep),
            "PYTHONUNBUFFERED": "1",
        })
        env.pop("VENV_PYTHON", None)
        return env

    def start(self) -> None:
        folder = ROOT / "artifacts"  # run products land in the project's artifacts folder; what this server adds is removed on stop
        if not self.process and folder.is_dir():
            self._artifacts_before = {p.name for p in folder.iterdir()}
        log = open(self.log_path, "ab")
        self.process = subprocess.Popen([sys.executable, "-m", "backend.cli", "--port", str(self.port), "--db", str(self.home / "backend.db")],
                                        cwd=ROOT, env=self.environment(), stdout=log, stderr=subprocess.STDOUT,
                                        start_new_session=True)  # one process group: the graph children go down with the server
        deadline = time.monotonic() + self.startup_seconds
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError(f"the server stopped at start:\n{self.log_tail()}")
            try:
                self.api("GET", "/api/v1/health")
                return
            except (OSError, ApiError):
                time.sleep(0.3)
        raise RuntimeError(f"the server did not answer in {self.startup_seconds:.0f} s:\n{self.log_tail()}")

    def stop(self) -> None:
        if self.process:
            for sig in (signal.SIGTERM, signal.SIGKILL):
                try:
                    os.killpg(self.process.pid, sig)
                except ProcessLookupError:
                    break
                try:
                    self.process.wait(10)
                    break
                except subprocess.TimeoutExpired:
                    continue
            try:
                os.killpg(self.process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        self._kill_graph_children()
        self.process = None

    def _kill_graph_children(self) -> None:
        """Graph children live in sessions of their own (a run outlives a restarted server), so they are found by the folder they work in."""
        marker = str(self.home)
        for line in subprocess.run(["ps", "-eo", "pid,args"], capture_output=True, text=True).stdout.splitlines()[1:]:
            pid, _, args = line.strip().partition(" ")
            if "graph_task_worker" in args and marker in args:
                try:
                    os.kill(int(pid), signal.SIGKILL)
                except ProcessLookupError:
                    pass

    def restart(self) -> None:
        self.stop()
        self.start()

    def log_tail(self, lines: int = 25) -> str:
        try:
            return "\n".join(self.log_path.read_text(errors="replace").splitlines()[-lines:])
        except OSError:
            return ""

    def api(self, method: str, path: str, body=None, *, timeout: float = 30.0):
        data = None if body is None else json.dumps(body).encode()
        request = urllib.request.Request(self.url + path, data=data, method=method, headers={"Content-Type": "application/json"} if data else {})
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                raw = response.read()
        except urllib.error.HTTPError as error:
            raw = error.read()
            try:
                parsed = json.loads(raw)
            except ValueError:
                parsed = raw.decode(errors="replace")
            raise ApiError(error.code, parsed) from None
        return json.loads(raw) if raw else None

    def wait_for(self, what: str, condition, timeout: float = 120.0, every: float = 0.5):
        """Poll ``condition()`` until it returns something truthy; the value is returned."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            value = condition()
            if value:
                return value
            time.sleep(every)
        raise TimeoutError(f"waited {timeout:.0f} s for {what}\n{self.log_tail(15)}")

    def __enter__(self) -> TestServer:
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop()
        datasets = ROOT / "datasets"
        if datasets.is_dir():
            for leftover in datasets.glob(DATASET_PREFIX + "*"):
                shutil.rmtree(leftover, ignore_errors=True)
        folder = ROOT / "artifacts"
        if folder.is_dir():
            for added in [p for p in folder.iterdir() if p.name not in self._artifacts_before]:
                shutil.rmtree(added, ignore_errors=True)
            if not self._artifacts_before and not any(folder.iterdir()):
                folder.rmdir()
        shutil.rmtree(self.home, ignore_errors=True)
