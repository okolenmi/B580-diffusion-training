"""Process management — launch, signal, and cleanup training subprocesses."""

import os
import signal
import subprocess
from pathlib import Path


def launch_training_process(
    cmd: list[str],
    comfy_dir: Path,
    project_root: Path,
    log_path: Path,
) -> subprocess.Popen:
    """Launch a training subprocess with proper environment setup.

    Parameters
    ----------
    cmd : list[str]
        Command to execute (e.g., [python, "-m", "core.cli", ...]).
    comfy_dir : Path
        ComfyUI directory (working directory for the subprocess).
    project_root : Path
        Project root (parent of this project's directory).
    log_path : Path
        Path to write stdout logs.

    Returns
    -------
    subprocess.Popen
        The launched process.
    """
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONPATH"] = str(project_root) + os.pathsep + env.get("PYTHONPATH", "")

    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = open(log_path, "w", buffering=1)

    proc = subprocess.Popen(
        cmd,
        cwd=str(comfy_dir),
        env=env,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )

    return proc


def send_signal(proc: subprocess.Popen, force: bool = False) -> bool:
    """Send a signal to the process group.

    Parameters
    ----------
    proc : subprocess.Popen
        The training subprocess.
    force : bool
        If True, send SIGKILL (immediate, no cleanup).
        If False, send SIGINT (graceful).

    Returns
    -------
    bool
        True if signal was sent successfully.
    """
    if not proc or not proc.pid:
        return False

    try:
        pgid = os.getpgid(proc.pid)
        sig = signal.SIGKILL if force else signal.SIGINT
        os.killpg(pgid, sig)
        return True
    except (ProcessLookupError, PermissionError, OSError):
        # Try direct kill on the main process
        try:
            sig = signal.SIGKILL if force else signal.SIGINT
            os.kill(proc.pid, sig)
            return True
        except (ProcessLookupError, PermissionError, OSError):
            return False


def kill_process_by_pid(pid: int) -> bool:
    """Kill a process by PID, including its process group.

    Parameters
    ----------
    pid : int
        Process ID to kill.

    Returns
    -------
    bool
        True if process was killed, False if not found.
    """
    try:
        pgid = os.getpgid(pid)
        os.killpg(pgid, signal.SIGKILL)
        return True
    except (ProcessLookupError, PermissionError, OSError):
        try:
            os.kill(pid, signal.SIGKILL)
            return True
        except (ProcessLookupError, PermissionError, OSError):
            return False


def is_process_alive(proc: subprocess.Popen) -> bool:
    """Check if a subprocess is actually alive.

    Uses os.kill(pid, 0) to verify the OS process exists, since
    poll() may be unreliable for dummy objects.

    Parameters
    ----------
    proc : subprocess.Popen
        The subprocess to check.

    Returns
    -------
    bool
        True if the process is alive.
    """
    if proc is None:
        return False

    poll_result = proc.poll()
    if poll_result is not None:
        return False

    try:
        os.kill(proc.pid, 0)
        return True
    except (ProcessLookupError, PermissionError, OSError):
        return False


def cleanup_orphaned_runs(db_path: Path) -> int:
    """Find runs marked as 'running' in the DB and kill their processes.

    Called on server startup to clean up processes left behind when
    the server was restarted or crashed.

    Parameters
    ----------
    db_path : Path
        Path to the server database.

    Returns
    -------
    int
        Number of orphaned runs cleaned up.
    """
    import sqlite3

    from . import db

    try:
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT id, pid, total_steps FROM runs WHERE status='running'",
        ).fetchall()
        killed = 0
        for row in rows:
            pid = row["pid"]
            run_id = row["id"]
            if pid:
                if kill_process_by_pid(pid):
                    db.update_run_status(db_path, run_id, "killed",
                                         error_msg="Orphan cleanup on server startup")
                    killed += 1
                else:
                    # PID stored but process gone — mark as failed
                    db.update_run_status(db_path, run_id, "failed",
                                         error_msg="Process disappeared (orphan cleanup)")
                    killed += 1
            else:
                # No PID stored — can't kill, mark as failed
                db.update_run_status(db_path, run_id, "failed",
                                     error_msg="No PID stored (orphan cleanup)")
                killed += 1
        conn.close()
        return killed
    except Exception as e:
        print(f"  Warning: Failed to cleanup orphaned runs ({e})")
        return 0
