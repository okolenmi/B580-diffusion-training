"""Runinfo — persists training state across sessions."""


def write_runinfo(path: str, step: int, total_steps: int,
                  resume_path: str, opt_path: str):
    """Write training state to a key=value text file."""
    with open(path, "w") as f:
        f.write(f"step={step}\n")
        f.write(f"total_steps={total_steps}\n")
        f.write(f"resume={resume_path}\n")
        f.write(f"optstate={opt_path}\n")


def read_runinfo(path: str) -> dict:
    """Parse a runinfo file into a dict."""
    info = {}
    with open(path) as f:
        for line in f:
            if "=" in line:
                k, v = line.strip().split("=", 1)
                info[k] = v
    return info
