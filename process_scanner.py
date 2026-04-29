"""
process_scanner.py – scan every available PID and return a uniform list[dict].

Each dict represents one process with all available metadata plus a numpy
embedding derived from the process name and command line (for downstream
similarity operations).  Results are sorted by RSS memory usage descending
so the heaviest consumers appear first (activity classification).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Optional

import psutil

from embeddings import embed_to_list


@dataclass
class ProcessInfo:
    """All metadata for a single running process."""

    pid: int
    name: str
    status: str
    cpu_percent: float
    memory_rss: int
    memory_vms: int
    memory_percent: float
    cmdline: list[str]
    username: str
    create_time: float
    num_threads: int
    num_fds: int
    connections: list[dict]
    open_files: list[str]
    exe: str
    cwd: str
    # text embedding stored as list[float] for JSON serialisation
    embedding: list[float] = field(default_factory=list)
    # filled in by pipeline for each snapshot
    time_attr: float = 0.0


def _collect(proc: psutil.Process) -> Optional[ProcessInfo]:
    """Attempt to collect all available information for *proc*.

    Returns None when the process has vanished or access is denied for the
    critical fields (pid, name, status).
    """
    try:
        with proc.oneshot():
            pid = proc.pid
            name = proc.name()
            status = proc.status()
            cpu_percent = proc.cpu_percent()

            try:
                mem = proc.memory_info()
                memory_rss = mem.rss
                memory_vms = mem.vms
            except (psutil.AccessDenied, psutil.NoSuchProcess):
                memory_rss = 0
                memory_vms = 0

            try:
                memory_percent = proc.memory_percent()
            except (psutil.AccessDenied, psutil.NoSuchProcess):
                memory_percent = 0.0

            try:
                cmdline = proc.cmdline()
            except (psutil.AccessDenied, psutil.NoSuchProcess):
                cmdline = []

            try:
                username = proc.username()
            except (psutil.AccessDenied, psutil.NoSuchProcess):
                username = ""

            try:
                create_time = proc.create_time()
            except (psutil.AccessDenied, psutil.NoSuchProcess):
                create_time = 0.0

            try:
                num_threads = proc.num_threads()
            except (psutil.AccessDenied, psutil.NoSuchProcess):
                num_threads = 0

            try:
                num_fds = proc.num_fds()
            except (psutil.AccessDenied, AttributeError, psutil.NoSuchProcess):
                num_fds = -1

            try:
                connections = [
                    dict(
                        fd=c.fd,
                        family=str(c.family),
                        type=str(c.type),
                        laddr=str(c.laddr),
                        raddr=str(c.raddr),
                        status=c.status,
                    )
                    for c in proc.net_connections()
                ]
            except (psutil.AccessDenied, psutil.NoSuchProcess, AttributeError):
                connections = []

            try:
                open_files = [f.path for f in proc.open_files()]
            except (psutil.AccessDenied, psutil.NoSuchProcess):
                open_files = []

            try:
                exe = proc.exe()
            except (psutil.AccessDenied, psutil.NoSuchProcess):
                exe = ""

            try:
                cwd = proc.cwd()
            except (psutil.AccessDenied, psutil.NoSuchProcess):
                cwd = ""

            text = f"{name} {exe} {' '.join(cmdline)}"
            embedding = embed_to_list(text)

        return ProcessInfo(
            pid=pid,
            name=name,
            status=status,
            cpu_percent=cpu_percent,
            memory_rss=memory_rss,
            memory_vms=memory_vms,
            memory_percent=memory_percent,
            cmdline=cmdline,
            username=username,
            create_time=create_time,
            num_threads=num_threads,
            num_fds=num_fds,
            connections=connections,
            open_files=open_files,
            exe=exe,
            cwd=cwd,
            embedding=embedding,
        )
    except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
        return None


def scan_all_pids() -> list[dict]:
    """Return a list[dict] of all running processes, sorted by RSS desc.

    Each dict is the full ProcessInfo dataclass converted to a plain Python
    dict (JSON-serialisable) so it can be embedded in snapshots, stored in
    DuckDB, or written to JSON without further conversion.
    """
    results: list[dict] = []
    for proc in psutil.process_iter():
        info = _collect(proc)
        if info is not None:
            results.append(asdict(info))

    # Classify by activity: highest memory consumers first
    results.sort(key=lambda x: x.get("memory_rss", 0), reverse=True)
    return results
