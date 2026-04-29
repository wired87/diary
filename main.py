"""
main.py – entry point.

Execution order
---------------
1.  Run the full process-scanning pipeline (10 iterations by default).
2.  Launch the FastMCP server (server.py) as a subprocess on port 8000.
"""

from __future__ import annotations

import subprocess
import sys
import os

from pipeline import run_full_pipeline


def start_server() -> subprocess.Popen:
    server_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "server.py")
    proc = subprocess.Popen(
        [sys.executable, server_path],
        stdout=sys.stdout,
        stderr=sys.stderr,
    )
    return proc


if __name__ == "__main__":
    run_full_pipeline(iterations=10, scan_delay=1.0)
    print("\n[main] Starting FastMCP server …")
    server_proc = start_server()
    print(f"[main] FastMCP server PID={server_proc.pid}  →  http://localhost:8000/mcp")
    server_proc.wait()
