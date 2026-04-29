"""
pipeline.py – full process-monitoring pipeline.

Workflow
--------
1.  Run 10 scan iterations (configurable); each iteration collects every PID
    on the machine together with its embedding and a ``time_attr`` timestamp.
2.  Build a Structure-of-Arrays (SOA) view segmented by ``time_attr``.
3.  Build CONTROLLER – for every (time-step, process) pair, find the index of
    the next time step where a process with cosine-similarity > THRESHOLD_N
    exists.  For the very first time-step the search is within that snapshot.
4.  Build PATHWAYS – a dict keyed by a short process-name tag; each value is
    the chain of process-dicts across time steps (or None where the chain
    breaks).
5.  Persist snapshots, SOA, CONTROLLER, and PATHWAYS to
    ``scan_history/scan_history.json``.
6.  Synchronise PATHWAYS to DuckDB / FastAPI route registry.
"""

from __future__ import annotations

import json
import os
import time
from pprint import pprint
from typing import Optional

import duckdb
import numpy as np

from embeddings import best_match, cosine_similarity, embed, embed_to_list
from process_scanner import scan_all_pids

# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------
THRESHOLD_N: float = 0.5   # CONTROLLER: similarity to next time step
THRESHOLD_J: float = 0.70  # pipeline → FastAPI-route docstring match
THRESHOLD_K: float = 0.85  # pipeline → DuckDB table-name match (strict: same name only)

# Maximum number of pathways to persist (top processes by memory)
MAX_PATHWAYS: int = 50

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCAN_HISTORY_DIR = os.path.join(os.path.dirname(__file__), "scan_history")
HISTORY_JSON = os.path.join(SCAN_HISTORY_DIR, "scan_history.json")
ROUTES_META_JSON = os.path.join(SCAN_HISTORY_DIR, "routes_meta.json")
DB_PATH = os.path.join(SCAN_HISTORY_DIR, "pathways.duckdb")


# ===========================================================================
# Step 1 – scan loop
# ===========================================================================

def run_scan_loop(iterations: int = 10, delay: float = 1.0) -> list[dict]:
    """Collect process snapshots for *iterations* steps.

    Each entry in the returned list is::

        {
            "time_attr": float,   # Unix timestamp
            "iteration": int,
            "processes": list[dict]  # each dict has all ProcessInfo fields
                                     # plus "time_attr" set to snapshot time
        }

    Results are pprint-ed as they arrive.
    """
    snapshots: list[dict] = []
    for i in range(iterations):
        ts = time.time()
        processes = scan_all_pids()
        for p in processes:
            p["time_attr"] = ts  # stamp each entry with the snapshot time

        snapshot = {"time_attr": ts, "iteration": i, "processes": processes}
        snapshots.append(snapshot)

        pprint(
            {
                "iteration": i,
                "time_attr": ts,
                "num_processes": len(processes),
                "top_process": processes[0]["name"] if processes else None,
            }
        )
        if i < iterations - 1:
            time.sleep(delay)

    return snapshots


# ===========================================================================
# Step 2 – SOA segmentation
# ===========================================================================

def build_soa(snapshots: list[dict]) -> dict:
    """Segment all entries by time_attr value (Structure-of-Arrays view).

    Returns::

        {
            "time_attrs":    list[float],
            "iterations":    list[int],
            "process_lists": list[list[dict]]
        }
    """
    return {
        "time_attrs": [s["time_attr"] for s in snapshots],
        "iterations": [s["iteration"] for s in snapshots],
        "process_lists": [s["processes"] for s in snapshots],
    }


# ===========================================================================
# Step 3 – CONTROLLER
# ===========================================================================

def build_controller(
    soa: dict, threshold: float = THRESHOLD_N
) -> list[list[Optional[int]]]:
    """Build CONTROLLER: list[list[int | None]].

    For every (time-step t, process p) pair find the index of the *next*
    time step t' > t where at least one process has cosine-similarity > threshold
    with p.  For t == 0 the search also considers the current time step (within
    t=0 entries) so that the highest-similarity neighbour is always captured.

    CONTROLLER[t][p_idx] = t' index, or None if no match was found.
    """
    process_lists: list[list[dict]] = soa["process_lists"]
    n = len(process_lists)
    controller: list[list[Optional[int]]] = []

    for t, procs_t in enumerate(process_lists):
        step: list[Optional[int]] = []

        for p_idx, proc in enumerate(procs_t):
            emb_p = proc.get("embedding")
            if not emb_p:
                step.append(None)
                continue

            emb_p = np.asarray(emb_p, dtype=np.float32)
            found_t: Optional[int] = None

            # For t=0, also search within t=0 excluding the process itself
            search_start = t if t > 0 else 0

            for t2 in range(search_start, n):
                for q_idx, proc_q in enumerate(process_lists[t2]):
                    if t2 == t and q_idx == p_idx:
                        continue  # skip self
                    emb_q = proc_q.get("embedding")
                    if not emb_q:
                        continue
                    score = cosine_similarity(emb_p, np.asarray(emb_q, dtype=np.float32))
                    if score > threshold:
                        found_t = t2
                        break
                if found_t is not None:
                    break

            step.append(found_t)

        controller.append(step)

    return controller


# ===========================================================================
# Step 4 – PATHWAYS
# ===========================================================================

def build_pathways(
    soa: dict, controller: list[list[Optional[int]]]
) -> dict[str, list[Optional[dict]]]:
    """Build PATHWAYS from CONTROLLER.

    Key: ``<process_name>_<pid>`` (short, describes the underlying SOA entry).
    Value: list of process-dicts across time steps, with None where the chain
    breaks.  One pathway is created for every process in time-step 0, capped
    at MAX_PATHWAYS (top memory consumers) to keep the system tractable.
    """
    process_lists: list[list[dict]] = soa["process_lists"]
    n = len(process_lists)

    pathways: dict[str, list[Optional[dict]]] = {}
    if not process_lists:
        return pathways

    # Cap to top MAX_PATHWAYS processes (already sorted by memory desc)
    seed_processes = process_lists[0][:MAX_PATHWAYS]

    for p_idx, proc0 in enumerate(seed_processes):
        key = f"{proc0.get('name', 'unknown')}_{proc0.get('pid', p_idx)}"

        pathway: list[Optional[dict]] = [proc0]
        current_t = 0
        current_p_idx = p_idx

        while current_t < n - 1:
            ctrl_row = controller[current_t] if current_t < len(controller) else []
            next_t: Optional[int] = (
                ctrl_row[current_p_idx]
                if current_p_idx < len(ctrl_row)
                else None
            )

            if next_t is None or next_t <= current_t:
                # Append None placeholders for remaining missing steps
                pathway.extend([None] * (n - 1 - current_t))
                break

            # Walk directly to next_t; fill gaps with None
            for gap_t in range(current_t + 1, next_t):
                pathway.append(None)

            # Find best matching process at next_t
            target_procs = process_lists[next_t]
            if not target_procs:
                pathway.append(None)
            else:
                emb_cur = proc0.get("embedding") or []
                emb_cur = np.asarray(emb_cur, dtype=np.float32)
                best_proc = max(
                    target_procs,
                    key=lambda q: cosine_similarity(
                        emb_cur,
                        np.asarray(q.get("embedding") or [], dtype=np.float32),
                    ),
                )
                pathway.append(best_proc)
                current_p_idx = target_procs.index(best_proc)

            current_t = next_t

        pathways[key] = pathway

    return pathways


# ===========================================================================
# Step 5 – JSON persistence
# ===========================================================================

def _serialisable(obj):
    """Recursively convert numpy types to Python scalars for JSON."""
    if isinstance(obj, dict):
        return {k: _serialisable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_serialisable(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def save_history(
    snapshots: list[dict],
    soa: dict,
    controller: list,
    pathways: dict,
) -> None:
    """Write all pipeline components to ``scan_history/scan_history.json``."""
    os.makedirs(SCAN_HISTORY_DIR, exist_ok=True)
    payload = {
        "snapshots": snapshots,
        "soa": soa,
        "controller": controller,
        "pathways": pathways,
    }
    with open(HISTORY_JSON, "w", encoding="utf-8") as fh:
        json.dump(_serialisable(payload), fh, indent=2)
    print(f"[pipeline] Saved scan history → {HISTORY_JSON}")


# ===========================================================================
# Step 6 – DuckDB / FastAPI route sync
# ===========================================================================

def _safe_table_name(raw: str) -> str:
    """Sanitise a string for use as a DuckDB table / FastAPI route name."""
    import re
    clean = re.sub(r"[^a-zA-Z0-9_]", "_", raw)
    if clean and clean[0].isdigit():
        clean = "t_" + clean
    return clean[:64]  # cap length


def _load_routes_meta() -> dict[str, list[float]]:
    """Load route-name → embedding mapping from routes_meta.json."""
    if not os.path.exists(ROUTES_META_JSON):
        return {}
    with open(ROUTES_META_JSON, encoding="utf-8") as fh:
        return json.load(fh)


def _save_routes_meta(meta: dict[str, list[float]]) -> None:
    os.makedirs(SCAN_HISTORY_DIR, exist_ok=True)
    with open(ROUTES_META_JSON, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)


def _get_duck_tables(conn: duckdb.DuckDBPyConnection) -> list[str]:
    rows = conn.execute("SHOW TABLES").fetchall()
    return [r[0] for r in rows]


def _ensure_pathway_table(
    conn: duckdb.DuckDBPyConnection, table: str
) -> None:
    conn.execute(
        f"""
        CREATE TABLE IF NOT EXISTS "{table}" (
            time_attr      DOUBLE,
            pid            INTEGER,
            name           VARCHAR,
            status         VARCHAR,
            cpu_percent    DOUBLE,
            memory_rss     BIGINT,
            memory_vms     BIGINT,
            memory_percent DOUBLE,
            num_threads    INTEGER,
            username       VARCHAR,
            exe            VARCHAR,
            cwd            VARCHAR,
            cmdline        VARCHAR,
            embedding      VARCHAR,
            PRIMARY KEY (time_attr, pid)
        )
        """
    )


def _upsert_pathway_entries(
    conn: duckdb.DuckDBPyConnection,
    table: str,
    entries: list[Optional[dict]],
) -> None:
    """Insert non-None pathway entries into *table*, skipping duplicates by (time_attr, pid)."""
    for entry in entries:
        if entry is None:
            continue
        conn.execute(
            f"""
            INSERT INTO "{table}"
                (time_attr, pid, name, status, cpu_percent,
                 memory_rss, memory_vms, memory_percent,
                 num_threads, username, exe, cwd, cmdline, embedding)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT DO NOTHING
            """,
            (
                entry.get("time_attr", 0.0),
                entry.get("pid"),
                entry.get("name", ""),
                entry.get("status", ""),
                entry.get("cpu_percent", 0.0),
                entry.get("memory_rss", 0),
                entry.get("memory_vms", 0),
                entry.get("memory_percent", 0.0),
                entry.get("num_threads", 0),
                entry.get("username", ""),
                entry.get("exe", ""),
                entry.get("cwd", ""),
                " ".join(entry.get("cmdline") or []),
                json.dumps(entry.get("embedding") or []),
            ),
        )


def sync_pathways_to_duckdb(pathways: dict, routes_meta: dict) -> None:
    """Upsert PATHWAYS entries into DuckDB and update the routes_meta registry.

    For each pathway key:
    * Perform similarity search against existing FastAPI route docstring
      embeddings (routes_meta).  If score > THRESHOLD_J → upsert to that
      table.
    * If no route match: check existing DuckDB table names.  If
      ss(table_name → key) > THRESHOLD_K → upsert to that table.
    * Otherwise create a new DuckDB table named after the sanitised key and
      register it in routes_meta so the FastMCP server can pick it up.
    """
    os.makedirs(SCAN_HISTORY_DIR, exist_ok=True)
    conn = duckdb.connect(DB_PATH)

    route_candidates = [(name, emb) for name, emb in routes_meta.items()]

    for raw_key, entries in pathways.items():
        key_emb = embed(raw_key)

        # 1) Match against known FastAPI route docstrings
        match = best_match(key_emb, route_candidates, THRESHOLD_J)
        if match:
            target_table = _safe_table_name(match[0])
            _ensure_pathway_table(conn, target_table)
            _upsert_pathway_entries(conn, target_table, entries)
            continue

        # 2) Match against existing DuckDB table names
        existing_tables = _get_duck_tables(conn)
        table_candidates = [(t, embed_to_list(t)) for t in existing_tables]
        match2 = best_match(key_emb, table_candidates, THRESHOLD_K)
        if match2:
            target_table = match2[0]
            _ensure_pathway_table(conn, target_table)
            _upsert_pathway_entries(conn, target_table, entries)
            continue

        # 3) Create new table
        new_table = _safe_table_name(raw_key)
        _ensure_pathway_table(conn, new_table)
        _upsert_pathway_entries(conn, new_table, entries)
        # Register in routes_meta so FastMCP can expose it as a tool
        if new_table not in routes_meta:
            routes_meta[new_table] = key_emb.tolist()
            route_candidates.append((new_table, key_emb.tolist()))

    conn.close()
    _save_routes_meta(routes_meta)
    print(f"[pipeline] DuckDB sync complete → {DB_PATH}")


def ensure_fastmcp_tools_for_new_tables(routes_meta: dict) -> None:
    """For every DuckDB table not yet in routes_meta, add an entry.

    The FastMCP server reads routes_meta at startup to register tools; this
    function ensures all tables are covered even if created outside the normal
    pathway sync flow.
    """
    if not os.path.exists(DB_PATH):
        return
    conn = duckdb.connect(DB_PATH)
    tables = _get_duck_tables(conn)
    conn.close()

    updated = False
    for table in tables:
        if table not in routes_meta:
            routes_meta[table] = embed_to_list(table)
            updated = True

    if updated:
        _save_routes_meta(routes_meta)


# ===========================================================================
# Orchestrator
# ===========================================================================

def run_full_pipeline(iterations: int = 10, scan_delay: float = 1.0) -> None:
    """Execute the complete pipeline end-to-end."""
    print("=" * 60)
    print("[pipeline] Starting process-scan loop …")
    print("=" * 60)

    # 1) Scan
    snapshots = run_scan_loop(iterations=iterations, delay=scan_delay)

    # 2) SOA
    soa = build_soa(snapshots)
    print(f"\n[pipeline] SOA built: {len(soa['time_attrs'])} time steps.")

    # 3) CONTROLLER
    print("[pipeline] Building CONTROLLER …")
    controller = build_controller(soa)
    print("\n[pipeline] CONTROLLER entries:")
    pprint(controller)

    # 4) PATHWAYS
    print("\n[pipeline] Building PATHWAYS …")
    pathways = build_pathways(soa, controller)
    print(f"[pipeline] {len(pathways)} pathways created.")

    # 5) Persist all components
    save_history(snapshots, soa, controller, pathways)

    # 6) Sync to DuckDB / FastAPI registry
    routes_meta = _load_routes_meta()
    sync_pathways_to_duckdb(pathways, routes_meta)
    routes_meta = _load_routes_meta()  # reload after sync
    ensure_fastmcp_tools_for_new_tables(routes_meta)

    print("\n[pipeline] Done.")
