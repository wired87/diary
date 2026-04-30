"""
maps.py – Location data collection, batch embedding, and DuckDB persistence.

Workflow
--------
1.  A background thread continuously reads location data at a configurable
    interval (default: 5 s).
2.  Collected entries are buffered; when the batch reaches BATCH_SIZE they are
    embedded and flushed to DuckDB (scan_history/pathways.duckdb).
3.  The DuckDB ``maps`` table is created when it does not exist.
4.  The FastMCP route registry (routes_meta.json) is checked; if the "maps"
    entry is absent it is added so the FastMCP server exposes it as a tool.
5.  ``get_embedded_rows()`` returns all rows stored in the maps table,
    including the stored embedding vectors parsed back to list[float].

Usage (continuous background mode)
------------------------------------
    from maps import start_maps_collection, stop_maps_collection, get_embedded_rows

    collector = start_maps_collection()   # starts background thread
    ...
    rows = get_embedded_rows()            # query maps table at any time
    stop_maps_collection()

Usage (finite pipeline run)
-----------------------------
    from maps import run_maps_pipeline

    rows = run_maps_pipeline(iterations=20, collect_interval=1.0, batch_size=5)
"""

from __future__ import annotations

import json
import os
import threading
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from typing import Any, Optional

import duckdb

from embeddings import embed_to_list

# ---------------------------------------------------------------------------
# Paths (must match pipeline.py / server.py)
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
SCAN_HISTORY_DIR = os.path.join(_HERE, "scan_history")
DB_PATH = os.path.join(SCAN_HISTORY_DIR, "pathways.duckdb")
ROUTES_META_JSON = os.path.join(SCAN_HISTORY_DIR, "routes_meta.json")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MAPS_TABLE: str = "maps"
BATCH_SIZE: int = 10
COLLECT_INTERVAL: float = 5.0   # seconds between location reads
_GEO_API_URL: str = "http://ip-api.com/json/"
_GEO_TIMEOUT: float = 3.0       # seconds
_IP_GEOLOCATION_ACCURACY_M: float = 5_000.0  # typical IP-geolocation accuracy in metres
_THREAD_JOIN_TIMEOUT_BUFFER: float = 5.0     # extra seconds added to collect_interval on join


# ---------------------------------------------------------------------------
# Location data model
# ---------------------------------------------------------------------------

@dataclass
class LocationData:
    """All metadata for a single location reading."""

    time_attr: float
    latitude: float
    longitude: float
    accuracy: float
    source: str
    address: str
    city: str
    country: str
    # text embedding stored as list[float] for JSON serialisation
    embedding: list[float] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Location reader
# ---------------------------------------------------------------------------

def _fetch_ip_location() -> Optional[dict[str, Any]]:
    """Query ip-api.com for the machine's approximate geolocation.

    Returns the parsed JSON dict on success, or *None* when the request
    fails, times out, or returns a non-success status.
    """
    try:
        req = urllib.request.Request(
            _GEO_API_URL,
            headers={"User-Agent": "diary-maps/1.0"},
        )
        with urllib.request.urlopen(req, timeout=_GEO_TIMEOUT) as resp:
            if resp.status == 200:
                return json.loads(resp.read().decode("utf-8"))
    except Exception:  # noqa: BLE001
        pass
    return None


def collect_location() -> LocationData:
    """Read the current location.

    Tries IP-based geolocation (ip-api.com) first; falls back to a
    placeholder entry when no network access is available or the
    service is unreachable.
    """
    ts = time.time()
    data = _fetch_ip_location()

    if data and data.get("status") == "success":
        lat = float(data.get("lat", 0.0))
        lon = float(data.get("lon", 0.0))
        city = data.get("city", "")
        country = data.get("country", "")
        region = data.get("regionName", "")
        address = ", ".join(part for part in [region, country] if part)
        source = "ip-api"
        accuracy = _IP_GEOLOCATION_ACCURACY_M
    else:
        lat, lon = 0.0, 0.0
        city = ""
        country = ""
        address = "unknown"
        source = "fallback"
        accuracy = 0.0

    text = f"{lat},{lon} {address} {city} {country}"
    embedding = embed_to_list(text)

    return LocationData(
        time_attr=ts,
        latitude=lat,
        longitude=lon,
        accuracy=accuracy,
        source=source,
        address=address,
        city=city,
        country=country,
        embedding=embedding,
    )


# ---------------------------------------------------------------------------
# DuckDB helpers
# ---------------------------------------------------------------------------

def _ensure_maps_table(conn: duckdb.DuckDBPyConnection) -> None:
    """Create the maps table if it does not already exist."""
    conn.execute(
        f"""
        CREATE TABLE IF NOT EXISTS "{MAPS_TABLE}" (
            time_attr   DOUBLE,
            latitude    DOUBLE,
            longitude   DOUBLE,
            accuracy    DOUBLE,
            source      VARCHAR,
            address     VARCHAR,
            city        VARCHAR,
            country     VARCHAR,
            embedding   VARCHAR,
            PRIMARY KEY (time_attr)
        )
        """
    )


def _insert_batch(
    conn: duckdb.DuckDBPyConnection,
    batch: list[LocationData],
) -> None:
    """Insert *batch* into the maps table, skipping duplicate timestamps."""
    for loc in batch:
        conn.execute(
            f"""
            INSERT INTO "{MAPS_TABLE}"
                (time_attr, latitude, longitude, accuracy, source,
                 address, city, country, embedding)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT DO NOTHING
            """,
            (
                loc.time_attr,
                loc.latitude,
                loc.longitude,
                loc.accuracy,
                loc.source,
                loc.address,
                loc.city,
                loc.country,
                json.dumps(loc.embedding),
            ),
        )


def _flush_batch(batch: list[LocationData]) -> None:
    """Open DuckDB, ensure the maps table exists, and persist *batch*."""
    os.makedirs(SCAN_HISTORY_DIR, exist_ok=True)
    conn = duckdb.connect(DB_PATH)
    try:
        _ensure_maps_table(conn)
        _insert_batch(conn, batch)
    finally:
        conn.close()
    print(f"[maps] Flushed {len(batch)} location entries → {DB_PATH}")


# ---------------------------------------------------------------------------
# FastMCP route registry
# ---------------------------------------------------------------------------

def _load_routes_meta() -> dict[str, list[float]]:
    if not os.path.exists(ROUTES_META_JSON):
        return {}
    with open(ROUTES_META_JSON, encoding="utf-8") as fh:
        return json.load(fh)


def _save_routes_meta(meta: dict[str, list[float]]) -> None:
    os.makedirs(SCAN_HISTORY_DIR, exist_ok=True)
    with open(ROUTES_META_JSON, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)


def ensure_maps_route() -> bool:
    """Check if the FastMCP 'maps' route exists; register it if not.

    Returns *True* when the route was newly added, *False* when it was
    already present.  The embedding stored in routes_meta.json is the
    character-3gram hash-vector of the string ``"maps"``, consistent with
    how all other table embeddings are registered.
    """
    meta = _load_routes_meta()
    if MAPS_TABLE in meta:
        print(f"[maps] FastMCP route '{MAPS_TABLE}' already registered.")
        return False
    meta[MAPS_TABLE] = embed_to_list(MAPS_TABLE)
    _save_routes_meta(meta)
    print(f"[maps] Registered FastMCP route '{MAPS_TABLE}' in routes_meta.json")
    return True


# ---------------------------------------------------------------------------
# Query helpers
# ---------------------------------------------------------------------------

def get_embedded_rows() -> list[dict[str, Any]]:
    """Return all rows from the maps table with embeddings as list[float].

    If the table or database does not yet exist, an empty list is returned.
    """
    if not os.path.exists(DB_PATH):
        return []
    conn = duckdb.connect(DB_PATH)
    try:
        rows = conn.execute(f'SELECT * FROM "{MAPS_TABLE}"').fetchall()
        cols = [d[0] for d in conn.description]
    except duckdb.CatalogException:
        return []
    finally:
        conn.close()

    result: list[dict[str, Any]] = []
    for row in rows:
        d = dict(zip(cols, row))
        # Parse the JSON-encoded embedding string back to list[float]
        raw_emb = d.get("embedding", "[]")
        try:
            d["embedding"] = json.loads(raw_emb) if isinstance(raw_emb, str) else raw_emb
        except (json.JSONDecodeError, TypeError):
            d["embedding"] = []
        result.append(d)
    return result


# ---------------------------------------------------------------------------
# Batch-streaming collector (background thread)
# ---------------------------------------------------------------------------

class MapsCollector:
    """Reads location data in a daemon thread, flushing to DuckDB in batches.

    Args:
        batch_size:        Number of entries to accumulate before flushing.
        collect_interval:  Seconds to wait between individual reads.
    """

    def __init__(
        self,
        batch_size: int = BATCH_SIZE,
        collect_interval: float = COLLECT_INTERVAL,
    ) -> None:
        self.batch_size = batch_size
        self.collect_interval = collect_interval
        self._batch: list[LocationData] = []
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None

    # ------------------------------------------------------------------
    # Public control interface

    def start(self) -> None:
        """Start the background collection thread (idempotent)."""
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run,
            name="MapsCollectorThread",
            daemon=True,
        )
        self._thread.start()
        print("[maps] MapsCollector started.")

    def stop(self, flush_remaining: bool = True) -> None:
        """Signal the thread to stop and optionally flush any buffered entries."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=self.collect_interval + _THREAD_JOIN_TIMEOUT_BUFFER)
        if flush_remaining:
            with self._lock:
                if self._batch:
                    _flush_batch(self._batch)
                    self._batch = []
        print("[maps] MapsCollector stopped.")

    def get_collected(self) -> list[LocationData]:
        """Return a snapshot of the current unflushed buffer."""
        with self._lock:
            return list(self._batch)

    # ------------------------------------------------------------------
    # Internal loop

    def _run(self) -> None:
        """Main thread loop: collect → embed → buffer → batch-flush."""
        iteration = 0
        pending_flush: list[LocationData] = []
        while not self._stop_event.is_set():
            loc = collect_location()
            with self._lock:
                self._batch.append(loc)
                if len(self._batch) >= self.batch_size:
                    pending_flush = self._batch[:]
                    self._batch = []

            print(f"[maps] iter={iteration}  src={loc.source}  city={loc.city}")
            if pending_flush:
                _flush_batch(pending_flush)
                pending_flush = []

            iteration += 1
            self._stop_event.wait(timeout=self.collect_interval)


# ---------------------------------------------------------------------------
# Module-level singleton for continuous background collection
# ---------------------------------------------------------------------------

_collector: Optional[MapsCollector] = None


def start_maps_collection(
    batch_size: int = BATCH_SIZE,
    collect_interval: float = COLLECT_INTERVAL,
) -> MapsCollector:
    """Start (or return the existing) background :class:`MapsCollector`.

    Also ensures the FastMCP route is registered before starting.
    """
    global _collector
    ensure_maps_route()
    if _collector is None or not (
        _collector._thread and _collector._thread.is_alive()
    ):
        _collector = MapsCollector(
            batch_size=batch_size,
            collect_interval=collect_interval,
        )
        _collector.start()
    return _collector


def stop_maps_collection() -> None:
    """Stop the background :class:`MapsCollector` if one is running."""
    global _collector
    if _collector is not None:
        _collector.stop()
        _collector = None


# ---------------------------------------------------------------------------
# Convenience finite-run orchestrator
# ---------------------------------------------------------------------------

def run_maps_pipeline(
    iterations: int = 10,
    collect_interval: float = COLLECT_INTERVAL,
    batch_size: int = BATCH_SIZE,
) -> list[dict[str, Any]]:
    """Collect *iterations* location entries, flush to DuckDB, ensure the
    FastMCP route exists, and return all embedded rows from the maps table.

    This is a synchronous, finite-run entry-point.  For continuous background
    collection use :func:`start_maps_collection` instead.

    Args:
        iterations:       Number of location reads to perform.
        collect_interval: Seconds to wait between reads (skipped on last).
        batch_size:       Entries to accumulate before flushing to DuckDB.

    Returns:
        All rows currently stored in the maps table (each row includes the
        embedding as ``list[float]``).
    """
    # 1) Ensure the FastMCP route is registered
    ensure_maps_route()

    # 2) Collect synchronously in batches inside a dedicated thread so the
    #    caller's thread is not blocked (joins after completion).
    collected: list[LocationData] = []
    errors: list[BaseException] = []

    def _collect_loop() -> None:
        batch: list[LocationData] = []
        for i in range(iterations):
            loc = collect_location()
            print(f"[maps] iter={i}  src={loc.source}  city={loc.city}")
            collected.append(loc)
            batch.append(loc)
            if len(batch) >= batch_size:
                try:
                    _flush_batch(batch)
                except Exception as exc:  # noqa: BLE001
                    errors.append(exc)
                batch = []
            if i < iterations - 1:
                time.sleep(collect_interval)
        if batch:
            try:
                _flush_batch(batch)
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

    t = threading.Thread(target=_collect_loop, name="MapsPipelineThread", daemon=True)
    t.start()
    t.join()

    if errors:
        print(f"[maps] {len(errors)} flush error(s) during pipeline run.")

    # 3) Return all embedded rows from the maps table
    rows = get_embedded_rows()
    print(f"[maps] Pipeline complete. {len(rows)} total rows in maps table.")
    return rows
