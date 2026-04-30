"""
maps.py – location data reader and DuckDB streamer.

Architecture
------------
1.  A background ``LocationReader`` thread polls for location fixes at a
    configurable interval and enqueues ``LocationPoint`` instances.
2.  The main ``run_location_pipeline()`` function drains the queue in
    configurable batches and writes each batch to a local DuckDB database.

Location sources (tried in order, first success wins per poll)
--------------------------------------------------------------
* IP-geolocation via ``ip-api.com`` (free, no key, requires internet).
* Simulated random-walk data (always available as fallback).

The DuckDB table ``location_readings`` is stored alongside the other
pipeline artefacts in ``scan_history/locations.duckdb``.
"""

from __future__ import annotations

import json
import math
import os
import queue
import random
import threading
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from typing import Optional

import duckdb

# ---------------------------------------------------------------------------
# Paths (mirrors pipeline.py conventions)
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
SCAN_HISTORY_DIR = os.path.join(_HERE, "scan_history")
DB_PATH = os.path.join(SCAN_HISTORY_DIR, "locations.duckdb")

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_POLL_INTERVAL: float = 5.0   # seconds between location reads
DEFAULT_BATCH_SIZE: int = 10         # flush to DuckDB after this many points
DEFAULT_FLUSH_INTERVAL: float = 15.0 # max seconds between flushes
DEFAULT_DURATION: float = 60.0       # total run time for run_location_pipeline()

_IP_API_URL = "http://ip-api.com/json/?fields=status,lat,lon,city,country,isp,query"
_IP_API_TIMEOUT = 5  # seconds


# ===========================================================================
# Data model
# ===========================================================================

@dataclass
class LocationPoint:
    """A single location fix."""

    timestamp: float           # Unix epoch (UTC)
    latitude: float
    longitude: float
    altitude: float = 0.0
    accuracy: float = 0.0      # metres, 0 = unknown
    speed: float = 0.0         # m/s, 0 = unknown
    heading: float = 0.0       # degrees [0, 360), 0 = unknown/North
    source: str = "unknown"    # "ip", "simulated"
    city: str = ""
    country: str = ""
    extra: str = ""            # JSON blob for additional provider fields


# ===========================================================================
# Location sources
# ===========================================================================

def _fetch_ip_location() -> Optional[LocationPoint]:
    """Try to obtain an approximate fix via ip-api.com.

    Returns a ``LocationPoint`` on success, ``None`` on any error (network
    unavailable, rate-limited, bad JSON, etc.).
    """
    try:
        req = urllib.request.Request(
            _IP_API_URL,
            headers={"User-Agent": "diary-maps/1.0"},
        )
        with urllib.request.urlopen(req, timeout=_IP_API_TIMEOUT) as resp:
            raw = resp.read().decode("utf-8", errors="ignore")
        data = json.loads(raw)
        if data.get("status") != "success":
            return None
        return LocationPoint(
            timestamp=time.time(),
            latitude=float(data["lat"]),
            longitude=float(data["lon"]),
            accuracy=5000.0,  # IP geolocation is city-level (~5 km)
            source="ip",
            city=data.get("city", ""),
            country=data.get("country", ""),
            extra=json.dumps({"isp": data.get("isp", ""), "ip": data.get("query", "")}),
        )
    except (urllib.error.URLError, OSError, json.JSONDecodeError, KeyError):
        return None


# ---------------------------------------------------------------------------
# Simulated random-walk source (always available as fallback)
# ---------------------------------------------------------------------------

class _RandomWalkState:
    """Mutable state for the simulated location random walk."""

    __slots__ = ("lat", "lon", "alt", "speed", "heading")

    def __init__(self) -> None:
        # Start somewhere in central Europe (Berlin) as a plausible default
        self.lat = 52.5200 + random.uniform(-0.5, 0.5)
        self.lon = 13.4050 + random.uniform(-0.5, 0.5)
        self.alt = random.uniform(30.0, 80.0)
        self.speed = random.uniform(0.0, 2.0)
        self.heading = random.uniform(0.0, 360.0)


_walk_state = _RandomWalkState()
_walk_lock = threading.Lock()


def _simulate_location() -> LocationPoint:
    """Generate a plausible location reading using a random walk."""
    with _walk_lock:
        # Small random updates to simulate slow movement
        _walk_state.heading = (_walk_state.heading + random.uniform(-15, 15)) % 360.0
        _walk_state.speed = max(0.0, _walk_state.speed + random.uniform(-0.5, 0.5))
        delta = _walk_state.speed * DEFAULT_POLL_INTERVAL  # metres
        # Convert metres to approximate degrees
        d_lat = (delta / 111_111.0) * math.cos(math.radians(_walk_state.heading))
        d_lon = (delta / (111_111.0 * math.cos(math.radians(_walk_state.lat)))) * math.sin(
            math.radians(_walk_state.heading)
        )
        _walk_state.lat += d_lat
        _walk_state.lon += d_lon
        _walk_state.alt += random.uniform(-1.0, 1.0)

        return LocationPoint(
            timestamp=time.time(),
            latitude=_walk_state.lat,
            longitude=_walk_state.lon,
            altitude=_walk_state.alt,
            accuracy=50.0,
            speed=_walk_state.speed,
            heading=_walk_state.heading,
            source="simulated",
        )


def _read_location() -> LocationPoint:
    """Return the best available location fix."""
    point = _fetch_ip_location()
    if point is not None:
        return point
    return _simulate_location()


# ===========================================================================
# Background reader thread
# ===========================================================================

class LocationReader:
    """Continuously reads location data in a background thread.

    Usage::

        reader = LocationReader(poll_interval=5.0)
        reader.start()
        # … consume reader.queue …
        reader.stop()
        reader.join()
    """

    def __init__(self, poll_interval: float = DEFAULT_POLL_INTERVAL) -> None:
        self.poll_interval = poll_interval
        self.queue: queue.Queue[LocationPoint] = queue.Queue()
        self._stop_event = threading.Event()
        self._thread = threading.Thread(
            target=self._run,
            name="LocationReader",
            daemon=True,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the background reader thread."""
        self._stop_event.clear()
        self._thread.start()
        print("[maps] LocationReader thread started "
              f"(poll_interval={self.poll_interval}s)")

    def stop(self) -> None:
        """Signal the background thread to stop after its current sleep."""
        self._stop_event.set()

    def join(self, timeout: Optional[float] = None) -> None:
        """Wait for the background thread to finish."""
        self._thread.join(timeout=timeout)

    @property
    def is_alive(self) -> bool:
        return self._thread.is_alive()

    # ------------------------------------------------------------------
    # Thread body
    # ------------------------------------------------------------------

    def _run(self) -> None:
        print("[maps] LocationReader: entering read loop")
        while not self._stop_event.is_set():
            try:
                point = _read_location()
                self.queue.put(point)
                print(
                    f"[maps] Read location: lat={point.latitude:.5f} "
                    f"lon={point.longitude:.5f} source={point.source}"
                )
            except Exception as exc:  # noqa: BLE001
                print(f"[maps] LocationReader error: {exc}")
            self._stop_event.wait(timeout=self.poll_interval)
        print("[maps] LocationReader: read loop exited")


# ===========================================================================
# DuckDB helpers
# ===========================================================================

def _ensure_table(conn: duckdb.DuckDBPyConnection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS location_readings (
            id          INTEGER,
            timestamp   DOUBLE      NOT NULL,
            latitude    DOUBLE      NOT NULL,
            longitude   DOUBLE      NOT NULL,
            altitude    DOUBLE,
            accuracy    DOUBLE,
            speed       DOUBLE,
            heading     DOUBLE,
            source      VARCHAR,
            city        VARCHAR,
            country     VARCHAR,
            extra       VARCHAR,
            PRIMARY KEY (id)
        )
        """
    )
    # Sequence for surrogate PK (DuckDB ≥ 0.9 supports CREATE SEQUENCE)
    conn.execute(
        "CREATE SEQUENCE IF NOT EXISTS location_readings_id_seq START 1"
    )


def _insert_batch(
    conn: duckdb.DuckDBPyConnection, batch: list[LocationPoint]
) -> None:
    """Insert a list of ``LocationPoint`` objects into ``location_readings``."""
    for pt in batch:
        conn.execute(
            """
            INSERT INTO location_readings
                (id, timestamp, latitude, longitude, altitude,
                 accuracy, speed, heading, source, city, country, extra)
            VALUES (
                nextval('location_readings_id_seq'),
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
            )
            """,
            (
                pt.timestamp,
                pt.latitude,
                pt.longitude,
                pt.altitude,
                pt.accuracy,
                pt.speed,
                pt.heading,
                pt.source,
                pt.city,
                pt.country,
                pt.extra,
            ),
        )


def flush_batch(
    batch: list[LocationPoint],
    db_path: str = DB_PATH,
) -> None:
    """Write *batch* to the DuckDB database at *db_path*.

    The database file and table are created automatically if they do not
    already exist.
    """
    if not batch:
        return
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = duckdb.connect(db_path)
    try:
        _ensure_table(conn)
        _insert_batch(conn, batch)
        conn.commit()
    finally:
        conn.close()
    print(f"[maps] Flushed {len(batch)} location point(s) → {db_path}")


# ===========================================================================
# Pipeline orchestrator
# ===========================================================================

def run_location_pipeline(
    duration: float = DEFAULT_DURATION,
    poll_interval: float = DEFAULT_POLL_INTERVAL,
    batch_size: int = DEFAULT_BATCH_SIZE,
    flush_interval: float = DEFAULT_FLUSH_INTERVAL,
    db_path: str = DB_PATH,
) -> None:
    """Read location data for *duration* seconds, streaming batches to DuckDB.

    Parameters
    ----------
    duration:       Total seconds to collect data (``float("inf")`` for
                    continuous operation).
    poll_interval:  Seconds between location reads in the background thread.
    batch_size:     Maximum number of points to accumulate before flushing.
    flush_interval: Maximum seconds between flushes (time-based trigger).
    db_path:        Path to the DuckDB file.
    """
    print("=" * 60)
    print("[maps] Starting location pipeline …")
    print(f"       duration={duration}s  poll={poll_interval}s  "
          f"batch_size={batch_size}  flush_every={flush_interval}s")
    print("=" * 60)

    reader = LocationReader(poll_interval=poll_interval)
    reader.start()

    batch: list[LocationPoint] = []
    deadline = time.monotonic() + duration
    last_flush = time.monotonic()

    try:
        while True:
            now = time.monotonic()
            if now >= deadline:
                break

            # Drain all currently available points from the queue
            while True:
                try:
                    point = reader.queue.get_nowait()
                    batch.append(point)
                except queue.Empty:
                    break

            # Flush if batch is full or flush interval has elapsed
            time_since_flush = now - last_flush
            if len(batch) >= batch_size or (batch and time_since_flush >= flush_interval):
                flush_batch(batch, db_path=db_path)
                batch = []
                last_flush = time.monotonic()

            # Sleep briefly so we don't busy-loop
            time.sleep(min(1.0, max(0.1, deadline - time.monotonic())))

    except KeyboardInterrupt:
        print("\n[maps] Interrupted by user.")
    finally:
        reader.stop()
        reader.join(timeout=poll_interval + 2)

        # Drain remaining items
        while True:
            try:
                batch.append(reader.queue.get_nowait())
            except queue.Empty:
                break

        # Final flush
        if batch:
            flush_batch(batch, db_path=db_path)

    print(f"[maps] Location pipeline complete. Data stored in {db_path}")


# ===========================================================================
# Standalone entry point
# ===========================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Collect location data into a local DuckDB database."
    )
    parser.add_argument(
        "--duration", type=float, default=DEFAULT_DURATION,
        help=f"Collection duration in seconds (default: {DEFAULT_DURATION})",
    )
    parser.add_argument(
        "--poll-interval", type=float, default=DEFAULT_POLL_INTERVAL,
        help=f"Seconds between location reads (default: {DEFAULT_POLL_INTERVAL})",
    )
    parser.add_argument(
        "--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
        help=f"Points per DuckDB flush (default: {DEFAULT_BATCH_SIZE})",
    )
    parser.add_argument(
        "--flush-interval", type=float, default=DEFAULT_FLUSH_INTERVAL,
        help=f"Max seconds between flushes (default: {DEFAULT_FLUSH_INTERVAL})",
    )
    parser.add_argument(
        "--db-path", type=str, default=DB_PATH,
        help=f"Path to DuckDB file (default: {DB_PATH})",
    )
    args = parser.parse_args()

    run_location_pipeline(
        duration=args.duration,
        poll_interval=args.poll_interval,
        batch_size=args.batch_size,
        flush_interval=args.flush_interval,
        db_path=args.db_path,
    )
