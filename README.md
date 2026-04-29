# diary

A local **process-monitoring pipeline** that continuously scans all running OS processes, builds semantic embeddings, tracks them across time via a CONTROLLER/PATHWAYS model, persists everything to DuckDB, and exposes the results as a live **FastMCP** tool server.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [File Structure](#file-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Pipeline Steps](#pipeline-steps)
- [MCP Server & Tools](#mcp-server--tools)
- [Configuration](#configuration)
- [Output Files](#output-files)

---

## Overview

`diary` answers the question: *which processes were running, how did they evolve over time, and which ones are semantically related?*

It does this by:

1. Scanning every PID on the machine repeatedly (default: 10 iterations, 1 s apart).
2. Embedding each process description (name + exe + cmdline) into a 128-dim character-3-gram vector.
3. Linking similar processes across time steps into **pathways** using cosine similarity.
4. Storing the full history as JSON and the pathways in a **DuckDB** database.
5. Serving the database contents as tools via a **FastMCP** HTTP endpoint on `localhost:8000`.

---

## Architecture

```
main.py
 ├─ pipeline.run_full_pipeline()
 │   ├─ Step 1 · process_scanner.scan_all_pids()   → snapshots[]
 │   ├─ Step 2 · build_soa()                        → SOA (Structure-of-Arrays)
 │   ├─ Step 3 · build_controller()                 → CONTROLLER[t][p_idx]
 │   ├─ Step 4 · build_pathways()                   → PATHWAYS {key: [proc|None]}
 │   ├─ Step 5 · save_history()                     → scan_history/scan_history.json
 │   └─ Step 6 · sync_pathways_to_duckdb()          → scan_history/pathways.duckdb
 └─ server.py (subprocess)
     └─ FastMCP tools: search_routes, list_tables, get_route_embedding,
                       + one dynamic tool per DuckDB table
```

**Embedding engine** (`embeddings.py`): pure NumPy, no ML framework required. Uses character tri-gram feature-hashing into a 128-dim L2-normalised vector.

---

## File Structure

```
diary/
├── main.py             # Entry point: runs pipeline then starts MCP server
├── pipeline.py         # Full scan → embed → controller → pathways → DuckDB pipeline
├── process_scanner.py  # psutil-based PID scanner; returns list[dict] sorted by RSS
├── embeddings.py       # Numpy-only char-3gram embeddings + cosine similarity
├── server.py           # FastMCP server; exposes DuckDB tables as tools
├── requirements.txt
└── scan_history/       # Generated at runtime
    ├── scan_history.json   # Snapshots, SOA, CONTROLLER, PATHWAYS
    ├── routes_meta.json    # Table-name → embedding registry
    └── pathways.duckdb     # DuckDB database with one table per pathway
```

---

## Requirements

- Python 3.10+
- Dependencies (see `requirements.txt`):

| Package | Purpose |
|---------|---------|
| `psutil` | OS process introspection |
| `numpy` | Embeddings and cosine similarity |
| `duckdb` | Pathway persistence |
| `fastmcp` | MCP server |
| `fastapi` | HTTP layer |
| `uvicorn` | ASGI server |

---

## Installation

```bash
git clone https://github.com/wired87/diary.git
cd diary
pip install -r requirements.txt
```

---

## Usage

### Run the full pipeline and start the MCP server

```bash
python main.py
```

This runs 10 scan iterations (≈10 s), saves all results to `scan_history/`, then starts the FastMCP server at `http://localhost:8000/mcp`.

### Run only the pipeline (no server)

```python
from pipeline import run_full_pipeline
run_full_pipeline(iterations=10, scan_delay=1.0)
```

### Run only the MCP server (after the pipeline has already produced data)

```bash
python server.py
```

---

## Pipeline Steps

| Step | Function | Description |
|------|----------|-------------|
| 1 | `run_scan_loop` | Collect process snapshots for N iterations. Each snapshot records all PIDs with full metadata and a vector embedding. |
| 2 | `build_soa` | Reorganise snapshots into a Structure-of-Arrays keyed by `time_attr`. |
| 3 | `build_controller` | For every `(time_step t, process p)` pair, find the index of the next time step where a similar process (cosine > `THRESHOLD_N = 0.5`) exists. |
| 4 | `build_pathways` | Follow CONTROLLER links from time step 0 to build a per-process chain across all time steps. Gaps where no match exists are represented as `None`. Capped at the top 50 memory consumers. |
| 5 | `save_history` | Write snapshots + SOA + CONTROLLER + PATHWAYS to `scan_history/scan_history.json`. |
| 6 | `sync_pathways_to_duckdb` | Upsert pathway entries into DuckDB. Matches pathway keys against existing FastAPI route embeddings (`THRESHOLD_J = 0.70`) or DuckDB table names (`THRESHOLD_K = 0.85`); otherwise creates a new table. |

---

## MCP Server & Tools

The FastMCP server exposes the following tools at `http://localhost:8000/mcp`:

| Tool | Description |
|------|-------------|
| `search_routes` | Semantic search over all registered tool/table names. Returns top-k results with similarity scores. |
| `list_tables` | List all pathway tables currently in the DuckDB database. |
| `get_route_embedding` | Return the stored 128-dim embedding vector for a given tool name. |
| `<table_name>` | One dynamic tool per DuckDB table — queries and returns all rows. Discovered at server startup from `routes_meta.json`. |

**Example – semantic search:**
```
search_routes(query="python interpreter", top_k=3)
```

---

## Configuration

Edit the constants at the top of `pipeline.py` and `server.py`:

| Constant | Default | Description |
|----------|---------|-------------|
| `THRESHOLD_N` | `0.5` | Minimum cosine similarity to link a process to the next time step (CONTROLLER) |
| `THRESHOLD_J` | `0.70` | Minimum similarity to match a pathway to an existing FastAPI route |
| `THRESHOLD_K` | `0.85` | Minimum similarity to match a pathway to an existing DuckDB table name |
| `MAX_PATHWAYS` | `50` | Maximum number of pathways to persist (top memory consumers) |
| `SEARCH_THRESHOLD` | `0.30` | Minimum similarity score returned by `search_routes` |
| `EMBED_DIM` | `128` | Dimensionality of character-3gram hash vectors |
| `iterations` | `10` | Number of scan iterations (passed to `run_full_pipeline`) |
| `scan_delay` | `1.0 s` | Sleep between scan iterations |

---

## Output Files

All output is written to `scan_history/` (created automatically):

- **`scan_history.json`** – complete pipeline state (snapshots, SOA, CONTROLLER, PATHWAYS).
- **`pathways.duckdb`** – DuckDB database; one table per tracked process pathway.
- **`routes_meta.json`** – JSON registry mapping table names to their embedding vectors, used by the MCP server for semantic tool discovery.
