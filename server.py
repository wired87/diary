"""
server.py – FastMCP server with dynamic tools per DuckDB table.

Startup sequence
----------------
1.  Read ``scan_history/routes_meta.json`` to discover all registered tables.
2.  For each table name, register a FastMCP tool whose docstring contains the
    embedding descriptor (enables downstream similarity search by other agents
    against the /mcp endpoint).
3.  Also read this very file (server.py) to extract the tool names already
    registered, then compare them against routes_meta so new tables that
    appeared after the last pipeline run are also exposed.
4.  Expose an HTTP MCP endpoint on 0.0.0.0:8000.

Similarity search on tool docstrings
-------------------------------------
The ``search_routes`` tool accepts a query string and returns the top-k tools
whose docstring embedding scores > a configurable threshold, relying purely on
numpy cosine similarity.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import os
import re
import sys
from typing import Any

import duckdb
from fastmcp import FastMCP

from embeddings import cosine_similarity, embed, embed_to_list

# ---------------------------------------------------------------------------
# Paths (must match pipeline.py)
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
SCAN_HISTORY_DIR = os.path.join(_HERE, "scan_history")
ROUTES_META_JSON = os.path.join(SCAN_HISTORY_DIR, "routes_meta.json")
DB_PATH = os.path.join(SCAN_HISTORY_DIR, "pathways.duckdb")

SEARCH_THRESHOLD: float = 0.30


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_routes_meta() -> dict[str, list[float]]:
    if not os.path.exists(ROUTES_META_JSON):
        return {}
    with open(ROUTES_META_JSON, encoding="utf-8") as fh:
        return json.load(fh)


def _get_duck_tables() -> list[str]:
    if not os.path.exists(DB_PATH):
        return []
    conn = duckdb.connect(DB_PATH)
    tables = [r[0] for r in conn.execute("SHOW TABLES").fetchall()]
    conn.close()
    return tables


def _query_table(table: str) -> list[dict[str, Any]]:
    conn = duckdb.connect(DB_PATH)
    rows = conn.execute(f'SELECT * FROM "{table}"').fetchall()
    cols = [d[0] for d in conn.description]
    conn.close()
    return [dict(zip(cols, row)) for row in rows]


def _read_tool_names_from_source(source_path: str) -> list[str]:
    """Parse server.py source with ast and return all @mcp.tool names."""
    try:
        with open(source_path, encoding="utf-8") as fh:
            tree = ast.parse(fh.read())
    except Exception:
        return []

    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            for deco in node.decorator_list:
                src = ast.unparse(deco) if hasattr(ast, "unparse") else ""
                if "mcp.tool" in src or "tool" in src:
                    # Check for explicit name= argument
                    if isinstance(deco, ast.Call):
                        for kw in deco.keywords:
                            if kw.arg == "name" and isinstance(kw.value, ast.Constant):
                                names.append(kw.value.value)
                                break
                        else:
                            names.append(node.name)
                    else:
                        names.append(node.name)
    return names


# ---------------------------------------------------------------------------
# FastMCP server
# ---------------------------------------------------------------------------

mcp = FastMCP(
    "ProcessPathwayServer",
    instructions=(
        "MCP server that exposes process-pathway tables from a local DuckDB "
        "database.  Each tool corresponds to one pathway table.  Use "
        "'search_routes' to find relevant tools by semantic query."
    ),
)


# ---------------------------------------------------------------------------
# Static tools
# ---------------------------------------------------------------------------

@mcp.tool()
def search_routes(query: str, top_k: int = 5) -> list[dict[str, Any]]:
    """Search all registered FastMCP tool docstrings by semantic similarity.

    Uses numpy cosine similarity on character-3gram embeddings (no sentence
    transformers).  Returns up to *top_k* tools whose score ≥ threshold.

    Args:
        query:  Natural-language search query.
        top_k:  Maximum number of results to return.
    """
    routes_meta = _load_routes_meta()
    q_vec = embed(query)
    scored: list[tuple[float, str]] = []
    for name, emb in routes_meta.items():
        score = cosine_similarity(q_vec, emb)
        if score >= SEARCH_THRESHOLD:
            scored.append((score, name))
    scored.sort(reverse=True)
    return [
        {"tool_name": name, "similarity": round(score, 4)}
        for score, name in scored[:top_k]
    ]


@mcp.tool()
def list_tables() -> list[str]:
    """List all pathway tables currently stored in the local DuckDB database."""
    return _get_duck_tables()


@mcp.tool()
def get_route_embedding(tool_name: str) -> list[float]:
    """Return the stored embedding vector for *tool_name*.

    The embedding is the character-3gram hash-vector of the table/route name
    used for similarity matching.
    """
    meta = _load_routes_meta()
    return meta.get(tool_name, [])


# ---------------------------------------------------------------------------
# Dynamic tools – one per DuckDB table
# ---------------------------------------------------------------------------

def _register_table_tools() -> None:
    """Register one FastMCP tool per DuckDB table found at startup.

    Reads routes_meta.json for the embedding descriptor; falls back to
    computing the embedding on-the-fly if the table is not in the registry.
    Compares against tool names already declared in this source file to avoid
    double-registration.
    """
    routes_meta = _load_routes_meta()
    tables = _get_duck_tables()
    existing_static_tools = _read_tool_names_from_source(__file__)

    for table in tables:
        if table in existing_static_tools:
            continue  # already registered as a static tool

        emb = routes_meta.get(table) or embed_to_list(table)
        doc = (
            f"Query all entries from pathway table '{table}'.\n\n"
            f"Embedding descriptor: {emb[:8]}…  (dim={len(emb)})\n"
            f"Use 'search_routes' to discover related tables."
        )

        # Build the query function with a stable name so FastMCP can register it
        def _make_query_fn(t: str = table):
            async def query_fn() -> list[dict[str, Any]]:
                return _query_table(t)

            query_fn.__name__ = t
            query_fn.__qualname__ = t
            query_fn.__doc__ = doc
            return query_fn

        mcp.tool(name=table)(_make_query_fn())

        # Ensure routes_meta is up to date
        if table not in routes_meta:
            routes_meta[table] = emb

    # Persist any additions
    if tables:
        os.makedirs(SCAN_HISTORY_DIR, exist_ok=True)
        with open(ROUTES_META_JSON, "w", encoding="utf-8") as fh:
            json.dump(routes_meta, fh, indent=2)


_register_table_tools()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    mcp.run(transport="streamable-http", host="0.0.0.0", port=8000)
