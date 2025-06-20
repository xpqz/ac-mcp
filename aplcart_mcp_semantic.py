#!/usr/bin/env python3
"""
MCP server that exposes APLCart idiom corpus with semantic search capabilities.

You need to run `generate_embeddings.py` first to create the FAISS index. You also need to set the `OPENAI_API_KEY` environment variable for semantic search to work.

If `APLCART_USE_DB` is set to "true", it will use SQLite for
exact match and substring search; otherwise, it will use the JSONL file directly.

Run aplcart2json.py to fetch, and convert the TSV data to JSONL format first.

Tools
  - lookup-syntax    – exact match on `syntax`
  - search           – substring search with limit
  - keywords-for     – return keyword list for a given `syntax`
  - semantic-search  – natural language search using embeddings
"""

from __future__ import annotations

import json
import os
import pickle
import sqlite3
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional

import faiss
import numpy as np
from mcp.server.fastmcp import FastMCP
from openai import OpenAI

USE_DB = os.environ.get("APLCART_USE_DB", "").lower() in ("1", "true", "yes")
DATA_FILE = Path(__file__).with_name("aplcart.jsonl")
DB_FILE = Path(__file__).with_name("aplcart.db")
MAX_LIMIT = 50

INDEX_FILE = Path(__file__).with_name("aplcart.index")
METADATA_FILE = Path(__file__).with_name("aplcart_metadata.pkl")

SEMANTIC_AVAILABLE = INDEX_FILE.exists() and METADATA_FILE.exists()

faiss_index: Optional[faiss.IndexFlatL2] = None
entries_metadata: Optional[List[Dict[str, Any]]] = None
openai_client: Optional[OpenAI] = None

if SEMANTIC_AVAILABLE:
    faiss_index = faiss.read_index(str(INDEX_FILE))
    with open(METADATA_FILE, "rb") as f:
        entries_metadata = pickle.load(f)
    if os.environ.get("OPENAI_API_KEY"):
        openai_client = OpenAI()


def _row_to_dict(row: sqlite3.Row) -> Dict[str, Any]:
    return {
        "syntax": row["syntax"],
        "description": row["description"],
        "arity": row["arity"],
        "placeholders": json.loads(row["placeholders"]) if row["placeholders"] else [],
        "class": row["class"],
        "type": row["type"],
        "group": row["group_name"],
        "category": row["category"],
        "keywords": json.loads(row["keywords"]) if row["keywords"] else [],
        "docs_url": row["docs_url"],
    }


def _get_db_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_FILE))
    conn.row_factory = sqlite3.Row
    return conn


def _records() -> Iterable[Dict[str, Any]]:
    """Yield one parsed JSON object per line."""
    with DATA_FILE.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line := line.strip():
                yield json.loads(line)


def _first(predicate: Callable[[Dict[str, Any]], bool]) -> Optional[Dict[str, Any]]:
    for rec in _records():
        if predicate(rec):
            return rec
    return None


def _collect(
    predicate: Callable[[Dict[str, Any]], bool], limit: int
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for rec in _records():
        if predicate(rec):
            out.append(rec)
            if len(out) >= limit:  # stop early
                break
    return out


def _db_lookup_syntax(syntax: str) -> Optional[Dict[str, Any]]:
    """Exact match lookup in database."""
    with _get_db_connection() as conn:
        row = conn.execute(
            "SELECT * FROM aplcart WHERE syntax = ? LIMIT 1", (syntax.strip(),)
        ).fetchone()
        return _row_to_dict(row) if row else None


def _db_search(query: str, limit: int) -> List[Dict[str, Any]]:
    """Search in database across multiple fields."""
    q = f"%{query}%"
    with _get_db_connection() as conn:
        # Search in syntax, description, and keywords (using LIKE on JSON array)
        rows = conn.execute(
            """
            SELECT * FROM aplcart 
            WHERE syntax LIKE ? 
               OR description LIKE ? 
               OR keywords LIKE ?
            LIMIT ?
            """,
            (q, q, q, limit),
        ).fetchall()
        return [_row_to_dict(row) for row in rows]


def _db_keywords_for(syntax: str) -> List[str]:
    """Get keywords for a syntax from database."""
    with _get_db_connection() as conn:
        row = conn.execute(
            "SELECT keywords FROM aplcart WHERE syntax = ? LIMIT 1", (syntax.strip(),)
        ).fetchone()
        return json.loads(row["keywords"]) if row and row["keywords"] else []


def _get_query_embedding(
    query: str, model: str = "text-embedding-3-small"
) -> np.ndarray:
    """Get embedding for a query string."""
    if not openai_client:
        raise ValueError(
            "OpenAI client not initialised. Set OPENAI_API_KEY environment variable."
        )

    response = openai_client.embeddings.create(model=model, input=query)
    return np.array([response.data[0].embedding], dtype="float32")


def _semantic_search(query: str, limit: int = 10) -> List[Dict[str, Any]]:
    """Search for similar entries using semantic search."""
    if not SEMANTIC_AVAILABLE or not faiss_index or not entries_metadata:
        return []

    try:
        query_embedding = _get_query_embedding(query)
        distances, indices = faiss_index.search(query_embedding, limit)  # type: ignore

        results = []
        for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            entry = entries_metadata[idx].copy()
            entry["similarity_score"] = float(
                1 / (1 + dist)
            )  # Convert distance to similarity
            results.append(entry)

        return results
    except Exception as e:
        print(f"Semantic search error: {e}")
        return []


mcp = FastMCP("APLCart-Semantic")


@mcp.tool(
    name="lookup-syntax",
    description="Return the record whose `syntax` exactly matches the input.",
)
def lookup_syntax(syntax: str) -> Optional[Dict[str, Any]]:
    """Exact-match lookup."""
    if USE_DB:
        return _db_lookup_syntax(syntax)
    else:
        return _first(lambda r: r["syntax"] == syntax.strip())


@mcp.tool(
    description="Substring search across syntax, description and keywords "
    "(case-insensitive)."
)
def search(query: str, limit: int = 10) -> List[Dict[str, Any]]:
    """Return up to *limit* matching records."""
    q = query.lower()
    limit = min(max(1, limit), MAX_LIMIT)

    if USE_DB:
        return _db_search(q, limit)
    else:
        return _collect(
            lambda r: (
                q in r["syntax"].lower()
                or q in r["description"].lower()
                or any(q in k for k in r["keywords"])
            ),
            limit,
        )


@mcp.tool(
    name="keywords-for", description="Return the keyword list for the given `syntax`."
)
def keywords_for(syntax: str) -> List[str]:
    """Retrieve keywords for a `syntax` entry, or empty list if none."""
    if USE_DB:
        return _db_keywords_for(syntax)
    else:
        rec = _first(lambda r: r["syntax"] == syntax.strip())
        return rec["keywords"] if rec else []


@mcp.tool(
    name="semantic-search",
    description="Natural language search using semantic embeddings. "
    "Use this for queries in everyday language when exact keywords might not match.",
)
def semantic_search(query: str, limit: int = 10) -> Dict[str, Any]:
    """
    Search using natural language queries.
    Returns entries ranked by semantic similarity.

    Examples:
      - "how do I combine two arrays"
      - "remove duplicates from a list"
      - "find the largest value"
    """
    if not SEMANTIC_AVAILABLE:
        return {
            "error": "Semantic search not available. Please run generate_embeddings.py first.",
            "available": False,
        }

    if not os.environ.get("OPENAI_API_KEY"):
        return {
            "error": "OPENAI_API_KEY environment variable not set",
            "available": False,
        }

    limit = min(max(1, limit), MAX_LIMIT)
    results = _semantic_search(query, limit)

    return {
        "query": query,
        "results": results,
        "count": len(results),
        "available": True,
    }


def main():
    """Entry point for the aplcart-mcp command."""
    mcp.run()  # stdio transport by default


if __name__ == "__main__":
    main()
