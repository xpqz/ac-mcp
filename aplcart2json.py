#!/usr/bin/env python3
"""
Convert APLCart TSV to a normalised JSONL corpus with typed-placeholder metadata.

Legend (used below):

  X Y Z  : any-type array
  M N    : numeric array
  I J    : integer array
  A B    : Boolean array
  C D    : character array
  f g h  : function
  ax     : bracket axis
  s      : scalar
  v      : vector
  m      : matrix
"""

from __future__ import annotations
import argparse
import csv
import json
import re
import sqlite3
import sys
from pathlib import Path
from typing import Iterable, List, Dict

import requests

TSV_URL = (
    "https://raw.githubusercontent.com/abrudz/aplcart/" "refs/heads/master/table.tsv"
)
OUTFILE = Path("aplcart.jsonl")
DBFILE = Path("aplcart.db")

PLACEHOLDER_RE = re.compile(r"\b(?:[A-Z]|ax|[fghsvm])\b")

PLACEHOLDER_TYPES: Dict[str, str] = {
    **dict.fromkeys("XYZ", "array-any"),
    **dict.fromkeys("MN", "array-numeric"),
    **dict.fromkeys("IJ", "array-integer"),
    **dict.fromkeys("AB", "array-boolean"),
    **dict.fromkeys("CD", "array-char"),
    "f": "function",
    "g": "function",
    "h": "function",
    "ax": "axis",
    "s": "scalar",
    "v": "vector",
    "m": "matrix",
}

DATA_KINDS = {
    "array-any",
    "array-numeric",
    "array-integer",
    "array-boolean",
    "array-char",
    "scalar",
    "vector",
    "matrix",
}


def extract_placeholders(syntax: str) -> List[Dict[str, str]]:
    """Return ordered unique placeholders with their semantic kind."""
    seen = set()
    placeholders = []
    for match in PLACEHOLDER_RE.finditer(syntax):
        token = match.group(0)
        if token not in seen:
            seen.add(token)
            kind = PLACEHOLDER_TYPES.get(token, "unknown")
            placeholders.append({"symbol": token, "kind": kind})
    return placeholders


def arity_from_placeholders(ph_list: List[Dict[str, str]]) -> str:
    """Derive niladic / monadic / dyadic / ambivalent from data-argument tokens."""
    data_args = [p for p in ph_list if p["kind"] in DATA_KINDS]
    n = len(data_args)
    return {0: "niladic", 1: "monadic", 2: "dyadic"}.get(n, "ambivalent")


SPLIT_KW_RE = re.compile(r"[ ,]+")


def kebabify(word: str) -> str:
    return re.sub(r"\s+", "-", word.strip().lower())


def explode_keywords(field: str) -> List[str]:
    if not field.strip():
        return []
    return [kebabify(x) for x in SPLIT_KW_RE.split(field) if x]


def fetch_tsv(url: str = TSV_URL) -> str:
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    r.encoding = "utf-8"
    return r.text


def parse_tsv(tsv_text: str) -> Iterable[dict[str, str]]:
    reader = csv.DictReader(tsv_text.splitlines(), delimiter="\t")
    for row in reader:
        yield row


def create_database(db_path: Path) -> sqlite3.Connection:
    """Create SQLite database with aplcart table."""
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS aplcart (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            syntax TEXT NOT NULL,
            description TEXT NOT NULL,
            arity TEXT,
            placeholders TEXT,
            class TEXT,
            type TEXT,
            group_name TEXT,
            category TEXT,
            keywords TEXT,
            docs_url TEXT
        )
    """
    )
    conn.execute("DELETE FROM aplcart")  # Clear existing data
    conn.commit()
    return conn


def convert_to_db(tsv_text: str, db_path: Path = DBFILE) -> int:
    """Convert TSV to SQLite database."""
    conn = create_database(db_path)
    n = 0

    for raw in parse_tsv(tsv_text):
        raw.pop("TIO", None)  # drop execution link
        placeholders = extract_placeholders(raw["SYNTAX"])

        conn.execute(
            """
            INSERT INTO aplcart (
                syntax, description, arity, placeholders, class, type,
                group_name, category, keywords, docs_url
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                raw["SYNTAX"].strip(),
                raw["DESCRIPTION"].strip(),
                arity_from_placeholders(placeholders),
                json.dumps(placeholders),
                raw["CLASS"].strip().lower() or None,
                raw["TYPE"].strip().lower() or None,
                raw["GROUP"].strip().lower() or None,
                raw["CATEGORY"].strip().lower().replace("/", "-") or None,
                json.dumps(explode_keywords(raw["KEYWORDS"])),
                raw["DOCS"].strip(),
            ),
        )
        n += 1

    conn.commit()
    conn.close()
    return n


def convert(tsv_text: str, out_path: Path = OUTFILE) -> int:
    n = 0
    with out_path.open("w", encoding="utf-8") as fh:
        for raw in parse_tsv(tsv_text):
            raw.pop("TIO", None)  # drop execution link
            placeholders = extract_placeholders(raw["SYNTAX"])
            record = {
                "syntax": raw["SYNTAX"].strip(),
                "description": raw["DESCRIPTION"].strip(),
                "arity": arity_from_placeholders(placeholders),
                "placeholders": placeholders,
                "class": raw["CLASS"].strip().lower() or None,
                "type": raw["TYPE"].strip().lower() or None,
                "group": raw["GROUP"].strip().lower() or None,
                "category": raw["CATEGORY"].strip().lower().replace("/", "-") or None,
                "keywords": explode_keywords(raw["KEYWORDS"]),
                "docs_url": raw["DOCS"].strip(),
            }
            json.dump(record, fh, ensure_ascii=False)
            fh.write("\n")
            n += 1
    return n


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert APLCart TSV to JSONL or SQLite database"
    )
    parser.add_argument(
        "--db",
        action="store_true",
        help="Store in SQLite database instead of JSONL file",
    )
    parser.add_argument(
        "--db-path",
        type=Path,
        default=DBFILE,
        help=f"Path to SQLite database file (default: {DBFILE})",
    )
    parser.add_argument(
        "--jsonl-path",
        type=Path,
        default=OUTFILE,
        help=f"Path to JSONL output file (default: {OUTFILE})",
    )
    args = parser.parse_args()

    print("Fetching TSV…", file=sys.stderr)
    tsv = fetch_tsv()
    print("Converting…", file=sys.stderr)

    if args.db:
        count = convert_to_db(tsv, args.db_path)
        print(f"Wrote {count} records to {args.db_path}", file=sys.stderr)
    else:
        count = convert(tsv, args.jsonl_path)
        print(f"Wrote {count} records to {args.jsonl_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
