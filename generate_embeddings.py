#!/usr/bin/env python3
"""Generate embeddings for APLCart entries and store in FAISS index."""

import json
import os
import numpy as np
import faiss
from openai import OpenAI
from typing import List, Dict, Any
import pickle
from tqdm import tqdm
import argparse


def load_aplcart_entries(jsonl_path: str) -> List[Dict[str, Any]]:
    """Load APLCart entries from JSONL file."""
    entries = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            entries.append(json.loads(line.strip()))
    return entries


def create_embedding_text(entry: Dict[str, Any]) -> str:
    """Create text to embed from an APLCart entry."""
    parts = [
        f"Syntax: {entry.get('syntax', '')}",
        f"Description: {entry.get('description', '')}",
    ]

    if entry.get("category"):
        parts.append(f"Category: {entry['category']}")
    if entry.get("type"):
        parts.append(f"Type: {entry['type']}")

    if entry.get("keywords"):
        parts.append(f"Keywords: {', '.join(entry['keywords'])}")

    if entry.get("placeholders"):
        placeholder_desc = []
        for p in entry["placeholders"]:
            desc = f"{p['symbol']} ({p['kind']})"
            placeholder_desc.append(desc)
        if placeholder_desc:
            parts.append(f"Parameters: {', '.join(placeholder_desc)}")

    return " | ".join(parts)


def get_embeddings(
    texts: List[str], model: str = "text-embedding-3-small"
) -> np.ndarray:
    """Get embeddings from OpenAI API."""
    client = OpenAI()  # Uses OPENAI_API_KEY from environment

    batch_size = 100
    all_embeddings = []

    for i in tqdm(range(0, len(texts), batch_size), desc="Getting embeddings"):
        batch = texts[i : i + batch_size]
        response = client.embeddings.create(model=model, input=batch)
        batch_embeddings = [e.embedding for e in response.data]
        all_embeddings.extend(batch_embeddings)

    return np.array(all_embeddings, dtype="float32")


def create_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatL2:
    """Create FAISS index from embeddings."""
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)  # type: ignore
    return index


def save_index_and_metadata(
    index: faiss.IndexFlatL2,
    entries: List[Dict[str, Any]],
    index_path: str = "aplcart.index",
    metadata_path: str = "aplcart_metadata.pkl",
):
    """Save FAISS index and metadata to disk."""
    faiss.write_index(index, index_path)
    with open(metadata_path, "wb") as f:
        pickle.dump(entries, f)

    print(f"Saved index to {index_path}")
    print(f"Saved metadata to {metadata_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate embeddings for APLCart entries"
    )
    parser.add_argument("--input", default="aplcart.jsonl", help="Input JSONL file")
    parser.add_argument(
        "--index", default="aplcart.index", help="Output FAISS index file"
    )
    parser.add_argument(
        "--metadata", default="aplcart_metadata.pkl", help="Output metadata file"
    )
    parser.add_argument(
        "--model",
        default="text-embedding-3-small",
        choices=["text-embedding-3-small", "text-embedding-3-large"],
        help="OpenAI embedding model to use",
    )
    args = parser.parse_args()

    # Check for OpenAI API key
    if not os.environ.get("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable not set")
        print("Please set it with: export OPENAI_API_KEY='your-key-here'")
        return

    print(f"Loading entries from {args.input}...")
    entries = load_aplcart_entries(args.input)
    print(f"Loaded {len(entries)} entries")

    print("Creating embedding texts...")
    embedding_texts = [create_embedding_text(entry) for entry in entries]

    print(f"Generating embeddings using {args.model}...")
    embeddings = get_embeddings(embedding_texts, model=args.model)
    print(f"Generated embeddings with shape: {embeddings.shape}")

    print("Creating FAISS index...")
    index = create_faiss_index(embeddings)

    save_index_and_metadata(index, entries, args.index, args.metadata)

    print("Done!")


if __name__ == "__main__":
    main()
