#!/usr/bin/env python3
"""Search APLCart entries using semantic embeddings."""

import json
import os
import numpy as np
import faiss
from openai import OpenAI
from typing import List, Dict, Any, Tuple
import pickle
import argparse


def load_index_and_metadata(
    index_path: str = "aplcart.index", metadata_path: str = "aplcart_metadata.pkl"
) -> Tuple[faiss.IndexFlatL2, List[Dict[str, Any]]]:
    """Load FAISS index and metadata from disk."""
    index = faiss.read_index(index_path)

    with open(metadata_path, "rb") as f:
        metadata = pickle.load(f)

    return index, metadata


def get_query_embedding(
    query: str, model: str = "text-embedding-3-small"
) -> np.ndarray:
    """Get embedding for a query string."""
    client = OpenAI()
    response = client.embeddings.create(model=model, input=query)
    return np.array([response.data[0].embedding], dtype="float32")


def search_similar(
    query: str,
    index: faiss.IndexFlatL2,
    entries: List[Dict[str, Any]],
    k: int = 10,
    model: str = "text-embedding-3-small",
) -> List[Dict[str, Any]]:
    """Search for similar entries using semantic search."""
    query_embedding = get_query_embedding(query, model)

    distances, indices = index.search(query_embedding, k)  # type: ignore

    results = []
    for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
        entry = entries[idx].copy()
        entry["score"] = float(dist)
        entry["rank"] = i + 1
        results.append(entry)

    return results


def format_result(entry: Dict[str, Any]) -> str:
    """Format a search result for display."""
    parts = [
        f"Rank {entry['rank']}: {entry['syntax']}",
        f"  Description: {entry['description']}",
        f"  Score: {entry['score']:.3f}",
    ]

    if entry.get("keywords"):
        parts.append(f"  Keywords: {', '.join(entry['keywords'][:5])}")

    return "\n".join(parts)


def interactive_search(
    index: faiss.IndexFlatL2,
    entries: List[Dict[str, Any]],
    model: str = "text-embedding-3-small",
):
    """Run interactive search loop."""
    print("APLCart Semantic Search")
    print("Type your query in natural language, or 'quit' to exit")
    print("-" * 50)

    while True:
        try:
            query = input("\nQuery: ").strip()
        except EOFError:
            print()
            break
        except KeyboardInterrupt:
            print()
            break

        if query.lower() in ["quit", "exit", "q"]:
            break

        if not query:
            continue

        try:
            results = search_similar(query, index, entries, k=5, model=model)

            print(f"\nTop {len(results)} results for: '{query}'")
            print("=" * 50)

            for result in results:
                print(format_result(result))
                print()

        except Exception as e:
            print(f"Error: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Search APLCart using semantic embeddings"
    )
    parser.add_argument(
        "query", nargs="?", help="Search query (if not provided, interactive mode)"
    )
    parser.add_argument("--index", default="aplcart.index", help="FAISS index file")
    parser.add_argument(
        "--metadata", default="aplcart_metadata.pkl", help="Metadata file"
    )
    parser.add_argument(
        "-k", "--top-k", type=int, default=5, help="Number of results to return"
    )
    parser.add_argument(
        "--model",
        default="text-embedding-3-small",
        choices=["text-embedding-3-small", "text-embedding-3-large"],
        help="OpenAI embedding model to use",
    )
    parser.add_argument("--json", action="store_true", help="Output results as JSON")
    args = parser.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable not set")
        print("Please set it with: export OPENAI_API_KEY='your-key-here'")
        return

    try:
        index, entries = load_index_and_metadata(args.index, args.metadata)
    except FileNotFoundError:
        print(f"Error: Index files not found. Please run generate_embeddings.py first.")
        return

    if args.query:
        results = search_similar(
            args.query, index, entries, k=args.top_k, model=args.model
        )

        if args.json:
            print(json.dumps(results, indent=2))
        else:
            print(f"\nTop {len(results)} results for: '{args.query}'")
            print("=" * 50)
            for result in results:
                print(format_result(result))
                print()
    else:
        interactive_search(index, entries, model=args.model)


if __name__ == "__main__":
    main()
