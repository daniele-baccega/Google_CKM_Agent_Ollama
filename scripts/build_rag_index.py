"""Build or update the CKM RAG index using Chroma."""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.rag import build_rag_index


def main() -> None:
    parser = argparse.ArgumentParser(description="Build CKM RAG index")
    parser.add_argument("--docs-dir", default="docs")
    parser.add_argument("--chroma-dir", default=".chroma")
    parser.add_argument("--collection", default="ckm_rules")
    parser.add_argument(
        "--embed-model",
        default="sentence-transformers/all-MiniLM-L6-v2",
    )
    parser.add_argument("--chunk-size", type=int, default=800)
    parser.add_argument("--chunk-overlap", type=int, default=120)

    args = parser.parse_args()
    result = build_rag_index(
        docs_dir=args.docs_dir,
        chroma_dir=args.chroma_dir,
        collection_name=args.collection,
        embedding_model=args.embed_model,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )

    print("RAG index status:")
    print(f"  Documents: {result['documents']}")
    print(f"  Chunks: {result['chunks']}")
    print(f"  Message: {result['message']}")


if __name__ == "__main__":
    main()
