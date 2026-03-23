"""RAG utilities for CKM Agent using Chroma and Markdown docs."""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from typing import Iterable, List
from typing import Optional
from typing import Tuple

import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import CrossEncoder

DEFAULT_DOCS_DIR = os.getenv("RAG_DOCS_DIR", "docs")
DEFAULT_CHROMA_DIR = os.getenv("RAG_CHROMA_DIR", ".chroma")
DEFAULT_COLLECTION = os.getenv("RAG_COLLECTION", "ckm_rules")
DEFAULT_EMBED_MODEL = os.getenv(
    "RAG_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)
DEFAULT_RERANKER_MODEL = os.getenv(
    "RAG_RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"
)
DEFAULT_TOP_K = int(os.getenv("RAG_TOP_K", "5"))
DEFAULT_CHUNK_SIZE = int(os.getenv("RAG_CHUNK_SIZE", "800"))
DEFAULT_CHUNK_OVERLAP = int(os.getenv("RAG_CHUNK_OVERLAP", "120"))
DEFAULT_ENABLE_RERANKING = os.getenv("RAG_ENABLE_RERANKING", "true").lower() == "true"

SUPPORTED_EXTENSIONS = (".md", ".txt")

_CACHED_CLIENT: Optional[Tuple[str, chromadb.PersistentClient]] = None
_CACHED_EMBED_FN: Optional[Tuple[str, embedding_functions.SentenceTransformerEmbeddingFunction]] = None
_CACHED_COLLECTION: Optional[tuple[str, str, str, chromadb.Collection]] = None
_CACHED_RERANKER: Optional[Tuple[str, CrossEncoder]] = None


@dataclass(frozen=True)
class RagChunk:
    """Represents a chunk of text and its source metadata."""

    source: str
    content: str
    chunk_index: int


def _iter_doc_paths(docs_dir: str) -> Iterable[str]:
    for root, _, files in os.walk(docs_dir):
        for name in files:
            if name.lower().endswith(SUPPORTED_EXTENSIONS):
                yield os.path.join(root, name)


def load_markdown_files(docs_dir: str) -> List[tuple[str, str]]:
    """Load markdown/text files from the docs directory."""
    docs: List[tuple[str, str]] = []
    if not os.path.isdir(docs_dir):
        return docs

    for path in _iter_doc_paths(docs_dir):
        with open(path, "r", encoding="utf-8", errors="ignore") as handle:
            text = handle.read().strip()
            if text:
                docs.append((path, text))
    return docs


def chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    """Split text into overlapping chunks using paragraph boundaries."""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: List[str] = []
    current: List[str] = []
    current_len = 0

    for paragraph in paragraphs:
        if current_len + len(paragraph) + 2 <= chunk_size:
            current.append(paragraph)
            current_len += len(paragraph) + 2
            continue

        if current:
            chunks.append("\n\n".join(current))

        if overlap > 0 and current:
            # Preserve overlap using full trailing paragraphs (not raw char slices)
            # to avoid broken tokens like "atins" in retrieval snippets.
            overlap_parts: List[str] = []
            overlap_len = 0
            for prior_paragraph in reversed(current):
                paragraph_len = len(prior_paragraph) + 2
                if overlap_len + paragraph_len > overlap and overlap_parts:
                    break
                overlap_parts.insert(0, prior_paragraph)
                overlap_len += paragraph_len

            current = overlap_parts + [paragraph]
            current_len = sum(len(p) + 2 for p in current)
        else:
            current = [paragraph]
            current_len = len(paragraph) + 2

    if current:
        chunks.append("\n\n".join(current))

    return chunks


def _make_chunk_id(source: str, chunk_index: int) -> str:
    digest = hashlib.sha1(source.encode("utf-8")).hexdigest()[:12]
    return f"{digest}-{chunk_index}"


def _get_client(chroma_dir: str) -> chromadb.PersistentClient:
    """Return a cached Chroma client for the given storage path."""
    global _CACHED_CLIENT
    if _CACHED_CLIENT and _CACHED_CLIENT[0] == chroma_dir:
        return _CACHED_CLIENT[1]

    client = chromadb.PersistentClient(path=chroma_dir)
    _CACHED_CLIENT = (chroma_dir, client)
    return client


def _get_embedding_fn(
    embedding_model: str,
) -> embedding_functions.SentenceTransformerEmbeddingFunction:
    """Return a cached sentence-transformers embedding function."""
    global _CACHED_EMBED_FN
    if _CACHED_EMBED_FN and _CACHED_EMBED_FN[0] == embedding_model:
        return _CACHED_EMBED_FN[1]

    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=embedding_model
    )
    _CACHED_EMBED_FN = (embedding_model, embedding_fn)
    return embedding_fn


def _get_reranker(reranker_model: str) -> CrossEncoder:
    """Return a cached cross-encoder reranker model."""
    global _CACHED_RERANKER
    if _CACHED_RERANKER and _CACHED_RERANKER[0] == reranker_model:
        return _CACHED_RERANKER[1]

    reranker = CrossEncoder(reranker_model)
    _CACHED_RERANKER = (reranker_model, reranker)
    return reranker


def _get_collection(
    chroma_dir: str,
    collection_name: str,
    embedding_model: str,
) -> chromadb.Collection:
    """Return a cached collection handle bound to the current embedding model."""
    global _CACHED_COLLECTION
    if _CACHED_COLLECTION:
        cached_dir, cached_name, cached_model, cached_collection = _CACHED_COLLECTION
        if (
            cached_dir == chroma_dir
            and cached_name == collection_name
            and cached_model == embedding_model
        ):
            return cached_collection

    client = _get_client(chroma_dir)
    embedding_fn = _get_embedding_fn(embedding_model)
    collection = client.get_or_create_collection(
        name=collection_name,
        embedding_function=embedding_fn,
    )
    _CACHED_COLLECTION = (chroma_dir, collection_name, embedding_model, collection)
    return collection


def build_rag_index(
    docs_dir: str = DEFAULT_DOCS_DIR,
    chroma_dir: str = DEFAULT_CHROMA_DIR,
    collection_name: str = DEFAULT_COLLECTION,
    embedding_model: str = DEFAULT_EMBED_MODEL,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> dict:
    """Build or update the Chroma index from local docs."""
    docs = load_markdown_files(docs_dir)
    if not docs:
        return {
            "documents": 0,
            "chunks": 0,
            "message": f"No docs found in '{docs_dir}'.",
        }

    collection = _get_collection(
        chroma_dir=chroma_dir,
        collection_name=collection_name,
        embedding_model=embedding_model,
    )

    ids: List[str] = []
    documents: List[str] = []
    metadatas: List[dict] = []

    for path, text in docs:
        chunks = chunk_text(text, chunk_size, chunk_overlap)
        for index, chunk in enumerate(chunks):
            ids.append(_make_chunk_id(path, index))
            documents.append(chunk)
            metadatas.append({
                "source": os.path.relpath(path, docs_dir),
                "chunk_index": index,
            })

    if ids:
        collection.upsert(ids=ids, documents=documents, metadatas=metadatas)

    return {
        "documents": len(docs),
        "chunks": len(ids),
        "message": "Index updated.",
    }


def retrieve_context(
    query: str,
    chroma_dir: str = DEFAULT_CHROMA_DIR,
    collection_name: str = DEFAULT_COLLECTION,
    embedding_model: str = DEFAULT_EMBED_MODEL,
    top_k: int = DEFAULT_TOP_K,
    reranker_model: str = DEFAULT_RERANKER_MODEL,
    enable_reranking: bool = DEFAULT_ENABLE_RERANKING,
) -> List[dict]:
    """Retrieve top-k context chunks for a query with optional re-ranking."""
    collection = _get_collection(
        chroma_dir=chroma_dir,
        collection_name=collection_name,
        embedding_model=embedding_model,
    )

    # Retrieve more candidates if re-ranking is enabled (for better re-ranking)
    retrieval_k = top_k * 2 if enable_reranking else top_k

    result = collection.query(
        query_texts=[query],
        n_results=retrieval_k,
        include=["documents", "metadatas", "distances"],
    )

    documents = result.get("documents", [[]])[0]
    metadatas = result.get("metadatas", [[]])[0]
    distances = result.get("distances", [[]])[0]

    matches = []
    for doc, meta, dist in zip(documents, metadatas, distances):
        matches.append({
            "content": doc,
            "source": meta.get("source", "unknown"),
            "chunk_index": meta.get("chunk_index", 0),
            "vector_distance": dist,
            "rerank_score": None,
        })

    # Re-rank if enabled
    if enable_reranking and matches:
        try:
            reranker = _get_reranker(reranker_model)
            
            # Prepare query-document pairs for reranking
            pairs = [[query, match["content"]] for match in matches]
            
            # Get re-ranker scores
            rerank_scores = reranker.predict(pairs)
            
            # Attach re-ranker scores
            for match, score in zip(matches, rerank_scores):
                match["rerank_score"] = float(score)
            
            # Sort by re-ranker score (higher is better)
            matches.sort(key=lambda m: m["rerank_score"], reverse=True)
            
            # Keep only top_k after re-ranking
            matches = matches[:top_k]
        except Exception as e:
            # Fallback to vector similarity if re-ranking fails
            print(f"Re-ranking failed: {e}. Using vector similarity ranking.")
            matches = matches[:top_k]
    else:
        matches = matches[:top_k]

    return matches


def format_rag_context(matches: List[dict], max_chars: int = 1200) -> str:
    """Format retrieved chunks into a compact context block."""
    if not matches:
        return "RAG_CONTEXT: None"

    lines: List[str] = ["RAG_CONTEXT:"]
    total = 0
    for idx, match in enumerate(matches, start=1):
        snippet = match.get("content", "").strip().replace("\n", " ")
        source = match.get("source", "unknown")
        line = f"{idx}. [{source}] {snippet}"
        if total + len(line) > max_chars:
            break
        lines.append(line)
        total += len(line)

    return "\n".join(lines)
