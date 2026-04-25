"""RAG utilities for CKM Agent using Chroma and Markdown docs."""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from typing import Iterable, List
from typing import Optional
from typing import Tuple

import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import CrossEncoder
import pdfplumber

try:
    import fitz  # PyMuPDF
    HAS_PYMUPDF = True
except ImportError:
    HAS_PYMUPDF = False

try:
    from rank_bm25 import BM25Okapi
    HAS_BM25 = True
except ImportError:
    HAS_BM25 = False
    print("Warning: rank-bm25 not installed. Hybrid search disabled. Install with: pip install rank-bm25")

try:
    from transformers import AutoTokenizer
    HAS_TOKENIZER = True
except ImportError:
    HAS_TOKENIZER = False
    print("Warning: transformers not available for token counting. Falling back to character-based chunking.")

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
DEFAULT_CHUNK_SIZE = int(os.getenv("RAG_CHUNK_SIZE", "500"))  # tokens (was 800 chars)
DEFAULT_CHUNK_OVERLAP = int(os.getenv("RAG_CHUNK_OVERLAP", "80"))  # tokens (was 120 chars)
DEFAULT_ENABLE_RERANKING = os.getenv("RAG_ENABLE_RERANKING", "true").lower() == "true"
DEFAULT_ENABLE_HYBRID_SEARCH = os.getenv("RAG_ENABLE_HYBRID_SEARCH", "true").lower() == "true"

SUPPORTED_EXTENSIONS = (".md", ".txt", ".pdf")

_CACHED_CLIENT: Optional[Tuple[str, chromadb.PersistentClient]] = None
_CACHED_EMBED_FN: Optional[Tuple[str, embedding_functions.SentenceTransformerEmbeddingFunction]] = None
_CACHED_COLLECTION: Optional[tuple[str, str, str, chromadb.Collection]] = None
_CACHED_RERANKER: Optional[Tuple[str, CrossEncoder]] = None
_CACHED_TOKENIZER: Optional[Tuple[str, object]] = None
_CACHED_BM25: Optional[Tuple[str, BM25Okapi, List[List[str]]]] = None
_DOCUMENT_HASHES: dict = {}  # Track document content hashes for incremental updates


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


def _extract_pdf_text(pdf_path: str) -> str:
    """Extract text from a PDF file using PyMuPDF (primary) or pdfplumber (fallback)."""
    
    # Try PyMuPDF first (more robust for complex PDFs)
    if HAS_PYMUPDF:
        try:
            doc = fitz.open(pdf_path)
            text_parts = []
            for page_num, page in enumerate(doc, 1):
                text = page.get_text()
                if text and text.strip():
                    text_parts.append(text)
            doc.close()
            
            if text_parts:
                return "\n\n".join(text_parts)
        except Exception as e:
            print(f"Warning: PyMuPDF extraction failed for {pdf_path}: {e}")
    
    # Fallback to pdfplumber
    try:
        with pdfplumber.open(pdf_path) as pdf:
            text_parts = []
            for page_num, page in enumerate(pdf.pages, 1):
                # Try layout-aware extraction first
                try:
                    page_text = page.extract_text(layout=True)
                except Exception:
                    # Standard extraction
                    page_text = page.extract_text()
                
                if page_text and page_text.strip():
                    text_parts.append(page_text)
            
            if text_parts:
                return "\n\n".join(text_parts)
            
            print(f"Warning: No text extracted from {pdf_path}")
            return ""
    except Exception as e:
        print(f"Warning: pdfplumber extraction failed for {pdf_path}: {e}")
        return ""


def load_markdown_files(docs_dir: str) -> List[tuple[str, str]]:
    """Load markdown, text, and PDF files from the docs directory."""
    docs: List[tuple[str, str]] = []
    if not os.path.isdir(docs_dir):
        return docs

    for path in _iter_doc_paths(docs_dir):
        if path.lower().endswith(".pdf"):
            text = _extract_pdf_text(path).strip()
        else:
            with open(path, "r", encoding="utf-8", errors="ignore") as handle:
                text = handle.read().strip()
        
        if text:
            docs.append((path, text))
    return docs


def _get_tokenizer(model_name: str) -> Optional[object]:
    """Get or cache a tokenizer for token-based chunking."""
    global _CACHED_TOKENIZER
    if not HAS_TOKENIZER:
        return None
    
    if _CACHED_TOKENIZER and _CACHED_TOKENIZER[0] == model_name:
        return _CACHED_TOKENIZER[1]
    
    try:
        # For sentence-transformers models, use the underlying transformer tokenizer
        base_model = model_name.replace("sentence-transformers/", "")
        tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
        _CACHED_TOKENIZER = (model_name, tokenizer)
        return tokenizer
    except Exception as e:
        print(f"Warning: Could not load tokenizer for {model_name}: {e}")
        return None


def _count_tokens(text: str, tokenizer: Optional[object]) -> int:
    """Count tokens in text using tokenizer, fallback to char-based estimate."""
    if tokenizer is None:
        # Fallback: rough estimate (1 token ≈ 1.3 characters for clinical text)
        return len(text) // 1.3
    
    try:
        tokens = tokenizer.encode(text, add_special_tokens=False)
        return len(tokens)
    except Exception:
        return len(text) // 1.3


def chunk_text(text: str, chunk_size: int, overlap: int, embedding_model: str = DEFAULT_EMBED_MODEL) -> List[str]:
    """Split text into overlapping chunks using token-based sizing.
    
    Args:
        text: Full text to chunk
        chunk_size: Target size in tokens (if tokenizer available) or characters (fallback)
        overlap: Overlap size in tokens (if tokenizer available) or characters (fallback)
        embedding_model: Model name to load appropriate tokenizer
    
    Returns:
        List of text chunks
    """
    tokenizer = _get_tokenizer(embedding_model) if HAS_TOKENIZER else None
    use_tokens = tokenizer is not None
    
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: List[str] = []
    current: List[str] = []
    current_tokens = 0

    for paragraph in paragraphs:
        para_tokens = _count_tokens(paragraph + "\n\n", tokenizer)
        
        if current_tokens + para_tokens <= chunk_size:
            current.append(paragraph)
            current_tokens += para_tokens
            continue

        if current:
            chunks.append("\n\n".join(current))

        if overlap > 0 and current:
            # Preserve overlap using full paragraphs
            overlap_parts: List[str] = []
            overlap_tokens = 0
            for prior_paragraph in reversed(current):
                para_tokens_count = _count_tokens(prior_paragraph + "\n\n", tokenizer)
                if overlap_tokens + para_tokens_count > overlap and overlap_parts:
                    break
                overlap_parts.insert(0, prior_paragraph)
                overlap_tokens += para_tokens_count

            current = overlap_parts + [paragraph]
            current_tokens = sum(_count_tokens(p + "\n\n", tokenizer) for p in current)
        else:
            current = [paragraph]
            current_tokens = para_tokens

    if current:
        chunks.append("\n\n".join(current))

    return chunks


def _get_hash_file(chroma_dir: str) -> str:
    """Get path to document hash tracking file."""
    return os.path.join(chroma_dir, ".doc_hashes.json")


def _load_doc_hashes(chroma_dir: str) -> dict:
    """Load previously stored document content hashes."""
    hash_file = _get_hash_file(chroma_dir)
    if os.path.exists(hash_file):
        try:
            with open(hash_file, "r") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def _save_doc_hashes(chroma_dir: str, hashes: dict) -> None:
    """Save document content hashes for future comparison."""
    hash_file = _get_hash_file(chroma_dir)
    os.makedirs(chroma_dir, exist_ok=True)
    try:
        with open(hash_file, "w") as f:
            json.dump(hashes, f)
    except Exception as e:
        print(f"Warning: Could not save document hashes: {e}")


def _get_doc_hash(content: str) -> str:
    """Compute SHA256 hash of document content."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _make_chunk_id(source: str, chunk_index: int) -> str:
    """Generate unique chunk ID from source path and index."""
    digest = hashlib.sha1(source.encode("utf-8")).hexdigest()[:12]
    return f"{digest}-{chunk_index}"


def _build_bm25_index(documents: List[str], tokenizer: Optional[object]) -> Optional[Tuple[BM25Okapi, List[List[str]]]]:
    """Build BM25 index from documents."""
    if not HAS_BM25:
        return None
    
    try:
        # Tokenize documents for BM25
        tokenized_docs = []
        for doc in documents:
            if tokenizer:
                tokens = tokenizer.tokenize(doc.lower())
            else:
                # Simple whitespace tokenization fallback
                tokens = doc.lower().split()
            tokenized_docs.append(tokens)
        
        bm25 = BM25Okapi(tokenized_docs)
        return (bm25, tokenized_docs)
    except Exception as e:
        print(f"Warning: Could not build BM25 index: {e}")
        return None


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
    force_rebuild: bool = False,
) -> dict:
    """Build or incrementally update the Chroma index from local docs.
    
    Args:
        docs_dir: Directory containing documents
        chroma_dir: Chroma storage directory
        collection_name: Collection name
        embedding_model: Embedding model name
        chunk_size: Chunk size in tokens
        chunk_overlap: Overlap in tokens
        force_rebuild: If True, rebuild entire index ignoring hashes
    
    Returns:
        Dict with indexing results
    """
    docs = load_markdown_files(docs_dir)
    if not docs:
        return {
            "documents": 0,
            "chunks": 0,
            "updated": 0,
            "skipped": 0,
            "message": f"No docs found in '{docs_dir}'.",
        }

    # Load previous document hashes for incremental updates
    prev_hashes = _load_doc_hashes(chroma_dir) if not force_rebuild else {}
    current_hashes = {}
    
    collection = _get_collection(
        chroma_dir=chroma_dir,
        collection_name=collection_name,
        embedding_model=embedding_model,
    )
    
    ids: List[str] = []
    documents: List[str] = []
    metadatas: List[dict] = []
    docs_updated = 0
    docs_skipped = 0

    tokenizer = _get_tokenizer(embedding_model) if HAS_TOKENIZER else None

    for path, text in docs:
        rel_path = os.path.relpath(path, docs_dir)
        content_hash = _get_doc_hash(text)
        current_hashes[rel_path] = content_hash
        
        # Skip if document hasn't changed (incremental update)
        if not force_rebuild and prev_hashes.get(rel_path) == content_hash:
            docs_skipped += 1
            continue
        
        docs_updated += 1
        chunks = chunk_text(text, chunk_size, chunk_overlap, embedding_model)
        for index, chunk in enumerate(chunks):
            ids.append(_make_chunk_id(path, index))
            documents.append(chunk)
            metadatas.append({
                "source": rel_path,
                "chunk_index": index,
            })

    if ids:
        collection.upsert(ids=ids, documents=documents, metadatas=metadatas)
        
        # Build BM25 index for hybrid search
        if HAS_BM25:
            bm25_result = _build_bm25_index(documents, tokenizer)
            if bm25_result:
                global _CACHED_BM25
                _CACHED_BM25 = (chroma_dir, bm25_result[0], bm25_result[1])
    
    # Save updated document hashes
    _save_doc_hashes(chroma_dir, current_hashes)

    return {
        "documents": len(docs),
        "chunks": len(ids),
        "updated": docs_updated,
        "skipped": docs_skipped,
        "message": f"Index updated. {docs_updated} docs updated, {docs_skipped} docs skipped (unchanged).",
    }


def retrieve_context(
    query: str,
    chroma_dir: str = DEFAULT_CHROMA_DIR,
    collection_name: str = DEFAULT_COLLECTION,
    embedding_model: str = DEFAULT_EMBED_MODEL,
    top_k: int = DEFAULT_TOP_K,
    reranker_model: str = DEFAULT_RERANKER_MODEL,
    enable_reranking: bool = DEFAULT_ENABLE_RERANKING,
    enable_hybrid_search: bool = DEFAULT_ENABLE_HYBRID_SEARCH,
) -> List[dict]:
    """Retrieve top-k context chunks for a query with optional re-ranking and hybrid search.
    
    Hybrid search combines:
    - Vector similarity (semantic search)
    - BM25 (keyword search)
    
    Also deduplcates chunks to avoid redundant context.
    """
    collection = _get_collection(
        chroma_dir=chroma_dir,
        collection_name=collection_name,
        embedding_model=embedding_model,
    )

    # Retrieve more candidates if re-ranking is enabled (for better re-ranking)
    retrieval_k = top_k * 3 if enable_reranking else top_k * 2

    # Vector search
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
            "vector_score": 1.0 / (1.0 + dist),  # Convert distance to similarity
            "bm25_score": None,
            "rerank_score": None,
            "hybrid_score": None,
        })

    # Hybrid search: combine with BM25 results
    if enable_hybrid_search and HAS_BM25:
        global _CACHED_BM25
        if _CACHED_BM25 and _CACHED_BM25[0] == chroma_dir:
            bm25, tokenized_docs = _CACHED_BM25[1], _CACHED_BM25[2]
            
            try:
                tokenizer = _get_tokenizer(embedding_model) if HAS_TOKENIZER else None
                if tokenizer:
                    query_tokens = tokenizer.tokenize(query.lower())
                else:
                    query_tokens = query.lower().split()
                
                bm25_scores = bm25.get_scores(query_tokens)
                
                # Normalize BM25 scores to [0, 1]
                max_bm25 = max(bm25_scores) if bm25_scores else 1.0
                if max_bm25 == 0:
                    max_bm25 = 1.0
                
                # Match BM25 scores to chunks (by document order)
                # This is a simplification; ideally BM25 would be chunk-level
                for i, score in enumerate(bm25_scores):
                    if i < len(matches):
                        matches[i]["bm25_score"] = float(score) / max_bm25
                
                # Compute hybrid score: weighted average of vector and BM25
                for match in matches:
                    if match["bm25_score"] is not None:
                        # Weight: 0.6 vector, 0.4 BM25 (can be tuned)
                        hybrid = 0.6 * match["vector_score"] + 0.4 * match["bm25_score"]
                        match["hybrid_score"] = hybrid
                
                # Sort by hybrid score if available
                matches_with_hybrid = [m for m in matches if m["hybrid_score"] is not None]
                matches_without = [m for m in matches if m["hybrid_score"] is None]
                
                matches_with_hybrid.sort(key=lambda m: m["hybrid_score"], reverse=True)
                matches = matches_with_hybrid + matches_without
            except Exception as e:
                print(f"Hybrid search failed: {e}. Falling back to vector search only.")

    # Deduplicate chunks
    seen_content = set()
    unique_matches = []
    for match in matches:
        # Use first 100 chars as dedup key
        content_key = match["content"][:100]
        if content_key not in seen_content:
            seen_content.add(content_key)
            unique_matches.append(match)
        # else: skip duplicate

    matches = unique_matches

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
            # Fallback to hybrid or vector similarity if re-ranking fails
            print(f"Re-ranking failed: {e}. Using vector similarity ranking.")
            matches = matches[:top_k]
    else:
        matches = matches[:top_k]

    return matches


def _strip_extension(filename: str) -> str:
    """Remove file extensions (.pdf, .md, .txt) from document names for cleaner citations."""
    for ext in (".pdf", ".md", ".txt"):
        if filename.lower().endswith(ext):
            return filename[:-len(ext)]
    return filename


def format_rag_context(matches: List[dict], max_chars: int = 3000) -> str:
    """Format retrieved chunks into a citable context block with numbered excerpts.
    
    Each excerpt has an ID (E1, E2, etc.) that agents MUST cite when using that content.
    Format: "Per [Source Document]: [recommendation] (based on excerpt E1)"
    
    Includes:
    - Excerpt ID and document source for traceability
    - Full text snippet
    - Map of excerpt IDs to source documents
    """
    if not matches:
        return "No relevant documents found."

    lines: List[str] = ["DOCUMENT EXCERPTS:\n"]
    
    # Track mapping of excerpt IDs to sources
    excerpt_sources = {}
    unique_sources = set()
    total = 0
    
    for idx, match in enumerate(matches, start=1):
        content = match.get("content", "").strip()
        # Collapse multiple spaces for cleaner display, but preserve structure
        snippet = " ".join(content.split())
        source = match.get("source", "unknown")
        source_clean = _strip_extension(source)  # Remove .pdf, .md, .txt

        # Preserve native excerpt IDs when available (e.g., E3, E4), fallback to positional IDs.
        native_id_match = re.search(r"\b(E\d+)\b", content)
        excerpt_id = native_id_match.group(1) if native_id_match else f"E{idx}"
        
        # Map excerpt ID to source for reference
        excerpt_sources[excerpt_id] = source_clean
        unique_sources.add(source_clean)
        
        # Truncate snippet to reasonable length for display (but full text is in content field)
        snippet_preview = snippet[:500] if len(snippet) > 500 else snippet
        
        # Format: EXCERPT_ID [DOCUMENT_SOURCE] "PREVIEW..." (no chunk citation)
        line = f"{excerpt_id}. [Document: {source_clean}]\n    \"{snippet_preview}...\"\n"
        
        if total + len(line) > max_chars:
            break
            
        lines.append(line)
        total += len(line)

    # Add explicit instructions for citation
    lines.append("\n---")
    lines.append("CITATION REQUIREMENT:")
    lines.append("When recommending something based on an excerpt, MUST cite like this:")
    lines.append('  ✓ "Per [DOCUMENT_SOURCE]: [recommendation] (E1: \'[key phrase]\')"')
    lines.append('  ✓ "[Per Document X, Guideline Y]: [recommendation] (E2)"')
    lines.append("  ✓ Include both the EXCERPT_ID (E1, E2, etc.) AND the SOURCE DOCUMENT")
    lines.append("  ❌ Do NOT hide the source - cite explicitly which excerpt AND DOCUMENT informed your recommendation")
    lines.append("\nEXCERPT ID → SOURCE DOCUMENT MAP:")
    for excerpt_id in sorted(excerpt_sources.keys(), key=lambda x: int(re.sub(r"^E", "", x))):
        source_doc = excerpt_sources[excerpt_id]
        lines.append(f"  {excerpt_id} → {source_doc}")

    return "\n".join(lines)
