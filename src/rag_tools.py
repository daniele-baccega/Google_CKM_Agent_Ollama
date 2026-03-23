"""Tool functions for retrieving RAG context inside ADK agents."""

import os
from .rag import format_rag_context, retrieve_context


def retrieve_rag_context(case_summary: str) -> str:
    """Retrieve relevant rule/doc context for a case summary.

    Returns a compact RAG_CONTEXT block to include in the case input.
    """
    chroma_dir = os.getenv("RAG_CHROMA_DIR", ".chroma")
    collection = os.getenv("RAG_COLLECTION", "ckm_rules")
    embed_model = os.getenv(
        "RAG_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
    )
    top_k = int(os.getenv("RAG_TOP_K", "5"))

    matches = retrieve_context(
        query=case_summary,
        chroma_dir=chroma_dir,
        collection_name=collection,
        embedding_model=embed_model,
        top_k=top_k,
    )
    return format_rag_context(matches)
