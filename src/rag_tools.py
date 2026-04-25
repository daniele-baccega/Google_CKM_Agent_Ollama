"""Tool functions for retrieving and injecting RAG context inside ADK agents."""

import os
import re
from typing import Optional

from google.adk.agents.callback_context import CallbackContext
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.genai import types

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
    reranker_model = os.getenv(
        "RAG_RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"
    )
    top_k = int(os.getenv("RAG_TOP_K", "5"))
    # Enable reranking by default for better citation quality
    enable_reranking = os.getenv("RAG_ENABLE_RERANKING", "true").lower() == "true"

    matches = retrieve_context(
        query=case_summary,
        chroma_dir=chroma_dir,
        collection_name=collection,
        embedding_model=embed_model,
        top_k=top_k,
        reranker_model=reranker_model,
        enable_reranking=enable_reranking,
    )
    result = format_rag_context(matches)
    return result


def _extract_text_from_content(content: types.Content) -> str:
    """Extract plain text from a Content object."""
    text_parts = []
    for part in content.parts or []:
        if getattr(part, "text", None):
            text_parts.append(part.text)
    return "\n".join(text_parts).strip()


def _latest_user_text(llm_request: LlmRequest) -> str:
    """Return the latest user-authored text in the current request."""
    for content in reversed(llm_request.contents):
        if getattr(content, "role", None) == "user":
            text = _extract_text_from_content(content)
            if text:
                return text
    return ""


def _all_user_text(llm_request: LlmRequest) -> str:
    """Return concatenated user text across the current request."""
    chunks = []
    for content in llm_request.contents:
        if getattr(content, "role", None) == "user":
            text = _extract_text_from_content(content)
            if text:
                chunks.append(text)
    return "\n\n".join(chunks)


def _clean_noise(text: str) -> str:
    """Remove common runtime log lines accidentally pasted into chat."""
    cleaned_lines = []
    for line in text.splitlines():
        if "Warning: You are sending unauthenticated requests to the HF Hub" in line:
            continue
        if line.startswith("Loading weights:"):
            continue
        if line.startswith("Batches:"):
            continue
        if "BertModel LOAD REPORT" in line:
            continue
        if line.strip().startswith("embeddings.position_ids"):
            continue
        if line.strip().startswith("Notes:"):
            continue
        if "UNEXPECTED" in line:
            continue
        if "LiteLLM:INFO" in line:
            continue
        if "completion() model=" in line and "provider =" in line:
            continue
        cleaned_lines.append(line)
    return "\n".join(cleaned_lines).strip()


def sanitize_user_input_before_model(
    callback_context: CallbackContext, llm_request: LlmRequest
) -> Optional[LlmResponse]:
    """Remove runtime log artifacts from user content before model reasoning."""
    try:
        updated_contents = []
        for content in llm_request.contents:
            if getattr(content, "role", None) != "user":
                updated_contents.append(content)
                continue

            original_text = _extract_text_from_content(content)
            cleaned_text = _clean_noise(original_text)
            if not cleaned_text:
                continue

            updated_contents.append(
                types.Content(role="user", parts=[types.Part(text=cleaned_text)])
            )

        if updated_contents:
            llm_request.contents = updated_contents
    except Exception as exc:  # pragma: no cover - best-effort safety net
        callback_context.state["input_sanitize_error"] = str(exc)

    return None


def flow_guard_before_model(
    callback_context: CallbackContext, llm_request: LlmRequest
) -> Optional[LlmResponse]:
    """Preserve consultation state and prevent accidental reset to intake."""
    try:
        agent_name = callback_context.agent_name
        latest_user = _clean_noise(_latest_user_text(llm_request)).strip().lower()

        if agent_name == "mediator":
            callback_context.state["consultation_started"] = True

        if agent_name == "intake_coordinator" and latest_user in {"1", "2"}:
            callback_context.state["consultation_started"] = False
            callback_context.state["rag_context_block"] = ""

        consultation_started = bool(callback_context.state.get("consultation_started"))
        if agent_name == "ckm_root_agent" and consultation_started and latest_user == "confirm":
            llm_request.contents.append(
                types.Content(
                    role="user",
                    parts=[
                        types.Part(
                            text=(
                                "FLOW_GUARD: A synthesis has already been generated in this conversation. "
                                "Treat 'Confirm' as 'keep current consultation context'. "
                                "Do NOT restart intake or ask to paste case again. "
                                "Offer expansion options A/B/C, Back, or ask whether to start a NEW case with 1/2."
                            )
                        )
                    ],
                )
            )
    except Exception as exc:  # pragma: no cover - best-effort safety net
        callback_context.state["flow_guard_error"] = str(exc)

    return None


def _extract_locked_case_facts(text: str) -> str:
    """Extract high-confidence demographics and medication facts from text."""
    text = _clean_noise(text)

    age = None
    sex = None

    # Typical pattern: "65-year-old male"
    age_sex = re.search(
        r"\b(\d{1,3})\s*[- ]?year[- ]?old\s+(male|female|man|woman|m|f)\b",
        text,
        flags=re.IGNORECASE,
    )
    if age_sex:
        age = age_sex.group(1)
        raw_sex = age_sex.group(2).lower()
        sex = "M" if raw_sex in {"male", "man", "m"} else "F"

    # Fallback: separate age and sex hints
    if not age:
        age_only = re.search(r"\b(\d{1,3})\s*(?:yo|y/o|years? old)\b", text, re.IGNORECASE)
        if age_only:
            age = age_only.group(1)
    if not sex:
        if re.search(r"\bmale\b|\bman\b", text, re.IGNORECASE):
            sex = "M"
        elif re.search(r"\bfemale\b|\bwoman\b", text, re.IGNORECASE):
            sex = "F"

    meds = None
    meds_line = re.search(
        r"\bcurrent medications?\s*:\s*(.+)", text, flags=re.IGNORECASE
    )
    if meds_line:
        meds = meds_line.group(1).strip()

    facts = []
    if age and sex:
        facts.append(f"Demographics: {age}{sex}")
    elif age:
        facts.append(f"Age: {age}")
    elif sex:
        facts.append(f"Sex: {sex}")

    if meds:
        facts.append(f"Current medications: {meds}")

    if not facts:
        return ""

    bullet_lines = "\n".join(f"- {fact}" for fact in facts)
    return (
        "PATIENT_DATA (do not alter or contradict):\n"
        f"{bullet_lines}\n"
        "If any generated text conflicts with these facts, correct the output to match these facts."
    )


def _has_rag_context(llm_request: LlmRequest) -> bool:
    """Check whether the request already contains an explicit RAG_CONTEXT block."""
    for content in llm_request.contents:
        text = _extract_text_from_content(content)
        if "RAG_CONTEXT:" in text:
            return True
    return False


def _has_locked_case_facts(llm_request: LlmRequest) -> bool:
    """Check whether the request already contains an explicit PATIENT_DATA block."""
    for content in llm_request.contents:
        text = _extract_text_from_content(content)
        if "PATIENT_DATA (do not alter or contradict):" in text:
            return True
    return False


def inject_locked_case_facts_before_model(
    callback_context: CallbackContext, llm_request: LlmRequest
) -> Optional[LlmResponse]:
    """Inject high-confidence demographics/meds facts to reduce demographic drift."""
    try:
        if _has_locked_case_facts(llm_request):
            return None

        text = _all_user_text(llm_request)
        if not text:
            return None

        locked_facts = _extract_locked_case_facts(text)
        if not locked_facts:
            return None

        llm_request.contents.append(
            types.Content(
                role="user",
                parts=[types.Part(text=locked_facts)],
            )
        )
    except Exception as exc:  # pragma: no cover - best-effort safety net
        callback_context.state["locked_facts_error"] = str(exc)

    return None


def inject_rag_context_before_model(
    callback_context: CallbackContext, llm_request: LlmRequest
) -> Optional[LlmResponse]:
    """Inject RAG_CONTEXT before model calls for deterministic specialist inputs.

    This avoids relying on an upstream LLM tool call to include retrieved rules.
    """
    try:
        agent_name = callback_context.agent_name
        # Keep mediator input focused on specialist outputs; inject RAG only for specialists.
        if agent_name not in {"cardiologist", "nephrologist", "diabetologist"}:
            return None

        full_user_text = _all_user_text(llm_request)

        if _has_rag_context(llm_request):
            return None

        query_text = _clean_noise(_latest_user_text(llm_request))
        if len(query_text) < 80:
            query_text = _clean_noise(full_user_text)

        # Only skip if TRULY empty (control commands like "A", "B", "C" shouldn't get RAG)
        if not query_text or len(query_text) < 10:
            return None

        rag_context = retrieve_rag_context(query_text)

        if not rag_context or rag_context.strip() == "RAG_CONTEXT: None":
            return None

        # Extract actual source labels for an explicit footer
        actual_sources = set()
        source_matches = re.findall(r"(?:E\d+|\d+)\.\s+\[([^\]]+)\]", rag_context)
        actual_sources.update(source_matches)
        
        # Build explicit source footer
        sources_footer = ""
        if actual_sources:
            sources_list = ", ".join(sorted(list(actual_sources)))
            sources_footer = f"\n\n---\n[AUTHORITATIVE_SOURCES: {sources_list}]\nDo NOT invent or cite guideline versions. Use ONLY the sources listed above when populating section F)."
        
        llm_request.contents.append(
            types.Content(
                role="user",
                parts=[
                    types.Part(
                        text=(
                            "Supplemental guideline context for this case:\n"
                            f"{rag_context}{sources_footer}"
                        )
                    )
                ],
            )
        )
    except Exception as exc:  # pragma: no cover - best-effort safety net
        callback_context.state["rag_injection_error"] = str(exc)

    return None


def hide_rag_internals_after_model(
    callback_context: CallbackContext, llm_response: LlmResponse
) -> Optional[LlmResponse]:
    """Clean up internal RAG machinery but preserve source attribution.
    
    Removes the verbose RAG_CONTEXT blocks (internal details) but keeps
    [AUTHORITATIVE_SOURCES] footer for transparency and mediator's section F).
    """
    try:
        if not llm_response or not llm_response.text:
            return None

        text = llm_response.text
        
        # Remove RAG_CONTEXT: ... blocks (the verbose internal machinery)
        # Pattern: "RAG_CONTEXT:" followed by anything until double newline or end
        text = re.sub(
            r"RAG_CONTEXT:.*?(?=\n\n|\Z)",
            "",
            text,
            flags=re.DOTALL
        )
        
        # Keep [AUTHORITATIVE_SOURCES: ...] footer!
        # This is needed for:
        # 1. Mediator to extract real PDF names for section F)
        # 2. Users to see which sources were cited
        # (It's not internal machinery—it's the source attribution)
        
        # Remove stray "---" dividers that may separate RAG content
        text = re.sub(r"^---\s*$", "", text, flags=re.MULTILINE)
        
        # Clean up excess blank lines (more than 2 consecutive)
        text = re.sub(r"\n\n\n+", "\n\n", text)
        
        # Strip leading/trailing whitespa':!*.pdf'ce
        text = text.strip()
        
        # Update response
        llm_response.text = text
    except Exception as exc:  # pragma: no cover - best-effort safety net
        callback_context.state["rag_hiding_error"] = str(exc)

    return None
