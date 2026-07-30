"""Utility functions for CKM agents."""

import time
import json
import os
from typing import Optional, Dict, Any
from google.adk.agents.callback_context import CallbackContext
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse


def _ensure_telemetry_state(state: Dict[str, Any], agent_name: str) -> Dict[str, Any]:
    telemetry = state.setdefault("telemetry", {})
    agents = telemetry.setdefault("agents", {})
    return agents.setdefault(agent_name, {})


def save_telemetry_data(state: Dict[str, Any], file_path: str = "telemetry_history.json"):
    """Save the current telemetry state to a JSON file for later analysis."""
    telemetry = state.get("telemetry", {})
    if not telemetry:
        return

    run_data = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "data": telemetry
    }

    history = []
    if os.path.exists(file_path):
        try:
            with open(file_path, "r") as f:
                history = json.load(f)
                if not isinstance(history, list):
                    history = []
        except (json.JSONDecodeError, IOError):
            history = []

    history.append(run_data)

    with open(file_path, "w") as f:
        json.dump(history, f, indent=2)


def _extract_text_from_response(llm_response: Optional[LlmResponse]) -> str:
    if not llm_response:
        return ""

    content = getattr(llm_response, "content", None)
    parts = getattr(content, "parts", None) if content else None
    if not parts:
        return ""

    texts = []
    for part in parts:
        text = getattr(part, "text", None)
        if text:
            texts.append(text)

    return "\n".join(texts).strip()


def _ns_to_s(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value) / 1e9
    except (TypeError, ValueError):
        return None


def _tokens_per_second(tokens: Any, latency_s: Any) -> Optional[float]:
    """Compute tokens per second."""
    try:
        tokens = float(tokens)
        latency_s = float(latency_s)
        if latency_s <= 0:
            return None
        return tokens / latency_s
    except (TypeError, ValueError):
        return None


def telemetry_before_model(
    callback_context: CallbackContext,
    llm_request: LlmRequest
) -> Optional[LlmResponse]:
    """Start timing for the current agent."""
    agent_name = callback_context.agent_name
    stats = _ensure_telemetry_state(callback_context.state, agent_name)

    now = time.time()
    stats["start_time"] = now

    if "e2e_start_time" not in callback_context.state.get("telemetry", {}):
        callback_context.state.setdefault("telemetry", {})["e2e_start_time"] = now

    return None


def telemetry_after_model(
    callback_context: CallbackContext,
    llm_response: LlmResponse
) -> Optional[LlmResponse]:
    """End timing and collect token counts for the current agent."""
    agent_name = callback_context.agent_name
    end_time = time.time()

    stats = _ensure_telemetry_state(callback_context.state, agent_name)

    start_time = stats.get("start_time")
    if start_time:
        stats["latency"] = end_time - start_time
        stats["end_time"] = end_time

    usage = getattr(llm_response, "usage_metadata", None) if llm_response else None
    if usage:
        stats["prompt_tokens"] = getattr(usage, "prompt_token_count", 0) or 0
        stats["completion_tokens"] = getattr(usage, "candidates_token_count", 0) or 0
        stats["total_tokens"] = getattr(usage, "total_token_count", 0) or 0

        # Optional hidden reasoning / thinking tokens for providers that expose them
        stats["generated_tokens"] = getattr(usage, "thoughts_token_count", 0) or 0

        # Total generated tokens = visible output + hidden reasoning tokens
        stats["total_generated_tokens"] = (
            stats["completion_tokens"] + stats["generated_tokens"]
        )

    text = _extract_text_from_response(llm_response)
    if text:
        stats["response_length"] = len(text)

    model_version = getattr(llm_response, "model_version", None) if llm_response else None
    if model_version:
        stats["model_version"] = model_version

    custom_metadata = getattr(llm_response, "custom_metadata", None) if llm_response else None
    if isinstance(custom_metadata, dict):
        stats["custom_metadata"] = custom_metadata

        load_duration_s = _ns_to_s(custom_metadata.get("load_duration"))
        if load_duration_s is not None:
            stats["model_loading_time"] = load_duration_s

        total_duration_s = _ns_to_s(custom_metadata.get("total_duration"))
        if total_duration_s is not None:
            stats["total_duration"] = total_duration_s

        prompt_eval_duration_s = _ns_to_s(custom_metadata.get("prompt_eval_duration"))
        if prompt_eval_duration_s is not None:
            stats["prompt_eval_duration"] = prompt_eval_duration_s

        eval_duration_s = _ns_to_s(custom_metadata.get("eval_duration"))
        if eval_duration_s is not None:
            stats["eval_duration"] = eval_duration_s

        prompt_eval_count = custom_metadata.get("prompt_eval_count")
        if prompt_eval_count is not None:
            stats["prompt_eval_count"] = prompt_eval_count

        eval_count = custom_metadata.get("eval_count")
        if eval_count is not None:
            stats["eval_count"] = eval_count

        # Ollama-style prompt evaluation speed
        if stats.get("prompt_eval_duration", 0) > 0 and stats.get("prompt_eval_count", 0) > 0:
            stats["prompt_tps"] = stats["prompt_eval_count"] / stats["prompt_eval_duration"]

        # Ollama-style generation speed
        if stats.get("eval_duration", 0) > 0 and stats.get("eval_count", 0) > 0:
            stats["generation_tps"] = stats["eval_count"] / stats["eval_duration"]

    # Generic end-to-end generated output tokens per second
    output_tps = _tokens_per_second(
        stats.get("total_generated_tokens", 0),
        stats.get("latency", 0)
    )
    if output_tps is not None:
        stats["tokens_per_second"] = output_tps

    if agent_name == "ckm_root_agent":
        callback_context.state["telemetry"]["e2e_end_time"] = end_time

    return None


def get_telemetry_report(state: Dict[str, Any]) -> str:
    """Generate a formatted telemetry report from the state."""
    telemetry = state.get("telemetry", {})
    if not telemetry:
        return ""

    agents_stats = telemetry.get("agents", {})
    report = ["\n\n---", "### 📊 Performance Statistics"]

    specialists = ["cardiologist", "nephrologist", "diabetologist"]
    spec_latencies = [agents_stats.get(s, {}).get("latency", 0) for s in specialists]

    e2e_start = telemetry.get("e2e_start_time")
    e2e_end = telemetry.get("e2e_end_time")
    if e2e_start and not e2e_end:
        e2e_end = time.time()

    if e2e_start and e2e_end:
        report.append(f"- **End-to-End Latency:** {e2e_end - e2e_start:.2f}s")

    if any(spec_latencies):
        report.append(f"- **Specialist Latency (parallel):** {max(spec_latencies):.2f}s")
        for s in specialists:
            s_stats = agents_stats.get(s, {})
            if s_stats:
                report.append(f"  - *{s.capitalize()}:* {s_stats.get('latency', 0):.2f}s")

    mediator_stats = agents_stats.get("mediator", {})
    if mediator_stats:
        report.append(f"- **Mediator Latency:** {mediator_stats.get('latency', 0):.2f}s")

    total_prompt = 0
    total_completion = 0
    total_generated = 0
    total_all = 0
    valid_agents = ["intake_coordinator"] + specialists + ["mediator", "ckm_root_agent"]

    for agent in valid_agents:
        stats = agents_stats.get(agent, {})
        total_prompt += stats.get("prompt_tokens", 0)
        total_completion += stats.get("completion_tokens", 0)
        total_generated += stats.get("generated_tokens", 0)
        total_all += stats.get("total_tokens", 0)

    if total_prompt or total_completion or total_generated or total_all:
        report.append(
            f"- **Total Token Counts:** {total_prompt} prompt / "
            f"{total_completion} completion / "
            f"{total_generated} hidden-generated / "
            f"{total_all} total"
        )

    report.append("- **Tokens per second (Generation):**")
    for agent in valid_agents:
        stats = agents_stats.get(agent, {})
        
        # Prefer provider-reported generation TPS
        tps = stats.get("generation_tps")
        if tps is None:
            # Fallback to calculated tokens per second
            tps = stats.get("tokens_per_second")
            
        if tps is not None:
            report.append(f"  - *{agent.capitalize()}:* {tps:.2f} tok/s")

    if mediator_stats:
        report.append(f"- **Final Response Length:** {mediator_stats.get('response_length', 0)} characters")

    load_times = [agents_stats.get(a, {}).get("model_loading_time", 0) for a in specialists + ["mediator"]]
    active_load_times = [t for t in load_times if t and t > 0]
    if active_load_times:
        report.append(f"- **Avg Model Loading Time:** {sum(active_load_times) / len(active_load_times):.2f}s")
        for agent in specialists + ["mediator"]:
            agent_stats = agents_stats.get(agent, {})
            load_time = agent_stats.get("model_loading_time")
            if load_time and load_time > 0:
                report.append(f"  - *{agent.capitalize()} load time:* {load_time:.2f}s")
    else:
        report.append("- **Model Loading Time:** Not reported by provider")

    report.append("- **Provider-reported speeds:**")
    for agent in specialists + ["mediator", "ckm_root_agent"]:
        stats = agents_stats.get(agent, {})
        prompt_tps = stats.get("prompt_tps")
        generation_tps = stats.get("generation_tps")
        if prompt_tps is not None or generation_tps is not None:
            prompt_tps_str = f"{prompt_tps:.2f}" if prompt_tps is not None else "n/a"
            generation_tps_str = f"{generation_tps:.2f}" if generation_tps is not None else "n/a"
            report.append(
                f"  - *{agent.capitalize()}:* "
                f"{prompt_tps_str} prompt tok/s, {generation_tps_str} gen tok/s"
            )

    report.append("---")
    return "\n".join(report)