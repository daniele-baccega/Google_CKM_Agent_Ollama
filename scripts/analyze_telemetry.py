import json
import os
import statistics
from typing import List, Dict, Any, DefaultDict
from collections import defaultdict

def analyze_telemetry(file_path: str = "telemetry_history.json"):
    if not os.path.exists(file_path):
        print(f"File {file_path} not found. Run the agent first.")
        return

    try:
        with open(file_path, "r") as f:
            history = json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        print(f"Error reading {file_path}: {e}")
        return

    if not history or not isinstance(history, list):
        print("No telemetry data found.")
        return

    num_runs = len(history)
    print(f"Analyzing {num_runs} runs...\n")

    # Overall metrics
    overall_metrics = {
        "e2e_latency": [],
        "total_prompt_tokens": [],
        "total_completion_tokens": [],
        "total_generated_tokens": [], # completion
        "total_tokens": [],
        "avg_model_loading_time": []
    }

    # Per-agent metrics
    agent_metrics = defaultdict(lambda: {
        "latency": [],
        "prompt_tokens": [],
        "completion_tokens": [],
        "total_tokens": [],
        "tokens_per_second": [],
        "prompt_tps": [],
        "generation_tps": [],
        "model_loading_time": [],
        "response_length": []
    })

    critical_path_latencies = []

    for run in history:
        data = run.get("data", {})
        agents = data.get("agents", {})
        
        # E2E Latency (Wall-clock)
        e2e_start = data.get("e2e_start_time")
        e2e_end = data.get("e2e_end_time")
        if e2e_start and e2e_end:
            overall_metrics["e2e_latency"].append(e2e_end - e2e_start)

        # Critical Path Calculation
        # Path: root + intake + max(specialists) + mediator
        root_lat = agents.get("ckm_root_agent", {}).get("latency", 0)
        intake_lat = agents.get("intake_coordinator", {}).get("latency", 0)
        mediator_lat = agents.get("mediator", {}).get("latency", 0)
        
        specialist_lats = [
            stats.get("latency", 0) 
            for name, stats in agents.items() 
            if name in ["cardiologist", "nephrologist", "diabetologist"]
        ]
        max_specialist = max(specialist_lats) if specialist_lats else 0
        
        if any([root_lat, intake_lat, max_specialist, mediator_lat]):
            critical_path_latencies.append(root_lat + intake_lat + max_specialist + mediator_lat)

        # Totals for this run
        run_prompt = 0
        run_completion = 0
        run_total = 0
        run_load_times = []

        for agent, stats in agents.items():
            # Basic stats
            lat = stats.get("latency")
            if lat is not None:
                agent_metrics[agent]["latency"].append(lat)

            p_tokens = stats.get("prompt_tokens", 0)
            c_tokens = stats.get("completion_tokens", 0)
            g_tokens = stats.get("generated_tokens", 0) # Generated tokens
            total = stats.get("total_tokens", 0)

            agent_metrics[agent]["prompt_tokens"].append(p_tokens)
            agent_metrics[agent]["completion_tokens"].append(c_tokens)
            agent_metrics[agent]["total_tokens"].append(total)

            run_prompt += p_tokens
            run_completion += c_tokens
            run_total += total

            # Speed metrics
            tps = stats.get("tokens_per_second")
            if tps is not None:
                agent_metrics[agent]["tokens_per_second"].append(tps)
            
            p_tps = stats.get("prompt_tps")
            if p_tps is not None:
                agent_metrics[agent]["prompt_tps"].append(p_tps)
            
            g_tps = stats.get("generation_tps")
            if g_tps is not None:
                agent_metrics[agent]["generation_tps"].append(g_tps)

            # Loading time
            load_time = stats.get("model_loading_time")
            if load_time is not None and load_time > 0:
                agent_metrics[agent]["model_loading_time"].append(load_time)
                run_load_times.append(load_time)

            # Response length
            resp_len = stats.get("response_length")
            if resp_len is not None:
                agent_metrics[agent]["response_length"].append(resp_len)

        # Aggregate run totals
        if run_total > 0:
            overall_metrics["total_prompt_tokens"].append(run_prompt)
            overall_metrics["total_completion_tokens"].append(run_completion)
            overall_metrics["total_generated_tokens"].append(run_completion + g_tokens)
            overall_metrics["total_tokens"].append(run_total)
        
        if run_load_times:
            overall_metrics["avg_model_loading_time"].append(statistics.mean(run_load_times))

    print("=== Overall Performance Statistics ===")
    for metric, values in overall_metrics.items():
        if values:
            avg = statistics.mean(values)
            stdev = statistics.stdev(values) if len(values) > 1 else 0.0
            unit = "s" if "latency" in metric or "time" in metric else "tokens"
            print(f"{metric.replace('_', ' ').capitalize()}: {avg:.2f} ± {stdev:.2f} {unit}")
    
    if critical_path_latencies:
        avg_cp = statistics.mean(critical_path_latencies)
        stdev_cp = statistics.stdev(critical_path_latencies) if len(critical_path_latencies) > 1 else 0.0
        print(f"Theoretical critical path: {avg_cp:.2f} ± {stdev_cp:.2f} s")
    
    print("\n=== Per-Agent Statistics ===")
    # Sort agents for consistent output
    sorted_agents = sorted(agent_metrics.keys())
    for agent in sorted_agents:
        metrics = agent_metrics[agent]
        print(f"\n> Agent: {agent.upper()}")
        
        # Primary metrics for agent
        primary = ["latency", "tokens_per_second", "generation_tps", "model_loading_time", "total_tokens", "completion_tokens"]
        for m in primary:
            values = metrics.get(m, [])
            if values:
                avg = statistics.mean(values)
                stdev = statistics.stdev(values) if len(values) > 1 else 0.0
                unit = "s" if m in ["latency", "model_loading_time"] else ("tok/s" if "tps" in m or "second" in m else "tokens")
                print(f"  - {m.replace('_', ' ').capitalize()}: {avg:.2f} ± {stdev:.2f} {unit}")
        
        # Optional: prompt vs completion
        p_vals = metrics.get("prompt_tokens", [])
        c_vals = metrics.get("completion_tokens", [])
        p = statistics.mean(p_vals) if p_vals else 0
        c = statistics.mean(c_vals) if c_vals else 0
        if p > 0 or c > 0:
            print(f"  - Avg Tokens Breakdown: {p:.0f} prompt / {c:.0f} completion")

if __name__ == "__main__":
    analyze_telemetry()
