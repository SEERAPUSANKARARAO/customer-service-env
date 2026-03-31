"""
evaluate.py
Official evaluation script for the Customer Service OpenEnv.
Runs each task multiple times across different seeds and reports
average scores, consistency, and per-task breakdowns.

Usage:
  python evaluate.py                  # 3 runs per task, default seeds
  python evaluate.py --runs 5         # 5 runs per task
  python evaluate.py --task hard      # evaluate one task only
"""

import os
import sys
import json
import argparse
import requests
import statistics

try:
    from dotenv import load_dotenv
    load_dotenv(override=False)
except ImportError:
    pass

# Reuse the agent from inference.py
sys.path.insert(0, os.path.dirname(__file__))
from inference import run_episode, MODEL_NAME, API_BASE_URL

EVAL_SEEDS = [42, 100, 777, 1234, 9999]


def check_server():
    """Make sure the OpenEnv server is running before evaluating."""
    try:
        r = requests.get(f"{API_BASE_URL}/health", timeout=5)
        r.raise_for_status()
        print(f"  [eval] Server OK: {API_BASE_URL}")
        return True
    except Exception as e:
        print(f"\n  [ERROR] Cannot reach server at {API_BASE_URL}")
        print(f"  Start it with: uvicorn app.main:app --host 0.0.0.0 --port 7860")
        return False


def evaluate_task(task_id: str, runs: int) -> dict:
    """Run a task `runs` times and return stats."""
    seeds  = EVAL_SEEDS[:runs]
    scores = []

    print(f"\n  {'─'*50}")
    print(f"  Evaluating task: {task_id.upper()}  ({runs} runs)")
    print(f"  {'─'*50}")

    for seed in seeds:
        score = run_episode(task_id=task_id, seed=seed, verbose=True)
        scores.append(score)

    avg    = sum(scores) / len(scores)
    stddev = statistics.stdev(scores) if len(scores) > 1 else 0.0
    mn     = min(scores)
    mx     = max(scores)

    return {
        "task_id": task_id,
        "runs":    runs,
        "seeds":   seeds,
        "scores":  scores,
        "avg":     round(avg,    4),
        "stddev":  round(stddev, 4),
        "min":     round(mn,     4),
        "max":     round(mx,     4),
        "passed":  avg >= 0.7,
    }


def print_report(results: list):
    print(f"\n{'='*60}")
    print(f"  EVALUATION REPORT")
    print(f"  Model : {MODEL_NAME}")
    print(f"  Server: {API_BASE_URL}")
    print(f"{'='*60}")

    overall_avgs = []
    for r in results:
        bar     = "█" * int(r["avg"] * 20) + "░" * (20 - int(r["avg"] * 20))
        status  = "✓ PASS" if r["passed"] else "✗ FAIL"
        print(f"\n  {r['task_id'].upper():<8} {status}")
        print(f"  [{bar}] avg={r['avg']:.4f}  min={r['min']:.4f}  max={r['max']:.4f}  σ={r['stddev']:.4f}")
        print(f"  Scores per seed: " + "  ".join(
            f"seed={s}→{sc:.2f}" for s, sc in zip(r["seeds"], r["scores"])
        ))
        overall_avgs.append(r["avg"])

    grand = sum(overall_avgs) / len(overall_avgs)
    bar   = "█" * int(grand * 20) + "░" * (20 - int(grand * 20))

    print(f"\n{'─'*60}")
    print(f"  GRAND AVERAGE : [{bar}] {grand:.4f} / 1.0")
    print(f"  OVERALL STATUS: {'✓ READY TO SUBMIT' if grand >= 0.7 else '✗ NEEDS IMPROVEMENT'}")
    print(f"{'='*60}\n")

    # Save JSON report
    report = {
        "model":         MODEL_NAME,
        "server":        API_BASE_URL,
        "grand_average": grand,
        "tasks":         results,
    }
    with open("evaluation_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"  Full report saved to: evaluation_report.json\n")

    return grand


def main():
    parser = argparse.ArgumentParser(description="Customer Service OpenEnv evaluator")
    parser.add_argument("--task",  choices=["easy", "medium", "hard", "all"], default="all")
    parser.add_argument("--runs",  type=int, default=3, help="Runs per task (default: 3)")
    args = parser.parse_args()

    if not check_server():
        sys.exit(1)

    tasks   = ["easy", "medium", "hard"] if args.task == "all" else [args.task]
    results = [evaluate_task(t, args.runs) for t in tasks]
    grand   = print_report(results)

    sys.exit(0 if grand >= 0.7 else 1)


if __name__ == "__main__":
    main()
