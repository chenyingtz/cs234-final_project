"""
Aggregate and summarize evaluation results from multiple benchmark runs.
Creates a comprehensive report comparing all models across all benchmarks.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Any
import pandas as pd


def load_results_json(results_path: Path) -> Dict[str, Any]:
    """Load results from lm-eval results.json file."""
    try:
        with open(results_path, "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"Warning: Could not load {results_path}: {e}")
        return {}


def extract_metric(results: Dict[str, Any], task: str, metric: str = "acc") -> float | None:
    """
    Extract metric from lm-eval results.
    Results structure: {task: {metric: value, ...}, ...}
    """
    # lm-eval stores metrics under a top-level "results" key:
    # {"results": {task: {...}}, ...}
    results_root = results.get("results", results)

    if task not in results_root:
        return None
    
    task_results = results_root[task]
    if isinstance(task_results, dict):
        # Try common metric names
        for m in [metric, f"{task}_{metric}", "acc", "accuracy", "exact_match"]:
            if m in task_results:
                value = task_results[m]
                if isinstance(value, dict) and "acc" in value:
                    return value["acc"]
                elif isinstance(value, (int, float)):
                    return float(value)
        # If no metric found, try to get first numeric value
        for key, value in task_results.items():
            if isinstance(value, (int, float)):
                return float(value)
    
    return None


def aggregate_evaluation_summary(summary_file: Path) -> pd.DataFrame:
    """Load evaluation summary JSON and create aggregated DataFrame."""
    with open(summary_file, "r") as f:
        data = json.load(f)
    
    rows = []
    for result in data.get("results", []):
        if not result.get("success"):
            continue
        
        output_path = Path(result["output_path"])
        results_file = output_path / "results.json"
        
        if not results_file.exists():
            # lm-eval often saves files like */results_YYYY-MM-DD*.json inside a
            # model-specific subdirectory. Fall back to the first such file.
            candidates = list(output_path.rglob("results_*.json"))
            if not candidates:
                continue
            results_file = candidates[0]
        
        results_data = load_results_json(results_file)
        benchmark = result["benchmark"]
        
        metric_value = extract_metric(results_data, benchmark, "acc")
        
        rows.append({
            "model": result.get("model_key", "unknown"),
            "benchmark": benchmark,
            "mode": result["mode"],
            "metric": metric_value,
            "elapsed_time": result.get("elapsed_time", 0),
            "output_path": str(output_path),
        })
    
    return pd.DataFrame(rows)


def create_summary_table(df: pd.DataFrame) -> str:
    """Create a formatted summary table from DataFrame."""
    if df.empty:
        return "No results to display."
    
    # Pivot table: models x (benchmark, mode)
    summary_lines = []
    summary_lines.append("=" * 100)
    summary_lines.append("Evaluation Results Summary")
    summary_lines.append("=" * 100)
    
    # Group by benchmark and mode
    for benchmark in df["benchmark"].unique():
        summary_lines.append(f"\nBenchmark: {benchmark.upper()}")
        summary_lines.append("-" * 100)
        
        for mode in ["greedy", "avg1", "avg32"]:
            mode_df = df[(df["benchmark"] == benchmark) & (df["mode"] == mode)]
            if mode_df.empty:
                continue
            
            summary_lines.append(f"\n  Mode: {mode}")
            summary_lines.append(f"  {'Model':<20} {'Metric':<15} {'Time (s)':<15}")
            summary_lines.append(f"  {'-'*50}")
            
            for _, row in mode_df.iterrows():
                metric_str = f"{row['metric']:.4f}" if row['metric'] is not None else "N/A"
                summary_lines.append(f"  {row['model']:<20} {metric_str:<15} {row['elapsed_time']:.1f}")
    
    summary_lines.append("\n" + "=" * 100)
    return "\n".join(summary_lines)


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate evaluation results from benchmark runs"
    )
    parser.add_argument(
        "summary_file",
        type=str,
        help="Path to evaluation summary JSON file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file for aggregated results (default: summary_file with .csv/.txt)"
    )
    parser.add_argument(
        "--format",
        choices=["csv", "txt", "json", "all"],
        default="all",
        help="Output format (default: all)"
    )
    
    args = parser.parse_args()
    
    summary_path = Path(args.summary_file)
    if not summary_path.exists():
        print(f"Error: Summary file not found: {summary_path}")
        return
    
    print(f"Loading summary from: {summary_path}")
    df = aggregate_evaluation_summary(summary_path)
    
    if df.empty:
        print("Warning: No successful evaluations found in summary.")
        return
    
    print(f"\nLoaded {len(df)} successful evaluations")
    
    # Determine output paths
    if args.output:
        base_path = Path(args.output)
    else:
        base_path = summary_path.with_suffix("")
    
    # Generate outputs
    if args.format in ["csv", "all"]:
        csv_path = base_path.with_suffix(".csv")
        df.to_csv(csv_path, index=False)
        print(f"CSV saved to: {csv_path}")
    
    if args.format in ["txt", "all"]:
        txt_path = base_path.with_suffix(".txt")
        summary_table = create_summary_table(df)
        with open(txt_path, "w") as f:
            f.write(summary_table)
        print(f"Text summary saved to: {txt_path}")
        print("\n" + summary_table)
    
    if args.format in ["json", "all"]:
        json_path = base_path.with_suffix(".json")
        df_dict = df.to_dict(orient="records")
        with open(json_path, "w") as f:
            json.dump(df_dict, f, indent=2)
        print(f"JSON saved to: {json_path}")


if __name__ == "__main__":
    main()
