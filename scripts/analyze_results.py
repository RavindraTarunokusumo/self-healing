#!/usr/bin/env python3
"""
Statistical Analysis Script for Multi-Run Benchmark Results

Aggregates results across multiple runs per configuration and computes:
- Mean/std/95% CI for pass rate
- Zero-shot extraction (iterations==1)
- Self-healing lift statistics
- Cost metrics
- Welch's t-test comparisons between configurations

Outputs:
- Console summary table
- results/summary.json
- results/comparison.csv
"""

import json
import os
import glob
import argparse
from dataclasses import dataclass
from typing import Dict, List, Optional
import math

# Try to import scipy for statistical tests, fall back gracefully
try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("Warning: scipy not available. Statistical tests will be skipped.")


@dataclass
class RunMetrics:
    """Metrics extracted from a single benchmark run."""
    config_name: str
    run_number: int
    total_problems: int
    passed: int
    pass_rate: float
    zero_shot_passed: int
    zero_shot_rate: float
    self_healed: int
    self_heal_lift: float
    total_cost: float
    coder_cost: float
    critic_cost: float
    cost_per_pass: float
    avg_iterations: float
    total_tokens: int


@dataclass
class ConfigStats:
    """Aggregated statistics for a configuration across multiple runs."""
    config_name: str
    runs: int
    pass_rate_mean: float
    pass_rate_std: float
    pass_rate_ci_low: float
    pass_rate_ci_high: float
    zero_shot_mean: float
    zero_shot_std: float
    self_heal_lift_mean: float
    self_heal_lift_std: float
    total_cost_mean: float
    total_cost_std: float
    cost_per_pass_mean: float
    cost_per_pass_std: float
    avg_iterations_mean: float


def calculate_ci_95(values: List[float]) -> tuple:
    """Calculate 95% confidence interval using t-distribution."""
    n = len(values)
    if n < 2:
        mean = values[0] if values else 0
        return mean, mean

    mean = sum(values) / n
    variance = sum((x - mean) ** 2 for x in values) / (n - 1)
    std = math.sqrt(variance)
    se = std / math.sqrt(n)

    if SCIPY_AVAILABLE:
        t_crit = stats.t.ppf(0.975, n - 1)
    else:
        # Approximate t-critical for small samples
        t_table = {2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228}
        t_crit = t_table.get(n, 1.96)

    margin = t_crit * se
    return mean - margin, mean + margin


def extract_run_metrics(filepath: str, config_name: str, run_number: int) -> Optional[RunMetrics]:
    """Extract metrics from a single result JSON file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Extract components
        problems_results = data[1].get("problems_results", [])
        summary = data[2].get("benchmark_summary", {})

        if not problems_results or not summary:
            print(f"Warning: Empty results in {filepath}")
            return None

        # Basic metrics from summary
        problems = summary.get("problems", {})
        total_problems = problems.get("total", 0)
        passed = problems.get("passed", 0)
        pass_rate = problems.get("pass_rate", 0.0)

        iterations = summary.get("iterations", {})
        avg_iterations = iterations.get("average", 0.0)

        cost = summary.get("cost", {})
        total_cost = cost.get("total_cost", 0.0)
        coder_cost = cost.get("coder_cost", 0.0)
        critic_cost = cost.get("critic_cost", 0.0)
        cost_per_pass = cost.get("cost_per_pass", 0.0)

        token_usage = summary.get("token_usage", {})
        total_tokens = token_usage.get("total", {}).get("total_tokens", 0)

        # Extract zero-shot metrics (problems with iterations == 1)
        zero_shot_problems = [p for p in problems_results if p.get("iterations", 0) == 1]
        zero_shot_passed = sum(1 for p in zero_shot_problems if p.get("is_complete", False))
        zero_shot_rate = (zero_shot_passed / len(problems_results) * 100) if len(problems_results) > 0 else 0.0

        # Self-healed = passed with iterations > 1
        self_healed = sum(1 for p in problems_results
                         if p.get("iterations", 0) > 1 and p.get("is_complete", False))

        # Self-heal lift = final pass rate - zero-shot rate
        self_heal_lift = pass_rate - zero_shot_rate

        return RunMetrics(
            config_name=config_name,
            run_number=run_number,
            total_problems=total_problems,
            passed=passed,
            pass_rate=pass_rate,
            zero_shot_passed=zero_shot_passed,
            zero_shot_rate=zero_shot_rate,
            self_healed=self_healed,
            self_heal_lift=self_heal_lift,
            total_cost=total_cost,
            coder_cost=coder_cost,
            critic_cost=critic_cost,
            cost_per_pass=cost_per_pass,
            avg_iterations=avg_iterations,
            total_tokens=total_tokens
        )
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return None


def aggregate_config_stats(runs: List[RunMetrics]) -> ConfigStats:
    """Aggregate metrics across multiple runs for a configuration."""
    if not runs:
        raise ValueError("No runs to aggregate")

    config_name = runs[0].config_name
    n = len(runs)

    # Extract metric lists
    pass_rates = [r.pass_rate for r in runs]
    zero_shot_rates = [r.zero_shot_rate for r in runs]
    self_heal_lifts = [r.self_heal_lift for r in runs]
    total_costs = [r.total_cost for r in runs]
    costs_per_pass = [r.cost_per_pass for r in runs]
    avg_iterations_list = [r.avg_iterations for r in runs]

    # Calculate means
    pass_rate_mean = sum(pass_rates) / n
    zero_shot_mean = sum(zero_shot_rates) / n
    self_heal_lift_mean = sum(self_heal_lifts) / n
    total_cost_mean = sum(total_costs) / n
    cost_per_pass_mean = sum(costs_per_pass) / n
    avg_iterations_mean = sum(avg_iterations_list) / n

    # Calculate standard deviations
    def calc_std(values):
        if len(values) < 2:
            return 0.0
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
        return math.sqrt(variance)

    pass_rate_std = calc_std(pass_rates)
    zero_shot_std = calc_std(zero_shot_rates)
    self_heal_lift_std = calc_std(self_heal_lifts)
    total_cost_std = calc_std(total_costs)
    cost_per_pass_std = calc_std(costs_per_pass)

    # Calculate 95% CI for pass rate
    ci_low, ci_high = calculate_ci_95(pass_rates)

    return ConfigStats(
        config_name=config_name,
        runs=n,
        pass_rate_mean=pass_rate_mean,
        pass_rate_std=pass_rate_std,
        pass_rate_ci_low=ci_low,
        pass_rate_ci_high=ci_high,
        zero_shot_mean=zero_shot_mean,
        zero_shot_std=zero_shot_std,
        self_heal_lift_mean=self_heal_lift_mean,
        self_heal_lift_std=self_heal_lift_std,
        total_cost_mean=total_cost_mean,
        total_cost_std=total_cost_std,
        cost_per_pass_mean=cost_per_pass_mean,
        cost_per_pass_std=cost_per_pass_std,
        avg_iterations_mean=avg_iterations_mean
    )


def perform_statistical_tests(config_runs: Dict[str, List[RunMetrics]]) -> Dict[str, dict]:
    """Perform Welch's t-test between all configuration pairs for pass rate."""
    if not SCIPY_AVAILABLE:
        return {}

    results = {}
    configs = list(config_runs.keys())

    for i, config1 in enumerate(configs):
        for config2 in configs[i+1:]:
            rates1 = [r.pass_rate for r in config_runs[config1]]
            rates2 = [r.pass_rate for r in config_runs[config2]]

            if len(rates1) < 2 or len(rates2) < 2:
                continue

            t_stat, p_value = stats.ttest_ind(rates1, rates2, equal_var=False)

            # Skip if NaN (occurs when variance is zero)
            if math.isnan(t_stat) or math.isnan(p_value):
                continue

            key = f"{config1}_vs_{config2}"
            results[key] = {
                "config1": config1,
                "config2": config2,
                "t_statistic": float(t_stat),
                "p_value": float(p_value),
                "significant": bool(p_value < 0.05)
            }

    return results


def print_summary_table(config_stats: Dict[str, ConfigStats]):
    """Print a formatted summary table to console."""
    print("\n" + "=" * 100)
    print("MULTI-RUN BENCHMARK ANALYSIS SUMMARY")
    print("=" * 100)

    # Header
    print(f"\n{'Config':<15} {'Runs':>5} {'Pass Rate':>20} {'Zero-Shot':>12} {'Self-Heal':>12} {'Cost/Pass':>12} {'Total Cost':>12}")
    print("-" * 100)

    for config_name, stats in sorted(config_stats.items()):
        pass_rate_str = f"{stats.pass_rate_mean:.2f}% +/- {stats.pass_rate_std:.2f}"
        zero_shot_str = f"{stats.zero_shot_mean:.2f}%"
        self_heal_str = f"+{stats.self_heal_lift_mean:.2f}%"
        cost_per_pass_str = f"${stats.cost_per_pass_mean:.5f}"
        total_cost_str = f"${stats.total_cost_mean:.4f}"

        print(f"{config_name:<15} {stats.runs:>5} {pass_rate_str:>20} {zero_shot_str:>12} {self_heal_str:>12} {cost_per_pass_str:>12} {total_cost_str:>12}")

    print("-" * 100)

    # Print 95% CIs
    print("\n95% Confidence Intervals (Pass Rate):")
    for config_name, stats in sorted(config_stats.items()):
        print(f"  {config_name}: [{stats.pass_rate_ci_low:.2f}%, {stats.pass_rate_ci_high:.2f}%]")


def print_statistical_tests(test_results: Dict[str, dict]):
    """Print statistical test results."""
    if not test_results:
        print("\n(Statistical tests skipped - scipy not available, insufficient runs, or zero variance in data)")
        return

    print("\n" + "=" * 60)
    print("STATISTICAL COMPARISONS (Welch's t-test)")
    print("=" * 60)

    for key, result in sorted(test_results.items()):
        sig_marker = "*" if result["significant"] else ""
        print(f"  {result['config1']} vs {result['config2']}: "
              f"p={result['p_value']:.4f} {sig_marker}")

    print("\n  * indicates p < 0.05 (statistically significant)")


def save_summary_json(config_stats: Dict[str, ConfigStats],
                      config_runs: Dict[str, List[RunMetrics]],
                      test_results: Dict[str, dict],
                      output_path: str):
    """Save aggregated results to JSON file."""
    output = {}

    for config_name, stats in config_stats.items():
        runs = config_runs[config_name]
        output[config_name] = {
            "runs": stats.runs,
            "pass_rate": {
                "mean": round(stats.pass_rate_mean, 4),
                "std": round(stats.pass_rate_std, 4),
                "ci_95": [round(stats.pass_rate_ci_low, 4), round(stats.pass_rate_ci_high, 4)],
                "values": [round(r.pass_rate, 4) for r in runs]
            },
            "zero_shot_rate": {
                "mean": round(stats.zero_shot_mean, 4),
                "std": round(stats.zero_shot_std, 4),
                "values": [round(r.zero_shot_rate, 4) for r in runs]
            },
            "self_heal_lift": {
                "mean": round(stats.self_heal_lift_mean, 4),
                "std": round(stats.self_heal_lift_std, 4),
                "values": [round(r.self_heal_lift, 4) for r in runs]
            },
            "total_cost": {
                "mean": round(stats.total_cost_mean, 6),
                "std": round(stats.total_cost_std, 6),
                "values": [round(r.total_cost, 6) for r in runs]
            },
            "cost_per_pass": {
                "mean": round(stats.cost_per_pass_mean, 8),
                "std": round(stats.cost_per_pass_std, 8),
                "values": [round(r.cost_per_pass, 8) for r in runs]
            },
            "avg_iterations": {
                "mean": round(stats.avg_iterations_mean, 4),
                "values": [round(r.avg_iterations, 4) for r in runs]
            }
        }

    # Add statistical tests
    if test_results:
        output["_statistical_tests"] = test_results

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2)

    print(f"\nSummary saved to: {output_path}")


def save_comparison_csv(config_stats: Dict[str, ConfigStats], output_path: str):
    """Save comparison data to CSV file for plotting."""
    headers = [
        "config",
        "runs",
        "pass_rate_mean",
        "pass_rate_std",
        "pass_rate_ci_low",
        "pass_rate_ci_high",
        "zero_shot_mean",
        "zero_shot_std",
        "self_heal_lift_mean",
        "self_heal_lift_std",
        "total_cost_mean",
        "total_cost_std",
        "cost_per_pass_mean",
        "cost_per_pass_std",
        "avg_iterations_mean"
    ]

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(",".join(headers) + "\n")

        for config_name, stats in sorted(config_stats.items()):
            row = [
                config_name,
                str(stats.runs),
                f"{stats.pass_rate_mean:.4f}",
                f"{stats.pass_rate_std:.4f}",
                f"{stats.pass_rate_ci_low:.4f}",
                f"{stats.pass_rate_ci_high:.4f}",
                f"{stats.zero_shot_mean:.4f}",
                f"{stats.zero_shot_std:.4f}",
                f"{stats.self_heal_lift_mean:.4f}",
                f"{stats.self_heal_lift_std:.4f}",
                f"{stats.total_cost_mean:.6f}",
                f"{stats.total_cost_std:.6f}",
                f"{stats.cost_per_pass_mean:.8f}",
                f"{stats.cost_per_pass_std:.8f}",
                f"{stats.avg_iterations_mean:.4f}"
            ]
            f.write(",".join(row) + "\n")

    print(f"CSV saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Analyze multi-run benchmark results")
    parser.add_argument("--results-dir", default="results",
                        help="Directory containing result subdirectories")
    parser.add_argument("--output-dir", default=None,
                        help="Directory for output files (default: same as results-dir)")
    args = parser.parse_args()

    results_dir = args.results_dir
    output_dir = args.output_dir or results_dir

    if not os.path.exists(results_dir):
        print(f"Error: Results directory '{results_dir}' not found")
        return 1

    # Expected configurations
    expected_configs = ["flash-max", "max-max", "flash-coder", "coder-coder", "flash-flash"]

    # Collect all run metrics
    config_runs: Dict[str, List[RunMetrics]] = {}

    for config_name in expected_configs:
        config_dir = os.path.join(results_dir, config_name)

        if not os.path.exists(config_dir):
            print(f"Warning: Directory not found for {config_name}")
            continue

        # Find all run*.json files
        pattern = os.path.join(config_dir, "run*.json")
        run_files = sorted(glob.glob(pattern))

        if not run_files:
            print(f"Warning: No result files found for {config_name}")
            continue

        runs = []
        for filepath in run_files:
            # Extract run number from filename
            filename = os.path.basename(filepath)
            run_num = int(filename.replace("run", "").replace(".json", ""))

            metrics = extract_run_metrics(filepath, config_name, run_num)
            if metrics:
                runs.append(metrics)

        if runs:
            config_runs[config_name] = runs
            print(f"Loaded {len(runs)} runs for {config_name}")

    if not config_runs:
        print("Error: No valid results found")
        return 1

    # Aggregate statistics per configuration
    config_stats: Dict[str, ConfigStats] = {}
    for config_name, runs in config_runs.items():
        config_stats[config_name] = aggregate_config_stats(runs)

    # Perform statistical tests
    test_results = perform_statistical_tests(config_runs)

    # Print summary
    print_summary_table(config_stats)
    print_statistical_tests(test_results)

    # Save outputs
    os.makedirs(output_dir, exist_ok=True)
    save_summary_json(config_stats, config_runs, test_results,
                      os.path.join(output_dir, "summary.json"))
    save_comparison_csv(config_stats, os.path.join(output_dir, "comparison.csv"))

    print("\nAnalysis complete!")
    return 0


if __name__ == "__main__":
    exit(main())
