import argparse
from datetime import datetime
import json
import logging
import sys
import traceback
from self_healing import SelfHealingAgent
from benchmark_loaders import get_loader, list_benchmarks
from pricing import calculate_total_cost, format_cost

# Initialize logger (configure with args later)
logger = logging.getLogger(__name__)


def main():
    """
    Main usage of the Self-Healing Code Agent.
    """
    logger.info("=" * 60)
    logger.info("SELF-HEALING CODE AGENT - Starting")
    logger.info("=" * 60)
    logger.debug(f"Python version: {sys.version}")
    logger.debug(f"Command line args: {sys.argv}")
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run the Self-Healing Code Agent")
    
    parser.add_argument(
        "--coder-provider", 
        type=str, 
        default="qwen", 
        choices=["openai", "anthropic", "qwen"], 
        help="Provider for the Coder LLM"
    )
    parser.add_argument(
        "--coder-model", 
        type=str, 
        default="qwen-flash", 
        help="Model name for the Coder LLM"
    )
    parser.add_argument(
        "--critic-provider", 
        type=str, 
        default="qwen", 
        choices=["openai", "anthropic", "qwen"], 
        help="Provider for the Critic LLM"
    )
    parser.add_argument(
        "--critic-model", 
        type=str, 
        default="qwen3-max", 
        help="Model name for the Critic LLM"
    )
    parser.add_argument(
        "--coder-max-tokens", 
        type=int, 
        default=1024, 
        help="Maximum number of tokens for the Coder LLM"
    )
    parser.add_argument(
        "--critic-max-tokens", 
        type=int, 
        default=4096, 
        help="Maximum number of tokens for the Critic LLM"
    )
    parser.add_argument(
        "--coder-temperature",
        type=float,
        default=0.7,
        help="Temperature setting for the Coder LLM"
    )
    parser.add_argument(
        "--critic-temperature",
        type=float,
        default=0.7,
        help="Temperature setting for the Critic LLM"
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=5,
        help="Maximum number of self-healing iterations"
    )

    # Stuck detection and early termination arguments
    parser.add_argument(
        "--enable-stuck-detection",
        action="store_true",
        help="Enable detection of stuck states (identical critic feedback)"
    )
    parser.add_argument(
        "--stuck-threshold",
        type=int,
        default=2,
        help="Number of identical consecutive feedbacks to consider stuck (default: 2)"
    )
    parser.add_argument(
        "--early-termination-on-stuck",
        action="store_true",
        help="Terminate immediately when stuck state detected (default: continue with warning)"
    )
    parser.add_argument(
        "--early-termination-on-truncation",
        action="store_true",
        help="Terminate immediately when critic response is truncated (default: continue)"
    )

    # Benchmark arguments
    parser.add_argument(
        "--benchmark",
        type=str,
        choices=list_benchmarks(),
        help=f"Benchmark to load problems from. Available: {', '.join(list_benchmarks())}"
    )
    parser.add_argument(
        "--benchmark-path",
        type=str,
        help="Path to benchmark data file (JSONL format)"
    )
    parser.add_argument(
        "--from-hub",
        action="store_true",
        help="Load benchmark directly from HuggingFace Hub"
    )
    parser.add_argument(
        "--task-id",
        type=str,
        help="Specific task ID to run (e.g., 'HumanEval/0')"
    )
    parser.add_argument(
        "--num-problems",
        type=int,
        default=None,
        help="Number of benchmark problems to run (default: all problems in benchmark)"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output file path for benchmark results (JSON format)"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    )

    args = parser.parse_args()
    
    logging.basicConfig(
    level=getattr(logging, args.log_level),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    logger.info("Parsed arguments:")
    logger.info(f"  benchmark: {args.benchmark}")
    logger.info(f"  from_hub: {args.from_hub}")
    logger.info(f"  task_id: {args.task_id}")
    logger.info(f"  benchmark_path: {args.benchmark_path}")
    logger.info(f"  num_problems: {args.num_problems}")
    logger.info(f"  coder_provider: {args.coder_provider}")
    logger.info(f"  coder_model: {args.coder_model}")
    logger.info(f"  critic_provider: {args.critic_provider}")
    logger.info(f"  critic_model: {args.critic_model}")
    logger.info(f"  max_iterations: {args.max_iterations}")

    # Initialize agent with command line arguments
    logger.info("Initializing SelfHealingAgent...")
    try:
        agent = SelfHealingAgent(
            coder_model_provider=args.coder_provider,
            coder_model_name=args.coder_model,
            critic_model_provider=args.critic_provider,
            critic_model_name=args.critic_model,
            coder_max_tokens=args.coder_max_tokens,
            critic_max_tokens=args.critic_max_tokens,
            coder_temperature=args.coder_temperature,
            critic_temperature=args.critic_temperature,
            max_iterations=args.max_iterations,
            enable_stuck_detection=args.enable_stuck_detection,
            stuck_detection_threshold=args.stuck_threshold,
            early_termination_on_stuck=args.early_termination_on_stuck,
            early_termination_on_truncation=args.early_termination_on_truncation
        )
        logger.info("SelfHealingAgent initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize SelfHealingAgent: {e}")
        logger.error(traceback.format_exc())
        return

    # Load benchmark problems if specified
    if args.benchmark:
        logger.info(f"Running in benchmark mode: {args.benchmark}")
        try:
            results = run_benchmark(agent, args)
            logger.info(f"Benchmark completed. Results count: {len(results)}")
            if args.output:
                save_results(results, args.output)
            print_benchmark_summary(results)
        except Exception as e:
            logger.error(f"Benchmark execution failed: {e}")
            logger.error(traceback.format_exc())
    else:
        # Run with a manual specification
        logger.info("Running in manual specification mode")
        problem = None
        if problem is None:
            logger.warning("No problem specified")
            print("No problem specified. Use --benchmark to load from a benchmark,")
            print("or modify the 'problem' variable in main.py.")
            return

        # Run the agent
        result = agent.run(problem)

        # Build results structure for manual runs
        problems_results = [{
            "task_id": "Manual",
            "entry_point": "",
            "is_complete": result["is_complete"],
            "termination_reason": result.get("termination_reason", ""),
            "iterations": result["iteration"],
            "token_usage": result.get("token_usage", {}),
            "cost": {}
        }]
        summary = calculate_benchmark_summary(problems_results)
        manual_results = [
            {"benchmark_run_metadata": {}, "models_metadata": {}},
            {"problems_results": problems_results},
            {"benchmark_summary": summary}
        ]
        print_benchmark_summary(manual_results)


def run_benchmark(agent: SelfHealingAgent, args) -> list:
    """
    Run the agent on benchmark problems.

    Args:
        agent: The SelfHealingAgent instance
        args: Parsed command line arguments

    Returns:
        List of results for each problem
    """
    logger.info(f"run_benchmark() called with benchmark={args.benchmark}")

    logger.debug("Getting benchmark loader...")
    loader = get_loader(args.benchmark)
    logger.info(f"Got loader for: {loader.name}")
    print(f"\nLoading {loader.name} benchmark...")

    # Load problems from file or HuggingFace Hub
    if args.from_hub:
        logger.info("Loading from HuggingFace Hub...")
        try:
            problems = loader.load_from_hub()
            logger.info(f"Successfully loaded {len(problems)} problems from Hub")
        except Exception as e:
            logger.error(f"Failed to load from Hub: {e}")
            logger.error(traceback.format_exc())
            raise
    elif args.benchmark_path:
        logger.info(f"Loading from file: {args.benchmark_path}")
        problems = loader.load(args.benchmark_path)
    else:
        logger.error("No data source specified (need --from-hub or --benchmark-path)")
        print("Error: Specify --benchmark-path or use --from-hub")
        return []

    print(f"Loaded {len(problems)} problems")
    logger.info(f"Total problems loaded: {len(problems)}")

    # Filter to specific task if requested
    if args.task_id:
        logger.info(f"Filtering for task_id: {args.task_id}")
        problems = [p for p in problems if p.task_id == args.task_id]
        logger.info(f"After filtering: {len(problems)} problems")
        if not problems:
            logger.error(f"Task ID '{args.task_id}' not found in dataset")
            print(f"Error: Task ID '{args.task_id}' not found")
            return []

    # Limit number of problems if specified
    if args.num_problems is not None:
        problems = problems[:args.num_problems]
        logger.info(f"Running {len(problems)} problem(s) (limited by --num-problems)")
    else:
        logger.info(f"Running all {len(problems)} problem(s) from benchmark")
    print(f"Running {len(problems)} problem(s)...\n")

    # Metadata of the benchmark run
    results = [{
        "benchmark_run_metadata": {
            "benchmark": args.benchmark,
            "from_hub": args.from_hub,
            "benchmark_path": args.benchmark_path,
            "task_id": args.task_id,
            "num_problems": len(problems),  # Actual number of problems being run
            "num_problems_requested": args.num_problems,  # What user requested (None = all)
            "timestamp": datetime.now().isoformat()
        },
        "models_metadata": {
            "coder_provider": args.coder_provider,
            "coder_model": args.coder_model,
            "critic_provider": args.critic_provider,
            "critic_model": args.critic_model,
            "coder_max_tokens": args.coder_max_tokens,
            "critic_max_tokens": args.critic_max_tokens,
            "coder_temperature": args.coder_temperature,
            "critic_temperature": args.critic_temperature,
            "max_iterations": args.max_iterations
        }
    }]

    # Write benchmark metadata to output file
    agent.write_benchmark_metadata({
        "benchmark": args.benchmark,
        "coder_provider": args.coder_provider,
        "coder_model": args.coder_model,
        "critic_provider": args.critic_provider,
        "critic_model": args.critic_model,
        "coder_temperature": args.coder_temperature,
        "critic_temperature": args.critic_temperature,
        "max_iterations": args.max_iterations
    })

    # Store problems
    results.append({"problems_results": []})

    for i, problem in enumerate(problems):
        logger.info(f"Processing problem {i+1}/{len(problems)}: {problem.task_id}")
        print(f"\n{'='*60}")
        print(f"Problem {i+1}/{len(problems)}: {problem.task_id}")
        print(f"Entry point: {problem.entry_point}")
        print(f"{'='*60}")

        logger.debug(f"Prompt length: {len(problem.prompt)} chars")
        logger.debug(f"Test code length: {len(problem.test_code)} chars")
        logger.debug(f"Entry point: {problem.entry_point}")

        logger.info("Calling agent.run()...")
        try:
            result = agent.run(
                specification=problem.prompt,
                test_code=problem.test_code,
                entry_point=problem.entry_point,
                task_id=problem.task_id
            )
            logger.info(f"agent.run() completed. is_complete={result['is_complete']}, iterations={result['iteration']}")
        except Exception as e:
            logger.error(f"agent.run() failed: {e}")
            logger.error(traceback.format_exc())
            raise

        # Calculate cost for this problem
        token_usage = result.get("token_usage", {})
        cost_info = calculate_total_cost(
            token_usage,
            args.coder_model,
            args.critic_model
        )

        results[1]["problems_results"].append({
            "task_id": problem.task_id,
            "entry_point": problem.entry_point,
            "is_complete": result["is_complete"],
            "termination_reason": result.get("termination_reason", ""),
            "iterations": result["iteration"],
            "token_usage": token_usage,
            "cost": cost_info
        })
        logger.info(f"Problem {problem.task_id} result appended")

    # Calculate and add summary to results
    summary = calculate_benchmark_summary(results[1]["problems_results"])
    results.append({"benchmark_summary": summary})
    logger.info(f"Benchmark summary calculated and added to results")

    logger.info(f"run_benchmark() completed. Total results: {len(results)}")
    return results


def calculate_benchmark_summary(problems_results: list) -> dict:
    """
    Calculate summary statistics from benchmark results.

    Args:
        problems_results: List of per-problem results

    Returns:
        Dictionary containing summary statistics
    """
    if not problems_results:
        return {}

    total = len(problems_results)
    passed = sum(1 for r in problems_results if r["is_complete"])
    total_iterations = sum(r["iterations"] for r in problems_results)
    avg_iterations = total_iterations / total if total > 0 else 0

    # Count termination reasons
    termination_counts = {}
    for r in problems_results:
        reason = r.get("termination_reason", "unknown")
        termination_counts[reason] = termination_counts.get(reason, 0) + 1

    # Aggregate token usage
    total_coder_input = 0
    total_coder_output = 0
    total_coder_calls = 0
    total_critic_input = 0
    total_critic_output = 0
    total_critic_calls = 0

    for r in problems_results:
        token_usage = r.get("token_usage", {})
        if token_usage:
            coder = token_usage.get("coder", {})
            critic = token_usage.get("critic", {})
            total_coder_input += coder.get("input_tokens", 0)
            total_coder_output += coder.get("output_tokens", 0)
            total_coder_calls += coder.get("calls", 0)
            total_critic_input += critic.get("input_tokens", 0)
            total_critic_output += critic.get("output_tokens", 0)
            total_critic_calls += critic.get("calls", 0)

    total_coder_tokens = total_coder_input + total_coder_output
    total_critic_tokens = total_critic_input + total_critic_output
    grand_total_tokens = total_coder_tokens + total_critic_tokens

    # Aggregate costs
    total_coder_cost = 0.0
    total_critic_cost = 0.0

    for r in problems_results:
        cost_info = r.get("cost", {})
        if cost_info:
            total_coder_cost += cost_info.get("coder_cost", 0.0)
            total_critic_cost += cost_info.get("critic_cost", 0.0)

    total_cost = total_coder_cost + total_critic_cost
    cost_per_pass = total_cost / passed if passed > 0 else 0.0

    return {
        "problems": {
            "total": total,
            "passed": passed,
            "failed": total - passed,
            "pass_rate": (passed / total * 100) if total > 0 else 0.0
        },
        "iterations": {
            "total": total_iterations,
            "average": avg_iterations
        },
        "termination_reasons": termination_counts,
        "token_usage": {
            "coder": {
                "input_tokens": total_coder_input,
                "output_tokens": total_coder_output,
                "total_tokens": total_coder_tokens,
                "calls": total_coder_calls,
                "avg_tokens_per_call": total_coder_tokens / total_coder_calls if total_coder_calls > 0 else 0
            },
            "critic": {
                "input_tokens": total_critic_input,
                "output_tokens": total_critic_output,
                "total_tokens": total_critic_tokens,
                "calls": total_critic_calls,
                "avg_tokens_per_call": total_critic_tokens / total_critic_calls if total_critic_calls > 0 else 0
            },
            "total": {
                "total_tokens": grand_total_tokens
            }
        },
        "cost": {
            "coder_cost": total_coder_cost,
            "critic_cost": total_critic_cost,
            "total_cost": total_cost,
            "cost_per_pass": cost_per_pass
        }
    }


def save_results(results: list, output_path: str):
    """Save benchmark results to a JSON file."""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")


def print_benchmark_summary(results: list):
    """
    Print a summary of benchmark results.

    Args:
        results: Full benchmark results list containing:
            - results[0]: benchmark_run_metadata and models_metadata
            - results[1]: problems_results list
            - results[2]: benchmark_summary dict
    """
    if not results or len(results) < 3:
        logger.warning("Results list incomplete, cannot print summary")
        return

    # Extract data from results
    problems_results = results[1].get("problems_results", [])
    summary = results[2].get("benchmark_summary", {})

    if not summary:
        logger.warning("No benchmark summary found in results")
        return

    # Extract summary data
    problems = summary.get("problems", {})
    iterations = summary.get("iterations", {})
    termination_reasons = summary.get("termination_reasons", {})
    token_usage = summary.get("token_usage", {})
    cost = summary.get("cost", {})

    total = problems.get("total", 0)
    passed = problems.get("passed", 0)
    pass_rate = problems.get("pass_rate", 0.0)
    avg_iterations = iterations.get("average", 0.0)

    print(f"\n{'='*60}")
    print("BENCHMARK SUMMARY")
    print(f"{'='*60}")
    print(f"Total problems: {total}")
    print(f"Passed: {passed}/{total} ({pass_rate:.1f}%)")
    print(f"Average iterations: {avg_iterations:.2f}")

    # Print termination reasons breakdown
    if termination_reasons:
        print(f"\nTermination reasons:")
        for reason, count in sorted(termination_reasons.items()):
            pct = (count / total * 100) if total > 0 else 0
            print(f"  {reason}: {count} ({pct:.1f}%)")

    print(f"{'='*60}")

    # Print token usage summary
    coder_usage = token_usage.get("coder", {})
    critic_usage = token_usage.get("critic", {})
    total_usage = token_usage.get("total", {})

    print("\nTOKEN USAGE SUMMARY")
    print(f"{'-'*60}")

    print(f"Coder:")
    print(f"  Total tokens:   {coder_usage.get('total_tokens', 0):,}")
    print(f"  Input tokens:   {coder_usage.get('input_tokens', 0):,}")
    print(f"  Output tokens:  {coder_usage.get('output_tokens', 0):,}")
    print(f"  Total calls:    {coder_usage.get('calls', 0)}")
    if coder_usage.get('calls', 0) > 0:
        print(f"  Avg tokens/call: {coder_usage.get('avg_tokens_per_call', 0):,.1f}")

    print(f"\nCritic:")
    print(f"  Total tokens:   {critic_usage.get('total_tokens', 0):,}")
    print(f"  Input tokens:   {critic_usage.get('input_tokens', 0):,}")
    print(f"  Output tokens:  {critic_usage.get('output_tokens', 0):,}")
    print(f"  Total calls:    {critic_usage.get('calls', 0)}")
    if critic_usage.get('calls', 0) > 0:
        print(f"  Avg tokens/call: {critic_usage.get('avg_tokens_per_call', 0):,.1f}")

    print(f"\nGrand Total: {total_usage.get('total_tokens', 0):,} tokens")
    print(f"{'='*60}")

    # Print cost summary
    print("\nCOST SUMMARY")
    print(f"{'-'*60}")
    print(f"Coder cost:      {format_cost(cost.get('coder_cost', 0.0))}")
    print(f"Critic cost:     {format_cost(cost.get('critic_cost', 0.0))}")
    print(f"Total cost:      {format_cost(cost.get('total_cost', 0.0))}")

    if passed > 0:
        print(f"Cost per pass:   {format_cost(cost.get('cost_per_pass', 0.0))}")

    print(f"{'='*60}")

    # Print per-problem results
    print("\nPer-problem results:")
    for r in problems_results:
        status = "PASS" if r["is_complete"] else "FAIL"
        term_reason = r.get("termination_reason", "N/A")
        r_token_usage = r.get("token_usage", {})
        total_tokens = r_token_usage.get("total", {}).get("total_tokens", 0)
        cost_info = r.get("cost", {})
        problem_cost = cost_info.get("total_cost", 0.0) if cost_info else 0.0
        print(f"  {r['task_id']}: {status} [{term_reason}] (iterations: {r['iterations']}, tokens: {total_tokens:,}, cost: {format_cost(problem_cost)})")

        
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error("An unhandled exception occurred:")
        logger.error(traceback.format_exc())
        sys.exit(1)
