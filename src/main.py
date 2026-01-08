import argparse
from datetime import datetime
import json
import logging
import sys
import traceback
from self_healing import SelfHealingAgent
from benchmark_loaders import get_loader, list_benchmarks

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
        default="openai", 
        choices=["openai", "anthropic", "qwen"], 
        help="Provider for the Coder LLM"
    )
    parser.add_argument(
        "--coder-model", 
        type=str, 
        default="gpt-4", 
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
        default="qwen", 
        help="Model name for the Critic LLM"
    )
    parser.add_argument(
        "--coder-max-tokens", 
        type=int, 
        default=32000, 
        help="Maximum number of tokens for the Coder LLM"
    )
    parser.add_argument(
        "--critic-max-tokens", 
        type=int, 
        default=32000, 
        help="Maximum number of tokens for the Critic LLM"
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=5,
        help="Maximum number of self-healing iterations"
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
        default=1,
        help="Number of benchmark problems to run (default: 1)"
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
            max_iterations=args.max_iterations
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
            print_benchmark_summary(results[1]["problems_results"])
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
        
        # Add manual run logic here if needed
        result = agent.run(problem)
        print_benchmark_summary([result])


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

    # Limit number of problems
    problems = problems[:args.num_problems]
    logger.info(f"Running {len(problems)} problem(s)")
    print(f"Running {len(problems)} problem(s)...\n")

    # Metadata of the benchmark run
    results = [{
        "benchmark_run_metadata": {
            "benchmark": args.benchmark,
            "from_hub": args.from_hub,
            "benchmark_path": args.benchmark_path,
            "task_id": args.task_id,
            "num_problems": args.num_problems,
            "timestamp": datetime.now().isoformat()
        },
        "models_metadata": {
            "coder_provider": args.coder_provider,
            "coder_model": args.coder_model,
            "critic_provider": args.critic_provider,
            "critic_model": args.critic_model,
            "coder_max_tokens": args.coder_max_tokens,
            "critic_max_tokens": args.critic_max_tokens,
            "max_iterations": args.max_iterations
        }
    }]
    
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
                entry_point=problem.entry_point
            )
            logger.info(f"agent.run() completed. is_complete={result['is_complete']}, iterations={result['iteration']}")
        except Exception as e:
            logger.error(f"agent.run() failed: {e}")
            logger.error(traceback.format_exc())
            raise

        results[1]["problems_results"].append({
            "task_id": problem.task_id,
            "entry_point": problem.entry_point,
            "prompt": problem.prompt,
            "generated_code": result["code"],
            "is_complete": result["is_complete"],
            "iterations": result["iteration"],
            "execution_output": result["execution_output"],
            "execution_error": result["execution_error"],
            "token_usage": result.get("token_usage", {})
        })
        logger.info(f"Problem {problem.task_id} result appended")

    logger.info(f"run_benchmark() completed. Total results: {len(results)}")
    return results


def save_results(results: list, output_path: str):
    """Save benchmark results to a JSON file."""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")


def print_benchmark_summary(results: list):
    """Print a summary of benchmark results."""
    if not results:
        return

    total = len(results)
    passed = sum(1 for r in results if r["is_complete"])
    avg_iterations = sum(r["iterations"] for r in results) / total

    # Aggregate token usage
    total_coder_input = 0
    total_coder_output = 0
    total_coder_calls = 0
    total_critic_input = 0
    total_critic_output = 0
    total_critic_calls = 0

    for r in results:
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

    print(f"\n{'='*60}")
    print("BENCHMARK SUMMARY")
    print(f"{'='*60}")
    print(f"Total problems: {total}")
    print(f"Passed: {passed}/{total} ({100*passed/total:.1f}%)")
    print(f"Average iterations: {avg_iterations:.2f}")
    print(f"{'='*60}")

    # Print token usage summary
    print("\nTOKEN USAGE SUMMARY")
    print(f"{'-'*60}")
    total_coder = total_coder_input + total_coder_output
    total_critic = total_critic_input + total_critic_output
    grand_total = total_coder + total_critic

    print(f"Coder:")
    print(f"  Total tokens:   {total_coder:,}")
    print(f"  Input tokens:   {total_coder_input:,}")
    print(f"  Output tokens:  {total_coder_output:,}")
    print(f"  Total calls:    {total_coder_calls}")
    if total_coder_calls > 0:
        print(f"  Avg tokens/call: {total_coder / total_coder_calls:,.1f}")

    print(f"\nCritic:")
    print(f"  Total tokens:   {total_critic:,}")
    print(f"  Input tokens:   {total_critic_input:,}")
    print(f"  Output tokens:  {total_critic_output:,}")
    print(f"  Total calls:    {total_critic_calls}")
    if total_critic_calls > 0:
        print(f"  Avg tokens/call: {total_critic / total_critic_calls:,.1f}")

    print(f"\nGrand Total: {grand_total:,} tokens")
    print(f"{'='*60}")

    # Print per-problem results
    print("\nPer-problem results:")
    for r in results:
        status = "PASS" if r["is_complete"] else "FAIL"
        token_usage = r.get("token_usage", {})
        total_tokens = token_usage.get("total", {}).get("total_tokens", 0)
        print(f"  {r['task_id']}: {status} (iterations: {r['iterations']}, tokens: {total_tokens:,})")
        
        
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error("An unhandled exception occurred:")
        logger.error(traceback.format_exc())
        sys.exit(1)
