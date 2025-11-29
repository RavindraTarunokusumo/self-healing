"""
Example usage of the Self-Healing Code Agent.

This file demonstrates various ways to use the agent for different code generation tasks.
Note: You'll need to set up your API keys in a .env file before running these examples.
"""

import logging
from main import SelfHealingAgent


def example_1_simple_function():
    """Example 1: Generate a simple utility function."""
    specification = """
    Create a function called 'is_prime' that checks if a number is prime.
    The function should handle edge cases (negative numbers, 0, 1, 2).
    Include a test with numbers 2, 3, 4, 15, 17, and print the results.
    """
    
    agent = SelfHealingAgent(
        coder_model_provider="openai",
        coder_model_name="gpt-4",
        critic_model_provider="openai",
        critic_model_name="gpt-4",
        max_iterations=5
    )
    
    result = agent.run(specification)
    return result


def example_2_data_processing():
    """Example 2: Generate data processing code."""
    specification = """
    Create a function that takes a list of dictionaries with keys 'name' and 'age',
    and returns a new list sorted by age in descending order. Only include people
    who are 18 or older. Test with sample data: [
        {'name': 'Alice', 'age': 25},
        {'name': 'Bob', 'age': 17},
        {'name': 'Charlie', 'age': 30},
        {'name': 'Diana', 'age': 22}
    ]
    """
    
    agent = SelfHealingAgent(
        coder_model_provider="openai",
        coder_model_name="gpt-4",
        critic_model_provider="openai",
        critic_model_name="gpt-4",
        max_iterations=5
    )
    
    result = agent.run(specification)
    return result


def example_3_file_operations():
    """Example 3: Generate code with file operations."""
    specification = """
    Create a function that counts the number of words in a string.
    The function should:
    1. Convert the string to lowercase
    2. Remove punctuation
    3. Split on whitespace
    4. Return the count
    
    Test it with the string: "Hello, World! This is a test."
    Expected output: 6
    """
    
    agent = SelfHealingAgent(
        coder_model_provider="openai",
        coder_model_name="gpt-4",
        critic_model_provider="openai",
        critic_model_name="gpt-4",
        max_iterations=5
    )
    
    result = agent.run(specification)
    return result


def example_4_with_anthropic():
    """Example 4: Using Anthropic/Claude instead of OpenAI."""
    specification = """
    Create a recursive function to calculate the nth Fibonacci number.
    Include proper base cases (fib(0) = 0, fib(1) = 1).
    Test with n=10 and print the result.
    """
    
    agent = SelfHealingAgent(
        coder_model_provider="anthropic",
        coder_model_name="claude-3-5-sonnet-20241022",
        critic_model_provider="anthropic",
        critic_model_name="claude-3-5-sonnet-20241022",
        max_iterations=5
    )
    
    result = agent.run(specification)
    return result


def example_5_complex_task():
    """Example 5: More complex algorithm."""
    specification = """
    Create a function that implements binary search on a sorted list.
    The function should:
    1. Take a sorted list and a target value
    2. Return the index if found, -1 if not found
    3. Use iterative approach (not recursive)
    
    Test with: list = [1, 3, 5, 7, 9, 11, 13, 15], target = 7
    Expected output: 3
    """
    
    agent = SelfHealingAgent(
        coder_model_provider="openai",
        coder_model_name="gpt-4",
        critic_model_provider="openai",
        critic_model_name="gpt-4",
        max_iterations=7  # More iterations for complex tasks
    )
    
    result = agent.run(specification)
    return result


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    
    print("Self-Healing Code Agent - Examples")
    print("=" * 60)
    print("\nThese examples demonstrate various use cases.")
    print("Make sure to set up your .env file with API keys before running.")
    print("\nAvailable examples:")
    print("1. Simple utility function (is_prime)")
    print("2. Data processing (filter and sort)")
    print("3. String manipulation (word counter)")
    print("4. Recursive algorithm with Anthropic (Fibonacci)")
    print("5. Complex algorithm (binary search)")
    print("\nTo run an example, uncomment the line below:")
    print("# result = example_1_simple_function()")
    print("# print(result['code'])")
