"""
Model pricing information and cost calculation utilities.

All prices are in USD per 1M tokens.
"""

# Pricing data: model_name -> (input_price_per_1m, output_price_per_1m)
MODEL_PRICING = {
    # OpenAI Models
    "gpt-4": (30.0, 60.0),
    "gpt-4-turbo": (10.0, 30.0),
    "gpt-4o": (2.5, 10.0),
    "gpt-4o-mini": (0.15, 0.6),
    "gpt-3.5-turbo": (0.5, 1.5),
    "gpt-3.5-turbo-16k": (3.0, 4.0),

    # Anthropic Models
    "claude-3-5-sonnet-20241022": (3.0, 15.0),
    "claude-3-5-sonnet-20240620": (3.0, 15.0),
    "claude-3-opus-20240229": (15.0, 75.0),
    "claude-3-sonnet-20240229": (3.0, 15.0),
    "claude-3-haiku-20240307": (0.25, 1.25),
    "claude-sonnet-4-5": (3.0, 15.0),  # Latest Claude Sonnet 4.5
    "claude-opus-4-5": (15.0, 75.0),   # Latest Claude Opus 4.5

    # Qwen Models
    "qwen3-max": (1.2, 6.0),
    "qwen-flash": (0.05, 0.4),
    "qwen3-coder-flash": (0.3, 1.5),
    "qwen": (1.2, 6.0),  # Default fallback for qwen
    "qwen2.5-72b-instruct": (0.5, 2.0),
    "qwen2.5-7b-instruct": (0.1, 0.3),
}


def get_model_pricing(model_name: str) -> tuple[float, float]:
    """
    Get pricing for a model.

    Args:
        model_name: Name of the model

    Returns:
        Tuple of (input_price_per_1m_tokens, output_price_per_1m_tokens)
        Returns (0.0, 0.0) if model not found
    """
    # Normalize model name (lowercase, strip whitespace)
    model_name = model_name.lower().strip()

    if model_name in MODEL_PRICING:
        return MODEL_PRICING[model_name]

    # Try partial matching for versioned models
    for key in MODEL_PRICING:
        if key in model_name or model_name in key:
            return MODEL_PRICING[key]

    # Return zero if not found (unknown model)
    return (0.0, 0.0)


def calculate_cost(input_tokens: int, output_tokens: int, model_name: str) -> float:
    """
    Calculate the cost of an LLM call.

    Args:
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens
        model_name: Name of the model

    Returns:
        Total cost in USD
    """
    input_price, output_price = get_model_pricing(model_name)

    # Convert tokens to millions and calculate cost
    input_cost = (input_tokens / 1_000_000) * input_price
    output_cost = (output_tokens / 1_000_000) * output_price

    return input_cost + output_cost


def calculate_total_cost(token_usage: dict, coder_model: str, critic_model: str) -> dict:
    """
    Calculate total costs for a benchmark run.

    Args:
        token_usage: Token usage dictionary from agent
        coder_model: Name of the coder model
        critic_model: Name of the critic model

    Returns:
        Dictionary with cost breakdown
    """
    coder_tokens = token_usage.get("coder", {})
    critic_tokens = token_usage.get("critic", {})

    coder_cost = calculate_cost(
        coder_tokens.get("input_tokens", 0),
        coder_tokens.get("output_tokens", 0),
        coder_model
    )

    critic_cost = calculate_cost(
        critic_tokens.get("input_tokens", 0),
        critic_tokens.get("output_tokens", 0),
        critic_model
    )

    return {
        "coder_cost": coder_cost,
        "critic_cost": critic_cost,
        "total_cost": coder_cost + critic_cost,
        "coder_pricing": get_model_pricing(coder_model),
        "critic_pricing": get_model_pricing(critic_model)
    }


def format_cost(cost: float) -> str:
    """
    Format a cost value for display.

    Args:
        cost: Cost in USD

    Returns:
        Formatted string (e.g., "$0.0123" or "$1.23")
    """
    if cost == 0.0:
        return "$0.00 (pricing unavailable)"
    elif cost < 0.01:
        return f"${cost:.4f}"
    elif cost < 1.0:
        return f"${cost:.3f}"
    else:
        return f"${cost:.2f}"
