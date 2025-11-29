# Self-Healing Code Agent

An autonomous, iterative Python code generation and fixing system using **LangGraph** for research on self-correction and Reflexion in code generation.

## 🎯 Overview

This project implements a self-healing code agent that automatically generates, executes, and fixes Python code through a cyclic workflow. The agent uses Large Language Models (OpenAI/Anthropic) for code generation and critique, and E2B sandboxes for safe code execution.

### Architecture

The agent follows a **cyclic three-node workflow** implemented with LangGraph:

```
┌─────────────┐
│  Generator  │  - Generates or fixes Python code using LLM
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  Executor   │  - Runs code in E2B sandbox
└──────┬──────┘  - Captures output and errors
       │
       ▼
┌─────────────┐
│   Critic    │  - Analyzes results and provides feedback
└──────┬──────┘  - Decides: iterate or complete
       │
       ▼
    [Loop back to Generator if needed]
```

**Key Features:**
- **Autonomous**: Iteratively fixes code without human intervention
- **Safe Execution**: Uses E2B sandboxes for isolated code execution
- **Reflexion Pattern**: Critic provides feedback that guides the next iteration
- **Configurable**: Supports OpenAI, Anthropic and Qwen models.
- **Research-Oriented**: Designed for experimentation with self-correction techniques

## 🚀 Quick Start

### Prerequisites

- Python 3.9 or higher
- API keys for LLM provider (OpenAI, Anthropic or Qwen)
- E2B API key for sandbox execution

### Installation

1. Clone the repository:
```bash
git clone https://github.com/RavindraTarunokusumo/self-healing.git
cd self-healing
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
# Create a .env file with your API keys
echo "OPENAI_API_KEY=your_openai_key_here" > .env
echo "ANTHROPIC_API_KEY=your_anthropic_key_here" >> .env
echo "QWEN_API_KEY=your_qwen_key_here" >> .env
echo "E2B_API_KEY=your_e2b_key_here" >> .env
```

### Usage

**Basic Usage:**

```python
from main import SelfHealingAgent

# Create an agent
agent = SelfHealingAgent(
    model_provider="openai",  # or "anthropic"
    model_name="gpt-4",       # or "claude-3-5-sonnet-20241022"
    max_iterations=5
)

# Define what you want to build
specification = """
Create a function that takes a list of numbers and returns 
the sum of all even numbers in the list.
"""

# Run the self-healing workflow
result = agent.run(specification)

# Access the final code
print(result["code"])
```

**Run the example:**

```bash
python main.py
```

## 🏗️ Architecture Details

### 1. Generator Node

**Purpose**: Generate or fix Python code based on specifications and feedback.

- **First iteration**: Generates code from scratch based on the specification
- **Subsequent iterations**: Fixes code based on execution results and critic feedback
- Uses LLM (GPT-4 or Claude) with temperature 0.7 for creativity

### 2. Executor Node

**Purpose**: Execute code safely and capture results.

- Runs code in isolated E2B sandbox environment
- Captures stdout, stderr, and error traces
- Prevents harmful code from affecting the host system
- Returns structured execution results

### 3. Critic Node

**Purpose**: Analyze execution results and guide the next iteration.

- Evaluates if code meets the specification
- Checks for errors and edge cases
- Provides specific, actionable feedback for fixes
- Approves code when it's correct (breaking the cycle)

### State Management

The `AgentState` TypedDict tracks information across nodes:

```python
{
    "specification": str,      # Original requirements
    "code": str,              # Current code version
    "execution_output": str,  # Sandbox output
    "execution_error": str,   # Error messages
    "critic_feedback": str,   # Critique and suggestions
    "iteration": int,         # Current iteration count
    "max_iterations": int,    # Iteration limit
    "is_complete": bool       # Success flag
}
```

### Workflow Control

The workflow uses conditional edges to control iteration:

- **Continue**: If code has errors and max iterations not reached
- **End**: If code is approved OR max iterations reached

## 🔧 Configuration

### Model Providers

**OpenAI:**
```python
agent = SelfHealingAgent(
    model_provider="openai",
    model_name="gpt-4",  # or "gpt-3.5-turbo", "gpt-4-turbo"
    max_iterations=5
)
```

**Anthropic:**
```python
agent = SelfHealingAgent(
    model_provider="anthropic",
    model_name="claude-3-5-sonnet-20241022",  # or other Claude models
    max_iterations=5
)
```

### Iteration Control

Adjust `max_iterations` to control how many times the agent will attempt to fix the code:

```python
agent = SelfHealingAgent(max_iterations=10)  # More attempts for complex tasks
```

## 📊 Research Applications

This framework is designed for research in:

1. **Self-Correction Mechanisms**: Study how LLMs improve code through iterative feedback
2. **Reflexion Patterns**: Analyze how critique influences subsequent generations
3. **Multi-Agent Systems**: Extend to multiple specialized agents (e.g., security checker, optimizer)
4. **Agent Reliability**: Measure success rates across different specifications
5. **Prompt Engineering**: Experiment with different prompting strategies

### Extending the Framework

**Add a new node:**

```python
def _optimizer_node(self, state: AgentState) -> AgentState:
    """Optimize the code for performance."""
    # Your optimization logic here
    return state

# Add to workflow
workflow.add_node("optimizer", self._optimizer_node)
workflow.add_edge("critic", "optimizer")
workflow.add_edge("optimizer", "generator")
```

**Custom evaluation metrics:**

```python
def evaluate_code_quality(code: str) -> dict:
    """Add custom metrics like complexity, coverage, etc."""
    return {
        "cyclomatic_complexity": calculate_complexity(code),
        "line_count": len(code.split("\n")),
        "has_docstrings": '"""' in code
    }
```

## 🛡️ Security

- All code execution happens in isolated E2B sandboxes
- No access to local filesystem or network from sandboxed code
- API keys are loaded from environment variables (never hardcoded)
- Sandboxes are automatically cleaned up after execution

## 📝 Dependencies

Core dependencies:
- `langgraph>=0.2.0` - Graph-based workflow orchestration
- `langchain>=0.3.0` - LLM integration framework
- `langchain-openai>=0.2.0` - OpenAI models
- `langchain-anthropic>=0.2.0` - Anthropic Claude models
- `e2b-code-interpreter>=1.0.0` - Sandbox code execution
- `python-dotenv>=1.0.0` - Environment variable management

See `requirements.txt` for complete list.

## 🎓 Learning Resources

### LangGraph
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [Building Autonomous Agents](https://blog.langchain.dev/langgraph-multi-agent-workflows/)

### Reflexion Pattern
- [Reflexion: Language Agents with Verbal Reinforcement Learning](https://arxiv.org/abs/2303.11366)
- [Self-Refine: Iterative Refinement with Self-Feedback](https://arxiv.org/abs/2303.17651)

### E2B Sandboxes
- [E2B Documentation](https://e2b.dev/docs)
- [Code Interpreter SDK](https://github.com/e2b-dev/code-interpreter)

## 🤝 Contributing

This is a research project. Contributions are welcome:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-node`)
3. Commit your changes (`git commit -am 'Add optimizer node'`)
4. Push to the branch (`git push origin feature/new-node`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🔮 Future Directions

- [ ] Add more specialized nodes (optimizer, security checker, tester)
- [ ] Implement multi-language support (JavaScript, Go, etc.)
- [ ] Add memory/learning capabilities across sessions
- [ ] Integrate with local code editors via LSP
- [ ] Build web interface for easier experimentation
- [ ] Collect and analyze success metrics for research
- [ ] Support for multi-file projects

## 📧 Contact

For questions or collaboration opportunities:
- GitHub: [@RavindraTarunokusumo](https://github.com/RavindraTarunokusumo)
- Repository: [self-healing](https://github.com/RavindraTarunokusumo/self-healing)

## 🙏 Acknowledgments

Built with:
- [LangGraph](https://github.com/langchain-ai/langgraph) by LangChain
- [E2B](https://e2b.dev/) for secure code execution
- Inspired by research in Reflexion and self-correction agents

---

**Happy Coding! 🚀**
