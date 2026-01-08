"""
Self-Healing Code Agent using LangGraph

This module implements an autonomous code generation and fixing system using LangGraph.
The workflow follows a cyclic pattern: Coder -> Executor -> Critic, with each node
capable of triggering iterations until the code is correct.

Architecture:
- Coder: Uses LLM to generate Python code based on a specification
- Executor: Runs code in E2B sandbox and captures output/errors
- Critic: Analyzes results and decides whether to iterate or complete
"""

import os
import logging
from datetime import datetime
from typing import Literal, TypedDict
import typing
import typing_extensions

# Monkeypatch Self for Python < 3.11
if not hasattr(typing, "Self"):
    typing.Self = typing_extensions.Self
from dotenv import load_dotenv
from colorama import Fore, Style, init as colorama_init

from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_qwq import ChatQwen
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END
from e2b_code_interpreter import Sandbox

# Initialize colorama for colored output
colorama_init(autoreset=True)

# Load environment variables
load_dotenv()

# Token-efficient prompt templates
CODER_SYSTEM_PROMPT = (
    "You are an expert Python programmer. Generate clean, working Python code. "
    "Output only the code, no explanations."
)

CODER_FIX_TEMPLATE = """SPEC: {specification}
CODE:
{code}
ERR: {execution_error}
CRITIC: {critic_feedback}
Output fixed code only."""

CRITIC_SYSTEM_PROMPT = """Review code against spec. Output ONLY in this format:

If correct: APPROVED

If issues exist:
STATUS: FAIL
TYPE: <syntax|runtime|logic|output|edge_case>
LOC: <line_number or "N/A">
FIX: <one sentence fix instruction>"""

CRITIC_USER_TEMPLATE = """SPEC: {specification}
CODE:
{code}
OUT: {execution_output}
ERR: {execution_error}"""


class AgentState(TypedDict):
    """State object that flows through the LangGraph workflow."""
    specification: str  # Original code specification/requirements
    code: str  # Current version of generated code
    test_code: str  # Unit tests to validate the code (from benchmarks)
    entry_point: str  # Function/class name to implement (from benchmarks)
    execution_output: str  # Output from code execution
    execution_error: str  # Error messages if any
    critic_feedback: str  # Feedback from the critic
    iteration: int  # Current iteration count
    max_iterations: int  # Maximum allowed iterations
    is_complete: bool  # Whether the code is correct and complete


class SelfHealingAgent:
    """
    Self-Healing Code Agent that generates, executes, and iteratively fixes code.
    """
    
    def __init__(
        self, 
        coder_model_provider: Literal["openai", "anthropic", "qwen"] = "openai",
        coder_model_name: str = "gpt-4",
        critic_model_provider: Literal["openai", "anthropic", "qwen"] = "openai",
        critic_model_name: str = "gpt-4",
        coder_max_tokens: int = 32000,
        critic_max_tokens: int = 32000,
        max_iterations: int = 5
    ):
        """
        Initialize the Self-Healing Agent.
        
        Args:
            coder_model_provider: Provider for the Coder LLM ("openai", "anthropic", "qwen")
            coder_model_name: Model name for the Coder LLM
            critic_model_provider: Provider for the Critic LLM ("openai", "anthropic", "qwen")
            critic_model_name: Model name for the Critic LLM
            max_iterations: Maximum number of self-healing iterations
        """
        self.max_iterations = max_iterations

        # Initialize token usage tracking
        self._reset_token_usage()

        # Initialize logger
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        
        # Create logs directory
        log_dir = os.path.join(os.getcwd(), "logs")
        os.makedirs(log_dir, exist_ok=True)
        
        # Create file handler with date-based filename
        current_date = datetime.now().strftime("%Y-%m-%d")
        current_time = datetime.now().strftime("%H-%M-%S")
        log_file = os.path.join(log_dir, f"{current_date}_{current_time}.log")
        file_handler = logging.FileHandler(log_file)
        
        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        
        # Add handlers
        self.logger.addHandler(file_handler)
        
        # Add stream handler
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        self.logger.addHandler(stream_handler)
        
        # Initialize the appropriate LLM for Coder and Critic
        self.coder_llm = self._create_llm(coder_model_provider, coder_model_name, coder_max_tokens)
        self.critic_llm = self._create_llm(critic_model_provider, critic_model_name, critic_max_tokens)
        
        # Build the LangGraph workflow
        self.workflow = self._build_workflow()

    def _create_llm(self, provider: str, model_name: str, max_tokens: int = 32000):
        """Helper method to create LLM instances."""
        if provider == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY not found in environment variables")
            return ChatOpenAI(model=model_name, temperature=0.7, max_tokens=max_tokens)
        elif provider == "anthropic":
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError("ANTHROPIC_API_KEY not found in environment variables")
            return ChatAnthropic(model=model_name, temperature=0.7, max_tokens=max_tokens)
        elif provider == "qwen":
            api_key = os.getenv("QWEN_API_KEY")
            if not api_key:
                raise ValueError("QWEN_API_KEY not found in environment variables")
            return ChatQwen(model=model_name, api_key=api_key, temperature=0.7, max_tokens=max_tokens)  
        else:
            raise ValueError(f"Unsupported model provider: {provider}")

    def _reset_token_usage(self):
        """Reset token usage counters for a new run."""
        self.coder_input_tokens = 0
        self.coder_output_tokens = 0
        self.coder_calls = 0
        self.critic_input_tokens = 0
        self.critic_output_tokens = 0
        self.critic_calls = 0

    def _extract_token_usage(self, response) -> tuple[int, int]:
        """
        Extract token usage from LLM response.

        Args:
            response: LangChain LLM response object

        Returns:
            Tuple of (input_tokens, output_tokens)
        """
        input_tokens = 0
        output_tokens = 0

        try:
            metadata = getattr(response, 'response_metadata', {}) or {}

            # OpenAI/Qwen format
            if 'token_usage' in metadata:
                usage = metadata['token_usage']
                input_tokens = usage.get('prompt_tokens', 0)
                output_tokens = usage.get('completion_tokens', 0)
            # Anthropic format
            elif 'usage' in metadata:
                usage = metadata['usage']
                input_tokens = usage.get('input_tokens', 0)
                output_tokens = usage.get('output_tokens', 0)
        except Exception:
            pass  # Silently handle if token extraction fails

        return input_tokens, output_tokens

    def get_token_usage(self) -> dict:
        """
        Get current token usage statistics.

        Returns:
            Dictionary with token usage metrics
        """
        coder_total = self.coder_input_tokens + self.coder_output_tokens
        critic_total = self.critic_input_tokens + self.critic_output_tokens

        return {
            "coder": {
                "calls": self.coder_calls,
                "input_tokens": self.coder_input_tokens,
                "output_tokens": self.coder_output_tokens,
                "total_tokens": coder_total,
                "avg_input_tokens": self.coder_input_tokens / self.coder_calls if self.coder_calls > 0 else 0,
                "avg_output_tokens": self.coder_output_tokens / self.coder_calls if self.coder_calls > 0 else 0,
            },
            "critic": {
                "calls": self.critic_calls,
                "input_tokens": self.critic_input_tokens,
                "output_tokens": self.critic_output_tokens,
                "total_tokens": critic_total,
                "avg_input_tokens": self.critic_input_tokens / self.critic_calls if self.critic_calls > 0 else 0,
                "avg_output_tokens": self.critic_output_tokens / self.critic_calls if self.critic_calls > 0 else 0,
            },
            "total": {
                "input_tokens": self.coder_input_tokens + self.critic_input_tokens,
                "output_tokens": self.coder_output_tokens + self.critic_output_tokens,
                "total_tokens": coder_total + critic_total,
            }
        }

    def _build_workflow(self) -> StateGraph:
        """
        Build the cyclic LangGraph workflow: Coder -> Executor -> Critic.
        
        Returns:
            Compiled StateGraph workflow
        """
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("coder", self._coder_node)
        workflow.add_node("executor", self._executor_node)
        workflow.add_node("critic", self._critic_node)
        
        # Define edges
        workflow.set_entry_point("coder")
        workflow.add_edge("coder", "executor")
        workflow.add_edge("executor", "critic")
        
        # Conditional edge from critic: either go back to coder or end
        workflow.add_conditional_edges(
            "critic",
            self._should_continue,
            {
                "continue": "coder",
                "end": END
            }
        )
        
        return workflow.compile()
    
    def _coder_node(self, state: AgentState) -> AgentState:
        """
        Coder Node: Creates or fixes Python code based on specification and feedback.
        
        Args:
            state: Current agent state
            
        Returns:
            Updated state with new/fixed code
        """
        self.logger.info(f"\n{Fore.CYAN}{'='*60}")
        self.logger.info(f"{Fore.CYAN}Coder - Iteration {state['iteration'] + 1}")
        self.logger.info(f"{Fore.CYAN}{'='*60}{Style.RESET_ALL}")
        
        # Build prompt based on whether this is initial generation or fixing
        if state["iteration"] == 0:
            # Initial code generation
            system_prompt = CODER_SYSTEM_PROMPT
            user_prompt = f"SPEC: {state['specification']}"
        else:
            # Code fixing based on feedback - use token-efficient template
            system_prompt = CODER_SYSTEM_PROMPT
            user_prompt = CODER_FIX_TEMPLATE.format(
                specification=state['specification'],
                code=state['code'],
                execution_error=state['execution_error'] or "None",
                critic_feedback=state['critic_feedback']
            )
        
        # Generate code using LLM
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]
        
        response = self.coder_llm.invoke(messages)
        generated_code = response.content

        # Track token usage
        input_tokens, output_tokens = self._extract_token_usage(response)
        self.coder_input_tokens += input_tokens
        self.coder_output_tokens += output_tokens
        self.coder_calls += 1

        # Extract code from markdown code blocks if present
        if "```python" in generated_code:
            generated_code = generated_code.split("```python")[1].split("```")[0].strip()
        elif "```" in generated_code:
            generated_code = generated_code.split("```")[1].split("```")[0].strip()
        
        self.logger.info(f"{Fore.GREEN}Generated Code:{Style.RESET_ALL}")
        self.logger.info("\n" + generated_code + "\n")
        
        return {
            **state,
            "code": generated_code
        }
    
    def _executor_node(self, state: AgentState) -> AgentState:
        """
        Executor Node: Runs the code in E2B sandbox and captures output.

        If test_code is provided (from benchmarks), it combines the generated
        code with the test code before execution.

        Args:
            state: Current agent state

        Returns:
            Updated state with execution results
        """
        self.logger.info(f"\n{Fore.YELLOW}{'='*60}")
        self.logger.info(f"{Fore.YELLOW}EXECUTOR - Running code in sandbox")
        self.logger.info(f"{Fore.YELLOW}{'='*60}{Style.RESET_ALL}")

        execution_output = ""
        execution_error = ""

        # Combine generated code with test code if available
        code_to_execute = state["code"]
        if state.get("test_code"):
            code_to_execute = f"{state['code']}\n\n{state['test_code']}"
            self.logger.info(f"{Fore.YELLOW}Appending test code for validation{Style.RESET_ALL}")

        try:
            # Create E2B sandbox and execute code
            with Sandbox.create() as sandbox:
                execution = sandbox.run_code(code_to_execute)
                
                # Capture output and errors
                if execution.logs:
                    execution_output = execution.logs.stdout + execution.logs.stderr
                
                if execution.error:
                    execution_error = f"Error: {execution.error.name}\n{execution.error.value}\n{execution.error.traceback}"
                
                if not execution_error and execution.results:
                    # Capture any results from the execution
                    execution_output += "\n" + str(execution.results)
        
        except Exception as e:
            execution_error = f"Sandbox execution failed: {str(e)}"
        
        self.logger.info(f"{Fore.YELLOW}Execution Output:{Style.RESET_ALL}")
        self.logger.info(execution_output if execution_output else "(no output)")
        
        if execution_error:
            self.logger.info(f"{Fore.RED}Execution Errors:{Style.RESET_ALL}")
            self.logger.info(execution_error)
        
        return {
            **state,
            "execution_output": execution_output,
            "execution_error": execution_error
        }
    
    def _critic_node(self, state: AgentState) -> AgentState:
        """
        Critic Node: Analyzes execution results and provides feedback.
        
        Args:
            state: Current agent state
            
        Returns:
            Updated state with critic feedback and completion status
        """
        self.logger.info(f"\n{Fore.MAGENTA}{'='*60}")
        self.logger.info(f"{Fore.MAGENTA}CRITIC - Analyzing results")
        self.logger.info(f"{Fore.MAGENTA}{'='*60}{Style.RESET_ALL}")
        
        # Build critic prompt - token-efficient format
        system_prompt = CRITIC_SYSTEM_PROMPT
        user_prompt = CRITIC_USER_TEMPLATE.format(
            specification=state['specification'],
            code=state['code'],
            execution_output=state['execution_output'] or "None",
            execution_error=state['execution_error'] or "None"
        )
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]
        
        response = self.critic_llm.invoke(messages)
        feedback = response.content

        # Track token usage
        input_tokens, output_tokens = self._extract_token_usage(response)
        self.critic_input_tokens += input_tokens
        self.critic_output_tokens += output_tokens
        self.critic_calls += 1

        self.logger.info(f"{Fore.MAGENTA}Critic Feedback:{Style.RESET_ALL}")
        self.logger.info("\n" + feedback + "\n")
        
        # Determine if code is complete (no errors and APPROVED by critic)
        is_complete = (
            not state["execution_error"] and 
            "APPROVED" in feedback.upper()
        )
        
        return {
            **state,
            "critic_feedback": feedback,
            "iteration": state["iteration"] + 1,
            "is_complete": is_complete
        }
    
    def _should_continue(self, state: AgentState) -> Literal["continue", "end"]:
        """
        Decide whether to continue iterating or end the workflow.
        
        Args:
            state: Current agent state
            
        Returns:
            "continue" to iterate again, "end" to stop
        """
        if state["is_complete"]:
            self.logger.info(f"\n{Fore.GREEN}✓ Code approved! Workflow complete.{Style.RESET_ALL}")
            return "end"
        
        if state["iteration"] >= state["max_iterations"]:
            self.logger.info(f"\n{Fore.RED}✗ Max iterations reached. Workflow ending.{Style.RESET_ALL}")
            return "end"
        
        self.logger.info(f"\n{Fore.YELLOW}→ Continuing to next iteration...{Style.RESET_ALL}")
        return "continue"
    
    def run(
        self,
        specification: str,
        test_code: str = "",
        entry_point: str = ""
    ) -> AgentState:
        """
        Run the self-healing agent on a code specification.

        Args:
            specification: Natural language description of desired code
            test_code: Optional unit tests to validate the solution (from benchmarks)
            entry_point: Optional function/class name to implement (from benchmarks)

        Returns:
            Final agent state with generated code and results
        """
        # Reset token usage for this run
        self._reset_token_usage()

        self.logger.info(f"\n{Fore.BLUE}{'='*60}")
        self.logger.info(f"{Fore.BLUE}SELF-HEALING CODE AGENT")
        self.logger.info(f"{Fore.BLUE}{'='*60}{Style.RESET_ALL}")
        self.logger.info(f"\nSpecification: {specification}\n")
        if entry_point:
            self.logger.info(f"Entry point: {entry_point}")
        if test_code:
            self.logger.info(f"Test code provided: {len(test_code)} chars\n")

        initial_state: AgentState = {
            "specification": specification,
            "code": "",
            "test_code": test_code,
            "entry_point": entry_point,
            "execution_output": "",
            "execution_error": "",
            "critic_feedback": "",
            "iteration": 0,
            "max_iterations": self.max_iterations,
            "is_complete": False
        }
        
        final_state = self.workflow.invoke(initial_state)
        
        self.logger.info(f"\n{Fore.BLUE}{'='*60}")
        self.logger.info(f"{Fore.BLUE}FINAL RESULTS")
        self.logger.info(f"{Fore.BLUE}{'='*60}{Style.RESET_ALL}")
        self.logger.info(f"\nTotal iterations: {final_state['iteration']}")
        self.logger.info(f"Status: {'Complete' if final_state['is_complete'] else 'Incomplete'}")

        # Log token usage
        token_usage = self.get_token_usage()
        self.logger.info(f"\n{Fore.CYAN}TOKEN USAGE:{Style.RESET_ALL}")
        self.logger.info(f"  Coder:  {token_usage['coder']['total_tokens']:,} tokens "
                        f"({token_usage['coder']['input_tokens']:,} in / {token_usage['coder']['output_tokens']:,} out) "
                        f"over {token_usage['coder']['calls']} call(s)")
        self.logger.info(f"  Critic: {token_usage['critic']['total_tokens']:,} tokens "
                        f"({token_usage['critic']['input_tokens']:,} in / {token_usage['critic']['output_tokens']:,} out) "
                        f"over {token_usage['critic']['calls']} call(s)")
        self.logger.info(f"  Total:  {token_usage['total']['total_tokens']:,} tokens")

        self.logger.info(f"\nFinal code:\n{Fore.GREEN}{final_state['code']}{Style.RESET_ALL}")

        # Add token usage to return value
        final_state['token_usage'] = token_usage

        return final_state