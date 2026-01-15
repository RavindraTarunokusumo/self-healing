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
    is_infrastructure_error: bool  # True if error is infrastructure (E2B, network) vs code error
    critic_feedback: str  # Feedback from the critic
    feedback_history: list  # Track all critic feedback for stuck detection
    iteration: int  # Current iteration count
    max_iterations: int  # Maximum allowed iterations
    is_complete: bool  # Whether the code is correct and complete
    termination_reason: str  # Why workflow ended: approved, max_iterations, stuck, truncated, infrastructure_error


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
        coder_temperature: float = 0.7,
        critic_temperature: float = 0.7,
        max_iterations: int = 5,
        enable_stuck_detection: bool = False,
        stuck_detection_threshold: int = 2,
        early_termination_on_stuck: bool = False,
        early_termination_on_truncation: bool = False
    ):
        """
        Initialize the Self-Healing Agent.

        Args:
            coder_model_provider: Provider for the Coder LLM ("openai", "anthropic", "qwen")
            coder_model_name: Model name for the Coder LLM
            critic_model_provider: Provider for the Critic LLM ("openai", "anthropic", "qwen")
            critic_model_name: Model name for the Critic LLM
            max_iterations: Maximum number of self-healing iterations
            coder_temperature: Temperature setting for the Coder LLM
            critic_temperature: Temperature setting for the Critic LLM
            enable_stuck_detection: Enable detection of stuck states (identical feedback)
            stuck_detection_threshold: Number of identical consecutive feedbacks to consider stuck
            early_termination_on_stuck: Terminate immediately when stuck state detected
            early_termination_on_truncation: Terminate immediately when critic response truncated
        """
        self.max_iterations = max_iterations
        self.enable_stuck_detection = enable_stuck_detection
        self.stuck_detection_threshold = stuck_detection_threshold
        self.early_termination_on_stuck = early_termination_on_stuck
        self.early_termination_on_truncation = early_termination_on_truncation

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

        # Create structured output file for detailed code tracking
        self.output_file = os.path.join(log_dir, f"{current_date}_{current_time}_output.txt")
        self.output_handle = None
        
        # Initialize the appropriate LLM for Coder and Critic
        self.coder_llm = self._create_llm(coder_model_provider, coder_model_name, coder_max_tokens, coder_temperature)
        self.critic_llm = self._create_llm(critic_model_provider, critic_model_name, critic_max_tokens, critic_temperature)
        
        # Build the LangGraph workflow
        self.workflow = self._build_workflow()

    def _create_llm(self, provider: str, model_name: str, max_tokens: int = 32000, temperature: float = 0.7):
        """Helper method to create LLM instances."""
        if provider == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY not found in environment variables")
            return ChatOpenAI(model=model_name, temperature=temperature, max_tokens=max_tokens)
        elif provider == "anthropic":
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError("ANTHROPIC_API_KEY not found in environment variables")
            return ChatAnthropic(model=model_name, temperature=temperature, max_tokens=max_tokens)
        elif provider == "qwen":
            api_key = os.getenv("QWEN_API_KEY")
            if not api_key:
                raise ValueError("QWEN_API_KEY not found in environment variables")
            return ChatQwen(model=model_name, api_key=api_key, temperature=temperature, max_tokens=max_tokens)  
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

    def _is_response_truncated(self, response) -> bool:
        """
        Detect if LLM response was truncated due to token limits.

        Args:
            response: LangChain LLM response object

        Returns:
            True if response was truncated, False otherwise
        """
        try:
            metadata = getattr(response, 'response_metadata', {}) or {}

            # OpenAI/Qwen format
            if 'finish_reason' in metadata:
                finish_reason = metadata.get('finish_reason', '')
                return finish_reason == 'length'

            # Anthropic format
            if 'stop_reason' in metadata:
                stop_reason = metadata.get('stop_reason', '')
                return stop_reason == 'max_tokens'

        except Exception:
            pass  # Silently handle errors

        return False

    def _detect_stuck_state(self, state: AgentState) -> bool:
        """
        Detect if agent is stuck in a loop with identical critic feedback.

        Args:
            state: Current agent state

        Returns:
            True if stuck state detected, False otherwise
        """
        if not self.enable_stuck_detection:
            return False

        feedback_history = state.get("feedback_history", [])

        # Need at least threshold number of feedbacks to detect stuck
        if len(feedback_history) < self.stuck_detection_threshold:
            return False

        # Check if last N feedbacks are identical
        recent_feedbacks = feedback_history[-self.stuck_detection_threshold:]

        # Normalize feedback (strip whitespace, lowercase)
        normalized = [f.strip().lower() for f in recent_feedbacks]

        # All must be identical and non-empty
        if len(set(normalized)) == 1 and normalized[0]:
            return True

        return False

    def _is_infrastructure_error(self, error_message: str) -> bool:
        """
        Detect if an error is an infrastructure error (E2B, network, etc.)
        vs a code execution error.

        Infrastructure errors should terminate the session immediately
        without wasting LLM tokens on the critic.

        Args:
            error_message: The error message from execution

        Returns:
            True if this is an infrastructure error, False if it's a code error
        """
        if not error_message:
            return False

        error_lower = error_message.lower()

        # E2B API errors (authentication, permission, quota)
        patterns = [
            "403:",  # Forbidden - API key issues
            "401:",  # Unauthorized
            "402:",  # Payment required (quota/billing)
            "429:",  # Rate limited
            "500:",  # Internal server error
            "502:",  # Bad gateway
            "503:",  # Service unavailable
            "504:",  # Gateway timeout
            "sandbox execution failed",
        ]

        for pattern in patterns:
            if pattern in error_lower:
                return True

        return False

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

    def _write_to_output(self, content: str):
        """Write content to the structured output file."""
        if self.output_handle:
            self.output_handle.write(content)
            self.output_handle.flush()

    def write_benchmark_metadata(self, benchmark_info: dict):
        """
        Write overall benchmark run metadata at the start of output file.
        Should be called once before running benchmark problems.
        """
        with open(self.output_file, 'a', encoding='utf-8') as f:
            f.write(f"\n{'#'*80}\n")
            f.write("RUN METADATA\n")
            f.write(f"{'#'*80}\n\n")
            f.write(f"Timestamp: {datetime.now().isoformat()}\n")
            f.write(f"Benchmark: {benchmark_info.get('benchmark', 'N/A')}\n")
            f.write(f"Coder Model: {benchmark_info.get('coder_provider', 'N/A')} - {benchmark_info.get('coder_model', 'N/A')}\n")
            f.write(f"Critic Model: {benchmark_info.get('critic_provider', 'N/A')} - {benchmark_info.get('critic_model', 'N/A')}\n")
            f.write(f"Coder Max Tokens: {benchmark_info.get('coder_max_tokens', 'N/A')}\n")
            f.write(f"Critic Max Tokens: {benchmark_info.get('critic_max_tokens', 'N/A')}\n")
            f.write(f"Coder Temperature: {benchmark_info.get('coder_temperature', 'N/A')}\n")
            f.write(f"Critic Temperature: {benchmark_info.get('critic_temperature', 'N/A')}\n")
            f.write(f"Max Iterations: {benchmark_info.get('max_iterations', 'N/A')}\n")
            f.write(f"\n{'#'*80}\n\n")
            f.flush()

    def _write_run_metadata(self, task_id: str, specification: str, entry_point: str = ""):
        """Write metadata for a new problem run."""
        self._write_to_output(f"\n{'='*80}\n")
        self._write_to_output(f"Problem: {task_id}\n")
        self._write_to_output(f"Specification: {specification}\n")
        if entry_point:
            self._write_to_output(f"Entry Point: {entry_point}\n")
        self._write_to_output(f"{'='*80}\n\n")

    def _write_iteration_header(self, iteration: int):
        """Write iteration header."""
        self._write_to_output(f"\n{'-'*80}\n")
        self._write_to_output(f"Iteration {iteration}\n")
        self._write_to_output(f"{'-'*80}\n\n")

    def _write_code(self, code: str):
        """Write generated code to output file."""
        self._write_to_output("CODE:\n")
        self._write_to_output(code)
        self._write_to_output("\n\n")

    def _write_execution_result(self, output: str, error: str):
        """Write execution results to output file."""
        self._write_to_output("EXECUTION STATUS:\n")
        if error:
            self._write_to_output(f"Error: {error}\n")
        else:
            self._write_to_output("Success\n")
        if output:
            self._write_to_output(f"Output: {output}\n")
        self._write_to_output("\n")

    def _write_critic_feedback(self, feedback: str):
        """Write critic feedback to output file."""
        self._write_to_output("CRITIC FEEDBACK:\n")
        self._write_to_output(feedback)
        self._write_to_output("\n\n")

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

        self.logger.info(f"{Fore.GREEN}Code generated{Style.RESET_ALL}")

        # Write iteration header and code to output file
        self._write_iteration_header(state["iteration"] + 1)
        self._write_code(generated_code)

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

        # Check if this is an infrastructure error (not a code error)
        is_infra_error = self._is_infrastructure_error(execution_error)

        if is_infra_error:
            self.logger.error(f"{Fore.RED}INFRASTRUCTURE ERROR DETECTED - Session will terminate{Style.RESET_ALL}")
            self.logger.error(f"{Fore.RED}This is not a code error. Check your E2B_API_KEY and network connection.{Style.RESET_ALL}")
        elif execution_error:
            self.logger.info(f"{Fore.RED}Execution failed{Style.RESET_ALL}")
        else:
            self.logger.info(f"{Fore.YELLOW}Execution completed{Style.RESET_ALL}")

        # Write execution results to output file
        self._write_execution_result(execution_output, execution_error)

        return {
            **state,
            "execution_output": execution_output,
            "execution_error": execution_error,
            "is_infrastructure_error": is_infra_error
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

        # Skip LLM call entirely if infrastructure error detected
        # This saves tokens - no point asking critic to review when code couldn't run
        if state.get("is_infrastructure_error", False):
            self.logger.info(f"{Fore.MAGENTA}Skipping critic LLM call - infrastructure error detected{Style.RESET_ALL}")
            feedback = "[INFRASTRUCTURE ERROR] Code execution failed due to infrastructure issue (E2B/network). Session terminated to save API costs."
            self._write_critic_feedback(feedback)
            return {
                **state,
                "critic_feedback": feedback,
                "feedback_history": state.get("feedback_history", []),
                "iteration": state["iteration"] + 1,
                "is_complete": False,
                "termination_reason": "infrastructure_error"
            }

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

        # Detect truncation/malformed response
        is_truncated = self._is_response_truncated(response)
        if is_truncated:
            self.logger.warning(f"{Fore.RED}Warning: Critic response was truncated due to token limit{Style.RESET_ALL}")
            feedback = feedback + "\n[TRUNCATED]"
            if self.early_termination_on_truncation:
                self.logger.error(f"{Fore.RED}Terminating due to truncated critic response{Style.RESET_ALL}")
                self._write_critic_feedback(feedback)
                # Update feedback history if enabled
                updated_history = state.get("feedback_history", [])
                if self.enable_stuck_detection:
                    updated_history = updated_history + [feedback]
                return {
                    **state,
                    "critic_feedback": feedback,
                    "feedback_history": updated_history,
                    "iteration": state["iteration"] + 1,
                    "is_complete": False,
                    "termination_reason": "truncated"
                }

        # Determine if code is complete (no errors and APPROVED by critic)
        is_complete = (
            not state["execution_error"] and
            "APPROVED" in feedback.upper()
        )

        if is_complete:
            self.logger.info(f"{Fore.MAGENTA}Critic: APPROVED{Style.RESET_ALL}")
        else:
            self.logger.info(f"{Fore.MAGENTA}Critic: Requesting fixes{Style.RESET_ALL}")

        # Write critic feedback to output file
        self._write_critic_feedback(feedback)

        # Update feedback history if stuck detection enabled
        updated_history = state.get("feedback_history", [])
        if self.enable_stuck_detection:
            updated_history = updated_history + [feedback]

        return {
            **state,
            "critic_feedback": feedback,
            "feedback_history": updated_history,
            "iteration": state["iteration"] + 1,
            "is_complete": is_complete,
            "termination_reason": "approved" if is_complete else state.get("termination_reason", "")
        }
    
    def _should_continue(self, state: AgentState) -> Literal["continue", "end"]:
        """
        Decide whether to continue iterating or end the workflow.

        Checks (in order):
        1. Code approved → end
        2. Infrastructure error → end (always, no config needed)
        3. Truncation with early termination → end
        4. Stuck state detected → warn/end based on config
        5. Max iterations reached → end
        6. Otherwise → continue

        Args:
            state: Current agent state

        Returns:
            "continue" to iterate again, "end" to stop
        """
        # Check 1: Code approved
        if state["is_complete"]:
            self.logger.info(f"\n{Fore.GREEN}✓ Code approved! Workflow complete.{Style.RESET_ALL}")
            return "end"

        # Check 2: Infrastructure error - always terminate immediately (no config needed)
        if state.get("termination_reason") == "infrastructure_error":
            self.logger.error(f"\n{Fore.RED}✗ Infrastructure error. Workflow terminated to save API costs.{Style.RESET_ALL}")
            return "end"

        # Check 3: Early termination on truncation (termination_reason already set in _critic_node)
        if state.get("termination_reason") == "truncated":
            return "end"

        # Check 4: Stuck state detection
        if self._detect_stuck_state(state):
            self.logger.warning(
                f"\n{Fore.YELLOW}Warning: Stuck state detected - "
                f"Critic provided identical feedback {self.stuck_detection_threshold} times in a row{Style.RESET_ALL}"
            )

            if self.early_termination_on_stuck:
                self.logger.error(f"{Fore.RED}Terminating due to stuck state{Style.RESET_ALL}")
                # Note: termination_reason will be set in run() since we can't modify state here
                return "end"
            else:
                self.logger.info(f"{Fore.YELLOW}→ Continuing despite stuck state (early_termination_on_stuck=False){Style.RESET_ALL}")

        # Check 5: Max iterations
        if state["iteration"] >= state["max_iterations"]:
            self.logger.info(f"\n{Fore.RED}✗ Max iterations reached. Workflow ending.{Style.RESET_ALL}")
            return "end"

        # Check 6: Continue
        self.logger.info(f"\n{Fore.YELLOW}→ Continuing to next iteration...{Style.RESET_ALL}")
        return "continue"
    
    def run(
        self,
        specification: str,
        test_code: str = "",
        entry_point: str = "",
        task_id: str = "Manual"
    ) -> AgentState:
        """
        Run the self-healing agent on a code specification.

        Args:
            specification: Natural language description of desired code
            test_code: Optional unit tests to validate the solution (from benchmarks)
            entry_point: Optional function/class name to implement (from benchmarks)
            task_id: Optional task identifier for tracking in output file

        Returns:
            Final agent state with generated code and results
        """
        # Reset token usage for this run
        self._reset_token_usage()

        # Open output file for this run
        self.output_handle = open(self.output_file, 'a', encoding='utf-8')

        self.logger.info(f"\n{Fore.BLUE}{'='*60}")
        self.logger.info(f"{Fore.BLUE}SELF-HEALING CODE AGENT")
        self.logger.info(f"{Fore.BLUE}{'='*60}{Style.RESET_ALL}")
        self.logger.info(f"\nTask ID: {task_id}")
        self.logger.info(f"Specification: {specification[:100]}...")
        if entry_point:
            self.logger.info(f"Entry point: {entry_point}")
        if test_code:
            self.logger.info(f"Test code provided: {len(test_code)} chars\n")

        # Write run metadata to output file
        self._write_run_metadata(task_id, specification, entry_point)

        initial_state: AgentState = {
            "specification": specification,
            "code": "",
            "test_code": test_code,
            "entry_point": entry_point,
            "execution_output": "",
            "execution_error": "",
            "is_infrastructure_error": False,
            "critic_feedback": "",
            "feedback_history": [],
            "iteration": 0,
            "max_iterations": self.max_iterations,
            "is_complete": False,
            "termination_reason": ""
        }

        final_state = self.workflow.invoke(initial_state)

        # Set termination_reason if not already set
        if not final_state.get("termination_reason"):
            if final_state["is_complete"]:
                final_state["termination_reason"] = "approved"
            elif self._detect_stuck_state(final_state):
                final_state["termination_reason"] = "stuck"
            else:
                final_state["termination_reason"] = "max_iterations"

        self.logger.info(f"\n{Fore.BLUE}{'='*60}")
        self.logger.info(f"{Fore.BLUE}FINAL RESULTS")
        self.logger.info(f"{Fore.BLUE}{'='*60}{Style.RESET_ALL}")
        self.logger.info(f"\nTotal iterations: {final_state['iteration']}")
        self.logger.info(f"Status: {'Complete' if final_state['is_complete'] else 'Incomplete'}")
        self.logger.info(f"Termination reason: {final_state.get('termination_reason', 'N/A')}")

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

        # Write final summary to output file
        self._write_to_output(f"\n{'='*80}\n")
        self._write_to_output(f"FINAL RESULT: {'COMPLETE' if final_state['is_complete'] else 'INCOMPLETE'}\n")
        self._write_to_output(f"Termination Reason: {final_state.get('termination_reason', 'N/A')}\n")
        self._write_to_output(f"Total Iterations: {final_state['iteration']}\n")
        self._write_to_output(f"Total Tokens: {token_usage['total']['total_tokens']:,}\n")
        self._write_to_output(f"{'='*80}\n\n")

        # Close output file
        if self.output_handle:
            self.output_handle.close()
            self.output_handle = None

        # Add token usage to return value
        final_state['token_usage'] = token_usage

        return final_state