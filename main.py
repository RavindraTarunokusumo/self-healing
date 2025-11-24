"""
Self-Healing Code Agent using LangGraph

This module implements an autonomous code generation and fixing system using LangGraph.
The workflow follows a cyclic pattern: Generator -> Executor -> Critic, with each node
capable of triggering iterations until the code is correct.

Architecture:
- Generator: Uses LLM to generate Python code based on a specification
- Executor: Runs code in E2B sandbox and captures output/errors
- Critic: Analyzes results and decides whether to iterate or complete
"""

import os
from typing import Literal, TypedDict
from dotenv import load_dotenv
from colorama import Fore, Style, init as colorama_init

from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END
from e2b_code_interpreter import Sandbox

# Initialize colorama for colored output
colorama_init(autoreset=True)

# Load environment variables
load_dotenv()


class AgentState(TypedDict):
    """State object that flows through the LangGraph workflow."""
    specification: str  # Original code specification/requirements
    code: str  # Current version of generated code
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
        model_provider: Literal["openai", "anthropic"] = "openai",
        model_name: str = "gpt-4",
        max_iterations: int = 5
    ):
        """
        Initialize the Self-Healing Agent.
        
        Args:
            model_provider: Which LLM provider to use ("openai" or "anthropic")
            model_name: Specific model to use (e.g., "gpt-4", "claude-3-5-sonnet-20241022")
            max_iterations: Maximum number of self-healing iterations
        """
        self.max_iterations = max_iterations
        
        # Initialize the appropriate LLM
        if model_provider == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY not found in environment variables")
            self.llm = ChatOpenAI(model=model_name, temperature=0.7)
        elif model_provider == "anthropic":
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError("ANTHROPIC_API_KEY not found in environment variables")
            self.llm = ChatAnthropic(model=model_name, temperature=0.7)
        else:
            raise ValueError(f"Unsupported model provider: {model_provider}")
        
        # Build the LangGraph workflow
        self.workflow = self._build_workflow()
    
    def _build_workflow(self) -> StateGraph:
        """
        Build the cyclic LangGraph workflow: Generator -> Executor -> Critic.
        
        Returns:
            Compiled StateGraph workflow
        """
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("generator", self._generator_node)
        workflow.add_node("executor", self._executor_node)
        workflow.add_node("critic", self._critic_node)
        
        # Define edges
        workflow.set_entry_point("generator")
        workflow.add_edge("generator", "executor")
        workflow.add_edge("executor", "critic")
        
        # Conditional edge from critic: either go back to generator or end
        workflow.add_conditional_edges(
            "critic",
            self._should_continue,
            {
                "continue": "generator",
                "end": END
            }
        )
        
        return workflow.compile()
    
    def _generator_node(self, state: AgentState) -> AgentState:
        """
        Generator Node: Creates or fixes Python code based on specification and feedback.
        
        Args:
            state: Current agent state
            
        Returns:
            Updated state with new/fixed code
        """
        print(f"\n{Fore.CYAN}{'='*60}")
        print(f"{Fore.CYAN}GENERATOR - Iteration {state['iteration'] + 1}")
        print(f"{Fore.CYAN}{'='*60}{Style.RESET_ALL}")
        
        # Build prompt based on whether this is initial generation or fixing
        if state["iteration"] == 0:
            # Initial code generation
            system_prompt = (
                "You are an expert Python programmer. Generate clean, well-documented "
                "Python code based on the given specification. Include proper error handling "
                "and follow best practices."
            )
            user_prompt = f"Generate Python code for the following specification:\n\n{state['specification']}"
        else:
            # Code fixing based on feedback
            system_prompt = (
                "You are an expert Python programmer fixing code. Based on the execution "
                "results and critic feedback, modify the code to fix all issues. "
                "Only output the corrected Python code, no explanations."
            )
            user_prompt = (
                f"Original specification:\n{state['specification']}\n\n"
                f"Previous code:\n```python\n{state['code']}\n```\n\n"
                f"Execution output:\n{state['execution_output']}\n\n"
                f"Execution errors:\n{state['execution_error']}\n\n"
                f"Critic feedback:\n{state['critic_feedback']}\n\n"
                f"Fix the code based on the feedback above."
            )
        
        # Generate code using LLM
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]
        
        response = self.llm.invoke(messages)
        generated_code = response.content
        
        # Extract code from markdown code blocks if present
        if "```python" in generated_code:
            generated_code = generated_code.split("```python")[1].split("```")[0].strip()
        elif "```" in generated_code:
            generated_code = generated_code.split("```")[1].split("```")[0].strip()
        
        print(f"{Fore.GREEN}Generated Code:{Style.RESET_ALL}")
        print(generated_code)
        
        return {
            **state,
            "code": generated_code
        }
    
    def _executor_node(self, state: AgentState) -> AgentState:
        """
        Executor Node: Runs the code in E2B sandbox and captures output.
        
        Args:
            state: Current agent state
            
        Returns:
            Updated state with execution results
        """
        print(f"\n{Fore.YELLOW}{'='*60}")
        print(f"{Fore.YELLOW}EXECUTOR - Running code in sandbox")
        print(f"{Fore.YELLOW}{'='*60}{Style.RESET_ALL}")
        
        execution_output = ""
        execution_error = ""
        
        try:
            # Create E2B sandbox and execute code
            with Sandbox() as sandbox:
                execution = sandbox.run_code(state["code"])
                
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
        
        print(f"{Fore.YELLOW}Execution Output:{Style.RESET_ALL}")
        print(execution_output if execution_output else "(no output)")
        
        if execution_error:
            print(f"{Fore.RED}Execution Errors:{Style.RESET_ALL}")
            print(execution_error)
        
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
        print(f"\n{Fore.MAGENTA}{'='*60}")
        print(f"{Fore.MAGENTA}CRITIC - Analyzing results")
        print(f"{Fore.MAGENTA}{'='*60}{Style.RESET_ALL}")
        
        # Build critic prompt
        system_prompt = (
            "You are a critical code reviewer. Analyze whether the code correctly "
            "implements the specification and runs without errors. If there are issues, "
            "provide specific, actionable feedback for fixing them. If the code is correct, "
            "state 'APPROVED' clearly."
        )
        
        user_prompt = (
            f"Specification:\n{state['specification']}\n\n"
            f"Generated code:\n```python\n{state['code']}\n```\n\n"
            f"Execution output:\n{state['execution_output']}\n\n"
            f"Execution errors:\n{state['execution_error']}\n\n"
            f"Does this code correctly implement the specification? "
            f"If not, what needs to be fixed?"
        )
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]
        
        response = self.llm.invoke(messages)
        feedback = response.content
        
        print(f"{Fore.MAGENTA}Critic Feedback:{Style.RESET_ALL}")
        print(feedback)
        
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
            print(f"\n{Fore.GREEN}✓ Code approved! Workflow complete.{Style.RESET_ALL}")
            return "end"
        
        if state["iteration"] >= state["max_iterations"]:
            print(f"\n{Fore.RED}✗ Max iterations reached. Workflow ending.{Style.RESET_ALL}")
            return "end"
        
        print(f"\n{Fore.YELLOW}→ Continuing to next iteration...{Style.RESET_ALL}")
        return "continue"
    
    def run(self, specification: str) -> AgentState:
        """
        Run the self-healing agent on a code specification.
        
        Args:
            specification: Natural language description of desired code
            
        Returns:
            Final agent state with generated code and results
        """
        print(f"\n{Fore.BLUE}{'='*60}")
        print(f"{Fore.BLUE}SELF-HEALING CODE AGENT")
        print(f"{Fore.BLUE}{'='*60}{Style.RESET_ALL}")
        print(f"\nSpecification: {specification}\n")
        
        initial_state: AgentState = {
            "specification": specification,
            "code": "",
            "execution_output": "",
            "execution_error": "",
            "critic_feedback": "",
            "iteration": 0,
            "max_iterations": self.max_iterations,
            "is_complete": False
        }
        
        final_state = self.workflow.invoke(initial_state)
        
        print(f"\n{Fore.BLUE}{'='*60}")
        print(f"{Fore.BLUE}FINAL RESULTS")
        print(f"{Fore.BLUE}{'='*60}{Style.RESET_ALL}")
        print(f"\nTotal iterations: {final_state['iteration']}")
        print(f"Status: {'✓ Complete' if final_state['is_complete'] else '✗ Incomplete'}")
        print(f"\nFinal code:\n{Fore.GREEN}{final_state['code']}{Style.RESET_ALL}")
        
        return final_state


def main():
    """
    Example usage of the Self-Healing Code Agent.
    """
    # Example specification
    specification = (
        "Create a function that takes a list of numbers and returns "
        "the sum of all even numbers in the list. Include a main block "
        "that tests the function with [1, 2, 3, 4, 5, 6]."
    )
    
    # Initialize agent (uses OpenAI GPT-4 by default)
    # Change to model_provider="anthropic" and model_name="claude-3-5-sonnet-20241022" for Claude
    agent = SelfHealingAgent(
        model_provider="openai",
        model_name="gpt-4",
        max_iterations=5
    )
    
    # Run the self-healing workflow
    result = agent.run(specification)
    
    # Access final code
    print(f"\n{'='*60}")
    print("You can now use the generated code:")
    print(f"{'='*60}")
    print(result["code"])


if __name__ == "__main__":
    main()
