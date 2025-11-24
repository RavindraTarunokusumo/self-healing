"""
Basic validation tests for the Self-Healing Code Agent.

These tests validate the structure and basic functionality without requiring API keys.
"""

import unittest
from unittest.mock import Mock, patch
import sys
import os

# Add parent directory to path to import main
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import SelfHealingAgent, AgentState


class TestSelfHealingAgentStructure(unittest.TestCase):
    """Test the basic structure of the Self-Healing Agent."""
    
    def test_agent_state_structure(self):
        """Test that AgentState has the correct fields."""
        state: AgentState = {
            "specification": "test spec",
            "code": "print('hello')",
            "execution_output": "",
            "execution_error": "",
            "critic_feedback": "",
            "iteration": 0,
            "max_iterations": 5,
            "is_complete": False
        }
        
        # Verify all required keys are present
        self.assertIn("specification", state)
        self.assertIn("code", state)
        self.assertIn("execution_output", state)
        self.assertIn("execution_error", state)
        self.assertIn("critic_feedback", state)
        self.assertIn("iteration", state)
        self.assertIn("max_iterations", state)
        self.assertIn("is_complete", state)
    
    @patch.dict(os.environ, {"OPENAI_API_KEY": "test_key"})
    @patch('main.ChatOpenAI')
    def test_agent_initialization_openai(self, mock_chat):
        """Test agent initialization with OpenAI provider."""
        mock_chat.return_value = Mock()
        
        agent = SelfHealingAgent(
            model_provider="openai",
            model_name="gpt-4",
            max_iterations=3
        )
        
        self.assertEqual(agent.max_iterations, 3)
        self.assertIsNotNone(agent.workflow)
        mock_chat.assert_called_once()
    
    @patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test_key"})
    @patch('main.ChatAnthropic')
    def test_agent_initialization_anthropic(self, mock_chat):
        """Test agent initialization with Anthropic provider."""
        mock_chat.return_value = Mock()
        
        agent = SelfHealingAgent(
            model_provider="anthropic",
            model_name="claude-3-5-sonnet-20241022",
            max_iterations=3
        )
        
        self.assertEqual(agent.max_iterations, 3)
        self.assertIsNotNone(agent.workflow)
        mock_chat.assert_called_once()
    
    def test_agent_initialization_missing_key(self):
        """Test that agent raises error when API key is missing."""
        with self.assertRaises(ValueError):
            SelfHealingAgent(model_provider="openai")
    
    def test_agent_initialization_invalid_provider(self):
        """Test that agent raises error for invalid provider."""
        with self.assertRaises(ValueError):
            SelfHealingAgent(model_provider="invalid")
    
    @patch.dict(os.environ, {"OPENAI_API_KEY": "test_key"})
    @patch('main.ChatOpenAI')
    def test_should_continue_logic(self, mock_chat):
        """Test the workflow continuation logic."""
        mock_chat.return_value = Mock()
        agent = SelfHealingAgent(model_provider="openai", max_iterations=5)
        
        # Test case 1: Code is complete
        state_complete: AgentState = {
            "specification": "test",
            "code": "test",
            "execution_output": "",
            "execution_error": "",
            "critic_feedback": "",
            "iteration": 2,
            "max_iterations": 5,
            "is_complete": True
        }
        result = agent._should_continue(state_complete)
        self.assertEqual(result, "end")
        
        # Test case 2: Max iterations reached
        state_max: AgentState = {
            "specification": "test",
            "code": "test",
            "execution_output": "",
            "execution_error": "",
            "critic_feedback": "",
            "iteration": 5,
            "max_iterations": 5,
            "is_complete": False
        }
        result = agent._should_continue(state_max)
        self.assertEqual(result, "end")
        
        # Test case 3: Should continue
        state_continue: AgentState = {
            "specification": "test",
            "code": "test",
            "execution_output": "",
            "execution_error": "",
            "critic_feedback": "",
            "iteration": 2,
            "max_iterations": 5,
            "is_complete": False
        }
        result = agent._should_continue(state_continue)
        self.assertEqual(result, "continue")


class TestCodeExtraction(unittest.TestCase):
    """Test code extraction from LLM responses."""
    
    @patch.dict(os.environ, {"OPENAI_API_KEY": "test_key"})
    @patch('main.ChatOpenAI')
    def test_code_extraction_from_markdown(self, mock_chat):
        """Test that code is properly extracted from markdown code blocks."""
        mock_llm = Mock()
        mock_response = Mock()
        mock_response.content = """Here's the code:
```python
def hello():
    print("Hello, world!")
```
This should work."""
        mock_llm.invoke.return_value = mock_response
        mock_chat.return_value = mock_llm
        
        agent = SelfHealingAgent(model_provider="openai")
        
        state: AgentState = {
            "specification": "Print hello world",
            "code": "",
            "execution_output": "",
            "execution_error": "",
            "critic_feedback": "",
            "iteration": 0,
            "max_iterations": 5,
            "is_complete": False
        }
        
        result = agent._generator_node(state)
        expected_code = 'def hello():\n    print("Hello, world!")'
        self.assertEqual(result["code"], expected_code)


if __name__ == "__main__":
    unittest.main()
