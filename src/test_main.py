"""
Unit tests for the self-healing loop.

These tests avoid live API calls by mocking LLM clients.
"""

import os
import unittest
from unittest.mock import Mock, patch

from self_healing import SelfHealingAgent


class SelfHealingTestCase(unittest.TestCase):
    def _base_state(self):
        return {
            "specification": "Write a function",
            "code": "def foo():\n    return 1",
            "test_code": "",
            "entry_point": "foo",
            "execution_output": "",
            "execution_error": "",
            "is_infrastructure_error": False,
            "critic_feedback": "",
            "critic_structured": {},
            "quality_score": 0.0,
            "quality_score_history": [],
            "failed_criteria": [],
            "critic_retries": 0,
            "feedback_history": [],
            "iteration": 0,
            "max_iterations": 5,
            "is_complete": False,
            "termination_reason": "",
        }


class TestCriticJsonSchema(SelfHealingTestCase):
    @patch.dict(os.environ, {"OPENAI_API_KEY": "test_key"})
    @patch("self_healing.ChatOpenAI")
    def test_parse_valid_critic_json(self, mock_chat):
        mock_chat.return_value = Mock()
        agent = SelfHealingAgent(
            coder_model_provider="openai",
            critic_model_provider="openai",
        )

        parsed = agent._parse_critic_feedback_json(
            '{"status":"FAIL","quality_score":0.62,"failed_criteria":["edge_cases"],'
            '"issues":[{"type":"logic","loc":"12","fix":"Handle empty input.","severity":"medium"}]}'
        )

        self.assertEqual(parsed["status"], "FAIL")
        self.assertEqual(parsed["quality_score"], 0.62)
        self.assertEqual(parsed["failed_criteria"], ["edge_cases"])
        self.assertEqual(parsed["issues"][0]["severity"], "medium")

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test_key"})
    @patch("self_healing.ChatOpenAI")
    def test_parse_invalid_critic_json_missing_field(self, mock_chat):
        mock_chat.return_value = Mock()
        agent = SelfHealingAgent(
            coder_model_provider="openai",
            critic_model_provider="openai",
        )

        with self.assertRaises(ValueError):
            agent._parse_critic_feedback_json(
                '{"status":"APPROVED","quality_score":0.95,"failed_criteria":[]}'
            )


class TestLoopPolicies(SelfHealingTestCase):
    @patch.dict(os.environ, {"OPENAI_API_KEY": "test_key"})
    @patch("self_healing.ChatOpenAI")
    def test_schema_retry_exhaustion_sets_schema_error(self, mock_chat):
        coder_llm = Mock()
        critic_llm = Mock()
        mock_chat.side_effect = [coder_llm, critic_llm]

        bad = Mock()
        bad.content = "not-json"
        bad.response_metadata = {}
        critic_llm.invoke.side_effect = [bad, bad]

        agent = SelfHealingAgent(
            coder_model_provider="openai",
            critic_model_provider="openai",
            max_critic_schema_retries=1,
        )

        result = agent._critic_node(self._base_state())

        self.assertEqual(result["termination_reason"], "schema_error")
        self.assertEqual(result["critic_retries"], 1)
        self.assertFalse(result["is_complete"])

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test_key"})
    @patch("self_healing.ChatOpenAI")
    def test_approval_threshold_boundary(self, mock_chat):
        coder_llm = Mock()
        critic_llm = Mock()
        mock_chat.side_effect = [coder_llm, critic_llm]

        below = Mock()
        below.content = (
            '{"status":"APPROVED","quality_score":0.89,'
            '"failed_criteria":[],"issues":[]}'
        )
        below.response_metadata = {}

        exact = Mock()
        exact.content = (
            '{"status":"APPROVED","quality_score":0.90,'
            '"failed_criteria":[],"issues":[]}'
        )
        exact.response_metadata = {}

        critic_llm.invoke.side_effect = [below, exact]

        agent = SelfHealingAgent(
            coder_model_provider="openai",
            critic_model_provider="openai",
            approval_score_threshold=0.90,
        )

        state = self._base_state()
        result_below = agent._critic_node(state)
        self.assertFalse(result_below["is_complete"])

        state = self._base_state()
        result_exact = agent._critic_node(state)
        self.assertTrue(result_exact["is_complete"])
        self.assertEqual(result_exact["termination_reason"], "approved")

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test_key"})
    @patch("self_healing.ChatOpenAI")
    def test_no_improvement_detection_and_stop(self, mock_chat):
        mock_chat.return_value = Mock()
        agent = SelfHealingAgent(
            coder_model_provider="openai",
            critic_model_provider="openai",
            no_improvement_patience=2,
            min_score_delta=0.02,
        )

        state = self._base_state()
        state["quality_score_history"] = [0.50, 0.51, 0.515]

        self.assertTrue(agent._detect_no_improvement(state))
        self.assertEqual(agent._should_continue(state), "end")

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test_key"})
    @patch("self_healing.ChatOpenAI")
    def test_termination_precedence_checks(self, mock_chat):
        mock_chat.return_value = Mock()
        agent = SelfHealingAgent(
            coder_model_provider="openai",
            critic_model_provider="openai",
            no_improvement_patience=2,
            min_score_delta=0.02,
        )

        infra = self._base_state()
        infra["termination_reason"] = "infrastructure_error"
        infra["quality_score_history"] = [0.2, 0.2, 0.2]
        self.assertEqual(agent._should_continue(infra), "end")

        trunc = self._base_state()
        trunc["termination_reason"] = "truncated"
        trunc["quality_score_history"] = [0.2, 0.2, 0.2]
        self.assertEqual(agent._should_continue(trunc), "end")

        schema = self._base_state()
        schema["termination_reason"] = "schema_error"
        schema["quality_score_history"] = [0.2, 0.2, 0.2]
        self.assertEqual(agent._should_continue(schema), "end")

        maxed = self._base_state()
        maxed["iteration"] = maxed["max_iterations"]
        self.assertEqual(agent._should_continue(maxed), "end")


if __name__ == "__main__":
    unittest.main()
