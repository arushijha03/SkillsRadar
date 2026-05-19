"""Tests for the RAGAs evaluation helpers."""

from unittest.mock import MagicMock

import pandas as pd

from evaluation.ragas_eval import (
    DEFAULT_QUESTIONS,
    build_eval_dataset,
    summarize_scores,
)


def test_default_questions_are_present():
    assert len(DEFAULT_QUESTIONS) >= 3
    assert all(q.endswith("?") for q in DEFAULT_QUESTIONS)


def test_build_eval_dataset_uses_retriever_and_agent_factory():
    retriever = MagicMock()
    retriever.retrieve.return_value = [
        {
            "text": "LangGraph and vector databases are trending.",
            "title": "Trends",
            "url": "https://example.com",
            "source": "test",
            "score": 0.9,
        }
    ]

    agent = MagicMock()
    agent.run.return_value = {
        "topics": [
            {"topic": "LangGraph", "community_score": 0.82},
            {"topic": "Vector Databases", "community_score": 0.74},
        ]
    }

    def agent_factory(passed_retriever):
        assert passed_retriever is retriever
        return agent

    dataset = build_eval_dataset(
        ["What skills are trending for ML engineers?"],
        retriever=retriever,
        agent_factory=agent_factory,
    )

    row = dataset[0]
    assert row["question"] == "What skills are trending for ML engineers?"
    assert row["contexts"] == ["LangGraph and vector databases are trending."]
    assert "LangGraph" in row["answer"]
    assert "Vector Databases" in row["answer"]
    retriever.retrieve.assert_called_once_with(
        "What skills are trending for ML engineers?"
    )
    agent.run.assert_called_once_with(
        role="What skills are trending for ML engineers?"
    )


def test_build_eval_dataset_uses_no_context_fallback():
    retriever = MagicMock()
    retriever.retrieve.return_value = []

    agent = MagicMock()
    agent.run.return_value = {"topics": []}

    dataset = build_eval_dataset(
        [""],
        retriever=retriever,
        agent_factory=lambda _retriever: agent,
    )

    row = dataset[0]
    assert row["answer"] == "No topics identified."
    assert row["contexts"] == ["No context retrieved."]
    agent.run.assert_called_once_with(role="engineer")


def test_summarize_scores_excludes_dataset_columns():
    result = MagicMock()
    result.to_pandas.return_value = pd.DataFrame(
        {
            "question": ["q1", "q2"],
            "answer": ["a1", "a2"],
            "contexts": [["c1"], ["c2"]],
            "ground_truth": ["g1", "g2"],
            "user_input": ["q1", "q2"],
            "response": ["a1", "a2"],
            "retrieved_contexts": [["c1"], ["c2"]],
            "reference": ["g1", "g2"],
            "faithfulness": [0.8, 0.6],
            "answer_relevancy": [0.7, 0.9],
        }
    )

    scores = summarize_scores(result)

    assert scores == {
        "faithfulness": 0.7,
        "answer_relevancy": 0.8,
    }
