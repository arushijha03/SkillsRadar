"""RAGAs evaluation for the SkillRadar RAG pipeline."""

import argparse
import json
import math
import os
import sys
import warnings
from collections.abc import Callable
from typing import Any

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from dotenv import load_dotenv

load_dotenv(os.path.join(ROOT, ".env"))

from datasets import Dataset
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from app.agents.rag_agent import RAGAgent
from app.rag.retriever import PineconeRetriever

DEFAULT_QUESTIONS = [
    "What skills are trending for machine learning engineers?",
    "What technologies are important for backend developers?",
    "What tools should a data scientist learn in 2025?",
]


def create_rag_agent(retriever: PineconeRetriever) -> RAGAgent:
    """Create a RAG agent for evaluation runs."""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    return RAGAgent(llm=llm, retriever=retriever)


def build_eval_dataset(
    questions: list[str],
    retriever: PineconeRetriever | None = None,
    agent_factory: Callable[[PineconeRetriever], RAGAgent] = create_rag_agent,
) -> Dataset:
    """Run retriever + RAG agent for each question and build RAGAs dataset."""
    retriever = retriever or PineconeRetriever()
    agent = agent_factory(retriever)

    rows = {
        "question": [],
        "answer": [],
        "contexts": [],
        "ground_truth": [],
    }

    for q in questions:
        chunks = retriever.retrieve(q)
        contexts = [c["text"] for c in chunks if c.get("text")]

        result = agent.run(role=q or "engineer")
        topics = result.get("topics") or []
        answer = ", ".join(
            f"{t.get('topic', '')} (score={t.get('community_score', 0)})"
            for t in topics[:5]
        ) or "No topics identified."

        rows["question"].append(q)
        rows["answer"].append(answer)
        rows["contexts"].append(contexts or ["No context retrieved."])
        rows["ground_truth"].append(
            "Relevant skills and technologies for the target role from community sources."
        )

    return Dataset.from_dict(rows)


def evaluate_dataset(dataset: Dataset):
    """Run RAGAs metrics against a prepared dataset."""
    from ragas import evaluate
    from ragas.embeddings import _LangchainEmbeddingsWrapper
    from ragas.llms import _LangchainLLMWrapper
    from ragas.metrics._answer_relevance import AnswerRelevancy
    from ragas.metrics._faithfulness import Faithfulness

    # RAGAs 0.4 accepts classic Metric objects in evaluate(), while the newer
    # collections metrics use a separate type hierarchy. Explicit wrappers keep
    # the evaluator compatible with the installed LangChain/OpenAI packages.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        evaluator_llm = _LangchainLLMWrapper(
            ChatOpenAI(model="gpt-4o-mini", temperature=0)
        )
        evaluator_embeddings = _LangchainEmbeddingsWrapper(
            OpenAIEmbeddings(model="text-embedding-3-small")
        )

    metrics = [
        Faithfulness(llm=evaluator_llm),
        AnswerRelevancy(llm=evaluator_llm, embeddings=evaluator_embeddings),
    ]

    return evaluate(
        dataset,
        metrics=metrics,
    )


def summarize_scores(result: Any) -> dict[str, float | None]:
    """Return mean metric scores from a RAGAs result object."""
    df = result.to_pandas()
    excluded = {
        "question",
        "contexts",
        "answer",
        "ground_truth",
        "user_input",
        "retrieved_contexts",
        "response",
        "reference",
    }
    scores = {}
    for col in df.columns:
        if col not in excluded:
            numeric = pd.to_numeric(df[col], errors="coerce")
            mean = numeric.mean()
            scores[col] = None if math.isnan(mean) else float(mean)
    return scores


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SkillRadar RAGAs evaluation.")
    parser.add_argument(
        "--question",
        action="append",
        dest="questions",
        help="Question to evaluate. Can be passed multiple times.",
    )
    parser.add_argument(
        "--output-json",
        help="Optional path to write mean metric scores as JSON.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    questions = args.questions or DEFAULT_QUESTIONS

    print("Building evaluation dataset...")
    dataset = build_eval_dataset(questions)

    print("Running RAGAs evaluation...")
    result = evaluate_dataset(dataset)

    print("\n=== RAGAs Results ===")
    print(result)

    scores = summarize_scores(result)
    print("\nScores:")
    for name, score in scores.items():
        if score is None:
            print(f"  {name}: n/a")
        else:
            print(f"  {name}: {score:.4f}")

    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(scores, f, indent=2)
        print(f"\nWrote scores to {args.output_json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
