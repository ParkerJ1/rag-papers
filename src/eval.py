import os
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import json
import logging
import requests
from datetime import datetime
from pathlib import Path

from query import query

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

EVAL_SET_PATH = Path("data/eval/eval_set.json")
EVAL_REPORTS_PATH = Path("docs/eval_reports")
JUDGE_URL = "http://localhost:8081/v1/chat/completions"

FAITHFULNESS_PROMPT = """
    You are an evaluation judge. Your task is to assess whether a generated answer is faithful to the retrieved context. 
    A faithful answer only makes claims that are supported by the context. It does not introduce information from outside the context.

    Retrieved context:
    {context}

    Generated answer:
    {answer}

    Score the faithfulness of the generated answer on a scale of 1 to 5:
    1 - The answer contradicts or ignores the context entirely.
    2 - the answer has significant unsupported claims.
    3 - The answer is partially supported by the context.
    4 - The answer is mostly supported by the context with minor unsupported claims.
    5 - The answer is fully supported by the context.

    Respond with a JSON object in this exact format:
    {{"score" : <integer 1-5>, "reason": "<one sentence explanation>"}}
    
    """

ANSWER_RELEVANCE_PROMPT = """
    You are evaluation judge. Your task is to assess whether a generated answer adequately addresses the question, given a reference answer as a guide to what a good answer looks like.

    Question:
    {question}

    Reference answer:
    {reference_answer}

    Generated answer:
    {answer}

    Score the relevance of the generated answer on a scale of 1 to 5:
    1 - The answer does not address the question at all
    2 - The answer addresses the question only superficially
    3 - The answer partially addresses the question
    4 - The answer addresses the question well with minor gaps
    5 - The answer fully addresses the question as well as the reference answer does

    Respond with a JSON object in this exact format:
    {{"score": <integer 1-5>, "reason": "<one sentence explanation>"}}


"""

CONTEXT_PRECISION_PROMPT = """You are an evaluation judge. Your task is to assess whether a retrieved text chunk is relevant to a given question.

Question:
{question}

Retrieved chunk:
{chunk}

Is this chunk relevant to the question? A chunk is relevant if it contains information that would help answer the question.

Respond with a JSON object in this exact format:
{{"relevant": true or false, "reason": "<one sentence explanation>"}}"""


def load_eval_set(path: Path) -> list[dict]:
    """Load evaluation set (question/answers) from JSON file"""
    with open(path, "r") as f:
        return json.load(f)
    
def judge(prompt: str) -> dict:

    try:
        response = requests.post(
            JUDGE_URL,
            json={
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "n_predict": 256,
                "temperature": 0.00
            
            }
        )

        response.raise_for_status()
        raw = response.json()["choices"][0]["message"]["content"]
        clean = raw.strip()
        clean = clean.removeprefix("```json").strip()
        clean = clean.removeprefix("```").strip()
        clean = clean.removesuffix("```").strip()


        parsed = json.loads(clean)
        if "score" in parsed:
            parsed["score"] = int(parsed["score"])
        return parsed
    
    except requests.exceptions.ConnectionError:
        logger.error(f"Could not connect to llama server at {JUDGE_URL}. Is it running?")
        return {"score": 0, "reason": f"Could not connect to the LLM server at {JUDGE_URL}."}
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse judge response as JSON: {e}. Raw response: {raw}")
        return {"score": 0, "reason": f"Failed to parse judge response as JSON: {e}."}
    
def evaluate_single(item: dict) -> dict:
    """Evaluate a single item in the evaluation set"""
    question = item["question"]
    reference_answer = item["reference_answer"]
    source_filter = item.get("source")

    logger.info(f"Evaluating: {item['id']} - {question[:60]}...")
    result = query(question, source_filter=source_filter)

    generated_answer = result["answer"]

    source = result["source"]
    chunks = result.get("chunks", [])

    context_precision = score_context_precision(question, chunks)
    logger.info(f"Context precision: {context_precision['precision']} ({context_precision['relevant_chunks']}/{context_precision['total_chunks']} chunks relevant)")

    context = "\n\n".join([c["text"] for c in chunks])

    if not question:
        logger.error(f"No question given.")
    if not context:
        logger.error(f"No context given.")
    if not generated_answer:
        logger.error(f"No generated answer given.")
    if not reference_answer:
        logger.error(f"No reference answer given.")



    faithfulness_score = judge(
        FAITHFULNESS_PROMPT.format(
            context=context,
            answer=generated_answer
        )
    )

    relevance_score = judge(
        ANSWER_RELEVANCE_PROMPT.format(
            question=question,
            reference_answer=reference_answer,
            answer=generated_answer
        )
    )

    return {
        "id": item["id"],
        "question": question,
        "generated_answer": generated_answer,
        "reference_answer": reference_answer,
        "confidence": result["confidence"],
        "context_precision": context_precision,
        "faithfulness": faithfulness_score,
        "answer_relevance": relevance_score,
    }    


def run_eval():
    logger.info(f"Loading eval set from {EVAL_SET_PATH}")
    eval_set = load_eval_set(EVAL_SET_PATH)

    logger.info(f"Loaded {len(eval_set)} items from {EVAL_SET_PATH}")

    results = []
    for item in eval_set:
        result = evaluate_single(item)
        results.append(result)

    valid_faithfulness = [r["faithfulness"]["score"] for r in results if r["faithfulness"]["score"] > 0]
    valid_relevance = [r["answer_relevance"]["score"] for r in results if r["answer_relevance"]["score"] > 0]
    valid_precision = [r["context_precision"]["precision"] for r in results]


    mean_faithfulness = sum(valid_faithfulness) / len(valid_faithfulness) if valid_faithfulness else 0
    mean_relevance = sum(valid_relevance) / len(valid_relevance) if valid_relevance else 0
    mean_precision = sum(valid_precision) / len(valid_precision) if valid_precision else 0


    report = {
        "timestamp": datetime.now().isoformat(),
        "n_questions": len(eval_set),
        "n_valid_faithfulness": len(valid_faithfulness),
        "n_valid_relevance": len(valid_relevance),
    
        "mean_faithfulness": mean_faithfulness,
        "mean_relevance": mean_relevance,
        "mean_context_precision": round(mean_precision, 3),
        "results": results
    }


    EVAL_REPORTS_PATH.mkdir(parents=True, exist_ok=True)
    report_path = EVAL_REPORTS_PATH / f"eval_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=1)

    logger.info(f"Evaluation report complete.")
    logger.info(f"Mean faithfulness: {mean_faithfulness}")
    logger.info(f"Mean relevance: {mean_relevance}")
    logger.info(f"Mean context precision: {mean_precision:.3f}")
    logger.info(f"Report saved to {report_path}")

    return report

def score_context_precision(question: str, chunks: list[dict]) -> dict:
    """Score what fraction of retrieved chunks are relevant to the question."""

    relevant_count = 0
    chunk_scores = []

    for i, chunk in enumerate(chunks):
        prompt = CONTEXT_PRECISION_PROMPT.format(
            question=question,
            chunk=chunk["text"]
        )
        result = judge(prompt)

        # judge() returns {"score": ..., "reason": ...} but here we expect {"relevant": ..., "reason": ...}
        # Handle both in case the model uses the wrong key
        relevant = result.get("relevant", False)
        if isinstance(relevant, str):
            relevant = relevant.lower() == "true"

        chunk_scores.append({
            "chunk_index": i,
            "source": chunk["source"],
            "relevant": relevant,
            "reason": result.get("reason", "")
        })

        if relevant:
            relevant_count += 1

    precision = relevant_count / len(chunks) if chunks else 0.0

    return {
        "precision": round(precision, 3),
        "relevant_chunks": relevant_count,
        "total_chunks": len(chunks),
        "chunk_scores": chunk_scores
    }

if __name__ == "__main__":
    run_eval()


