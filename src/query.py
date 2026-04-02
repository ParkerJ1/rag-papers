import logging
import requests
import sys 
from sentence_transformers import SentenceTransformer, CrossEncoder

import chromadb

from config import *

import os
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

logger.info("Loading embedding model and cross encoder model")
_embed_model = SentenceTransformer(EMBED_MODEL)
_cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL)

def rerank_chunks(question: str, chunks: list[dict]) -> list[dict]:
    """ Takes the retrieved chunks and uses a cross encoder to rerank them."""

    #create pairs
    pairs = [[question, chunk["text"]] for chunk in chunks]
    scores = _cross_encoder.predict(pairs)

    for chunk, score in zip(chunks, scores):
        chunk["rerank_score"] = float(score)

    reranked = sorted(chunks, key=lambda x: x["rerank_score"], reverse=True)

    return reranked[:TOP_K]




def assess_confidence(chunks: list[dict]) -> dict:
    """ Returns mean confidence rating of returned chunks."""

    if not chunks:
        return {"level": "none",
                "mean_distance": None}

    total_distance = sum([chunk["distance"] for chunk in chunks])
    mean_distance = total_distance / len(chunks)

    level = "high" if mean_distance < 0.5 else "medium" if mean_distance < 1.0 else "low"

    return {"level": level,
            "mean_distance": mean_distance}

def retrieve_chunks(question: str, collection, source_filter: str | None = None) -> list[dict]:
    """Embed the question and retrieve top K relevant chunks"""

    logging.info("Embedding query")
    question_embedding = _embed_model.encode(question).tolist()

    query_kwargs = {
        "query_embeddings": [question_embedding],
        "n_results": RETRIEVAL_K,
        "include": ["documents", "metadatas", "distances"]
    }

    if source_filter:
        query_kwargs["where"] = {"source": {"$eq": source_filter}}
    
    results = collection.query(**query_kwargs)

    #Rework results into chunks
    chunks = []

    for document, metadata, distance in zip(results["documents"][0], results["metadatas"][0], results["distances"][0]):
        chunk = {
            "text": document,
            "source": metadata["source"],
            "distance": distance
        }
        chunks.append(chunk)

    return chunks

def build_context(chunks: list[dict]) -> str:
    """Build context string from retrieved chunks."""
    return "\n\n".join([chunk["text"] for chunk in chunks])

def generate_answer(context: str, question: str) -> str:
    """Send a prompt to llama-server and return the generated answer."""

    try:
        response = requests.post(
            LLAMA_SERVER_URL,
            json={
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant. Use the provided context to answer the Question. Only use information from the context. If the answer is not in the context, say so."},
                    {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"}
                ],
                "n_predict": 515,
                "temperature": 0.2
            })
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except requests.exceptions.ConnectionError:
        logger.error(f"Could not connect to llama server at {LLAMA_SERVER_URL}. Is it running?")
        return f"Error: Could not connect to the LLM server at {LLAMA_SERVER_URL}. Please start it and try again."
    except requests.exceptions.HTTPError as e:
        logger.error(f"LLM server returned an error: {e}")
        return f"Error: LLM server returned an error: {e}"

def query(question: str, source_filter: str | None = None) -> dict:
    """Full query pipeline"""
    logger.info(f"Query: {question}")

    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_collection(name=COLLECTION_NAME)

    chunks = retrieve_chunks(question, collection, source_filter=source_filter)
    logger.info(f"Retrieved {len(chunks)} chunks")
    logger.info(f"Reranking {RETRIEVAL_K} chunks, keeping top {TOP_K}")
    chunks = rerank_chunks(question, chunks)

    context = build_context(chunks)
    answer = generate_answer(context, question)
    if answer.startswith("Error:"):
        confidence = {"level": "error", "mean_distance": None}
    else:
        confidence = assess_confidence(chunks)
    logger.info(f"Confidence: {confidence}")


    return {
        "question": question,
        "answer": answer,
        "source": [c["source"] for c in chunks],
        "confidence": confidence,
        "chunks": chunks
    }



if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python src/query.py <question>")
        sys.exit(1)

    question = sys.argv[1]
    result = query(question)
    print(f"Question: {result['question']}")
    print(f"Answer: {result['answer']}")
    print(f"Source: {result['source']}")
       