import os
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["ANONYMIZED_TELEMETRY"] = "False"

import logging
import sys
from pathlib import Path

import fitz
import chromadb
from sentence_transformers import SentenceTransformer

from config import *

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)


logger = logging.getLogger(__name__)


def extract_text(pdf_path: Path) -> str:
    """Extract text from a PDF file."""

    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def chunk_text(text :str, chunk_size: int, overlap: int, split_sentences: bool = True) -> list[str]:
    """Split text into overlapping chunks - possibly respecting sentence boundaries."""
    chunks = []
    
    if split_sentences:
        import nltk
        sentences = nltk.sent_tokenize(text)
        current_chunk = []
        current_size = 0

        for sentence in sentences:
            sentence_size = len(sentence)

            if current_size + sentence_size > chunk_size and current_chunk:
                chunks.append(" ".join(current_chunk))

                # Keep last few sentences for overlap
                # Reach back into current_chunk to grab overlap number of characters, 
                # use this to initialise next iteration of current_chunk
                overlap_chunk = []
                overlap_size = 0
                for s in reversed(current_chunk): #Goes through them in reverse order
                    if overlap_size + len(s) > overlap:
                        break
                    overlap_chunk.insert(0, s) # inserts at the front, so counteracts the reversing
                    overlap_size += len(s)

                current_chunk = overlap_chunk
                current_size = overlap_size

            current_chunk.append(sentence)
            current_size += sentence_size

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks

    else:
        for i in range(0, len(text), chunk_size - overlap):
            chunk = text[i:i + chunk_size]
            chunks.append(chunk)

        return chunks

def ingest_pdf(pdf_path: Path):
    """Full ingestion pipeline for pdf"""

    logger.info(f"Ingesting: {pdf_path.name}")
    text = extract_text(pdf_path)

    logger.info(f"Chunking: {pdf_path.name}")
    chunks = chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP)

    logger.info(f"Embedding: {pdf_path.name}")
    model = SentenceTransformer(EMBED_MODEL)
    embeddings = model.encode(chunks)

    # Storing
    logger.info(f"Storing: {pdf_path.name} into {CHROMA_PATH}")
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_or_create_collection(COLLECTION_NAME)

    ids = [f"{pdf_path.stem}_chunk_{i}" for i, _ in enumerate(chunks)]
    metadatas = [{"source": pdf_path.name, "chunk_index": i} for i, _ in enumerate(chunks)]

    collection.add(
        ids=ids,
        documents=chunks,
        embeddings=embeddings.tolist(),
        metadatas=metadatas,
    )

    logger.info(f"Ingestion complete: {len(chunks)} chunks stored from {pdf_path.name}")

def is_already_ingested(pdf_path: Path, collection) -> bool:
    """Check if a pdf has already been ingested by looking for its chunks"""
    results = collection.get(where={"source":pdf_path.name})
    return len(results["ids"]) > 0

def ingest_folder(folder_path: Path):
    """Ingest all pdfs in a folder"""
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_or_create_collection(COLLECTION_NAME)

    pdfs = list(folder_path.glob("*.pdf"))
    if not pdfs:
        logger.info(f"No pdfs found in {folder_path}")
        return
    
    logger.info(f"Found {len(pdfs)} pdfs in {folder_path}")

    for pdf_path in pdfs:
        if is_already_ingested(pdf_path, collection):
            logger.info(f"Skipping {pdf_path.name} as it has already been ingested, skipping...")
            continue
        ingest_pdf(pdf_path)

        



if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python src/ingest.py <path_to_pdf_or_folder>")
        sys.exit(1)

    pdf_path = Path(sys.argv[1])
    if not pdf_path.exists():
        print(f"File not found: {pdf_path}")
        sys.exit(1)

    if pdf_path.is_dir():
        ingest_folder(pdf_path)
    else:
        client = chromadb.PersistentClient(path=CHROMA_PATH)
        collection = client.get_or_create_collection(COLLECTION_NAME)
        if is_already_ingested(pdf_path, collection):
            print(f"File already ingested: {pdf_path}")
            logger.info(f"File already ingested: {pdf_path}, skipping...")
        else:
            ingest_pdf(pdf_path)








    
