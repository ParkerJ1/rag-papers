import fitz
import chromadb
import chromadb.config
chromadb.config.Settings(anonymized_telemetry=False)

from sentence_transformers import SentenceTransformer
from pathlib import Path

from config import *

import sys

import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def extract_text(pdf_path: Path) -> str:
    """Extract text from a PDF file."""

    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def chunk_text(text :str, chunk_size: int, overlap: int) -> list[str]:
    """Split text into overlapping chunks"""
    chunks = []
    
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

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python src/ingest.py <path_to_pdf>")
        sys.exit(1)

    pdf_path = Path(sys.argv[1])
    if not pdf_path.exists():
        print(f"File not found: {pdf_path}")
        sys.exit(1)

    ingest_pdf(pdf_path)







    
