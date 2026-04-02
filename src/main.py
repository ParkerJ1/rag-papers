import argparse
import logging
import sys
from pathlib import Path

from ingest import ingest_folder, ingest_pdf
from query import query

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

def main():
    parser = argparse.ArgumentParser(description="RAG pipeline for querying research papers")
    subparsers = parser.add_subparsers(dest="command", required=True)

    #Ingest subcommand
    ingest_parser = subparsers.add_parser("ingest", help="Ingest a pdf or folder of pdfs into vector store")
    ingest_parser.add_argument("path", type=Path, help="Path to pdf or folder of pdfs")

    # Query subcommand
    query_parser = subparsers.add_parser("query", help="Query the vector store with a question")
    query_parser.add_argument("question", type=str, help="Question to ask the vector store")
    query_parser.add_argument("--filter", type=str, default = None, help="Filter results to a specific source PDF by filename")

    zotero_parser = subparsers.add_parser("zotero", help="Ingest all PDFs from a Zotero collection")
    zotero_parser.add_argument("collection", type=str, help="Name of the Zotero collection to ingest")
    zotero_parser.add_argument("--parent", type=str, default=None, help="Name of the parent Zotero collection")

    args = parser.parse_args()

    if args.command == "ingest":
        if not args.path.exists():
            logger.error(f"File not found: {args.path}")
            sys.exit(1)
        if args.path.is_dir():
            ingest_folder(args.path)
        else:
            ingest_pdf(args.path)
    elif args.command == "query":
        result = query(args.question, source_filter=args.filter)
        print(f"Question: {result['question']}")
        print(f"Answer: {result['answer']}")
        print(f"Source: {set(result['source'])}")
        print(f"Confidence: {result['confidence']['level']} -- {result['confidence']['mean_distance']}")

    elif args.command == "zotero":
        from zotero import ingest_zotero_collection
        ingest_zotero_collection(args.collection, parent_collection_name=args.parent)



if __name__ == "__main__":
    main()