import logging
import requests
from pathlib import Path

from config import *
from ingest import ingest_pdf, is_already_ingested
import chromadb
from urllib.parse import unquote

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

logger = logging.getLogger(__name__)

ZOTERO_BASE_URL = "http://localhost:23119/api/users/0"
ZOTERO_HEADERS = {"zotero-allowed-request": "1"}

def get_collections() -> list[dict]:
    """Fetch collections from local Zotero library"""
    response = requests.get(
        f"{ZOTERO_BASE_URL}/collections",
            headers=ZOTERO_HEADERS
        )
    response.raise_for_status()
    return response.json()

def find_collection_key(name: str, parent_key: str | None = None) -> str | None:
    collections = get_collections()
    for collection in collections:
        data = collection.get("data", {})
        if data.get("name") == name:
            if parent_key is None:
                return collection["key"]
            if data.get("parentCollection") == parent_key:
                return collection["key"]
    return None

def get_collection_items(collection_key: str) -> list[dict]:
    """Fetch all items in a collection, handling pagination."""
    items = []
    start = 0
    limit = 25

    while True:
        response = requests.get(
            f"{ZOTERO_BASE_URL}/collections/{collection_key}/items",
            headers=ZOTERO_HEADERS,
            params={"limit": limit, "start": start}
        )
        response.raise_for_status()
        batch = response.json()
        if not batch:
            break
        items.extend(batch)
        total = int(response.headers.get("Total-Results", 0))
        start += limit
        if start >= total:
            break

    logger.info(f"Fetched {len(items)} items from collection")
    return items

def get_attachment_path(item_key: str) -> Path | None:
    """Get the local file path of a PDF attachment for a given item key."""
    response = requests.get(
        f"{ZOTERO_BASE_URL}/items/{item_key}/file",
        headers=ZOTERO_HEADERS,
        allow_redirects=False
    )
    
    if response.status_code in (301, 302, 303, 307, 308):
        location = response.headers.get("Location", "")
        logger.info(f"Redirect location for {item_key}: {location}")
        if location.startswith("file:///"):
            file_path = unquote(location.replace("file:///", "").replace("/", "\\"))
            path = Path(file_path)
            if path.exists() and path.suffix.lower() == ".pdf":
                return path
            else:
                logger.warning(f"Resolved path does not exist: {path}")
    return None

def get_item_children(item_key: str) -> list[dict]:
    """Fetch all children of a given item."""
    response = requests.get(
        f"{ZOTERO_BASE_URL}/items/{item_key}/children",
        headers=ZOTERO_HEADERS
    )
    response.raise_for_status()
    return response.json()

PARENT_ITEM_TYPES = {"journalArticle", "conferencePaper", "preprint", "report", "book", "bookSection", "thesis"}

def get_pdf_attachments(collection_key: str) -> list[Path]:
    """Get local PDF paths for all items in a collection. Only processes PDFs
    attached to parent items (journalArticle, preprint, etc.) — standalone
    attachments are excluded."""
    items = get_collection_items(collection_key)
    pdfs = []

    for item in items:
        data = item.get("data", {})
        item_type = data.get("itemType")

        if item_type in PARENT_ITEM_TYPES:
            children = get_item_children(item["key"])
            for child in children:
                child_data = child.get("data", {})
                if child_data.get("itemType") == "attachment" and child_data.get("contentType") == "application/pdf":
                    path = get_attachment_path(child["key"])
                    if path:
                        pdfs.append(path)
                        logger.info(f"Found PDF (child of {item_type}): {path.name}")
                    else:
                        logger.warning(f"Could not resolve path for child attachment {child['key']}")

    return pdfs

def ingest_zotero_collection(collection_name: str, parent_collection_name: str | None = None, dry_run: bool = False):
    """Ingest all PDFs from a Zotero collection, by name"""

    parent_key = None
    if parent_collection_name:  
        parent_key = find_collection_key(parent_collection_name)
        if not parent_key:
            logger.error(f"Parent collection {parent_collection_name} not found.")
            return
        logger.info(f"Parent collection {parent_collection_name} found with key: {parent_key}")

    collection_key = find_collection_key(collection_name, parent_key=parent_key)
    if not collection_key:
        logger.error(f"Collection {collection_name} not found.")
        return
    logger.info(f"Collection {collection_name} found with key: {collection_key}")

    pdf_paths = get_pdf_attachments(collection_key)
    if not pdf_paths:
        logger.info(f"No PDFs found in collection {collection_name}.")
        return
    logger.info(f"Found {len(pdf_paths)} PDFs in collection {collection_name}.")

    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_or_create_collection(COLLECTION_NAME)

    for pdf_path in pdf_paths:
        if is_already_ingested(pdf_path, collection):
            logger.info(f"Skipping {pdf_path.name} as it has already been ingested, skipping...")
            continue
        ingest_pdf(pdf_path)

