import json
from pathlib import Path
from typing import Any

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from services.vectorstore import get_vector_store

RAW_ARTICLES_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100


def load_unique_articles(raw_dir: Path = RAW_ARTICLES_DIR) -> dict[str, dict[str, Any]]:
    articles: dict[str, dict[str, Any]] = {}
    for path in raw_dir.rglob("*.json"):
        data = json.loads(path.read_text(encoding="utf-8"))
        articles.setdefault(data["slug"], data)
    return articles


def flatten_content(blocks: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for block in blocks:
        if block["_type"] == "block":
            parts.append("".join(child.get("text", "") for child in block["children"]))
        elif block["_type"] == "accordion":
            for item in block["items"]:
                parts.append(item["title"])
                parts.append(flatten_content(item["content"]))
        elif block["_type"] == "callToActionLink":
            parts.append(block["title"])

    return "\n".join(part for part in parts if part)


def build_metadata(article: dict[str, Any]) -> dict[str, str]:
    categories = ", ".join(sorted({c["category"] for c in article["categories"]}))
    subcategories = ", ".join(sorted({c["subcategory"] for c in article["categories"]}))

    return {
        "slug": article["slug"],
        "url": article["url"],
        "title": article["title"],
        "category": categories,
        "subcategory": subcategories,
    }


def build_documents(article: dict[str, Any], splitter: RecursiveCharacterTextSplitter) -> tuple[list[Document], list[str]]:
    text = f"{article['title']}\n\n{flatten_content(article['content'])}"
    chunks = splitter.split_text(text)
    metadata = build_metadata(article)

    documents = [Document(page_content=chunk, metadata=metadata) for chunk in chunks]
    ids = [f"{article['slug']}-{index}" for index in range(len(chunks))]

    return documents, ids


def ingest_articles() -> None:
    articles = load_unique_articles()
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    vector_store = get_vector_store()

    print(f"Found {len(articles)} unique articles to ingest")

    for index, (slug, article) in enumerate(articles.items(), start=1):
        documents, ids = build_documents(article, splitter)
        vector_store.add_documents(documents, ids=ids)
        print(f"[{index}/{len(articles)}] Ingested {slug} -> {len(documents)} chunk(s)")


if __name__ == "__main__":
    ingest_articles()
