import json
import re
import time
from pathlib import Path
from typing import Any

import httpx

BASE_URL = "https://support.spotify.com"
LOCALE = "br-pt"
SITEMAP_INDEX_URL = f"{BASE_URL}/api/web/sitemap/"
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; spotify-support-rag-bot/1.0)"}
REQUEST_DELAY_SECONDS = 0.5
MAX_RETRIES = 3
RETRY_BACKOFF_SECONDS = 2.0

RAW_ARTICLES_DIR = Path(__file__).resolve().parent.parent / "data" / "raw" / "articles"

NEXT_DATA_PATTERN = re.compile(r'<script id="__NEXT_DATA__"[^>]*>(.*?)</script>', re.DOTALL)


def _get_with_retries(client: httpx.Client, url: str) -> httpx.Response:
    for attempt in range(1, MAX_RETRIES + 1):
        response = client.get(url, headers=HEADERS)
        if response.status_code < 500:
            response.raise_for_status()
            return response
        if attempt == MAX_RETRIES:
            response.raise_for_status()
        time.sleep(RETRY_BACKOFF_SECONDS * attempt)
    raise AssertionError("unreachable")


def _fetch_next_data(client: httpx.Client, url: str) -> dict[str, Any]:
    response = _get_with_retries(client, url)
    match = NEXT_DATA_PATTERN.search(response.text)
    if not match:
        raise ValueError(f"No __NEXT_DATA__ block found at {url}")
    return json.loads(match.group(1))


def _get_sitemap_locs(client: httpx.Client, sitemap_url: str) -> list[str]:
    response = _get_with_retries(client, sitemap_url)
    return re.findall(r"<loc>(.*?)</loc>", response.text)


def get_article_urls(client: httpx.Client) -> list[str]:
    vertical_sitemaps = _get_sitemap_locs(client, SITEMAP_INDEX_URL)

    article_urls: set[str] = set()
    for sitemap_url in vertical_sitemaps:
        for url in _get_sitemap_locs(client, sitemap_url):
            if f"/{LOCALE}/article/" in url:
                article_urls.add(url)

    return sorted(article_urls)


def build_category_map(client: httpx.Client) -> dict[str, list[dict[str, str]]]:
    home_data = _fetch_next_data(client, f"{BASE_URL}/{LOCALE}/")
    categories = home_data["props"]["pageProps"]["sharedPageData"]["categories"]

    category_map: dict[str, list[dict[str, str]]] = {}
    for category in categories:
        for subcategory in category["childCategories"]:
            for article in subcategory["articles"]:
                category_map.setdefault(article["slug"], []).append(
                    {
                        "category": category["title"],
                        "category_slug": category["slug"],
                        "subcategory": subcategory["title"],
                        "subcategory_slug": subcategory["slug"],
                    }
                )

    return category_map


def _slug_from_url(url: str) -> str:
    return url.rstrip("/").rsplit("/", 1)[-1]


def scrape_article(client: httpx.Client, url: str, categories: list[dict[str, str]]) -> dict[str, Any]:
    data = _fetch_next_data(client, url)
    article = data["props"]["pageProps"]["article"]

    return {
        "slug": _slug_from_url(url),
        "url": url,
        "title": article["title"],
        "updated_at": article.get("updatedAt"),
        "categories": categories,
        "related_articles": article.get("relatedArticles", []),
        "content": article.get("content", []),
    }


def _output_paths_for_article(article: dict[str, Any]) -> list[Path]:
    category_slugs = sorted({c["category_slug"] for c in article["categories"]})
    if not category_slugs:
        return [RAW_ARTICLES_DIR / f"{article['slug']}.json"]
    return [RAW_ARTICLES_DIR / category_slug / f"{article['slug']}.json" for category_slug in category_slugs]


def scrape_all_articles() -> None:
    RAW_ARTICLES_DIR.mkdir(parents=True, exist_ok=True)

    with httpx.Client(timeout=30.0) as client:
        category_map = build_category_map(client)
        article_urls = get_article_urls(client)

        print(f"Found {len(article_urls)} articles for locale '{LOCALE}'")

        for index, url in enumerate(article_urls, start=1):
            slug = _slug_from_url(url)
            categories = category_map.get(slug, [])

            try:
                article = scrape_article(client, url, categories)
            except (httpx.HTTPError, ValueError, KeyError) as exc:
                print(f"[{index}/{len(article_urls)}] Failed {url}: {exc}")
                continue

            payload = json.dumps(article, ensure_ascii=False, indent=2)
            for output_path in _output_paths_for_article(article):
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_text(payload, encoding="utf-8")
            print(f"[{index}/{len(article_urls)}] Saved {slug} -> {len(_output_paths_for_article(article))} folder(s)")

            time.sleep(REQUEST_DELAY_SECONDS)


if __name__ == "__main__":
    scrape_all_articles()
