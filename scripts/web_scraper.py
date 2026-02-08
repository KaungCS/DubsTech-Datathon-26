from urllib.parse import urlparse

from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeoutError


def infer_domain_category(hostname: str) -> str:
    host = hostname.lower()
    if ".edu" in host or host.endswith(".edu"):
        return "Educational Platforms"
    if ".gov" in host or host.endswith(".gov"):
        return "Government and Public Services"
    if any(token in host for token in ["bbc", "news", "journal", "press", "times", "mirror", "media", "post", "cnn", "reuters", "ap"]):
        return "News and Media"
    if any(token in host for token in ["shop", "store", "ecommerce", "market", "cart"]):
        return "E-commerce"
    if any(
        token in host
        for token in ["video", "stream", "tv", "movie", "music", "youtube", "vimeo", "twitch", "netflix", "hulu", "spotify"]
    ):
        return "Streaming Platforms"
    if any(
        token in host
        for token in ["health", "clinic", "hospital", "med", "wellness", "care"]
    ):
        return "Health and Wellness"
    if any(
        token in host
        for token in ["tech", "science", "research", "lab", "ai", "data", "software", "developer"]
    ):
        return "Technology Science and Research"
    return "Technology Science and Research"


def build_affected_elements(soup: BeautifulSoup, max_items: int = 12) -> str:
    snippets = []

    def add_snippet(tag):
        if len(snippets) >= max_items:
            return
        snippet = str(tag)
        snippet = " ".join(snippet.split())
        snippets.append(snippet[:200])

    title = (soup.title.string or "").strip() if soup.title else ""
    if title:
        snippets.append(f"title: {title[:120]}")

    for heading in soup.find_all(["h1", "h2", "h3"]):
        if len(snippets) >= max_items:
            break
        text = (heading.get_text() or "").strip()
        if text:
            snippets.append(f"heading: {text[:120]}")

    for img in soup.find_all("img"):
        if not img.get("alt"):
            add_snippet(img)
        if len(snippets) >= max_items:
            break

    for link in soup.find_all("a"):
        text = (link.get_text() or "").strip()
        if not text and not link.get("aria-label") and not link.get("title"):
            add_snippet(link)
        if len(snippets) >= max_items:
            break

    for button in soup.find_all("button"):
        text = (button.get_text() or "").strip()
        if not text and not button.get("aria-label"):
            add_snippet(button)
        if len(snippets) >= max_items:
            break

    for input_tag in soup.find_all("input"):
        if input_tag.get("type") in {"hidden", "submit", "button", "image"}:
            continue
        if not input_tag.get("aria-label") and not input_tag.get("placeholder"):
            add_snippet(input_tag)
        if len(snippets) >= max_items:
            break

    for link in soup.find_all("a"):
        if len(snippets) >= max_items:
            break
        text = (link.get_text() or "").strip()
        if text:
            snippets.append(f"link: {text[:120]}")

    for paragraph in soup.find_all(["p", "li", "label"]):
        if len(snippets) >= max_items:
            break
        text = (paragraph.get_text() or "").strip()
        if text:
            snippets.append(f"text: {text[:160]}")

    if not snippets:
        tag_counts = {}
        for tag in soup.find_all(True):
            tag_counts[tag.name] = tag_counts.get(tag.name, 0) + 1
        top_tags = sorted(tag_counts.items(), key=lambda item: item[1], reverse=True)[:5]
        snippets = [name for name, _ in top_tags]

    return ", ".join(snippets)


def fetch_html(url: str, use_playwright: bool = True) -> str:
    """
    Fetch HTML from a URL.
    
    Args:
        url: URL to fetch
        use_playwright: If True, use Playwright to execute JavaScript frameworks.
                       If False, fall back to simple requests (faster but won't handle JS).
    
    Returns:
        HTML content as string, or empty string if fetch fails.
    """
    if not use_playwright:
        # Fallback to requests for simple static pages
        return _fetch_html_requests(url)
    
    return _fetch_html_playwright(url)


def _fetch_html_requests(url: str) -> str:
    """Fallback: Fetch HTML using requests (no JavaScript execution)."""
    import requests
    
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; AccessibilityML/1.0)",
        "Accept": "text/html,application/xhtml+xml",
        "Accept-Language": "en-US,en;q=0.9",
    }
    try:
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        if response.text and len(response.text) > 500:
            return response.text
    except Exception:
        pass
    return ""


def _fetch_html_playwright(url: str, retry_count: int = 0) -> str:
    """Fetch HTML using Playwright to execute JavaScript frameworks.
    
    Args:
        url: URL to fetch
        retry_count: Number of retries already attempted
    
    Returns:
        HTML content as string, or empty string if fetch fails.
    """
    max_retries = 2
    
    try:
        with sync_playwright() as p:
            # Launch browser (headless mode for speed)
            browser = p.chromium.launch(headless=True)
            context = browser.new_context(
                user_agent="Mozilla/5.0 (compatible; AccessibilityML/1.0)"
            )
            page = context.new_page()
            
            try:
                # Navigate to URL with longer timeout
                page.goto(url, timeout=30000, wait_until="networkidle")
                
                # Wait for network idle to ensure JS has finished rendering
                try:
                    page.wait_for_load_state("networkidle", timeout=10000)
                except PlaywrightTimeoutError:
                    pass  # Continue even if network idle times out
                
                # Wait for common content elements to be present
                try:
                    # Try to wait for body or main content
                    page.wait_for_selector("body", timeout=5000)
                except PlaywrightTimeoutError:
                    pass  # Continue if body doesn't load
                
                # Get fully rendered HTML
                html = page.content()
                
                # Check if we got meaningful content
                if html and len(html) > 1000:
                    return html
                
                # If content is sparse and we haven't retried, try again
                if len(html) < 1000 and retry_count < max_retries:
                    context.close()
                    browser.close()
                    return _fetch_html_playwright(url, retry_count + 1)
                
                # Return what we have if it's at least some content
                if html and len(html) > 500:
                    return html
                    
            except PlaywrightTimeoutError:
                # If page load times out, try to get whatever content loaded
                html = page.content()
                if html and len(html) > 500:
                    return html
                # Retry if we got too little content
                if retry_count < max_retries:
                    context.close()
                    browser.close()
                    return _fetch_html_playwright(url, retry_count + 1)
            except Exception:
                pass
            finally:
                context.close()
                browser.close()
    except Exception:
        pass
    
    return ""


def scrape_metadata(url, use_playwright: bool = True):
    """
    Scrape metadata from a URL.
    
    Args:
        url: URL to scrape
        use_playwright: If True, uses Playwright to handle JS frameworks.
                       Set to False for faster scraping of static pages.
    
    Returns:
        Dictionary with metadata about the page.
    """
    parsed = urlparse(url)
    path = parsed.path or "/"
    html_file_name = path.split("/")[-1] or "index.html"
    html_file_path = path
    domain_category = infer_domain_category(parsed.hostname or "")

    html = fetch_html(url, use_playwright=use_playwright)

    if html:
        soup = BeautifulSoup(html, "html.parser")
        affected_html_elements = build_affected_elements(soup)
    else:
        affected_html_elements = ""

    metadata = {
        "html_file_name": html_file_name,
        "html_file_path": html_file_path,
        "web_URL": url,
        "domain_category": domain_category,
        "affected_html_elements": affected_html_elements,
    }
    return metadata

# Example usage
if __name__ == "__main__":
    url = "https://example.com"
    metadata = scrape_metadata(url)
    print(metadata)