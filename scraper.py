"""
Scrape the ENTIRE cyberathon.com website -> clean -> chunk.
Discovery: sitemap.xml  +  every internal <a href> found while crawling (breadth-first).
Output:    data/raw_pages.json (full cleaned page text)  and  data/chunks.json (RAG chunks)

Run:  python scraper.py
"""
import io, json, os, re, time
from collections import Counter, deque
from urllib.parse import urljoin, urlparse, urldefrag
import xml.etree.ElementTree as ET

import requests
from bs4 import BeautifulSoup

BASE = "https://cyberathon.com"
HOST = "cyberathon.com"
SEEDS = ["/", "/About", "/Services", "/WebDesigning", "/DigitalMarketing", "/GraphicDesigning",
         "/AndroidDevelopment", "/WebDevelopment", "/SeoMarketing", "/DigitalSurvrlence",
         "/ITconsulting", "/SkillsDevelopment", "/Projects", "/Internship", "/Careers",
         "/Contact", "/Blog", "/Gallery", "/Home/qa", "/Home/privacy_policy"]
BROCHURE = "https://cyberathon.com/assets/CyberAthon_Technology_Brochure.pdf.pdf"
SKIP_EXT = (".jpg", ".jpeg", ".png", ".gif", ".webp", ".svg", ".pdf", ".mp4", ".css", ".js",
            ".ico", ".zip", ".woff", ".woff2", ".ttf", ".xml", ".txt", ".mp3")
SKIP_PATH = ("/Admin", "/assets", "/cdn-cgi")
MAX_PAGES = 400
DELAY = 0.4
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; CyberathonChatbot/1.0)"}
NOISE = re.compile(r"(book demo|enquiry form|inquiry|search here|all rights reserved|enroll now|apply now|"
                   r"please enter|cannot be empty|captcha|must contain|^submit$|^×$|^details$|no details found)", re.I)


def norm(url):
    url, _ = urldefrag(url.strip())
    p = urlparse(url)
    if not p.netloc:
        return None
    return f"https://{p.netloc.replace('www.', '')}{p.path.rstrip('/') or '/'}" + (f"?{p.query}" if p.query else "")


def crawlable(url):
    if not url:
        return False
    p = urlparse(url)
    return (p.netloc == HOST and not p.path.lower().endswith(SKIP_EXT)
            and not p.path.startswith(SKIP_PATH) and "logout" not in p.path.lower())


def from_sitemap():
    out = []
    for sm in (f"{BASE}/sitemap.xml", f"https://www.{HOST}/sitemap.xml"):
        try:
            r = requests.get(sm, headers=HEADERS, timeout=20)
            if r.status_code == 200 and "<loc>" in r.text:
                for loc in ET.fromstring(r.content).iter():
                    if loc.tag.endswith("loc") and loc.text:
                        out.append(norm(loc.text))
                print(f"[sitemap] {len(out)} urls from {sm}")
                break
        except Exception as e:
            print("[sitemap] skipped:", e)
    return out


def parse(html):
    soup = BeautifulSoup(html, "lxml")
    links = [a["href"] for a in soup.find_all("a", href=True)]
    for t in soup(["script", "style", "noscript", "nav", "footer", "header", "form", "img", "svg", "iframe", "button"]):
        t.decompose()
    title = re.sub(r"\s+", " ", soup.title.get_text()).strip() if soup.title else ""
    m = soup.find("meta", attrs={"name": "description"})
    desc = m["content"].strip() if m and m.get("content") else ""
    h1 = soup.find(["h1", "h2"])
    heading = h1.get_text(" ", strip=True) if h1 else ""
    lines = []
    for el in soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "td", "th", "blockquote", "span", "div"]):
        # only leaf-ish blocks, to avoid duplicating nested text
        if el.name in ("span", "div") and el.find(["p", "li", "h1", "h2", "h3", "h4", "div", "span"]):
            continue
        txt = re.sub(r"\s+", " ", el.get_text(" ", strip=True))
        if len(txt) < 3 or NOISE.search(txt) or re.fullmatch(r"[\d\s+\-/.]+", txt):
            continue
        lines.append(txt)
    return title, desc, heading, lines, links


def crawl():
    q = deque(dict.fromkeys([norm(urljoin(BASE, s)) for s in SEEDS] + from_sitemap()))
    seen, pages = set(), {}
    while q and len(pages) < MAX_PAGES:
        url = q.popleft()
        if url in seen or not crawlable(url):
            continue
        seen.add(url)
        try:
            r = requests.get(url, headers=HEADERS, timeout=25)
            if r.status_code != 200 or "text/html" not in r.headers.get("content-type", ""):
                print(f"[--] {r.status_code} {url}")
                continue
        except requests.RequestException as e:
            print("[err]", url, e)
            continue
        title, desc, heading, lines, links = parse(r.text)
        pages[url] = {"title": title, "heading": heading, "desc": desc, "lines": lines}
        print(f"[ok] {len(pages):3d} {url} ({len(lines)} lines)")
        for href in links:
            nl = norm(urljoin(r.url, href))
            if nl and nl not in seen and crawlable(nl):
                q.append(nl)
        time.sleep(DELAY)
    return pages


def brochure_text():
    try:
        from pypdf import PdfReader
        r = requests.get(BROCHURE, headers=HEADERS, timeout=40)
        r.raise_for_status()
        text = "\n".join((p.extract_text() or "") for p in PdfReader(io.BytesIO(r.content)).pages)
        lines = [re.sub(r"\s+", " ", l).strip() for l in text.splitlines() if len(l.strip()) > 3]
        print(f"[ok] brochure PDF ({len(lines)} lines)")
        return {"title": "Company brochure", "heading": "Company brochure", "desc": "", "lines": lines}
    except Exception as e:
        print("[--] brochure skipped:", e)
        return None


def to_chunks(pages, max_chars=750):
    n = len(pages)
    freq = Counter(l for p in pages.values() for l in set(p["lines"]))
    chunks = []
    for url, p in pages.items():
        label = urlparse(url).path or "/"
        title = p["heading"] or p["title"]
        kept, seen_local = [], set()
        for l in ([p["desc"]] if p["desc"] else []) + p["lines"]:
            if l in seen_local or (n > 6 and freq[l] > max(3, n * 0.4)):
                continue  # drop duplicates and boilerplate repeated on many pages
            seen_local.add(l)
            kept.append(l)
        texts, cur = [], ""
        for l in kept:
            piece = l if l.endswith((".", "?", "!", ":")) else l + "."
            if cur and len(cur) + len(piece) > max_chars:
                texts.append(cur.strip()); cur = ""
            cur += piece + " "
        if cur.strip():
            texts.append(cur.strip())
        chunks += [{"source": url, "page": label, "title": title, "text": t} for t in texts]
    return chunks


def run(progress=print):
    pages = crawl()
    b = brochure_text()
    if b:
        pages[BROCHURE] = b
    if not pages:
        raise RuntimeError("No pages scraped - check internet access / site availability.")
    os.makedirs("data", exist_ok=True)
    json.dump(pages, open("data/raw_pages.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    chunks = to_chunks(pages)
    json.dump(chunks, open("data/chunks.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    progress(f"Pages: {len(pages)} | Chunks: {len(chunks)} -> data/chunks.json")
    return len(pages), len(chunks)


if __name__ == "__main__":
    run()
    try:  # also pick up the website's brand colours so the chatbot UI matches it
        import extract_theme
        extract_theme.run()
    except Exception as e:
        print("[--] theme extraction skipped:", e)
