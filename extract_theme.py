"""
Read cyberathon.com's own stylesheets and work out its brand colours -> data/theme.json
The chatbot (app.py) loads this file so the interface automatically matches the website.

Run:  python extract_theme.py        (scraper.py also runs it for you)
"""
import colorsys, json, os, re
from collections import Counter
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

SITE = "https://cyberathon.com/"
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; CyberathonChatbot/1.0)"}
# third-party/library CSS whose colours are not the brand's
SKIP_CSS = re.compile(r"bootstrap|fontawesome|font-awesome|flaticon|animate|slick|owl|swiper|magnific|"
                      r"aos|nice-select|jquery|fancybox|lightbox|icomoon|googleapis|gstatic|cdn", re.I)
VAR_HINT = re.compile(r"--[\w-]*(primary|theme|main|brand|base|accent|secondary|heading|dark)[\w-]*\s*:\s*([^;}{]+)", re.I)
HEX = re.compile(r"#(?:[0-9a-fA-F]{6}|[0-9a-fA-F]{3})\b")
RGB = re.compile(r"rgba?\(\s*(\d{1,3})\s*,\s*(\d{1,3})\s*,\s*(\d{1,3})")


def to_rgb(token):
    token = token.strip()
    m = HEX.search(token)
    if m:
        h = m.group(0)[1:]
        if len(h) == 3:
            h = "".join(c * 2 for c in h)
        return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))
    m = RGB.search(token)
    if m:
        return tuple(min(255, int(x)) for x in m.groups())
    return None


def hsl(rgb):
    h, l, s = colorsys.rgb_to_hls(*(c / 255 for c in rgb))
    return h * 360, s, l


def hex_of(rgb):
    return "#%02x%02x%02x" % rgb


def chromatic(rgb):
    h, s, l = hsl(rgb)
    return s >= 0.30 and 0.18 <= l <= 0.80


def collect_css():
    css = []
    r = requests.get(SITE, headers=HEADERS, timeout=25)
    soup = BeautifulSoup(r.text, "lxml")
    css += [s.get_text() for s in soup.find_all("style")]
    for link in soup.find_all("link", rel=lambda v: v and "stylesheet" in v):
        href = urljoin(r.url, link.get("href", ""))
        if SKIP_CSS.search(href):
            continue
        try:
            css.append(requests.get(href, headers=HEADERS, timeout=25).text)
        except requests.RequestException:
            pass
    return "\n".join(css)


def analyse(css):
    # 1) named theme variables are the most reliable signal
    named = [to_rgb(v) for _, v in VAR_HINT.findall(css)]
    named = [c for c in named if c]
    # 2) otherwise count every colour used in the stylesheet
    counts = Counter()
    for tok in HEX.findall(css) + [m.group(0) for m in RGB.finditer(css)]:
        c = to_rgb(tok)
        if c:
            counts[c] += 1
    for c in named:
        counts[c] += 25  # boost
    ranked = [c for c, _ in counts.most_common(80)]
    chroma = [c for c in ranked if chromatic(c)]
    if not chroma:
        raise RuntimeError("no brand colours found")
    primary = chroma[0]
    p_h = hsl(primary)[0]
    # dark: the most used genuinely dark colour (navy / near-black headings, footer)
    darks = [c for c in ranked if hsl(c)[2] < 0.28 and not (c == (0, 0, 0))]
    dark = darks[0] if darks else tuple(int(x * 0.35) for x in primary)
    accent = next((c for c in chroma[1:] if min(abs(hsl(c)[0] - p_h), 360 - abs(hsl(c)[0] - p_h)) > 35), None)
    return {"primary": hex_of(primary), "dark": hex_of(dark), "accent": hex_of(accent) if accent else None,
            "palette_seen": [hex_of(c) for c in chroma[:6]]}


def run():
    theme = analyse(collect_css())
    os.makedirs("data", exist_ok=True)
    json.dump(theme, open("data/theme.json", "w"), indent=1)
    print("[theme]", theme)
    return theme


if __name__ == "__main__":
    run()
