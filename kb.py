"""Step 3: Knowledge base + retriever (TF-IDF, no heavy model downloads, runs on free tiers)."""
import json, os, re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

DATA = os.path.join(os.path.dirname(__file__), "data")


SEED_PAGES = [  # (keywords in the seed heading, page that backs that topic)
    (("contact", "office hours", "social", "demo"), "https://cyberathon.com/Contact"),
    (("internship",), "https://cyberathon.com/Internship"),
    (("career", "job"), "https://cyberathon.com/Careers"),
    (("service", "web development", "mobile", "marketing", "it services"), "https://cyberathon.com/Services"),
    (("product", "project"), "https://cyberathon.com/Projects"),
]


def seed_source(title):
    t = title.lower()
    for keys, url in SEED_PAGES:
        if any(k in t for k in keys):
            return url
    return "https://cyberathon.com/About"


def load_seed():
    text = open(os.path.join(DATA, "seed_knowledge.md"), encoding="utf-8").read()
    chunks = []
    for block in re.split(r"\n(?=## )", text.strip()):
        head, _, body = block.partition("\n")
        title = head.replace("## ", "").strip()
        chunks.append({"source": seed_source(title), "page": "curated", "title": title,
                       "text": f"{title}: {body.strip()}"})
    return chunks


def load_scraped():
    p = os.path.join(DATA, "chunks.json")
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else []


SYNONYMS = {  # trigger words in the question -> extra words that steer retrieval to the right topic
    "phone numbers whatsapp call": ["phone", "call", "number", "mobile", "whatsapp", "dial", "contact"],
    "email addresses mail": ["email", "mail", "gmail", "write"],
    "office address location": ["address", "location", "located", "where", "map", "visit", "reach", "directions"],
    "office hours timings open": ["timing", "timings", "hours", "open", "close", "closed", "sunday", "saturday", "time", "when"],
    "job openings vacancies careers": ["job", "jobs", "vacancy", "vacancies", "hiring", "opening", "openings",
                                       "career", "careers", "recruit", "position", "salary"],
    "apply registration application form": ["apply", "register", "registration", "enroll", "join", "admission"],
    "internship certificate benefits": ["certificate", "ojt", "fresher", "freshers"],
    "services offer provide": ["service", "services", "offer", "do", "build", "make", "create"],
    "mobile app android ios development": ["app", "apps", "android", "ios", "mobile", "application", "applications"],
    "CEO manager Devanand": ["ceo", "founder", "owner", "director", "boss", "head"],
    "HR team human resource Surabhi": ["hr", "human", "resource", "resources"],
    "Dot Net ASP.NET Razor Entity Framework": [".net", "dotnet", "asp.net", "c#", "asp"],
    "fees stipend cost price duration months": ["fee", "fees", "cost", "stipend", "charges", "duration",
                                                "long", "months"],
}


FILLER = ("tell give show want need know detail details information info please hello hi hey thank thanks help available like "
          "looking interested get list name good yes ok okay also more company cyberathon offer provide join work learn "
          "can could would will explain describe about ask question questions know").split()


def stem(w):
    for suf in ("ies", "es", "s"):
        if len(w) > 3 and w.endswith(suf):
            return w[: -len(suf)] + ("y" if suf == "ies" else "")
    return w


def analyzer(doc):
    words = [stem(w) for w in re.findall(r"[a-z0-9+#.]+", doc.lower()) if w not in ENGLISH_STOP_WORDS and len(w) > 1]
    return words + [a + " " + b for a, b in zip(words, words[1:])]


def expand(query):
    words = set(re.findall(r"[a-z]+", query.lower()))
    extra = [k for k, trig in SYNONYMS.items() if words & set(trig)]
    return query + " " + " ".join(extra)


class KnowledgeBase:
    def __init__(self):
        self.chunks = load_seed() + load_scraped()
        docs = [(c["title"] + " ") * 2 + c["text"] for c in self.chunks]  # headings count double
        self.vec = TfidfVectorizer(analyzer=analyzer, sublinear_tf=True)
        self.matrix = self.vec.fit_transform(docs)
        # rare-keyword index: words found in only 1-2 passages (e.g. "MBA", "Kotlin") are strong topic signals
        self.tokens = [set(w for w in analyzer(d) if " " not in w) for d in docs]
        df = {}
        for t in self.tokens:
            for w in t:
                df[w] = df.get(w, 0) + 1
        self.rare = {w for w, n in df.items() if n <= 2 and len(w) > 2}
        # words the bot "knows": everything in the KB + synonym triggers + everyday filler words
        trig = {stem(t) for words in SYNONYMS.values() for t in words}
        self.vocab = set(df) | trig | {stem(w) for w in FILLER}

    def in_domain(self, query):
        """False when most of the question's content words are unknown to the bot (e.g. 'weather in nagpur today')."""
        toks = [w for w in analyzer(query) if " " not in w]
        unknown = [w for w in toks if w not in self.vocab]
        return not (len(unknown) >= 2 and len(unknown) / max(1, len(toks)) >= 0.5)

    def search(self, query, k=4):
        if not self.in_domain(query):
            return []
        sims = linear_kernel(self.vec.transform([expand(query)]), self.matrix).ravel()
        qw = {w for w in analyzer(query) if " " not in w} & self.rare
        if qw:
            for i, toks in enumerate(self.tokens):
                sims[i] += min(0.4, 0.2 * len(qw & toks))
        idx = sims.argsort()[::-1][:k]
        return [(self.chunks[i], float(sims[i])) for i in idx if sims[i] > 0]

    def score_sentences(self, query, sentences):
        """Relevance of each sentence to the query (used to pick the exact details a question asks for)."""
        if not sentences:
            return []
        sims = linear_kernel(self.vec.transform([expand(query)]), self.vec.transform(sentences)).ravel()
        return [float(x) for x in sims]
