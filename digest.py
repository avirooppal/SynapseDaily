#!/usr/bin/env python3
"""Daily AI research digest -> Telegram.

Free stack: GitHub Actions (cron) + Gemini free tier (summaries) + Telegram Bot API.

Pipeline: fetch -> merge duplicates across sources -> rank -> enrich with real
page text -> LLM summary (grounded, JSON) -> Telegram.

Env vars:
  TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID   (required for sending)
  OPENROUTER_API_KEY or GEMINI_API_KEY   (required for summaries)
  OPENROUTER_MODEL  optional, overrides OpenRouter model (default: nvidia/nemotron-3-ultra-550b-a55b:free)
  GEMINI_MODEL      optional, overrides Gemini model (default: flash "latest" aliases)
  LLM_PROVIDER      optional, 'openrouter' or 'gemini' (auto-detected if omitted)
  TOP_N             optional, items per digest (default 8)
  TZ_NAME           optional, timezone for the header date (default Asia/Kolkata)
"""
import argparse
import calendar
import html
import json
import logging
import os
import re
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from difflib import SequenceMatcher
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlparse
from zoneinfo import ZoneInfo

import feedparser
import requests

log = logging.getLogger("digest")

# --------------------------------------------------------------------------
# Config — edit freely
# --------------------------------------------------------------------------
TOP_N = int(os.getenv("TOP_N", "8"))
LOOKBACK_HOURS = 30            # a bit over 24h so a late run never misses items
MIN_SCORE = 1.0                # drops weak, single-source noise
KIND_CAPS = {"paper": 4, "model": 3, "discussion": 2}   # keeps the digest varied
SEEN_FILE = Path(__file__).with_name("seen.json")
SEEN_TTL_DAYS = 30
HTTP_TIMEOUT = 20
MAX_TEXT_CHARS = 7000
LLM_SLEEP_SECONDS = 7          # stays under ~10 requests/min on the free tier
MAX_LLM_CALLS = 2 * TOP_N + 2  # hard cap on daily LLM requests

HEADERS = {"User-Agent": "ai-digest-bot/1.0 (personal research digest)"}

# Cheap prefilter for HN titles. The LLM can still discard non-AI items later.
AI_WORDS = re.compile(
    r"\b(ai|a\.i\.|llms?|gpt\S*|gemini|claude|llama\S*|qwen\S*|deepseek\S*|mistral|grok|"
    r"transformers?|diffusion|neural|rag|agents?|agentic|fine-?tun\w*|inference|quantiz\w*|"
    r"moe|rlhf|reasoning|multimodal|openai|anthropic|deepmind|hugging ?face|nvidia|vllm|"
    r"ollama|embeddings?|machine learning|deep learning|pytorch|cuda|tpu|gpu)\b",
    re.I,
)

# Hugging Face orgs whose new model repos count as "model releases".
HF_ORGS = [
    "deepseek-ai", "Qwen", "meta-llama", "mistralai", "google", "openai",
    "microsoft", "nvidia", "moonshotai", "zai-org", "MiniMaxAI", "allenai",
    "ibm-granite", "black-forest-labs",
]
HF_SKIP_VARIANTS = re.compile(r"(gguf|awq|gptq|fp8|fp4|int4|int8|mlx|onnx|bnb|quantized)", re.I)

# Official blogs (RSS/Atom). Dead feeds are skipped automatically.
LAB_FEEDS = {
    "OpenAI": "https://openai.com/news/rss.xml",
    "Google DeepMind": "https://deepmind.google/blog/rss.xml",
    "Google Research": "https://research.google/blog/rss/",
    "Hugging Face Blog": "https://huggingface.co/blog/feed.xml",
    "NVIDIA": "https://developer.nvidia.com/blog/feed/",
    "Microsoft Research": "https://www.microsoft.com/en-us/research/feed/",
}

REDDIT_SUBS = ["LocalLLaMA", "MachineLearning"]
ARXIV_QUERY = "cat:cs.AI OR cat:cs.LG OR cat:cs.CL"

GEMINI_MODELS = ["gemini-flash-latest", "gemini-flash-lite-latest"]
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

OPENROUTER_DEFAULT_MODEL = "nvidia/nemotron-3-ultra-550b-a55b:free"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

KIND_PRIORITY = {"discussion": 0, "paper": 1, "news": 2, "model": 3}


# --------------------------------------------------------------------------
# Data model + helpers
# --------------------------------------------------------------------------
ARXIV_RE = re.compile(r"arxiv\.org/(?:abs|pdf|html)/(\d{4}\.\d{4,5})", re.I)
HF_NON_MODEL = {"papers", "blog", "spaces", "datasets", "collections", "docs", "models", "api", "learn"}


def canon_key(url: str) -> str:
    """Stable identity for a URL so the same story from different sources merges."""
    m = ARXIV_RE.search(url)
    if m:
        return "arxiv:" + m.group(1)
    p = urlparse(url)
    host = p.netloc.lower().removeprefix("www.")
    path = p.path.rstrip("/")
    segs = [s for s in path.split("/") if s]
    if host == "huggingface.co" and len(segs) == 2 and segs[0].lower() not in HF_NON_MODEL:
        return f"hf:{segs[0]}/{segs[1]}".lower()
    if host == "github.com" and len(segs) == 2:
        return f"gh:{segs[0]}/{segs[1]}".lower()
    q = [(k, v) for k, v in parse_qsl(p.query) if not k.startswith("utm_") and k not in {"ref", "source"}]
    return f"{host}{path}" + (("?" + urlencode(q)) if q else "")


def _tok(t: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", t.lower()).strip()


def similar(a: str, b: str) -> bool:
    a, b = _tok(a), _tok(b)
    if not a or not b:
        return False
    if a == b:
        return True
    sm = SequenceMatcher(None, a, b)
    return sm.real_quick_ratio() >= 0.85 and sm.quick_ratio() >= 0.85 and sm.ratio() >= 0.85


@dataclass
class Item:
    title: str
    url: str
    kind: str = "news"                      # paper | model | news | discussion
    snippet: str = ""
    sources: dict = field(default_factory=dict)   # label -> link
    score: float = 0.0
    key: str = field(init=False, default="")

    def __post_init__(self):
        self.key = canon_key(self.url)


def get(url, **kw) -> requests.Response:
    r = requests.get(url, headers=HEADERS, timeout=HTTP_TIMEOUT, **kw)
    r.raise_for_status()
    return r


def entry_time(e):
    for attr in ("published_parsed", "updated_parsed"):
        t = getattr(e, attr, None)
        if t:
            return datetime.fromtimestamp(calendar.timegm(t), tz=timezone.utc)
    return None


def cutoff() -> datetime:
    return datetime.now(timezone.utc) - timedelta(hours=LOOKBACK_HOURS)


# --------------------------------------------------------------------------
# Fetchers — each returns list[Item]; failures are isolated in main()
# --------------------------------------------------------------------------
def fetch_hn():
    since = int(cutoff().timestamp())
    requests_ = [
        ("https://hn.algolia.com/api/v1/search", {"tags": "front_page", "hitsPerPage": 100}),
        ("https://hn.algolia.com/api/v1/search_by_date", {
            "tags": "story", "hitsPerPage": 100,
            "numericFilters": f"created_at_i>{since},points>40"}),
    ]
    out = []
    for url, params in requests_:
        for h in get(url, params=params).json().get("hits", []):
            title = h.get("title") or ""
            pts = h.get("points") or 0
            if pts < 30 or not AI_WORDS.search(title):
                continue
            hn_link = f"https://news.ycombinator.com/item?id={h['objectID']}"
            link = h.get("url") or hn_link
            kind = "paper" if ARXIV_RE.search(link) else "news"
            score = min(pts / 40, 8) + min((h.get("num_comments") or 0) / 150, 2)
            out.append(Item(title, link, kind, sources={"HN": hn_link}, score=score))
    return out


def fetch_hf_papers():
    out = []
    days = [None, (datetime.now(timezone.utc) - timedelta(days=1)).strftime("%Y-%m-%d")]
    for day in days:
        params = {"date": day} if day else {}
        try:
            data = get("https://huggingface.co/api/daily_papers", params=params).json()
        except Exception as e:  # the dated call is best-effort
            log.warning("hf daily_papers (%s) failed: %s", day, e)
            continue
        for e in data:
            p = e.get("paper", e)
            pid = p.get("id")
            if not pid:
                continue
            title = p.get("title") or e.get("title") or pid
            summary = p.get("summary") or e.get("summary") or ""
            ups = p.get("upvotes") or 0
            out.append(Item(
                title, f"https://arxiv.org/abs/{pid}", "paper", summary,
                sources={"HF Papers": f"https://huggingface.co/papers/{pid}"},
                score=min(ups / 15, 8)))
    return out


def fetch_arxiv():
    r = get("https://export.arxiv.org/api/query", params={
        "search_query": ARXIV_QUERY, "sortBy": "submittedDate",
        "sortOrder": "descending", "max_results": 60})
    out = []
    for e in feedparser.parse(r.content).entries:
        link = e.get("id", "")
        out.append(Item(re.sub(r"\s+", " ", e.title), link, "paper",
                        re.sub(r"\s+", " ", e.get("summary", "")),
                        sources={"arXiv": link}, score=0.3))
    return out


def fetch_reddit():
    out = []
    for sub in REDDIT_SUBS:
        try:
            feed = feedparser.parse(get(f"https://www.reddit.com/r/{sub}/top/.rss", params={"t": "day"}).content)
        except Exception as e:  # Reddit often blocks datacenter IPs; that's fine
            log.warning("reddit r/%s failed: %s", sub, e)
            continue
        for i, e in enumerate(feed.entries[:15]):
            body = e.content[0].value if e.get("content") else ""
            m = re.search(r'href="([^"]+)">\[link\]', body)
            ext = m.group(1) if m else ""
            thread = e.link
            is_external = ext and "reddit.com" not in ext
            out.append(Item(e.title, ext if is_external else thread,
                            "news" if is_external else "discussion",
                            sources={f"r/{sub}": thread}, score=max(0.5, 3 - 0.2 * i)))
    return out


def fetch_labs():
    out, since = [], cutoff()
    for name, url in LAB_FEEDS.items():
        try:
            feed = feedparser.parse(get(url).content)
        except Exception as e:
            log.warning("feed %s failed: %s", name, e)
            continue
        base = 3.0 if name in {"NVIDIA", "Microsoft Research", "Hugging Face Blog"} else 4.5
        for e in feed.entries[:10]:
            t = entry_time(e)
            if t and t < since:
                continue
            snippet = re.sub(r"<[^>]+>", " ", e.get("summary", ""))
            out.append(Item(re.sub(r"\s+", " ", e.title), e.link, "news",
                            re.sub(r"\s+", " ", snippet).strip(),
                            sources={name: e.link}, score=base))
    return out


def fetch_hf_models():
    out, since = [], cutoff()
    for org in HF_ORGS:
        try:
            data = get("https://huggingface.co/api/models", params={
                "author": org, "sort": "createdAt", "direction": "-1", "limit": 6}).json()
        except Exception as e:
            log.warning("hf models %s failed: %s", org, e)
            continue
        for m in data:
            mid = m.get("id") or m.get("modelId")
            created = m.get("createdAt")
            if not mid or not created or HF_SKIP_VARIANTS.search(mid):
                continue
            try:
                when = datetime.fromisoformat(created.replace("Z", "+00:00"))
            except ValueError:
                continue
            if when < since:
                continue
            url = f"https://huggingface.co/{mid}"
            out.append(Item(mid, url, "model", sources={"HF Hub": url}, score=2.5))
    return out


FETCHERS = [fetch_hn, fetch_hf_papers, fetch_arxiv, fetch_reddit, fetch_labs, fetch_hf_models]


# --------------------------------------------------------------------------
# Merge / seen / rank
# --------------------------------------------------------------------------
def merge(items):
    """Collapse the same story seen on several sources into one Item."""
    out = {}
    for it in items:
        target = out.get(it.key)
        if target is None:
            target = next((o for o in out.values() if similar(o.title, it.title)), None)
        if target is None:
            out[it.key] = it
            continue
        target.sources.update(it.sources)
        target.score += it.score
        if len(it.snippet) > len(target.snippet):
            target.snippet = it.snippet
        if KIND_PRIORITY[it.kind] > KIND_PRIORITY[target.kind]:
            target.kind = it.kind
    return list(out.values())


def load_seen():
    try:
        return json.loads(SEEN_FILE.read_text())
    except Exception:
        return {}


def save_seen(seen, delivered):
    today = datetime.now(timezone.utc).date()
    for it in delivered:
        seen[it.key] = {"d": today.isoformat(), "t": it.title}
    keep = {k: v for k, v in seen.items()
            if (today - datetime.fromisoformat(v["d"]).date()).days <= SEEN_TTL_DAYS}
    SEEN_FILE.write_text(json.dumps(keep, indent=1, ensure_ascii=False))


def is_seen(it, seen):
    return it.key in seen or any(similar(it.title, v.get("t", "")) for v in seen.values())


def rank(items, seen):
    fresh = [it for it in items if not is_seen(it, seen)]
    for it in fresh:
        it.score += 1.5 * (len(it.sources) - 1)     # cross-source mentions = signal
    fresh = [it for it in fresh if it.score >= MIN_SCORE]
    fresh.sort(key=lambda x: -x.score)
    picked, counts = [], {}
    for it in fresh:
        if counts.get(it.kind, 0) >= KIND_CAPS.get(it.kind, 99):
            continue
        counts[it.kind] = counts.get(it.kind, 0) + 1
        picked.append(it)
    return picked


# --------------------------------------------------------------------------
# Enrichment: get real text so the LLM summarizes facts, not vibes
# --------------------------------------------------------------------------
CONFIG_KEYS = ["num_hidden_layers", "hidden_size", "num_attention_heads", "num_key_value_heads",
               "intermediate_size", "num_experts", "n_routed_experts", "num_local_experts",
               "num_experts_per_tok", "max_position_embeddings", "vocab_size", "sliding_window"]


def hf_model_text(model_id: str) -> str:
    parts = []
    try:
        cfg = get(f"https://huggingface.co/{model_id}/raw/main/config.json").json()
        core = cfg.get("text_config") or cfg
        facts = {k: cfg[k] for k in ("architectures", "model_type") if cfg.get(k)}
        facts.update({k: core[k] for k in CONFIG_KEYS if core.get(k) is not None})
        parts.append("config.json: " + json.dumps(facts))
    except Exception:
        pass
    try:
        card = get(f"https://huggingface.co/{model_id}/raw/main/README.md").text
        parts.append(re.sub(r"^---.*?---\s*", "", card, flags=re.S))
    except Exception:
        pass
    return "\n\n".join(parts)


def arxiv_abstract(arxiv_id: str) -> str:
    feed = feedparser.parse(get("https://export.arxiv.org/api/query", params={"id_list": arxiv_id}).content)
    return re.sub(r"\s+", " ", feed.entries[0].get("summary", "")) if feed.entries else ""


def page_text(url: str) -> str:
    r = get(url)
    ctype = r.headers.get("content-type", "")
    if "html" not in ctype and "text" not in ctype:
        return ""
    import trafilatura
    return trafilatura.extract(r.text, include_comments=False, include_tables=False) or ""


def enrich(it: Item) -> str:
    text = ""
    try:
        if it.key.startswith("hf:"):
            text = hf_model_text(it.url.removeprefix("https://huggingface.co/"))
        elif it.key.startswith("arxiv:"):
            text = it.snippet if len(it.snippet) > 300 else arxiv_abstract(it.key[6:])
        else:
            text = page_text(it.url)
    except Exception as e:
        log.info("enrich failed for %s: %s", it.url, e)
    text = text or it.snippet
    return text[:MAX_TEXT_CHARS]


# --------------------------------------------------------------------------
# Gemini summarizer (REST, structured JSON output)
# --------------------------------------------------------------------------
SYSTEM_PROMPT = """You write a daily AI briefing for a software engineer who builds AI systems.
Rules:
- Use ONLY facts present in the SOURCE TEXT. Never guess numbers, architectures, benchmarks or licenses. If something is not stated, leave it out.
- headline: one line that starts with the actor and what they did, e.g. "DeepSeek released V4, a mixture-of-experts model".
- bullets: 3 to 4 short bullets, each a concrete technical fact (architecture, size, training method, results, license, how it differs from prior work). No hype words.
- why_it_matters: one sentence on the practical significance for someone building AI products.
- importance: integer 1-10 (10 = field-shifting release, 1 = minor).
- skip: true if the text is not about AI/ML, or is too thin to say anything concrete."""

SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "skip": {"type": "BOOLEAN"},
        "headline": {"type": "STRING"},
        "bullets": {"type": "ARRAY", "items": {"type": "STRING"}},
        "why_it_matters": {"type": "STRING"},
        "importance": {"type": "INTEGER"},
    },
    "required": ["skip", "headline", "bullets", "why_it_matters", "importance"],
}


class Gemini:
    def __init__(self, api_key: str):
        self.key = api_key
        self.models = [m for m in [os.getenv("GEMINI_MODEL"), *GEMINI_MODELS] if m]

    def summarize(self, it: Item, text: str, retries: int = 4):
        user = (f"TITLE: {it.title}\nKIND: {it.kind}\n"
                f"SEEN ON: {', '.join(it.sources)}\nSOURCE TEXT:\n{text or '(none available)'}")
        body = {
            "systemInstruction": {"parts": [{"text": SYSTEM_PROMPT}]},
            "contents": [{"role": "user", "parts": [{"text": user}]}],
            "generationConfig": {
                "responseMimeType": "application/json",
                "responseSchema": SCHEMA,
                "temperature": 0.2,
                "maxOutputTokens": 4096,
            },
        }
        for model in list(self.models):
            for attempt in range(retries):
                r = requests.post(GEMINI_URL.format(model=model), json=body, timeout=90,
                                  headers={"x-goog-api-key": self.key, "Content-Type": "application/json"})
                if r.status_code == 404:
                    log.warning("Gemini model %s unavailable, trying next", model)
                    break
                if r.status_code == 429 or r.status_code >= 500:
                    wait = min(60, 8 * 2 ** attempt)
                    log.warning("Gemini %s, retrying in %ss", r.status_code, wait)
                    time.sleep(wait)
                    continue
                r.raise_for_status()
                self.models = [model]          # stick with whatever works
                return self._parse(r.json())
        raise RuntimeError("No Gemini model worked; set GEMINI_MODEL to a model your key can use")

    @staticmethod
    def _parse(data):
        try:
            parts = data["candidates"][0]["content"]["parts"]
            text = "".join(p.get("text", "") for p in parts if not p.get("thought"))
            out = json.loads(text)
            out["bullets"] = [str(b) for b in out.get("bullets", [])][:4]
            return out
        except (KeyError, IndexError, ValueError, TypeError):
            return None


class OpenRouter:
    def __init__(self, api_key: str):
        self.key = api_key
        self.model = os.getenv("OPENROUTER_MODEL", OPENROUTER_DEFAULT_MODEL)

    def summarize(self, it: Item, text: str, retries: int = 4):
        user = (f"TITLE: {it.title}\nKIND: {it.kind}\n"
                f"SEEN ON: {', '.join(it.sources)}\nSOURCE TEXT:\n{text or '(none available)'}")
        system = (
            f"{SYSTEM_PROMPT}\n"
            'Respond ONLY with a valid JSON object matching: '
            '{"skip": bool, "headline": string, "bullets": [string, ...], "why_it_matters": string, "importance": int}'
        )
        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0.2,
        }
        headers = {
            "Authorization": f"Bearer {self.key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/ai-digest",
            "X-Title": "AI Digest",
        }
        for attempt in range(retries):
            try:
                r = requests.post(OPENROUTER_URL, json=body, headers=headers, timeout=90)
                if r.status_code == 429 or r.status_code >= 500:
                    wait = min(60, 8 * 2 ** attempt)
                    log.warning("OpenRouter %s, retrying in %ss", r.status_code, wait)
                    time.sleep(wait)
                    continue
                r.raise_for_status()
                return self._parse(r.json())
            except requests.RequestException as e:
                if attempt == retries - 1:
                    raise
                wait = min(60, 8 * 2 ** attempt)
                log.warning("OpenRouter error (%s), retrying in %ss", e, wait)
                time.sleep(wait)
        raise RuntimeError(f"OpenRouter model {self.model} failed after {retries} retries")

    @staticmethod
    def _parse(data):
        try:
            content = data["choices"][0]["message"].get("content", "")
            if not content:
                return None
            clean = content.strip()
            fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", clean)
            if fence_match:
                clean = fence_match.group(1).strip()
            else:
                brace_match = re.search(r"(\{[\s\S]*\})", clean)
                if brace_match:
                    clean = brace_match.group(1).strip()
            out = json.loads(clean)
            out["bullets"] = [str(b) for b in out.get("bullets", [])][:4]
            return out
        except (KeyError, IndexError, ValueError, TypeError):
            return None


def get_llm():
    provider = os.getenv("LLM_PROVIDER", "").lower()
    or_key = os.getenv("OPENROUTER_API_KEY")
    gemini_key = os.getenv("GEMINI_API_KEY")

    if provider == "openrouter":
        if not or_key:
            raise ValueError("LLM_PROVIDER is 'openrouter' but OPENROUTER_API_KEY is not set")
        return OpenRouter(or_key)
    if provider == "gemini":
        if not gemini_key:
            raise ValueError("LLM_PROVIDER is 'gemini' but GEMINI_API_KEY is not set")
        return Gemini(gemini_key)

    if or_key:
        return OpenRouter(or_key)
    if gemini_key:
        return Gemini(gemini_key)
    return None


# --------------------------------------------------------------------------
# Telegram
# --------------------------------------------------------------------------
def esc(s) -> str:
    return html.escape(str(s), quote=False)


def format_item(n: int, it: Item, s: dict) -> str:
    lines = [f"<b>{n}. {esc(s['headline'])}</b>", ""]
    lines += [f"• {esc(b)}" for b in s["bullets"]]
    if s.get("why_it_matters"):
        lines += ["", f"💡 <i>{esc(s['why_it_matters'])}</i>"]
    links = {"Read": it.url, **it.sources}
    seen_urls, parts = set(), []
    for label, url in links.items():
        if url in seen_urls:
            continue
        seen_urls.add(url)
        parts.append(f'<a href="{html.escape(url, quote=True)}">{esc(label)}</a>')
    lines += ["", "🔗 " + " · ".join(parts)]
    return "\n".join(lines)[:4000]


def tg_send(token: str, chat_id: str, text: str):
    for _ in range(3):
        r = requests.post(f"https://api.telegram.org/bot{token}/sendMessage", timeout=30, json={
            "chat_id": chat_id, "text": text, "parse_mode": "HTML",
            "disable_web_page_preview": True})
        if r.status_code == 429:
            time.sleep(r.json().get("parameters", {}).get("retry_after", 5) + 1)
            continue
        if not r.ok:
            raise RuntimeError(f"Telegram error {r.status_code}: {r.text}")
        return
    raise RuntimeError("Telegram rate-limited repeatedly")


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------
def collect():
    raw = []
    for f in FETCHERS:
        try:
            got = f()
            log.info("%-18s %d items", f.__name__, len(got))
            raw += got
        except Exception as e:
            log.warning("%s failed: %s", f.__name__, e)
    return merge(raw)


def build_digest(llm, seen):
    candidates = rank(collect(), seen)
    log.info("%d ranked candidates", len(candidates))
    kept, calls = [], 0
    for it in candidates:
        if len(kept) >= TOP_N or calls >= MAX_LLM_CALLS:
            break
        text = enrich(it)
        calls += 1
        try:
            s = llm.summarize(it, text)
        except requests.HTTPError as e:
            log.warning("summarize failed for %s: %s", it.title, e)
            s = None
        time.sleep(LLM_SLEEP_SECONDS)
        if not s or s.get("skip") or not s.get("bullets"):
            log.info("skipped: %s", it.title)
            continue
        kept.append((it, s))
    kept.sort(key=lambda p: -int(p[1].get("importance", 0)))   # stable: ties keep rank order
    return kept


def main():
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="print instead of sending; don't update seen.json")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    token, chat = os.getenv("TELEGRAM_BOT_TOKEN"), os.getenv("TELEGRAM_CHAT_ID")
    llm = get_llm()
    if not llm or (not args.dry_run and not (token and chat)):
        sys.exit("Missing env: OPENROUTER_API_KEY or GEMINI_API_KEY, and (unless --dry-run) TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID")

    def send(text):
        if args.dry_run:
            print("\n" + text + "\n" + "-" * 60)
        else:
            tg_send(token, chat, text)
            time.sleep(1.1)

    try:
        seen = load_seen()
        kept = build_digest(llm, seen)
        today = datetime.now(ZoneInfo(os.getenv("TZ_NAME", "Asia/Kolkata"))).strftime("%a %d %b")
        if not kept:
            send(f"📡 <b>AI Digest — {today}</b>\nNothing new worth your time today.")
            return
        send(f"📡 <b>AI Digest — {today}</b>\n{len(kept)} items · HN, arXiv, HF, Reddit, lab blogs")
        for n, (it, s) in enumerate(kept, 1):
            send(format_item(n, it, s))
        if not args.dry_run:
            save_seen(seen, [it for it, _ in kept])
    except Exception as e:
        log.exception("digest failed")
        if token and chat and not args.dry_run:
            try:
                tg_send(token, chat, f"⚠️ AI digest failed: {esc(str(e))[:500]}")
            except Exception:
                pass
        sys.exit(1)


if __name__ == "__main__":
    main()
