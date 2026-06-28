from __future__ import annotations

import json
import re
from collections import Counter

from bs4 import BeautifulSoup

from diagnostics import log


def _fallback_extract(html):
    # ponytail: raw get_text(). trafilatura covers 95%+ of pages, this is the last resort.
    # ceiling: naive get_text() chunking. upgrade: readability-lxml when fallback rate exceeds 10% of fetches.
    return BeautifulSoup(html, "html.parser").get_text("\n", strip=True)


def _extract_publish_date(html):
    from datetime import datetime

    soup = BeautifulSoup(html, "html.parser")
    date_str = None

    meta_selectors = [
        ("meta", {"property": "article:published_time"}),
        ("meta", {"name": "publication_date"}),
        ("meta", {"name": "publishdate"}),
        ("meta", {"property": "og:published_time"}),
        ("meta", {"name": "date"}),
        ("meta", {"itemprop": "datePublished"}),
        ("time", {"datetime": True}),
    ]

    for tag, attrs in meta_selectors:
        elem = soup.find(tag, attrs)
        if elem:
            date_str = elem.get("content") or elem.get("datetime")
            if date_str:
                break

    if not date_str:
        for script in soup.find_all("script", {"type": "application/ld+json"}):
            try:
                data = json.loads(script.string)
                if isinstance(data, dict):
                    date_str = data.get("datePublished") or data.get("dateCreated")
                    if date_str:
                        break
            except Exception:
                pass

    if date_str:
        try:
            if "T" in date_str or "-" in date_str:
                date_str = date_str.split("+")[0].split("Z")[0]
                return datetime.fromisoformat(date_str.replace("T", " ")[:19])
        except Exception:
            pass

    return None


def extract_content(html, url=None):
    import core

    if core.HAS_TRAFILATURA:
        text = core.trafilatura.extract(
            html,
            url=url,
            include_comments=False,
            include_tables=True,
            no_fallback=False,
            favor_recall=True,
            output_format="markdown",
        )
        if text and len(text) > 100:
            return text

    return _fallback_extract(html)


def chunk_text(text, chunk_size=None, overlap=None, lang="en"):
    import core

    if chunk_size is None:
        chunk_size = core.CHUNK_SIZE
    if overlap is None:
        overlap = core.CHUNK_OVERLAP
    if not text:
        return []

    paragraphs = re.split(r"\n{2,}", text.strip())
    chunks = []
    current = ""

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        if current and len(current) + len(para) + 2 > chunk_size:
            chunks.append(current.strip())
            if overlap > 0 and len(current) > overlap:
                current = current[-overlap:] + "\n\n" + para
            else:
                current = para
        else:
            current = current + "\n\n" + para if current else para

        while len(current) > chunk_size * 1.5:
            split_pos = chunk_size
            for delim in [". ", "! ", "? ", ".\n", ";\n", "\n"]:
                pos = current.rfind(delim, 0, chunk_size + 50)
                if pos > chunk_size * 0.3:
                    split_pos = pos + len(delim)
                    break

            chunk_part = current[:split_pos].strip()
            if chunk_part:
                chunks.append(chunk_part)

            remainder = current[split_pos:].strip()
            if overlap > 0 and len(chunk_part) > overlap:
                current = chunk_part[-overlap:] + " " + remainder
            else:
                current = remainder

    if current.strip():
        chunks.append(current.strip())

    return [chunk for chunk in chunks if len(chunk) > 40]


def _is_incomplete_chunk(text):
    """Only reject clear mid-sentence fragments."""
    if not text or len(text) < 30:
        return True

    text_stripped = text.strip()
    # starts mid-sentence: lowercase letter
    if len(text_stripped) < 100 and text_stripped[0].islower():
        return True
    # starts with punctuation (continuation)
    if text_stripped[0] in ".,-—–…":
        return True
    # trailing dash = cut off mid-word
    if text_stripped.endswith("-") or text_stripped.endswith("—"):
        return True
    return False



def _is_low_quality_chunk(text):
    """One-pass quality check. ponytail: merged _remove_garbage_lines patterns."""
    if not text or len(text) < 30:
        return True

    words = text.split()
    if not words:
        return True
    if len(words) / len(text) < 0.08:
        return True

    text_lower = text.lower()
    # merged garbage + boilerplate patterns
    garbage = [
        "подписаться", "подпис", "subscribe", "sign up", "telegram", "whatsapp",
        "вконтакте", "следите за нами", "follow us", "поделиться",
        "share on", "tweet", "facebook", "twitter", "комментар", "comment",
        "оставьте отзыв", "читайте также", "read also", "related articles",
        "рекомендуем", "recommended", "похожие статьи", "subscribe now",
        "email updates", "daily digest",
        "cookie policy", "privacy policy", "terms of service", "all rights reserved", "© 20",
        "copyright ©", "sign up for", "follow us on",
        "share this article", "advertisement", "sponsored content", "click here to",
        "read more »", "loading...", "please wait", "javascript is disabled",
        "enable javascript", "accept cookies", "we use cookies",
        "in your inbox", "sign up for our", "get the latest",
    ]
    if sum(1 for p in garbage if p in text_lower) >= 2:
        return True

    word_counts = Counter(w.lower() for w in words if len(w) > 3)
    if word_counts and max(word_counts.values()) > len(words) * 0.3:
        return True

    long_words = [w for w in words if len(w) > 4]
    if len(long_words) / len(words) < 0.2:
        return True
    return False


def is_content_page(text, query=None, lang="en", min_avg_sentence_len=25, min_sentences=3):
    """Return True when extracted text has enough sentence or line-level substance."""
    if not text or len(text) < 100:
        return False

    sentences = [s.strip() for s in re.split(r"[.!?]\s+", text) if len(s.strip()) > 20]
    lines = [l.strip() for l in text.split("\n") if len(l.strip()) > 25]
    if len(sentences) < min_sentences and len(lines) < min_sentences:
        return False
    all_parts = sentences + lines
    avg_len = sum(len(s) for s in all_parts) / len(all_parts)
    return avg_len >= min_avg_sentence_len or len(all_parts) >= 8  # enough distinct items = real content


def filter_low_quality_chunks(chunks):
    filtered = []
    for chunk in chunks:
        if not _is_incomplete_chunk(chunk) and not _is_low_quality_chunk(chunk):
            filtered.append(chunk)

    removed = len(chunks) - len(filtered)
    if removed:
        log.info("[FILTER] Removed %s low-quality/incomplete chunks", removed)
    return filtered
