from __future__ import annotations

import json
import re
from collections import Counter

from bs4 import BeautifulSoup

from .diagnostics import log


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
    from . import core

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


# ── Tables ─────────────────────────────────────────────────────────────────
#
# A table is a grid, not prose: its numbers mean nothing without the column
# names, and a split in the middle of a row invents a value that was never on
# the page. The prose path did both — it cut on the nearest newline and left
# every part after the first headerless — so tables are routed around it and
# split on row boundaries only, with the header repeated in each part.

_TABLE_SEPARATOR_RE = re.compile(r"^[\s|:+-]+$")


_COLUMN_GAP_RE = re.compile(r"\s{2,}")


def _looks_like_column_gaps(lines):
    """A grid drawn with spaces instead of pipes: fixed-width columns.

    Kept deliberately strict — prose wrapped at a fixed width and indented code
    both survive a naive gap test — so a block qualifies only when nearly every
    line splits into the same number of fields and the body carries digits,
    which is the case the figures are lost in.
    """
    fields = [_COLUMN_GAP_RE.split(line.strip()) for line in lines]
    counts = [len(row) for row in fields]
    if min(counts) < 2:
        return False
    common = max(set(counts), key=counts.count)
    if common < 2 or counts.count(common) < len(counts) * 0.8:
        return False
    numeric = sum(1 for row in fields if any(any(ch.isdigit() for ch in cell) for cell in row[1:]))
    return numeric >= len(fields) * 0.5


def _table_rows(block):
    """Return the block's rows when it reads as a table, else None."""
    lines = [line for line in block.splitlines() if line.strip()]
    if len(lines) < 3:
        return None

    delimited = [line for line in lines if "|" in line or "\t" in line]
    if len(delimited) >= max(3, len(lines) * 0.6):
        counts = [line.count("|") or line.count("\t") for line in delimited]
        # A real grid keeps its column count; a paragraph that happens to
        # contain a pipe does not.
        if len(set(counts)) <= max(2, len(counts) // 4):
            return lines
        return None

    return lines if _looks_like_column_gaps(lines) else None


def _table_header(rows):
    """The leading label rows: the column names plus a --- separator if present."""
    header = rows[:1]
    if len(rows) > 1 and _TABLE_SEPARATOR_RE.match(rows[1]):
        header = rows[:2]
    return header


def _chunk_table(block, chunk_size):
    """Split a table on row boundaries, repeating the header in every part."""
    rows = _table_rows(block)
    if rows is None:
        return None

    header = _table_header(rows)
    body = rows[len(header):]
    if not body:
        return [block.strip()]

    header_text = "\n".join(header)
    parts = []
    current = list(header)
    size = len(header_text)

    for row in body:
        # Never break inside a row: an oversized single row goes out whole.
        if len(current) > len(header) and size + len(row) + 1 > chunk_size:
            parts.append("\n".join(current))
            current = list(header)
            size = len(header_text)
        current.append(row)
        size += len(row) + 1

    if len(current) > len(header):
        parts.append("\n".join(current))
    return parts


def chunk_text(text, chunk_size=None, overlap=None, lang="en"):
    from . import core

    if chunk_size is None:
        chunk_size = core.CHUNK_SIZE
    if overlap is None:
        overlap = core.CHUNK_OVERLAP
    if not text:
        return []

    # A table is split into whole rows under a repeated header, so the prose
    # budget buys far fewer rows per part than it buys sentences. Give a grid a
    # budget of its own: at 600 characters a modest table still arrived in
    # pieces, each a fragment of one answer.
    table_chunk_size = max(chunk_size, getattr(core, "TABLE_CHUNK_SIZE", chunk_size))

    paragraphs = re.split(r"\n{2,}", text.strip())
    chunks = []
    current = ""

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        table_parts = _chunk_table(para, table_chunk_size)
        if table_parts is not None:
            # Close the prose chunk in progress rather than glue a grid onto it,
            # then emit the table's own parts untouched by the prose splitter.
            if current.strip():
                chunks.append(current.strip())
                current = ""
            chunks.extend(table_parts)
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
    # A table has no sentences to be in the middle of, and its first cell is
    # routinely lowercase or a digit.
    if _table_rows(text_stripped) is not None:
        return False
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



# Boilerplate a page wraps around its content; two or more hits mean the chunk
# is navigation furniture rather than substance.
_GARBAGE_PATTERNS = [
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
    # The prose heuristics below measure vocabulary: share of long words, and
    # repetition of any one token. A numeric table fails both by construction —
    # its cells are digits and its column names repeat in every part — and used
    # to be dropped in full, taking the figures with it. Only the boilerplate
    # patterns apply to a grid.
    if _table_rows(text.strip()) is not None:
        return sum(1 for p in _GARBAGE_PATTERNS if p in text_lower) >= 2
    if sum(1 for p in _GARBAGE_PATTERNS if p in text_lower) >= 2:
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
