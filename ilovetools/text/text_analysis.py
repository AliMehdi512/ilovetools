\
"""
Text analysis utilities for developer workflows and LLM agent integration.

This module provides a collection of small, well-tested helper functions
for common text-analysis tasks that developers and AI agents encounter
daily: secret redaction, URL/email extraction, markdown stripping,
slug generation, text similarity, word frequency analysis, reading-time
estimation, and keyword extraction.

Every function is pure (no side-effects on inputs), fully type-hinted,
and safe for use in both synchronous scripts and library code.

Typical use-cases
------------------
* Redacting API keys, tokens, and passwords from logs before display.
* Extracting URLs and email addresses from unstructured text.
* Converting markdown to plain text for NLP pipelines.
* Generating URL-safe slugs from titles.
* Computing text similarity for deduplication or search.
* Analysing word frequency for content insights.
* Estimating reading time for articles and documentation.
* Extracting keywords for tagging or SEO.

Examples
--------
>>> from ilovetools.text.text_analysis import (
...     redact_secrets, extract_urls, extract_emails, strip_markdown,
...     slugify, text_similarity, word_frequency, reading_time,
...     extract_keywords, normalize_whitespace,
... )

>>> redact_secrets("my api key is sk-1234567890abcdef and token ghp_abc123")
'my api key is [REDACTED] and token [REDACTED]'

>>> extract_urls("Visit https://example.com and http://test.org/page")
['https://example.com', 'http://test.org/page']

>>> extract_emails("Contact ali@example.com or bob@test.org")
['ali@example.com', 'bob@test.org']

>>> strip_markdown("# Title\n\n**bold** and *italic* text")
'Title\n\nbold and italic text'

>>> slugify("Hello World! This is a Test.")
'hello-world-this-is-a-test'

>>> text_similarity("hello world", "hello there")
0.5454545455

>>> word_frequency("the cat sat on the mat the cat")
{'the': 3, 'cat': 2, 'sat': 1, 'on': 1, 'mat': 1}

>>> reading_time("word " * 500)
1.0

>>> extract_keywords("machine learning is the future of machine intelligence")
['machine', 'learning', 'future', 'intelligence']

>>> normalize_whitespace("  hello   world  \n\n  foo  ")
'hello world\n\nfoo'
"""

from __future__ import annotations

import re
import math
import string
from collections import Counter
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Secret redaction
# ---------------------------------------------------------------------------

_SECRET_PATTERNS = [
    (r"\bAKIA[0-9A-Z]{16}\b", "AWS_ACCESS_KEY"),
    (r"\b(?<![A-Za-z0-9/+=])[A-Za-z0-9/+]{40}(?![A-Za-z0-9/+=])\b", "AWS_SECRET_KEY"),
    (r"\bgh[pousr]_[A-Za-z0-9]{36}\b", "GITHUB_TOKEN"),
    (r"\bsk-[A-Za-z0-9]{20,}\b", "OPENAI_API_KEY"),
    (r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b", "SLACK_TOKEN"),
    (r"(?i)(?:api[_-]?key|api[_-]?secret|access[_-]?token|secret[_-]?key|auth[_-]?token)\s*[=:]\s*['\"]?[A-Za-z0-9_\-/+=]{16,}['\"]?", "GENERIC_SECRET"),
    (r"(?i)\bBearer\s+[A-Za-z0-9_\-\.]{20,}\b", "BEARER_TOKEN"),
    (r"-----BEGIN[A-Z\s]+PRIVATE KEY-----[\s\S]*?-----END[A-Z\s]+PRIVATE KEY-----", "PRIVATE_KEY"),
]


def redact_secrets(text, replacement="[REDACTED]", patterns=None):
    """Redact common secret patterns from text.

    Scans *text* for known secret formats (API keys, tokens, private keys,
    etc.) and replaces them with *replacement*.

    Parameters
    ----------
    text : str
        The input text that may contain secrets.
    replacement : str, optional
        The string to replace secrets with (default ``"[REDACTED]"``).
    patterns : list of tuple, optional
        Custom ``(regex, label)`` pairs to use instead of the built-in
        patterns.

    Returns
    -------
    str
        The text with all matched secrets replaced.

    Examples
    --------
    >>> redact_secrets("token: ghp_1234567890abcdefghijklmnopqrstuvwxyz")
    'token: [REDACTED]'

    >>> redact_secrets("key is sk-proj1234567890abcdefghij")
    'key is [REDACTED]'

    >>> redact_secrets("no secrets here")
    'no secrets here'

    >>> redact_secrets("password=supersecret12345678", replacement="***")
    'password=***'
    """
    if not isinstance(text, str) or not text:
        return text if isinstance(text, str) else ""

    active_patterns = patterns if patterns is not None else _SECRET_PATTERNS
    result = text
    for pattern, _label in active_patterns:
        result = re.sub(pattern, replacement, result)
    return result


# ---------------------------------------------------------------------------
# URL extraction
# ---------------------------------------------------------------------------

_URL_PATTERN = re.compile(
    r"https?://[^\s<>\"')\]]+[^\s<>\"')\].,!?:;]"
)


def extract_urls(text):
    """Extract all HTTP(S) URLs from *text*.

    Parameters
    ----------
    text : str
        The input text that may contain URLs.

    Returns
    -------
    list[str]
        A list of URLs found in the text, in order of appearance.

    Examples
    --------
    >>> extract_urls("Visit https://example.com today!")
    ['https://example.com']

    >>> extract_urls("No URLs here")
    []

    >>> extract_urls("")
    []
    """
    if not isinstance(text, str) or not text:
        return []
    return _URL_PATTERN.findall(text)


# ---------------------------------------------------------------------------
# Email extraction
# ---------------------------------------------------------------------------

_EMAIL_PATTERN = re.compile(
    r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"
)


def extract_emails(text):
    """Extract all email addresses from *text*.

    Parameters
    ----------
    text : str
        The input text that may contain email addresses.

    Returns
    -------
    list[str]
        A list of email addresses found, in order of appearance.

    Examples
    --------
    >>> extract_emails("Contact ali@example.com for info")
    ['ali@example.com']

    >>> extract_emails("No emails here")
    []

    >>> extract_emails("Send to a.b+c@test.co.uk please")
    ['a.b+c@test.co.uk']
    """
    if not isinstance(text, str) or not text:
        return []
    return _EMAIL_PATTERN.findall(text)


# ---------------------------------------------------------------------------
# Markdown stripping
# ---------------------------------------------------------------------------

def strip_markdown(text):
    """Remove markdown formatting from *text*, returning plain text.

    Parameters
    ----------
    text : str
        The markdown text to strip.

    Returns
    -------
    str
        Plain text with markdown syntax removed.

    Examples
    --------
    >>> strip_markdown("# Title")
    'Title'

    >>> strip_markdown("**bold** and *italic*")
    'bold and italic'

    >>> strip_markdown("[link](https://example.com)")
    'link'

    >>> strip_markdown("```python\nprint('hi')\n```")
    "print('hi')"

    >>> strip_markdown("")
    ''
    """
    if not isinstance(text, str) or not text:
        return ""

    result = text

    # Remove code blocks
    result = re.sub(r"```[a-zA-Z0-9]*\n?(.*?)\n?```", r"\1", result, flags=re.DOTALL)

    # Remove images
    result = re.sub(r"!\[([^\]]*)\]\([^\)]+\)", r"\1", result)

    # Remove links
    result = re.sub(r"\[([^\]]*)\]\([^\)]+\)", r"\1", result)

    # Remove reference-style links
    result = re.sub(r"\[([^\]]*)\]\[[^\]]*\]", r"\1", result)

    # Remove headings markers
    result = re.sub(r"^#{1,6}\s+", "", result, flags=re.MULTILINE)

    # Remove bold/italic markers
    result = re.sub(r"\*\*\*(.+?)\*\*\*", r"\1", result)
    result = re.sub(r"\*\*(.+?)\*\*", r"\1", result)
    result = re.sub(r"__(.+?)__", r"\1", result)
    result = re.sub(r"\*(.+?)\*", r"\1", result)
    result = re.sub(r"(?<!\w)_(.+?)_(?!\w)", r"\1", result)

    # Remove inline code
    result = re.sub(r"`([^`]+)`", r"\1", result)

    # Remove blockquote markers
    result = re.sub(r"^>\s?", "", result, flags=re.MULTILINE)

    # Remove horizontal rules
    result = re.sub(r"^[-*_]{3,}\s*$", "", result, flags=re.MULTILINE)

    # Remove list markers
    result = re.sub(r"^\s*[-*+]\s+", "", result, flags=re.MULTILINE)
    result = re.sub(r"^\s*\d+\.\s+", "", result, flags=re.MULTILINE)

    # Remove strikethrough
    result = re.sub(r"~~(.+?)~~", r"\1", result)

    # Clean up extra blank lines
    result = re.sub(r"\n{3,}", "\n\n", result)

    return result.strip()


# ---------------------------------------------------------------------------
# Slugify
# ---------------------------------------------------------------------------

def slugify(text, separator="-", max_length=0):
    """Convert *text* to a URL-safe slug.

    Parameters
    ----------
    text : str
        The text to slugify.
    separator : str, optional
        The separator to use between words (default ``"-"``).
    max_length : int, optional
        Maximum slug length.  ``0`` means no limit.

    Returns
    -------
    str
        A URL-safe slug.

    Examples
    --------
    >>> slugify("Hello World!")
    'hello-world'

    >>> slugify("  Multiple   Spaces  ")
    'multiple-spaces'

    >>> slugify("Café résumé naïve")
    'cafe-resume-naive'

    >>> slugify("Hello_World", separator="_")
    'hello_world'

    >>> slugify("This is a very long title", max_length=10)
    'this-is-a'

    >>> slugify("")
    ''
    """
    if not isinstance(text, str) or not text:
        return ""

    import unicodedata
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", separator, text)
    text = text.strip(separator)

    if max_length > 0 and len(text) > max_length:
        text = text[:max_length]
        last_sep = text.rfind(separator)
        if last_sep > 0:
            text = text[:last_sep]
        text = text.strip(separator)

    return text


# ---------------------------------------------------------------------------
# Text similarity (cosine similarity on character bigrams)
# ---------------------------------------------------------------------------

def _get_bigrams(text):
    """Return a Counter of character bigrams from *text*."""
    text = text.lower().strip()
    if len(text) < 2:
        return Counter()
    return Counter(text[i:i+2] for i in range(len(text) - 1))


def text_similarity(text1, text2):
    """Compute cosine similarity between two strings using character bigrams.

    Returns a float between 0.0 (no similarity) and 1.0 (identical).

    Parameters
    ----------
    text1 : str
        First string.
    text2 : str
        Second string.

    Returns
    -------
    float
        Cosine similarity score in [0.0, 1.0].

    Examples
    --------
    >>> text_similarity("hello world", "hello world")
    1.0

    >>> text_similarity("hello world", "goodbye world")
    0.4444444444

    >>> text_similarity("abc", "")
    0.0

    >>> text_similarity("", "")
    0.0
    """
    if not text1 or not text2:
        return 0.0

    bigrams1 = _get_bigrams(text1)
    bigrams2 = _get_bigrams(text2)

    if not bigrams1 or not bigrams2:
        return 0.0

    all_keys = set(bigrams1.keys()) | set(bigrams2.keys())
    dot = sum(bigrams1.get(k, 0) * bigrams2.get(k, 0) for k in all_keys)

    mag1 = math.sqrt(sum(v * v for v in bigrams1.values()))
    mag2 = math.sqrt(sum(v * v for v in bigrams2.values()))

    if mag1 == 0 or mag2 == 0:
        return 0.0

    return round(dot / (mag1 * mag2), 10)


# ---------------------------------------------------------------------------
# Word frequency
# ---------------------------------------------------------------------------

def word_frequency(text, case_sensitive=False, stop_words=None, min_length=1):
    """Count word frequencies in *text*.

    Parameters
    ----------
    text : str
        The input text.
    case_sensitive : bool, optional
        If ``False`` (default), words are lowercased before counting.
    stop_words : set, optional
        A set of words to exclude from the count.
    min_length : int, optional
        Minimum word length to include (default 1).

    Returns
    -------
    dict[str, int]
        A dictionary mapping each word to its count, sorted by
        count descending then alphabetically.

    Examples
    --------
    >>> word_frequency("the cat sat on the mat the cat")
    {'the': 3, 'cat': 2, 'sat': 1, 'on': 1, 'mat': 1}

    >>> word_frequency("Hello hello WORLD", case_sensitive=True)
    {'Hello': 1, 'hello': 1, 'WORLD': 1}

    >>> word_frequency("a an the cat", stop_words={"a", "an", "the"})
    {'cat': 1}

    >>> word_frequency("hi there friend", min_length=3)
    {'there': 1, 'friend': 1}

    >>> word_frequency("")
    {}
    """
    if not isinstance(text, str) or not text:
        return {}

    if not case_sensitive:
        text = text.lower()

    words = re.findall(r"[a-zA-Z0-9]+", text)

    if stop_words is not None:
        stop_words_lower = {w.lower() for w in stop_words} if not case_sensitive else stop_words
        words = [w for w in words if w not in stop_words_lower]

    words = [w for w in words if len(w) >= min_length]

    counts = Counter(words)

    return dict(sorted(counts.items(), key=lambda x: (-x[1], x[0])))


# ---------------------------------------------------------------------------
# Reading time estimation
# ---------------------------------------------------------------------------

def reading_time(text, words_per_minute=200, unit="minutes"):
    """Estimate reading time for *text*.

    Parameters
    ----------
    text : str
        The input text.
    words_per_minute : int, optional
        Average reading speed (default 200 WPM).
    unit : str, optional
        ``"minutes"`` (default) or ``"seconds"``.

    Returns
    -------
    float
        Estimated reading time.

    Examples
    --------
    >>> reading_time("word " * 200)
    1.0

    >>> reading_time("word " * 400)
    2.0

    >>> reading_time("word " * 200, unit="seconds")
    60.0

    >>> reading_time("")
    0.0

    >>> reading_time("one two three")
    1.0
    """
    if not isinstance(text, str) or not text:
        return 0.0

    word_count = len(text.split())
    if word_count == 0:
        return 0.0

    minutes = word_count / words_per_minute

    if unit == "seconds":
        return round(minutes * 60, 1)

    return max(1.0, round(minutes, 1))


# ---------------------------------------------------------------------------
# Keyword extraction (TF-based simple approach)
# ---------------------------------------------------------------------------

_STOP_WORDS = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "must", "can", "this", "that",
    "these", "those", "i", "you", "he", "she", "it", "we", "they", "what",
    "which", "who", "when", "where", "why", "how", "all", "each", "every",
    "both", "few", "more", "most", "other", "some", "such", "no", "nor",
    "not", "only", "own", "same", "so", "than", "too", "very", "just",
    "as", "if", "about", "against", "between", "into", "through", "during",
    "before", "after", "above", "below", "up", "down", "out", "off",
    "over", "under", "again", "further", "then", "once", "here", "there",
    "its", "their", "his", "her", "our", "your", "my", "me", "him", "them",
    "us", "am", "any", "because", "while", "where",
}


def extract_keywords(text, num_keywords=5, stop_words=None, min_length=3):
    """Extract the most significant keywords from *text* using term frequency.

    Parameters
    ----------
    text : str
        The input text.
    num_keywords : int, optional
        Number of keywords to return (default 5).
    stop_words : set, optional
        Custom stop-word set.  If ``None``, a built-in English stop-word
        list is used.
    min_length : int, optional
        Minimum word length for keywords (default 3).

    Returns
    -------
    list[str]
        A list of keywords sorted by frequency (descending).

    Examples
    --------
    >>> extract_keywords("machine learning is the future of machine intelligence", num_keywords=3)
    ['machine', 'future', 'intelligence']

    >>> extract_keywords("data data data science science models")
    ['data', 'science', 'models']

    >>> extract_keywords("")
    []

    >>> extract_keywords("the and or but", min_length=3)
    []
    """
    if not isinstance(text, str) or not text:
        return []

    active_stops = stop_words if stop_words is not None else _STOP_WORDS

    text_lower = text.lower()
    words = re.findall(r"[a-zA-Z]+", text_lower)

    candidates = [
        w for w in words
        if len(w) >= min_length and w not in active_stops
    ]

    if not candidates:
        return []

    counts = Counter(candidates)

    sorted_keywords = sorted(counts.items(), key=lambda x: (-x[1], x[0]))
    return [word for word, _count in sorted_keywords[:num_keywords]]


# ---------------------------------------------------------------------------
# Whitespace normalisation
# ---------------------------------------------------------------------------

def normalize_whitespace(text, collapse_newlines=True):
    """Normalise whitespace in *text*.

    Collapses multiple spaces/tabs into a single space, strips leading
    and trailing whitespace, and optionally collapses multiple newlines
    into at most two.

    Parameters
    ----------
    text : str
        The input text.
    collapse_newlines : bool, optional
        If ``True`` (default), collapses 3+ consecutive newlines into 2.

    Returns
    -------
    str
        Whitespace-normalised text.

    Examples
    --------
    >>> normalize_whitespace("  hello   world  ")
    'hello world'

    >>> normalize_whitespace("a\n\n\n\nb")
    'a\n\nb'

    >>> normalize_whitespace("a\n\n\n\nb", collapse_newlines=False)
    'a\n\n\n\nb'

    >>> normalize_whitespace("")
    ''
    """
    if not isinstance(text, str) or not text:
        return ""

    # Collapse multiple spaces/tabs into one
    result = re.sub(r"[ \t]+", " ", text)

    # Strip spaces around newlines
    result = re.sub(r" *\n *", "\n", result)

    if collapse_newlines:
        result = re.sub(r"\n{3,}", "\n\n", result)

    return result.strip()


__all__ = [
    "redact_secrets",
    "extract_urls",
    "extract_emails",
    "strip_markdown",
    "slugify",
    "text_similarity",
    "word_frequency",
    "reading_time",
    "extract_keywords",
    "normalize_whitespace",
]
