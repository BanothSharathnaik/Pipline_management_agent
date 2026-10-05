from __future__ import annotations

import re
import unicodedata

# Deliberately small. Negations (not, no, cannot) are NOT stopwords:
# "does not have SELECT" and "does have SELECT" must stay different.
STOPWORDS = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been",
    "of", "on", "in", "to", "and", "or", "for", "at", "by", "with",
    "it", "its", "this", "that", "as", "from",
    "do", "does", "did", "has", "have", "had",
    "there", "these", "those", "which", "what", "why", "how", "when",
    "i", "me", "my", "you", "your", "we", "our",
})

_LOWER_TO_UPPER = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")    # OutOf -> Out Of
_ACRONYM_TO_WORD = re.compile(r"(?<=[A-Z])(?=[A-Z][a-z])")  # SQLException -> SQL Exception
_TOKEN = re.compile(r"[^\W_]+")  # runs of letters/digits; underscore is a separator


def normalize_text(text: str) -> str:
    """NFKC, split camelCase, case-fold. For MATCHING only; never store the result."""
    if not isinstance(text, str):
        raise TypeError(f"text must be a str, got {type(text).__name__}")
    text = unicodedata.normalize("NFKC", text)
    text = _LOWER_TO_UPPER.sub(" ", text)
    text = _ACRONYM_TO_WORD.sub(" ", text)
    return text.casefold()


def tokenize(text: str, remove_stopwords: bool = True) -> list[str]:
    """Text -> list of tokens. Duplicates are kept (term frequency matters later)."""
    tokens = _TOKEN.findall(normalize_text(text))
    if remove_stopwords:
        tokens = [t for t in tokens if t not in STOPWORDS]
    return tokens