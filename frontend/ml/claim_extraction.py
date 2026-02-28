"""Sentence-split fallback for claim extraction."""

MAX_SENTENCES = 500


def sentence_split_claims(text: str) -> list:
    """Split text into sentence-level claims using period/question/exclamation delimiters.

    Returns a list of stripped, non-empty sentence strings.
    Bounded to MAX_SENTENCES iterations.
    """
    normalized = text.replace("? ", ".\n").replace("! ", ".\n")
    parts = normalized.split(".")

    results: list = []
    count = 0
    for i in range(len(parts)):
        if count >= MAX_SENTENCES:
            break
        segment = parts[i].strip()
        if len(segment) == 0:
            continue
        results.append(segment)
        count += 1

    return results
