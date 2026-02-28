"""Deterministic safe-answer builder for the backend.

Consumes ML pipeline outputs (clusters + cluster_scores) and produces a
human-readable summary with supported/rejected cluster ID lists.

This module has zero ML imports — it is pure backend logic.
"""

from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Internal dataclasses
# ---------------------------------------------------------------------------


@dataclass
class _ClusterData:
    cluster_id: str
    representative_text: str


@dataclass
class _ScoreData:
    cluster_id: str
    trust_score: int
    verdict: str


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_VERDICT_ORDER = {"SAFE": 0, "CAUTION": 1, "REJECT": 2}
_ALLOWED_VERDICTS = set(_VERDICT_ORDER)
_MAX_CLUSTERS = 1000  # Aligned with ml/modal_app.py
_MAX_INPUT = 10000    # Upper bound for parse/dedup loops

_ALLOWED_TRANSITION_PREFIXES = [
    "In addition,", "Additionally,", "However,", "Because of this,",
    "As a result,", "Meanwhile,", "Overall,", "Still,",
]

# Prefixes valid at sentence index 0 (openers only, no contrastive)
_OPENER_PREFIXES = {""}

_FALLBACK_TEXT = (
    "There is insufficient verified support to provide a reliable summary."
)


# ---------------------------------------------------------------------------
# 1. _try_parse_cluster
# ---------------------------------------------------------------------------


def _try_parse_cluster(raw):
    """Return _ClusterData from a raw dict, or None on any problem."""
    try:
        if not isinstance(raw, dict):
            return None
        cid = raw.get("cluster_id")
        text = raw.get("representative_text")
        if not isinstance(cid, str) or not cid:
            return None
        if not isinstance(text, str) or not text:
            return None
        return _ClusterData(cluster_id=cid, representative_text=text)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# 2. _try_parse_score
# ---------------------------------------------------------------------------


def _try_parse_score(raw):
    """Return _ScoreData from a raw dict, or None on any problem."""
    try:
        if not isinstance(raw, dict):
            return None
        cid = raw.get("cluster_id")
        ts = raw.get("trust_score")
        verdict = raw.get("verdict")
        if not isinstance(cid, str) or not cid:
            return None
        if not isinstance(ts, int) or isinstance(ts, bool):
            return None
        if ts < 0 or ts > 100:
            return None
        if not isinstance(verdict, str) or verdict not in _ALLOWED_VERDICTS:
            return None
        return _ScoreData(cluster_id=cid, trust_score=ts, verdict=verdict)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# 3. _dedup_clusters
# ---------------------------------------------------------------------------


def _dedup_clusters(parsed, warnings):
    """Deduplicate clusters by cluster_id.

    Tie-break: longest representative_text, then lex-smallest text.
    Returns dict[str, _ClusterData].
    """
    index = {}
    for i in range(min(len(parsed), _MAX_INPUT)):
        item = parsed[i]
        cid = item.cluster_id
        if cid in index:
            warnings.append(f"Duplicate cluster_id '{cid}' in clusters")
            existing = index[cid]
            # Keep longest text; on equal length keep lex-smallest
            if len(item.representative_text) > len(existing.representative_text):
                index[cid] = item
            elif (len(item.representative_text) == len(existing.representative_text)
                  and item.representative_text < existing.representative_text):
                index[cid] = item
        else:
            index[cid] = item
    return index


# ---------------------------------------------------------------------------
# 4. _dedup_scores
# ---------------------------------------------------------------------------


def _dedup_scores(parsed, warnings):
    """Deduplicate scores by cluster_id.

    Tie-break: highest trust_score, then best verdict rank,
    then lex-smallest cluster_id (self-documenting).
    Returns list[_ScoreData].
    """
    index = {}
    for i in range(min(len(parsed), _MAX_INPUT)):
        item = parsed[i]
        cid = item.cluster_id
        if cid in index:
            warnings.append(f"Duplicate cluster_id '{cid}' in scores")
            existing = index[cid]
            new_key = (-item.trust_score, _verdict_rank(item.verdict), cid)
            old_key = (-existing.trust_score, _verdict_rank(existing.verdict), cid)
            if new_key < old_key:
                index[cid] = item
        else:
            index[cid] = item
    return list(index.values())


# ---------------------------------------------------------------------------
# 5. _verdict_rank
# ---------------------------------------------------------------------------


def _verdict_rank(verdict):
    """Return sort rank for a verdict string. Unknown defaults to 2."""
    return _VERDICT_ORDER.get(verdict, 2)


# ---------------------------------------------------------------------------
# 6. _sorted_scores
# ---------------------------------------------------------------------------


def _sorted_scores(cluster_scores):
    """Sort scores by (verdict_rank, -trust_score, cluster_id)."""
    return sorted(
        cluster_scores,
        key=lambda s: (_verdict_rank(s.verdict), -s.trust_score, s.cluster_id),
    )


# ---------------------------------------------------------------------------
# 7. _validate_prefix
# ---------------------------------------------------------------------------


def _validate_prefix(prefix, index):
    """Return True if the prefix is acceptable at the given sentence index."""
    if prefix == "":
        return True
    if prefix not in _ALLOWED_TRANSITION_PREFIXES:
        return False
    if index == 0 and prefix not in _OPENER_PREFIXES:
        return False
    return True


# ---------------------------------------------------------------------------
# 8. _apply_transition_rewrite
# ---------------------------------------------------------------------------


def _apply_transition_rewrite(sentences, rewrite_client):
    """Apply rewrite_client prefixes to sentences with per-sentence fallback."""
    try:
        prefixes = rewrite_client(sentences)
        if len(prefixes) != len(sentences):
            return list(sentences)
        result = []
        for i in range(len(sentences)):
            prefix = prefixes[i]
            if _validate_prefix(prefix, i) and prefix != "":
                result.append(f"{prefix} {sentences[i]}")
            else:
                result.append(sentences[i])
        return result
    except Exception:
        return list(sentences)


# ---------------------------------------------------------------------------
# 9. _base_sentences
# ---------------------------------------------------------------------------


def _base_sentences(sorted_scores, cluster_index):
    """Build transition-free factual sentences from sorted scores."""
    safe_scores = []
    caution_scores = []
    reject_scores = []

    for i in range(len(sorted_scores)):
        if i >= _MAX_CLUSTERS:
            break
        s = sorted_scores[i]
        if s.cluster_id not in cluster_index:
            continue
        verdict = s.verdict
        if verdict == "SAFE":
            safe_scores.append(s)
        elif verdict == "CAUTION":
            caution_scores.append(s)
        else:
            reject_scores.append(s)

    sentences = []

    if len(safe_scores) > 0:
        # Lead with top SAFE
        lead = cluster_index[safe_scores[0].cluster_id].representative_text
        sentences.append(f"Verified sources support the following: {lead}.")
        # Up to 2 more SAFE support
        for j in range(1, min(3, len(safe_scores))):
            text = cluster_index[safe_scores[j].cluster_id].representative_text
            sentences.append(f"{text}.")
        # 1 CAUTION caveat
        if len(caution_scores) > 0:
            text = cluster_index[caution_scores[0].cluster_id].representative_text
            sentences.append(
                f"It is worth noting that {text}, though this claim has"
                " limited support."
            )
        # 1 REJECT conflict
        if len(reject_scores) > 0:
            text = cluster_index[reject_scores[0].cluster_id].representative_text
            sentences.append(f"The claim that {text} was not verified.")
    elif len(caution_scores) > 0:
        # Lead with uncertainty phrasing
        lead = cluster_index[caution_scores[0].cluster_id].representative_text
        sentences.append(
            "The available evidence is uncertain."
            f" One unconfirmed claim suggests: {lead}."
        )
        # 2nd CAUTION
        if len(caution_scores) > 1:
            text = cluster_index[caution_scores[1].cluster_id].representative_text
            sentences.append(f"{text}, though support is limited.")
    else:
        sentences.append(_FALLBACK_TEXT)

    return sentences


# ---------------------------------------------------------------------------
# 10. build_safe_answer (public entry)
# ---------------------------------------------------------------------------


def build_safe_answer(clusters, cluster_scores, *, rewrite_client=None):
    """Build a safe-answer summary from ML pipeline outputs.

    Never raises. Returns (safe_answer_dict, warnings_list).
    """
    try:
        warnings = []

        # -- Parse clusters --
        parsed_clusters = []
        for i in range(min(len(clusters), _MAX_INPUT)):
            result = _try_parse_cluster(clusters[i])
            if result is None:
                warnings.append(
                    f"Skipped malformed cluster at index {i}"
                )
            else:
                parsed_clusters.append(result)

        # -- Parse scores --
        parsed_scores = []
        for i in range(min(len(cluster_scores), _MAX_INPUT)):
            result = _try_parse_score(cluster_scores[i])
            if result is None:
                warnings.append(
                    f"Skipped malformed score at index {i}"
                )
            else:
                parsed_scores.append(result)

        # -- Dedup --
        cluster_index = _dedup_clusters(parsed_clusters, warnings)
        deduped_scores = _dedup_scores(parsed_scores, warnings)

        # -- Sort --
        ordered = _sorted_scores(deduped_scores)

        # -- Filter to scores that still have a matching cluster --
        ordered = [s for s in ordered if s.cluster_id in cluster_index]

        # -- Cap --
        if len(ordered) > _MAX_CLUSTERS:
            warnings.append(
                f"Truncated to top {_MAX_CLUSTERS} ranked clusters"
            )
            ordered = ordered[:_MAX_CLUSTERS]

        # -- Derive ID lists --
        supported = []
        rejected = []
        for i in range(len(ordered)):
            s = ordered[i]
            if s.verdict == "SAFE":
                supported.append(s.cluster_id)
            elif s.verdict == "REJECT":
                rejected.append(s.cluster_id)

        # -- Build sentences --
        sentences = _base_sentences(ordered, cluster_index)

        # -- Optional rewrite --
        if rewrite_client is not None:
            sentences = _apply_transition_rewrite(sentences, rewrite_client)

        # -- Assemble output --
        text = " ".join(sentences)
        safe_answer = {
            "text": text,
            "supported_cluster_ids": supported,
            "rejected_cluster_ids": rejected,
        }
        return (safe_answer, warnings)

    except Exception as exc:
        return (
            {
                "text": _FALLBACK_TEXT,
                "supported_cluster_ids": [],
                "rejected_cluster_ids": [],
            },
            [f"build_safe_answer failed: {exc!r}"],
        )
