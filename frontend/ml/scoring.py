"""Trust score algorithm for TruthLens cluster scoring."""

TOTAL_MODELS = 5  # Number of LLM providers in the arena


def clamp(value: float, min_val: float, max_val: float) -> float:
    """Clamp value to [min_val, max_val] range."""
    if value < min_val:
        return min_val
    if value > max_val:
        return max_val
    return value


def compute_agreement_score(supporting_count: int, total_models: int) -> float:
    """agreement_score = 100 * (supporting_count / total_models)."""
    if total_models <= 0:
        return 0.0
    return 100.0 * (supporting_count / total_models)


def compute_verification_score(
    best_entailment: float, best_contradiction: float
) -> float:
    """verification_score = 100 * best_entailment - 100 * best_contradiction."""
    return 100.0 * best_entailment - 100.0 * best_contradiction


def compute_trust_score(
    agreement_score: float,
    verification_score: float,
    agreement_weight: float,
    verification_weight: float,
) -> int:
    """trust_score = round(aw * agreement + vw * clamp(verification, 0, 100))."""
    clamped_verification = clamp(verification_score, 0.0, 100.0)
    raw = agreement_weight * agreement_score + verification_weight * clamped_verification
    return round(clamp(raw, 0.0, 100.0))


def determine_verdict(
    trust_score: int,
    best_contradiction_prob: float,
    safe_min: int,
    caution_min: int,
) -> str:
    """Classify trust score into SAFE, CAUTION, or REJECT.

    SAFE:    trust_score >= safe_min AND best_contradiction_prob <= 0.2
    CAUTION: trust_score >= caution_min
    REJECT:  otherwise
    """
    if trust_score >= safe_min and best_contradiction_prob <= 0.2:
        return "SAFE"
    if trust_score >= caution_min:
        return "CAUTION"
    return "REJECT"


def find_supporting_models(claim_ids: list, claims: dict) -> list:
    """Return list of unique model_ids that contributed claims to this cluster.

    Args:
        claim_ids: List of claim IDs in the cluster.
        claims: Dict mapping claim_id to an object with a model_id attribute.

    Bounded loop over claim_ids (max 500).
    """
    seen: dict = {}
    models: list = []
    max_iter = 500
    count = 0
    for i in range(len(claim_ids)):
        if count >= max_iter:
            break
        count += 1
        cid = claim_ids[i]
        if cid in claims:
            mid = claims[cid].model_id
            if mid not in seen:
                seen[mid] = True
                models.append(mid)
    return models


def find_best_nli_for_cluster(
    claim_ids: list,
    nli_results: list,
) -> tuple:
    """Find the best entailment and worst contradiction for a cluster.

    Returns: (best_entailment_prob, best_contradiction_prob, evidence_passage_id)

    Bounded loops over nli_results (max 5000) and claim_ids.
    """
    best_ent = 0.0
    best_contra = 0.0
    evidence_pid = ""
    claim_set: dict = {}
    for i in range(len(claim_ids)):
        claim_set[claim_ids[i]] = True

    max_iter = 5000
    count = 0
    for i in range(len(nli_results)):
        if count >= max_iter:
            break
        count += 1
        result = nli_results[i]
        if result.claim_id not in claim_set:
            continue
        ent = result.probs.get("entailment", 0.0)
        contra = result.probs.get("contradiction", 0.0)
        if ent > best_ent:
            best_ent = ent
            evidence_pid = result.passage_id
        if contra > best_contra:
            best_contra = contra

    return (best_ent, best_contra, evidence_pid)
