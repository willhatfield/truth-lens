import asyncio
import logging
from hashlib import sha1
from typing import Any
logger = logging.getLogger(__name__)


async def retrieve_evidence(
    claims: list[dict[str, Any]],
    max_results: int = 5,
) -> dict[str, list[dict[str, str]]]:
    try:
        sem = asyncio.Semaphore(5)
        loop = asyncio.get_running_loop()

        async def _fetch_one(claim: dict[str, Any]) -> tuple[str, list[dict[str, str]]]:
            claim_id = str(claim.get("claim_id", ""))
            claim_text = str(claim.get("claim_text", ""))
            if not claim_id or not claim_text:
                return claim_id, []

            async with sem:
                def _search() -> list[dict]:
                    from duckduckgo_search import DDGS
                    return list(DDGS().text(claim_text, max_results=max_results))

                results = await loop.run_in_executor(None, _search)

            passages: list[dict[str, str]] = []
            for r in results:
                url = str(r.get("href", "") or r.get("url", ""))
                body = str(r.get("body", "") or r.get("snippet", ""))
                if not body:
                    continue
                passage_id = "p_" + sha1(
                    f"{claim_id}:{url}".encode()
                ).hexdigest()[:16]
                passages.append({"passage_id": passage_id, "text": body, "source": url})
            return claim_id, passages

        tasks = [_fetch_one(c) for c in claims]
        try:
            pairs = await asyncio.gather(*tasks, return_exceptions=True)
            
            for item in pairs:
                if isinstance(item, Exception):
                    logger.error(f"A DuckDuckGo search task failed: {item}")

            return {
                cid: passages
                for item in pairs
                if isinstance(item, tuple)
                for cid, passages in [item]
                if cid
            }

        except Exception as e:
            logger.exception(f"Unexpected error during concurrent evidence retrieval: {e}")
            return {}
    except Exception:
        return {}
