"""LLM provider API calls for the TruthLens pipeline.

Simple synchronous HTTP calls to LLM APIs (OpenAI, Claude, Gemini).
No streaming needed since the frontend polls for results.
Runs on Modal CPU containers.
"""

import concurrent.futures

_MAX_PROVIDERS = 10
_LLM_TIMEOUT_SECONDS = 90
_MAX_RESPONSE_LENGTH = 100_000


def call_openai(prompt, api_key):
    """Call OpenAI API and return response.

    Args:
        prompt: str, the user prompt
        api_key: str, OpenAI API key

    Returns:
        dict: {"model_id": "openai_gpt4", "response_text": str}
        On error: {"model_id": "openai_gpt4", "response_text": "", "error": str}
    """
    import httpx

    model_id = "openai_gpt4"
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    body = {
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 4096,
    }
    try:
        with httpx.Client(timeout=_LLM_TIMEOUT_SECONDS) as client:
            resp = client.post(url, headers=headers, json=body)
            resp.raise_for_status()
            data = resp.json()
            text = data["choices"][0]["message"]["content"]
            return {"model_id": model_id, "response_text": text[:_MAX_RESPONSE_LENGTH]}
    except Exception as exc:
        return {"model_id": model_id, "response_text": "", "error": str(exc)}


def call_claude(prompt, api_key):
    """Call Anthropic Claude API and return response.

    Args:
        prompt: str, the user prompt
        api_key: str, Anthropic API key

    Returns:
        dict: {"model_id": "claude_sonnet_4", "response_text": str}
        On error: {"model_id": "claude_sonnet_4", "response_text": "", "error": str}
    """
    import httpx

    model_id = "claude_sonnet_4"
    url = "https://api.anthropic.com/v1/messages"
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "Content-Type": "application/json",
    }
    body = {
        "model": "claude-sonnet-4-20250514",
        "max_tokens": 4096,
        "messages": [{"role": "user", "content": prompt}],
    }
    try:
        with httpx.Client(timeout=_LLM_TIMEOUT_SECONDS) as client:
            resp = client.post(url, headers=headers, json=body)
            resp.raise_for_status()
            data = resp.json()
            text = data["content"][0]["text"]
            return {"model_id": model_id, "response_text": text[:_MAX_RESPONSE_LENGTH]}
    except Exception as exc:
        return {"model_id": model_id, "response_text": "", "error": str(exc)}


def call_gemini(prompt, api_key):
    """Call Google Gemini API and return response.

    Args:
        prompt: str, the user prompt
        api_key: str, Google Gemini API key

    Returns:
        dict: {"model_id": "gemini_2_0", "response_text": str}
        On error: {"model_id": "gemini_2_0", "response_text": "", "error": str}
    """
    import httpx

    model_id = "gemini_2_0"
    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"gemini-2.0-flash:generateContent?key={api_key}"
    )
    headers = {"Content-Type": "application/json"}
    body = {
        "contents": [{"parts": [{"text": prompt}]}],
    }
    try:
        with httpx.Client(timeout=_LLM_TIMEOUT_SECONDS) as client:
            resp = client.post(url, headers=headers, json=body)
            resp.raise_for_status()
            data = resp.json()
            text = data["candidates"][0]["content"]["parts"][0]["text"]
            return {"model_id": model_id, "response_text": text[:_MAX_RESPONSE_LENGTH]}
    except Exception as exc:
        return {"model_id": model_id, "response_text": "", "error": str(exc)}


def call_all_llms(prompt, api_keys):
    """Call all enabled LLM providers concurrently.

    Args:
        prompt: str, the user prompt
        api_keys: dict with keys "openai", "anthropic", "gemini"

    Returns:
        tuple: (responses: list[dict], warnings: list[str])
            Each response: {"model_id": str, "response_text": str}
            Warnings for any provider that failed.
    """
    responses = []
    warnings = []

    providers = []
    if api_keys.get("openai"):
        providers.append(("openai", call_openai, api_keys["openai"]))
    if api_keys.get("anthropic"):
        providers.append(("anthropic", call_claude, api_keys["anthropic"]))
    if api_keys.get("gemini"):
        providers.append(("gemini", call_gemini, api_keys["gemini"]))

    with concurrent.futures.ThreadPoolExecutor(max_workers=_MAX_PROVIDERS) as executor:
        future_map = {}
        for idx in range(min(len(providers), _MAX_PROVIDERS)):
            name, fn, key = providers[idx]
            future = executor.submit(fn, prompt, key)
            future_map[future] = name

        for future in concurrent.futures.as_completed(future_map):
            name = future_map[future]
            try:
                result = future.result(timeout=_LLM_TIMEOUT_SECONDS + 10)
                if result.get("error"):
                    warnings.append(f"LLM {name} error: {result['error']}")
                if result.get("response_text"):
                    responses.append(result)
            except Exception as exc:
                warnings.append(f"LLM {name} failed: {str(exc)}")

    return responses, warnings
