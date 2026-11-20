import json
import re
from typing import Any, Dict, List, Set, Tuple
import requests


OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "qwen2.5:7b"  # Change to whichever model you pull in Ollama


SYSTEM_PROMPT = """You are a precise information extraction engine for technical and research papers.
Extract factual relationships as triplets with keys: "subject", "predicate", and "object".
Rules:
- Keep entities concise and canonical (e.g., "bridge deck", "fatigue crack", "ultrasound inspection").
- Use clear, short predicates (e.g., "causes", "detects", "improves", "occurs_in", "measured_by").
- Return ONLY a JSON array of objects, with no extra commentary or markdown fences.
Example:
[{"subject": "cyclic loading", "predicate": "initiates", "object": "fatigue crack"}]
"""


def _parse_triplets_from_response(raw_text: str) -> List[Dict[str, str]]:
    """Extracts and validates a list of {subject, predicate, object} dicts from LLM output."""
    cleaned = raw_text.strip()

    # Strip ```json ... ``` fences if the model includes them
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)

    # Try parsing directly, or extract the first JSON array substring
    data: Any = None
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\[.*\]", cleaned, flags=re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(0))
            except json.JSONDecodeError:
                return []
        else:
            return []

    if not isinstance(data, list):
        return []

    valid_triplets: List[Dict[str, str]] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        subj = str(item.get("subject", "")).strip()
        pred = str(item.get("predicate", "")).strip()
        obj = str(item.get("object", "")).strip()
        if subj and pred and obj:
            valid_triplets.append({
                "subject": subj,
                "predicate": pred,
                "object": obj,
            })

    return valid_triplets


def _call_ollama_for_chunk(
    text: str,
    model: str = DEFAULT_MODEL,
    timeout: int = 120,
) -> List[Dict[str, str]]:
    prompt = f"{SYSTEM_PROMPT}\n\nText:\n{text}\n\nTriplets JSON:"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.1,  # Low temperature for consistent structured extraction
        },
    }

    response = requests.post(OLLAMA_URL, json=payload, timeout=timeout)
    response.raise_for_status()
    body = response.json()
    llm_output = body.get("response", "")
    return _parse_triplets_from_response(llm_output)


def extract_triplets_from_chunks(
    chunks: List[Dict[str, Any]],
    model: str = DEFAULT_MODEL,
    deduplicate: bool = True,
) -> List[Dict[str, str]]:
    """
    Runs Ollama triplet extraction across chunks produced by chunk_pdf_structured_recursive.
    Returns a list of {"subject": ..., "predicate": ..., "object": ...} dicts.
    """
    all_triplets: List[Dict[str, str]] = []
    seen: Set[Tuple[str, str, str]] = set()

    total = len(chunks)
    for idx, chunk in enumerate(chunks):
        text = str(chunk.get("text", "")).strip()
        if not text:
            continue

        chunk_id = chunk.get("chunk_id", idx)
        page = chunk.get("page", "?")

        try:
            extracted = _call_ollama_for_chunk(text=text, model=model)
        except Exception as exc:
            print(f"      [WARN] Chunk {chunk_id} (p.{page}) extraction failed: {exc}")
            continue

        for t in extracted:
            key = (
                t["subject"].lower(),
                t["predicate"].lower(),
                t["object"].lower(),
            )
            if deduplicate and key in seen:
                continue
            seen.add(key)
            all_triplets.append(t)

    print(f"      -> Extracted {len(all_triplets)} unique triplets across {total} chunks.")
    return all_triplets