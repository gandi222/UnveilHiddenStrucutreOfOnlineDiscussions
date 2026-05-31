"""
Low-level LLM communication: sending prompts to Ollama and parsing responses.
"""

import json
import logging
import re
import time

from langchain_ollama import ChatOllama

log = logging.getLogger(__name__)


def call_api(llm: ChatOllama, prompt: str, max_retries: int) -> str:
    """Send a prompt to the Ollama server and return the raw response text."""
    for attempt in range(1, max_retries + 1):
        try:
            response = llm.invoke(prompt)
            return response.content
        except Exception as exc:
            log.warning("API call failed (attempt %d/%d): %s", attempt, max_retries, exc)
            if attempt < max_retries:
                time.sleep(5)
            else:
                raise RuntimeError("Max retries exceeded") from exc


def parse_json_array(text: str, expected: int) -> list:
    """Extract a JSON array from the model response and verify its length.
    Strips markdown code fences (```json ... ```) that the model sometimes adds."""
    text = re.sub(r"```(?:json)?", "", text).strip().rstrip("`").strip()
    match = re.search(r"\[.*?\]", text, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON array found in response: {text!r}")
    result = json.loads(match.group())
    if len(result) != expected:
        raise ValueError(f"Expected {expected} items in response, got {len(result)}")
    return result
