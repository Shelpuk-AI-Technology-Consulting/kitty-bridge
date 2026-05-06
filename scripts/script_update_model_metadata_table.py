#!/usr/bin/env python3
"""Fetch model metadata from OpenRouter and write src/kitty/providers/model_metadata.json.

Usage:
    OPENROUTER_API_KEY=sk-... python scripts/update_model_metadata.py

Reads OPENROUTER_API_KEY from the environment.
Writes a sorted JSON array of model records to model_metadata.json.
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

API_URL = "https://openrouter.ai/api/v1/models"
OUTPUT_PATH = Path(__file__).resolve().parent.parent / "src" / "kitty" / "providers" / "model_metadata.json"

FIELDS = ("id", "name", "context_length", "max_completion_tokens", "created")


def fetch_models(api_key: str) -> list[dict]:
    """Fetch model list from OpenRouter API."""
    req = urllib.request.Request(
        API_URL,
        headers={
            "Authorization": f"Bearer {api_key}",
            "User-Agent": "kitty-bridge-metadata-update",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        print(f"HTTP error {exc.code}: {exc.reason}", file=sys.stderr)
        sys.exit(1)
    except urllib.error.URLError as exc:
        print(f"Network error: {exc.reason}", file=sys.stderr)
        sys.exit(1)

    if not isinstance(data, dict) or "data" not in data:
        print("Unexpected API response format", file=sys.stderr)
        sys.exit(1)

    return data["data"]


def extract_record(model: dict) -> dict | None:
    """Extract relevant fields from an OpenRouter model object."""
    if not isinstance(model, dict) or "id" not in model:
        return None

    # Skip models without context_length
    context_length = model.get("context_length")
    if not context_length or not isinstance(context_length, (int, float)):
        return None

    # Get max_completion_tokens from top_provider if available
    top_provider = model.get("top_provider") or {}
    max_completion = top_provider.get("max_completion_tokens")

    record: dict = {
        "id": model["id"],
        "name": model.get("name", model["id"]),
        "context_length": int(context_length),
        "max_completion_tokens": int(max_completion) if max_completion else None,
        "created": model.get("created"),
    }
    return record


def write_metadata(records: list[dict], path: Path) -> None:
    """Write records as sorted JSON array, one line per record."""
    records.sort(key=lambda r: r["id"])

    # Compact JSON: one object per line inside the array
    lines = [json.dumps(r, ensure_ascii=False) for r in records]
    content = "[\n" + ",\n".join(lines) + "\n]\n"

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def main() -> None:
    api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        print("OPENROUTER_API_KEY environment variable is required", file=sys.stderr)
        sys.exit(1)

    print(f"Fetching models from {API_URL} ...")
    raw_models = fetch_models(api_key)
    print(f"Received {len(raw_models)} models from OpenRouter")

    records = []
    for m in raw_models:
        rec = extract_record(m)
        if rec is not None:
            records.append(rec)

    if not records:
        print("Warning: extracted 0 model records from OpenRouter response", file=sys.stderr)

    write_metadata(records, OUTPUT_PATH)

    size_kb = OUTPUT_PATH.stat().st_size / 1024
    print(f"Wrote {len(records)} models to {OUTPUT_PATH} ({size_kb:.1f} KB)")


if __name__ == "__main__":
    main()
