"""Download the public customer-support ticket dataset and commit a fixed 2,000-ticket sample.

    uv run soa-fetch

Source: Tobi-Bueck/customer-support-tickets on Hugging Face (CC BY-NC 4.0; see ATTRIBUTION.md).
20,000 tickets, each with subject, body, a reference answer, and labels for queue, priority
and type. We keep the 11,923 English rows, drop empties, and sample 2,000 with a fixed seed.
The first 1,500 are the training split for the classifier; the last 500 are held out and
never seen by training. Output: data/raw/tickets_sample.json.gz (about 600 KB).
"""

from __future__ import annotations

import csv
import gzip
import io
import json
import random
from pathlib import Path

import httpx

URL = (
    "https://huggingface.co/datasets/Tobi-Bueck/customer-support-tickets/resolve/main/"
    "dataset-tickets-multi-lang-4-20k.csv"
)
OUT = Path("data/raw/tickets_sample.json.gz")
SEED = 42
N = 2000
TRAIN = 1500


def _clean(text: str) -> str:
    # Some rows contain the two characters backslash-n instead of a newline.
    return " ".join((text or "").replace("\\n", " ").replace("\r", " ").split())


def build(rows: list[dict], seed: int = SEED, n: int = N) -> list[dict]:
    english = [
        r
        for r in rows
        if r.get("language") == "en" and _clean(r.get("subject")) and _clean(r.get("body")) and r.get("queue")
    ]
    rng = random.Random(seed)
    picked = rng.sample(english, n)
    out = []
    for i, r in enumerate(picked, start=1):
        tags = [r[f"tag_{k}"] for k in range(1, 9) if r.get(f"tag_{k}")]
        out.append(
            {
                "id": f"T-{i:04d}",
                "split": "train" if i <= TRAIN else "heldout",
                "subject": _clean(r["subject"]),
                "body": _clean(r["body"]),
                "reference_answer": _clean(r.get("answer", "")),
                "type": r.get("type", ""),
                "queue": r["queue"],
                "priority": r.get("priority", ""),
                "tags": tags,
            }
        )
    return out


def main() -> None:
    print(f"downloading {URL}")
    raw = httpx.get(URL, follow_redirects=True, timeout=120).content
    rows = list(csv.DictReader(io.StringIO(raw.decode("utf-8"))))
    sample = build(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(OUT, "wt", encoding="utf-8") as fh:
        json.dump(sample, fh)
    print(f"wrote {len(sample)} tickets to {OUT} ({OUT.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
