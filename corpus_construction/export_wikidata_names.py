#!/usr/bin/env python3
import argparse
import csv
import json
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


API = "https://www.wikidata.org/w/api.php"


def batches(items, size=25):
    for start in range(0, len(items), size):
        yield items[start : start + size]


def save_output(path, input_file, qid_count, entities, complete):
    output = {
        "source": API,
        "languages": ["hi", "mr", "en"],
        "sites": ["hiwiki", "mrwiki", "enwiki"],
        "input_file": Path(input_file).name,
        "qid_count": qid_count,
        "downloaded_count": len(entities),
        "complete": complete,
        "entities": entities,
    }
    destination = Path(path)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(output, handle, ensure_ascii=False, indent=2)
    temporary.replace(destination)


def download_batch(batch, retries=7):
    query = urlencode(
        {
            "action": "wbgetentities",
            "ids": "|".join(batch),
            "props": "labels|aliases|sitelinks",
            "languages": "hi|mr|en",
            "sitefilter": "hiwiki|mrwiki|enwiki",
            "format": "json",
            "formatversion": "2",
            "maxlag": "5",
        }
    )
    request = Request(
        f"{API}?{query}",
        headers={
            "User-Agent": "DevanagariNameAudit/1.0 (research corpus verification)"
        },
    )

    for attempt in range(1, retries + 1):
        try:
            with urlopen(request, timeout=60) as response:
                return json.load(response).get("entities", {})
        except HTTPError as error:
            if error.code not in (429, 503) or attempt == retries:
                raise
            retry_after = error.headers.get("Retry-After")
            wait = int(retry_after) if retry_after and retry_after.isdigit() else min(300, 15 * (2 ** (attempt - 1)))
            print(f"HTTP {error.code}; waiting {wait} seconds before retry {attempt + 1}/{retries}")
            time.sleep(wait)
        except URLError:
            if attempt == retries:
                raise
            wait = min(120, 10 * (2 ** (attempt - 1)))
            print(f"Network error; waiting {wait} seconds before retry {attempt + 1}/{retries}")
            time.sleep(wait)

    raise RuntimeError("Batch download failed after retries")


def main():
    parser = argparse.ArgumentParser(
        description="Export Hindi/Marathi/English Wikidata names and Wikipedia titles."
    )
    parser.add_argument("input_csv", help="The 601-row Hindi corpus CSV")
    parser.add_argument(
        "output_json", nargs="?", default="wikidata_hi_mr_export.json"
    )
    args = parser.parse_args()

    with open(args.input_csv, encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))

    qids = []
    for row in rows:
        qid = (row.get("qid") or row.get("fact_id", "")).upper()
        if not qid.startswith("Q"):
            raise ValueError(f"Invalid QID in row: {row.get('fact_id')}")
        qids.append(qid)

    if len(qids) != 601 or len(set(qids)) != 601:
        raise ValueError(
            f"Expected 601 unique QIDs; found {len(qids)} rows and {len(set(qids))} unique QIDs"
        )

    entities = {}
    output_path = Path(args.output_json)
    if output_path.exists():
        try:
            with open(output_path, encoding="utf-8") as handle:
                checkpoint = json.load(handle)
            entities = checkpoint.get("entities", {})
            print(f"Resuming from checkpoint with {len(entities)} entities")
        except (OSError, json.JSONDecodeError):
            print("Existing output is not a valid checkpoint; starting again")

    remaining = [qid for qid in qids if qid not in entities]
    if not remaining:
        save_output(args.output_json, args.input_csv, len(qids), entities, True)
        print(f"All {len(entities)} entities are already downloaded")
        return

    total_batches = (len(remaining) + 24) // 25
    for number, batch in enumerate(batches(remaining), start=1):
        entities.update(download_batch(batch))
        complete = len(entities) >= len(qids)
        save_output(args.output_json, args.input_csv, len(qids), entities, complete)
        print(f"Downloaded batch {number}/{total_batches}; checkpoint has {len(entities)} entities")
        time.sleep(2.0)

    print(f"Saved {len(entities)} entities to {args.output_json}")


if __name__ == "__main__":
    main()
