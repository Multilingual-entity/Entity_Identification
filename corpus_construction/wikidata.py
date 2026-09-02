"""Thin, polite Wikidata client: label lookup via the API, entity search via SPARQL.

Both endpoints are rate-limited and ask for a descriptive User-Agent. Results are cached
on disk so a re-run costs nothing and an interrupted run resumes.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import requests

API = "https://www.wikidata.org/w/api.php"
SPARQL = "https://query.wikidata.org/sparql"
UA = "cross-script-entity-matching/1.0 (research; contact via repository)"

CACHE = Path(__file__).resolve().parent / "cache"
CACHE.mkdir(exist_ok=True)

_LAST_CALL = [0.0]
MIN_INTERVAL = 2.5          # seconds between calls
BACKOFF = (5, 15, 45, 90, 180)   # a 429 needs minutes, not seconds


def _throttle() -> None:
    gap = time.time() - _LAST_CALL[0]
    if gap < MIN_INTERVAL:
        time.sleep(MIN_INTERVAL - gap)
    _LAST_CALL[0] = time.time()


def _wait(attempt: int, exc: Exception, what: str) -> None:
    """Back off. Wikimedia rate-limits by IP across its services, so a burst of SPARQL
    can produce a 429 on the label API even though they are different endpoints. The
    server tells us how long to wait when it can; otherwise escalate."""
    delay = BACKOFF[min(attempt, len(BACKOFF) - 1)]
    response = getattr(exc, "response", None)
    if response is not None and 500 <= response.status_code < 600:
        # 502 and 504 mean the query service itself is struggling, usually because we have
        # been asking too fast. Backing off harder is the only thing that helps, and the
        # earlier schedule was tuned for rate limits rather than for an overloaded server.
        delay = max(delay, 60 * (attempt + 1))
        what = f"{what} (server error {response.status_code})"
    if response is not None:
        header = response.headers.get("Retry-After")
        if header:
            try:
                delay = max(delay, int(float(header)))
            except ValueError:
                pass
        if response.status_code == 429:
            what = f"{what} (rate limited)"
    print(f"    {what}: waiting {delay}s before retry {attempt + 2}", flush=True)
    time.sleep(delay)


def _cache_path(kind: str, key: str) -> Path:
    return CACHE / f"{kind}_{hashlib.sha256(key.encode()).hexdigest()[:16]}.json"


def _cached(kind: str, key: str):
    p = _cache_path(kind, key)
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            p.unlink()
    return None


def _store(kind: str, key: str, value) -> None:
    _cache_path(kind, key).write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def labels(qids, langs, retries: int = 5) -> dict:
    """{qid: {lang: label}} for up to 50 QIDs per call. Missing labels are simply absent."""
    qids = list(qids)
    out: dict = {}
    for i in range(0, len(qids), 50):
        chunk = qids[i:i + 50]
        key = "|".join(chunk) + "::" + "|".join(sorted(langs))
        hit = _cached("labels", key)
        if hit is not None:
            out.update(hit)
            continue
        params = {
            "action": "wbgetentities",
            "ids": "|".join(chunk),
            "props": "labels|aliases",
            "languages": "|".join(langs),
            "format": "json",
        }
        data = None
        for attempt in range(retries):
            _throttle()
            try:
                r = requests.get(API, params=params, headers={"User-Agent": UA}, timeout=60)
                r.raise_for_status()
                data = r.json()
                break
            except Exception as exc:                                    # noqa: BLE001
                if attempt == retries - 1:
                    raise RuntimeError(f"label fetch failed for {chunk[:3]}...: {exc}") from exc
                _wait(attempt, exc, f"labels {i // 50 + 1}/{(len(qids) + 49) // 50}")
        block = {}
        for qid, ent in (data or {}).get("entities", {}).items():
            block[qid] = {
                "labels": {lg: v["value"] for lg, v in ent.get("labels", {}).items()},
                "aliases": {lg: [a["value"] for a in vs]
                            for lg, vs in ent.get("aliases", {}).items()},
            }
        _store("labels", key, block)
        out.update(block)
    return out


def entities(qids, props="claims|sitelinks", langs=None, retries: int = 5) -> dict:
    """Raw wbgetentities blocks, cached and throttled exactly like labels().

    Separate from labels() because the second name has to come from a statement rather
    than an alias list. Wikidata's alias lists are per-language and unlinked -- the Hindi
    list and the English list are two independent sets with nothing tying one entry to
    another -- so pairing across them requires guessing which entry corresponds to which.
    A statement does not have that problem: P1477 with a Hindi value and an English value
    is one claim about one name, written twice by editors who could read both.
    """
    qids = list(qids)
    out: dict = {}
    for i in range(0, len(qids), 50):
        chunk = qids[i:i + 50]
        key = "|".join(chunk) + "::" + props + "::" + "|".join(sorted(langs or []))
        hit = _cached("entities", key)
        if hit is not None:
            out.update(hit)
            continue
        params = {
            "action": "wbgetentities",
            "ids": "|".join(chunk),
            "props": props,
            "format": "json",
        }
        if langs:
            params["languages"] = "|".join(langs)
        data = None
        for attempt in range(retries):
            _throttle()
            try:
                r = requests.get(API, params=params, headers={"User-Agent": UA}, timeout=60)
                r.raise_for_status()
                data = r.json()
                break
            except Exception as exc:                                        # noqa: BLE001
                if attempt == retries - 1:
                    raise RuntimeError(f"entity fetch failed for {chunk[:3]}...: {exc}") from exc
                _wait(attempt, exc, f"entities {i // 50 + 1}/{(len(qids) + 49) // 50}")
        block = dict((data or {}).get("entities", {}))
        _store("entities", key, block)
        out.update(block)
    return out


class SparqlTooLarge(RuntimeError):
    """The endpoint returned a body that will not parse as JSON.

    On the public query service this is not a transient fault and not a rate limit. It
    means the query hit the sixty-second limit and the response was cut off mid-stream,
    so the JSON ends in the middle of a string or an object. Retrying the identical query
    produces the identical truncation: the first run of this pipeline spent 155 seconds
    of backoff per failure discovering that five times over, then dropped the band.

    The caller has to make the query smaller. Raising a distinct type lets it do that
    immediately instead of waiting out a backoff schedule that cannot help.
    """


def _parse_sparql(text: str) -> dict:
    """Parse a results body, tolerating control characters but not truncation.

    Wikidata labels genuinely contain raw control characters, and Python's JSON parser
    rejects those by default with "Invalid control character at". That is a real response
    that merely needs a laxer parser, and it is worth separating from a truncated one,
    because the two failures look similar and want opposite responses: accept the first,
    split the query for the second.
    """
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    try:
        return json.loads(text, strict=False)
    except json.JSONDecodeError as exc:
        raise SparqlTooLarge(f"unparseable response ({exc}); "
                             f"{len(text)} bytes, likely truncated") from exc


def sparql(query: str, retries: int = 5) -> list:
    """Run a SPARQL query, returning a list of {var: value} dicts.

    Raises SparqlTooLarge when the body will not parse even leniently, and RuntimeError
    for transport failures, which are retried with backoff.
    """
    hit = _cached("sparql", query)
    if hit is not None:
        return hit
    data = None
    for attempt in range(retries):
        _throttle()
        try:
            r = requests.get(SPARQL, params={"query": query, "format": "json"},
                             headers={"User-Agent": UA, "Accept": "application/sparql-results+json"},
                             timeout=180)
            r.raise_for_status()
            data = _parse_sparql(r.text)
            break
        except SparqlTooLarge:
            # One retry only: an occasional truncation is transient, a repeated one means
            # the query is genuinely too big and the caller must split it.
            if attempt >= 1:
                raise
            _wait(attempt, RuntimeError("truncated"), "sparql (truncated response)")
        except Exception as exc:                                        # noqa: BLE001
            if attempt == retries - 1:
                raise RuntimeError(f"SPARQL failed: {exc}") from exc
            _wait(attempt, exc, "sparql")
    rows = [{k: v["value"] for k, v in row.items()}
            for row in (data or {}).get("results", {}).get("bindings", [])]
    _store("sparql", query, rows)
    return rows


def qid_from_uri(uri: str) -> str:
    return uri.rsplit("/", 1)[-1]
