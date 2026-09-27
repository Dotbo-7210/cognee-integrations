"""Embeddable web chat widget powered by cognee memory.

Run a tiny FastAPI backend that any site can talk to via a single ``<script>``
tag. It doubles as an "ask our docs" assistant: seed a docs corpus once and
every visitor conversation can recall from it, with inline **citations** to the
source material.

The backend is a thin, CORS-enabled proxy — all memory behavior lives in
``ChatMemoryAdapter`` (``adapter.py``), which talks to a running cognee server
over HTTP and never imports cognee in-process. Point it at that server with
``COGNEE_BASE_URL`` (and ``COGNEE_API_KEY`` if the server has auth on).

Run it::

    cognee-web-widget            # or: python -m cognee_integration_web_widget.server

Then open http://127.0.0.1:8000 for the "ask our docs" demo page, or embed the
widget on your own site::

    <script src="http://127.0.0.1:8000/widget.js"
            data-site-id="acme" data-api="http://127.0.0.1:8000"></script>

Endpoints:
    POST /api/chat    -> {answer, citations, session_id}
    POST /api/forget  -> clear one conversation's memory
    GET  /            -> demo "ask our docs" page
    GET  /widget.js   -> the embeddable snippet
"""

from __future__ import annotations

import asyncio
import os
import re
import secrets
import time
from collections import Counter
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from pydantic import BaseModel

from .adapter import ChatMemoryAdapter
from .docs_drift import drift_for_items
from .docs_ingest import item_name, render_for_ingest, upload_filename

STATIC_DIR = Path(__file__).resolve().parent / "static"

# A couple of "docs" so the demo returns something without external setup.
# Replace these with your real docs/site content.
DEMO_DOCS = [
    "Cognee is an open-source AI memory platform. It turns raw data into a "
    "knowledge graph that AI agents can recall from, replacing plain RAG "
    "with an Extract-Cognify-Load pipeline.",
    "You store data with remember(), build the graph, then query it with "
    "recall(). Each conversation is isolated by a session_id, so one user's "
    "chat never leaks into another's.",
    "Use forget() to delete memory for a conversation or dataset. Visitors "
    "can opt out of being remembered at any time.",
]

DEMO_SITE_ID = os.getenv("WIDGET_SITE_ID", "demo")

# Public root of the site whose pages were ingested. Set it and every citation
# links to the page; leave it unset and citations name the document only.
DOCS_BASE_URL = os.getenv("WIDGET_DOCS_BASE_URL", "").strip() or None

adapter = ChatMemoryAdapter(top_k=8, docs_base_url=DOCS_BASE_URL)


# Operator dashboard. Unset means the routes 404 — the dashboard is opt-in, so a
# stock deployment never exposes corpus contents or the visitor question log.
DASHBOARD_TOKEN = os.getenv("WIDGET_DASHBOARD_TOKEN", "").strip() or None


def _require_dashboard(token: Optional[str]) -> None:
    """Gate the dashboard routes.

    404 rather than 401 when the feature is off, so an unconfigured deployment
    is indistinguishable from one without the route at all. Compared with
    ``compare_digest`` so a wrong token cannot be recovered a byte at a time.
    """
    if DASHBOARD_TOKEN is None:
        raise HTTPException(status_code=404)
    if not token or not secrets.compare_digest(token, DASHBOARD_TOKEN):
        raise HTTPException(status_code=401, detail="bad or missing dashboard token")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Seed the demo "ask our docs" corpus once, best-effort, on startup."""
    try:
        await adapter.ingest_docs(site_id=DEMO_SITE_ID, documents=DEMO_DOCS)
    except Exception as error:  # noqa: BLE001 - the demo should still boot
        print(f"[web_widget] docs seeding skipped: {error}")
    # Not awaited: the graph takes ~24s to fetch and nothing should wait for it,
    # least of all the server accepting its first request.
    warm = asyncio.create_task(_warm_graph_cache())
    yield
    warm.cancel()


app = FastAPI(title="cognee web chat widget", lifespan=lifespan)

# The widget is embedded cross-origin on customer sites, so the browser needs
# CORS on /api/*. "*" is the sensible default for a public docs assistant; set
# WIDGET_ALLOWED_ORIGINS (comma-separated) to restrict it.
_origins = [o.strip() for o in os.getenv("WIDGET_ALLOWED_ORIGINS", "*").split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins or ["*"],
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str
    conversation_id: str
    visitor_id: str = "anonymous"
    site_id: str = DEMO_SITE_ID
    opt_in: bool = True


class ForgetRequest(BaseModel):
    conversation_id: str
    visitor_id: str = "anonymous"
    site_id: str = DEMO_SITE_ID


async def _store_exchange(conversation, question: str, answer: str) -> None:
    """Keep this exchange, in the conversations corpus.

    Two jobs in one call. It stores the turn where it outlives the session
    cache - which holds turns for seven days of inactivity and is dropped
    outright when its dataset is deleted - and it is the trigger for cognee's
    session bridge, which persists the session's Q&A into that dataset's graph
    and distils lessons from it. Nothing in cognee does that on a timer; the
    bridge only runs because a remember() with a session id asked it to.

    Not the docs corpus, so none of it can answer a visitor. In the background,
    so the visitor does not wait on a cognify for a reply they already have. And
    never fatal: they have the reply, and losing the record of it is not a
    reason to hand them an error instead.

    Only when the visitor opted in - the same consent that governs whether the
    turn is remembered at all.
    """
    try:
        await adapter.client.remember(
            f"Q: {question}\n\nA: {answer}",
            dataset_name=adapter.conversations_dataset(conversation.site_id),
            session_id=conversation.session_id,
            run_in_background=True,
        )
    except Exception as error:  # noqa: BLE001 - the answer is already out
        print(f"[web_widget] storing the exchange failed: {error}")


@app.post("/api/chat")
async def chat(req: ChatRequest) -> JSONResponse:
    conversation = adapter.conversation(
        site_id=req.site_id, visitor_id=req.visitor_id, conversation_id=req.conversation_id
    )

    # "/forget" is a first-class command, not a question to answer.
    if req.message.strip().lower() in ("/forget", "forget me"):
        try:
            await adapter.forget(conversation=conversation)
            answer_text = "Done — I've forgotten this conversation."
        except Exception:  # noqa: BLE001 - surface a friendly message, not a 500
            answer_text = "Sorry — I couldn't reach memory right now."
        return JSONResponse(
            {
                "answer": answer_text,
                "citations": [],
                "session_id": conversation.session_id,
            }
        )

    answer = await adapter.answer(
        conversation=conversation,
        query=req.message,
        remember=req.opt_in,
    )
    if req.opt_in:
        await _store_exchange(conversation, req.message, answer.text)
    return JSONResponse(answer.as_dict())


@app.post("/api/forget")
async def forget(req: ForgetRequest) -> JSONResponse:
    conversation = adapter.conversation(
        site_id=req.site_id, visitor_id=req.visitor_id, conversation_id=req.conversation_id
    )
    try:
        cleared = await adapter.forget(conversation=conversation)
    except Exception:  # noqa: BLE001 - a failing backend must not 500 the widget
        cleared = False
    return JSONResponse({"cleared": bool(cleared), "session_id": conversation.session_id})


@app.get("/")
async def demo_page() -> FileResponse:
    return FileResponse(STATIC_DIR / "demo.html")


@app.get("/widget.js")
async def widget_js() -> FileResponse:
    return FileResponse(STATIC_DIR / "widget.js", media_type="application/javascript")


def _field(item, *names, default=""):
    """First present value among ``names`` — the APIs vary by endpoint."""
    if not isinstance(item, dict):
        return default
    for n in names:
        v = item.get(n)
        if v not in (None, ""):
            return v
    return default


DOCS_PATH = os.getenv("WIDGET_DOCS_PATH", "").strip() or None
# Used only to stamp a Source line into ingested text.
DOCS_URL = os.getenv("WIDGET_DOCS_URL", "https://docs.cognee.ai").strip()


def _corpus_sync(drift: dict) -> dict:
    """Summarise documentation drift for the header badge.

    Deliberately not derived from updatedAt. That field moves whenever cognee
    reprocesses a record, so a dataset-wide re-cognify reported every source as
    changed while nothing had been edited.

    A page deleted from the docs counts as needing attention just as much as an
    edited one: its content still answers questions and still cites a URL that
    now 404s.
    """
    base = {"state": "unknown", "matched": 0, "drifted": 0, "removed": 0}
    if not drift.get("enabled") or not (drift["matched"] or drift["removed"]):
        return base
    needs_attention = drift["drifted"] + drift["removed"]
    return {
        "state": "stale" if needs_attention else "synced",
        "matched": drift["matched"],
        "drifted": drift["drifted"],
        "removed": drift["removed"],
    }


async def _dashboard_data() -> dict:
    """Everything the dashboard shows, gathered read-only from cognee."""
    client = adapter.client
    docs_dataset = adapter.docs_dataset(DEMO_SITE_ID)

    # Three round trips to cognee, each about a second against a hosted tenant,
    # and they were originally run in series for ~2.5s before the page could
    # paint. Only one ordering is forced: the dataset list names the corpus, so
    # its items cannot be asked for until it lands. The recall history is
    # addressed to the principal rather than to a dataset, so it is started
    # first and collected last - it costs nothing but the wait it removes.
    history_task = asyncio.create_task(client.recall_history())
    datasets = await client.list_datasets()
    match = next(
        (d for d in datasets if isinstance(d, dict) and d.get("name") == docs_dataset), None
    )
    items = await (client.dataset_data(str(match.get("id"))) if match else _empty_list())
    history = await history_task

    # Recall history is the whole principal's, and it interleaves both sides of
    # each exchange: `user` rows are the questions asked, `system` rows the
    # answers returned. Keep the questions, and scope to the docs dataset so the
    # list is what visitors asked *this* widget rather than every recall the key
    # has ever run. If nothing carries the dataset id (older rows record none),
    # fall back to unscoped so the panel is never mysteriously empty.
    dataset_id = str(match.get("id")) if match else None
    asked = [h for h in history if isinstance(h, dict) and _field(h, "user") == "user"]
    scoped = [h for h in asked if dataset_id and str(_field(h, "datasetId")) == dataset_id]
    rows, is_scoped = (scoped, True) if scoped else (asked, False)
    rows.sort(key=lambda h: str(_field(h, "createdAt")), reverse=True)

    # Reads and hashes the source files, so it is local work rather than another
    # round trip. DOCS_URL is the one ingest stamps into each document, and the
    # comparison is over the rendered bytes, so the two have to agree.
    drift = drift_for_items(items, DOCS_PATH, DOCS_URL)

    questions = []
    for h in rows[:25]:
        q = _field(h, "text", "query", "question")
        if q:
            questions.append({"query": str(q)[:300], "at": str(_field(h, "createdAt"))})

    return {
        "config": {
            "cognee_base_url": client.base_url,
            "authenticated": bool(client.api_key),
            "docs_dataset": docs_dataset,
            "site_id": DEMO_SITE_ID,
            # Seeding is unconditional here: DEMO_DOCS is written on every boot.
            # Surfaced because against a real corpus that is a bug, not a demo.
            "seeds_demo_docs": True,
            "allowed_origins": _origins or ["*"],
            "cloud_reachable": bool(datasets),
        },
        "corpus": {
            "dataset": docs_dataset,
            "dataset_id": dataset_id,
            # Whether the corpus still matches the documentation it came from,
            # decided by hashing each page as ingest would render it and
            # comparing that with the digest in the item's storage path.
            #
            # This says nothing about the graph. A corpus can be current and its
            # graph still be built from an older pass; the breakdown below
            # counts the graph itself, and the two questions stay separate.
            "sync": _corpus_sync(drift),
            "exists": match is not None,
            "item_count": len(items),
            "items": [
                {
                    "id": str(_field(i, "id")),
                    "name": str(_field(i, "name", "rawDataLocation"))[:120],
                    "created": str(_field(i, "createdAt")),
                    # "Last synced" is updatedAt: it moves when the item is
                    # re-ingested, not when the underlying page is edited.
                    "updated": str(_field(i, "updatedAt")),
                    "source": str(
                        (_field(i, "externalMetadata", default={}) or {})
                        .get("_cognee", {})
                        .get("source_uri", "")
                    ),
                    # Same comparison the header badge makes, decided once here
                    # so a row can never disagree with the summary above it.
                    # current | edited | removed | foreign, or None when drift
                    # checking is switched off.
                    "doc_state": drift["states"].get(str(_field(i, "id"))),
                }
                for i in items
            ],
            "all_datasets": [str(_field(d, "name")) for d in datasets],
        },
        # No graph counts here. /graph-summary reports the latest pipeline run
        # rather than the dataset - it returns zeros after a small run, and its
        # history goes stale - so the counts come from the breakdown endpoint,
        # which counts the graph itself.
        "questions": questions,
        "questions_scoped_to_dataset": is_scoped,
    }


@app.get("/api/dashboard")
async def dashboard_data(token: Optional[str] = Query(default=None)) -> JSONResponse:
    _require_dashboard(token)
    return JSONResponse(await _dashboard_data())


async def _empty_list() -> list:
    """An already-satisfied empty result, so the gather above stays symmetrical."""
    return []


async def _docs_dataset_id() -> str:
    """The id of the widget's own dataset, resolved server-side.

    Every mutating route goes through this rather than accepting a dataset id
    from the page: the key can write to anything it can reach, and the
    dashboard must not be a way to reach the rest of the tenant.
    """
    datasets = await adapter.client.list_datasets()
    docs_dataset = adapter.docs_dataset(DEMO_SITE_ID)
    match = next(
        (d for d in datasets if isinstance(d, dict) and d.get("name") == docs_dataset), None
    )
    if not match:
        raise HTTPException(status_code=404, detail="docs dataset not found")
    return str(match.get("id"))


# cognee renders this page itself - the same HTML artifact visualize_graph()
# writes - but the call takes ~40s, is not cached upstream, and two in flight at
# once was enough to make the tenant answer 503. So it is cached here and
# serialised: one upstream render at a time, shared until stale.
_VIZ_TTL_SECONDS = 900
_viz_cache: dict = {"html": None, "at": 0.0, "dataset": None}
_viz_lock = asyncio.Lock()


def _invalidate_viz_cache() -> None:
    """Drop the cached render because the corpus it pictures has changed.

    The cache is keyed on the dataset id, which stays the same while its
    contents do not, so nothing else expires it: ingesting or deleting a source
    left the graph serving a picture of the corpus as it was, for the rest of
    its fifteen minutes, with no sign that it was describing something gone.

    Dropping it does not rebuild anything. The next opener pays the render, and
    cognee builds the graph behind an ingest anyway, so an eager rebuild here
    would spend forty seconds drawing the shape the ingest has not reached yet.
    """
    _viz_cache.update({"html": None, "at": 0.0, "dataset": None})
    # The counts and the repository list describe the same corpus, so they go
    # stale at the same moment.
    _graph_cache.update({"summary": None, "at": 0.0, "dataset": None})


def _prefer_dark(html: str) -> str:
    """Make cognee's graph page open dark, without taking the choice away.

    The page ships ``<html class="light">`` and its script restores
    ``cognee-viz-theme`` from storage, defaulting to light. Two small edits:
    drop the class so the first paint uses the dark ``:root`` variables the
    stylesheet already defines, and seed the stored preference *only when it is
    unset*, so a later click of its own Dark/Light button still wins and sticks.
    """
    html = html.replace('<html lang="en" class="light">', '<html lang="en">', 1)
    seed = (
        "<script>try{if(!localStorage.getItem('cognee-viz-theme'))"
        "localStorage.setItem('cognee-viz-theme','dark');}catch(e){}</script>"
    )
    # Before their scripts, so the seeded value is what the restore reads.
    return html.replace("<head>", "<head>" + seed, 1)


@app.get("/api/dashboard/graph-html", response_class=HTMLResponse)
async def dashboard_graph_html(
    token: Optional[str] = Query(default=None),
    refresh: bool = Query(default=False),
) -> HTMLResponse:
    """cognee's own graph rendering, cached. ``refresh=true`` forces a rebuild."""
    _require_dashboard(token)
    dataset_id = await _docs_dataset_id()

    def _fresh() -> bool:
        return bool(
            _viz_cache["html"]
            and _viz_cache["dataset"] == dataset_id
            and time.time() - _viz_cache["at"] < _VIZ_TTL_SECONDS
        )

    if _fresh() and not refresh:
        age = int(time.time() - _viz_cache["at"])
        return HTMLResponse(_viz_cache["html"], headers={"X-Cache": "hit", "X-Cache-Age": str(age)})

    # One render at a time: a second opener waits for the first rather than
    # starting another 40s job against the same tenant.
    async with _viz_lock:
        if _fresh() and not refresh:
            return HTMLResponse(_viz_cache["html"], headers={"X-Cache": "hit-after-wait"})
        html = await adapter.client.visualize_html(dataset_id)
        if not html:
            raise HTTPException(status_code=502, detail="cognee could not render the graph")
        html = _prefer_dark(html)
        _viz_cache.update({"html": html, "at": time.time(), "dataset": dataset_id})
    return HTMLResponse(html, headers={"X-Cache": "miss"})


_EMPTY_ANSWER_MARKER = "I don't have anything in memory for that yet."


def _active_since(session, cutoff: str) -> bool:
    """Could this conversation hold a question at or after ``cutoff``?

    Both sides are ISO-8601 in UTC, so comparing the strings orders them. A
    session carrying no timestamp at all is kept: it cannot be ruled out, and
    paying for one extra round trip beats under-reporting the period.
    """
    at = str(_field(session, "last_activity_at", "ended_at", "started_at"))
    return not at or at >= cutoff


def _dense_days(per_day: dict, days: int) -> list:
    """One entry per day in the window, zero-filled, oldest first."""
    today = datetime.now(timezone.utc).date()
    out = []
    for offset in range(days - 1, -1, -1):
        key = (today - timedelta(days=offset)).isoformat()
        out.append(per_day.get(key, {"day": key, "answered": 0, "unanswered": 0}))
    return out


@app.get("/api/dashboard/analytics")
async def dashboard_analytics(
    days: int = Query(default=7, ge=1, le=90),
    token: Optional[str] = Query(default=None),
) -> JSONResponse:
    """Usage of this widget: who asked what, when, and whether it could answer.

    Session listing gives no per-question detail, so each conversation in the
    window is fetched for its ``qas``. That is one call per session, and each
    costs about a second against a hosted tenant, so it is the slowest thing
    the dashboard asks for - the place to add a cache if the widget ever gets
    busy enough that a window's worth of conversations is itself a lot.

    "Unanswered" is exact, not inferred: the adapter returns one fixed string
    when recall finds nothing, so a question whose answer is that string is one
    the corpus could not serve. That is the single most actionable number here -
    it names the documentation gaps.
    """
    _require_dashboard(token)
    prefix = f"web:{DEMO_SITE_ID}:"
    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()

    # Conversations that cannot contribute to this window are dropped before
    # the fan-out rather than after it: a question is never newer than its
    # session's last activity, so a session that went quiet before the cutoff
    # has nothing inside it to count. Filtering afterwards spent a round trip
    # on every conversation the tenant had ever held, which made the default
    # seven-day view cost more every month regardless of the period asked for.
    #
    # The totals below are therefore the window's, matching the question counts
    # beside them rather than reporting lifetime figures under a "last N days"
    # heading.
    sessions = [
        x
        for x in await adapter.client.list_sessions()
        if str(_field(x, "session_id")).startswith(prefix) and _active_since(x, cutoff)
    ]

    details = await asyncio.gather(
        *(adapter.client.session_detail(str(_field(x, "session_id"))) for x in sessions)
    )

    visitors, per_day, questions = set(), {}, []
    answered = unanswered = 0

    for session, detail in zip(sessions, details):
        sid = str(_field(session, "session_id"))
        parts = sid.split(":")
        visitors.add(parts[2] if len(parts) > 2 else sid)
        for qa in detail.get("qas") or []:
            when = str(_field(qa, "time"))
            if when and when < cutoff:
                continue
            is_empty = _EMPTY_ANSWER_MARKER in str(_field(qa, "answer"))
            answered += 0 if is_empty else 1
            unanswered += 1 if is_empty else 0
            day = when[:10] or "unknown"
            bucket = per_day.setdefault(day, {"day": day, "answered": 0, "unanswered": 0})
            bucket["unanswered" if is_empty else "answered"] += 1
            questions.append(
                {
                    "question": str(_field(qa, "question"))[:200],
                    "time": when,
                    "answered": not is_empty,
                    "visitor": parts[2] if len(parts) > 2 else "",
                }
            )

    # Same question asked by different people is the signal worth ranking.
    counts: Counter = Counter(q["question"].strip().lower() for q in questions if q["question"])
    display = {}
    for q in questions:
        key = q["question"].strip().lower()
        display.setdefault(key, q["question"])

    return JSONResponse(
        {
            "days": days,
            "totals": {
                "conversations": len(sessions),
                "questions": len(questions),
                "visitors": len(visitors),
                "answered": answered,
                "unanswered": unanswered,
                "tokens_in": sum(int(_field(x, "tokens_in", default=0) or 0) for x in sessions),
                "tokens_out": sum(int(_field(x, "tokens_out", default=0) or 0) for x in sessions),
            },
            # Dense series: a day with no traffic is a zero, not a gap, or the
            # chart implies activity it did not have.
            "per_day": _dense_days(per_day, days),
            "top_questions": [
                {"question": display[k], "count": c} for k, c in counts.most_common(10)
            ],
            "recent": sorted(questions, key=lambda q: q["time"], reverse=True)[:10],
        }
    )


def _repositories(nodes: list) -> list:
    """The code repositories indexed into this dataset.

    cognee records each as a CodeRepository node carrying where it was cloned
    from and what the last load added, which is everything needed to say a
    repository is in without listing ten thousand symbols.
    """
    out = []
    for node in nodes:
        if not isinstance(node, dict) or str(node.get("type")) != "CodeRepository":
            continue
        props = node.get("properties") or {}
        delta = props.get("last_delta") or {}
        out.append(
            {
                "name": str(node.get("label") or "repository"),
                # /var/cognee/repos/github.com-topoteretes-cognee -> the origin,
                # as close to the URL that was typed as cognee keeps.
                "origin": str(props.get("path") or "").rsplit("/", 1)[-1],
                "loaded_at": str(delta.get("loaded_at") or ""),
                "nodes": delta.get("nodes_added"),
                "edges": delta.get("edges_added"),
            }
        )
    return sorted(out, key=lambda r: r["name"])


# The graph arrives whole or not at all: this deployment has no schema route,
# no cypher, and no filter on the graph endpoint, so asking which repositories
# are indexed costs the same 46MB and 24 seconds as counting every node. It is
# therefore fetched once, summarised, and kept - and warmed at startup, so the
# first person to open the dashboard is usually not the one paying for it.
_GRAPH_TTL_SECONDS = 900
_graph_cache: dict = {"summary": None, "at": 0.0, "dataset": None}
_graph_lock = asyncio.Lock()


def _summarise_graph(graph: dict) -> dict:
    nodes = graph.get("nodes") or []
    edges = graph.get("edges") or []
    node_types = Counter(str(_field(n, "type") or "unknown") for n in nodes)
    edge_labels = Counter(str(_field(e, "label") or "unknown") for e in edges)
    return {
        "node_total": len(nodes),
        "edge_total": len(edges),
        "node_types": [{"name": k, "count": v} for k, v in node_types.most_common()],
        # Edge labels have a long tail; the top ten carry the shape and the
        # rest is summarised rather than rendered as a forest of hairlines.
        "edge_labels": [{"name": k, "count": v} for k, v in edge_labels.most_common(10)],
        "edge_label_other": sum(c for _, c in edge_labels.most_common()[10:]),
        "edge_label_distinct": len(edge_labels),
        # Indexed repositories. They create no dataset items, so the corpus
        # table can never show one; this is the only place they exist.
        "repositories": _repositories(nodes),
    }


def _graph_cache_fresh(dataset_id: str) -> bool:
    return bool(
        _graph_cache["summary"]
        and _graph_cache["dataset"] == dataset_id
        and time.time() - _graph_cache["at"] < _GRAPH_TTL_SECONDS
    )


async def _graph_summary(dataset_id: str) -> dict:
    """The summary, from cache when it is warm.

    One fetch at a time: two dashboards opening together would otherwise pull
    46MB each to compute the same counts.
    """
    if _graph_cache_fresh(dataset_id):
        return dict(_graph_cache["summary"], cached=True)
    async with _graph_lock:
        if _graph_cache_fresh(dataset_id):
            return dict(_graph_cache["summary"], cached=True)
        summary = _summarise_graph(await adapter.client.graph(dataset_id))
        _graph_cache.update({"summary": summary, "at": time.time(), "dataset": dataset_id})
    return dict(summary, cached=False)


async def _warm_graph_cache() -> None:
    """Fill the cache in the background, so opening the page does not wait."""
    try:
        await _graph_summary(await _docs_dataset_id())
    except Exception as error:  # noqa: BLE001 - a warm cache is an optimisation
        print(f"[web_widget] graph warm-up skipped: {error}")


@app.get("/api/dashboard/repositories")
async def dashboard_repositories(token: Optional[str] = Query(default=None)) -> JSONResponse:
    """The indexed repositories, and only if answering is already cheap.

    Never triggers the fetch. A repository is one line on a page that paints in
    three seconds, so it either comes from the warm cache or it comes later with
    the breakdown - what it must not do is hold the page for 24 seconds, or
    report "none indexed" because the answer was not ready yet.
    """
    _require_dashboard(token)
    # No round trip at all, not even to resolve the dataset id: the cache is
    # dropped whenever the corpus changes, so a warm entry is this dataset's by
    # construction. Resolving it first cost a second against the tenant, on a
    # route whose whole reason to exist is being cheap enough to call on load.
    summary = _graph_cache["summary"]
    if not summary or time.time() - _graph_cache["at"] >= _GRAPH_TTL_SECONDS:
        return JSONResponse({"ready": False, "repositories": []})
    return JSONResponse({"ready": True, "repositories": summary["repositories"]})


@app.get("/api/dashboard/graph")
async def dashboard_graph(token: Optional[str] = Query(default=None)) -> JSONResponse:
    """What the knowledge graph is made of, as counts.

    The graph itself is 46MB for this corpus - 35k nodes and 116k edges - and a
    node-link rendering of that is an unreadable hairball, so the payload is
    aggregated here and the browser receives about a kilobyte. cognee Cloud's
    own graph canvas is the right tool for exploring the structure.

    Deliberately not part of /api/dashboard: the fetch takes seconds, and the
    page should paint without waiting for something most visits do not open.
    """
    _require_dashboard(token)
    summary = await _graph_summary(await _docs_dataset_id())
    return JSONResponse(
        {
            **summary,
        }
    )


# One upload is a file's path relative to the root the operator chose, plus its
# text. The browser reads both; this process opens nothing.
class UploadedFile(BaseModel):
    path: str
    text: str


class IngestRequest(BaseModel):
    files: list[UploadedFile]
    # The name of the folder that was chosen, when one was. Paths are relative
    # to it, so it is the only thing that tells two ingests of a "guides" folder
    # apart.
    root: str = ""


class RepoRequest(BaseModel):
    url: str


# A repository spec is resolved by cognee, not by this process, so "local path"
# here would mean a path on the tenant's own filesystem - useless to the
# operator and a way to point the tenant at its own disk. Only http(s) URLs
# leave this backend.
_REPO_URL = re.compile(r"^https?://[A-Za-z0-9.-]+(?::\d+)?/[\w.~%+-]+(?:/[\w.~%+-]+)+/?$")


def repo_label(url: str) -> str:
    """The owner/name a repository URL ends in, for tagging and for saying so."""
    path = url.split("://", 1)[-1].split("/", 1)[-1].rstrip("/")
    if path.endswith(".git"):
        path = path[: -len(".git")]
    parts = [p for p in path.split("/") if p]
    return "/".join(parts[-2:]) if len(parts) >= 2 else (parts[-1] if parts else "repository")


class ClearRequest(BaseModel):
    # Typing the dataset name is the confirmation. A checkbox is too easy to
    # click through for something that destroys a corpus that cost money to
    # build and takes an ingest run to restore.
    confirm: str


# Bounds on one request. The body is held in memory while it is parsed, and the
# page can send whatever the operator selected, so a mistyped folder should be
# refused rather than absorbed.
MAX_INGEST_FILES = 2000
MAX_INGEST_CHARS = 2_000_000
MAX_INGEST_TOTAL_CHARS = 50_000_000

# Uploads within one request go out together rather than one after another. Each
# is a round trip costing about a second, so a folder of 250 pages sent in series
# held a single request open for minutes. Four at a time: the tenant's throughput
# flattens well before a dozen, and a burst large enough to make it answer 503
# would cost more than it saves.
INGEST_CONCURRENCY = 4


def _safe_relative(path: str) -> Optional[str]:
    """``path`` as a corpus-safe relative path, or ``None`` if it is not one.

    It never opens a file - nothing here touches the filesystem - but it does
    become an item name and a citation path, so a value that escapes its root
    or carries separators cognee would not round-trip is refused rather than
    normalised into something the operator did not choose.
    """
    candidate = (path or "").strip().replace("\\", "/")
    if not candidate or candidate.startswith("/") or len(candidate) > 300:
        return None
    if "\x00" in candidate or "//" in candidate:
        return None
    parts = candidate.split("/")
    if any(p in ("", ".", "..") for p in parts):
        return None
    return candidate


def _safe_tag(value: str) -> str:
    """``value`` as a node-set tag, or empty if it is not usable as one."""
    tag = (value or "").strip().strip("/")
    if not tag or len(tag) > 120 or any(c < " " for c in tag):
        return ""
    return tag


def _node_set_for(relative: str, root: str) -> str:
    """The node set a file joins: the folder it came from, under the chosen root.

    One tag per folder, so a folder can be recalled on its own - cognee's recall
    takes these same values as ``node_name``. The root is part of the tag
    because paths are relative to it, and two ingests of some other project's
    ``guides`` folder would otherwise answer as one.

    Falls back to the site id for a lone file picked with no folder around it:
    every document should carry a tag, and an untagged one is invisible to every
    filtered recall rather than merely ungrouped.
    """
    folder = relative.rsplit("/", 1)[0] if "/" in relative else ""
    return "/".join([p for p in (root, folder) if p]) or DEMO_SITE_ID


@app.post("/api/dashboard/ingest")
async def dashboard_ingest(
    body: IngestRequest, token: Optional[str] = Query(default=None)
) -> JSONResponse:
    """Ingest files the operator picked in the browser.

    The page sends each file's text. This used to take a list of paths and read
    them here, which meant the dashboard could name any file this process had
    permission to read and have its contents uploaded to the tenant. The guard
    against that was a prefix check against one configured root - which also
    made ingesting anything else impossible.

    Reading in the browser answers both. The operator picks with the OS chooser,
    this process opens nothing, and a hosted backend - which can never see the
    operator's disk - works exactly the same way.

    Uploads are queued with run_in_background, so this returns once cognee has
    accepted them rather than once it has built the graph.
    """
    _require_dashboard(token)
    dataset = adapter.docs_dataset(DEMO_SITE_ID)
    root = _safe_tag(body.root)

    if len(body.files) > MAX_INGEST_FILES:
        raise HTTPException(
            status_code=413, detail=f"at most {MAX_INGEST_FILES} files in one ingest"
        )
    if sum(len(f.text) for f in body.files) > MAX_INGEST_TOTAL_CHARS:
        raise HTTPException(status_code=413, detail="that selection is too large for one ingest")

    queued, skipped, sendable = [], [], []
    claimed: dict = {}
    for upload in body.files:
        relative = _safe_relative(upload.path)
        if relative is None:
            skipped.append({"path": upload.path[:120], "why": "not a usable relative path"})
            continue
        if len(upload.text) > MAX_INGEST_CHARS:
            skipped.append({"path": relative, "why": "file is too large"})
            continue
        # A NUL says this was never text, whatever its extension claims. Storing
        # it would put bytes in the corpus that answer nothing and cost a
        # cognify run to find out.
        if "\x00" in upload.text:
            skipped.append({"path": relative, "why": "looks binary"})
            continue
        if not upload.text.strip():
            skipped.append({"path": relative, "why": "empty"})
            continue
        # Two paths differing only by extension flatten to one name, because
        # cognee strips a trailing extension to make it. Ingesting both would
        # put two items under one name, and nothing downstream could tell them
        # apart - so the second is refused rather than silently shadowing.
        name = item_name(relative)
        if name in claimed:
            skipped.append(
                {"path": relative, "why": f"name {name} already taken by {claimed[name]}"}
            )
            continue
        claimed[name] = relative
        sendable.append((relative, upload.text))

    limit = asyncio.Semaphore(INGEST_CONCURRENCY)

    async def send(relative: str, source: str) -> dict:
        name = item_name(relative)
        node_set = _node_set_for(relative, root)
        async with limit:
            ok = await adapter.client.remember_background(
                render_for_ingest(source, relative, DOCS_URL).encode("utf-8"),
                dataset_name=dataset,
                filename=upload_filename(relative),
                node_set=[node_set],
                # The name loses the extension, so the path it came from is
                # recorded where it survives intact.
                external_metadata={"widget_path": relative, "widget_root": root},
            )
        return (
            {"path": relative, "name": name, "node_set": node_set}
            if ok
            else {"path": relative, "why": "refused"}
        )

    for result in await asyncio.gather(*(send(r, t) for r, t in sendable)):
        (queued if "name" in result else skipped).append(result)
    _invalidate_viz_cache()
    return JSONResponse(
        {
            "queued": len(queued),
            "skipped": skipped,
            "dataset": dataset,
            # What was tagged, so the page can say which sets are now recallable
            # rather than leaving the operator to infer them from the folders.
            "node_sets": sorted({q["node_set"] for q in queued}),
        }
    )


@app.post("/api/dashboard/ingest-repo")
async def dashboard_ingest_repo(
    body: RepoRequest, token: Optional[str] = Query(default=None)
) -> JSONResponse:
    """Index a git repository as a code graph.

    Uploaded code files are stored as prose - chunks that happen to contain
    code, with no functions, classes or call edges in them. cognee builds a real
    code graph instead when it is given the repository itself, which is why this
    takes a URL rather than going through the picker: the route refuses uploads,
    because it clones and walks the repository on its own side.

    Only http(s) URLs are accepted. cognee resolves the spec, so a path here
    would be read from the tenant's filesystem rather than the operator's.
    """
    _require_dashboard(token)
    url = (body.url or "").strip()
    if not _REPO_URL.match(url):
        raise HTTPException(
            status_code=400,
            detail=(
                "Pass an https URL to a git repository, for example "
                "https://github.com/owner/name. A filesystem path would be read on "
                "the cognee server, not on this machine."
            ),
        )

    label = repo_label(url)
    dataset = adapter.docs_dataset(DEMO_SITE_ID)
    ok, why = await adapter.client.remember_repo(url, dataset_name=dataset)
    if not ok:
        raise HTTPException(status_code=502, detail=why)
    _invalidate_viz_cache()
    # A repository produces graph nodes and no dataset items, so nothing about
    # it will appear in the sources table. Saying where it does appear is the
    # difference between "indexed" and "apparently nothing happened".
    return JSONResponse({"repository": label, "url": url, "shows_in": "knowledge graph"})


# Both node sets are cognee's own, and they are the only thing telling the two
# kinds of writing in this dataset apart: the node type is TextDocument for both
# and the label is a content hash for both.
PERSISTED_SESSIONS_NODE_SET = "user_sessions_from_cache"
LESSONS_NODE_SET = "session_learnings"


def _conversation_memory(graph: dict) -> dict:
    """Persisted sessions, their turns, and the lessons distilled from them.

    The bridge does not invent node types. It cognifies each session like any
    other document and tags the result with cognee's own node set, so what comes
    back is a TextDocument per session and a DocumentChunk per piece of it -
    identified by the tag, not by a type name. Reading them apart is what makes
    the dataset worth having: a flat node list says nothing about which
    conversation taught what.

    Lessons are matched by name because distillation writes "standalone lesson
    documents" and this tenant has produced none yet to model against. Anything
    unmatched is counted rather than dropped, so a shape this does not know is
    visible instead of silently missing.
    """
    nodes = [n for n in (graph.get("nodes") or []) if isinstance(n, dict)]

    def tags(node) -> str:
        props = node.get("properties") or {}
        return str(props.get("source_node_set") or props.get("node_set") or "")

    def lesson_session(tag: str) -> str:
        """The conversation a lesson was distilled from.

        Distillation tags a lesson twice: ``session_learnings`` and
        ``session_learnings:<session id>``. The second answers "which
        conversation taught this", so it is read rather than treated as a
        duplicate of the first.
        """
        prefix = LESSONS_NODE_SET + ":"
        for part in tag.replace("'", " ").replace('"', " ").replace(",", " ").split():
            if part.startswith(prefix):
                return part[len(prefix) :].strip("[]")
        return ""

    turns, sessions, lessons, other = [], [], [], Counter()
    for node in nodes:
        kind = str(_field(node, "type") or "unknown")
        label = str(_field(node, "label"))[:200]
        props = node.get("properties") or {}
        entry = {
            "id": str(_field(node, "id")),
            "label": label,
            "type": kind,
            "text": str(props.get("text") or "")[:400],
            "at": str(props.get("created_at") or props.get("time") or ""),
            "session": "",
        }
        # Read off the first real distillation rather than guessed. Matching on
        # the type or the label - which is what this did first - filed every
        # lesson as a persisted session, because both are TextDocuments named
        # after their content hash.
        tag = tags(node)
        if LESSONS_NODE_SET in tag:
            entry["session"] = lesson_session(tag)
            if kind == "TextDocument":
                lessons.append(entry)
            else:
                other[kind] += 1  # chunks and entities derived from a lesson
        elif PERSISTED_SESSIONS_NODE_SET not in tag:
            other[kind] += 1
        elif kind == "TextDocument":
            sessions.append(entry)
        elif kind == "DocumentChunk":
            turns.append(entry)
        else:
            # Entities and summaries cognify derives from a persisted session:
            # real content, but not one of the three things this view is about.
            other[kind] += 1

    return {
        "sessions": sessions[:200],
        "turns": turns[:500],
        "lessons": lessons[:200],
        "counts": {
            "sessions": len(sessions),
            "turns": len(turns),
            "lessons": len(lessons),
        },
        "unclassified": [{"type": k, "count": v} for k, v in other.most_common(10)],
    }


# cognee's own format for a persisted turn, written by the session bridge:
#   Session ID: web:demo:visitor:conv
#
#   Question: ...
#
#   Answer: ...
_PERSISTED_TURN = re.compile(
    r"Session ID:\s*(?P<session>\S+)\s*"
    r"Question:\s*(?P<question>.*?)\s*"
    r"Answer:\s*(?P<answer>.*)\s*\Z",
    re.S,
)


def _documents_from_graph(graph: dict) -> list:
    """Rebuild each stored document from its chunks, in order.

    A document is split across DocumentChunks, each carrying document_id and
    chunk_index, so joining by the first and sorting by the second reassembles
    the text exactly - and costs one graph read rather than a round trip per
    document, which is what listing them item by item would have cost.
    """
    by_document: dict = {}
    for node in graph.get("nodes") or []:
        if not isinstance(node, dict) or str(node.get("type")) != "DocumentChunk":
            continue
        props = node.get("properties") or {}
        key = str(props.get("document_id") or node.get("id"))
        by_document.setdefault(key, []).append(
            (int(props.get("chunk_index") or 0), str(props.get("text") or ""))
        )
    return ["".join(text for _, text in sorted(parts)) for parts in by_document.values()]


def _conversations_from_documents(documents: list) -> list:
    """Group persisted turns into conversations, newest first.

    The session id travels inside the stored text, which is what makes this
    exact: turns are grouped by who asked them, not by when they happened near
    each other. Correlating by time was the alternative, and on this tenant it
    attributed one visitor's questions to another's conversation.
    """
    conversations: dict = {}
    for document in documents:
        match = _PERSISTED_TURN.search(document)
        if not match:
            continue
        session = match.group("session").strip()
        turn = {
            "question": match.group("question").strip()[:2000],
            "answer": match.group("answer").strip()[:4000],
        }
        conversations.setdefault(session, []).append(turn)
    return [
        {
            "session_id": session,
            # web:demo:visitor-x:conv-y -> the identifying half
            "visitor": session.split(":")[2] if len(session.split(":")) > 2 else session,
            "turns": turns,
        }
        for session, turns in conversations.items()
    ]


@app.get("/api/dashboard/conversations")
async def dashboard_conversations(token: Optional[str] = Query(default=None)) -> JSONResponse:
    """Conversations rebuilt from what was actually stored.

    Not from the session cache, which is where this list used to come from. That
    cache holds an index whose entries outlive their own contents: a dataset
    delete drops every turn it holds while leaving the session rows and their
    token totals behind, so conversations with real traffic showed as empty and
    there was no way to tell them from ones that never had a turn.

    This shows a conversation when the turns exist to show. Nothing appears
    until the bridge has persisted it, and what appears can be read.
    """
    _require_dashboard(token)
    dataset = adapter.conversations_dataset(DEMO_SITE_ID)
    datasets = await adapter.client.list_datasets()
    match = next((d for d in datasets if isinstance(d, dict) and d.get("name") == dataset), None)
    if not match:
        return JSONResponse({"dataset": dataset, "exists": False, "conversations": []})
    graph = await adapter.client.graph(str(match.get("id")))
    conversations = _conversations_from_documents(_documents_from_graph(graph))
    return JSONResponse({"dataset": dataset, "exists": True, "conversations": conversations})


@app.get("/api/dashboard/conversation-memory")
async def dashboard_conversation_memory(
    token: Optional[str] = Query(default=None),
) -> JSONResponse:
    """What has been persisted and distilled from conversations so far."""
    _require_dashboard(token)
    dataset = adapter.conversations_dataset(DEMO_SITE_ID)
    datasets = await adapter.client.list_datasets()
    match = next((d for d in datasets if isinstance(d, dict) and d.get("name") == dataset), None)
    if not match:
        # Nothing has been stored yet: the dataset is created by the first
        # exchange, so its absence is a normal empty state, not an error.
        return JSONResponse({"dataset": dataset, "exists": False, "item_count": 0})
    dataset_id = str(match.get("id"))
    items, graph = await asyncio.gather(
        adapter.client.dataset_data(dataset_id),
        adapter.client.graph(dataset_id),
    )
    return JSONResponse(
        {
            "dataset": dataset,
            "exists": True,
            "item_count": len(items),
            **_conversation_memory(graph),
        }
    )


@app.post("/api/dashboard/clear-conversations")
async def dashboard_clear_conversations(
    body: ClearRequest, token: Optional[str] = Query(default=None)
) -> JSONResponse:
    """Delete the conversations corpus: stored exchanges, sessions, lessons.

    Separate from clearing the docs corpus, and gated the same way, because it
    holds other people's words rather than our own documents - the one thing
    here that cannot be rebuilt by ingesting a folder again.

    Deleting it also invalidates the sessions attributed to it, which is how
    cognee keeps session memory from quoting deleted content.
    """
    _require_dashboard(token)
    dataset = adapter.conversations_dataset(DEMO_SITE_ID)
    if body.confirm != dataset:
        raise HTTPException(status_code=400, detail=f"type the dataset name exactly: {dataset}")
    if not await adapter.client.forget_dataset(dataset):
        raise HTTPException(status_code=502, detail="cognee refused to clear the dataset")
    return JSONResponse({"cleared": dataset})


@app.get("/api/dashboard/ingest-progress")
async def dashboard_ingest_progress(
    token: Optional[str] = Query(default=None),
    pipeline: str = Query(default="cognify_pipeline"),
) -> JSONResponse:
    """How far cognee has got building the graph for what was ingested.

    An upload returns once cognee has accepted it; the cognify that follows runs
    for minutes with nothing to show for it, which is why an ingest used to end
    at "queued" and go quiet. The corpus count is reported alongside the
    pipeline's own figures because it moves even before the first progress tick.
    """
    _require_dashboard(token)
    dataset_id = await _docs_dataset_id()
    # A repository is indexed by code_graph_pipeline, ordinary files by
    # cognify_pipeline, and a caller watching the wrong one sees a run that
    # never starts.
    if pipeline not in ("cognify_pipeline", "code_graph_pipeline"):
        raise HTTPException(status_code=400, detail="unknown pipeline")
    state, items = await asyncio.gather(
        adapter.client.dataset_progress(dataset_id, pipeline),
        adapter.client.dataset_data(dataset_id),
    )
    progress = state.get("progress") or {}
    return JSONResponse(
        {
            "status": str(state.get("status") or "unknown"),
            # False when this cognee has no progress endpoint at all, which is
            # not the same as a run that has not reported numbers yet - the page
            # should stop waiting for counts rather than wait forever.
            "counts_available": bool(state.get("counts_available")),
            "completed_items": progress.get("completed_items"),
            "total_items": progress.get("total_items"),
            "current_stage": progress.get("current_stage"),
            "item_count": len(items),
        }
    )


@app.post("/api/dashboard/clear")
async def dashboard_clear(
    body: ClearRequest, token: Optional[str] = Query(default=None)
) -> JSONResponse:
    """Ask cognee to delete the widget's dataset and everything in it.

    Scoped to the widget's own dataset, and gated on the caller typing that
    dataset's name. The next ingest recreates it.

    cognee accepts the delete and drains the dataset behind it - a 255-item
    corpus empties over several minutes - so the count here is what was queued,
    not what is gone. It is read before the call for that reason, and named
    ``items_queued``: reporting it as removed claimed a finish that had not
    happened, and anything rendered from the next read showed a half-emptied
    corpus, which reads as the clear having failed.
    """
    _require_dashboard(token)
    dataset = adapter.docs_dataset(DEMO_SITE_ID)
    if body.confirm != dataset:
        raise HTTPException(status_code=400, detail=f"type the dataset name exactly: {dataset}")
    items = await adapter.client.dataset_data(await _docs_dataset_id())
    if not await adapter.client.forget_dataset(dataset):
        raise HTTPException(status_code=502, detail="cognee refused to clear the dataset")
    _invalidate_viz_cache()
    return JSONResponse({"cleared": dataset, "items_queued": len(items)})


@app.post("/api/dashboard/data/{data_id}/reingest")
async def dashboard_reingest(
    data_id: str, token: Optional[str] = Query(default=None)
) -> JSONResponse:
    """Re-ingest one item: delete it, then add its bytes back.

    cognee has no per-item refresh. Re-uploading alongside the original is a
    no-op — identical content is deduplicated, so nothing is rebuilt and no
    timestamp moves — which means the only honest way to force a fresh pass is
    to remove the item first.

    What this does NOT do is re-read the page the item came from. source_uri
    points into whatever filesystem performed the original ingest, not one this
    backend can see, so this replays the stored bytes. It is useful for
    rebuilding an item under changed cognify settings; it will not pick up an
    edit made to the underlying documentation.

    The bytes are fetched before anything is destroyed. If the re-upload fails
    afterwards the item is genuinely gone, and the response says so rather than
    reporting a success that lost data.
    """
    _require_dashboard(token)
    dataset_id = await _docs_dataset_id()

    items = await adapter.client.dataset_data(dataset_id)
    item = next((i for i in items if str(_field(i, "id")) == data_id), None)
    if not item:
        raise HTTPException(status_code=404, detail="item not found in the docs dataset")

    raw = await adapter.client.fetch_raw(dataset_id=dataset_id, data_id=data_id)
    if raw is None:
        # Nothing has been touched yet, so this is a clean refusal.
        raise HTTPException(
            status_code=502, detail="could not read the stored copy; nothing changed"
        )

    name = str(_field(item, "name")) or "document"
    extension = str(_field(item, "extension")) or "txt"
    filename = name if name.endswith(f".{extension}") else f"{name}.{extension}"
    content_type = str(_field(item, "mimeType")) or "text/plain"

    if not await adapter.client.delete_data(dataset_id=dataset_id, data_id=data_id):
        raise HTTPException(status_code=502, detail="cognee refused the delete; nothing changed")

    try:
        await adapter.client.remember_bytes(
            raw,
            dataset_name=adapter.docs_dataset(DEMO_SITE_ID),
            filename=filename,
            content_type=content_type,
        )
    except Exception as error:  # noqa: BLE001 - the item is already gone; say so
        raise HTTPException(
            status_code=500,
            detail=(
                f"'{name}' was removed but could not be re-added ({error}). "
                "Its content is no longer in the corpus."
            ),
        ) from error

    _invalidate_viz_cache()
    return JSONResponse({"reingested": data_id, "name": name, "bytes": len(raw)})


@app.delete("/api/dashboard/data/{data_id}")
async def dashboard_delete_data(
    data_id: str, token: Optional[str] = Query(default=None)
) -> JSONResponse:
    """Permanently remove one ingested item from the docs corpus.

    Scoped to the widget's own dataset on purpose: the key can delete from any
    dataset it can write, and this dashboard should not be a way to reach the
    rest of the tenant.
    """
    _require_dashboard(token)
    dataset_id = await _docs_dataset_id()
    ok = await adapter.client.delete_data(dataset_id=dataset_id, data_id=data_id)
    if not ok:
        raise HTTPException(status_code=502, detail="cognee refused the delete")
    _invalidate_viz_cache()
    return JSONResponse({"deleted": data_id})


@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(token: Optional[str] = Query(default=None)) -> HTMLResponse:
    """The operator page.

    A shell only: it re-fetches /api/dashboard and the session routes from the
    browser, so acting on a row (deleting, opening a conversation) refreshes in
    place rather than reloading a server-rendered snapshot. The token travels in
    the query string it was opened with, so it is never stored by the page.
    """
    _require_dashboard(token)
    return HTMLResponse((STATIC_DIR / "dashboard.html").read_text(encoding="utf-8"))


def main() -> None:
    import uvicorn

    host = os.getenv("WEB_HOST", "127.0.0.1")
    port = int(os.getenv("WEB_PORT", "8000"))
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
