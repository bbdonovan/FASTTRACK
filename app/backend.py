# johny5_agentic/app/backend.py
"""
Backend service layer for Johny 5 Agentic.

This module provides:
- SQLite-backed storage (j5_agentic.db).
- Simple ingestion helpers (profiles, candles, news).
- Knowledge graph tables (kg_nodes, kg_edges).
- A single agentic entrypoint: llm_analyze(question).

In a real deployment at work, llm_analyze is where you:
- Call your internal LLM / LangGraph agent.
- Orchestrate tools over profiles, candles, news, and KG.
- Return both the final answer and a provenance trace that shows
  which data sources were used to produce the answer.
"""

from __future__ import annotations

import logging
import pathlib
import sqlite3
import re
from typing import Any, Dict, List, Optional

from email.utils import parsedate_to_datetime
from xml.etree import ElementTree as ET

import os
import json
import textwrap
import requests 

logger = logging.getLogger(__name__)

# Basic stopwords so we don't treat every word as a ticker.
_TICKER_STOPWORDS = {
    "WHAT",
    "DOES",
    "THE",
    "LLM",
    "KNOW",
    "ABOUT",
    "CURRENTLY",
    "BASED",
    "INFORMATION",
    "YOU",
    "HAVE",
    "HOW",
    "DO",
    "GO",
    "IS",
    "ARE",
    "AND",
    "OR",
    "FOR",
    "WITH",
    "THIS",
    "THAT",
}

# ---------------------------------------------------------------------------
# Ollama Setup
# ---------------------------------------------------------------------------
def _get_ollama_settings() -> tuple[str, str]:
    """
    Read Ollama connection settings from environment.

    Environment variables:
        J5_OLLAMA_BASE_URL  (default: http://127.0.0.1:11434)
        J5_OLLAMA_MODEL     (default: llama3:latest)

    Returns:
        (base_url, model)
    """
    base = os.getenv("J5_OLLAMA_BASE_URL", "http://127.0.0.1:11434")
    model = os.getenv("J5_OLLAMA_MODEL", "llama3:latest")
    return base.rstrip("/"), model


def _ollama_generate(prompt: str, temperature: float = 0.2) -> str:
    """
    Call a local Ollama model to get a completion for the given prompt.

    This uses Ollama's /api/generate endpoint with stream=False so we get
    a single JSON object back.

    If the call fails, we return a clear error message instead of faking
    an answer.
    """
    base_url, model = _get_ollama_settings()
    url = f"{base_url}/api/generate"

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
        },
    }

    try:
        resp = requests.post(url, json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        return (data.get("response") or "").strip() or "LLM returned an empty response."
    except Exception as exc:  # noqa: BLE001
        # We do NOT fabricate an answer here – just report the error.
        return f"LLM call via Ollama failed: {exc}"
    
def _build_agentic_prompt_for_llm(
    question: str,
    tickers: list[str],
    profiles: list[dict],
    metrics: list[dict],
    articles: list[dict],
    geo_notes: list[dict] | object,
    graph_neighborhood: dict | None,
) -> str:
    """
    Turn structured tool outputs into a single prompt string for the LLM.

    The agentic part (tool orchestration) happens in Python; the LLM's job
    is to reason over this context and write a clear answer.
    """
    parts: list[str] = []

    parts.append(
        "You are Johny 5, an equity + geopolitics analyst working inside a bank.\n"
        "You receive a user question and a set of structured data sources.\n"
        "Use ONLY this context to answer. If something is unknown, say so explicitly.\n"
    )

    parts.append("QUESTION:")
    parts.append(question.strip())
    parts.append("")

    parts.append("TICKERS DETECTED:")
    parts.append(", ".join(tickers) if tickers else "(none)")
    parts.append("")

    if profiles:
        parts.append("COMPANY PROFILES (subset):")
        for p in profiles[:3]:
            parts.append(
                f"- {p.get('ticker', '?')}: {p.get('name', '(no name)')} "
                f"(sector={p.get('sector', '?')}, industry={p.get('industry', '?')})"
            )
        parts.append("")

    if metrics:
        parts.append("EXTERNAL METRICS (subset):")
        for m in metrics[:3]:
            parts.append(
                f"- {m.get('ticker', '?')}: "
                f"PE={m.get('pe_ratio', '?')}, "
                f"mkt_cap_usd_billion={m.get('market_cap_usd_billion', '?')}, "
                f"1y_volatility={m.get('one_year_volatility', '?')}"
            )
        parts.append("")

    if articles:
        parts.append("NEWS / FILINGS (subset):")
        for a in articles[:5]:
            parts.append(
                f"- [{a.get('published_at', '?')}] "
                f"{a.get('ticker', a.get('symbol', '?'))}: "
                f"{a.get('title', '(no title)')}"
            )
        parts.append("")

    geo_list = _normalize_geo_notes(geo_notes)
    if geo_list:
        parts.append("GEOPOLITICAL / SANCTIONS INTEL NOTES (subset):")
        for g in geo_list[:6]:
            snippet = (g.get("snippet") or "").replace("\n", " ")
            if len(snippet) > 260:
                snippet = snippet[:260] + "..."
            parts.append(
                f"- [{g.get('region', 'Global')}] {g.get('title', '(no title)')}: "
                f"{snippet}"
            )
        parts.append("")
    if graph_neighborhood and graph_neighborhood.get("center"):
        center = graph_neighborhood["center"]
        parts.append(
            "GRAPH NEIGHBORHOOD SUMMARY "
            f"(center={center.get('key')} · {center.get('label')} · "
            f"type={center.get('node_type')}):"
        )
        parts.append(
            f"- Neighbors: {len(graph_neighborhood.get('neighbors', []))}; "
            f"Edges: {len(graph_neighborhood.get('edges', []))}"
        )
        parts.append("")

    parts.append(
        textwrap.dedent(
            """
            INSTRUCTIONS FOR YOUR ANSWER:

            - Write a concise, professional answer (2–4 short paragraphs).
            - Explicitly tie your reasoning back to the data above
              (profiles, metrics, news, intel notes, and graph structure).
            - Highlight risk/exposure angles and, where relevant, how China vs U.S.
              constraints show up in this data.
            - If key information is missing, call that out instead of guessing.
            - At the end, add a short bullet list: "Key Points" with 3–6 bullets.
            """
        ).strip()
    )

    return "\n".join(parts)
# ---------------------------------------------------------------------------
# Paths & DB helpers
# ---------------------------------------------------------------------------

# 👈 THIS is the constant dashboard.py wants to import
BASE_DIR = pathlib.Path(__file__).resolve().parents[1]
DB_PATH = BASE_DIR / "j5_agentic.db"


def _ensure_db() -> None:
    """
    Ensure the SQLite database file and required tables exist.

    This function is idempotent. It will:
    - Create the DB file if it does not exist.
    - Create core tables (companies, articles, kg_nodes, kg_edges, candles)
      if they do not already exist.
    """
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS companies (
            id      INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker  TEXT UNIQUE,
            name    TEXT
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS articles (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker       TEXT,
            title        TEXT,
            published_at TEXT
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS kg_nodes (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            key       TEXT UNIQUE,
            label     TEXT,
            node_type TEXT
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS kg_edges (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            src       TEXT,
            dst       TEXT,
            edge_type TEXT
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS candles (
            id     INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT,
            dt     TEXT,
            open   REAL,
            high   REAL,
            low    REAL,
            close  REAL,
            volume INTEGER
        )
        """
    )
    # Geopolitics / sanctions / macro intelligence notes
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS intel_articles (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            topic        TEXT,    -- e.g. 'china_chip_sanctions', 'brics_dollar'
            region       TEXT,    -- e.g. 'CN', 'US', 'Global'
            entities     TEXT,    -- comma-separated list of key actors (companies, people, states)
            source       TEXT,    -- e.g. 'SyntheticDemo', 'InternalReport'
            published_at TEXT,
            title        TEXT NOT NULL,
            snippet      TEXT
        )
        """
    )

    conn.commit()
    conn.close()

def _detect_tickers_from_text(text: str) -> List[str]:
    """
    Detect valid tickers mentioned in free text by intersecting tokens with the
    tickers present in the companies table.

    This prevents us from treating every capitalized word as a ticker.
    """
    _ensure_db()

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute("SELECT ticker FROM companies")
    rows = cur.fetchall()
    conn.close()

    known = {r["ticker"].upper() for r in rows}

    # Extract alphabetic tokens up to 5 chars, in uppercase.
    tokens = re.findall(r"[A-Za-z]{1,5}", (text or "").upper())

    seen: set[str] = set()
    tickers: List[str] = []
    for tok in tokens:
        if tok in known and tok not in seen:
            seen.add(tok)
            tickers.append(tok)

    # Simple synonym mapping for common names → tickers.
    upper_text = (text or "").upper()
    synonyms = [
        ("NVIDIA", "NVDA"),
        ("APPLE", "AAPL"),
        ("AMAZON", "AMZN"),
        ("MICROSOFT", "MSFT"),
    ]
    for name, ticker in synonyms:
        if name in upper_text and ticker in known and ticker not in seen:
            seen.add(ticker)
            tickers.append(ticker)

    return tickers

def _extract_candidate_tickers(question: str) -> list[str]:
    """
    Heuristic ticker detector.

    - Finds 2–6 letter alphabetic tokens in the question.
    - Uppercases them.
    - Drops obvious non-ticker stopwords.
    - Deduplicates while preserving order.

    Downstream lookups (companies_lookup) will further filter out
    anything that isn't in the profiles database.
    """
    import re

    if not question:
        return []

    tokens = re.findall(r"[A-Za-z]{2,6}", question.upper())
    seen: list[str] = []
    for token in tokens:
        if token in _TICKER_STOPWORDS:
            continue
        if token not in seen:
            seen.append(token)
    return seen

def get_stats() -> Dict[str, int]:
    """
    Return high-level counts used by the dashboard cards.

    The dictionary includes:
        companies: number of rows in the companies table.
        articles:  number of rows in the articles table.
        kg_nodes:  number of rows in the kg_nodes table.
        kg_edges:  number of rows in the kg_edges table.
        candles:   number of rows in the candles table.

    Returns:
        A dictionary of table names to row counts.
    """
    _ensure_db()

    stats = {
        "companies": 0,
        "articles": 0,
        "kg_nodes": 0,
        "kg_edges": 0,
        "candles": 0,
    }

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    mapping = [
        ("companies", "companies"),
        ("articles", "articles"),
        ("kg_nodes", "kg_nodes"),
        ("kg_edges", "kg_edges"),
        ("candles", "candles"),
    ]

    for key, table in mapping:
        try:
            cur.execute(f"SELECT COUNT(1) FROM {table}")
            row = cur.fetchone()
            stats[key] = int(row[0]) if row and row[0] is not None else 0
        except sqlite3.Error:
            # Table might not exist yet; that's fine in a fresh DB.
            continue

    conn.close()
    return stats


# ---------------------------------------------------------------------------
# Fallback ingestion implementations (simple demo data)
# ---------------------------------------------------------------------------

def pull_profiles(limit: int = 50) -> Dict[str, Any]:
    """
    Pull or create company profiles.

    For the prototype:
    - Seeds a tiny demo universe into the companies table.

    Args:
        limit: Maximum number of demo companies to insert.

    Returns:
        A dictionary describing what was done.
    """
    logger.info("Using demo pull_profiles implementation")
    _ensure_db()

    demo_companies = [
        ("AAPL", "Apple Inc."),
        ("MSFT", "Microsoft Corporation"),
        ("NVDA", "NVIDIA Corporation"),
        ("AMZN", "Amazon.com, Inc."),
        ("TSLA", "Tesla, Inc."),
        ("META", "Meta Platforms, Inc."),
        ("GOOGL", "Alphabet Inc. (Class A)"),
    ][:limit]

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    inserted = 0
    for ticker, name in demo_companies:
        try:
            cur.execute(
                "INSERT OR IGNORE INTO companies (ticker, name) VALUES (?, ?)",
                (ticker, name),
            )
            if cur.rowcount:
                inserted += 1
        except sqlite3.Error as exc:  # noqa: BLE001
            logger.warning("Failed to insert company %s: %s", ticker, exc)

    conn.commit()
    conn.close()

    return {"inserted": inserted, "total_demo": len(demo_companies)}

def _lookup_companies_by_tickers(tickers: List[str]) -> List[Dict[str, Any]]:
    """
    Helper: fetch company rows for a list of tickers from the companies table.

    Args:
        tickers: List of ticker symbols (e.g. ["AAPL", "NVDA"]).

    Returns:
        A list of dictionaries with basic profile information for each ticker
        that exists in the table.
    """
    _ensure_db()

    if not tickers:
        return []

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    # Build a simple IN clause like (..., ?, ?, ...)
    placeholders = ",".join("?" for _ in tickers)
    sql = f"""
        SELECT ticker, name
        FROM companies
        WHERE ticker IN ({placeholders})
        ORDER BY ticker
    """
    cur.execute(sql, [t.upper() for t in tickers])
    rows = cur.fetchall()
    conn.close()

    return [
        {
            "ticker": r["ticker"],
            "name": r["name"],
        }
        for r in rows
    ]

def build_graph_from_profiles() -> Dict[str, Any]:
    """
    Build a simple knowledge graph from the company profiles.

    Behavior:
    - Create a MARKET hub node.
    - Create one KG node per company.
    - Create a member_of edge from MARKET to each company node.

    Returns:
        A dictionary summarizing how many nodes and edges were added.
    """
    logger.info("Using demo build_graph_from_profiles implementation")
    _ensure_db()

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.execute("SELECT ticker, name FROM companies")
    companies = cur.fetchall()

    # Upsert MARKET node
    cur.execute(
        """
        INSERT OR IGNORE INTO kg_nodes (key, label, node_type)
        VALUES (?, ?, ?)
        """,
        ("MARKET", "Equity Market", "hub"),
    )

    nodes_added = 0
    edges_added = 0

    for ticker, name in companies:
        key = f"COMP::{ticker}"
        cur.execute(
            """
            INSERT OR IGNORE INTO kg_nodes (key, label, node_type)
            VALUES (?, ?, ?)
            """,
            (key, name, "company"),
        )
        if cur.rowcount:
            nodes_added += 1

        # Edge MARKET -> company
        cur.execute(
            """
            INSERT INTO kg_edges (src, dst, edge_type)
            VALUES (?, ?, ?)
            """,
            ("MARKET", key, "member_of"),
        )
        edges_added += 1

    conn.commit()
    conn.close()

    return {
        "nodes_added": nodes_added,
        "edges_added": edges_added,
        "companies_seen": len(companies),
    }


def pull_candles_for_tickers(tickers: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Pull OHLCV candles for tickers (demo seeding).

    For the prototype, this function:
    - Seeds a tiny synthetic OHLCV series per ticker.

    Args:
        tickers: Optional list of tickers to seed. If None, all companies
                 in the DB will be used.

    Returns:
        A dictionary summarizing what was inserted.
    """
    logger.info("Using demo pull_candles_for_tickers implementation")
    _ensure_db()

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    if not tickers:
        cur.execute("SELECT ticker FROM companies")
        tickers = [row[0] for row in cur.fetchall()]

    inserted = 0
    for ticker in tickers:
        demo_rows = [
            ("2025-01-01", 100.0, 105.0, 99.0, 104.0, 1_000_000),
            ("2025-01-02", 104.0, 106.0, 103.0, 105.5, 900_000),
            ("2025-01-03", 105.5, 107.0, 104.0, 106.0, 950_000),
        ]
        for dt, o, h, l, c, v in demo_rows:
            cur.execute(
                """
                INSERT INTO candles (ticker, dt, open, high, low, close, volume)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (ticker, dt, o, h, l, c, v),
            )
            inserted += 1

    conn.commit()
    conn.close()

    return {"inserted": inserted, "tickers": tickers}


def pull_sector(sector: str) -> Dict[str, Any]:
    """
    Dummy sector pull.

    In a real system you might:
    - Resolve a sector ETF (e.g. XLE) into a list of component tickers.
    - Insert or update those tickers into the companies table.

    Args:
        sector: Sector code (e.g. "XLE").

    Returns:
        A dictionary describing the action taken.
    """
    logger.info("Demo pull_sector called for sector=%s", sector)
    return {"sector": sector, "note": "demo implementation; no-op"}


def ingest_news(max_per_ticker: int = 5) -> Dict[str, Any]:
    """
    Ingest news articles tied to companies via Yahoo Finance RSS, with fallback,
    and project them into the knowledge graph as NEWS nodes.

    For each company ticker in the DB, this function:
        - Attempts to fetch the Yahoo Finance RSS feed for that ticker.
        - If successful:
            * Clears existing rows in the articles table for that ticker.
            * Clears existing KG NEWS nodes and has_news edges for that ticker.
            * Inserts up to `max_per_ticker` of the latest headlines into:
                - articles (ticker, title, published_at)
                - kg_nodes as NEWS::<TICKER>::N
                - kg_edges as COMP::<TICKER> --has_news--> NEWS::<TICKER>::N
        - If the request fails (429 rate limit, network issue, etc.):
            * Keeps existing rows if any.
            * If there are no existing rows, inserts a single placeholder
              article explaining that news is unavailable.

    This keeps the demo resilient and enriches the graph, so the graph
    neighborhood for COMP::<TICKER> includes its recent news items.

    Args:
        max_per_ticker: Maximum headlines to store per ticker.

    Returns:
        A dictionary summarizing inserted rows and any errors:
            {
              "tickers": [...],
              "inserted_total": int,
              "errors": {ticker: "error message", ...},
              "source": "yahoo-finance-rss+fallback+kg-news"
            }
    """
    logger.info("Ingesting news from Yahoo Finance RSS (with fallback + KG news)")
    _ensure_db()

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute("SELECT ticker FROM companies")
    tickers = [row["ticker"] for row in cur.fetchall()]

    total_inserted = 0
    errors: Dict[str, str] = {}

    for ticker in tickers:
        symbol = ticker.upper()
        url = (
            "https://feeds.finance.yahoo.com/rss/2.0/headline"
            f"?s={symbol}&region=US&lang=en-US"
        )

        inserted_for_symbol = 0

        try:
            resp = requests.get(
                url,
                timeout=5.0,
                headers={"User-Agent": "Johny5-Agentic-Demo/1.0"},
            )
            resp.raise_for_status()
            xml_text = resp.text
            root = ET.fromstring(xml_text)
            channel = root.find("channel")
            items = channel.findall("item") if channel is not None else []
        except Exception as exc:  # noqa: BLE001
            msg = f"Failed to fetch RSS for {symbol}: {exc}"
            logger.warning(msg)
            errors[symbol] = str(exc)
            items = []

        if items:
            # We have real RSS items → overwrite previous news for this ticker.
            cur.execute("DELETE FROM articles WHERE ticker = ?", (symbol,))

            # Also clear KG NEWS nodes and has_news edges for this ticker.
            comp_key = f"COMP::{symbol}"
            cur.execute(
                "DELETE FROM kg_edges WHERE src = ? AND edge_type = 'has_news'",
                (comp_key,),
            )
            cur.execute(
                "DELETE FROM kg_nodes WHERE key LIKE ?",
                (f"NEWS::{symbol}::%",),
            )

            for idx, item in enumerate(items[:max_per_ticker]):
                title_el = item.find("title")
                date_el = item.find("pubDate")

                if title_el is None:
                    continue

                title = (title_el.text or "").strip()
                raw_pub = (date_el.text or "").strip() if date_el is not None else ""

                published_at = raw_pub
                if raw_pub:
                    try:
                        published_at = parsedate_to_datetime(raw_pub).isoformat()
                    except Exception:  # noqa: BLE001
                        pass

                # Store in articles table.
                cur.execute(
                    """
                    INSERT INTO articles (ticker, title, published_at)
                    VALUES (?, ?, ?)
                    """,
                    (symbol, title, published_at),
                )
                inserted_for_symbol += 1

                # Project into KG as a NEWS node + edge from company.
                article_key = f"NEWS::{symbol}::{idx}"
                cur.execute(
                    """
                    INSERT INTO kg_nodes (key, label, node_type)
                    VALUES (?, ?, ?)
                    """,
                    (article_key, title, "news"),
                )
                cur.execute(
                    """
                    INSERT INTO kg_edges (src, dst, edge_type)
                    VALUES (?, ?, ?)
                    """,
                    (comp_key, article_key, "has_news"),
                )

            logger.info("Inserted %s real headlines for %s", inserted_for_symbol, symbol)
        else:
            # No items (rate limit, network, or empty feed).
            # Check if this ticker already has any rows.
            cur.execute(
                "SELECT COUNT(1) AS cnt FROM articles WHERE ticker = ?",
                (symbol,),
            )
            row = cur.fetchone()
            existing = int(row["cnt"]) if row and row["cnt"] is not None else 0

            if existing == 0:
                # Insert a single placeholder article so the UI and agent trace
                # still have something to show.
                placeholder_title = (
                    f"Demo placeholder: news unavailable for {symbol} "
                    "(likely rate-limited or offline)"
                )
                cur.execute(
                    """
                    INSERT INTO articles (ticker, title, published_at)
                    VALUES (?, ?, datetime('now'))
                    """,
                    (symbol, placeholder_title),
                )
                inserted_for_symbol += 1
                logger.info(
                    "Inserted placeholder headline for %s (no real RSS items)",
                    symbol,
                )

        conn.commit()
        total_inserted += inserted_for_symbol

    conn.close()

    return {
        "tickers": tickers,
        "inserted_total": total_inserted,
        "errors": errors,
        "source": "yahoo-finance-rss+fallback+kg-news",
    }

def fetch_external_metrics(ticker: str) -> Dict[str, Any]:
    """
    Placeholder for an external API or microservice call.

    In your real environment this is where you would:
        - Call an internal REST/gRPC service (e.g. http://internal-api/metrics)
        - Or call a vendor API
        - Parse the JSON and normalize it into a stable shape.

    For the demo, we synthesize some metrics so that the agent trace
    shows a distinct "external metrics" tool.

    Args:
        ticker: Company ticker, e.g. "NVDA".

    Returns:
        A dictionary of synthetic metrics, including a 'source' field
        that makes it clear this came from an external system placeholder.
    """
    t = ticker.upper()
    # simple fake values, just to show structure
    demo_metrics = {
        "ticker": t,
        "source": "demo-external-metrics",
        "pe_ratio": 42.0 if t == "NVDA" else 20.0,
        "market_cap_usd_billion": 1200.0 if t == "NVDA" else 500.0,
        "one_year_volatility": 0.35 if t == "NVDA" else 0.25,
        "note": "Synthetic metrics for demo; replace with real API call.",
    }
    return demo_metrics

def _intel_query_to_themes(
    query: str,
    geo_payload: object | None,
) -> list[str]:
    """
    Turn a user query (keyword or full sentence) into one or more
    'themes' we should center the intel graph on.

    Strategy:
      - If query is short (<= 4 tokens), use it directly as a theme.
      - Otherwise, prefer any 'keywords' field returned by geopolitics_lookup().
      - Fallback: just use the raw query.
    """
    q = (query or "").strip()
    if not q:
        return []

    tokens = q.split()
    if len(tokens) <= 4:
        # e.g. "BRICS", "CIPS", "China chip policy"
        return [q]

    # Longer sentence → try to use geopolitics_lookup keywords.
    keywords: list[str] = []
    if isinstance(geo_payload, dict):
        kw = geo_payload.get("keywords")
        if isinstance(kw, list):
            keywords = [str(k) for k in kw if isinstance(k, (str, int, float))]

    if keywords:
        return keywords

    # Last resort, just treat the sentence itself as a single theme.
    return [q]

def get_intel_graph_neighbors(query: str) -> dict:
    """
    Build a lightweight intel / actors graph around a sanctions / geopolitics theme.

    This uses geopolitics_lookup(), which is already LLM-backed in our pipeline,
    so it works both for:
      - short keywords (e.g. 'BRICS', 'CIPS', 'dedollarization')
      - full questions ('how does China circumvent US chip export controls?')

    Returns a structure consumed directly by the Cytoscape.js intel graph:

      {
        "center": {...},
        "neighbors": [...],
        "edges": [...]
      }
    """
    q = (query or "").strip()
    if not q:
        return {}

    # 1) Ask the geopolitics tool (LLM-backed) for relevant intel notes.
    try:
        geo_raw = geopolitics_lookup(q)  # type: ignore[name-defined]
    except NameError:
        # If geopolitics_lookup isn't wired yet, just return empty.
        return {}

    matches: list[dict] = []
    if isinstance(geo_raw, dict):
        # Our LLM-based tool returns {"topic": ..., "keywords": [...], "matches": [...]}
        m = geo_raw.get("matches") or geo_raw.get("results")
        if isinstance(m, list):
            matches = [x for x in m if isinstance(x, dict)]
    elif isinstance(geo_raw, list):
        matches = [x for x in geo_raw if isinstance(x, dict)]

    if not matches:
        return {}

    # 2) Decide what theme(s) to center on (keyword vs full sentence).
    themes = _intel_query_to_themes(q, geo_raw)
    if not themes:
        themes = [q]

    center_label = themes[0]
    center_key = f"THEME::{center_label}"

    center_node = {
        "id": 0,
        "key": center_key,
        "label": center_label,
        "node_type": "actor",  # treat as actor / main subject
    }

    neighbors: list[dict] = []
    edges: list[dict] = []

    # Simple registry to avoid duplicate nodes.
    node_by_key: dict[str, dict] = {center_key: center_node}
    next_id = 1

    def ensure_node(key: str, label: str, node_type: str) -> dict:
        nonlocal next_id
        if key in node_by_key:
            return node_by_key[key]
        node = {
            "id": next_id,
            "key": key,
            "label": label,
            "node_type": node_type,
        }
        node_by_key[key] = node
        neighbors.append(node)
        next_id += 1
        return node

    # 3) Build nodes/edges:
    #    - Center THEME node
    #    - One INTEL node per matched note
    #    - ACTOR/INSTITUTION nodes for each entity in the note
    for note in matches:
        note_id = note.get("id")
        note_title = note.get("title") or f"Intel note {note_id}"
        note_key = f"INTEL::{note_id}"

        intel_node = ensure_node(note_key, note_title, "intel")

        # Edge from theme center to intel note
        edges.append(
            {
                "id": f"{center_key}->{note_key}::relevant_intel",
                "src": center_key,
                "dst": note_key,
                "edge_type": "relevant_intel",
            }
        )

        for ent in note.get("entities", []):
            if not isinstance(ent, str):
                continue
            ent_label = ent.strip()
            if not ent_label:
                continue

            # 👇 slug to match project_intel_to_kg() convention
            slug = ent_label.upper().replace(" ", "_")
            ent_key = f"ACTOR::{slug}"

            # crude but decent heuristic: names with spaces → person/actor
            node_type = "actor" if " " in ent_label else "institution"

            actor_node = ensure_node(ent_key, ent_label, node_type)
            edges.append(
                {
                    "id": f"{note_key}->{ent_key}::mentioned_in",
                    "src": note_key,
                    "dst": ent_key,
                    "edge_type": "mentioned_in",
                }
            )

    return {
        "center": center_node,
        "neighbors": neighbors,
        "edges": edges,
    }

def seed_intel_articles_demo() -> Dict[str, Any]:
    """
    Seed the intel_articles table with a small but rich set of synthetic
    intelligence notes focused on:
        - Chinese semiconductor industry under US export controls
        - Tactics to circumvent sanctions (front companies, third-country routing)
        - BRICS and dedollarization / alternative payment rails
        - Key actors (states, firms, leaders, institutions)

    This is *demo data* designed to behave like internal research notes:
        - It is not scraped from the internet.
        - It gives the agent something realistic to reason over.
        - At work you would replace this with your own ETL / feeds.

    Returns:
        Summary of how many rows were inserted.
    """
    _ensure_db()

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    # Clear any previous demo intel so rebuilds are deterministic.
    cur.execute("DELETE FROM intel_articles")

    demo_rows = [
        {
            "topic": "china_chip_sanctions",
            "region": "CN",
            "entities": "Chinese government, Ministry of Industry and Information Technology, SMIC, Huawei, US Commerce Department",
            "source": "SyntheticDemo",
            "published_at": "2023-10-15T00:00:00",
            "title": "China expands domestic chip ecosystem to reduce reliance on US tooling",
            "snippet": (
                "Beijing accelerates subsidies for local fabs and equipment makers "
                "to offset US export controls on advanced lithography and EDA tools. "
                "SMIC and Huawei collaborate on workarounds using older nodes and "
                "multi-patterning to approximate leading-edge performance."
            ),
        },
        {
            "topic": "china_chip_sanctions",
            "region": "CN",
            "entities": "Chinese government, state-owned banks, front companies, Hong Kong intermediaries",
            "source": "SyntheticDemo",
            "published_at": "2024-02-01T00:00:00",
            "title": "Use of front companies and Hong Kong intermediaries to access restricted chip equipment",
            "snippet": (
                "Chinese buyers increasingly route orders for restricted semiconductor "
                "equipment through shell entities and trading firms in Hong Kong and "
                "third countries. Payments are structured to obscure the ultimate "
                "beneficiary and avoid direct links to sanctioned entities."
            ),
        },
        {
            "topic": "china_chip_sanctions",
            "region": "Global",
            "entities": "SMIC, Huawei, Taiwanese foundries, US Commerce Department",
            "source": "SyntheticDemo",
            "published_at": "2024-03-12T00:00:00",
            "title": "Shifts in foundry orders as export controls tighten",
            "snippet": (
                "Export controls on advanced nodes lead some Chinese design houses to "
                "split production between domestic fabs and non-US-aligned foundries. "
                "This reduces transparency on true end-use and complicates compliance "
                "for global suppliers."
            ),
        },
        {
            "topic": "brics_dedollarization",
            "region": "Global",
            "entities": "China, Russia, India, Brazil, South Africa, BRICS, SWIFT, CIPS, PBoC, Russian central bank",
            "source": "SyntheticDemo",
            "published_at": "2023-09-05T00:00:00",
            "title": "BRICS explore alternatives to dollar-denominated trade",
            "snippet": (
                "BRICS members expand the use of local currencies and explore common "
                "settlement mechanisms to reduce reliance on the US dollar. China "
                "promotes CIPS as a complement to SWIFT for cross-border RMB payments, "
                "especially in energy and commodity trade with Russia."
            ),
        },
        {
            "topic": "brics_dedollarization",
            "region": "CN",
            "entities": "People's Bank of China, CIPS, Chinese state-owned banks, Gulf energy exporters",
            "source": "SyntheticDemo",
            "published_at": "2024-01-20T00:00:00",
            "title": "Expansion of RMB-settled commodity contracts via CIPS",
            "snippet": (
                "Pilot projects for RMB-settled oil and gas contracts expand through "
                "CIPS, giving Chinese banks greater visibility and control over trade "
                "flows while reducing direct exposure to US financial sanctions."
            ),
        },
        {
            "topic": "sanctions_circumvention_finance",
            "region": "CN",
            "entities": "Chinese state-owned banks, regional lenders, Russian corporates, offshore SPVs",
            "source": "SyntheticDemo",
            "published_at": "2023-12-10T00:00:00",
            "title": "Regional Chinese banks deepen ties with sanctioned counterparties",
            "snippet": (
                "While large Chinese banks remain cautious about secondary sanctions, "
                "regional lenders and offshore special purpose vehicles play a larger "
                "role in financing trade with sanctioned Russian entities, often using "
                "complex ownership chains and non-dollar currencies."
            ),
        },
        {
            "topic": "technology_transfer_channels",
            "region": "CN",
            "entities": "Chinese universities, research institutes, Western chip firms, joint ventures",
            "source": "SyntheticDemo",
            "published_at": "2022-11-03T00:00:00",
            "title": "Research collaborations used as channels for incremental technology transfer",
            "snippet": (
                "Joint labs and university partnerships provide Chinese researchers with "
                "access to know-how in design tools, advanced packaging, and materials. "
                "While not always covered by export controls, these relationships can "
                "accelerate domestic capability upgrades."
            ),
        },
        {
            "topic": "china_chip_industrial_policy",
            "region": "CN",
            "entities": "Chinese government, National IC Fund, local governments, foundry startups",
            "source": "SyntheticDemo",
            "published_at": "2021-08-15T00:00:00",
            "title": "National IC Fund and local subsidies reshape Chinese fab landscape",
            "snippet": (
                "Central and provincial funding channels support a wave of smaller fabs, "
                "some focused on legacy nodes and specialty processes. This broad base "
                "of capacity reduces dependence on a small set of flagship players and "
                "creates many potential counterparties for foreign suppliers."
            ),
        },
        {
            "topic": "key_figures_policy",
            "region": "CN",
            "entities": "Xi Jinping, Liu He, senior economic planners, MIIT leadership",
            "source": "SyntheticDemo",
            "published_at": "2022-03-01T00:00:00",
            "title": "Senior leadership frames semiconductors as core national security priority",
            "snippet": (
                "Public speeches by Xi Jinping and senior economic planners explicitly "
                "link semiconductor self-sufficiency to national security, elevating "
                "chip policy within broader industrial and geopolitical strategy."
            ),
        },
        {
            "topic": "third_country_routing",
            "region": "Asia",
            "entities": "Chinese traders, Southeast Asian intermediaries, US exporters, dual-use goods",
            "source": "SyntheticDemo",
            "published_at": "2024-04-02T00:00:00",
            "title": "Third-country routing of dual-use components through Southeast Asia",
            "snippet": (
                "Some Chinese buyers shift procurement of dual-use chips and modules "
                "to distributors in Southeast Asia, who then re-export to China. "
                "This complicates enforcement because customs declarations often list "
                "benign end-users in the intermediary countries."
            ),
        },
        {
            "topic": "brics_coordination",
            "region": "Global",
            "entities": "BRICS finance ministers, multilateral development banks, sanctions-hit borrowers",
            "source": "SyntheticDemo",
            "published_at": "2023-07-22T00:00:00",
            "title": "BRICS explore coordinated financing tools for sanctioned borrowers",
            "snippet": (
                "Discussions within BRICS forums include options for syndicated lending "
                "and guarantee structures that reduce unilateral leverage of any single "
                "jurisdiction over cross-border capital flows."
            ),
        },
    ]

    for row in demo_rows:
        cur.execute(
            """
            INSERT INTO intel_articles
                (topic, region, entities, source, published_at, title, snippet)
            VALUES
                (:topic, :region, :entities, :source, :published_at, :title, :snippet)
            """,
            row,
        )

    conn.commit()
    conn.close()

    return {"inserted": len(demo_rows), "source": "seed_intel_articles_demo"}

def project_intel_to_kg() -> Dict[str, Any]:
    """
    Project intel_articles into the knowledge graph.

    For each row in intel_articles:
        - Create an INTEL::<id> node with node_type='intel'.
        - Parse entities (comma-separated) and, for each entity:
            * Create an ACTOR::<SLUG> node with node_type='actor' if not present.
            * Create an edge ACTOR::<SLUG> --mentioned_in--> INTEL::<id>.
        - If an entity looks like a known company ticker, also link:
            COMP::<TICKER> --intel_link--> INTEL::<id>.

    This gives us:
        - Macro/thematic INTEL nodes in the KG.
        - Actor nodes (people, institutions, states) that analysts care about.
    """
    _ensure_db()

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    # Preload all company tickers so we can link intel notes to them if mentioned.
    cur.execute("SELECT ticker FROM companies")
    tickers = {row["ticker"].upper() for row in cur.fetchall()}

    # Clear previous INTEL / ACTOR nodes and related edges so rebuild is clean.
    cur.execute("DELETE FROM kg_edges WHERE edge_type IN ('mentioned_in', 'intel_link')")
    cur.execute("DELETE FROM kg_nodes WHERE key LIKE 'INTEL::%'")
    cur.execute("DELETE FROM kg_nodes WHERE key LIKE 'ACTOR::%'")

    # Fetch intel notes.
    cur.execute(
        """
        SELECT id, topic, region, entities, title
        FROM intel_articles
        ORDER BY published_at DESC, id DESC
        """
    )
    rows = cur.fetchall()

    intel_count = 0
    actor_nodes: set[str] = set()
    intel_links = 0

    for row in rows:
        intel_id = row["id"]
        title = row["title"] or f"Intel note {intel_id}"
        entities_raw = row["entities"] or ""
        intel_key = f"INTEL::{intel_id}"

        # Create INTEL node.
        cur.execute(
            """
            INSERT INTO kg_nodes (key, label, node_type)
            VALUES (?, ?, ?)
            """,
            (intel_key, title, "intel"),
        )
        intel_count += 1

        # Parse entities; simple comma-separated list.
        entities = [
            e.strip() for e in entities_raw.split(",") if e.strip()
        ]
        for ent in entities:
            slug = ent.upper().replace(" ", "_")
            actor_key = f"ACTOR::{slug}"

            if actor_key not in actor_nodes:
                cur.execute(
                    """
                    INSERT INTO kg_nodes (key, label, node_type)
                    VALUES (?, ?, ?)
                    """,
                    (actor_key, ent, "actor"),
                )
                actor_nodes.add(actor_key)

            # ACTOR -> INTEL edge.
            cur.execute(
                """
                INSERT INTO kg_edges (src, dst, edge_type)
                VALUES (?, ?, ?)
                """,
                (actor_key, intel_key, "mentioned_in"),
            )

            # If this entity looks like a company ticker we know, link that too.
            candidate = ent.upper()
            if candidate in tickers:
                comp_key = f"COMP::{candidate}"
                cur.execute(
                    """
                    INSERT INTO kg_edges (src, dst, edge_type)
                    VALUES (?, ?, ?)
                    """,
                    (comp_key, intel_key, "intel_link"),
                )
                intel_links += 1

    conn.commit()
    conn.close()

    return {
        "intel_nodes": intel_count,
        "actor_nodes": len(actor_nodes),
        "intel_links": intel_links,
    }

def rebuild_all() -> Dict[str, Any]:
    """
    Run the full demo pipeline:

        1. Pull / refresh company profiles into the companies table.
        2. Build the base knowledge graph (companies + market hub).
        3. Pull demo candles for those tickers.
        4. Ingest news headlines from Yahoo Finance RSS (with fallback) and
           project them into the KG as NEWS nodes.
        5. Seed geopolitical / sanctions / BRICS intel notes.
        6. Project those intel notes into the KG as INTEL and ACTOR nodes.

    In production, these steps would be replaced with:
        - ETL pipelines for real reference data.
        - Internal news / filings feeds.
        - Internal sanctions / geopolitical intelligence sources.

    Returns:
        A dictionary summarizing each stage so the UI and logs can inspect it.
    """
    logger.info("Running demo rebuild_all pipeline")

    profiles_result = pull_profiles()
    graph_result = build_graph_from_profiles()
    candles_result = pull_candles_for_tickers()
    news_result = ingest_news()

    intel_seed_result = seed_intel_articles_demo()
    intel_graph_result = project_intel_to_kg()

    return {
        "profiles": profiles_result,
        "graph": graph_result,
        "candles": candles_result,
        "news": news_result,
        "intel_seed": intel_seed_result,
        "intel_graph": intel_graph_result,
    }


# ---------------------------------------------------------------------------
# Agentic QA entrypoint (stubbed locally, ready for real LLM)
# ---------------------------------------------------------------------------
def get_node_details_for_ticker(ticker: str) -> Dict[str, Any]:
    """
    Aggregate detailed information about a company ticker.

    This function joins multiple internal sources:
        - companies table (profiles)
        - external metrics (placeholder API)
        - articles table (news / filings)
        - candles table (recent OHLCV)
        - graph neighborhood (kg_nodes + kg_edges)

    It returns both raw data structures and a list of human-readable
    summary lines that the UI can render in the "Node Details" panel.

    Args:
        ticker: Company ticker symbol, e.g. "NVDA".

    Returns:
        A dictionary with keys:
            ticker: str
            company: dict or None
            metrics: dict or None
            articles: list[dict]
            candles: list[dict]
            graph: dict (same shape as get_graph_neighbors_for_ticker)
            summary_lines: list[str]
    """
    symbol = (ticker or "").strip().upper()
    _ensure_db()

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    # Company profile
    cur.execute(
        "SELECT id, ticker, name FROM companies WHERE ticker = ?",
        (symbol,),
    )
    row = cur.fetchone()
    company = dict(row) if row is not None else None

    # Latest news
    cur.execute(
        """
        SELECT id, title, published_at
        FROM articles
        WHERE ticker = ?
        ORDER BY published_at DESC
        LIMIT 5
        """,
        (symbol,),
    )
    articles = [dict(r) for r in cur.fetchall()]

    # Recent candles
    cur.execute(
        """
        SELECT dt, open, high, low, close, volume
        FROM candles
        WHERE ticker = ?
        ORDER BY dt DESC
        LIMIT 3
        """,
        (symbol,),
    )
    candles = [dict(r) for r in cur.fetchall()]

    conn.close()

    # External metrics (placeholder API)
    metrics: Optional[Dict[str, Any]] = None
    try:
        metrics = fetch_external_metrics(symbol)
    except Exception as exc:  # noqa: BLE001
        logger.warning("fetch_external_metrics failed in node details for %s: %s", symbol, exc)

    # Graph neighborhood (reuse existing helper)
    graph = get_graph_neighbors_for_ticker(symbol)

    # Build human-readable summary.
    summary_lines: List[str] = [f"Node details for {symbol}:", ""]

    if company:
        summary_lines.append(f"• Company: {company.get('name')}")
    else:
        summary_lines.append("• Company: (not found in profiles table)")

    if metrics:
        summary_lines.append("• External metrics:")
        pe = metrics.get("pe_ratio")
        mc = metrics.get("market_cap_usd_billion")
        vol = metrics.get("one_year_volatility")
        if pe is not None:
            summary_lines.append(f"    - P/E ratio: {pe}")
        if mc is not None:
            summary_lines.append(f"    - Market cap: {mc} B USD")
        if vol is not None:
            summary_lines.append(f"    - 1y volatility: {vol}")
    else:
        summary_lines.append("• External metrics: (unavailable)")

    if articles:
        summary_lines.append(f"• Latest articles ({len(articles)}):")
        for art in articles[:3]:
            title = art.get("title") or "(no title)"
            published_at = art.get("published_at") or ""
            summary_lines.append(f"    - {title} [{published_at}]")
    else:
        summary_lines.append("• Latest articles: (none in articles table)")

    if candles:
        summary_lines.append(f"• Recent candles ({len(candles)} rows):")
        for c in candles[:3]:
            summary_lines.append(
                f"    - {c.get('dt')}: close={c.get('close')} vol={c.get('volume')}"
            )
    else:
        summary_lines.append("• Recent candles: (none in candles table)")

    center = graph.get("center")
    neighbors = graph.get("neighbors") or []
    edges = graph.get("edges") or []
    if center:
        summary_lines.append(
            f"• Graph center: {center.get('key')} ({center.get('node_type')})"
        )
        summary_lines.append(
            f"    - Neighbors: {len(neighbors)}; Edges: {len(edges)}"
        )
    else:
        summary_lines.append("• Graph: no center node found for this ticker.")

    return {
        "ticker": symbol,
        "company": company,
        "metrics": metrics,
        "articles": articles,
        "candles": candles,
        "graph": graph,
        "summary_lines": summary_lines,
    }

def get_intel_actor_neighbors(query: str) -> Dict[str, Any]:
    """
    Public intel / actors graph API.

    1) Use an LLM guardrail to decide if the query belongs in the
       sanctions / BRICS / chip-intel domain.
    2) If out-of-scope → return an empty graph plus guardrail info.
    3) If in-scope → build a graph via get_intel_graph_neighbors() and
       attach node/edge counts + guardrail to the payload.
    """
    q = (query or "").strip()

    guard = _llm_guardrail_intel_query(q)
    in_scope = guard.get("in_scope", False)
    confidence = guard.get("confidence", 0.0)
    reason = guard.get("reason", "unknown")

    MIN_CONF = 0.65  # tweak as you like

    # Out-of-scope or low confidence → empty graph, but explicit about why.
    if not in_scope or confidence < MIN_CONF:
        return {
            "query": q,
            "center": None,
            "neighbors": [],
            "edges": [],
            "node_count": 0,
            "edge_count": 0,
            "guardrail": guard,
        }

    # In-scope → build the actual intel graph.
    graph = get_intel_graph_neighbors(q)

    center = graph.get("center")
    neighbors = graph.get("neighbors", [])
    edges = graph.get("edges", [])

    node_count = (1 if center else 0) + len(neighbors)
    edge_count = len(edges)

    # Flatten: we keep the structure the frontend already expects
    # (center / neighbors / edges at top level) and just add metadata.
    return {
        "query": q,
        "center": center,
        "neighbors": neighbors,
        "edges": edges,
        "node_count": node_count,
        "edge_count": edge_count,
        "guardrail": guard,
    }

def get_intel_node_details(node_key: str) -> Dict[str, Any]:
    """
    Return rich text for an intel / actor / company node key.

    Used by the dashboard when a user clicks a node in the Intel graph.
    """
    _ensure_db()
    key = (node_key or "").strip()

    if not key:
        return {
            "key": key,
            "kind": "unknown",
            "text": "No node key provided.",
        }

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    # Try to enrich from kg_nodes if present, but don't require it.
    cur.execute(
        "SELECT key, label, node_type FROM kg_nodes WHERE key = ?",
        (key,),
    )
    node = cur.fetchone()
    label = node["label"] if node else key
    node_type = node["node_type"] or "unknown" if node else "unknown"

    # --- INTEL::<id> → pull back full intel note from intel_articles
    if key.startswith("INTEL::"):
        intel_id_str = key.split("::", 1)[1]
        try:
            intel_id = int(intel_id_str)
        except ValueError:
            intel_id = None

        row = None
        if intel_id is not None:
            cur.execute(
                """
                SELECT id, topic, region, entities, source,
                       published_at, title, snippet
                FROM intel_articles
                WHERE id = ?
                """,
                (intel_id,),
            )
            row = cur.fetchone()

        conn.close()

        if not row:
            return {
                "key": key,
                "kind": "intel",
                "title": label,
                "text": "No intel_articles row found for this note.",
            }

        entities_raw = row["entities"] or ""
        entities = [e.strip() for e in entities_raw.split(",") if e.strip()]

        text_lines = [
            f"[{row['region']}] {row['title']} ({row['published_at']})",
            "",
        ]
        if entities:
            text_lines.append("Entities: " + ", ".join(entities))
            text_lines.append("")
        if row["snippet"]:
            text_lines.append(row["snippet"])

        return {
            "key": key,
            "kind": "intel",
            "title": row["title"],
            "topic": row["topic"],
            "region": row["region"],
            "source": row["source"],
            "published_at": row["published_at"],
            "entities": entities,
            "snippet": row["snippet"],
            "text": "\n".join(text_lines),
        }

    # --- ACTOR::<...> → show intel notes this actor appears in
    if key.startswith("ACTOR::"):
        cur.execute(
            """
            SELECT dst
            FROM kg_edges
            WHERE src = ?
              AND dst LIKE 'INTEL::%'
            """,
            (key,),
        )
        rows = cur.fetchall()
        intel_ids: list[int] = []
        for r in rows:
            dst_key = r["dst"]
            parts = dst_key.split("::", 1)
            if len(parts) == 2:
                try:
                    intel_ids.append(int(parts[1]))
                except ValueError:
                    continue

        notes: list[dict] = []
        if intel_ids:
            placeholders = ",".join("?" for _ in intel_ids)
            cur.execute(
                f"""
                SELECT id, title, region, published_at
                FROM intel_articles
                WHERE id IN ({placeholders})
                ORDER BY published_at DESC, id DESC
                """,
                intel_ids,
            )
            for r in cur.fetchall():
                notes.append(
                    {
                        "id": r["id"],
                        "title": r["title"],
                        "region": r["region"],
                        "published_at": r["published_at"],
                    }
                )

        conn.close()

        lines = [f"Actor: {label}", ""]
        if notes:
            lines.append("Mentioned in intel notes:")
            for n in notes:
                lines.append(
                    f" - [{n['region']}] {n['title']} "
                    f"[{n['published_at']}] (INTEL::{n['id']})"
                )
        else:
            lines.append("No intel notes found for this actor in the current graph.")

        return {
            "key": key,
            "kind": "actor",
            "label": label,
            "notes": notes,
            "text": "\n".join(lines),
        }

    # --- COMP::<TICKER> → reuse node-details helper
    if key.startswith("COMP::"):
        conn.close()
        ticker = key.split("::", 1)[1]
        details = get_node_details_for_ticker(ticker)
        return {
            "key": key,
            "kind": "company",
            "ticker": ticker,
            "text": "\n".join(details.get("summary_lines", [])),
        }

    # --- Fallback generic case
    conn.close()
    return {
        "key": key,
        "kind": node_type,
        "label": label,
        "text": f"{label} (node_type={node_type})",
    }

def get_graph_neighbors_for_ticker(ticker: str) -> Dict[str, Any]:
    """
    Return a small graph neighborhood around a company ticker.

    The neighborhood is defined as:
        - center: the kg_nodes row for COMP::<TICKER>
        - neighbors: all nodes connected to the center by any edge
        - edges: all edges where src or dst is the center key

    Args:
        ticker: Company ticker symbol (e.g. "NVDA").

    Returns:
        A dictionary with keys:
            center:   dict or None
            neighbors: list[dict]
            edges:    list[dict]
    """
    _ensure_db()

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    key = f"COMP::{ticker.upper()}"

    # Look up the center node.
    cur.execute(
        "SELECT id, key, label, node_type FROM kg_nodes WHERE key = ?",
        (key,),
    )
    center_row = cur.fetchone()
    if not center_row:
        conn.close()
        return {"center": None, "neighbors": [], "edges": []}

    center = dict(center_row)

    # Fetch edges touching this node.
    cur.execute(
        """
        SELECT src, dst, edge_type
        FROM kg_edges
        WHERE src = ? OR dst = ?
        """,
        (key, key),
    )
    edges = [dict(r) for r in cur.fetchall()]

    # Collect neighbor keys (the other endpoint in each edge).
    neighbor_keys = set()
    for edge in edges:
        if edge["src"] == key:
            neighbor_keys.add(edge["dst"])
        if edge["dst"] == key:
            neighbor_keys.add(edge["src"])

    neighbors: List[Dict[str, Any]] = []
    if neighbor_keys:
        placeholders = ",".join("?" for _ in neighbor_keys)
        cur.execute(
            f"""
            SELECT id, key, label, node_type
            FROM kg_nodes
            WHERE key IN ({placeholders})
            """,
            tuple(neighbor_keys),
        )
        neighbors = [dict(r) for r in cur.fetchall()]

    conn.close()

    return {
        "center": center,
        "neighbors": neighbors,
        "edges": edges,
    }

def _lookup_news_for_tickers(tickers: List[str]) -> List[Dict[str, Any]]:
    """
    Helper: fetch recent news headlines for one or more tickers from articles.

    Args:
        tickers: List of ticker symbols.

    Returns:
        A list of dictionaries with ticker, title, and published_at fields.
    """
    _ensure_db()

    if not tickers:
        return []

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    placeholders = ",".join("?" for _ in tickers)
    sql = f"""
        SELECT ticker, title, published_at
        FROM articles
        WHERE ticker IN ({placeholders})
        ORDER BY published_at DESC, id DESC
    """
    cur.execute(sql, [t.upper() for t in tickers])
    rows = cur.fetchall()
    conn.close()

    return [
        {
            "ticker": r["ticker"],
            "title": r["title"],
            "published_at": r["published_at"],
        }
        for r in rows
    ]

def companies_lookup(tickers: list[str]) -> list[dict[str, Any]]:
    """
    Demo implementation of companies_lookup for the agent trace.

    In production you can replace this function to call your internal
    reference-data service instead of the local SQLite DB.
    """
    return _lookup_companies_by_tickers(tickers)


def external_metrics_api(tickers: list[str]) -> list[dict[str, Any]]:
    """
    Demo external metrics tool, backed by fetch_external_metrics().
    """
    return [fetch_external_metrics(t) for t in tickers]


def news_lookup(tickers: list[str]) -> list[dict[str, Any]]:
    """
    Demo news tool, backed by the local articles table.
    """
    return _lookup_news_for_tickers(tickers)

def _run_demo_agentic_flow(question: str) -> Dict[str, Any]:
    """
    Run a small, fully local "agentic" flow for a given question.

    For demo purposes, this function does not call an external LLM. Instead it:
        1. Classifies the question and extracts any mentioned tickers.
        2. Retrieves company profiles for those tickers.
        3. Fetches placeholder external metrics.
        4. Retrieves recent news headlines for those tickers.
        5. If the question appears to be about China / BRICS / sanctions /
           chips / semiconductors, it also:
              - Calls geopolitics_lookup() over intel_articles.
        6. Synthesizes a textual answer and returns:
              - answer: str
              - steps: high-level agent steps
              - trace: raw tool inputs/outputs
              - stats: small summary

    In production, this flow would be implemented using your internal LLM and
    orchestration framework (e.g. LangGraph), but the tool boundaries and
    trace structure can remain the same.
    """
    q = (question or "").strip()
    lower_q = q.lower()

    # ── Step 1: detect *real* tickers from the companies table.
    tickers = _detect_tickers_from_text(q)

    steps: List[Dict[str, Any]] = []
    trace: List[Dict[str, Any]] = []

    # Step: understand_question
    steps.append(
        {
            "name": "understand_question",
            "description": "Classify question and detect relevant tickers and themes.",
            "inputs": {"question": q},
            "outputs": {
                "mentioned_tickers": tickers,
            },
        }
    )

    # ── Step 2: retrieve_profiles
    profiles_info: List[Dict[str, Any]] = []
    if tickers:
        profiles_info = _lookup_companies_by_tickers(tickers)
    steps.append(
        {
            "name": "retrieve_profiles",
            "description": "Look up basic company profiles.",
            "inputs": {"tickers": tickers},
            "outputs": {"profiles_found": len(profiles_info)},
        }
    )
    trace.append(
        {
            "tool": "companies_lookup",
            "args": {"tickers": tickers},
            "results": profiles_info,
        }
    )

    # ── Step 3: external_metrics_api (placeholder).
    metrics_results: List[Dict[str, Any]] = []
    for t in tickers:
        metrics_results.append(fetch_external_metrics(t))
    steps.append(
        {
            "name": "fetch_external_metrics",
            "description": "Call external metrics API/service for each ticker.",
            "inputs": {"tickers": tickers},
            "outputs": {"count": len(metrics_results)},
        }
    )
    trace.append(
        {
            "tool": "external_metrics_api",
            "args": {"tickers": tickers},
            "results": metrics_results,
        }
    )

    # ── Step 4: news_lookup from articles table.
    news_results: List[Dict[str, Any]] = []
    if tickers:
        news_results = _lookup_news_for_tickers(tickers)
    steps.append(
        {
            "name": "retrieve_news",
            "description": "Retrieve recent news Reuters/filings-style items for each ticker.",
            "inputs": {"tickers": tickers},
            "outputs": {"articles_found": len(news_results)},
        }
    )
    trace.append(
        {
            "tool": "news_lookup",
            "args": {"tickers": tickers},
            "results": news_results,
        }
    )

    # ── Step 5: geopolitics_lookup if question is about China / sanctions / BRICS / chips.
    geo_keywords = [
        "china",
        "chinese",
        "brics",
        "sanction",
        "export control",
        "chip",
        "semiconductor",
        "fab",
        "cips",
        "yuan",
        "renminbi",
    ]
    needs_geo = any(kw in lower_q for kw in geo_keywords)

    geo_result: Optional[Dict[str, Any]] = None
    if needs_geo:
        geo_result = geopolitics_lookup(q)
        matches = geo_result.get("matches", []) if geo_result else []
        steps.append(
            {
                "name": "geopolitics_lookup",
                "description": "Search geopolitical / sanctions / BRICS intelligence notes.",
                "inputs": {"topic": q},
                "outputs": {"articles_found": len(matches)},
            }
        )
        trace.append(
            {
                "tool": "geopolitics_lookup",
                "args": {"topic": q},
                "results": matches,
            }
        )

    # ── Synthesis: build a readable answer string.
    lines: List[str] = []
    if tickers:
        focus = ", ".join(tickers)
        lines.append(f"For the question '{q}', I focused on: {focus}.")
        lines.append("")
    else:
        lines.append(f"For the question '{q}', I treated this as a thematic inquiry.")
        lines.append("")

    if profiles_info:
        lines.append(
            f"There are {len(profiles_info)} company profiles in the graph for these tickers."
        )

    if news_results:
        lines.append(
            f"There are {len(news_results)} recent news articles linked to these names."
        )

    if metrics_results:
        lines.append(
            "External metrics (placeholder API) provide a snapshot of valuation and volatility."
        )

    if geo_result:
        matches = geo_result.get("matches", [])
        if matches:
            lines.append("")
            lines.append(
                f"The question includes China/BRICS/sanctions themes, so I also consulted "
                f"{len(matches)} geopolitical intelligence notes."
            )
            lines.append("Examples:")
            for m in matches[:3]:
                title = m.get("title") or "(untitled intel note)"
                region = m.get("region") or "Global"
                lines.append(f" - [{region}] {title}")
            lines.append("")

            # High-level synthesis of what those notes are about.
            lines.append("Taken together, these notes highlight several recurring tactics:")
            lines.append(
                " - Expanding the *domestic chip ecosystem* via subsidies, the National IC Fund, "
                "and a broad base of smaller fabs to reduce reliance on US tooling."
            )
            lines.append(
                " - Using *front companies, Hong Kong intermediaries, and third-country routing* "
                "to acquire restricted semiconductor equipment and dual-use components."
            )
            lines.append(
                " - *Splitting production* across Chinese and non-US-aligned foundries to obscure "
                "true end-use and make export controls harder to enforce."
            )
            lines.append(
                " - Building *non-dollar financial channels* through BRICS coordination, RMB-"
                "settled commodity contracts, and CIPS to lower exposure to US financial sanctions."
            )
            lines.append(
                " - Leveraging *research collaborations and joint ventures* with foreign firms and "
                "universities as channels for incremental technology transfer."
            )
            lines.append(
                " - Relying on *regional banks and offshore SPVs* to finance trade with sanctioned "
                "counterparties while larger banks remain more cautious."
            )
        else:
            lines.append("")
            lines.append(
                "I attempted to consult geopolitical intelligence notes, but none matched this topic."
            )

    lines.append("")
    lines.append(
        "In a real agentic setup, these tools would point at internal reference data, "
        "sanctions/OSINT feeds, and portfolio exposures. The trace below records which "
        "tools I called and what I used from each to answer the question."
    )

    answer = "\n".join(lines)

    stats = {
        "tickers": tickers,
        "profiles": len(profiles_info),
        "news_articles": len(news_results),
        "geo_notes": len(geo_result.get("matches", [])) if geo_result else 0,
    }

    return {
        "answer": answer,
        "steps": steps,
        "trace": trace,
        "stats": stats,
    }


def llm_analyze(question: str) -> dict:
    """
    Main agentic controller for the Johny 5 prototype, backed by a REAL LLM
    via Ollama instead of hard-coded text.

    Orchestration steps:

    1) Use simple heuristics to detect possible tickers in the question.
    2) Call companies_lookup() to filter for real tickers and fetch profiles.
    3) Call external_metrics_api() for placeholder valuation/volatility.
    4) Call news_lookup() to fetch recent articles per ticker.
    5) Call geopolitics_lookup() for China/BRICS/sanctions-style intel notes.
    6) If we have at least one ticker, call get_graph_neighbors_for_ticker()
       to capture a local graph neighborhood.
    7) Build a structured prompt that includes all of this context and send it
       to a local Ollama model to get the final answer text.

    Returns a dict with:
        {
          "answer": <str>,       # LLM-written answer
          "steps": [...],        # high-level logical steps
          "trace": [...],        # low-level tool call trace
          "stats": {...},        # summary counts
        }

    The UI simply renders "answer" and shows "steps"/"trace" as JSON.
    """
    q = (question or "").strip()
    if not q:
        return {
            "answer": "Please provide a non-empty question.",
            "steps": [],
            "trace": [],
            "stats": {},
        }

    steps: list[dict] = []
    trace: list[dict] = []
    stats: dict[str, object] = {}

    #
    # 1) Detect candidate tickers from question text.
    #
    candidates = _extract_candidate_tickers(q)
    steps.append(
        {
            "name": "understand_question",
            "description": "Classify question and detect relevant tickers and themes.",
            "inputs": {"question": q},
            "outputs": {"candidate_tickers": candidates},
        }
    )

    #
    # 2) Company profiles via companies_lookup().
    #
    profiles: list[dict] = []
    tickers: list[str] = []
    if candidates:
        try:
            profiles = companies_lookup(candidates)  # type: ignore[name-defined]
        except NameError:
            # If companies_lookup is not implemented yet, just skip gracefully.
            profiles = []

    if profiles:
        seen: set[str] = set()
        for p in profiles:
            t = (p.get("ticker") or "").upper()
            if t and t not in seen:
                seen.add(t)
        tickers = sorted(seen)

    stats["tickers"] = tickers
    stats["profiles"] = len(profiles)

    trace.append(
        {
            "tool": "companies_lookup",
            "args": {"tickers": candidates},
            "results": profiles,
        }
    )

    if profiles:
        steps.append(
            {
                "name": "retrieve_profiles",
                "description": "Look up basic company profiles for detected tickers.",
                "inputs": {"tickers": candidates},
                "outputs": {"profiles_found": len(profiles)},
            }
        )

    #
    # 3) External metrics.
    #
    metrics: list[dict] = []
    if tickers:
        try:
            metrics = external_metrics_api(tickers)  # type: ignore[name-defined]
        except NameError:
            metrics = []

    stats["metrics"] = len(metrics)

    trace.append(
        {
            "tool": "external_metrics_api",
            "args": {"tickers": tickers},
            "results": metrics,
        }
    )

    if metrics:
        steps.append(
            {
                "name": "fetch_external_metrics",
                "description": "Call external metrics API/service for each ticker.",
                "inputs": {"tickers": tickers},
                "outputs": {"count": len(metrics)},
            }
        )

    #
    # 4) News / filings.
    #
    articles: list[dict] = []
    if tickers:
        try:
            articles = news_lookup(tickers)  # type: ignore[name-defined]
        except NameError:
            articles = []

    stats["news_articles"] = len(articles)

    trace.append(
        {
            "tool": "news_lookup",
            "args": {"tickers": tickers},
            "results": articles,
        }
    )

    if articles:
        steps.append(
            {
                "name": "retrieve_news",
                "description": "Retrieve recent news / filings-style items for each ticker.",
                "inputs": {"tickers": tickers},
                "outputs": {"articles_found": len(articles)},
            }
        )

    #
    # 5) Geopolitics / sanctions / BRICS intel, even if there are no tickers.
    #
    raw_geo: object = []
    try:
        raw_geo = geopolitics_lookup(q)  # type: ignore[name-defined]
    except NameError:
        raw_geo = []

    geo_notes: list[dict] = _normalize_geo_notes(raw_geo)
    stats["geo_notes"] = len(geo_notes)

    trace.append(
        {
            "tool": "geopolitics_lookup",
            "args": {"topic": q},
            "results": raw_geo,
        }
    )

    if geo_notes:
        steps.append(
            {
                "name": "geopolitics_lookup",
                "description": "Search geopolitical / sanctions / BRICS intelligence notes.",
                "inputs": {"topic": q},
                "outputs": {"articles_found": len(geo_notes)},
            }
        )

    #
    # 6) Optional graph neighborhood for the first ticker.
    #
    graph_neighborhood: dict | None = None
    if tickers:
        first = tickers[0]
        try:
            graph_neighborhood = get_graph_neighbors_for_ticker(first)  # type: ignore[name-defined]
            stats["graph_center"] = graph_neighborhood.get("center", {}).get("key")
            stats["graph_neighbors"] = len(graph_neighborhood.get("neighbors", []))
            stats["graph_edges"] = len(graph_neighborhood.get("edges", []))

            steps.append(
                {
                    "name": "graph_neighbors",
                    "description": "Retrieve local graph neighborhood for the first ticker.",
                    "inputs": {"ticker": first},
                    "outputs": {
                        "neighbors": stats["graph_neighbors"],
                        "edges": stats["graph_edges"],
                    },
                }
            )

            trace.append(
                {
                    "tool": "graph_neighbors",
                    "args": {"ticker": first},
                    "results": graph_neighborhood,
                }
            )
        except Exception as exc:  # noqa: BLE001
            steps.append(
                {
                    "name": "graph_neighbors",
                    "description": "Attempted to retrieve graph neighborhood but failed.",
                    "inputs": {"ticker": first},
                    "outputs": {"error": str(exc)},
                }
            )

    #
    # 7) Build LLM prompt and call Ollama.
    #
    prompt = _build_agentic_prompt_for_llm(
        question=q,
        tickers=tickers,
        profiles=profiles,
        metrics=metrics,
        articles=articles,
        geo_notes=geo_notes,
        graph_neighborhood=graph_neighborhood,
    )

    answer_text = _ollama_generate(prompt)

    return {
        "answer": answer_text,
        "steps": steps,
        "trace": trace,
        "stats": stats,
    }

def _normalize_geo_notes(raw_geo: object) -> list[dict]:
    """
    Normalize whatever geopolitics_lookup() returns into a list[dict].

    Supports:
      - list[dict]
      - {"matches": [...]} style dict
      - {"results": [...]} style dict
      - {"notes": [...]} style dict

    Any non-dict entries are filtered out.
    """
    if isinstance(raw_geo, list):
        return [g for g in raw_geo if isinstance(g, dict)]

    if isinstance(raw_geo, dict):
        for key in ("matches", "results", "notes"):
            val = raw_geo.get(key)
            if isinstance(val, list):
                return [g for g in val if isinstance(g, dict)]

    return []

def _ollama_structured_geointent(query: str) -> dict:
    """
    Use the local LLM (via Ollama) to translate an analyst query into a
    structured search intent for intel_articles.

    Expected JSON shape from the model:

        {
          "keywords": ["china", "chips", "sanctions"],
          "entities": ["Chinese state-owned banks", "CIPS"],
          "regions": ["CN", "Global"],
          "risk_tags": ["sanctions_circumvention", "payments"]
        }

    All fields are optional; we fall back to heuristics if parsing fails.
    """
    q = (query or "").strip()
    if not q:
        return {}

    prompt = textwrap.dedent(f"""
        You are a sanctions / BRICS / geopolitics intelligence planner.

        Your job is to translate a free-text analyst question into a compact
        JSON object that can drive search over an intel_notes database.

        The database contains notes on:
          - Chinese semiconductor policy and US export controls.
          - Tactics to circumvent sanctions (front companies, third-country routing,
            regional banks, offshore SPVs, dual-use goods).
          - BRICS and dedollarization (CIPS, local-currency settlement, RMB trade).
          - Key actors (governments, state-owned banks, leaders, companies).

        For the USER_QUERY below, identify:
          - 3–7 short "keywords" for searching titles/snippets/entities.
          - 2–6 "entities" (people, institutions, countries, organisations).
          - 0–4 "regions" (e.g. "CN", "US", "Global", "Asia", "EU").
          - 0–6 "risk_tags" like
              ["sanctions_circumvention","chips","payments","energy","banks"].

        Return ONLY valid JSON, with this exact top-level schema
        and lowercase strings where reasonable:

        {{
          "keywords": ["...", "..."],
          "entities": ["...", "..."],
          "regions": ["...", "..."],
          "risk_tags": ["...", "..."]
        }}

        No commentary, no backticks, no explanation – just the JSON.

        USER_QUERY:
        {q}
    """).strip()

    raw = _ollama_generate(prompt, temperature=0.0)

    # Try straight JSON first.
    try:
        return json.loads(raw)
    except Exception:
        # Try to salvage a JSON block if the model added extra text.
        start = raw.find("{")
        end = raw.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(raw[start : end + 1])
            except Exception:
                pass

    logger.warning(
        "ollama_structured_geointent: failed to parse JSON from LLM output: %r",
        raw[:200],
    )
    return {}

def geopolitics_lookup(topic: str) -> Dict[str, Any]:
    """
    LLM-assisted lookup of geopolitical / sanctions / BRICS intel notes.

    Pipeline:
      1) Ask the local LLM (via Ollama) for a structured search intent
         (keywords, entities, regions, risk_tags).
      2) Merge that with a small heuristic fallback so we always have
         at least one keyword.
      3) Query intel_articles using those keywords across topic/title/
         snippet/entities.
      4) Return a normalized payload with the matches plus the intent
         the model produced.

    This is used both by:
      - llm_analyze() for context in the main answer.
      - get_intel_graph_neighbors() to drive the Intel / Actors graph.
    """
    _ensure_db()

    raw = (topic or "").strip()
    lower = raw.lower()

    # 1) Structured intent from the LLM (best effort).
    intent = _ollama_structured_geointent(raw)

    kw_from_llm = [
        str(k).strip().lower()
        for k in intent.get("keywords", [])
        if str(k).strip()
    ]
    entities_from_llm = [
        str(e).strip()
        for e in intent.get("entities", [])
        if str(e).strip()
    ]
    regions_from_llm = [
        str(r).strip().upper()
        for r in intent.get("regions", [])
        if str(r).strip()
    ]
    risk_tags = [
        str(r).strip().lower()
        for r in intent.get("risk_tags", [])
        if str(r).strip()
    ]

    keywords: list[str] = kw_from_llm.copy()

    # 2) Heuristic reinforcement / fallback so we never return an empty search.
    if "china" in lower and "china" not in keywords:
        keywords.append("china")
    if any(tok in lower for tok in ("chip", "semiconductor", "fab")) and "chip" not in keywords:
        keywords.append("chip")
    if "sanction" in lower or "export control" in lower:
        if "sanction" not in keywords:
            keywords.append("sanction")
    if "brics" in lower and "brics" not in keywords:
        keywords.append("brics")
    if any(tok in lower for tok in ("cips", "renminbi", "yuan")) and "cips" not in keywords:
        keywords.append("cips")
    if any(tok in lower for tok in ("dollar", "dedollar")) and "dollar" not in keywords:
        keywords.append("dollar")

    # Final backstop: always have at least one anchor.
    if not keywords:
        keywords = ["china"]

    # Deduplicate while preserving order.
    seen_kw: set[str] = set()
    final_keywords: list[str] = []
    for kw in keywords:
        if kw not in seen_kw:
            seen_kw.add(kw)
            final_keywords.append(kw)

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    where_clauses: list[str] = []
    params: list[Any] = []

    for kw in final_keywords:
        pattern = f"%{kw}%"
        where_clauses.append(
            "(LOWER(topic) LIKE ? OR LOWER(title) LIKE ? "
            "OR LOWER(snippet) LIKE ? OR LOWER(entities) LIKE ?)"
        )
        params.extend([pattern, pattern, pattern, pattern])

    where_sql = " OR ".join(where_clauses)

    sql = f"""
        SELECT id, topic, region, entities, source, published_at, title, snippet
        FROM intel_articles
        WHERE {where_sql}
        ORDER BY published_at DESC, id DESC
    """
    cur.execute(sql, params)
    rows = cur.fetchall()
    conn.close()

    matches: List[Dict[str, Any]] = []
    for r in rows:
        entities_raw = r["entities"] or ""
        entities = [e.strip() for e in entities_raw.split(",") if e.strip()]
        matches.append(
            {
                "id": r["id"],
                "title": r["title"],
                "source": r["source"],
                "published_at": r["published_at"],
                "region": r["region"],
                "entities": entities,
                "snippet": r["snippet"],
            }
        )

    return {
        "topic": raw,
        "keywords": final_keywords,
        "entities_hint": entities_from_llm,
        "regions_hint": regions_from_llm,
        "risk_tags": risk_tags,
        "matches": matches,
    }

def _llm_guardrail_intel_query(query: str) -> Dict[str, Any]:
    """
    Use the local LLM (via Ollama) as a gatekeeper to decide whether a query
    belongs in the geopolitics / sanctions / BRICS intel domain.

    Returns:
        {
          "in_scope": bool,
          "confidence": float,   # 0.0–1.0
          "reason": str
        }
    """
    q = (query or "").strip()
    if not q:
        return {"in_scope": False, "confidence": 0.0, "reason": "empty_query"}

    prompt = textwrap.dedent(f"""
        You are a strict gatekeeper for a geopolitical / sanctions
        intelligence knowledge graph. The graph ONLY covers:

        - sanctions, export controls, and trade restrictions
        - BRICS, CIPS, SWIFT, and alternative payment / settlement systems
        - central banks, state-owned banks, and multilateral lenders
        - chip / semiconductor supply chains and dual-use exports
        - energy flows tied to these topics (oil, gas, LNG, etc.)

        Read the user's query and decide if it is clearly about THIS domain.
        If it is obviously about sports, entertainment, weather, or anything
        unrelated, mark it out_of_scope.

        Respond ONLY with compact JSON:

        {{
          "in_scope": true or false,
          "confidence": number between 0 and 1,
          "reason": "short explanation"
        }}

        User query: {q!r}
    """).strip()

    raw = _ollama_generate(prompt, temperature=0.0)

    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        # Fail closed – if the guardrail can't parse, treat as out-of-scope.
        return {
            "in_scope": False,
            "confidence": 0.0,
            "reason": "guardrail_llm_invalid_json",
        }

    in_scope = bool(obj.get("in_scope"))
    confidence = float(obj.get("confidence", 0.0))
    reason = str(obj.get("reason", "") or "").strip() or "no_reason"

    return {
        "in_scope": in_scope,
        "confidence": confidence,
        "reason": reason,
    }


class BackendService:
    ...
    def get_intel_node_details(self, node_key: str) -> dict:
        return get_intel_node_details(node_key)

    def get_intel_actor_neighbors(self, query: str) -> dict:
        return get_intel_actor_neighbors(query)