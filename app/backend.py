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
from typing import Any, Dict, List, Optional

from email.utils import parsedate_to_datetime
from xml.etree import ElementTree as ET

import requests

logger = logging.getLogger(__name__)

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

    conn.commit()
    conn.close()


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
    Ingest news articles tied to companies via Yahoo Finance RSS, with fallback.

    For each company ticker in the DB, this function:
        - Attempts to fetch the Yahoo Finance RSS feed for that ticker.
        - If successful:
            * Clears existing rows in the articles table for that ticker.
            * Inserts up to `max_per_ticker` of the latest headlines.
        - If the request fails (429 rate limit, network issue, etc.):
            * Keeps existing rows if any.
            * If there are no existing rows, inserts a single placeholder
              article explaining that news is unavailable and this is
              demo data.

    This keeps the demo resilient: network / rate limit issues won't
    leave the articles table empty, and the agent's trace still has
    something to show.

    Args:
        max_per_ticker: Maximum headlines to store per ticker.

    Returns:
        A dictionary summarizing inserted rows and any errors:
            {
              "tickers": [...],
              "inserted_total": int,
              "errors": {ticker: "error message", ...},
              "source": "yahoo-finance-rss+fallback"
            }
    """
    logger.info("Ingesting news from Yahoo Finance RSS (with fallback)")
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

        have_real_news = False
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
            for item in items[:max_per_ticker]:
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

                cur.execute(
                    """
                    INSERT INTO articles (ticker, title, published_at)
                    VALUES (?, ?, ?)
                    """,
                    (symbol, title, published_at),
                )
                inserted_for_symbol += 1

            have_real_news = True
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
        "source": "yahoo-finance-rss+fallback",
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

def rebuild_all() -> Dict[str, Any]:
    """
    Run a top-level data pipeline: profiles → graph → candles → news.

    For the demo, this calls local helper functions in sequence.

    Returns:
        A dictionary summarizing each stage and final stats.
    """
    logger.info("Running demo rebuild_all pipeline")
    out_profiles = pull_profiles()
    out_graph = build_graph_from_profiles()
    out_candles = pull_candles_for_tickers()
    out_news = ingest_news()
    stats = get_stats()

    return {
        "profiles": out_profiles,
        "graph": out_graph,
        "candles": out_candles,
        "news": out_news,
        "stats": stats,
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

def _run_demo_agentic_flow(question: str) -> Dict[str, Any]:
    """
    Run a simple, local "agentic-style" flow without an external LLM.

    This is a stand-in for the real agent you will build at work.

    The logic:
        1. Parse the question in a trivial way (no real NLP).
        2. Query the DB for companies and articles.
        3. Build a small trace that mimics tool calls and results.
        4. Compose a short answer summarizing what we have.

    Args:
        question: Natural-language question from the analyst.

    Returns:
        A dictionary containing:
            answer: str
            stats: dict
            trace: list of "tool call" records
            steps: high-level list of reasoning steps
    """
    stats = get_stats()
    _ensure_db()

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    # Very naive "ticker detection": look for any company ticker in the question.
    cur.execute("SELECT ticker, name FROM companies")
    companies = cur.fetchall()
    mentioned: List[Dict[str, Any]] = []
    lowered = question.lower()

    for row in companies:
        ticker = row["ticker"]
        name = row["name"]
        if ticker.lower() in lowered or name.lower() in lowered:
            mentioned.append({"ticker": ticker, "name": name})

    # If no explicit mention, just take top 3 companies as "context universe".
    if not mentioned and companies:
        mentioned = [
            {"ticker": row["ticker"], "name": row["name"]}
            for row in companies[:3]
        ]

    # Get news for mentioned tickers (if any).
    news_results: List[Dict[str, Any]] = []
    for comp in mentioned:
        ticker = comp["ticker"]
        cur.execute(
            """
            SELECT id, title, published_at
            FROM articles
            WHERE ticker = ?
            ORDER BY published_at DESC
            LIMIT 3
            """,
            (ticker,),
        )
        rows = cur.fetchall()
        for r in rows:
            news_results.append(
                {
                    "article_id": r["id"],
                    "ticker": ticker,
                    "title": r["title"],
                    "published_at": r["published_at"],
                }
            )

    conn.close()

    # Fetch synthetic "external API" metrics for each mentioned ticker.
    external_metrics: List[Dict[str, Any]] = []
    for comp in mentioned:
        ticker = comp["ticker"]
        try:
            metrics = fetch_external_metrics(ticker)
            external_metrics.append(metrics)
        except Exception as exc:  # noqa: BLE001
            logger.warning("fetch_external_metrics failed for %s: %s", ticker, exc)

    # Build an agent-style "steps" trace.
    steps = [
        {
            "name": "understand_question",
            "description": "Classify question and detect relevant tickers.",
            "inputs": {"question": question},
            "outputs": {"mentioned_tickers": [m["ticker"] for m in mentioned]},
        },
        {
            "name": "retrieve_profiles",
            "description": "Look up basic company profiles.",
            "inputs": {"tickers": [m["ticker"] for m in mentioned]},
            "outputs": {"profiles_found": len(mentioned)},
        },
        {
            "name": "fetch_external_metrics",
            "description": "Call external metrics API/service for each ticker.",
            "inputs": {"tickers": [m["ticker"] for m in mentioned]},
            "outputs": {"metrics_count": len(external_metrics)},
        },
        {
            "name": "retrieve_news",
            "description": "Pull recent news for mentioned tickers.",
            "inputs": {"tickers": [m["ticker"] for m in mentioned]},
            "outputs": {"articles_found": len(news_results)},
        },
    ]

    trace = [
        {
            "tool": "companies_lookup",
            "dataset": "companies",
            "results": mentioned,
        },
        {
            "tool": "external_metrics_api",
            "dataset": "external",
            "results": external_metrics,
        },
        {
            "tool": "news_lookup",
            "dataset": "articles",
            "results": news_results,
        },
    ]

    # Synthesize a tiny human-readable answer.
    if not mentioned:
        answer_lines = [
            "I could not detect a specific ticker in your question,",
            "but the knowledge base currently contains:",
            f"- {stats['companies']} companies",
            f"- {stats['articles']} articles",
            f"- {stats['kg_nodes']} knowledge graph nodes",
            f"- {stats['kg_edges']} knowledge graph edges",
        ]
    else:
        summary_tickers = ", ".join(m["ticker"] for m in mentioned)
        answer_lines = [
            f"For the question {question!r}, I focused on: {summary_tickers}.",
            "",
            f"There are {len(news_results)} recent demo news articles for these names.",
            "In a real agentic setup, this step would:",
            "- Retrieve structured fundamentals and time series.",
            "- Retrieve relevant news & filings.",
            "- Combine them to answer risk/exposure questions.",
        ]

    return {
        "answer": "\n".join(answer_lines),
        "stats": stats,
        "trace": trace,
        "steps": steps,
    }


def llm_analyze(question: str) -> Dict[str, Any]:
    """
    High-level analysis entrypoint used by /api/ask.

    In production at work, this function should:
    - Call your internal LLM / LangGraph agent.
    - Orchestrate tools over profiles, candles, news, and KG.
    - Return a payload with:
        answer: final natural language answer
        stats: snapshot of the DB
        trace: list of tool calls and their results (provenance)
        steps: high-level reasoning steps (for debugging / demo)

    For the prototype at home, this function uses a local demo flow
    implemented by _run_demo_agentic_flow().

    Args:
        question: Natural-language question from the analyst.

    Returns:
        A dictionary payload suitable to be returned from /api/ask.
    """
    question = (question or "").strip()
    if not question:
        stats = get_stats()
        return {
            "answer": "Please provide a non-empty question.",
            "stats": stats,
            "trace": [],
            "steps": [],
            "source": "empty-question",
        }

    # In the future, drop your real agent here:
    #
    #   return internal_agent.run(question=question, db_path=str(DB_PATH))
    #
    # For now, we run a local demo flow.
    result = _run_demo_agentic_flow(question)
    result.setdefault("source", "demo-agentic")
    return result
