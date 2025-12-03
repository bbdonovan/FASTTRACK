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


def ingest_news() -> Dict[str, Any]:
    """
    Ingest or seed news articles tied to companies.

    For the prototype:
    - Creates one synthetic article per company.

    Returns:
        A dictionary summarizing how many articles were inserted.
    """
    logger.info("Using demo ingest_news implementation")
    _ensure_db()

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT ticker FROM companies")
    tickers = [row[0] for row in cur.fetchall()]

    inserted = 0
    for ticker in tickers:
        title = f"Demo article for {ticker}"
        cur.execute(
            """
            INSERT INTO articles (ticker, title, published_at)
            VALUES (?, ?, datetime('now'))
            """,
            (ticker, title),
        )
        inserted += 1

    conn.commit()
    conn.close()

    return {"inserted": inserted, "tickers": tickers}


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
