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
    Return a small neighborhood around an ACTOR or INTEL node that matches
    the given free-text query.

    Search strategy:
        1. Try to match ACTOR nodes (node_type='actor') whose label contains
           the query (case-insensitive).
        2. If no actor matches, try to match INTEL nodes (node_type='intel').
        3. If still nothing, fall back to any node whose label contains query.
        4. Once a center node is chosen:
            - Fetch all edges where src = center or dst = center.
            - Pull neighbor nodes referenced by those edges.
            - Return a payload similar to get_graph_neighbors_for_ticker:
                  {
                    "query": "...",
                    "center": {...},
                    "neighbors": [...],
                    "edges": [...]
                  }

    This lets the UI explore themes like "Xi Jinping", "CIPS", "BRICS",
    "Chinese state-owned banks", etc., and see which intel notes and
    companies are connected to those actors.
    """
    _ensure_db()

    q = (query or "").strip()
    if not q:
        return {"query": q, "center": None, "neighbors": [], "edges": []}

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    like = f"%{q.lower()}%"

    # 1) Prefer ACTOR nodes.
    cur.execute(
        """
        SELECT key, label, node_type
        FROM kg_nodes
        WHERE node_type = 'actor'
          AND LOWER(label) LIKE ?
        ORDER BY key
        LIMIT 5
        """,
        (like,),
    )
    rows = cur.fetchall()
    center_row = rows[0] if rows else None

    # 2) Fallback to INTEL nodes.
    if center_row is None:
        cur.execute(
            """
            SELECT key, label, node_type
            FROM kg_nodes
            WHERE node_type = 'intel'
              AND LOWER(label) LIKE ?
            ORDER BY key
            LIMIT 5
            """,
            (like,),
        )
        rows = cur.fetchall()
        center_row = rows[0] if rows else None

    # 3) Fallback to any node label.
    if center_row is None:
        cur.execute(
            """
            SELECT key, label, node_type
            FROM kg_nodes
            WHERE LOWER(label) LIKE ?
            ORDER BY key
            LIMIT 5
            """,
            (like,),
        )
        rows = cur.fetchall()
        center_row = rows[0] if rows else None

    if center_row is None:
        conn.close()
        return {"query": q, "center": None, "neighbors": [], "edges": []}

    center_key = center_row["key"]
    center = {
        "key": center_row["key"],
        "label": center_row["label"],
        "node_type": center_row["node_type"],
    }

    # Fetch edges where the center participates.
    cur.execute(
        """
        SELECT src, dst, edge_type, rowid AS edge_id
        FROM kg_edges
        WHERE src = ? OR dst = ?
        """,
        (center_key, center_key),
    )
    edge_rows = cur.fetchall()

    neighbor_keys: set[str] = set()
    edges: List[Dict[str, Any]] = []

    for e in edge_rows:
        src = e["src"]
        dst = e["dst"]
        edge_type = e["edge_type"] or ""
        neighbor_keys.add(src)
        neighbor_keys.add(dst)

        edge_id = f"{src}->{dst}::{edge_type}"
        edges.append(
            {
                "src": src,
                "dst": dst,
                "edge_type": edge_type,
                "id": edge_id,
            }
        )

    # Remove the center from the neighbor set so we only return true neighbors.
    neighbor_keys.discard(center_key)

    neighbors: List[Dict[str, Any]] = []
    if neighbor_keys:
        placeholders = ",".join("?" for _ in neighbor_keys)
        sql = f"""
            SELECT key, label, node_type
            FROM kg_nodes
            WHERE key IN ({placeholders})
        """
        cur.execute(sql, list(neighbor_keys))
        for n in cur.fetchall():
            neighbors.append(
                {
                    "key": n["key"],
                    "label": n["label"],
                    "node_type": n["node_type"],
                }
            )

    conn.close()

    return {
        "query": q,
        "center": center,
        "neighbors": neighbors,
        "edges": edges,
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

def geopolitics_lookup(topic: str) -> Dict[str, Any]:
    """
    Lookup geopolitical / sanctions / BRICS intel related to a topic.

    This demo implementation:
        - Extracts a small set of thematic keywords from the question.
        - Searches intel_articles for any rows whose topic/title/snippet/
          entities contain one or more of those keywords.

    In production this would call an internal search / intel system instead.
    """
    _ensure_db()

    raw = (topic or "").strip()
    lower = raw.lower()

    # Hand-crafted thematic keywords that matter for our seeded intel.
    keywords: List[str] = []

    if "china" in lower or "chinese" in lower:
        keywords.append("china")
    if "chip" in lower or "semiconductor" in lower or "fab" in lower:
        keywords.append("chip")
    if "sanction" in lower or "export control" in lower:
        keywords.append("sanction")
    if "brics" in lower:
        keywords.append("brics")
    if "cips" in lower or "renminbi" in lower or "yuan" in lower:
        keywords.append("cips")
    if "dollar" in lower or "dedollar" in lower:
        keywords.append("dollar")

    # Fallback: if nothing matched, just use a generic token so we at least try.
    if not keywords:
        keywords = ["china"]

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    where_clauses: List[str] = []
    params: List[Any] = []

    # Build OR of (topic/title/snippet/entities LIKE %kw%) for each keyword.
    for kw in keywords:
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
        "keywords": keywords,
        "matches": matches,
    }