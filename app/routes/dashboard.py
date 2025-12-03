# johny5_agentic/app/routes/dashboard.py
"""
Flask blueprint for the Johny 5 Agentic dashboard.

This module:
- Defines the main routes (Overview, DB, Vectors) exactly once.
- Provides POST-only action routes for pipeline triggers.
- Exposes /api/ask as the main agentic endpoint.
- Uses a lazy import helper S() to avoid circular imports with backend code.
"""

from __future__ import annotations

import os
import sqlite3
from typing import Any, Dict, List

from flask import (
    Blueprint,
    current_app,
    flash,
    jsonify,
    redirect,
    render_template,
    request,
    session,
    url_for,
)

from app.backend import DB_PATH, get_stats as backend_get_stats

bp = Blueprint("dashboard", __name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def S():
    """
    Lazily import and return the backend service module.

    This helper prevents circular imports by resolving backend only at
    call time, never at import time of this routes module.

    Returns:
        The imported app.backend module (aliased as s).
    """
    import app.backend as s  # noqa: WPS433

    return s


def _get_stats() -> Dict[str, int]:
    """
    Fetch aggregated stats for the dashboard cards.

    This wraps backend_get_stats() so all routes use a single helper.
    """
    return backend_get_stats()


def _inspect_db(max_rows: int = 1) -> List[Dict[str, Any]]:
    """
    Inspect the SQLite database and collect simple table metadata.

    Args:
        max_rows: Maximum number of sample rows to fetch per table.

    Returns:
        A list of dictionaries with:
            name:   table name
            count:  row count
            sample: a single sample row as dict (or None if table empty)
    """
    tables: List[Dict[str, Any]] = []

    if not DB_PATH.exists():
        return tables

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    cur.execute(
        "SELECT name FROM sqlite_master "
        "WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
    )
    for row in cur.fetchall():
        name = row["name"]
        try:
            cur.execute(f"SELECT COUNT(1) AS cnt FROM {name}")
            count = int(cur.fetchone()["cnt"])
        except sqlite3.Error as exc:  # noqa: BLE001
            current_app.logger.warning(
                "Failed to count table %s: %s", name, exc
            )
            count = 0

        sample: Any = None
        if count and max_rows > 0:
            try:
                cur.execute(f"SELECT * FROM {name} LIMIT ?", (max_rows,))
                row_sample = cur.fetchone()
                if row_sample is not None:
                    sample = dict(row_sample)
            except sqlite3.Error:
                sample = None

        tables.append({"name": name, "count": count, "sample": sample})

    conn.close()
    return tables


# ---------------------------------------------------------------------------
# Page routes
# ---------------------------------------------------------------------------

@bp.get("/")
def index():
    """
    Render the main dashboard overview.

    Template:
        index.html

    Context:
        s: dictionary of global stats (companies, articles, KG nodes/edges, candles)
    """
    stats = _get_stats()
    return render_template("index.html", s=stats)


@bp.get("/db")
def db_tab():
    """
    Render the database explorer tab.

    Template:
        db.html

    Context:
        s: stats dictionary
        tables: list of table metadata from _inspect_db()
    """
    stats = _get_stats()
    tables = _inspect_db()
    return render_template("db.html", s=stats, tables=tables)


@bp.get("/vectors")
def vectors_tab():
    """
    Render the vectors / RAG playground tab.

    Template:
        vectors.html

    Context:
        s: stats dictionary
    """
    stats = _get_stats()
    return render_template("vectors.html", s=stats)


# ---------------------------------------------------------------------------
# Action routes (POST-only)
# ---------------------------------------------------------------------------

@bp.post("/action/pull_profiles")
def action_pull_profiles():
    """
    Trigger profile ingestion via backend.pull_profiles().

    After running, this route redirects back to the index page and
    uses flash messages to signal success or failure.
    """
    svc = S()
    try:
        svc.pull_profiles()
        flash("Profiles pull triggered.", "success")
    except Exception as exc:  # noqa: BLE001
        current_app.logger.exception("pull_profiles failed: %s", exc)
        flash(f"Error pulling profiles: {exc}", "error")
    return redirect(url_for("dashboard.index"))


@bp.post("/action/build_graph")
def action_build_graph():
    """
    Trigger knowledge graph build via backend.build_graph_from_profiles().

    After running, this route redirects back to the index page and
    uses flash messages to signal success or failure.
    """
    svc = S()
    try:
        svc.build_graph_from_profiles()
        flash("Knowledge graph build triggered.", "success")
    except Exception as exc:  # noqa: BLE001
        current_app.logger.exception("build_graph failed: %s", exc)
        flash(f"Error building graph: {exc}", "error")
    return redirect(url_for("dashboard.index"))


@bp.post("/action/pull_candles")
def action_pull_candles():
    """
    Trigger OHLCV ingestion via backend.pull_candles_for_tickers().

    After running, this route redirects back to the index page and
    uses flash messages to signal success or failure.
    """
    svc = S()
    try:
        svc.pull_candles_for_tickers()
        flash("Candle ingestion triggered.", "success")
    except Exception as exc:  # noqa: BLE001
        current_app.logger.exception("pull_candles failed: %s", exc)
        flash(f"Error pulling candles: {exc}", "error")
    return redirect(url_for("dashboard.index"))


@bp.post("/action/pull_sector")
def action_pull_sector():
    """
    Trigger sector universe ingestion via backend.pull_sector().

    Expects:
        form["sector"]: sector code such as "XLE". Defaults to "XLE" if absent.

    After running, this route redirects back to the index page and
    uses flash messages to signal success or failure.
    """
    sector = (request.form.get("sector") or "XLE").strip().upper()
    svc = S()
    try:
        svc.pull_sector(sector)
        flash(f"Sector pull triggered for {sector}.", "success")
    except Exception as exc:  # noqa: BLE001
        current_app.logger.exception("pull_sector failed: %s", exc)
        flash(f"Error pulling sector {sector}: {exc}", "error")
    return redirect(url_for("dashboard.index"))


@bp.post("/action/rebuild_all")
def action_rebuild_all():
    """
    Trigger the full pipeline via backend.rebuild_all().

    This usually runs:
        profiles → graph → candles → news

    After running, this route redirects back to the index page and
    uses flash messages to signal success or failure.
    """
    svc = S()
    try:
        svc.rebuild_all()
        flash("Full rebuild pipeline triggered.", "success")
    except Exception as exc:  # noqa: BLE001
        current_app.logger.exception("rebuild_all failed: %s", exc)
        flash(f"Error running rebuild_all: {exc}", "error")
    return redirect(url_for("dashboard.index"))


@bp.post("/action/ingest_news")
def action_ingest_news():
    """
    Trigger news ingestion via backend.ingest_news().

    After running, this route redirects back to the index page and
    uses flash messages to signal success or failure.
    """
    svc = S()
    try:
        svc.ingest_news()
        flash("News ingestion triggered.", "success")
    except Exception as exc:  # noqa: BLE001
        current_app.logger.exception("ingest_news failed: %s", exc)
        flash(f"Error ingesting news: {exc}", "error")
    return redirect(url_for("dashboard.index"))


@bp.post("/action/save_gemini_key")
def save_gemini_key():
    """
    Persist a Gemini API key into the session and environment.

    Behavior:
        - Reads gemini_api_key from POST form data.
        - Stores it in Flask session["gemini_api_key"].
        - Stores it in os.environ["GEMINI_API_KEY"].
        - Redirects back to the index with a flash message.
    """
    key = (request.form.get("gemini_api_key") or "").strip()

    if not key:
        flash("No Gemini API key provided.", "error")
        return redirect(url_for("dashboard.index"))

    session["gemini_api_key"] = key
    os.environ["GEMINI_API_KEY"] = key
    flash("Gemini API key saved for this session.", "success")
    return redirect(url_for("dashboard.index"))


# ---------------------------------------------------------------------------
# API routes
# ---------------------------------------------------------------------------

@bp.post("/api/ask")
def api_ask():
    """
    Primary agentic endpoint for the UI.

    Expects JSON body:
        { "question": "<analyst question>" }

    Response:
        On success:
            { "ok": true, "data": { ...llm_analyze payload... } }
        On error:
            { "ok": false, "error": "<message>" }
    """
    payload = request.get_json(silent=True) or {}
    question = (payload.get("question") or "").strip()

    if not question:
        return jsonify({"ok": False, "error": "Missing 'question'"}), 400

    svc = S()
    try:
        result = svc.llm_analyze(question)
    except Exception as exc:  # noqa: BLE001
        current_app.logger.exception("llm_analyze failed: %s", exc)
        return jsonify({"ok": False, "error": str(exc)}), 500

    return jsonify({"ok": True, "data": result})
