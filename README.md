# Johny 5 Agentic — Graph RAG Prototype

This is a minimal, **agentic-first** version of the Johny 5 dashboard.

The goal is to demonstrate how an **Agentic Graph RAG** system can:
- Ingest company data (profiles, candles, news).
- Build a simple knowledge graph.
- Answer analyst questions via an agentic endpoint that:
  - orchestrates tools, and
  - returns both an **answer** and a **provenance trace**.

---

## 1. Tech Stack

- **Backend:** Flask (Python)
- **Database:** SQLite (`j5_agentic.db` in the project root)
- **Graph / Vector (future):** plug in NetworkX, FAISS, SentenceTransformers
- **Orchestration:** your internal LLM / LangGraph agent, wired into `llm_analyze`
- **Frontend:** HTML + Jinja2 + vanilla CSS (dark mode)

---

## 2. Project Layout

```text
johny5_agentic/
├── run.py               # Entry point
├── requirements.txt     # Python dependencies
├── README.md            # This file
└── app/
    ├── __init__.py      # Flask app factory
    ├── backend.py       # Service layer + agentic entrypoint
    ├── routes/
    │   ├── __init__.py
    │   └── dashboard.py # All routes defined exactly once
    └── templates/
        ├── base.html    # Layout
        ├── index.html   # Main dashboard (answer + trace)
        ├── db.html      # DB explorer
        └── vectors.html # Agentic playground
