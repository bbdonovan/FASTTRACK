# johny5_agentic/run.py
"""Entry point for the Johny 5 Agentic Graph RAG prototype."""

from app import create_app

app = create_app()


if __name__ == "__main__":
    # For work you may want debug=False and a different host/port.
    app.run(debug=True, port=5000)