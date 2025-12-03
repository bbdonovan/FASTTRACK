# johny5_agentic/app/__init__.py
"""Flask application factory for the Johny 5 Agentic prototype."""

from flask import Flask


def create_app() -> Flask:
    """
    Create and configure the Flask application instance.

    This function:
    - Instantiates the Flask app.
    - Sets a development secret key.
    - Registers the dashboard blueprint.

    Returns:
        A configured Flask application object.
    """
    app = Flask(__name__)
    # Simple dev secret key (change or externalize for production).
    app.secret_key = "dev_key"

    # Import and register blueprints.
    from app.routes.dashboard import bp as dashboard_bp  # noqa: WPS433

    app.register_blueprint(dashboard_bp)

    # Optional: adjust logging verbosity.
    app.logger.setLevel("INFO")

    return app
