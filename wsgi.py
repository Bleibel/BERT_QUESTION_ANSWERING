"""WSGI entry point for production deployment (e.g., Gunicorn).

Usage:
    gunicorn --bind 0.0.0.0:8000 --workers 2 --timeout 120 wsgi:app
"""

from demo.app import app

if __name__ == "__main__":
    app.run()
