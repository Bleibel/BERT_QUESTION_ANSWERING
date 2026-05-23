"""Entry-point launcher for the BERT QA demo.

Run with:
    python app.py

Then open http://127.0.0.1:5000 in your browser.
"""

import sys
from pathlib import Path

# Ensure demo package is importable
sys.path.insert(0, str(Path(__file__).parent))

from demo.app import app

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
