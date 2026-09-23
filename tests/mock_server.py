"""Run PaperFeed with offline fixtures: python tests/mock_server.py  → http://127.0.0.1:8000"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import uvicorn  # noqa: E402

import app  # noqa: E402
import fixtures  # noqa: E402

fixtures.install()
if __name__ == "__main__":
    uvicorn.run(app.app, host="127.0.0.1", port=int(sys.argv[1]) if len(sys.argv) > 1 else 8000)
