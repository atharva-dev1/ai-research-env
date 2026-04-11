"""
server/app.py — OpenEnv-required entry point.
Re-exports the FastAPI app from backend.server for multi-mode deployment.
"""
import os, sys

# Ensure project root is on PYTHONPATH
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from backend.server import app  # noqa: F401 — re-export


def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
