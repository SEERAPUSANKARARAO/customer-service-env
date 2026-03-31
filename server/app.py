"""
server/app.py
OpenEnv standard server entrypoint.
Imports the FastAPI app from app.main and exposes it for uvicorn.
"""

import uvicorn
from app.main import app  # noqa: F401  — re-exported for OpenEnv compatibility


def main():
    """Entry point for [project.scripts] server command."""
    uvicorn.run("app.main:app", host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
