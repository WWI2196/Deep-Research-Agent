#!/usr/bin/env python3
"""Entry point for the Deep Research Agent backend."""

import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "backend.server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
