#!/usr/bin/env python3
"""VisualATC – ATC Stream Transcriber. Entry point."""

import argparse
import sys
import uvicorn


def main():
    parser = argparse.ArgumentParser(description="VisualATC – ATC Stream Transcriber")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8765, help="Port to bind (default: 8765)")
    parser.add_argument("--reload", action="store_true", help="Auto-reload on code changes (dev)")
    args = parser.parse_args()

    print(f"\n  VisualATC – ATC Stream Transcriber")
    print(f"  http://{args.host}:{args.port}\n")

    uvicorn.run(
        "server.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )


if __name__ == "__main__":
    main()
