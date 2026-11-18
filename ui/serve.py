#!/usr/bin/env python3
"""Static file server for the PathSnap evaluator UI (laptop 2).

The UI is plain HTML/CSS/JavaScript with Leaflet bundled under ui/vendor/, so
it needs no Python web framework - only the standard library. It talks to the
FastAPI service on laptop 1 over HTTP (default http://localhost:8000, editable
in the page header or via ?api=http://<laptop-1-ip>:8000).

Usage (from the project root):

    python3 ui/serve.py [--port 8501] [--bind 127.0.0.1]
"""
from __future__ import annotations

import argparse
import functools
import http.server
import socketserver
import sys
from pathlib import Path

UI_DIR = Path(__file__).resolve().parent


class Handler(http.server.SimpleHTTPRequestHandler):
    """Static handler rooted at the ui/ directory."""

    def end_headers(self) -> None:
        # Always serve fresh files so a browser refresh picks up edits.
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def log_message(self, fmt: str, *args) -> None:
        sys.stdout.write("[ui] %s - %s\n" % (self.address_string(), fmt % args))
        sys.stdout.flush()


class Server(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Serve the PathSnap evaluator UI (run on laptop 2)."
    )
    parser.add_argument("--port", type=int, default=8501, help="port to listen on (default 8501)")
    parser.add_argument(
        "--bind",
        default="127.0.0.1",
        help="interface to bind (default 127.0.0.1 = this laptop only)",
    )
    args = parser.parse_args()

    handler = functools.partial(Handler, directory=str(UI_DIR))
    try:
        with Server((args.bind, args.port), handler) as httpd:
            url = "http://localhost:%d/" % args.port
            print("PathSnap evaluator UI served from %s" % UI_DIR)
            print("Open %s" % url)
            print("API base URL defaults to http://localhost:8000; set it in the page header,")
            print("or open %s?api=http://<laptop-1-ip>:8000" % url)
            print("Press Ctrl+C to stop.")
            sys.stdout.flush()
            httpd.serve_forever()
    except OSError as exc:
        print(
            "Could not start the UI server on %s:%d - %s" % (args.bind, args.port, exc),
            file=sys.stderr,
        )
        print("Is another process already using port %d?" % args.port, file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nStopped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
