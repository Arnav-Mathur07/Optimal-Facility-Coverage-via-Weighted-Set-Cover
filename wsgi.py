"""
WSGI / ASGI Fallback Entrypoint for Vercel Serverless Runtime.
Serves the MotionSites Cinematic Web Simulator if Vercel's Python engine is invoked.
Requires 0 external pip dependencies (pure Python standard library).
"""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
INDEX_FILE = BASE_DIR / "index.html"

def app(environ, start_response):
    """Standard WSGI entrypoint callable expected by Vercel."""
    if not INDEX_FILE.exists():
        start_response("404 Not Found", [("Content-Type", "text/plain; charset=utf-8")])
        return [b"index.html not found"]
    
    with open(INDEX_FILE, "rb") as f:
        content = f.read()
        
    start_response("200 OK", [
        ("Content-Type", "text/html; charset=utf-8"),
        ("Content-Length", str(len(content))),
        ("Cache-Control", "public, max-age=3600")
    ])
    return [content]

# Aliases expected by various serverless runtimes
application = app
handler = app

if __name__ == "__main__":
    from wsgiref.simple_server import make_server
    print("Serving on http://localhost:8000 ...")
    server = make_server("127.0.0.1", 8000, app)
    server.serve_forever()
