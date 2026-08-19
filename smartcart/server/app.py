"""HTTP API server and static web dashboard handler."""

import argparse
from http.server import HTTPServer, SimpleHTTPRequestHandler
import json
from pathlib import Path
import sys
from urllib.parse import parse_qs, urlparse

# Ensure repository root is on sys.path
repo_root = Path(__file__).resolve().parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from smartcart.server.service import RecommendationService


class SmartCartRequestHandler(SimpleHTTPRequestHandler):
    """Custom HTTP request handler with REST API routing and CORS support."""

    service: RecommendationService = None
    web_dir: Path = None

    def end_headers(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        super().end_headers()

    def do_OPTIONS(self) -> None:
        self.send_response(200)
        self.end_headers()

    def do_GET(self) -> None:
        parsed_url = urlparse(self.path)
        path = parsed_url.path
        query_params = parse_qs(parsed_url.query)

        if path == "/api/health":
            self._send_json({"status": "healthy", "service": "smartcart-recommender"})
        elif path == "/api/catalog":
            category = query_params.get("category", [None])[0]
            search = query_params.get("search", [None])[0]
            limit = int(query_params.get("limit", [100])[0])
            items = self.service.get_catalog_items(category=category, search=search, limit=limit)
            self._send_json({"items": items, "total": len(items)})
        elif path == "/api/categories":
            categories = ["All"] + list(self.service.catalog.CATEGORIES)
            self._send_json({"categories": categories})
        elif path == "/api/benchmark":
            self._send_json({"benchmarks": self.service.benchmark_data})
        else:
            # Serve static files from web_dir
            if self.web_dir and self.web_dir.exists():
                if path == "/" or path == "":
                    req_file = self.web_dir / "index.html"
                else:
                    clean_path = path.lstrip("/")
                    req_file = self.web_dir / clean_path

                if req_file.exists() and req_file.is_file():
                    content_type = "text/html"
                    if req_file.suffix == ".css":
                        content_type = "text/css"
                    elif req_file.suffix == ".js":
                        content_type = "application/javascript"
                    elif req_file.suffix == ".json":
                        content_type = "application/json"

                    self.send_response(200)
                    self.send_header("Content-Type", content_type)
                    self.end_headers()
                    with open(req_file, "rb") as f:
                        self.wfile.write(f.read())
                    return

            self.send_error(404, f"Path not found: {path}")

    def do_POST(self) -> None:
        parsed_url = urlparse(self.path)
        path = parsed_url.path

        content_len = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_len).decode("utf-8") if content_len > 0 else "{}"
        try:
            payload = json.loads(body)
        except Exception:
            payload = {}

        if path == "/api/recommend":
            user_id = int(payload.get("user_id", 0))
            cart_items = [int(i) for i in payload.get("cart_items", [])]
            top_k = int(payload.get("top_k", 4))

            recs = self.service.get_recommendations(
                user_id=user_id,
                cart_item_ids=cart_items,
                top_k=top_k,
            )
            self._send_json({"user_id": user_id, "recommendations": recs})

        elif path == "/api/simulate-ab":
            num_users = int(payload.get("num_users", 5000))
            traffic_split = float(payload.get("traffic_split", 0.5))

            try:
                result = self.service.run_live_ab_simulation(
                    num_users=num_users,
                    traffic_split=traffic_split,
                )
                self._send_json(result)
            except Exception as e:
                self._send_json({"error": str(e)}, status=500)
        else:
            self.send_error(404, f"API endpoint not found: {path}")

    def _send_json(self, data: Any, status: int = 200) -> None:
        response_bytes = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(response_bytes)))
        self.end_headers()
        self.wfile.write(response_bytes)


def run_server(host: str = "127.0.0.1", port: int = 8000) -> None:
    web_dir = Path(__file__).resolve().parent.parent / "web"
    service = RecommendationService()

    SmartCartRequestHandler.service = service
    SmartCartRequestHandler.web_dir = web_dir

    server_address = (host, port)
    httpd = HTTPServer(server_address, SmartCartRequestHandler)
    print(f"Smart Cart Recommender Server running at http://{host}:{port}/")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server...")
        httpd.server_close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Start Smart Cart API Server & Web UI.")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host address")
    parser.add_argument("--port", type=int, default=8000, help="Port number")
    args = parser.parse_args()

    run_server(host=args.host, port=args.port)
