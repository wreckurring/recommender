"""HTTP API server and service runtime for Smart Cart."""

from smartcart.server.app import run_server
from smartcart.server.service import RecommendationService

__all__ = ["RecommendationService", "run_server"]
