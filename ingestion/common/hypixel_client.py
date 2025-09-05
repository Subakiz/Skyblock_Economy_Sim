import os
import time
import threading
import requests
from typing import Any, Dict, Optional
from urllib.parse import urljoin

class TokenBucket:
    def __init__(self, rate_per_minute: int, burst: Optional[int] = None):
        self.capacity = burst or rate_per_minute
        self.tokens = self.capacity
        self.rate_per_sec = rate_per_minute / 60.0
        self.timestamp = time.monotonic()
        self.lock = threading.Lock()

    def consume(self, tokens: int = 1):
        with self.lock:
            now = time.monotonic()
            elapsed = now - self.timestamp
            self.timestamp = now
            self.tokens = min(self.capacity, self.tokens + elapsed * self.rate_per_sec)
            if self.tokens < tokens:
                # Sleep until enough tokens accumulate
                needed = tokens - self.tokens
                sleep_time = needed / self.rate_per_sec
                time.sleep(max(0.0, sleep_time))
                self.tokens = max(0.0, self.tokens - tokens)  # consume after sleep
            else:
                self.tokens -= tokens

class HypixelClient:
    def __init__(
        self,
        base_url: str,
        api_key: Optional[str],
        max_requests_per_minute: int = 110,
        timeout_seconds: int = 10,
        session: Optional[requests.Session] = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key or os.getenv("HYPIXEL_API_KEY")
        if not self.api_key:
            raise ValueError("HYPIXEL_API_KEY not provided")
        self.timeout = timeout_seconds
        self.bucket = TokenBucket(max_requests_per_minute)
        self.session = session or requests.Session()
        self.session.headers.update({"User-Agent": "Skyblock-Econ-Model/0.1"})

    def get_json(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        self.bucket.consume(1)
        url = urljoin(self.base_url + "/", endpoint.lstrip("/"))
        q = dict(params or {})
        q["key"] = self.api_key
        for attempt in range(5):
            try:
                resp = self.session.get(url, params=q, timeout=self.timeout)
                if resp.status_code == 429:
                    # Backoff on rate-limit
                    time.sleep(2 ** attempt)
                    continue
                resp.raise_for_status()
                data = resp.json()
                if not data.get("success", True):
                    # Some endpoints use success=false with cause/message
                    raise RuntimeError(f"Hypixel API error: {data}")
                return data
            except (requests.RequestException, ValueError) as e:
                if attempt == 4:
                    raise
                time.sleep(1.5 ** attempt)
        raise RuntimeError("Unreachable")