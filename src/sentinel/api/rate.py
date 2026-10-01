import math
import threading
from collections.abc import Callable


class TokenBucket:
    def __init__(
        self,
        rate: float,
        burst: int,
        clock: Callable[[], float],
        idle_ttl: float = 3600,
        max_buckets: int = 10000,
    ) -> None:
        self.rate = rate
        self.burst = burst
        self.clock = clock
        self.idle_ttl = idle_ttl
        self.max_buckets = max_buckets
        self.buckets: dict[str, tuple[float, float]] = {}
        self.lock = threading.Lock()

    def consume(self, identity: str) -> tuple[bool, int, int]:
        with self.lock:
            now = self.clock()
            for name, (balance, updated_at) in list(self.buckets.items()):
                if (now - updated_at) >= self.idle_ttl and (
                    balance + max(0, now - updated_at) * self.rate >= self.burst
                ):
                    del self.buckets[name]
            if identity not in self.buckets and len(self.buckets) >= self.max_buckets:
                oldest = min(self.buckets, key=lambda name: self.buckets[name][1])
                del self.buckets[oldest]
            tokens, updated = self.buckets.get(identity, (float(self.burst), now))
            tokens = min(self.burst, tokens + max(0, now - updated) * self.rate)
            allowed = tokens >= 1
            if allowed:
                tokens -= 1
            self.buckets[identity] = (tokens, now)
            retry = 0 if allowed else max(1, math.ceil((1 - tokens) / self.rate))
            return allowed, math.floor(tokens), retry
