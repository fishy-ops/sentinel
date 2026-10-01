import math
import threading
from collections.abc import Callable


class TokenBucket:
    def __init__(self, rate: float, burst: int, clock: Callable[[], float]) -> None:
        self.rate = rate
        self.burst = burst
        self.clock = clock
        self.buckets: dict[str, tuple[float, float]] = {}
        self.lock = threading.Lock()

    def consume(self, identity: str) -> tuple[bool, int, int]:
        with self.lock:
            now = self.clock()
            tokens, updated = self.buckets.get(identity, (float(self.burst), now))
            tokens = min(self.burst, tokens + max(0, now - updated) * self.rate)
            allowed = tokens >= 1
            if allowed:
                tokens -= 1
            self.buckets[identity] = (tokens, now)
            retry = 0 if allowed else max(1, math.ceil((1 - tokens) / self.rate))
            return allowed, math.floor(tokens), retry
