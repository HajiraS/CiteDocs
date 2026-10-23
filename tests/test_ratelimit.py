"""Sliding-window rate limiter, driven by a fake clock."""

import pytest

from app import ratelimit
from app.ratelimit import RateLimiter


class Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def test_allows_up_to_the_limit_then_reports_the_wait():
    clock = Clock()
    limiter = RateLimiter(limit=2, window_seconds=60, clock=clock)
    assert limiter.check("a") == 0.0
    clock.now = 10
    assert limiter.check("a") == 0.0
    assert limiter.check("a") == pytest.approx(50.0)  # oldest hit leaves the window at t=60


def test_window_slides_so_requests_are_allowed_again():
    clock = Clock()
    limiter = RateLimiter(limit=1, window_seconds=60, clock=clock)
    assert limiter.check("a") == 0.0
    assert limiter.check("a") > 0
    clock.now = 61
    assert limiter.check("a") == 0.0


def test_clients_are_limited_independently():
    limiter = RateLimiter(limit=1, window_seconds=60, clock=Clock())
    assert limiter.check("a") == 0.0
    assert limiter.check("b") == 0.0
    assert limiter.check("a") > 0


def test_idle_clients_are_swept_so_memory_stays_bounded(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(ratelimit, "_SWEEP_THRESHOLD", 1)
    clock = Clock()
    limiter = RateLimiter(limit=5, window_seconds=60, clock=clock)
    limiter.check("a")
    limiter.check("b")
    clock.now = 100
    limiter.check("c")
    assert list(limiter._hits) == ["c"]
