"""
Streaming (online) descriptive statistics.

Long-lived, high-rate flows can accumulate a very large number of packets
before they expire (up to `flow_timeout_seconds` of idle time). Storing raw
per-packet samples (lengths, inter-arrival times) for the lifetime of a flow
would mean unbounded memory growth per flow. `RunningStats` instead keeps
O(1) accumulators (count, sum, sum-of-squares, min, max) and derives
mean/std/min/max from those -- the standard streaming approach.

Deliberately placed in a neutral `app.common` package rather than under
`app.flows` or `app.features`: it's used by the flow aggregator (Phase 3) to
build up per-flow distributions, and conceptually belongs to neither layer
specifically.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class RunningStats:
    """
    Streaming mean/std/min/max over a sequence of values added one at a time.

    Standard deviation is POPULATION standard deviation (divide by n, not
    n-1) -- this matches numpy's default (`ddof=0`) and is the convention
    CICFlowMeter-derived datasets (e.g. CIC-IDS2017) use, so live-computed
    values stay comparable to values derived from the training dataset in
    later phases.

    With zero or one observed value, `std` is defined as 0.0 (no variance is
    observable from fewer than two samples) and `min`/`max`/`mean` are
    likewise defined as 0.0 for zero observations -- never NaN or infinity.
    This is a deliberate modeling choice, not an oversight; see
    docs/methodology.md for the full feature contract this supports.
    """

    count: int = 0
    _sum: float = field(default=0.0, repr=False)
    _sum_sq: float = field(default=0.0, repr=False)
    _min: float = field(default=math.inf, repr=False)
    _max: float = field(default=-math.inf, repr=False)

    def add(self, value: float) -> None:
        self.count += 1
        self._sum += value
        self._sum_sq += value * value
        if value < self._min:
            self._min = value
        if value > self._max:
            self._max = value

    @property
    def mean(self) -> float:
        if self.count == 0:
            return 0.0
        return self._sum / self.count

    @property
    def std(self) -> float:
        if self.count < 2:
            return 0.0
        variance = (self._sum_sq / self.count) - (self.mean ** 2)
        # Floating-point cancellation can push a near-zero true variance
        # slightly negative -- clamp rather than let sqrt() raise/NaN.
        return math.sqrt(max(variance, 0.0))

    @property
    def min(self) -> float:
        return 0.0 if self.count == 0 else self._min

    @property
    def max(self) -> float:
        return 0.0 if self.count == 0 else self._max
