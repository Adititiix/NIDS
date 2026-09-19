"""High byte rate heuristic rule."""

from __future__ import annotations

from app.config import DetectionThresholds
from app.detection.models import RuleTrigger
from app.detection.rules.base import DetectionRule
from app.flows.models import FinalizedFlow


class HighByteRateRule(DetectionRule):
    """
    Flags a flow whose overall throughput (total_bytes_per_second) exceeds
    a configurable threshold.

    Independent of HighPacketRateRule: a flow can have few, large packets
    (high byte rate, moderate packet rate) or many small packets (high
    packet rate, moderate byte rate) -- both patterns are worth surfacing
    separately. Heuristic indicator only; large legitimate transfers will
    also trigger this.

    Requires at least `rate_rule_min_packets` packets before evaluating the
    rate at all -- see HighPacketRateRule's docstring for why a tiny flow's
    rate (byte_count/flow_duration_seconds here) can be mathematically
    enormous but statistically meaningless.
    """

    name = "high_byte_rate"
    base_score = 25

    def __init__(self, thresholds: DetectionThresholds) -> None:
        self._threshold = thresholds.high_byte_rate_bps
        self._min_packets = thresholds.rate_rule_min_packets

    def evaluate(self, features: dict[str, float], flow: FinalizedFlow) -> RuleTrigger | None:
        if features["total_packets"] < self._min_packets:
            return None

        rate = features["total_bytes_per_second"]
        if rate < self._threshold:
            return None
        return RuleTrigger(
            rule_name=self.name,
            reason=f"Flow byte rate {rate:.0f} bytes/s meets or exceeds the {self._threshold:.0f} bytes/s threshold.",
            score=self.base_score,
            evidence={
                "total_bytes_per_second": rate,
                "total_bytes": features["total_bytes"],
                "flow_duration_seconds": features["flow_duration_seconds"],
            },
        )