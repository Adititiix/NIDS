"""High packet rate heuristic rule."""

from __future__ import annotations

from app.config import DetectionThresholds
from app.detection.models import RuleTrigger
from app.detection.rules.base import DetectionRule
from app.flows.models import FinalizedFlow


class HighPacketRateRule(DetectionRule):
    """
    Flags a flow whose overall packet rate (total_packets_per_second)
    exceeds a configurable threshold.

    This is a heuristic indicator, not proof of an attack: a legitimate
    high-throughput transfer (e.g. a large file download, a video call)
    can also have a high packet rate. It's meant to be combined with other
    rules and, later, ML confidence -- not acted on alone.

    Requires at least `rate_rule_min_packets` packets before evaluating the
    rate at all. Rate is packet_count/flow_duration_seconds, and a flow
    with only a couple of packets can have a flow_duration_seconds so tiny
    (microseconds) that the resulting rate is mathematically enormous but
    statistically meaningless -- observed in real Windows traffic
    validation with ordinary 2-3 packet HTTPS flows. This guard doesn't
    change the rate calculation itself, only when this rule trusts it.
    """

    name = "high_packet_rate"
    base_score = 30

    def __init__(self, thresholds: DetectionThresholds) -> None:
        self._threshold = thresholds.high_packet_rate_pps
        self._min_packets = thresholds.rate_rule_min_packets

    def evaluate(self, features: dict[str, float], flow: FinalizedFlow) -> RuleTrigger | None:
        if features["total_packets"] < self._min_packets:
            return None

        rate = features["total_packets_per_second"]
        if rate < self._threshold:
            return None
        return RuleTrigger(
            rule_name=self.name,
            reason=f"Flow packet rate {rate:.1f} pkt/s meets or exceeds the {self._threshold:.1f} pkt/s threshold.",
            score=self.base_score,
            evidence={
                "total_packets_per_second": rate,
                "total_packets": features["total_packets"],
                "flow_duration_seconds": features["flow_duration_seconds"],
            },
        )