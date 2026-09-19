"""Short-duration, high-packet-count "burst" heuristic rule."""

from __future__ import annotations

from app.config import DetectionThresholds
from app.detection.models import RuleTrigger
from app.detection.rules.base import DetectionRule
from app.flows.models import FinalizedFlow


class ShortHighVolumeRule(DetectionRule):
    """
    Flags a flow that packed an unusually large number of packets into an
    unusually short duration -- a "burst" pattern. This is distinct from
    HighPacketRateRule: a flow can have a high average rate over a long
    duration without this pattern, and a short burst might not cross the
    rate threshold if it's just barely over the duration cutoff. Evaluating
    both duration and count directly (rather than only their ratio) avoids
    a single edge case masking the other.
    """

    name = "short_high_volume_burst"
    base_score = 20

    def __init__(self, thresholds: DetectionThresholds) -> None:
        self._max_duration = thresholds.short_flow_max_duration_seconds
        self._min_packets = thresholds.short_flow_min_packets

    def evaluate(self, features: dict[str, float], flow: FinalizedFlow) -> RuleTrigger | None:
        duration = features["flow_duration_seconds"]
        total_packets = features["total_packets"]

        if duration > self._max_duration or total_packets < self._min_packets:
            return None

        return RuleTrigger(
            rule_name=self.name,
            reason=(
                f"{int(total_packets)} packets arrived within {duration:.3f}s "
                f"(<= {self._max_duration:.3f}s threshold, >= {self._min_packets} packet threshold)."
            ),
            score=self.base_score,
            evidence={"flow_duration_seconds": duration, "total_packets": total_packets},
        )
