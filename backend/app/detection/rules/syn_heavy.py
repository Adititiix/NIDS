"""SYN-dominant traffic heuristic rule."""

from __future__ import annotations

from app.config import DetectionThresholds
from app.detection.models import RuleTrigger
from app.detection.rules.base import DetectionRule
from app.flows.models import FinalizedFlow


class SynHeavyRule(DetectionRule):
    """
    Flags a flow whose packets are overwhelmingly TCP SYNs -- a pattern
    associated with SYN scanning or a SYN-flood attempt, but also produced
    by a perfectly ordinary unanswered/blocked connection attempt (e.g. a
    client retrying against a closed or firewalled port).

    Requires a minimum packet count (`syn_heavy_min_packets`) so a 1-2
    packet flow doesn't trigger purely on ratio noise -- a single SYN with
    no reply would otherwise be a 100% "SYN ratio" trivially.
    """

    name = "syn_heavy"
    base_score = 35

    def __init__(self, thresholds: DetectionThresholds) -> None:
        self._ratio_threshold = thresholds.syn_heavy_ratio_threshold
        self._min_packets = thresholds.syn_heavy_min_packets

    def evaluate(self, features: dict[str, float], flow: FinalizedFlow) -> RuleTrigger | None:
        total_packets = features["total_packets"]
        syn_count = features["syn_count"]

        if total_packets < self._min_packets or syn_count <= 0:
            return None

        ratio = syn_count / total_packets
        if ratio < self._ratio_threshold:
            return None

        return RuleTrigger(
            rule_name=self.name,
            reason=(
                f"SYN packets make up {ratio:.0%} of {int(total_packets)} total packets "
                f"(threshold {self._ratio_threshold:.0%}), with only {int(features['ack_count'])} ACKs observed."
            ),
            score=self.base_score,
            evidence={
                "syn_count": syn_count,
                "ack_count": features["ack_count"],
                "total_packets": total_packets,
                "syn_ratio": ratio,
            },
        )
