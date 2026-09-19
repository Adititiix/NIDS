"""TCP flag combination anomaly heuristic rule."""

from __future__ import annotations

from app.capture.models import TransportProtocol
from app.config import DetectionThresholds
from app.detection.models import RuleTrigger
from app.detection.rules.base import DetectionRule
from app.flows.models import FinalizedFlow


class TcpFlagAnomalyRule(DetectionRule):
    """
    Flags TCP flag combinations that are unusual for ordinary application
    traffic but well-documented in scanning tooling:

    - SYN+FIN with no ACK anywhere in the flow: a legitimate TCP handshake
      always involves an ACK; a flow containing both SYN and FIN packets
      but never an ACK is a known crafted-packet scanning pattern.
    - "Null/Xmas"-style probes: no SYN and no ACK across the entire flow,
      but FIN, PSH, or URG flags are present. Ordinary connections almost
      always open with a SYN; packets carrying only FIN/PSH/URG with
      neither SYN nor ACK are characteristic of stealth scan techniques
      (e.g. Null and Xmas scans), though a truncated capture of a
      long-lived connection could theoretically also produce this.

    Both checks are heuristic pattern-matching on flag COUNTS aggregated
    over the whole flow, not packet-by-packet sequence analysis -- they
    indicate "this combination of flags appeared somewhere in the flow",
    not the exact order. Only evaluated for TCP flows with at least
    `flag_anomaly_min_packets` packets, to avoid flagging trivial flows.
    """

    name = "tcp_flag_anomaly"
    base_score = 40

    def __init__(self, thresholds: DetectionThresholds) -> None:
        self._min_packets = thresholds.flag_anomaly_min_packets

    def evaluate(self, features: dict[str, float], flow: FinalizedFlow) -> RuleTrigger | None:
        if flow.key.protocol != TransportProtocol.TCP:
            return None
        if features["total_packets"] < self._min_packets:
            return None

        syn = features["syn_count"]
        ack = features["ack_count"]
        fin = features["fin_count"]
        psh = features["psh_count"]
        urg = features["urg_count"]

        if syn > 0 and fin > 0 and ack == 0:
            return RuleTrigger(
                rule_name=self.name,
                reason=(
                    f"Flow contains {int(syn)} SYN and {int(fin)} FIN packets but zero ACKs -- "
                    "unusual for a normal TCP handshake, consistent with a crafted-packet scan pattern."
                ),
                score=self.base_score,
                evidence={"syn_count": syn, "fin_count": fin, "ack_count": ack},
            )

        if syn == 0 and ack == 0 and (fin > 0 or psh > 0 or urg > 0):
            return RuleTrigger(
                rule_name=self.name,
                reason=(
                    "Flow has no SYN or ACK packets but does have FIN/PSH/URG activity -- "
                    "pattern consistent with a Null/Xmas-style stealth scan probe."
                ),
                score=self.base_score,
                evidence={"syn_count": syn, "ack_count": ack, "fin_count": fin, "psh_count": psh, "urg_count": urg},
            )

        return None
