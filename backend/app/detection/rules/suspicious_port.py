"""Suspicious destination port heuristic rule."""

from __future__ import annotations

from app.config import DetectionThresholds
from app.detection.models import RuleTrigger
from app.detection.rules.base import DetectionRule
from app.flows.models import FinalizedFlow


class SuspiciousPortRule(DetectionRule):
    """
    Flags a flow whose destination port is on a small, configurable list
    of ports commonly associated with high-risk services or known attack
    tooling defaults.

    This is the weakest signal in the rule set by design (hence the lowest
    base_score): a destination port alone proves nothing -- Telnet, SMB,
    and RDP are all legitimately used within many networks, and this rule
    cannot tell a legitimate administrative connection from a malicious
    one. It exists to surface a flow for review, not to accuse it.
    """

    name = "suspicious_destination_port"
    base_score = 10

    def __init__(self, thresholds: DetectionThresholds) -> None:
        self._suspicious_ports = set(thresholds.suspicious_destination_ports)

    def evaluate(self, features: dict[str, float], flow: FinalizedFlow) -> RuleTrigger | None:
        dst_port = features["dst_port"]
        if dst_port not in self._suspicious_ports:
            return None

        return RuleTrigger(
            rule_name=self.name,
            reason=(
                f"Destination port {int(dst_port)} is on the configured suspicious-ports list. "
                "This alone does not indicate malicious activity."
            ),
            score=self.base_score,
            evidence={"dst_port": dst_port},
        )
