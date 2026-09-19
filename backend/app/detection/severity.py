"""
Shared risk-score -> Severity classification.

Extracted out of app.detection.rule_engine.RuleDetector so Phase 8's
HybridDetector can classify its own (rule+ML combined) risk score using
the exact same bucket boundaries, without duplicating the logic.
"""

from __future__ import annotations

from app.config import RiskScoringConfig
from app.detection.models import Severity


def classify_severity(risk_score: int, risk_scoring: RiskScoringConfig) -> Severity:
    """
    Map a 0-100 risk score to a Severity bucket using the existing
    RiskScoringConfig boundaries (0-29 LOW, 30-59 MEDIUM, 60-79 HIGH,
    80-100 CRITICAL by default) -- the same operational, not
    scientifically-derived, scheme documented since Phase 1.
    """
    if risk_score <= 0:
        return Severity.NONE
    if risk_score <= risk_scoring.low_max:
        return Severity.LOW
    if risk_score <= risk_scoring.medium_max:
        return Severity.MEDIUM
    if risk_score <= risk_scoring.high_max:
        return Severity.HIGH
    return Severity.CRITICAL
