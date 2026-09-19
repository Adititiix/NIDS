"""
Centralized configuration for the NIDS backend.

Nothing in this project should hardcode a network interface, database URL,
model path, threshold, or timeout. Everything is loaded from environment
variables (with sane lab-safe defaults), typically supplied via a `.env`
file in local/VM development. See `.env.example` for the full list.

This module is intentionally the ONLY place that reads os.environ directly.
Every other module should import `settings` from here.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class DetectionThresholds(BaseSettings):
    """
    Rule-based detection thresholds.

    These are operational defaults chosen to be reasonable for a small lab
    network, NOT values derived from a formal ROC analysis. They are meant
    to be tuned per-environment and are documented individually in
    docs/methodology.md once Phase 5 (rule engine) lands.
    """

    # --- Reserved for a future cross-flow/multi-flow correlation layer ---
    # These fields describe behavior of a SOURCE IP across MANY flows over
    # a time window (e.g. "how many distinct ports has this IP touched in
    # the last 10 seconds"). Phase 5's RuleDetector operates on a SINGLE
    # FinalizedFlow at a time and has no cross-flow state, so these are
    # NOT read by any Phase 5 rule yet -- they're kept here, unused, for
    # whichever future phase adds that stateful tracking layer.
    port_scan_distinct_ports: int = Field(
        default=15, description="Distinct destination ports from one source IP within the window that triggers MEDIUM severity."
    )
    port_scan_distinct_ports_high: int = Field(
        default=40, description="Distinct destination ports from one source IP within the window that triggers HIGH severity."
    )
    port_scan_window_seconds: int = Field(default=10, description="Sliding window for port-scan counting.")

    high_conn_rate_per_sec: float = Field(
        default=20.0, description="New connections/sec from a single source IP considered anomalous."
    )

    syn_flood_ratio_threshold: float = Field(
        default=5.0,
        description="SYN-to-completed-handshake ratio considered a heuristic indicator (NOT a confirmed DDoS classification).",
    )

    packet_rate_zscore_threshold: float = Field(
        default=3.0, description="Rolling z-score of packets/sec from a source IP considered anomalous."
    )

    # --- Phase 5: single-flow, stateless rule thresholds ---
    # These ARE used by app.detection.rules -- each operates on one flow's
    # FeatureVector alone, no cross-flow state required.
    high_packet_rate_pps: float = Field(
        default=100.0,
        description="total_packets_per_second above which a single flow is flagged as unusually fast. "
        "A legitimate bulk transfer can also exceed this -- heuristic, not proof of an attack.",
    )
    high_byte_rate_bps: float = Field(
        default=1_000_000.0,
        description="total_bytes_per_second (bytes/sec) above which a single flow is flagged as high-throughput.",
    )
    rate_rule_min_packets: int = Field(
        default=5,
        description="Minimum number of packets a flow must have before HighPacketRateRule or HighByteRateRule "
        "are evaluated at all. Rate is packet_count/byte_count divided by flow_duration_seconds -- for a flow "
        "with very few packets, flow_duration_seconds can be a tiny fraction of a second (e.g. a 2-packet HTTPS "
        "handshake observed ~60 microseconds apart), which mathematically produces an enormous rate (tens of "
        "thousands of packets/sec) from a sample too small to say anything reliable. This guard does not change "
        "the rate calculation itself (FeatureExtractor's total_packets_per_second/total_bytes_per_second remain "
        "the exact mathematical rate) -- it only withholds these two heuristic rules' judgment until there's "
        "enough of a sample to trust, avoiding false positives observed during real Windows traffic validation.",
    )
    syn_heavy_ratio_threshold: float = Field(
        default=0.8,
        description="Fraction of a flow's packets that are SYN above which the flow is considered SYN-dominant "
        "(associated with SYN scanning/flooding attempts, but also produced by a simple unanswered connection attempt).",
    )
    syn_heavy_min_packets: int = Field(
        default=5, description="Minimum total packets before the SYN-heavy ratio is evaluated, to avoid noise on tiny flows."
    )
    short_flow_max_duration_seconds: float = Field(
        default=0.5, description="Flows at or below this duration are considered 'short' for the short/high-volume rule."
    )
    short_flow_min_packets: int = Field(
        default=20, description="Packet count at/above which a 'short' flow is considered a suspicious burst."
    )
    flag_anomaly_min_packets: int = Field(
        default=2, description="Minimum packet count before TCP flag-combination anomalies are evaluated."
    )
    suspicious_destination_ports: tuple[int, ...] = Field(
        default=(23, 445, 3389, 4444, 31337),
        description="Destination ports commonly associated with high-risk services or known attack tooling "
        "(Telnet, SMB, RDP, a common Metasploit default, 'elite'/backdoor convention). A single match is a weak "
        "signal on its own -- see docs/methodology.md -- and is NOT proof of an attack.",
    )


class RiskScoringConfig(BaseSettings):
    """
    Configurable risk-score bucket boundaries (0-100 scale).

    These bucket boundaries are an operational scoring scheme chosen for
    readability on the dashboard -- they are explicitly NOT claimed to be
    a scientifically optimal cutoff. See docs/methodology.md.
    """

    low_max: int = 29
    medium_max: int = 59
    high_max: int = 79
    # 80-100 = CRITICAL


class HybridScoringConfig(BaseSettings):
    """
    Phase 8 hybrid (rule + ML) risk-score fusion weights.

    final_risk_score = round(rule_weight * rule_risk_score + ml_weight * ml_score),
    where ml_score is the model's ATTACK-class probability scaled to 0-100
    (or `ml_no_probability_fallback_score` if the model doesn't expose a
    probability). Weights default to an even 0.5/0.5 split -- a deliberate,
    documented starting point rather than an arbitrary one: neither the
    rule thresholds (Phase 5) nor any trained model (Phase 6/7) have been
    empirically validated against real labeled attack traffic yet, so there
    is no evidence-based reason to trust one detector more than the other
    by default. These weights are meant to be re-tuned once Phase 12
    benchmarking produces real precision/recall numbers for each detector.
    rule_weight + ml_weight must sum to 1.0 -- enforced by HybridDetector,
    not here, so a caller gets an immediate, clear error at construction
    time rather than a silently-wrong score later.
    """

    rule_weight: float = 0.5
    ml_weight: float = 0.5
    ml_no_probability_fallback_score: float = Field(
        default=60.0,
        description="Score (0-100) attributed to the ML component when the model predicts ATTACK but "
        "cannot report a probability (e.g. a classifier without predict_proba, or a degenerate "
        "single-class model). Chosen as a moderate-confidence placeholder -- an ATTACK prediction "
        "without a probability is still a signal worth weighting, just not one we can scale by "
        "model confidence. Configurable; not claimed to be statistically derived.",
    )


class Settings(BaseSettings):
    """Top-level application settings, populated from environment variables."""

    model_config = SettingsConfigDict(env_file=".env", env_nested_delimiter="__", extra="ignore")

    # --- General ---
    environment: Literal["lab", "development", "production"] = Field(default="lab")
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(default="INFO")

    # --- Packet capture ---
    capture_interface: str = Field(
        default="eth0",
        description="Network interface to sniff on. On a VirtualBox/VMware lab VM this is typically "
        "the interface attached to the internal/host-only network segment shared with the "
        "traffic-generator VM -- run `ip link` inside the sensor VM to confirm the name.",
    )
    capture_promiscuous: bool = Field(default=True)
    capture_queue_max_size: int = Field(
        default=10_000, description="Bounded queue size between the sniff loop and the processing consumer."
    )

    # --- Flow aggregation ---
    flow_timeout_seconds: int = Field(default=120, description="Idle time before a flow is considered expired and finalized.")
    flow_table_max_size: int = Field(default=50_000, description="Safety cap on concurrently tracked flows.")

    # --- Database ---
    database_url: str = Field(
        default="postgresql+psycopg://nids:nids@localhost:5432/nids",
        description="SQLAlchemy connection string. Override via DATABASE_URL env var.",
    )

    # --- ML ---
    # NOTE (discovered during Phase 7 implementation): these two fields were
    # Phase 1 placeholders for a single-file model format ("Populated after
    # Phase 7"). Phase 6's actual `save_model_artifact()` produces THREE
    # files per model (model.joblib, model.scaler.joblib,
    # model.metadata.json -- schema info lives inside the metadata JSON, not
    # a separate file), so these two fields don't match the real shape and
    # are not used by app.ml.inference. Kept, unused, for backward `.env`
    # compatibility rather than silently deleted. See ml_model_dir /
    # ml_active_model_name below for the fields Phase 7 actually reads.
    ml_model_path: str = Field(
        default="ml/models/latest_model.joblib",
        description="Superseded by ml_model_dir/ml_active_model_name -- see note above. Not read by app.ml.inference.",
    )
    ml_feature_schema_path: str = Field(
        default="ml/models/feature_schema.json",
        description="Superseded -- Phase 6 embeds schema info in each model's own metadata.json instead. Not read by app.ml.inference.",
    )

    ml_model_dir: str = Field(
        default="ml/models",
        description="Directory containing saved Phase 6 model artifact triples (model/scaler/metadata).",
    )
    ml_active_model_name: str = Field(
        default="random_forest",
        description="Base filename (without extension) of the saved model to load for live inference, "
        "e.g. 'random_forest' for random_forest.joblib/.scaler.joblib/.metadata.json. Must match a model "
        "actually produced by `python -m app.ml.train_cli`.",
    )

    # --- WebSocket ---
    websocket_broadcast_interval_ms: int = Field(default=500, description="Minimum interval between metrics broadcasts.")

    # --- Detection ---
    thresholds: DetectionThresholds = Field(default_factory=DetectionThresholds)
    risk_scoring: RiskScoringConfig = Field(default_factory=RiskScoringConfig)
    hybrid_scoring: HybridScoringConfig = Field(default_factory=HybridScoringConfig)


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings instance (loaded once per process)."""
    return Settings()


settings = get_settings()