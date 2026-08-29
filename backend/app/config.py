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
    ml_model_path: str = Field(
        default="ml/models/latest_model.joblib",
        description="Path to the serialized model used by real-time inference. Populated after Phase 7.",
    )
    ml_feature_schema_path: str = Field(
        default="ml/models/feature_schema.json",
        description="JSON file pinning the exact ordered feature list the model expects -- the contract between offline training and real-time inference.",
    )

    # --- WebSocket ---
    websocket_broadcast_interval_ms: int = Field(default=500, description="Minimum interval between metrics broadcasts.")

    # --- Detection ---
    thresholds: DetectionThresholds = Field(default_factory=DetectionThresholds)
    risk_scoring: RiskScoringConfig = Field(default_factory=RiskScoringConfig)


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings instance (loaded once per process)."""
    return Settings()


settings = get_settings()
