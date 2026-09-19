# Architecture

## Real-time detection path

```
NIC (lab interface, promiscuous mode)
  -> Packet Capture (Scapy, sniff loop in dedicated thread/process)
  -> Bounded Queue (backpressure-aware; drops are counted, never silent)
  -> Packet Processor (parse L2/L3/L4 headers)
  -> Flow Aggregator (5-tuple flow table, sliding expiration)
  -> Feature Extractor (per-flow feature vector matching the training schema)
  -> Rule Engine -----+
  -> ML Model       ---+--> Risk Scorer -> Alert Manager -> Dedup/Suppression
                                              |
                              +---------------+----------------+
                              v                                v
                          PostgreSQL                  WebSocket Broadcaster
                                                              |
                                                              v
                                                     React SOC Dashboard
```

## Offline training path

```
CIC-IDS2017 (CICFlowMeter CSVs)
  -> Cleaning (duplicates, missing/invalid values)
  -> Feature Alignment (drop anything not reproducible from live capture --
     see the feature-compatibility table in docs/methodology.md)
  -> Scenario-aware Train/Val/Test Split
  -> Model Training (Logistic Regression, Decision Tree, Random Forest[, XGBoost])
  -> Evaluation (metrics + plots, Section 11/34 of the original spec)
  -> Serialized model (joblib) + pinned feature_schema.json
  -> Loaded by real-time inference (ml/inference/)
```

The **feature schema is the contract** between the two paths: nothing enters
the model that the real-time flow aggregator can't compute from live packets.

## Phase 4 addition: Flow -> FeatureVector

`FinalizedFlow` (Phase 3) is consumed by `FeatureExtractor` (Phase 4),
producing a fixed-length, versioned `FeatureVector` -- see
`docs/methodology.md` for the full 41-feature contract. Building this
required an additive extension to `Flow`/`FinalizedFlow` (streaming
packet-length/IAT statistics, TCP flag counts) since the original Phase 3
aggregate counters alone couldn't support several required features. The
extension is backward-compatible: every Phase 3 field keeps its exact prior
meaning; new fields only add capability.

## Phase 5 addition: FeatureVector -> DetectionResult

`app.detection.rule_engine.RuleDetector` consumes a `FeatureVector` +
`FinalizedFlow` pair and produces a `DetectionResult` (risk score,
severity, triggered rules with reasons/evidence). Stateless and
single-flow only -- no cross-flow correlation layer exists yet. See
`docs/methodology.md` for the full rule catalog and the risk-scoring
scheme.

## Phase 6 addition: offline training pipeline (`app.ml.*`)

A fully separate, offline path that does NOT touch the real-time pipeline
above:

```
CIC-IDS2017 CSV -> app.ml.feature_mapping (align to the SAME 41-feature
schema used live) -> app.ml.preprocessing (clean, split, scale)
-> app.ml.training (Logistic Regression / Decision Tree / Random Forest)
-> app.ml.persistence (save model + scaler + metadata)
```

The live 41-feature schema (`app.features.schema`) is imported by
`app.ml.feature_mapping` and `app.ml.persistence` as the single source of
truth -- the training pipeline never maintains its own separate copy of
the feature list. No model trained here is loaded by any live-inference
code yet; that connection is explicitly Phase 7's job.

## Phase 7 addition: live ML inference (`app.ml.inference`)

Connects the FeatureVector output of the existing real-time pipeline to a
Phase 6-trained model artifact:

```
FinalizedFlow -> FeatureExtractor.extract() -> FeatureVector
                                                    |
                                                    v
                                      MLPredictor.predict(vector)
                                                    |
                                                    v
                                          MLInferenceResult
```

`MLPredictor.load()` reuses the existing `app.ml.persistence.load_model_artifact()`
for schema-version/feature-name validation at load time (not duplicated),
and applies exactly the `StandardScaler` saved during Phase 6 training (no
separate preprocessing pipeline invented). `MLPredictor` does not extract
features itself -- it strictly consumes `FeatureVector` instances from the
existing `FeatureExtractor`. Errors are typed (`ModelArtifactNotFoundError`,
`ModelArtifactCorruptedError`, `InvalidFeatureVectorError`,
`SchemaVersionMismatchError`) so a future orchestrator can catch-and-skip
rather than crash. No orchestrator/main-loop wiring capture through to
inference continuously exists yet -- Phase 7 provides the inference
*capability*; a later phase (per the project's own phase ordering, Phase
8's hybrid detector, and eventually a real-time service loop) is
responsible for calling it per finalized flow.

## Phase 8 addition: hybrid rule + ML detection (`app.detection.hybrid_detector`)

`HybridDetector` orchestrates the existing `RuleDetector` (Phase 5) and
`MLPredictor` (Phase 7) unchanged -- it does not reimplement either:

```
FeatureVector ---> RuleDetector.detect()   -> DetectionResult   --+
              \                                                   |--> fuse -> HybridDetectionResult
               -> MLPredictor.predict()    -> MLInferenceResult --+
```

`RuleDetector` remains fully independent and usable on its own (verified by
a test asserting identical behavior standalone vs. inside a
`HybridDetector`). ML is optional (`ml_predictor=None` runs rules-only,
useful since no model has been trained on real attack data yet). ML
inference failures are caught (`MLInferenceError` only -- anything else
still propagates) and degrade to rules-only for that flow rather than
crashing detection. Severity classification was extracted into
`app.detection.severity.classify_severity()`, which `RuleDetector` itself
now delegates to (specifically to avoid `HybridDetector` duplicating that
bucket logic) -- `RuleDetector`'s public behavior, including its existing
test suite, is unchanged. See `docs/methodology.md` for the full
fusion-weight rationale.

## Phase 9 addition: PostgreSQL persistence (`app.database`)

```
FinalizedFlow ---> FlowDetectionRepository.save_flow()   -> flows table
                                                                |
HybridDetectionResult ---> .save_detection(flow_id, result) -> detections table (FK -> flows.id)
```

Three-module split:
- `app.database.models` -- SQLAlchemy ORM tables (`FlowRecord`, `DetectionRecord`)
  plus plain, frozen read-view dataclasses (`PersistedFlow`, `PersistedDetection`)
  that the repository returns instead of raw ORM instances, avoiding
  DetachedInstanceError pitfalls and keeping callers SQLAlchemy-agnostic.
- `app.database.session` -- engine/session-factory construction from
  `settings.database_url`, and `init_db()` (a `create_all`-based schema
  initializer -- see docs/methodology.md for why Alembic is deferred
  rather than added now).
- `app.database.repository` -- `FlowDetectionRepository`, the only layer
  that imports SQLAlchemy exceptions. Every database failure surfaces as
  one typed `DatabasePersistenceError`, so the capture/detection pipeline
  never needs to know it's talking to SQLAlchemy/PostgreSQL at all.

No raw packet payloads are stored -- only flow-level metadata, aggregate
counters, and (optionally) the already-computed 41-feature snapshot.

## Phase 10 addition: FastAPI REST API + WebSocket (`app.api`, `app.websocket`)

```
Client -> GET /api/flows, /api/alerts, /api/stats/*, /api/status
              |
              v
      app.api.routes.* (FastAPI routers)
              |
              v
      FlowDetectionRepository (Phase 9, unmodified interface -- only additively extended)
              |
              v
         PostgreSQL / SQLite
```

```
HybridDetectionResult (future orchestrator) -> ConnectionManager.broadcast()
                                                        |
                                                        v
                                          every connected /ws/alerts client
```

Four routers (`system`, `flows`, `alerts`, `stats`) share one process-wide
`FlowDetectionRepository` via a FastAPI dependency
(`app.api.dependencies.get_repository`), overridable in tests. The
pre-existing `/api/health` liveness check is untouched; `/api/status` is
new and separately reports live database reachability (never fails the
request itself if the DB is down -- that's exactly the situation a status
endpoint needs to report).

`app.websocket.manager.ConnectionManager` is a small, independently
testable connection registry: `connect()`/`disconnect()` track clients in
a set guarded by an `asyncio.Lock`, and `broadcast()` sends to all of them,
catching and dropping any client whose send fails rather than letting one
broken connection affect delivery to the others or raise out of
`broadcast()` itself. As with Phases 7-9's capabilities, no live
capture-to-detection loop calls `broadcast()` yet -- this phase delivers
the mechanism, fully exercised by tests using fake WebSocket objects plus
FastAPI's real `TestClient.websocket_connect()`.

Phase 9's `FlowDetectionRepository` was extended **additively** with new
read/aggregate methods (`count_flows`, `count_detections`, `list_flows`,
`get_recent_alerts`, `get_traffic_summary`) needed by these endpoints --
every pre-existing Phase 9 method is untouched, confirmed by re-running
`test_database.py` before writing any Phase 10 route code.

## Lab topology (VM-based, VirtualBox/VMware)

```
+------------------+       internal / host-only        +--------------------+
|  VM A: Traffic   |  <------- network segment ------>  |  VM B: Sensor       |
|  Generator       |     (isolated from the internet;    |  - Capture + Flow   |
|  - normal traffic|      NIC in promiscuous mode on      |    Aggregator      |
|  - controlled     |      the virtual switch for VM B)    |  - Detection Engine|
|    port scans     |                                     |  - FastAPI backend |
|  - high-rate load |                                     +---------+----------+
+------------------+                                                |
                                                                     v
                                                     +--------------------------+
                                                     |  VM C (or same as VM B):  |
                                                     |  PostgreSQL + Dashboard   |
                                                     +--------------------------+
```

Notes for a VirtualBox/VMware setup:
- Use an **internal network** (VirtualBox) or **host-only/custom vSwitch**
  (VMware) so lab traffic never touches your real LAN or the internet.
- Enable **promiscuous mode: Allow All** on that virtual network adapter/switch
  — by default hypervisors drop frames not addressed to a VM's own MAC, which
  silently breaks sniffing.
- VM B needs `CAP_NET_RAW`/root (or a capability grant on the capture binary)
  to sniff — see docs/methodology.md, Phase 2, for the least-privilege setup.
- PostgreSQL and the dashboard can run on VM B or a separate VM C; docker-compose.yml
  currently starts Postgres standalone for local development convenience.

## Component responsibilities

See the top-level design writeup in the conversation (Section B) — this file
will be expanded with sequence diagrams as each phase lands.
