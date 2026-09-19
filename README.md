# Real-Time Network Intrusion Detection System (NIDS)

A hybrid rule-based + machine-learning network intrusion detection system,
built for an authorized VM-based lab environment, with a real-time
WebSocket dashboard and rigorously measured (never fabricated) performance
metrics.

> **Status:** Phase 10 of 15 complete — FastAPI REST API (`/api/status`,
> `/api/flows`, `/api/alerts`, `/api/stats/*`) and a real-time
> `/ws/alerts` WebSocket, both backed by the existing Phase 9 persistence
> layer with no logic duplicated. `/api/health` is unchanged from Phase 1.
> See `docs/methodology.md` for the full writeup.

## Problem statement

_(Expanded in Phase 17 final documentation pass.)_ Network intrusion
detection systems need to identify malicious or anomalous traffic in real
time, at line rate, with a defensible false-positive rate — not just score
well on a static offline dataset.

## Motivation

_(Expanded in Phase 17.)_

## Architecture

See [`docs/architecture.md`](docs/architecture.md) for the full real-time
pipeline, offline training pipeline, and the VM lab topology this project
targets (VirtualBox/VMware internal network, isolated from production
traffic).

## Scope & authorization

This project is designed **only** for the operator's own machine or an
authorized, isolated VM lab. See [`docs/threat_model.md`](docs/threat_model.md).

## Project structure

```
nids-project/
├── backend/            FastAPI app: capture, flows, features, detection, ML, alerts, DB, WebSocket
├── frontend/            React + Vite SOC-style dashboard (Phase 13)
├── ml/                   Offline training/evaluation pipeline, kept separate from real-time inference
├── experiments/          Benchmark harness + results (Phase 15)
├── database/             Migrations
├── docker/               Dockerfiles (Phase 16)
├── docs/                 Architecture, methodology, evaluation, threat model
├── docker-compose.yml    PostgreSQL for local dev (backend/frontend services added Phase 16)
└── .env.example
```

## Setup (Phase 1 state)

```bash
# 1. Start PostgreSQL (used from Phase 10 onward; harmless to start now)
docker compose up -d postgres

# 2. Backend
cd backend
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp ../.env.example ../.env   # adjust CAPTURE_INTERFACE etc. for your VM

# 3. Run the (currently minimal) API
uvicorn app.main:app --reload

# 4. Health check
curl http://localhost:8000/api/health

# 5. Run tests
pytest
```

### Trying the capture engine directly (Phase 2)

Packet capture needs raw-socket privileges. On the Kali sensor VM:

```bash
# Either run as root, or grant the capability once:
sudo setcap cap_net_raw,cap_net_admin=eip $(readlink -f $(which python3))

cd backend
python3 -c "
from app.capture.packet_capture import PacketCapture
import time

capture = PacketCapture(interface='eth0')  # falls back to auto-detect if 'eth0' is invalid
capture.start()
time.sleep(10)
capture.stop()
print(capture.stats_snapshot())
"
```

This does not require the internal lab network yet — the current VirtualBox
NAT `eth0` (10.0.2.0/24) is enough to validate the capture engine. The
isolated sensor/target network is added when Phase 14 (controlled lab
testing) needs actual attack scenarios.

### Trying capture + flow aggregation together (Phase 3)

```bash
cd backend
python3 -c "
from app.capture.packet_capture import PacketCapture
from app.flows.flow_aggregator import FlowAggregator
import time

capture = PacketCapture(interface='eth0')
aggregator = FlowAggregator(source_queue=capture.queue)

capture.start()
aggregator.start()
time.sleep(10)

print(capture.stats_snapshot())
print(aggregator.stats_snapshot())

aggregator.stop()   # force-finalizes any still-active flows
capture.stop()

for flow in aggregator.get_expired_flows():
    print(flow)
"
```

### Running rule-based detection on a flow (Phase 5)

```bash
cd backend
python3 -c "
from app.features.feature_extractor import FeatureExtractor
from app.detection.rule_engine import RuleDetector

# ... obtain a FinalizedFlow from aggregator.get_expired_flows() as above ...
extractor = FeatureExtractor()
detector = RuleDetector()  # uses DetectionThresholds/RiskScoringConfig from .env

vector = extractor.extract(flow)
result = detector.detect(vector, flow)
print(result)
"
```

### Training a model on CIC-IDS2017 (Phase 6, offline only)

```bash
cd backend
python -m app.ml.train_cli \
    --dataset-path /path/to/cic_ids2017.csv \
    --output-dir ml/models \
    --random-seed 42

# Quick pipeline test on a subsample first, before running the full dataset:
python -m app.ml.train_cli --dataset-path /path/to/cic_ids2017.csv --output-dir /tmp/test_models --sample-size 5000
```

This validates dataset compatibility against the live 41-feature schema
BEFORE training, reports data-cleaning counts, trains and compares all
three baseline models, and saves the best one (by macro F1) with full
metadata. It does not connect to live capture -- that's Phase 7.

### Running live ML inference on a flow (Phase 7)

```bash
cd backend
python3 -c "
from app.features.feature_extractor import FeatureExtractor
from app.ml.inference import MLPredictor

# ... obtain a FinalizedFlow from aggregator.get_expired_flows() as above ...
predictor = MLPredictor.load(
    model_path='ml/models/random_forest.joblib',
    scaler_path='ml/models/random_forest.scaler.joblib',
    metadata_path='ml/models/random_forest.metadata.json',
)
vector = FeatureExtractor().extract(flow)
result = predictor.predict(vector)
print(result)
"
```

Requires a model already trained via `python -m app.ml.train_cli` (Phase 6).
Not yet fused with Phase 5's `RuleDetector` -- that's Phase 8.

### Running hybrid rule + ML detection on a flow (Phase 8)

```bash
cd backend
python3 -c "
from app.features.feature_extractor import FeatureExtractor
from app.ml.inference import MLPredictor
from app.detection.hybrid_detector import HybridDetector

# ... obtain a FinalizedFlow from aggregator.get_expired_flows() as above ...
predictor = MLPredictor.load(
    model_path='ml/models/random_forest.joblib',
    scaler_path='ml/models/random_forest.scaler.joblib',
    metadata_path='ml/models/random_forest.metadata.json',
)
detector = HybridDetector(ml_predictor=predictor)  # omit ml_predictor for rules-only mode

vector = FeatureExtractor().extract(flow)
result = detector.detect(vector, flow)
print(result)
"
```

### Persisting flows and detections to PostgreSQL (Phase 9)

```bash
# 1. Start PostgreSQL (docker-compose.yml already matches the default DATABASE_URL)
docker compose up -d postgres

# 2. Initialize the schema (creates the flows/detections tables)
cd backend
python -m app.database.init_db_cli

# 3. Use the repository from Python
python3 -c "
from app.database.session import create_db_engine, create_session_factory
from app.database.repository import FlowDetectionRepository
from app.features.feature_extractor import FeatureExtractor
from app.detection.hybrid_detector import HybridDetector
from app.config import settings

engine = create_db_engine(settings.database_url)
repo = FlowDetectionRepository(create_session_factory(engine))

# ... obtain a FinalizedFlow from aggregator.get_expired_flows() as above ...
vector = FeatureExtractor().extract(flow)
result = HybridDetector().detect(vector, flow)

flow_id = repo.save_flow(flow, feature_vector=vector)
repo.save_detection(flow_id, result)
print(repo.get_recent_detections(limit=5))
"
```

Every database failure raises a single `DatabasePersistenceError` --
callers never need to import SQLAlchemy directly. Run the real-PostgreSQL
integration test (skipped by default, requires the above `docker compose`
step) via:
```bash
NIDS_TEST_POSTGRES_URL='postgresql+psycopg://nids:nids@localhost:5432/nids' \
    pytest tests/test_database.py -k postgres
```

### Running the API + WebSocket server (Phase 10)

```bash
cd backend
uvicorn app.main:app --reload

# In another terminal:
curl http://localhost:8000/api/health
curl http://localhost:8000/api/status
curl http://localhost:8000/api/flows
curl http://localhost:8000/api/alerts?limit=10
curl "http://localhost:8000/api/alerts?severity=HIGH"
curl http://localhost:8000/api/stats/traffic
curl http://localhost:8000/api/stats/detections

# Interactive API docs (FastAPI auto-generated):
#   http://localhost:8000/docs

# WebSocket (e.g. with websocat, or any WS client):
#   ws://localhost:8000/ws/alerts
```

No live capture-to-detection loop pushes real events through `/ws/alerts`
yet -- see `docs/methodology.md` for what's wired vs. what's still a ready
capability awaiting a later phase.

## Technologies (and why)

| Tech | Why |
|---|---|
| Python 3.11+ / FastAPI | Async-friendly, type-hinted, fast to build a real-time API around |
| Scapy | Flexible packet parsing without requiring a compiled capture pipeline; tradeoffs vs. PyShark/Zeek documented in `docs/methodology.md` once Phase 2 lands |
| PostgreSQL + SQLAlchemy | Relational storage for flows/alerts with strong indexing support for time/IP/severity queries |
| WebSockets | Push-based updates so the dashboard never needs polling or manual refresh |
| React + Vite | Fast dev loop for a data-dense SOC dashboard |
| pandas / NumPy | Dataset loading, cleaning, and feature alignment for offline ML training (Phase 6) |
| scikit-learn | Interpretable baselines (LR/DT/RF) before reaching for anything heavier |
| joblib | Model/scaler serialization for the saved training artifacts |

## Development phases

1. ✅ Project architecture and repository structure
2. ✅ Packet capture
3. ✅ Flow aggregation
4. ✅ Feature extraction
5. ✅ Rule-based detection
6. ✅ Dataset preprocessing and ML training (offline only)
7. ✅ Real-time ML inference
8. ✅ Hybrid detection / risk scoring
9. ✅ PostgreSQL integration
10. ✅ FastAPI APIs + WebSocket event streaming
11. ⬜ React dashboard
12. ⬜ Controlled lab testing
13. ⬜ Benchmarking and metrics
14. ⬜ Dockerization
15. ⬜ Documentation and final README

## Evaluation

See [`docs/evaluation.md`](docs/evaluation.md). Every metric is marked
**"Not measured yet"** until an experiment has actually produced it —
this project does not fabricate accuracy, latency, or throughput numbers.

## Limitations

Tracked honestly and continuously in [`docs/threat_model.md`](docs/threat_model.md).

## Future work

Distributed sensors, Zeek/Suricata integration, TLS metadata analysis,
online learning, threat intelligence feeds, Kubernetes monitoring —
revisited in the Phase 17 final README.
