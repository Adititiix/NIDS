# Real-Time Network Intrusion Detection System (NIDS)

A hybrid rule-based + machine-learning network intrusion detection system,
built for an authorized VM-based lab environment, with a real-time
WebSocket dashboard and rigorously measured (never fabricated) performance
metrics.

> **Status:** Phase 3 of 17 complete — in-memory flow aggregation
> (bidirectional 5-tuple flow table, forward/backward tracking,
> inactivity-based expiration). Verified end-to-end with real `PacketCapture`
> traffic feeding the aggregator on a live interface. See
> `docs/architecture.md` for the design and the phase list below for what's
> next.

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

## Technologies (and why)

| Tech | Why |
|---|---|
| Python 3.11+ / FastAPI | Async-friendly, type-hinted, fast to build a real-time API around |
| Scapy | Flexible packet parsing without requiring a compiled capture pipeline; tradeoffs vs. PyShark/Zeek documented in `docs/methodology.md` once Phase 2 lands |
| PostgreSQL + SQLAlchemy | Relational storage for flows/alerts with strong indexing support for time/IP/severity queries |
| WebSockets | Push-based updates so the dashboard never needs polling or manual refresh |
| React + Vite | Fast dev loop for a data-dense SOC dashboard |
| scikit-learn | Interpretable baselines (LR/DT/RF) before reaching for anything heavier |

## Development phases

1. ✅ Project architecture and repository structure
2. ✅ Packet capture
3. ✅ Flow aggregation
4. ⬜ Feature extraction
5. ⬜ Rule-based detection
6. ⬜ Dataset preprocessing
7. ⬜ ML training and evaluation
8. ⬜ Real-time ML inference
9. ⬜ Hybrid detection / risk scoring
10. ⬜ PostgreSQL integration
11. ⬜ FastAPI APIs
12. ⬜ WebSocket event streaming
13. ⬜ React dashboard
14. ⬜ Controlled lab testing
15. ⬜ Benchmarking and metrics
16. ⬜ Dockerization
17. ⬜ Documentation and final README

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
