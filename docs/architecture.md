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
