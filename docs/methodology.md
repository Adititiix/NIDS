# Methodology

_This file is populated incrementally as each phase lands:_

- Phase 4/6: feature-compatibility table (dataset vs. live) — finalized version
- Phase 5: full rule catalog (threshold, window, rationale, severity per rule)
- Phase 7: model selection rationale, class-imbalance handling decision
- Phase 9: train/val/test split strategy (scenario-aware, leakage avoidance)

## Phase 3: flow direction semantics (populated now, relevant to Phase 4)

Flows are bidirectional and keyed by a canonical (order-independent) 5-tuple,
so a request and its reply hash to the same flow. "Forward" vs "backward"
is defined relative to whichever packet the aggregator *observed first* for
that 5-tuple — not necessarily the TCP handshake initiator in a strict
sense.

This has one known, deliberate limitation: if capture starts mid-flow (e.g.
the sensor comes online after a long-lived connection already exists), the
first packet we happen to see defines "forward," which may not match the
conventional client→server direction. This is acceptable for this project's
purposes because:
- Forward/backward *counts* are still internally consistent for feature
  extraction (the ratios and magnitudes are what matter to the model/rules,
  not which literal direction is labeled "forward").
- CIC-IDS2017's own forward/backward labeling has the same practical
  ambiguity for any flow not captured from its very first packet.

This will be re-examined if Phase 4 feature validation against CICFlowMeter
reference values shows it matters in practice.

## Phase 4: feature contract (schema version 1.0.0)

**Source of truth:** `backend/app/features/schema.py`. The table below is
generated directly from that module (not hand-maintained), so it cannot
drift out of sync with the actual code -- regenerate it with:

```bash
cd backend && PYTHONPATH=. python3 -c "
from app.features.schema import FEATURE_SCHEMA
for s in FEATURE_SCHEMA: print(s.index, s.name)
"
```

**41 features total** (indices 0-40), fixed order, versioned as a unit --
a model trained against one schema version must never be fed vectors from
a different version without re-validation.

**Scope decisions:**
- `src_port` is deliberately excluded (ephemeral, not predictive; matches
  CICFlowMeter's own choice).
- Packet-length and inter-arrival-time statistics are provided both
  combined (`packet_length_*`, `flow_iat_*`) and split by direction
  (`forward_packet_length_*` / `backward_packet_length_*`,
  `forward_iat_*` / `backward_iat_*`), matching CICFlowMeter's convention
  of providing both views -- this is about training-data compatibility,
  not feature-count inflation.
- All rates (`*_packets_per_second`, `*_bytes_per_second`) are defined as
  **0.0, never NaN/Inf**, when flow duration is 0. This is a deliberate
  convention for an otherwise-undefined case (a zero-duration flow has no
  meaningful "rate"), and it needs to be checked against how the eventual
  CIC-IDS2017 CSVs represent the same edge case once Phase 6 preprocessing
  begins -- see "Open assumptions" below.

**Explicitly excluded (not silently dropped):**
- `init_win_bytes_forward/backward` -- Phase 1 incorrectly listed this as
  "Live Available." On inspection, `PacketMetadata` (Phase 2) does not
  capture TCP window size at all. Excluded from v1; would require an
  additive Phase 2 change to add if judged worth it later.
- `active_mean` / `idle_mean` -- CICFlowMeter's algorithm for these
  requires tracking active-burst/idle-gap boundaries as they happen, which
  is meaningfully more state than a streaming accumulator and risks a
  subtly wrong reimplementation. Deferred rather than guessed at.
- `subflow_forward_bytes` / `subflow_backward_bytes` -- already excluded in
  the Phase 1 compatibility table; not faithfully reproducible from live
  capture without reimplementing CICFlowMeter's subflow segmentation.

**Open assumptions to validate once the CIC-IDS2017 dataset is loaded
(Phase 6):**
1. Whether CICFlowMeter's own zero-duration-flow rate convention matches
   our 0.0 choice, or produces something else (0, NaN, or a sentinel) that
   would need reconciling before training.
2. Whether population (not sample) standard deviation is indeed
   CICFlowMeter's convention -- assumed here based on prior knowledge of
   the tool, not yet verified against the actual CSV documentation.
3. Whether `origin_dst_port` (the destination port of whichever packet
   created the flow) lines up with how CICFlowMeter assigns "Destination
   Port" for flows it captures mid-session.

| Idx | Name | Type | Unit | Direction | Description | Zero/Edge-Case Handling | Live | Dataset |
|---|---|---|---|---|---|---|---|---|
| 0 | `dst_port` | int | port number | n/a | Destination port of the flow's originating packet. Source port is deliberately excluded: it's ephemeral/high-cardinality and not predictive on its own -- CICFlowMeter drops it too. | -1 for protocols without ports (e.g. ICMP). | yes | yes |
| 1 | `protocol_is_tcp` | int | boolean (0/1) | n/a | One-hot: 1 if the flow's transport protocol is TCP, else 0. | n/a -- always defined. | yes | yes |
| 2 | `protocol_is_udp` | int | boolean (0/1) | n/a | One-hot: 1 if the flow's transport protocol is UDP, else 0. | n/a -- always defined. | yes | yes |
| 3 | `protocol_is_icmp` | int | boolean (0/1) | n/a | One-hot: 1 if the flow's transport protocol is ICMP, else 0. Any other protocol (TransportProtocol.OTHER) is represented as all three one-hot columns being 0. | n/a -- always defined. | yes | yes |
| 4 | `flow_duration_seconds` | float | seconds | overall | last_seen - first_seen for the flow. | 0.0 for a single-packet flow (first_seen == last_seen). | yes | yes |
| 5 | `total_packets` | int | count | overall | Total number of packets observed in the flow (both directions). | Always >= 1 -- a FinalizedFlow cannot exist with zero packets. | yes | yes |
| 6 | `total_bytes` | int | bytes | overall | Total bytes observed in the flow (both directions), on-the-wire packet length. | Always >= 1. | yes | yes |
| 7 | `total_forward_packets` | int | count | forward | Packets traveling in the same direction as the flow's originating packet. | Always >= 1 (the originating packet itself is forward). | yes | yes |
| 8 | `total_backward_packets` | int | count | backward | Packets traveling in the reverse direction of the flow's originating packet. | 0 for a one-directional flow (e.g. an unanswered probe). | yes | yes |
| 9 | `total_forward_bytes` | int | bytes | forward | Sum of on-the-wire lengths of forward packets. | Always >= 1. | yes | yes |
| 10 | `total_backward_bytes` | int | bytes | backward | Sum of on-the-wire lengths of backward packets. | 0 for a one-directional flow. | yes | yes |
| 11 | `packet_length_mean` | float | bytes | overall | Mean packet length across all packets in the flow. Note: mathematically equal to total_bytes / total_packets, kept as its own feature for readability and dataset-column parity. | Equal to the single packet's length for a 1-packet flow. | yes | yes |
| 12 | `packet_length_std` | float | bytes | overall | Population standard deviation of packet length across all packets. | 0.0 for a flow with fewer than 2 packets. | yes | yes |
| 13 | `packet_length_min` | float | bytes | overall | Minimum packet length observed in the flow. | Equal to the single packet's length for a 1-packet flow. | yes | yes |
| 14 | `packet_length_max` | float | bytes | overall | Maximum packet length observed in the flow. | Equal to the single packet's length for a 1-packet flow. | yes | yes |
| 15 | `forward_packet_length_mean` | float | bytes | forward | Mean packet length, forward packets only. | 0.0 if there are no forward packets (cannot occur -- see total_forward_packets). | yes | yes |
| 16 | `forward_packet_length_std` | float | bytes | forward | Population standard deviation of packet length, forward packets only. | 0.0 for fewer than 2 forward packets. | yes | yes |
| 17 | `forward_packet_length_min` | float | bytes | forward | Minimum packet length, forward packets only. | Equal to the single forward packet's length if there's only one. | yes | yes |
| 18 | `forward_packet_length_max` | float | bytes | forward | Maximum packet length, forward packets only. | Equal to the single forward packet's length if there's only one. | yes | yes |
| 19 | `backward_packet_length_mean` | float | bytes | backward | Mean packet length, backward packets only. | 0.0 if there are no backward packets (one-directional flow). | yes | yes |
| 20 | `backward_packet_length_std` | float | bytes | backward | Population standard deviation of packet length, backward packets only. | 0.0 for fewer than 2 backward packets, including zero. | yes | yes |
| 21 | `backward_packet_length_min` | float | bytes | backward | Minimum packet length, backward packets only. | 0.0 if there are no backward packets. | yes | yes |
| 22 | `backward_packet_length_max` | float | bytes | backward | Maximum packet length, backward packets only. | 0.0 if there are no backward packets. | yes | yes |
| 23 | `flow_iat_mean` | float | seconds | overall | Mean inter-arrival time between consecutive packets, regardless of direction. | 0.0 for a flow with fewer than 2 packets (no gap to measure). | yes | yes |
| 24 | `flow_iat_std` | float | seconds | overall | Population standard deviation of inter-arrival time, regardless of direction. | 0.0 for fewer than 3 packets (need >=2 gaps for a variance to exist). | yes | yes |
| 25 | `flow_iat_min` | float | seconds | overall | Minimum inter-arrival time observed. | 0.0 for a flow with fewer than 2 packets. | yes | yes |
| 26 | `flow_iat_max` | float | seconds | overall | Maximum inter-arrival time observed. | 0.0 for a flow with fewer than 2 packets. | yes | yes |
| 27 | `forward_iat_mean` | float | seconds | forward | Mean inter-arrival time between consecutive FORWARD packets only. | 0.0 for fewer than 2 forward packets. | yes | yes |
| 28 | `forward_iat_std` | float | seconds | forward | Population standard deviation of inter-arrival time, forward packets only. | 0.0 for fewer than 3 forward packets. | yes | yes |
| 29 | `backward_iat_mean` | float | seconds | backward | Mean inter-arrival time between consecutive BACKWARD packets only. | 0.0 for fewer than 2 backward packets, including zero backward packets. | yes | yes |
| 30 | `backward_iat_std` | float | seconds | backward | Population standard deviation of inter-arrival time, backward packets only. | 0.0 for fewer than 3 backward packets. | yes | yes |
| 31 | `syn_count` | int | count | overall | Number of packets in the flow with the TCP SYN flag set. | 0 for non-TCP flows. | yes | yes |
| 32 | `ack_count` | int | count | overall | Number of packets in the flow with the TCP ACK flag set. | 0 for non-TCP flows. | yes | yes |
| 33 | `fin_count` | int | count | overall | Number of packets in the flow with the TCP FIN flag set. | 0 for non-TCP flows. | yes | yes |
| 34 | `rst_count` | int | count | overall | Number of packets in the flow with the TCP RST flag set. | 0 for non-TCP flows. | yes | yes |
| 35 | `psh_count` | int | count | overall | Number of packets in the flow with the TCP PSH flag set. | 0 for non-TCP flows. | yes | yes |
| 36 | `urg_count` | int | count | overall | Number of packets in the flow with the TCP URG flag set. | 0 for non-TCP flows. | yes | yes |
| 37 | `forward_packets_per_second` | float | packets/sec | forward | total_forward_packets / flow_duration_seconds. | Defined as 0.0 (not NaN/Inf) when flow_duration_seconds == 0 -- i.e. a single-timestamp flow's rate is undefined and we define it as 0 by convention. See docs/methodology.md for why this needs validating against how the training dataset handles the same case. | yes | yes |
| 38 | `backward_packets_per_second` | float | packets/sec | backward | total_backward_packets / flow_duration_seconds. | 0.0 when flow_duration_seconds == 0, same convention as forward_packets_per_second. | yes | yes |
| 39 | `total_packets_per_second` | float | packets/sec | overall | total_packets / flow_duration_seconds. | 0.0 when flow_duration_seconds == 0. | yes | yes |
| 40 | `total_bytes_per_second` | float | bytes/sec | overall | total_bytes / flow_duration_seconds. | 0.0 when flow_duration_seconds == 0. | yes | yes |

## Phase 5: rule-based detection

**What it does.** `app.detection.rule_engine.RuleDetector` runs a fixed set
of stateless, single-flow rules (`app.detection.rules.*`) against one
flow's `FeatureVector`, and combines any triggers into a `DetectionResult`
carrying a risk score (0-100), a severity bucket, and one `RuleTrigger` per
triggered rule (name, human-readable reason, score contribution, and the
specific feature values that justified it).

**Why heuristics, not ground truth.** Every rule here is explicitly a
*heuristic indicator* -- a pattern statistically associated with suspicious
behavior, never proof of an attack on its own. A legitimate bulk file
transfer can trip `high_packet_rate`; an ordinary blocked connection
attempt can trip `syn_heavy`. `RuleDetector` deliberately has no concept of
"confirmed malicious" -- only "worth a closer look." This framing is
intentional and matches the project's stated goal of measurable, honest
detection rather than overclaimed accuracy.

**Rules implemented** (all single-flow, stateless -- see below for what's
deliberately NOT implemented):

| Rule | Signal | Base score | Rationale |
|---|---|---|---|
| `high_packet_rate` | `total_packets_per_second` > threshold | 30 | Unusually fast flow; also true of legitimate high-throughput transfers |
| `high_byte_rate` | `total_bytes_per_second` > threshold | 25 | High-throughput flow; independent signal from packet rate (few large packets vs. many small ones) |
| `syn_heavy` | SYN packets ≥ ratio of total, above a minimum packet count | 35 | Associated with SYN scanning/flooding; also produced by an ordinary unanswered connection attempt |
| `tcp_flag_anomaly` | SYN+FIN with no ACK, or FIN/PSH/URG with no SYN/ACK (Null/Xmas pattern) | 40 | Known crafted-packet scanning patterns; flag-count based, not full packet-sequence analysis |
| `short_high_volume_burst` | High packet count within a short duration | 20 | Burst pattern distinct from a sustained high rate |
| `suspicious_destination_port` | Destination port on a configurable list (23, 445, 3389, 4444, 31337 by default) | 10 | Weakest signal by design -- a port alone proves nothing |

**Risk score aggregation:** sum of triggered rules' `base_score`, capped at
100. This is a simple, explainable, operational scheme -- multiple weak
signals add up to a stronger one -- not a statistically fitted or
ROC-calibrated score. Severity buckets reuse the existing
`RiskScoringConfig` boundaries from `app.config` (0-29 LOW, 30-59 MEDIUM,
60-79 HIGH, 80-100 CRITICAL), the same scheme documented as "operational,
not scientifically optimal" since Phase 1.

**Configuration:** every threshold lives in `DetectionThresholds`
(`app.config`) -- `high_packet_rate_pps`, `high_byte_rate_bps`,
`syn_heavy_ratio_threshold`, `syn_heavy_min_packets`,
`short_flow_max_duration_seconds`, `short_flow_min_packets`,
`flag_anomaly_min_packets`, `suspicious_destination_ports`. No magic
numbers are scattered in rule logic; each rule reads its own named
threshold field.

**Deliberately NOT implemented in Phase 5 (documented, not silently
skipped):** the pre-existing `DetectionThresholds.port_scan_*`,
`.high_conn_rate_per_sec`, and `.packet_rate_zscore_threshold` fields
describe **cross-flow** behavior (e.g. how many distinct ports one source
IP has touched across many flows in a time window). `RuleDetector`
operates on exactly one flow at a time and has no such state. Those fields
remain in config, unused, reserved for whichever future phase adds a
stateful multi-flow correlation layer -- they were not repurposed or
quietly redefined to fit Phase 5's scope.

**Limitations:**
- Flag-combination rules look at aggregate flag *counts* over the whole
  flow, not packet-by-packet sequence order.
- Thresholds are reasonable operational defaults, not fitted against any
  labeled traffic yet -- Phase 6's models are trained independently and
  are not yet fused with rule output (that fusion is Phase 8/9 territory).
- No cross-flow / multi-source correlation (see above).

## Phase 6: dataset preprocessing + offline ML training

**Scope.** Fully offline: dataset CSV in, trained model + metadata out.
No code in `app.ml.*` is imported by the capture/flow/feature/detection
pipeline, and no live inference exists yet (Phase 7).

**Pipeline:** `app.ml.pipeline.run_training_pipeline()` --

```
CSV (CICFlowMeter-style)
  -> load_dataset_csv                (app.ml.dataset)
  -> normalize_columns + build_compatibility_report   (app.ml.feature_mapping) -- STOP if incompatible
  -> align_features                   -> exactly the 41 live features, in FEATURE_NAMES order
  -> normalize_labels_binary           (app.ml.preprocessing) -- BENIGN=0, everything else=1
  -> clean_aligned_dataset             -- duplicates / infinite / NaN rows, each counted and logged
  -> split_and_scale                   -- stratified train/val/test, StandardScaler fit on train only
  -> train_and_evaluate (x3 models)    (app.ml.training)
  -> select_best_model (macro F1 on validation)
  -> evaluate_fitted_model (test set, once, no refit)
  -> save_model_artifact               (app.ml.persistence)
```

**Feature alignment -- the core Phase 6 requirement.** `app.ml.feature_mapping`
is the single source of the dataset-column -> live-feature mapping. Every
mapping is an explicit, documented assertion (`DIRECT_COLUMN_MAP`), not a
guess:

- Column-name whitespace (a known CIC-IDS2017 CSV quirk, e.g.
  `" Destination Port"`) is stripped before matching.
- `Flow Duration` and all `*IAT*` columns are converted from the dataset's
  **microseconds** to the live schema's **seconds** (`* 1e-6`) -- this unit
  mismatch would silently corrupt every duration/IAT/rate feature if
  missed, so it's handled explicitly with a named scale factor per column,
  not inferred.
- `total_packets` / `total_bytes` are **computed** (forward + backward),
  not copied from a single column, because CICFlowMeter's CSV doesn't
  export a combined total directly.
- `Protocol` (an IANA protocol number: 6/17/1) is converted to the
  `protocol_is_tcp/udp/icmp` one-hot triple used live.
- `build_compatibility_report()` reports matched-direct, matched-computed,
  matched-protocol, and (critically) **missing** features plus unmapped
  dataset columns, before any training starts. `align_features()` calls
  this and raises `FeatureAlignmentError` -- refusing to proceed -- if any
  of the 41 live features or the label column can't be obtained. Nothing
  is silently substituted or invented.

**Label handling.** Binary only for Phase 6 (`BINARY_LABEL_MAP = {"BENIGN": 0, "ATTACK": 1}`
in `app.ml.preprocessing`): any label not exactly "BENIGN" (case-insensitive,
whitespace-stripped) becomes ATTACK. The original label string is preserved
alongside the binary one (`normalize_labels_binary` returns both), so a
future multiclass phase doesn't need to re-load raw data. Multiclass was
deliberately not attempted now, per the "don't over-engineer" instruction.

**Cleaning.** `clean_aligned_dataset()` removes exact duplicate rows,
then rows with any infinite feature value, then rows with any missing
(NaN) feature value -- each count reported in `CleaningReport.summary()`,
never silently dropped without visibility.

**Split and leakage avoidance.** Stratified (on the binary label)
train/validation/test split via `sklearn.train_test_split`, fixed
`random_state`. `StandardScaler` is fit **only** on the training split
(`split_and_scale`) and applied unchanged to validation/test -- verified
by a test asserting the scaled training data has exactly zero mean/unit
variance (which only holds if the scaler was fit on that exact data).

**KNOWN LIMITATION, stated plainly:** this is a random row-level
stratified split, **not** the scenario/time-aware split described in the
original project methodology (to prevent the same attack session
appearing in both train and test). CICFlowMeter's standard CSV export
carries no explicit session/scenario identifier to split on without
additional, dataset-specific engineering. Deferred rather than
approximated incorrectly.

**Models.** Logistic Regression (`class_weight="balanced"`, `max_iter=1000`),
Decision Tree (`max_depth=15`, `class_weight="balanced"`), Random Forest
(`n_estimators=100`, `max_depth=15`, `class_weight="balanced"`) --
`app.ml.training.build_baseline_models()`, all with `random_state=42` for
reproducibility, modest hyperparameters chosen for understandability over
tuning. `class_weight="balanced"` is used on all three specifically
because intrusion datasets are class-imbalanced (per the project's
explicit warning against relying on plain accuracy).

**Evaluation.** `ModelEvalResult` reports accuracy, precision, recall, F1,
macro F1, weighted F1, ROC-AUC, PR-AUC (when both classes are present in
the eval split), false positive rate, false negative rate, raw TP/TN/FP/FN
counts, the full confusion matrix, training time, and inference time --
computed by actual `sklearn.metrics` calls against real predictions, never
hardcoded. `select_best_model()` picks the highest **macro F1** on the
validation split specifically because macro F1 (unweighted average across
classes) doesn't let a large BENIGN majority mask poor ATTACK-class
performance the way plain accuracy or weighted F1 could -- a documented
tie-breaking choice, not a claim of universal correctness. The selected
model is evaluated exactly once more on the held-out test set (no
refitting), so the test set is never touched during model selection.

**Model persistence.** `save_model_artifact()` writes three files per
model: the joblib-serialized model, the joblib-serialized fitted scaler,
and a JSON metadata file recording `feature_schema_version`,
`feature_names` (exact order), `feature_count`, `label_mapping`,
`training_config`, the saved model's evaluation metrics, and a UTC
timestamp. `load_model_artifact()` raises `SchemaVersionMismatchError` if
the saved schema version OR the exact feature name/order doesn't match the
CURRENT live schema -- this is what makes it safe for a future Phase 7 to
load an artifact without silently misinterpreting its feature vector.

**No fake metrics.** Every number in this document and in any training run
output comes from an actual `sklearn` computation against actual
predictions on actual (synthetic, for testing, or real CIC-IDS2017, for
production use) data. No example accuracy/precision/recall/F1 figures are
quoted here as if they were empirical results, because Phase 6 has only
been run against the synthetic test fixture so far -- see "Open items"
below.

**Open items / not yet empirically validated:**
1. **The full pipeline has not yet been run against the real
   CIC-IDS2017 dataset** -- only against a small synthetic fixture built
   for unit testing (deliberately, trivially separable, so its ~100%
   accuracy numbers mean nothing about real-world performance). Real
   metrics require running `app.ml.train_cli` against the actual dataset.
2. Whether the population-vs-sample standard deviation assumption (Phase 4)
   matches CICFlowMeter's actual convention is still unverified against
   source documentation.
3. Whether the real CIC-IDS2017 CSV's exact column names match
   `DIRECT_COLUMN_MAP` -- the mapping is built from well-documented public
   CICFlowMeter column conventions, but `build_compatibility_report()`
   should be run against the actual downloaded file before trusting it
   blindly; if column names differ (e.g. a different CICFlowMeter version),
   the report will say so explicitly rather than silently mismatching.

## Phase 7: live ML inference

**What it does.** `app.ml.inference.MLPredictor` loads a Phase 6 model
artifact triple (model + scaler + metadata) and runs inference on a single
flow's `FeatureVector`, returning a typed `MLInferenceResult`.

**Reuse, not duplication.** Two things were explicitly NOT reimplemented:

1. **Feature extraction** -- `MLPredictor` only ever accepts an already-built
   `FeatureVector` (from the existing `FeatureExtractor`). It has no
   knowledge of `FinalizedFlow` internals.
2. **Schema/artifact validation at load time** -- `MLPredictor.load()`
   delegates entirely to the existing `app.ml.persistence.load_model_artifact()`,
   which already raises `SchemaVersionMismatchError` if the saved feature
   schema version or exact feature name/order doesn't match the current
   live schema. `MLPredictor` adds one further, narrower check at
   *predict* time: that the specific `FeatureVector` passed in matches
   what *this* loaded model's metadata expects (schema version string and
   feature count) -- a defense-in-depth check distinct from the load-time
   one, since `FeatureVector.__post_init__` only guards against the
   currently-imported live schema, not against a specific model's
   recorded expectations.

**Preprocessing.** Exactly the `StandardScaler` instance saved during
Phase 6 training is applied (`scaler.transform`) -- no separate scaling
logic exists in the inference path.

**Probability handling.** `predict_proba`'s column order follows the
model's `classes_` attribute, which is not guaranteed to be `[0, 1]`.
`MLPredictor._compute_probability` looks up the ATTACK class's column
explicitly via `classes_.index(attack_label)` rather than assuming a fixed
index -- verified by a test using a stub model with **reversed**
`classes_ = [1, 0]`, confirming the probability comes from the correct
column regardless of order. `probability` is `None` (never a guessed or
default value) when: the model has no `predict_proba`, or its `classes_`
never includes the ATTACK label (e.g. a degenerate single-class model).

**Error handling.** Four typed exceptions, all raised with an actionable
message: `ModelArtifactNotFoundError` (missing file), `ModelArtifactCorruptedError`
(malformed JSON / missing metadata key / joblib deserialization failure),
`InvalidFeatureVectorError` (schema version or feature-count mismatch at
predict time), and `SchemaVersionMismatchError` (re-exported from
`app.ml.persistence`, raised at load time).

**Integration status.** `MLPredictor` is a ready-to-use capability, not yet
wired into a continuous capture-processing loop -- no such loop exists in
the project yet (Phases 2-5 are reusable components exercised via tests
and manual scripts). A single-row sklearn `predict()`/`predict_proba()`
call is a fast, synchronous, CPU-bound computation with no I/O, so there is
nothing to make asynchronous yet; genuine non-blocking integration (e.g. a
thread pool, if a future model were large enough to matter) is deferred
until an actual orchestrator exists to decouple it from.

**Limitations:**
- Not yet connected to Phase 5's `RuleDetector` -- that fusion is Phase 8.
- Every test in `test_ml_inference.py` uses a real but trivially-trained
  model (fit on 40 random points, `X[20:, 0] += 10.0` to make it
  separable) to exercise the inference *plumbing*. This is explicitly not
  evidence of real detection accuracy -- that requires training against
  CIC-IDS2017 (Phase 6's open item, still unresolved).
- `MLPredictor.load()` reads three separate files per model
  (`{name}.joblib`, `{name}.scaler.joblib`, `{name}.metadata.json`) via
  `app.config.Settings.ml_model_dir` / `ml_active_model_name`. During
  Phase 7 implementation I found the Phase 1 placeholder fields
  `ml_model_path` / `ml_feature_schema_path` assumed a single-file format
  that doesn't match Phase 6's actual three-file persistence design --
  they're left in `config.py`, unused, for backward `.env` compatibility,
  rather than silently repurposed to mean something they don't say.

## Phase 8: hybrid rule + ML detection

**What it does.** `app.detection.hybrid_detector.HybridDetector` runs the
existing `RuleDetector` (Phase 5) and, optionally, the existing
`MLPredictor` (Phase 7) against one flow's `FeatureVector`, then fuses
their outputs into a single `HybridDetectionResult`. Neither existing
component is modified or reimplemented -- `HybridDetector` is purely an
orchestrator on top of both, verified by a test confirming `RuleDetector`
produces byte-identical results whether used standalone or from inside a
`HybridDetector`.

**Fusion formula:**

```
final_risk_score = round(rule_weight * rule_risk_score + ml_weight * ml_score)
```

where `ml_score` is:
- `0.0` if the model predicts BENIGN,
- `probability * 100.0` if the model predicts ATTACK and reports a probability,
- `ml_no_probability_fallback_score` (default 60.0) if the model predicts
  ATTACK but can't report a probability.

**Why 0.5/0.5 by default, explicitly.** Per `HybridScoringConfig` in
`app.config`: neither the Phase 5 rule thresholds nor any Phase 6/7 model
have been empirically validated against real labeled attack traffic yet
(no training run against CIC-IDS2017 has happened). With no evidence that
one detector is more trustworthy than the other, an even split is the
only defensible default -- it is explicitly NOT claimed to be optimal, and
is meant to be re-tuned once Phase 12 benchmarking produces real
precision/recall numbers per detector. `rule_weight + ml_weight` is
validated to sum to 1.0 at `HybridDetector` construction time, failing
loudly rather than silently producing a miscalibrated score.

**The four required combination cases** (each with a dedicated test in
`test_hybrid_detector.py`):

| Rules | ML | Behavior |
|---|---|---|
| benign | benign | `final_risk_score = 0`, `Severity.NONE` |
| suspicious | benign | ML pulls the combined score down from the pure-rule score, but a strongly-flagged flow still registers as somewhat suspicious (score > 0) |
| benign | suspicious | ML alone can surface a flow rules missed (e.g. rule-benign + 90% ATTACK probability → score 45, `Severity.MEDIUM`) |
| suspicious | suspicious | Both contribute; combined score typically higher than either alone, capped at 100 |

**Graceful degradation.** Two distinct "no ML signal" cases, both handled
the same way -- fall back to the rule score alone, NOT scaled down by
`rule_weight` as if ML had said "benign" (that would incorrectly punish a
rule-flagged flow just because ML wasn't available):
1. `ml_predictor=None` (rules-only mode) -- e.g. no model has been trained yet.
2. `MLPredictor.predict()` raises any `MLInferenceError` subclass (missing
   artifact, schema mismatch, etc.) -- caught, logged, and recorded in
   `HybridDetectionResult.ml_error` as a human-readable string. Any
   *other* exception type is deliberately NOT caught here, since that
   would hide a real bug rather than a known, typed ML-unavailability
   condition -- verified by a test using a stub that raises `RuntimeError`.

**Severity classification** was extracted from `RuleDetector` into a
shared `app.detection.severity.classify_severity()` function during this
phase specifically so `HybridDetector` wouldn't duplicate the bucket
logic. `RuleDetector._classify_severity()` now delegates to it; its
signature, behavior, and existing test (`test_severity_classification_boundaries_directly`)
are unchanged.

**Limitations:**
- The 0.5/0.5 default weighting (and the `ml_no_probability_fallback_score`
  of 60.0) are operational placeholders, not fitted values -- see rationale
  above.
- No orchestrator continuously feeds live flows through `HybridDetector`
  yet -- same status as Phase 7's `MLPredictor`, this is a ready-to-use
  capability awaiting a real-time service loop.
- All hybrid tests use either a stub predictor (for fusion-logic tests) or
  a real-but-trivially-trained model (for the one full-pipeline
  integration test) -- neither is evidence of real detection accuracy.

## Phase 9: PostgreSQL persistence

**What it does.** `app.database.repository.FlowDetectionRepository` persists
`FinalizedFlow` (Phase 3) and `HybridDetectionResult` (Phase 8) domain
objects into two tables (`flows`, `detections`, FK-linked), and provides
read queries (`get_flow`, `get_detection`, `get_recent_detections`,
`get_detections_for_flow`, `count_detections_by_severity`) for a future
dashboard/API (Phase 10/11).

**Schema.**
- `flows`: timestamps (`first_seen`/`last_seen`), 5-tuple (`src_ip`,
  `dst_ip`, `src_port`, `dst_port`, `protocol`), duration, packet/byte
  counts (total + forward/backward), and an *optional* JSON
  `feature_snapshot` (the full 41-feature dict, when the caller passes a
  `FeatureVector`). No raw packet payloads are ever stored -- only
  already-aggregated flow metadata and already-computed features.
- `detections`: FK to `flows`, timestamp, rule score, triggered rule names
  (JSON list), rule evidence (JSON dict keyed by rule name), ML fields
  (model type, predicted class, probability, error -- each nullable, since
  ML is optional per Phase 8), final fused risk score, severity, and the
  fusion weights used.
- Indexes on `flows.src_ip`, `flows.dst_ip`, `flows.first_seen`, and
  `detections.flow_id`, `.timestamp`, `.final_risk_score`, `.severity` --
  the columns this project's own dashboard design (`docs/architecture.md`)
  identifies as commonly queried.

**Decoupling.** `FlowDetectionRepository` is the *only* module that
imports `sqlalchemy.exc` -- every failure (missing row, foreign-key
violation, connection error surfaced through the ORM) is wrapped in one
typed `DatabasePersistenceError`, so a future orchestrator or API layer
never needs to import SQLAlchemy or know it's PostgreSQL underneath.
Verified by a test asserting the original SQLAlchemy exception is
preserved via `__cause__` for debugging, without leaking its type to the
caller's `except` clause.

**Known, documented gap in that decoupling:** if the *session factory
itself* raises (e.g. a connection pool that fails outside of any ORM call)
rather than an operation raised through an open session, that exception is
NOT currently wrapped -- verified explicitly by
`test_broken_session_factory_raises_persistence_error_not_a_crash`, which
shows a `RuntimeError` from a broken `session_factory()` propagates
as-is. In practice, `sessionmaker`-produced factories raise through
`SQLAlchemyError` subclasses for real connection failures, so this gap is
narrow, but it's called out rather than silently assumed away.

**Schema initialization.** `app.database.session.init_db()` uses
SQLAlchemy's `Base.metadata.create_all()` -- sufficient for this project's
current single schema version. Alembic (the natural migration tool, given
SQLAlchemy is already a dependency) is deferred rather than introduced now:
there is no second schema version yet to migrate between, so adding a full
migration framework ahead of that need would be complexity without a
corresponding requirement. The reserved `database/migrations/` directory
at the repo root is where that tooling would live once it's needed.

**Testing.** All 19 non-skipped tests run against in-memory SQLite
(`sqlite:///:memory:`), with foreign-key enforcement explicitly turned on
via a `PRAGMA foreign_keys=ON` connect-event listener (SQLite defaults to
FK enforcement OFF, unlike PostgreSQL, which would have made the
constraint-violation test a false pass). One additional, clearly
documented real-PostgreSQL integration test exists
(`test_real_postgres_save_and_query_round_trip`), skipped unless
`NIDS_TEST_POSTGRES_URL` is set -- run it locally via:

```bash
docker compose up -d postgres
NIDS_TEST_POSTGRES_URL='postgresql+psycopg://nids:nids@localhost:5432/nids' \
    pytest tests/test_database.py -k postgres
```

**This test has not been run against a real PostgreSQL instance in this
development environment** (no Docker/Postgres available in the sandbox
used to build this phase) -- it is verified to skip cleanly with a clear
reason, but its actual pass/fail against real PostgreSQL is unconfirmed
and should be run in your local Windows/Docker environment before relying
on it.

**Limitations:**
- No connection pooling/retry policy beyond SQLAlchemy's defaults --
  acceptable for this project's current stage, worth revisiting under
  Phase 12 benchmarking if write throughput under load matters.
- `count_detections_by_severity()` loads all severity values into Python
  to count them rather than using a `GROUP BY` -- simple and fine at this
  project's current scale; would want a real aggregate query if the
  `detections` table grows large.
- No orchestrator yet calls `save_flow`/`save_detection` continuously from
  live traffic -- same status as Phases 7/8, this is a ready capability
  awaiting a real-time service loop (or Phase 10's API layer, which may
  itself be that first caller).

## Phase 10: FastAPI REST API + WebSocket

**What it does.** Exposes the existing detection/persistence stack over
HTTP and WebSocket:

- `GET /api/health` -- unchanged from Phase 1 (basic liveness).
- `GET /api/status` -- environment, capture interface, active ML model
  name, live database reachability, and totals. Queries the database but
  never fails the *request* if the DB is unreachable -- `database_reachable: false`
  is itself a valid, useful status response.
- `GET /api/flows`, `GET /api/flows/{id}`, `GET /api/flows/{id}/alerts`
- `GET /api/alerts` (paginated, optional `severity` filter),
  `GET /api/alerts/{id}` (full detail including rule evidence)
- `GET /api/stats/detections`, `GET /api/stats/traffic`
- `WS /ws/alerts` -- real-time detection event stream

**No duplicated logic.** Every route is a thin translation layer: FastAPI
routers call the existing `FlowDetectionRepository` (Phase 9) and convert
its read-view dataclasses to Pydantic response models
(`app.api.schemas`). No route reimplements a database query, a detection
rule, or ML inference -- confirmed by inspection and by the full test
suite passing unchanged before any Phase 10 code was written.

**Repository extension, not rewrite.** Phase 9's `FlowDetectionRepository`
gained five new, additive methods (`count_flows`, `count_detections`,
`list_flows`, `get_recent_alerts`, `get_traffic_summary`) needed by the
new endpoints. Every pre-existing Phase 9 method, and all 19 of its
existing tests, are unchanged -- verified by running `test_database.py`
immediately after the extension, before writing any route code. The new
aggregate methods use real SQL (`func.count`/`func.sum`/`func.avg`,
`GROUP BY`) rather than loading rows into Python, in contrast to Phase 9's
`count_detections_by_severity()` (whose Python-side counting was already a
documented Phase 9 limitation) -- that pre-existing method was left
exactly as it was rather than retroactively rewritten.

**`detection_source` field.** Alerts are labeled `"rule"`, `"ml"`,
`"hybrid"`, or `"none"` by `derive_detection_source()` in `app.api.schemas`,
computed from the persisted `rule_score` and `ml_predicted_class` at
serialization time (not stored in the database -- it's a presentation
concern, kept out of the persistence layer per the project's "keep API
logic separate from database implementation where practical"
requirement).

**Typed models, input validation, HTTP errors.** All responses are typed
Pydantic models. `limit` (1-500) and `offset` (>=0) query parameters are
validated by FastAPI/Pydantic automatically (invalid values -> 422).
`severity` is validated explicitly against the `Severity` enum's values
(422 with a clear message on mismatch, rather than silently returning zero
rows for a typo'd severity). Not-found resources return 404. A database
failure surfaces as 503 with the wrapped `DatabasePersistenceError`
message -- never a raw 500 traceback.

**WebSocket.** `ConnectionManager` (`app.websocket.manager`) tracks
connections in an `asyncio.Lock`-guarded set. `broadcast()` iterates a
snapshot of connections, catches any per-connection send failure, and
removes only the failed connection -- verified by a test using a
deliberately-broken fake WebSocket alongside a working one, confirming the
working client still receives the message and `broadcast()` doesn't raise.
Client disconnects are handled via `WebSocketDisconnect` in the route,
with `finally: await connection_manager.disconnect(websocket)` guaranteeing
cleanup even on an unclean disconnect -- verified by a test that opens and
closes a real `TestClient` WebSocket connection and polls until the
server-side connection count returns to its prior value.

**Testing approach.** All 27 API tests use FastAPI's `app.dependency_overrides`
to swap in a `FlowDetectionRepository` backed by in-memory SQLite -- no
live PostgreSQL required. One non-obvious fix was necessary: SQLite's
`:memory:` database is per-connection, and FastAPI's `TestClient` runs
route handlers in a different thread than the test itself, so without
`poolclass=StaticPool` (and `connect_args={"check_same_thread": False}`),
the test's seeded data and the app's own queries silently pointed at two
different, unconnected in-memory databases (surfaced as "no such table"
errors during initial development of this test file, not a hidden fix --
documented here since it's a genuinely easy mistake to reintroduce).

**Limitations:**
- No live orchestrator calls `ConnectionManager.broadcast()` from real
  detections yet -- same status as every "ready capability" noted since
  Phase 7. `/ws/alerts` is fully functional and tested, but nothing
  currently pushes real detection events through it outside of tests.
- No authentication/authorization on any endpoint -- appropriate for a lab
  project, not for any non-lab deployment.
- No rate limiting.
- `get_recent_alerts`/`list_flows` pagination is offset-based, which is
  simple but not the most efficient approach at very large table sizes
  (keyset pagination would scale better) -- acceptable at this project's
  current scale.
