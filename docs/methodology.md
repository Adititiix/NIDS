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

