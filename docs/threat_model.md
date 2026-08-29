# Threat Model & Scope

## In scope

- Detecting suspicious network behavior (port scanning, high connection
  rates, SYN-ratio anomalies, packet-rate anomalies, and ML-classified
  known attack patterns) on traffic the operator is authorized to observe:
  their own machine, a personally-owned VM lab, or a network with explicit
  written authorization.
- All test/attack traffic used to validate detection is generated **inside**
  the isolated lab network described in `architecture.md` and never directed
  at third-party or public IP addresses.

## Explicitly out of scope

- Attacking, scanning, or probing any system the operator does not own or
  does not have explicit authorization to test.
- Evading or bypassing production security controls (this is a detector,
  not an evasion tool).
- Any guidance on attacking public infrastructure. Lab scenarios in
  `experiments/` are restricted to VM-to-VM traffic on an isolated virtual
  network segment.

## Known limitations (tracked honestly, not hidden)

- Encrypted traffic limits visibility into payload content; detection relies
  on flow/timing features, which is realistic but not complete.
- Offline dataset (CIC-IDS2017) traffic patterns will not perfectly match
  lab-generated traffic — dataset metrics and live-lab metrics are always
  reported and labeled separately, never conflated.
- Packet capture requires elevated privileges on the sensor VM.
- Rule thresholds are operational defaults, not values derived from a formal
  ROC/cost analysis — they're configurable and meant to be tuned per lab.
