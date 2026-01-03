# Step 2b — Phase Margin Calibration (Supplement)

## Purpose
This supplementary experiment **calibrates the phase margin** of the environment used in Step 2.

While Step 2 demonstrates the existence of a sharp phase collapse under delayed feedback,  
Step 2b explicitly verifies **where that collapse boundary lies** by systematically scanning
feedback delay against known physical consumption rates.

This experiment is **not a discovery**, but a **calibration and consistency check**.

---

## What Is Being Calibrated
The system exhibits two distinct failure regimes:

- **Resource-Limited Regime**  
  Collapse occurs due to long-term structural or metabolic depletion.

- **Phase-Collapse Regime**  
  Collapse occurs immediately once **feedback delay exceeds the survivable blind interval**
  (phase margin), even when sufficient resources are available.

Step 2b isolates the second regime.

---

## Experimental Setup (Summary)
- Initial buffers are increased to exclude “birth-death” artifacts.
- Metabolic cost is fixed and known.
- Feedback delay is scanned across a wide range.
- Collapse time is averaged across multiple random seeds.

No learning, prediction, or adaptation mechanisms are introduced.

---

## Key Observation
A **sharp transition** is observed once delay exceeds the calibrated phase margin:

- For delays below the margin, agents survive until long-term structural exhaustion.
- For delays beyond the margin, agents collapse rapidly and deterministically.

The observed transition point matches the blind-interval estimate derived from physical consumption rates.
In our calibration runs (m = 0.05), this boundary is empirically located between delay = 25 (stable regime, avg. extinction ≈ 226 steps) and delay = 30 (phase-collapse regime, avg. extinction ≈ 57 steps).

## Role in the Overall Study
Step 2b serves as a **supporting calibration** for Step 2:

- Step 2 establishes the existence of delay-induced phase collapse.
- Step 2b confirms that the collapse boundary is **physically grounded**, not an artifact of parameter tuning.

All core claims remain supported by Step 2 alone.
