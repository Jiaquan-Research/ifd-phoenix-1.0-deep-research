# Step 1 — Gradient Revelation

## Purpose
This experiment demonstrates that an **optimal action gradient can emerge purely from survival selection**, without learning, prediction, memory, or delayed feedback.

Agents do **not** discover the gradient internally.  
The gradient is **revealed by the environment** through differential survival under fixed physics.

## What This Experiment Is (and Is Not)
- **Is**: An existence proof that environmental dynamics alone can imprint a stable mixed strategy.
- **Is NOT**: Learning, optimization, reinforcement learning, or intelligence.

There is no System-2, no adaptation, and no feedback delay in this step.

## Role in the Paper
This step establishes the **baseline geometry of the environment**.
Later steps (Step 2–4) introduce delay, blindness, and higher-order control to show how this revealed gradient becomes unstable and collapses.

## Reproducibility Note

This directory does not include pre-generated result files.

All figures and numerical outcomes described in the accompanying text
are deterministically reproducible by executing the script in this folder
under the frozen core (`src_frozen/`).

This design is intentional:
- to avoid cherry-picked artifacts
- to emphasize causal structure over numerical tuning