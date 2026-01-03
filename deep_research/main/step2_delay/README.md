# Step 2 — Delay-Induced Phase Collapse

## Purpose
This experiment demonstrates that **introducing feedback delay alone is sufficient to destabilize an otherwise viable control policy**.

Agents use the same fixed action policy as in Step 1.  
No learning, no adaptation, and no internal state reconstruction are allowed.

The only change from Step 1 is the presence of **delayed state observation**.

## Physics Definition (Critical)
- **Delay Type**: Feedback delay (observation lag)
- **Not** actuation delay (actions are applied immediately)
- Agent commits to an action at time *t* using state information from *t − k*

This explicitly models **blindness**, not motor lag.

## Expected Phenomenon
- For small delay: agents survive with reduced margin.
- Beyond a critical delay (phase margin): agents collapse catastrophically.
- Collapse timing becomes insensitive to further increases in delay.

This defines a **delay-induced phase transition** in survival dynamics.

## What This Experiment Is (and Is Not)
- **Is**: A causal demonstration that delay alone can destroy a valid control loop.
- **Is NOT**:
  - Learning failure
  - Poor reward shaping
  - Exploration noise
  - Insufficient optimization

The policy is held constant across all conditions.

## Role in the Paper
This is the **core experiment** supporting *Law 4: Delay-Induced Phase Collapse*.

Step 1 shows that a stable gradient exists.  
Step 2 shows that **the same gradient becomes lethal once feedback delay exceeds the phase margin**.

All subsequent steps (2b–4) refine, bound, or extend this phenomenon.
