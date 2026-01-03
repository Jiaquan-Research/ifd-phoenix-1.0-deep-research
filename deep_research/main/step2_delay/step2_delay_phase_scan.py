"""
Phoenix / IFD 1.0
Experiment 2: Delay-Induced Phase Collapse (Law 4)

---------------------------------------------------------
PHYSICS INTENT (Concept Anchor)
---------------------------------------------------------
Concept:
    Law 4 — Delay-Induced Phase Collapse

Physical Variable:
    Feedback Delay (Perceptual Blindness)

Explicitly NOT:
    - Actuation Delay (execution lag)
    - World-level physics latency

Implementation Definition:
    - Agent observes its own internal state with a fixed delay K.
    - Decision is made based on stale (t - K) information.
    - Action effects are applied immediately (no actuation lag).

Expected Phenomenon:
    - If Delay < Phase Margin:
          Agent stabilizes via reactive control.
    - If Delay > Phase Margin:
          Control becomes phase-inverted → oscillation → extinction.

Purpose:
    This is the PRIMARY experiment demonstrating Law 4.
    Supplementary calibration is handled in Step 2b.
---------------------------------------------------------
"""

from __future__ import annotations

import csv
import json
import random
from collections import deque
from pathlib import Path
from statistics import mean

# ============================================================
# A. Parameters (Frozen — calibrated via Step 2b)
# ============================================================

SEEDS = [2026, 2027, 2028, 2029, 2030]
POPULATION = 100
STEPS = 600

# High Gain to strictly rule out Poverty Trap
GAIN = 8.0
METABOLIC_COST = 0.05

# Scan Range (crosses phase margin)
DELAYS = [10, 20, 30, 40, 50]

# Robust initial buffer (survives blindness window)
INITIAL_STATE = 20.0
ALARM_THRESHOLD = 10.0


# ============================================================
# B. System-1 Agent with Explicit Feedback Delay
# ============================================================

class System1BlindAgent:
    """
    Reflex-based System-1 agent with delayed perception.
    """

    def __init__(self, delay: int):
        self.E = INITIAL_STATE
        self.H = INITIAL_STATE
        self.I = INITIAL_STATE

        self.delay = delay

        # Feedback delay queue: stores historical internal states
        self.obs_queue = deque(
            [(INITIAL_STATE, INITIAL_STATE, INITIAL_STATE)] * (delay + 1),
            maxlen=delay + 1,
        )

    def step(self):
        # 1. Observe true current state
        current_state = (self.E, self.H, self.I)
        self.obs_queue.append(current_state)

        # 2. Perceive delayed state (blindness)
        perceived_E, perceived_H, perceived_I = self.obs_queue[0]

        # 3. System-1 decision rule (reflex hysteresis)
        if perceived_H < ALARM_THRESHOLD:
            action = "REST"
        else:
            action = "FORAGE"

        # 4. Immediate actuation (no execution delay)
        if action == "FORAGE":
            dE = GAIN
            dH = -0.30
            dI = -0.05
        else:  # REST
            dE = -0.5
            dH = 0.50
            dI = -0.05

        # 5. State update
        self.E += dE
        self.H += dH
        self.I += dI

        # Metabolic tax
        self.E -= METABOLIC_COST
        self.H -= METABOLIC_COST
        self.I -= METABOLIC_COST

        # Clamp upper bound (avoid runaway)
        limit = 40.0
        self.E = min(self.E, limit)
        self.H = min(self.H, limit)
        self.I = min(self.I, limit)

    def alive(self) -> bool:
        return self.E > 0 and self.H > 0 and self.I > 0


# ============================================================
# C. Runner
# ============================================================

def run_once(delay: int, seed: int) -> int:
    random.seed(seed)
    population = [System1BlindAgent(delay) for _ in range(POPULATION)]

    for t in range(STEPS):
        survivors = []
        for agent in population:
            agent.step()
            if agent.alive():
                survivors.append(agent)

        if not survivors:
            return t

        population = survivors

    return STEPS


# ============================================================
# D. Entry Point & Output
# ============================================================

def repo_root() -> Path:
    # deep_research/main/step2_delay/step2_delay_phase_scan.py
    # parents[3] -> deep_research
    return Path(__file__).resolve().parents[3]

def main():
    out_dir = Path("deep_research/results/step2_phase_collapse")
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []

    print("[*] PHOENIX — STEP 2 (Feedback Delay / Blindness)")
    print(f"[*] Gain={GAIN} | Pop={POPULATION} | Steps={STEPS}")
    print("=" * 70)
    print(f"{'DELAY':<10} | {'AVG DEATH':<12} | {'STATUS':<10}")
    print("-" * 70)

    for delay in DELAYS:
        deaths = [run_once(delay, seed) for seed in SEEDS]
        avg_death = int(mean(deaths))
        status = f"X_{avg_death:03d}" if avg_death < STEPS else "ALIVE"

        print(f"{delay:<10} | {avg_death:<12} | {status:<10}")

        rows.append({
            "gain": GAIN,
            "delay": delay,
            "avg_extinction_step": avg_death,
            "seeds": SEEDS,
        })

    print("=" * 70)

    # Save CSV
    csv_path = out_dir / "phase_collapse_scan.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["gain", "delay", "avg_extinction_step", "seeds"],
        )
        writer.writeheader()
        writer.writerows(rows)

    # Save config
    config = {
        "experiment": "Step 2 — Delay-Induced Phase Collapse",
        "law": "Law 4",
        "delay_type": "Feedback / Perceptual Blindness",
        "gain": GAIN,
        "initial_state": INITIAL_STATE,
        "alarm_threshold": ALARM_THRESHOLD,
        "note": "Primary evidence for phase collapse under delayed perception.",
    }

    (out_dir / "config.json").write_text(json.dumps(config, indent=2))
    print(f"[✓] Saved: {csv_path}")


if __name__ == "__main__":
    main()
