"""
Phoenix / IFD — Step 2b (Diagnostic)
Phase-Margin Scan under a minimal System-1 reflex policy.

OBJECTIVE:
Validate that 'Phase Margin' is a real physical quantity.
We explicitly model Feedback Delay (Blindness).
The agent should survive short delays (within margin) and collapse under long delays.

PHYSICS CALIBRATION (Final):
- Initial Buffer: 20.0 (Robust start)
- Consumption: ~0.35/step (0.3 action + 0.05 metabolic)
- Alarm Threshold: 10.0 (Early warning system)
- Phase Margin Time: ~28 steps (Time from Action to Threshold Cross)
- Critical Delay: ~28 steps.
  -> Delay 20: Safe.
  -> Delay 30: Collapse.
"""

from __future__ import annotations

import csv
import json
import random
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np


# ---------------------------
# Scan grid
# ---------------------------
POP_SIZE = 200
STEPS = 600
SEEDS = [2026, 2027, 2028]

DELAYS = list(range(0, 51, 5))  # 0, 5, 10 ... 50
METABOLIC_RATES = [0.05]        # Fixed for clarity

# Initial state (High buffer to allow observable blind flight)
E0 = 20.0
H0 = 20.0
I0 = 20.0

# Physics Constants
# FORAGE: High Energy Gain, Moderate Structure Cost
FORAGE_PHYSICS = (8.0, -0.30, -0.05)   # (dE, dH, dI)
# REST: Energy Cost, Structure Recovery
REST_PHYSICS   = (-0.5, 0.50, -0.02)   # (dE, dH, dI)

# Reflex thresholds (System-1)
# Panic early (at 50% health) to maximize survival chance
H_THR = 10.0


@dataclass(frozen=True)
class Config:
    pop_size: int = POP_SIZE
    steps: int = STEPS
    seeds: List[int] = None
    delays: List[int] = None
    metabolic_rates: List[float] = None
    init_state: Tuple[float, float, float] = (E0, H0, I0)
    h_thr: float = H_THR


class Agent:
    def __init__(self, delay: int):
        self.delay = delay
        self.E = E0
        self.H = H0
        self.I = I0

        # Blindness Queue: Stores historical observations
        # Init with perfect knowledge of initial state
        self.q = deque([(E0, H0, I0)] * (delay + 1), maxlen=delay + 1)

    def step(self, metabolic_rate: float) -> None:
        # 1. Observation (Record current reality)
        current_state = (self.E, self.H, self.I)
        self.q.append(current_state)

        # 2. Perception (Read delayed reality)
        # q[0] is the oldest state (t - delay)
        perceived_E, perceived_H, perceived_I = self.q[0]

        # 3. Policy (Reflex based on PERCEPTION)
        if perceived_H < H_THR:
            act = "REST"
        else:
            act = "FORAGE"

        # 4. Actuation (Apply to REALITY)
        dE, dH, dI = FORAGE_PHYSICS if act == "FORAGE" else REST_PHYSICS

        self.E += dE
        self.H += dH
        self.I += dI

        # 5. Metabolic Tax
        self.E -= metabolic_rate
        self.H -= metabolic_rate
        self.I -= metabolic_rate

        # 6. Clamp (Prevent infinite hoarding hiding the crash)
        limit = 40.0
        self.E = min(self.E, limit)
        self.H = min(self.H, limit)
        self.I = min(self.I, limit)

    def alive(self) -> bool:
        return self.E > 0 and self.H > 0 and self.I > 0


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def run_once(delay: int, metabolic_rate: float, seed: int) -> int:
    random.seed(seed)
    np.random.seed(seed)

    pop = [Agent(delay) for _ in range(POP_SIZE)]

    for t in range(STEPS):
        alive = []
        for a in pop:
            a.step(metabolic_rate)
            if a.alive():
                alive.append(a)
        if not alive:
            return t
        pop = alive
    return STEPS


def main() -> None:
    out_dir = repo_root() / "results" / "step2b_phase_margin"
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = Config(
        seeds=SEEDS,
        delays=DELAYS,
        metabolic_rates=METABOLIC_RATES,
    )
    (out_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2), encoding="utf-8")

    rows = []
    print("[*] STEP 2b | Phase-Margin Scan (Calibrated)")
    print(f"[*] Pop={POP_SIZE} | Steps={STEPS} | Seeds={len(SEEDS)}")
    print("-" * 86)

    for m in METABOLIC_RATES:
        for d in DELAYS:
            deaths: List[int] = []
            for s in SEEDS:
                deaths.append(run_once(d, m, s))
            avg_death = sum(deaths) / len(deaths)

            rows.append([m, d, avg_death, int(avg_death >= STEPS)])

            status = "ALIVE" if avg_death >= STEPS else f"DIED@{avg_death:.1f}"

            # Print clarity
            if d % 5 == 0:
                print(f"m={m:.2f} | delay={d:>2} | avg_death={avg_death:6.1f} | {status}")

        print("-" * 86)

    csv_path = out_dir / "phase_margin_scan.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["metabolic_rate", "delay", "avg_extinction_step", "survived_full_horizon"])
        w.writerows(rows)

    print(f"[✓] Saved: {csv_path}")


if __name__ == "__main__":
    main()