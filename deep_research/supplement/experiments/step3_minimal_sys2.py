"""
Phoenix / IFD Deep Research Package
Step 3 — Minimal System-2 (Predictive Gating)  [SUPPLEMENT]

------------------------------------------------------------
PHYSICS INTENT (Concept Anchor)
------------------------------------------------------------
Concept:       Law 4 follow-up — Why "minimal System-2" is non-trivial.
Physical Var:  Feedback Delay / Blindness (observation is delayed by K).
               NOT Actuation Delay (action effects are immediate).
Goal:          Compare:
               - System-1: reactive policy driven by delayed observations.
               - System-2: minimal predictor + gating that acts on predicted current state.

Expected Phenomenon:
- Under high delay, System-1 collapses by acting on stale state.
- System-2 may survive longer (or still fail) depending on phase margin & buffer.
- This step is a *supplement* because it's an illustrative mechanism probe,
  not the core Law-4 demonstration (which is Step 2).
------------------------------------------------------------

Implementation Notes:
- We do NOT touch src_frozen.
- We intentionally keep the model minimal and auditable.
- We add an initial buffer to avoid "birth-death" artifacts (startup blindness).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
from dataclasses import asdict, dataclass
from typing import Dict, List, Tuple


# -----------------------------
# A. Parameters (Frozen Defaults)
# -----------------------------
DEFAULT_SEED = 2026
DEFAULT_POP = 100
DEFAULT_STEPS = 1000

# Delay = number of steps the agent's observation lags behind reality
DEFAULT_DELAY = 20

# Gain = food energy gained when FORAGE succeeds (deterministic here)
DEFAULT_GAIN = 4.0

# Physics: minimal survival bookkeeping
DEFAULT_INITIAL_BUFFER = 25.0  # <-- accepted change (prevents "birth-death")
DEFAULT_METABOLIC_COST = 0.05
DEFAULT_FORAGE_COST = 0.10     # effort cost per FORAGE step
DEFAULT_REST_RECOVERY = 0.00   # we keep REST "neutral" besides metabolism

# Structural / fatigue proxy (simple)
DEFAULT_DIST_INC_FORAGE = 0.06
DEFAULT_DIST_DECAY_REST = 0.03

# System-1 thresholds (hysteresis on *observed* dist)
DEFAULT_THR_LO = 0.30
DEFAULT_THR_HI = 0.60


# -----------------------------
# B. Utility: paths & IO
# -----------------------------
def _script_dir() -> str:
    return os.path.dirname(os.path.abspath(__file__))


def _results_dir() -> str:
    # deep_research/supplement/experiments -> deep_research/supplement/results/step3_sys2
    base = os.path.normpath(os.path.join(_script_dir(), "..","..", "results", "step3_sys2"))
    os.makedirs(base, exist_ok=True)
    return base


def _write_json(path: str, payload: Dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def _write_csv(path: str, header: List[str], rows: List[List]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


# -----------------------------
# C. Config (for reproducibility)
# -----------------------------
@dataclass(frozen=True)
class Step3Config:
    seed: int
    pop: int
    steps: int
    delay: int
    gain: float
    initial_buffer: float
    metabolic_cost: float
    forage_cost: float
    dist_inc_forage: float
    dist_decay_rest: float
    thr_lo: float
    thr_hi: float


# -----------------------------
# D. Agents
# -----------------------------
class System1Agent:
    """
    System-1: reactive policy driven by delayed observations only.
    No prediction, no learning.
    """
    __slots__ = ("H", "dist", "state")

    def __init__(self, initial_buffer: float, rng: random.Random):
        # Add initial buffer to survive the startup blindness window.
        # This is not "making agent smarter"; it's removing a numerical artifact.
        self.H = initial_buffer + rng.uniform(-1.0, 1.0)
        self.dist = 0.40 + rng.uniform(-0.02, 0.02)
        self.state = "ACT"  # start in ACT to stress the loop

    def decide(self, obs_dist: float, thr_lo: float, thr_hi: float) -> str:
        # Hysteresis on delayed observation
        if self.state == "ACT":
            self.state = "IDLE" if obs_dist > thr_hi else "ACT"
        else:
            self.state = "ACT" if obs_dist < thr_lo else "IDLE"
        return self.state


class System2Agent:
    """
    Minimal System-2: maintains a one-step-ahead predictor of current state
    (by simulating its own actions through known physics), and gates System-1.

    It does NOT see the true current state directly (still blind), but it can
    maintain a *model-based estimate* because action effects are immediate and known.
    """
    __slots__ = ("H", "dist", "state", "H_hat", "dist_hat")

    def __init__(self, initial_buffer: float, rng: random.Random):
        self.H = initial_buffer + rng.uniform(-1.0, 1.0)
        self.dist = 0.40 + rng.uniform(-0.02, 0.02)
        self.state = "ACT"

        # internal predictor states (start aligned)
        self.H_hat = self.H
        self.dist_hat = self.dist

    def decide(self, obs_dist: float, cfg: Step3Config) -> str:
        # System-1 proposes action based on delayed obs
        if self.state == "ACT":
            proposed = "IDLE" if obs_dist > cfg.thr_hi else "ACT"
        else:
            proposed = "ACT" if obs_dist < cfg.thr_lo else "IDLE"

        # System-2 gating: if predicted H_hat is too low, force REST (IDLE).
        # Minimal "safety gate" — no planning, just prevents suicidal ACT streaks.
        # You can tune this threshold later; keep it conservative for auditability.
        SAFE_H_MIN = 1.0

        gated = proposed
        if proposed == "ACT" and self.H_hat <= SAFE_H_MIN:
            gated = "IDLE"

        self.state = gated
        return gated

    def predictor_update(self, action: str, cfg: Step3Config) -> None:
        # advance the internal estimate using the same physics as the world
        if action == "ACT":  # forage
            self.H_hat += cfg.gain
            self.H_hat -= (cfg.metabolic_cost + cfg.forage_cost)
            self.dist_hat = min(1.0, self.dist_hat + cfg.dist_inc_forage)
        else:  # rest
            self.H_hat -= cfg.metabolic_cost
            self.dist_hat = max(0.0, self.dist_hat - cfg.dist_decay_rest)


# -----------------------------
# E. World stepping (feedback delay)
# -----------------------------
def step_physics(H: float, dist: float, action: str, cfg: Step3Config) -> Tuple[float, float]:
    if action == "ACT":
        H += cfg.gain
        H -= (cfg.metabolic_cost + cfg.forage_cost)
        dist = min(1.0, dist + cfg.dist_inc_forage)
    else:
        H -= cfg.metabolic_cost
        dist = max(0.0, dist - cfg.dist_decay_rest)
    return H, dist


def run_population(cfg: Step3Config) -> Dict[str, List[int]]:
    rng = random.Random(cfg.seed)

    # init populations
    s1: List[System1Agent] = [System1Agent(cfg.initial_buffer, rng) for _ in range(cfg.pop)]
    s2: List[System2Agent] = [System2Agent(cfg.initial_buffer, rng) for _ in range(cfg.pop)]

    # feedback delay buffers store observed dist (stale sensor)
    # each agent has its own delay line
    s1_obs = [[a.dist] * cfg.delay for a in s1]
    s2_obs = [[a.dist] * cfg.delay for a in s2]

    alive_s1_curve: List[int] = []
    alive_s2_curve: List[int] = []

    for t in range(cfg.steps):
        # Count alive at start-of-step
        alive_s1 = sum(1 for a in s1 if a.H > 0.0)
        alive_s2 = sum(1 for a in s2 if a.H > 0.0)
        alive_s1_curve.append(alive_s1)
        alive_s2_curve.append(alive_s2)

        if alive_s1 == 0 and alive_s2 == 0:
            break

        # Step all agents (deterministic order)
        for i, a in enumerate(s1):
            if a.H <= 0.0:
                continue
            obs_dist = s1_obs[i][0]  # delayed
            action = a.decide(obs_dist, cfg.thr_lo, cfg.thr_hi)
            a.H, a.dist = step_physics(a.H, a.dist, action, cfg)

            # push new observation into delay line (sensor sees dist with lag)
            s1_obs[i].append(a.dist)
            if len(s1_obs[i]) > cfg.delay:
                s1_obs[i].pop(0)

        for i, a in enumerate(s2):
            if a.H <= 0.0:
                continue
            obs_dist = s2_obs[i][0]
            action = a.decide(obs_dist, cfg)
            # predictor update first (System-2 internal bookkeeping)
            a.predictor_update(action, cfg)

            a.H, a.dist = step_physics(a.H, a.dist, action, cfg)

            s2_obs[i].append(a.dist)
            if len(s2_obs[i]) > cfg.delay:
                s2_obs[i].pop(0)

    return {"sys1_alive": alive_s1_curve, "sys2_alive": alive_s2_curve}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--pop", type=int, default=DEFAULT_POP)
    ap.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    ap.add_argument("--delay", type=int, default=DEFAULT_DELAY)
    ap.add_argument("--gain", type=float, default=DEFAULT_GAIN)
    ap.add_argument("--buffer", type=float, default=DEFAULT_INITIAL_BUFFER)
    args = ap.parse_args()

    cfg = Step3Config(
        seed=args.seed,
        pop=args.pop,
        steps=args.steps,
        delay=args.delay,
        gain=args.gain,
        initial_buffer=args.buffer,
        metabolic_cost=DEFAULT_METABOLIC_COST,
        forage_cost=DEFAULT_FORAGE_COST,
        dist_inc_forage=DEFAULT_DIST_INC_FORAGE,
        dist_decay_rest=DEFAULT_DIST_DECAY_REST,
        thr_lo=DEFAULT_THR_LO,
        thr_hi=DEFAULT_THR_HI,
    )

    print(f"[*] STEP 3 (SUPPLEMENT) | Delay={cfg.delay} | Gain={cfg.gain} | Pop={cfg.pop} | Steps={cfg.steps}")
    print("STEP   | SYS-1 ALIVE  | SYS-2 ALIVE ")
    print("----------------------------------------")

    curves = run_population(cfg)

    # Print sparse log
    for t in range(0, len(curves["sys1_alive"]), 50):
        print(f"{t:<6} | {curves['sys1_alive'][t]:<11} | {curves['sys2_alive'][t]:<11}")

    # summarize
    def extinction_step(series: List[int]) -> int:
        for i, v in enumerate(series):
            if v == 0:
                return i
        return len(series) - 1

    s1_ext = extinction_step(curves["sys1_alive"])
    s2_ext = extinction_step(curves["sys2_alive"])

    if curves["sys1_alive"][-1] == 0 and curves["sys2_alive"][-1] == 0:
        print("[!] Both extinct.")
    elif curves["sys1_alive"][-1] == 0 and curves["sys2_alive"][-1] > 0:
        print("[✓] System-2 outlived System-1.")
    else:
        print("[i] No full separation observed under these parameters (still valid probe).")

    out_dir = _results_dir()
    _write_json(os.path.join(out_dir, "config.json"), asdict(cfg))

    rows = [[t, curves["sys1_alive"][t], curves["sys2_alive"][t]] for t in range(len(curves["sys1_alive"]))]
    _write_csv(os.path.join(out_dir, "alive_curves.csv"), ["step", "sys1_alive", "sys2_alive"], rows)

    summary = {
        "sys1_extinction_step": s1_ext,
        "sys2_extinction_step": s2_ext,
        "final_sys1_alive": curves["sys1_alive"][-1],
        "final_sys2_alive": curves["sys2_alive"][-1],
    }
    _write_json(os.path.join(out_dir, "summary.json"), summary)

    print(f"[✓] Saved: {os.path.join(out_dir, 'config.json')}")
    print(f"[✓] Saved: {os.path.join(out_dir, 'alive_curves.csv')}")
    print(f"[✓] Saved: {os.path.join(out_dir, 'summary.json')}")


if __name__ == "__main__":
    main()
