"""
Phoenix / IFD Deep Research Package
Step 4 — Delayed Causal Learning (Eligibility Traces)  [SUPPLEMENT]

------------------------------------------------------------
PHYSICS INTENT (Concept Anchor)
------------------------------------------------------------
Concept:       Law 4 follow-up — "Learning under delay" is not a free lunch.
Physical Var:  Feedback Delay / Blindness (observation is delayed by K).
               NOT Actuation Delay (action effects are immediate).
Goal:          Show that even if an agent *tries* to learn the gain/feedback link
              via delayed credit assignment (eligibility traces),
              collapse can still happen in the high-delay regime.

Expected Phenomenon:
- Learning signal appears (knowledge rises) but may arrive too late.
- Under delay beyond phase margin, naive causal learning collapses before it stabilizes.

Important:
- This is a mechanism probe, not the core Law-4 proof (Step 2).
- We add an initial buffer to remove startup "birth-death" artifacts.
------------------------------------------------------------
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
# A. Frozen Defaults
# -----------------------------
DEFAULT_SEED = 2026
DEFAULT_POP = 100
DEFAULT_STEPS = 600

DEFAULT_DELAY = 20
DEFAULT_GAIN = 4.0

DEFAULT_INITIAL_BUFFER = 25.0  # <-- accepted change
DEFAULT_METABOLIC_COST = 0.05
DEFAULT_FORAGE_COST = 0.10

# For the learner, "knowledge" is a scalar in [0, 1] tracking confidence in gain estimate.
# We keep this deliberately simple for auditability.
DEFAULT_LR = 0.15
DEFAULT_TRACE_DECAY = 0.90  # eligibility trace decay per step


# -----------------------------
# B. Paths & IO
# -----------------------------
def _script_dir() -> str:
    return os.path.dirname(os.path.abspath(__file__))


def _results_dir() -> str:
    base = os.path.normpath(os.path.join(_script_dir(), "..","..", "results", "step4_learning"))
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
# C. Config
# -----------------------------
@dataclass(frozen=True)
class Step4Config:
    seed: int
    pop: int
    steps: int
    delay: int
    gain: float
    initial_buffer: float
    metabolic_cost: float
    forage_cost: float
    lr: float
    trace_decay: float


# -----------------------------
# D. Learner agent
# -----------------------------
class LearnerAgent:
    """
    A minimal learner that tries to estimate whether FORAGE is "worth it"
    using delayed credit assignment.
    """
    __slots__ = ("H", "elig_trace", "knowledge")

    def __init__(self, initial_buffer: float, rng: random.Random):
        self.H = initial_buffer + rng.uniform(-1.0, 1.0)

        # eligibility trace is a rolling buffer of past actions (FORAGE=1, REST=0)
        self.elig_trace = [0.0] * 1  # will be resized per delay at runtime

        # knowledge in [0,1]
        self.knowledge = 0.0

    def ensure_delay(self, delay: int) -> None:
        if len(self.elig_trace) != delay:
            self.elig_trace = [0.0] * delay

    def policy(self, rng: random.Random) -> str:
        """
        Very simple policy:
        - As knowledge rises, it forages more aggressively.
        """
        p_forage = 0.20 + 0.75 * self.knowledge  # in [0.20, 0.95]
        return "FORAGE" if rng.random() < p_forage else "REST"

    def update_trace(self, action: str, cfg: Step4Config) -> None:
        # shift left
        for i in range(cfg.delay - 1):
            self.elig_trace[i] = cfg.trace_decay * self.elig_trace[i + 1]
        self.elig_trace[-1] = 1.0 if action == "FORAGE" else 0.0

    def learn_from_delayed_reward(self, delayed_reward: float, cfg: Step4Config) -> None:
        """
        Update knowledge using the oldest trace component as the credit proxy.
        """
        credit = self.elig_trace[0]  # action that occurred delay steps ago
        # map reward to a [0,1] "signal" with a soft clamp
        signal = max(0.0, min(1.0, delayed_reward / max(1e-6, cfg.gain)))
        self.knowledge += cfg.lr * credit * (signal - self.knowledge)
        self.knowledge = max(0.0, min(1.0, self.knowledge))


# -----------------------------
# E. Physics & simulation
# -----------------------------
def step_physics(H: float, action: str, cfg: Step4Config) -> Tuple[float, float]:
    """
    Returns (new_H, reward).
    reward is the immediate energy gained from FORAGE (gain) minus costs,
    but learning will see it only after delay.
    """
    if action == "FORAGE":
        reward = cfg.gain - (cfg.metabolic_cost + cfg.forage_cost)
        H += reward
    else:
        reward = -cfg.metabolic_cost
        H += reward
    return H, reward


def run(cfg: Step4Config) -> Dict[str, List[float]]:
    rng = random.Random(cfg.seed)
    agents: List[LearnerAgent] = [LearnerAgent(cfg.initial_buffer, rng) for _ in range(cfg.pop)]
    for a in agents:
        a.ensure_delay(cfg.delay)

    # delayed reward pipe per agent
    reward_pipe: List[List[float]] = [[0.0] * cfg.delay for _ in range(cfg.pop)]

    alive_curve: List[int] = []
    avg_knowledge_curve: List[float] = []

    for t in range(cfg.steps):
        alive = 0
        know_sum = 0.0

        for i, a in enumerate(agents):
            if a.H <= 0.0:
                continue
            alive += 1
            know_sum += a.knowledge

            action = a.policy(rng)
            a.update_trace(action, cfg)

            a.H, reward = step_physics(a.H, action, cfg)

            # deliver delayed reward to learner
            delayed = reward_pipe[i][0]
            a.learn_from_delayed_reward(delayed, cfg)

            # advance reward pipe
            reward_pipe[i].append(reward)
            if len(reward_pipe[i]) > cfg.delay:
                reward_pipe[i].pop(0)

        alive_curve.append(alive)
        avg_knowledge_curve.append((know_sum / alive) if alive > 0 else 0.0)

        if alive == 0:
            break

    return {"alive": alive_curve, "avg_knowledge": avg_knowledge_curve}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--pop", type=int, default=DEFAULT_POP)
    ap.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    ap.add_argument("--delay", type=int, default=DEFAULT_DELAY)
    ap.add_argument("--gain", type=float, default=DEFAULT_GAIN)
    ap.add_argument("--buffer", type=float, default=DEFAULT_INITIAL_BUFFER)
    args = ap.parse_args()

    cfg = Step4Config(
        seed=args.seed,
        pop=args.pop,
        steps=args.steps,
        delay=args.delay,
        gain=args.gain,
        initial_buffer=args.buffer,
        metabolic_cost=DEFAULT_METABOLIC_COST,
        forage_cost=DEFAULT_FORAGE_COST,
        lr=DEFAULT_LR,
        trace_decay=DEFAULT_TRACE_DECAY,
    )

    print(f"[*] PHOENIX STEP 4 (SUPPLEMENT) | Eligibility Traces | Delay={cfg.delay} | Gain={cfg.gain}")
    print("STEP   | ALIVE  | AVG GAIN KNOWLEDGE   | STATUS")
    print("-----------------------------------------------------------------")

    out = run(cfg)

    # sparse log
    for t in range(0, len(out["alive"]), 5):
        alive = out["alive"][t]
        k = out["avg_knowledge"][t]
        status = "IGNORANT" if k < 0.2 else ("LEARNING" if k < 0.8 else "CONFIDENT")
        print(f"{t:<6} | {alive:<6} | {k:<20.3f} | {status}")
        if alive == 0:
            break

    if out["alive"][-1] == 0:
        print("[!] EXTINCTION.")
    else:
        print("[✓] Survived full horizon (under this configuration).")

    out_dir = _results_dir()
    _write_json(os.path.join(out_dir, "config.json"), asdict(cfg))

    rows = [[t, out["alive"][t], out["avg_knowledge"][t]] for t in range(len(out["alive"]))]
    _write_csv(os.path.join(out_dir, "learning_curve.csv"), ["step", "alive", "avg_knowledge"], rows)

    summary = {
        "final_alive": out["alive"][-1],
        "final_avg_knowledge": out["avg_knowledge"][-1],
        "extinction_step": next((i for i, v in enumerate(out["alive"]) if v == 0), None),
    }
    _write_json(os.path.join(out_dir, "summary.json"), summary)

    print(f"[✓] Saved: {os.path.join(out_dir, 'config.json')}")
    print(f"[✓] Saved: {os.path.join(out_dir, 'learning_curve.csv')}")
    print(f"[✓] Saved: {os.path.join(out_dir, 'summary.json')}")


if __name__ == "__main__":
    main()
