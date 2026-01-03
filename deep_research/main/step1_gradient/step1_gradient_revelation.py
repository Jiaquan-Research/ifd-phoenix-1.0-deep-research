"""
=========================================================
PHYSICS INTENT (Concept Anchor)
=========================================================
Concept:
    Step 1 — Gradient Revelation

Purpose:
    Demonstrate that an optimal action gradient can emerge
    purely from survival selection, without learning,
    prediction, or delayed feedback.

Key Constraints:
    - No learning
    - No System-2
    - No delay
    - No memory beyond current state
    - Deterministic replay (fixed seed)

Interpretation:
    The gradient is NOT discovered by the agent.
    It is revealed by the environment through differential survival.

Evidence Role:
    Main Text — Existence proof of environmental gradient geometry.
=========================================================
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np

from ifd.core.world import World
from ifd.config.specs import PhysicsConfig, AgentConfig, ExperimentConfig, System1Config


# ---------------------------------------------------------
# Path Utilities
# ---------------------------------------------------------
def package_root() -> Path:
    """
    Resolve repository root:
    IFD_Phoenix_1_0_Package/
    """
    return Path(__file__).resolve().parents[3]


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------
# Experiment Setup
# ---------------------------------------------------------
SEED = 2026
STEPS = 1000
POPULATION = 200


def build_config() -> System1Config:
    physics = PhysicsConfig(
        env_gain_mu=0.037,
        metabolic_cost=0.025,
        sigma_env=0.0,
        reflux_rate=0.0,
    )

    agent = AgentConfig(
        thr_lo=0.30,
        thr_hi=0.60,
    )

    experiment = ExperimentConfig(
        n_prey=POPULATION,
        n_predators=1,
        max_steps=STEPS,
        seed=SEED,
    )

    return System1Config(
        physics=physics,
        agent=agent,
        experiment=experiment,
    )


# ---------------------------------------------------------
# Main Experiment
# ---------------------------------------------------------
def main() -> None:
    cfg = build_config()
    world = World(cfg)

    print(f"[*] STEP 1 — Gradient Revelation")
    print(f"[*] Seed={SEED} | Population={POPULATION} | Steps={STEPS}")

    for t in range(STEPS):
        world.step()

        if t % 100 == 0:
            alive = world.n_alive_prey
            avg_margin = np.mean(
                [p.energy - p.dist for p in world.prey if p.alive]
            ) if alive > 0 else 0.0
            print(f"Step {t:04d} | Alive: {alive:3d} | Avg Margin: {avg_margin:.2f}")

        if world.n_alive_prey == 0:
            print("[!] All prey extinct.")
            break

    # -----------------------------------------------------
    # Collect Results
    # -----------------------------------------------------
    genomes: List[np.ndarray] = [
        p.genome for p in world.prey if p.alive
    ]

    global_avg_genome = (
        np.mean(genomes, axis=0).tolist() if genomes else []
    )

    summary: Dict[str, object] = {
        "alive_prey": world.n_alive_prey,
        "global_avg_genome": global_avg_genome,
        "note": (
            "Global average genome approximates the optimal mixed strategy "
            "revealed by survival selection under current physics."
        ),
    }

    # -----------------------------------------------------
    # Output (Deep Research Scoped)
    # -----------------------------------------------------
    out_dir = (
        package_root()
        / "deep_research"
        / "main"
        / "step1_gradient"
        / "results"
    )
    ensure_dir(out_dir)

    with open(out_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(cfg.to_dict(), f, indent=2)

    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    population_csv = out_dir / "population.csv"
    with open(population_csv, "w", encoding="utf-8") as f:
        f.write("genome_0,genome_1,genome_2,genome_3\n")
        for g in genomes:
            f.write(",".join(f"{x:.6f}" for x in g) + "\n")

    print(f"[✓] Saved config.json")
    print(f"[✓] Saved summary.json")
    print(f"[✓] Saved population.csv")


# ---------------------------------------------------------
# Entry Point
# ---------------------------------------------------------
if __name__ == "__main__":
    main()
