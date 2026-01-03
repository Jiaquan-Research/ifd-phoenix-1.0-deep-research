# src/ifd/core/world.py
"""
IFD System-1 World
Law 3.0 Frozen Baseline (T=0)

This implementation is the first structurally auditable,
deterministic, and freeze-worthy realization of Law 3.

All future Law 3.x / Law 4 work MUST layer on top of this file.
DO NOT modify without explicit version bump and audit.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import random

from ifd.config.specs import System1Config
from ifd.core.agents import StandardAgent


@dataclass(frozen=True)
class StepMetrics:
    t: int
    alive: int
    total_env_energy: float
    total_pred_energy: float


@dataclass(frozen=True)
class WorldResult:
    extinction_step: int
    energy_from_environment: float
    energy_from_predation: float
    max_steps: int


class World:
    """
    System-1 simulation engine (Law 3.0 baseline).

    Critical invariants for bit-exact migration:
      - Decision happens BEFORE env gain.
      - Metabolic cost applies every step regardless of action.
      - ACT: env gain (mu + gauss), clamp >= 0, then dist += dist_inc_act
      - IDLE: dist = max(0, dist - dist_decay_idle)
      - Predation: rng.random() then rng.randrange
      - Reflux: only for dist < thr_hi, and modes group/victim/env
      - Cleanup: alive iff (energy > 0) and (dist < 1.0)
    """

    def __init__(self, cfg: System1Config):
        self.cfg = cfg

        # NOTE: World must own the only RNG to ensure causal reproducibility.
        # Agents must never introduce their own randomness source.
        self.rng = random.Random(cfg.experiment.seed)

        # --- agents ---
        self.prey: List[StandardAgent] = [
            StandardAgent(uid=i, energy=1.0, structural_integrity=0.0, state="ACT")
            for i in range(cfg.experiment.n_prey)
        ]

        # --- bookkeeping ---
        self.t: int = 0
        self.total_env_energy: float = 0.0
        self.total_pred_energy: float = 0.0

        self._stats: List[StepMetrics] = []

    @property
    def stats(self) -> List[StepMetrics]:
        return self._stats

    def step(self) -> StepMetrics:
        """
        Execute one full simulation tick.

        Order MUST match legacy run_law3_world:
          1) Prey dynamics: Decide -> Metabolism -> Action consequence
          2) Predation
          3) Structural reflux
          4) Cleanup / extinction check
        """
        p = self.cfg.physics
        a = self.cfg.agent
        e = self.cfg.experiment

        # Track bitten victims for "victim" reflux mode
        bitten_indices: List[int] = []

        # =====================
        # 1. Prey Dynamics (Metabolism & Behavior)
        # =====================
        for i in range(len(self.prey)):
            agent = self.prey[i]

            # A. Decision Phase (updates agent.state immediately)
            action = agent.decide(a)

            # B. Metabolic Cost (always applied)
            agent.energy -= p.metabolic_cost

            # C. Action Consequence
            if action == "ACT":
                gain = p.env_gain_mu + self.rng.gauss(0.0, p.sigma_env)
                gain = max(gain, 0.0)  # Prevent negative gain
                agent.energy += gain
                agent.structural_integrity += p.dist_inc_act
                self.total_env_energy += gain
            else:  # IDLE
                agent.structural_integrity = max(
                    0.0, agent.structural_integrity - p.dist_decay_idle
                )

        # =====================
        # 2. Predation Dynamics
        # =====================
        if e.n_predators > 0 and len(self.prey) > 0:
            for _ in range(e.n_predators):
                if self.rng.random() < p.bite_success_prob and len(self.prey) > 0:
                    j = self.rng.randrange(len(self.prey))
                    bitten_indices.append(j)

                    victim = self.prey[j]
                    victim.energy -= p.bite_damage_to_prey
                    victim.structural_integrity += p.bite_extra_dist_cost

                    gained = p.bite_damage_to_prey * p.bite_gain_eff
                    self.total_pred_energy += gained

        # =====================
        # 3. Law 3: Structural Reflux
        # =====================
        if p.reflux_rate > 0.0 and len(self.prey) > 0:
            self._apply_reflux(bitten_indices)

        # =====================
        # 4. Extinction Check & Cleanup
        # =====================
        alive: List[StandardAgent] = []
        for agent in self.prey:
            # legacy condition: energy > 0 and dist < 1.0
            if agent.energy > 0.0 and agent.structural_integrity < 1.0:
                alive.append(agent)
        self.prey = alive

        m = StepMetrics(
            t=self.t,
            alive=len(self.prey),
            total_env_energy=self.total_env_energy,
            total_pred_energy=self.total_pred_energy,
        )
        self._stats.append(m)
        self.t += 1
        return m

    def _apply_reflux(self, bitten_indices: List[int]) -> None:
        """
        Structural reflux (Law 3.0 family).

        Definition:
        Reflux is a purely structural relaxation mechanism (energy/entropy redistribution).
        It MUST NOT depend on agent intention, history, or prediction.

        Modes:
          - group: everyone in reversible zone (dist < thr_hi) gets multiplicative relief
          - victim: only bitten victims get relief
          - env: convert a portion of dist into shared energy pool
        """
        p = self.cfg.physics
        a = self.cfg.agent

        mode = p.reflux_mode

        if mode == "group":
            for agent in self.prey:
                if agent.structural_integrity < a.thr_hi:
                    agent.structural_integrity *= (1.0 - p.reflux_rate)

        elif mode == "victim":
            # Only bitten victims get relief
            for j in set(bitten_indices):
                if 0 <= j < len(self.prey):
                    agent = self.prey[j]
                    if agent.structural_integrity < a.thr_hi:
                        agent.structural_integrity *= (1.0 - p.reflux_rate)

        elif mode == "env":
            # Dist relief converted into energy pool
            recovered = 0.0
            for agent in self.prey:
                if agent.structural_integrity < a.thr_hi:
                    delta = agent.structural_integrity * p.reflux_rate
                    agent.structural_integrity -= delta
                    recovered += delta

            if recovered > 0.0 and len(self.prey) > 0:
                per = recovered / len(self.prey)
                for agent in self.prey:
                    agent.energy += per

        else:
            # safe fail
            return

    def run(self) -> WorldResult:
        """
        Run until max_steps or extinction.
        Mirrors legacy return semantics.
        """
        max_steps = self.cfg.experiment.max_steps
        for _ in range(max_steps):
            m = self.step()
            if m.alive == 0:
                return WorldResult(
                    extinction_step=m.t,
                    energy_from_environment=self.total_env_energy,
                    energy_from_predation=self.total_pred_energy,
                    max_steps=max_steps,
                )

        return WorldResult(
            extinction_step=max_steps,
            energy_from_environment=self.total_env_energy,
            energy_from_predation=self.total_pred_energy,
            max_steps=max_steps,
        )