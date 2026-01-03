# src/ifd/config/specs.py
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, Literal, Optional


RefluxMode = Literal["group", "victim", "env"]


@dataclass(frozen=True)
class PhysicsConfig:
    """
    Universe constants / shared physics.
    Must NOT contain per-agent thresholds.
    """

    # --- Environment energy gain (baseline)
    env_gain_mu: float = 0.037  # from law3_world.WorldConfig

    # --- Metabolic tax (always applies)
    metabolic_cost: float = 0.025

    # --- Environmental noise (sigma on env gain)
    sigma_env: float = 0.0

    # --- Structural dynamics
    dist_inc_act: float = 0.02       # fatigue accumulation when ACT
    dist_decay_idle: float = 0.01    # recovery when IDLE

    # --- Predation / bite physics
    bite_success_prob: float = 0.20
    bite_damage_to_prey: float = 0.12
    bite_gain_eff: float = 0.80
    bite_extra_dist_cost: float = 0.015
    bite_energy_cost: float = 0.01

    # --- Structural reflux (Law 3.0)
    reflux_rate: float = 0.0
    reflux_mode: RefluxMode = "group"

    # --- Constraints
    max_prey_cap: int = 300

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AgentConfig:
    """
    Per-agent control parameters (System-1 state machine thresholds).
    """

    thr_lo: float = 0.30
    thr_hi: float = 0.60

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ExperimentConfig:
    """
    Run-time experiment settings (population, steps, seed).
    """

    n_prey: int = 50
    n_predators: int = 1
    max_steps: int = 2000
    seed: int = 0

    # Optional overrides (for backward compatibility during refactor).
    # Use this ONLY in migration stage; later remove it.
    cfg_override: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class System1Config:
    """
    Bundle config for a single simulation.
    This is the object you persist to JSON for reproducibility.
    """

    physics: PhysicsConfig
    agent: AgentConfig
    experiment: ExperimentConfig

    def to_dict(self) -> Dict[str, Any]:
        return {
            "physics": self.physics.to_dict(),
            "agent": self.agent.to_dict(),
            "experiment": self.experiment.to_dict(),
        }