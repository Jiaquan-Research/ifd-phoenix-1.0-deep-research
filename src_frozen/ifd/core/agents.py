# src/ifd/core/agents.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol, Literal

from ifd.config.specs import AgentConfig


Action = Literal["ACT", "IDLE"]


class ISystem1Agent(Protocol):
    uid: int
    energy: float
    structural_integrity: float  # a.k.a. dist
    state: Action

    def decide(self, cfg: AgentConfig) -> Action: ...
    def metabolize(self, metabolic_cost: float) -> None: ...
    def apply_structural_act_cost(self, dist_inc_act: float) -> None: ...
    def apply_structural_idle_recovery(self, dist_decay_idle: float) -> None: ...
    def is_dead(self) -> bool: ...


@dataclass
class StandardAgent:
    """
    System-1 Agent (Frozen Baseline).

    Properties:
    - No world model
    - No prediction
    - No learning
    - No memory beyond current state (Markovian)

    This class defines the maximal capability of System-1.
    Any extension belongs to System-2 and MUST NOT modify this class.

    Legacy semantics:
    - decision logic mirrors the specific hysteresis of the legacy codebase.
    """

    uid: int
    energy: float = 1.0
    structural_integrity: float = 0.0  # legacy: dist
    state: Action = "ACT"

    # Optional bookkeeping
    last_victim_uid: Optional[int] = None

    def decide(self, cfg: AgentConfig) -> Action:
        """
        Legacy equivalence:
          - If last action ACT, keep acting until dist > thr_hi
          - If last action IDLE, keep resting until dist < thr_lo
        """
        dist = self.structural_integrity
        if self.state == "ACT":
            self.state = "IDLE" if dist > cfg.thr_hi else "ACT"
        else:
            self.state = "ACT" if dist < cfg.thr_lo else "IDLE"
        return self.state

    def metabolize(self, metabolic_cost: float) -> None:
        self.energy -= metabolic_cost

    def apply_structural_act_cost(self, dist_inc_act: float) -> None:
        if self.state == "ACT":
            self.structural_integrity += dist_inc_act

    def apply_structural_idle_recovery(self, dist_decay_idle: float) -> None:
        if self.state == "IDLE":
            self.structural_integrity = max(0.0, self.structural_integrity - dist_decay_idle)

    def is_dead(self) -> bool:
        return self.energy <= 0.0