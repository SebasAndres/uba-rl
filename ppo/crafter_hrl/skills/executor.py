"""Ejecuta UNA habilidad con acciones primitivas hasta cumplirla o timeout.

Es el "nivel bajo" del loop jerárquico: lo usan PlannerEnv (fase 2, juego,
evaluación). El agente de habilidades recibe el mismo goal_vec durante toda la
ejecución.
"""

from dataclasses import dataclass, field
from typing import Callable, Protocol

import numpy as np

from ..env import CrafterEnv
from .catalog import Skill, succeeded


class SkillPolicy(Protocol):
    name: str

    def reset(self, skill: Skill) -> None: ...
    def act(self, env: CrafterEnv, obs: np.ndarray, info: dict, skill: Skill) -> int: ...


@dataclass
class SkillResult:
    skill: Skill
    success: bool
    steps: int
    obs: np.ndarray
    info: dict
    terminated: bool             # murió
    truncated: bool              # se terminó el episodio de Crafter (length)
    stopped_early: bool = False  # `stop_fn` cortó (ej. la misión se cumplió a mitad de camino)
    actions: list[int] = field(default_factory=list)


def run_skill(
    env: CrafterEnv,
    policy: SkillPolicy,
    skill: Skill,
    obs: np.ndarray,
    info: dict,
    max_steps: int | None = None,
    stop_fn: Callable[[dict], bool] | None = None,
    on_step: Callable[[CrafterEnv, int, dict], None] | None = None,
) -> SkillResult:
    max_steps = max_steps or skill.timeout
    prev_ach = dict(info["achievements"])
    policy.reset(skill)
    actions: list[int] = []
    terminated = truncated = success = stopped = False
    for _ in range(max_steps):
        action = policy.act(env, obs, info, skill)
        obs, _, terminated, truncated, info = env.step(action)
        actions.append(action)
        if on_step is not None:
            on_step(env, action, info)
        success = succeeded(skill, prev_ach, info["achievements"])
        stopped = stop_fn is not None and stop_fn(info)
        if success or terminated or truncated or stopped:
            break
    return SkillResult(skill, success, len(actions), obs, info, terminated, truncated, stopped, actions)
