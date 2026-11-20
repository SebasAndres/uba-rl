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

StopFn = Callable[[dict], bool]
StepCallback = Callable[[CrafterEnv, int, dict], None]


class SkillPolicy(Protocol):
    """Interfaz común del agente de habilidades (PPO o scripteado)."""

    name: str

    def reset(self, skill: Skill) -> None:
        """Prepara la política para ejecutar una habilidad nueva.

        Parameters
        ----------
        skill : Skill
            Habilidad que se va a ejecutar.
        """
        ...

    def act(self, env: CrafterEnv, obs: np.ndarray, info: dict, skill: Skill) -> int:
        """Elige la próxima acción primitiva.

        Parameters
        ----------
        env : CrafterEnv
            Entorno (las políticas scripteadas leen su estado interno).
        obs : np.ndarray
            Imagen RGB actual.
        info : dict
            `info` actual de Crafter.
        skill : Skill
            Habilidad en ejecución.

        Returns
        -------
        int
            Índice de la acción primitiva.
        """
        ...


@dataclass
class SkillResult:
    """Resultado de ejecutar una habilidad.

    Attributes
    ----------
    skill : Skill
        Habilidad ejecutada.
    success : bool
        Si se cumplió.
    steps : int
        Pasos primitivos usados.
    obs : np.ndarray
        Última observación.
    info : dict
        Último `info` de Crafter.
    terminated : bool
        El jugador murió.
    truncated : bool
        Se terminó el episodio de Crafter (límite `length`).
    stopped_early : bool
        `stop_fn` cortó la ejecución (ej. misión cumplida o daño).
    actions : list of int
        Acciones primitivas tomadas.
    """

    skill: Skill
    success: bool
    steps: int
    obs: np.ndarray
    info: dict
    terminated: bool
    truncated: bool
    stopped_early: bool = False
    actions: list[int] = field(default_factory=list)


def run_skill(env: CrafterEnv, policy: SkillPolicy, skill: Skill, obs: np.ndarray, info: dict,
              max_steps: int | None = None, stop_fn: StopFn | None = None,
              on_step: StepCallback | None = None) -> SkillResult:
    """Ejecuta una habilidad paso a paso con la política dada.

    Parameters
    ----------
    env : CrafterEnv
        Entorno donde se juega.
    policy : SkillPolicy
        Agente de habilidades (PPO o scripteado).
    skill : Skill
        Habilidad a ejecutar.
    obs : np.ndarray
        Observación actual.
    info : dict
        `info` actual de Crafter.
    max_steps : int or None, optional
        Pasos máximos; por defecto `skill.timeout`.
    stop_fn : callable or None, optional
        Recibe `info` después de cada paso; si devuelve True se corta la ejecución.
    on_step : callable or None, optional
        Callback `(env, action, info)` después de cada paso (ej. render).

    Returns
    -------
    SkillResult
        Resultado de la ejecución.
    """
    max_steps = max_steps or skill.timeout
    prev_achievements = dict(info["achievements"])
    policy.reset(skill)
    actions: list[int] = []
    terminated = truncated = success = stopped = False
    for _ in range(max_steps):
        action = policy.act(env, obs, info, skill)
        obs, _, terminated, truncated, info = env.step(action)
        actions.append(action)
        if on_step is not None:
            on_step(env, action, info)
        success = succeeded(skill, prev_achievements, info["achievements"])
        stopped = stop_fn is not None and stop_fn(info)
        if success or terminated or truncated or stopped:
            break
    return SkillResult(skill, success, len(actions), obs, info, terminated, truncated, stopped, actions)
