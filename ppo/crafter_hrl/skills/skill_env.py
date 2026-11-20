"""Entorno de la FASE 1: Crafter condicionado a una habilidad (gymnasium).

Un episodio de gymnasium = un intento de UNA habilidad (máx. `skill.timeout` pasos).
El mundo de Crafter persiste entre habilidades (salvo que el jugador muera o se
termine el episodio de Crafter), así el inventario progresa naturalmente y no
pagamos la generación del mundo en cada reset.

Observación: {"image": uint8 64x64x3, "goal": float32 GOAL_DIM, "time": float32 (1,)}
("time" = fracción del timeout que queda, ver `catalog.time_left`).
Recompensa (IDEA.md): +1 si se cumple la habilidad, -0.01 por paso.
"""

from typing import Any

import gymnasium as gym
import numpy as np

from ..env import CrafterEnv
from ..env.constants import NUM_ACTIONS
from ..env.perception import is_near
from .catalog import GOAL_DIM, SKILLS, Skill, encode, is_feasible, stations_needed, succeeded, time_left

# Inventario "de regalo" para practicar habilidades avanzadas (curriculum): ítem -> (min, max).
DEFAULT_BOOST = {"wood": (0, 6), "stone": (0, 6), "coal": (0, 3), "iron": (0, 2), "sapling": (0, 2),
                 "wood_pickaxe": (0, 1), "stone_pickaxe": (0, 1), "iron_pickaxe": (0, 1)}


class SkillEnv(gym.Env):
    """Crafter donde cada episodio es un intento de una habilidad sorteada.

    Parameters
    ----------
    seed : int or None, optional
        Semilla del mundo de Crafter.
    skills : list of Skill or None, optional
        Habilidades que se pueden sortear (por defecto todo el catálogo).
    success_reward : float, optional
        Recompensa al cumplir la habilidad.
    step_penalty : float, optional
        Penalización por paso.
    death_penalty : float, optional
        Penalización al morir.
    boost_prob : float, optional
        Probabilidad de regalar inventario en cada reset (curriculum).
    boost : dict or None, optional
        Rango (min, max) a regalar por ítem; por defecto `DEFAULT_BOOST`.
    station_radius : int, optional
        Para sortear "fabricar", la mesa/horno tiene que estar a este radio.
    world_length : int, optional
        Pasos máximos de un mundo de Crafter.
    """

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, seed: int | None = None, skills: list[Skill] | None = None,
                 success_reward: float = 1.0, step_penalty: float = 0.01, death_penalty: float = 0.0,
                 boost_prob: float = 0.3, boost: dict | None = None, station_radius: int = 8,
                 world_length: int = 10000) -> None:
        super().__init__()
        self.crafter = CrafterEnv(seed=seed, length=world_length)
        self.skills = skills or SKILLS
        self.success_reward = success_reward
        self.step_penalty = step_penalty
        self.death_penalty = death_penalty
        self.boost_prob = boost_prob
        self.boost = boost or DEFAULT_BOOST
        self.station_radius = station_radius
        self.observation_space = gym.spaces.Dict({
            "image": gym.spaces.Box(0, 255, (64, 64, 3), np.uint8),
            "goal": gym.spaces.Box(0.0, 1.0, (GOAL_DIM,), np.float32),
            "time": gym.spaces.Box(0.0, 1.0, (1,), np.float32),
        })
        self.action_space = gym.spaces.Discrete(NUM_ACTIONS)
        self._world_alive = False
        self._obs = self._info = None
        self.skill: Skill | None = None
        self._t = 0
        self._prev_achievements: dict = {}

    # ------------------------------------------------------------------ gym API
    def reset(self, *, seed: int | None = None,
              options: dict[str, Any] | None = None) -> tuple[dict[str, np.ndarray], dict]:
        """Empieza un intento de habilidad (genera un mundo sólo si el anterior terminó).

        Parameters
        ----------
        seed : int or None, optional
            Semilla del `np_random` de gymnasium.
        options : dict or None, optional
            ``{"skill": Skill}`` fuerza la habilidad en vez de sortearla.

        Returns
        -------
        observation : dict
            {"image": imagen RGB, "goal": goal_vec, "time": fracción de timeout restante}.
        info : dict
            {"skill": nombre de la habilidad}.
        """
        super().reset(seed=seed)
        if not self._world_alive:
            self._obs, self._info = self.crafter.reset()
            self._world_alive = True
        if self.np_random.random() < self.boost_prob:
            self._boost_inventory()
        forced_skill = (options or {}).get("skill")
        self.skill = forced_skill or self._sample_skill()
        self._t = 0
        self._prev_achievements = dict(self._info["achievements"])
        return self._observation(), {"skill": str(self.skill)}

    def step(self, action: int) -> tuple[dict[str, np.ndarray], float, bool, bool, dict]:
        """Ejecuta una acción primitiva.

        Parameters
        ----------
        action : int
            Índice de la acción.

        Returns
        -------
        observation : dict
            {"image": imagen RGB, "goal": goal_vec, "time": fracción de timeout restante}.
        reward : float
            Recompensa de la fase 1.
        terminated : bool
            True si se cumplió la habilidad o el jugador murió.
        truncated : bool
            True si se agotó el timeout de la habilidad o terminó el mundo.
        info : dict
            {"skill", "success", "died"}.
        """
        self._obs, _, died, world_over, self._info = self.crafter.step(action)
        self._t += 1
        success = succeeded(self.skill, self._prev_achievements, self._info["achievements"])
        reward = -self.step_penalty
        if success:
            reward += self.success_reward
        if died:
            reward -= self.death_penalty
        if died or world_over:
            self._world_alive = False
        terminated = success or died
        truncated = (not terminated) and (world_over or self._t >= self.skill.timeout)
        info = {"skill": str(self.skill), "success": success, "died": died}
        return self._observation(), reward, terminated, truncated, info

    def render(self) -> np.ndarray:
        """Frame RGB del estado actual.

        Returns
        -------
        np.ndarray
            Imagen RGB.
        """
        return self.crafter.render()

    # ------------------------------------------------------------------ helpers
    def _observation(self) -> dict[str, np.ndarray]:
        """Arma la observación del agente.

        Returns
        -------
        dict
            {"image": imagen RGB, "goal": goal_vec de la habilidad actual,
            "time": fracción de su timeout que queda}.
        """
        return {"image": self._obs, "goal": encode(self.skill), "time": time_left(self.skill, self._t)}

    def _boost_inventory(self) -> None:
        """Regala un inventario aleatorio (dentro de `self.boost`) para el curriculum."""
        amounts = {item: self.np_random.integers(low, high + 1) for item, (low, high) in self.boost.items()}
        self.crafter.set_inventory(**amounts)
        self._info["inventory"] = dict(self.crafter.player.inventory)

    def _sample_skill(self) -> Skill:
        """Sortea una habilidad posible con el inventario y el mundo actual.

        Returns
        -------
        Skill
            Habilidad factible y, si fabrica, con las estaciones cerca.
        """
        inv, sem, pos = self._info["inventory"], self._info["semantic"], self._info["player_pos"]
        options = [
            s for s in self.skills
            if is_feasible(s, inv)
            and all(is_near(sem, pos, st, self.station_radius) for st in stations_needed(s))
        ]
        # TODO(A): pesos por habilidad (ej. sobremuestrear las que peor andan).
        return options[self.np_random.integers(len(options))]
