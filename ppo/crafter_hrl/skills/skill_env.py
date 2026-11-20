"""Entorno de la FASE 1: Crafter condicionado a una habilidad (gymnasium).

Un episodio de gymnasium = un intento de UNA habilidad (máx. `skill.timeout` pasos).
El mundo de Crafter persiste entre habilidades (salvo que el jugador muera o se
termine el episodio de Crafter), así el inventario progresa naturalmente y no
pagamos la generación del mundo en cada reset.

Observación: {"image": uint8 64x64x3, "goal": float32 GOAL_DIM}
Recompensa (IDEA.md): +1 si se cumple la habilidad, -0.01 por paso.
"""

import gymnasium as gym
import numpy as np

from ..env import CrafterEnv
from ..env.constants import NUM_ACTIONS
from ..env.perception import is_near
from .catalog import GOAL_DIM, SKILLS, Skill, encode, is_feasible, stations_needed, succeeded

# Inventario "de regalo" para practicar habilidades avanzadas (curriculum).
DEFAULT_BOOST = {"wood": (0, 6), "stone": (0, 6), "coal": (0, 3), "iron": (0, 2), "sapling": (0, 2),
                 "wood_pickaxe": (0, 1), "stone_pickaxe": (0, 1), "iron_pickaxe": (0, 1)}


class SkillEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        seed: int | None = None,
        skills: list[Skill] | None = None,
        success_reward: float = 1.0,
        step_penalty: float = 0.01,
        death_penalty: float = 0.0,
        boost_prob: float = 0.3,
        boost: dict | None = None,
        station_radius: int = 8,
        world_length: int = 10000,
    ):
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
        })
        self.action_space = gym.spaces.Discrete(NUM_ACTIONS)
        self._world_alive = False
        self._obs = self._info = None
        self.skill: Skill | None = None
        self._t = 0
        self._prev_ach: dict = {}

    # ------------------------------------------------------------------ gym API
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if not self._world_alive:
            self._obs, self._info = self.crafter.reset()
            self._world_alive = True
        if self.np_random.random() < self.boost_prob:
            self.crafter.set_inventory(**{k: self.np_random.integers(lo, hi + 1)
                                          for k, (lo, hi) in self.boost.items()})
            self._info["inventory"] = dict(self.crafter.player.inventory)
        forced = (options or {}).get("skill")
        self.skill = forced or self._sample_skill()
        self._t = 0
        self._prev_ach = dict(self._info["achievements"])
        return self._observation(), {"skill": str(self.skill)}

    def step(self, action):
        self._obs, _, died, world_over, self._info = self.crafter.step(action)
        self._t += 1
        reward = -self.step_penalty
        success = succeeded(self.skill, self._prev_ach, self._info["achievements"])
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

    def render(self):
        return self.crafter.render()

    # ------------------------------------------------------------------ helpers
    def _observation(self):
        return {"image": self._obs, "goal": encode(self.skill)}

    def _sample_skill(self) -> Skill:
        """Sólo se sortean habilidades posibles con el inventario y el mundo actual."""
        inv, sem, pos = self._info["inventory"], self._info["semantic"], self._info["player_pos"]
        options = [
            s for s in self.skills
            if is_feasible(s, inv)
            and all(is_near(sem, pos, st, self.station_radius) for st in stations_needed(s))
        ]
        # TODO(A): pesos por habilidad (ej. sobremuestrear las que peor andan).
        return options[self.np_random.integers(len(options))]
