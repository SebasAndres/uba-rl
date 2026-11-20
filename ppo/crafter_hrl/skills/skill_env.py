"""Entorno de la FASE 1: Crafter condicionado a una habilidad (gymnasium).

Un episodio de gymnasium = un intento de UNA habilidad (máx. `skill.timeout` pasos).
El mundo de Crafter persiste entre habilidades (salvo que el jugador muera o se
termine el episodio de Crafter), así el inventario progresa naturalmente y no
pagamos la generación del mundo en cada reset.

Observación: {"image": uint8 64x64x3, "goal": float32 GOAL_DIM, "time": float32 (1,)}
("time" = fracción del timeout que queda, ver `catalog.time_left`).
Recompensa (IDEA.md): +1 si se cumple la habilidad, -0.01 por paso, más el
shaping por distancia BFS al objetivo (`shaping.py`, basado en potencial).

Curriculum (`curriculum.py`): las habilidades se desbloquean por etapas y se
sortean más las que peor andan. Con prob. `prereq_prob` se regalan los
prerequisitos de la habilidad sorteada (inventario y mesa/horno cerca).
"""

from typing import Any

import gymnasium as gym
import numpy as np

from ..env import CrafterEnv
from ..env.constants import MAT_ID, NUM_ACTIONS
from ..env.perception import is_near
from .catalog import (GOAL_DIM, SKILLS, Skill, encode, is_feasible, missing_requirements, stations_needed,
                      succeeded, time_left)
from .curriculum import Curriculum
from .shaping import DEFAULT_MAX_RADIUS, potential, shaping_reward

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
    shaping_coef : float, optional
        Escala del shaping por distancia BFS (0 lo desactiva).
    gamma : float, optional
        Descuento del shaping (tiene que coincidir con el de PPO).
    use_curriculum : bool, optional
        Si es True, sortea con `Curriculum` (etapas + pesos por dificultad).
    start_stage : int, optional
        Etapa inicial del curriculum.
    advance_threshold : float, optional
        Éxito medio necesario para pasar de etapa.
    prereq_prob : float, optional
        Prob. de regalar los prerequisitos de la habilidad sorteada.
    station_place_radius : int, optional
        Radio (Chebyshev) donde se pone la mesa/horno regalada.
    """

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, seed: int | None = None, skills: list[Skill] | None = None,
                 success_reward: float = 1.0, step_penalty: float = 0.01, death_penalty: float = 0.0,
                 boost_prob: float = 0.3, boost: dict | None = None, station_radius: int = 8,
                 world_length: int = 10000, shaping_coef: float = 0.1, gamma: float = 0.99,
                 use_curriculum: bool = True, start_stage: int = 0, advance_threshold: float = 0.6,
                 prereq_prob: float = 0.5, station_place_radius: int = 4) -> None:
        super().__init__()
        self.crafter = CrafterEnv(seed=seed, length=world_length)
        self.skills = skills or SKILLS
        self.success_reward = success_reward
        self.step_penalty = step_penalty
        self.death_penalty = death_penalty
        self.boost_prob = boost_prob
        self.boost = boost or DEFAULT_BOOST
        self.station_radius = station_radius
        self.shaping_coef = shaping_coef
        self.gamma = gamma
        self.prereq_prob = prereq_prob
        self.station_place_radius = station_place_radius
        self.curriculum = (Curriculum(self.skills, start_stage, advance_threshold)
                           if use_curriculum else None)
        self._phi = 0.0
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
        self._phi = potential(self.skill, self._info, DEFAULT_MAX_RADIUS) if self.shaping_coef else 0.0
        return self._observation(), {"skill": str(self.skill), "stage": self.stage}

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
            {"skill", "success", "died", "shaping", "stage"}.
        """
        self._obs, _, died, world_over, self._info = self.crafter.step(action)
        self._t += 1
        success = succeeded(self.skill, self._prev_achievements, self._info["achievements"])
        if died or world_over:
            self._world_alive = False
        terminated = success or died
        truncated = (not terminated) and (world_over or self._t >= self.skill.timeout)
        shaping = self._shaping(terminated or truncated)
        reward = -self.step_penalty + shaping + self.success_reward * success - self.death_penalty * died
        if (terminated or truncated) and self.curriculum is not None:
            self.curriculum.update(self.skill, success)
        info = {"skill": str(self.skill), "success": success, "died": died,
                "shaping": shaping, "stage": self.stage}
        return self._observation(), reward, terminated, truncated, info

    @property
    def stage(self) -> int:
        """int: etapa actual del curriculum (-1 si no se usa)."""
        return self.curriculum.stage if self.curriculum is not None else -1

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

    def _shaping(self, done: bool) -> float:
        """Calcula el shaping del paso y actualiza el potencial guardado.

        Parameters
        ----------
        done : bool
            Si el intento de la habilidad terminó en este paso.

        Returns
        -------
        float
            coef * (gamma * Phi(s') - Phi(s)), con Phi(terminal) = 0.
        """
        if not self.shaping_coef:
            return 0.0
        phi_next = 0.0 if done else potential(self.skill, self._info, DEFAULT_MAX_RADIUS)
        reward = shaping_reward(self._phi, phi_next, self.gamma, self.shaping_coef, done)
        self._phi = phi_next
        return reward

    def _boost_inventory(self) -> None:
        """Regala un inventario aleatorio (dentro de `self.boost`) para el curriculum."""
        amounts = {item: self.np_random.integers(low, high + 1) for item, (low, high) in self.boost.items()}
        self.crafter.set_inventory(**amounts)
        self._obs, self._info = self.crafter.refresh()

    def _sample_skill(self) -> Skill:
        """Sortea una habilidad: con curriculum, de lo desbloqueado y ponderada por dificultad.

        Con prob. `prereq_prob` se regalan los prerequisitos de la sorteada; si
        aun así no está lista, se sortea entre las que sí lo están.

        Returns
        -------
        Skill
            Habilidad factible y, si fabrica, con las estaciones cerca.
        """
        candidates = self.curriculum.unlocked() if self.curriculum is not None else self.skills
        if self.np_random.random() < self.prereq_prob:
            skill = self._choose(candidates)
            self._grant_prerequisites(skill)
            if self._is_ready(skill):
                return skill
        ready = ([s for s in candidates if self._is_ready(s)]
                 or [s for s in self.skills if self._is_ready(s)] or [SKILLS[0]])
        return self._choose(ready)

    def _choose(self, skills: list[Skill]) -> Skill:
        """Sortea entre `skills` (con los pesos del curriculum si hay).

        Parameters
        ----------
        skills : list of Skill
            Candidatas (no vacía).

        Returns
        -------
        Skill
            Habilidad sorteada.
        """
        if self.curriculum is not None:
            return self.curriculum.sample(skills, self.np_random)
        return skills[self.np_random.integers(len(skills))]

    def _is_ready(self, skill: Skill) -> bool:
        """Indica si la habilidad es factible ahora (inventario y estaciones cerca).

        Parameters
        ----------
        skill : Skill
            Habilidad a chequear.

        Returns
        -------
        bool
            True si se puede intentar.
        """
        sem, pos = self._info["semantic"], self._info["player_pos"]
        return is_feasible(skill, self._info["inventory"]) and all(
            is_near(sem, pos, st, self.station_radius) for st in stations_needed(skill))

    def _grant_prerequisites(self, skill: Skill) -> None:
        """Regala lo que le falta a la habilidad: ítems, energía baja (dormir) y mesa/horno cerca.

        Parameters
        ----------
        skill : Skill
            Habilidad sorteada.
        """
        inv = self._info["inventory"]
        items = {k: inv.get(k, 0) + v for k, v in missing_requirements(skill, inv).items()}
        if skill.verb == "dormir":
            items["energy"] = int(self.np_random.integers(2, 7))
        if items:
            self.crafter.set_inventory(**items)
        self._obs, self._info = self.crafter.refresh()
        self._place_stations(stations_needed(skill))

    def _place_stations(self, stations: list[str]) -> None:
        """Pone las estaciones que falten, todas a <= 2 celdas entre sí.

        Crafter exige tener TODAS las estaciones a 1 celda para fabricar, así
        que la segunda (ej. el horno) se pone cerca de la primera (la mesa).

        Parameters
        ----------
        stations : list of str
            Estaciones que necesita la habilidad ("table", "furnace").
        """
        sem, pos = self._info["semantic"], self._info["player_pos"]
        present = [st for st in stations if is_near(sem, pos, st, self.station_radius)]
        anchor = self._nearest_cell(present[0]) if present else None
        for station in stations:
            if station not in present:
                anchor = self._place_near(station, anchor) or anchor

    def _nearest_cell(self, name: str) -> tuple[int, int]:
        """Celda de `name` más cercana al jugador (Chebyshev).

        Parameters
        ----------
        name : str
            Material interno de Crafter.

        Returns
        -------
        tuple of int
            Celda (x, y).
        """
        x0, y0 = int(self._info["player_pos"][0]), int(self._info["player_pos"][1])
        xs, ys = np.nonzero(self._info["semantic"] == MAT_ID[name])
        i = int(np.argmin(np.maximum(abs(xs - x0), abs(ys - y0))))
        return int(xs[i]), int(ys[i])

    def _place_near(self, name: str, anchor: tuple[int, int] | None = None) -> tuple[int, int] | None:
        """Pone `name` en una celda de pasto al azar cerca del jugador (o a <= 2 de `anchor`).

        Parameters
        ----------
        name : str
            Material interno de Crafter ("table" o "furnace").
        anchor : tuple of int or None, optional
            Otra estación a la que tiene que quedar cerca.

        Returns
        -------
        tuple of int or None
            Celda donde se puso, o None si no había pasto libre.
        """
        sem, player = self._info["semantic"], tuple(int(v) for v in self._info["player_pos"])
        (x0, y0), r = (anchor, 2) if anchor else (player, self.station_place_radius)
        cells = [(x, y) for x in range(max(0, x0 - r), min(sem.shape[0], x0 + r + 1))
                 for y in range(max(0, y0 - r), min(sem.shape[1], y0 + r + 1))
                 if (x, y) != player and sem[x, y] == MAT_ID["grass"]]
        if not cells:
            return None
        cell = cells[self.np_random.integers(len(cells))]
        self.crafter.set_material(cell, name)
        self._obs, self._info = self.crafter.refresh()
        return cell
