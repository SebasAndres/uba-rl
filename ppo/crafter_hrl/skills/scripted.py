"""Habilidades scripteadas (BFS + reglas). Sirven para:

  1. Que el pipeline completo corra HOY, sin nada entrenado.
  2. Desbloquear al resto del equipo (B y C) mientras A entrena el agente PPO.
  3. La ablación "habilidades scripteadas vs entrenadas con RL".

Hacen trampa a propósito: leen el mapa semántico completo (dentro de un radio)
y la orientación del jugador, cosas que el agente PPO no ve (sólo ve píxeles).
"""

import numpy as np

from ..env import CrafterEnv
from ..env.constants import ACTION_ID, DIR_TO_MOVE, ID_TO_NAME, PLACE
from ..env.perception import find_path_to_adjacent, is_near, name_to_ids
from .catalog import Skill, stations_needed

DO = ACTION_ID["do"]
SLEEP = ACTION_ID["sleep"]
NOOP = ACTION_ID["noop"]
_DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]


class ScriptedSkillPolicy:
    name = "scripted"

    def __init__(self, search_radius: int = 12, seed: int = 0):
        self.search_radius = search_radius
        self.rng = np.random.default_rng(seed)
        self._explore_dir = (1, 0)
        self._last_pos = None

    def reset(self, skill: Skill) -> None:
        self._explore_dir = _DIRS[self.rng.integers(4)]
        self._last_pos = None

    def act(self, env: CrafterEnv, obs: np.ndarray, info: dict, skill: Skill) -> int:
        v = skill.verb
        if v == "dormir":
            return SLEEP
        if v == "colocar":
            return self._place(env, info, skill)
        if v == "fabricar":
            sem, pos = info["semantic"], info["player_pos"]
            missing = [s for s in stations_needed(skill) if not is_near(sem, pos, s, radius=1)]
            if not missing:
                return ACTION_ID[f"make_{skill.crafted_item}"]
            return self._goto(env, info, missing[0], interact=False)
        if v == "recolectar" and skill.obj == "brote":
            # El pasto es caminable: no se "llega al lado", se golpea el de enfrente.
            return DO if self._facing_name(env, info) == "grass" else self._explore(info)
        return self._goto(env, info, skill.target, interact=True)

    # ------------------------------------------------------------------ helpers
    def _facing_cell(self, env, info):
        x, y = info["player_pos"]
        fx, fy = env.facing
        return int(x + fx), int(y + fy)

    def _facing_name(self, env, info):
        sem = info["semantic"]
        cx, cy = self._facing_cell(env, info)
        if not (0 <= cx < sem.shape[0] and 0 <= cy < sem.shape[1]):
            return None
        return ID_TO_NAME.get(int(sem[cx, cy]))

    def _goto(self, env, info, target: str, interact: bool) -> int:
        """Caminar hasta quedar al lado de `target`, mirarlo y (opcional) golpearlo."""
        if interact and self._facing_name(env, info) == target:
            return DO
        sem, pos = info["semantic"], info["player_pos"]
        path, cell = find_path_to_adjacent(sem, pos, name_to_ids(target), self.search_radius)
        if path is None:
            return self._explore(info)
        if path:
            return ACTION_ID[DIR_TO_MOVE[path[0]]]
        # Ya estamos al lado: girar hacia el objetivo (no se mueve porque está ocupado).
        d = (cell[0] - int(pos[0]), cell[1] - int(pos[1]))
        return ACTION_ID[DIR_TO_MOVE[d]] if interact else NOOP

    def _place(self, env, info, skill: Skill) -> int:
        where = PLACE[skill.place_name]["where"]
        if self._facing_name(env, info) in where:
            return ACTION_ID[f"place_{skill.place_name}"]
        # Girar hacia algún vecino donde se pueda colocar.
        sem, (x, y) = info["semantic"], info["player_pos"]
        options = [d for d in _DIRS
                   if 0 <= x + d[0] < sem.shape[0] and 0 <= y + d[1] < sem.shape[1]
                   and ID_TO_NAME.get(int(sem[x + d[0], y + d[1]])) in where]
        if options:
            return ACTION_ID[DIR_TO_MOVE[options[self.rng.integers(len(options))]]]
        return self._explore(info)

    def _explore(self, info) -> int:
        """Caminata con dirección persistente; cambia si se traba o al azar."""
        pos = tuple(int(v) for v in info["player_pos"])
        sem = info["semantic"]

        def lava(d):
            x, y = pos[0] + d[0], pos[1] + d[1]
            inside = 0 <= x < sem.shape[0] and 0 <= y < sem.shape[1]
            return not inside or ID_TO_NAME.get(int(sem[x, y])) == "lava"

        if pos == self._last_pos or lava(self._explore_dir) or self.rng.random() < 0.05:
            safe = [d for d in _DIRS if not lava(d)] or _DIRS
            self._explore_dir = safe[self.rng.integers(len(safe))]
        self._last_pos = pos
        return ACTION_ID[DIR_TO_MOVE[self._explore_dir]]
