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
_EXPLORE_TURN_PROB = 0.05  # prob. de cambiar de dirección al azar mientras explora


def _inside(semantic: np.ndarray, x: int, y: int) -> bool:
    """Indica si (x, y) está dentro del mapa.

    Parameters
    ----------
    semantic : np.ndarray
        Mapa semántico [x, y] de ids.
    x, y : int
        Celda a chequear.

    Returns
    -------
    bool
        True si la celda existe en el mapa.
    """
    return 0 <= x < semantic.shape[0] and 0 <= y < semantic.shape[1]


def _cell_name(semantic: np.ndarray, x: int, y: int) -> str | None:
    """Nombre de lo que hay en la celda (x, y), o None si está fuera del mapa.

    Parameters
    ----------
    semantic : np.ndarray
        Mapa semántico [x, y] de ids.
    x, y : int
        Celda a consultar.

    Returns
    -------
    str or None
        Nombre interno de Crafter (ej. "tree"), o None.
    """
    if not _inside(semantic, x, y):
        return None
    return ID_TO_NAME.get(int(semantic[x, y]))


def _move_action(direction: tuple[int, int]) -> int:
    """Acción de movimiento hacia `direction`.

    Parameters
    ----------
    direction : tuple of int
        Paso (dx, dy) en una de las 4 direcciones.

    Returns
    -------
    int
        Índice de la acción "move_*" correspondiente.
    """
    return ACTION_ID[DIR_TO_MOVE[direction]]


class ScriptedSkillPolicy:
    """Agente de habilidades scripteado, con la misma interfaz que `PPOSkillPolicy`.

    Parameters
    ----------
    search_radius : int, optional
        Radio máximo (Manhattan) del BFS para buscar objetivos.
    seed : int, optional
        Semilla para las decisiones aleatorias (exploración, dónde colocar).
    """

    name = "scripted"

    def __init__(self, search_radius: int = 12, seed: int = 0) -> None:
        self.search_radius = search_radius
        self.rng = np.random.default_rng(seed)
        self._explore_dir = (1, 0)
        self._last_pos: tuple[int, int] | None = None

    def reset(self, skill: Skill) -> None:
        """Sortea una dirección de exploración nueva para la habilidad.

        Parameters
        ----------
        skill : Skill
            Habilidad que se va a ejecutar (no se usa).
        """
        self._explore_dir = _DIRS[self.rng.integers(4)]
        self._last_pos = None

    def act(self, env: CrafterEnv, obs: np.ndarray, info: dict, skill: Skill) -> int:
        """Elige la próxima acción primitiva para la habilidad.

        Parameters
        ----------
        env : CrafterEnv
            Entorno (se lee hacia dónde mira el jugador).
        obs : np.ndarray
            Imagen RGB (no se usa: el script lee el mapa semántico).
        info : dict
            `info` actual de Crafter.
        skill : Skill
            Habilidad en ejecución.

        Returns
        -------
        int
            Índice de la acción primitiva.
        """
        if skill.verb == "dormir":
            return SLEEP
        if skill.verb == "colocar":
            return self._place(env, info, skill)
        if skill.verb == "fabricar":
            return self._craft(env, info, skill)
        if skill.verb == "recolectar" and skill.obj == "brote":
            # El pasto es caminable: no se "llega al lado", se golpea el de enfrente.
            return DO if self._facing_name(env, info) == "grass" else self._explore(info)
        return self._goto(env, info, skill.target, interact=True)

    # ------------------------------------------------------------------ helpers
    def _facing_name(self, env: CrafterEnv, info: dict) -> str | None:
        """Nombre de lo que hay en la celda que mira el jugador.

        Parameters
        ----------
        env : CrafterEnv
            Entorno (de ahí sale la orientación).
        info : dict
            `info` actual de Crafter.

        Returns
        -------
        str or None
            Nombre interno de Crafter, o None si mira fuera del mapa.
        """
        x, y = info["player_pos"]
        fx, fy = env.facing
        return _cell_name(info["semantic"], int(x + fx), int(y + fy))

    def _craft(self, env: CrafterEnv, info: dict, skill: Skill) -> int:
        """Fabrica si las estaciones están al lado; si no, camina hacia la que falta.

        Parameters
        ----------
        env : CrafterEnv
            Entorno.
        info : dict
            `info` actual de Crafter.
        skill : Skill
            Habilidad "fabricar(...)".

        Returns
        -------
        int
            Acción "make_*" o de movimiento.
        """
        sem, pos = info["semantic"], info["player_pos"]
        missing = [s for s in stations_needed(skill) if not is_near(sem, pos, s, radius=1)]
        if not missing:
            return ACTION_ID[f"make_{skill.crafted_item}"]
        return self._goto(env, info, missing[0], interact=False)

    def _goto(self, env: CrafterEnv, info: dict, target: str, interact: bool) -> int:
        """Camina hasta quedar al lado de `target`, lo mira y (opcional) lo golpea.

        Parameters
        ----------
        env : CrafterEnv
            Entorno.
        info : dict
            `info` actual de Crafter.
        target : str
            Material u objeto destino (ej. "tree").
        interact : bool
            Si es True, golpea (`do`) al llegar.

        Returns
        -------
        int
            Acción primitiva.
        """
        if interact and self._facing_name(env, info) == target:
            return DO
        sem, pos = info["semantic"], info["player_pos"]
        path, cell = find_path_to_adjacent(sem, pos, name_to_ids(target), self.search_radius)
        if path is None:
            return self._explore(info)
        if path:
            return _move_action(path[0])
        # Ya estamos al lado: girar hacia el objetivo (no se mueve porque está ocupado).
        direction = (cell[0] - int(pos[0]), cell[1] - int(pos[1]))
        return _move_action(direction) if interact else NOOP

    def _place(self, env: CrafterEnv, info: dict, skill: Skill) -> int:
        """Coloca el objeto si la celda de enfrente lo permite; si no, gira o explora.

        Parameters
        ----------
        env : CrafterEnv
            Entorno.
        info : dict
            `info` actual de Crafter.
        skill : Skill
            Habilidad "colocar(...)".

        Returns
        -------
        int
            Acción "place_*", de giro o de exploración.
        """
        allowed = PLACE[skill.place_name]["where"]
        if self._facing_name(env, info) in allowed:
            return ACTION_ID[f"place_{skill.place_name}"]
        sem, (x, y) = info["semantic"], info["player_pos"]
        options = [d for d in _DIRS if _cell_name(sem, x + d[0], y + d[1]) in allowed]
        if options:
            return _move_action(options[self.rng.integers(len(options))])
        return self._explore(info)

    def _is_unsafe(self, semantic: np.ndarray, pos: tuple[int, int], direction: tuple[int, int]) -> bool:
        """Indica si dar un paso en `direction` sale del mapa o pisa lava.

        Parameters
        ----------
        semantic : np.ndarray
            Mapa semántico [x, y] de ids.
        pos : tuple of int
            Posición del jugador.
        direction : tuple of int
            Paso (dx, dy).

        Returns
        -------
        bool
            True si la celda destino no existe o es lava.
        """
        x, y = pos[0] + direction[0], pos[1] + direction[1]
        return not _inside(semantic, x, y) or _cell_name(semantic, x, y) == "lava"

    def _explore(self, info: dict) -> int:
        """Caminata con dirección persistente; cambia si se traba, hay lava o al azar.

        Parameters
        ----------
        info : dict
            `info` actual de Crafter.

        Returns
        -------
        int
            Acción de movimiento.
        """
        pos = tuple(int(v) for v in info["player_pos"])
        sem = info["semantic"]
        stuck = pos == self._last_pos
        if stuck or self._is_unsafe(sem, pos, self._explore_dir) or self.rng.random() < _EXPLORE_TURN_PROB:
            safe = [d for d in _DIRS if not self._is_unsafe(sem, pos, d)] or _DIRS
            self._explore_dir = safe[self.rng.integers(len(safe))]
        self._last_pos = pos
        return _move_action(self._explore_dir)
