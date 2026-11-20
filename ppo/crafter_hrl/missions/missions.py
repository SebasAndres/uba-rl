"""Misiones: texto que recibe el LLM + chequeo de éxito sobre `info`.

`goal` es la versión estructurada que usa el oráculo para planear:
  ("item", nombre, cantidad)        -> inventory[nombre] >= cantidad
  ("achievement", nombre, cantidad) -> achievements[nombre] >= cantidad
Las misiones compuestas tienen chequeo propio (`check`) y una lista de goals.

La separación train/test es el experimento clave: el LLM se entrena (SFT y PPO)
sólo con misiones "train" y se evalúa en "test" para medir generalización.
"""

from dataclasses import dataclass, field
from typing import Callable

from ..env.perception import is_near

Goal = tuple[str, str, int]


@dataclass(frozen=True)
class Mission:
    """Una misión del LLM planificador.

    Attributes
    ----------
    key : str
        Identificador corto (ej. "espada_piedra").
    text : str
        Enunciado en castellano que se le muestra al LLM.
    goals : tuple of Goal
        Objetivos estructurados ("item" | "achievement", nombre, cantidad).
    split : str
        "train" o "test".
    extra_check : callable or None
        Chequeo adicional para misiones compuestas (ej. mesa y horno juntos).
    """

    key: str
    text: str
    goals: tuple[Goal, ...]
    split: str
    extra_check: Callable[[dict], bool] | None = field(default=None, compare=False)

    def check(self, info: dict) -> bool:
        """Indica si la misión está cumplida.

        Parameters
        ----------
        info : dict
            `info` de Crafter.

        Returns
        -------
        bool
            True si se cumplen todos los goals y el chequeo extra (si lo hay).
        """
        goals_met = all(_goal_met(goal, info) for goal in self.goals)
        return goals_met and (self.extra_check is None or self.extra_check(info))


def _goal_met(goal: Goal, info: dict) -> bool:
    """Indica si se cumple un goal individual.

    Parameters
    ----------
    goal : Goal
        Tupla (tipo, nombre, cantidad).
    info : dict
        `info` de Crafter.

    Returns
    -------
    bool
        True si el inventario/logro `nombre` alcanza `cantidad`.
    """
    kind, name, amount = goal
    counters = info["inventory"] if kind == "item" else info["achievements"]
    return counters.get(name, 0) >= amount


# ----------------------------------------------------------- chequeos compuestos
def _has_base(info: dict) -> bool:
    """Base: mesa y horno a <= 2 celdas del jugador.

    Parameters
    ----------
    info : dict
        `info` de Crafter.

    Returns
    -------
    bool
        True si hay una mesa y un horno cerca del jugador.
    """
    sem, pos = info["semantic"], info["player_pos"]
    return is_near(sem, pos, "table", 2) and is_near(sem, pos, "furnace", 2)


MISSIONS: dict[str, Mission] = {m.key: m for m in [
    # --- nativas (logros de Crafter) ---
    Mission("pico_madera", "Fabricá un pico de madera.", (("item", "wood_pickaxe", 1),), "train"),
    Mission("espada_madera", "Fabricá una espada de madera.", (("item", "wood_sword", 1),), "train"),
    Mission("espada_piedra", "Fabricá una espada de piedra.", (("item", "stone_sword", 1),), "train"),
    Mission("horno", "Colocá un horno.", (("achievement", "place_furnace", 1),), "train"),
    Mission("carbon_5", "Juntá 5 de carbón.", (("item", "coal", 5),), "train"),
    Mission("pico_piedra", "Fabricá un pico de piedra.", (("item", "stone_pickaxe", 1),), "test"),
    Mission("pico_hierro", "Fabricá un pico de hierro.", (("item", "iron_pickaxe", 1),), "test"),
    # --- compuestas (chequeo propio) ---
    Mission("base", "Prepará una base: dejá una mesa y un horno juntos y quedate al lado.",
            (("achievement", "place_table", 1), ("achievement", "place_furnace", 1)), "test",
            extra_check=_has_base),
    Mission("equipo_piedra", "Equipate con un pico de piedra y una espada de piedra.",
            (("item", "stone_pickaxe", 1), ("item", "stone_sword", 1)), "test"),
]}


def missions_by_split(split: str) -> list[Mission]:
    """Devuelve las misiones de un split.

    Parameters
    ----------
    split : str
        "train" o "test".

    Returns
    -------
    list of Mission
        Misiones de ese split, en el orden de `MISSIONS`.
    """
    return [m for m in MISSIONS.values() if m.split == split]
