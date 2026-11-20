"""Catálogo de habilidades: sintaxis, goal_vec, chequeo de éxito y factibilidad.

Este archivo es el CONTRATO entre el LLM (que escribe "fabricar(pico, madera)")
y el agente PPO (que recibe un goal_vec). Si se agrega/quita una habilidad,
cambia GOAL_DIM y hay que reentrenar el agente de la fase 1.

goal_vec = [one-hot verbo | one-hot objeto | one-hot material]
"""

from dataclasses import dataclass

import numpy as np

from ..env.constants import COLLECT, MAKE, PLACE

VERBS = ["recolectar", "beber", "comer", "dormir", "pelear", "colocar", "fabricar"]
OBJECTS = [
    "madera", "piedra", "carbon", "hierro", "diamante", "brote",  # recolectar
    "vaca", "zombi",                                              # comer / pelear
    "mesa", "horno",                                              # colocar (+ piedra)
    "pico", "espada",                                             # fabricar
]
MATERIALS = ["madera", "piedra", "hierro"]                        # material de fabricar
GOAL_DIM = len(VERBS) + len(OBJECTS) + len(MATERIALS)

# castellano -> nombre interno de Crafter
_RESOURCE = {"madera": "wood", "piedra": "stone", "carbon": "coal", "hierro": "iron",
             "diamante": "diamond", "brote": "sapling"}
_RESOURCE_MAT = {"madera": "tree", "piedra": "stone", "carbon": "coal", "hierro": "iron",
                 "diamante": "diamond", "brote": "grass"}
_PLACEABLE = {"mesa": "table", "horno": "furnace", "piedra": "stone"}
_TOOL = {"pico": "pickaxe", "espada": "sword"}
_TOOL_MAT = {"madera": "wood", "piedra": "stone", "hierro": "iron"}


@dataclass(frozen=True)
class Skill:
    verb: str
    obj: str | None = None
    material: str | None = None

    def __str__(self) -> str:
        args = [a for a in (self.obj, self.material) if a]
        return f"{self.verb}({', '.join(args)})"

    # ---------------------------------------------------------------- crafter
    @property
    def achievement(self) -> str:
        """Logro de Crafter cuyo contador sube cuando la habilidad se cumple.

        Usamos los contadores de info["achievements"] en lugar del inventario
        porque el inventario satura en 9 y se consume (ej. la madera al fabricar).
        """
        v, o = self.verb, self.obj
        if v == "recolectar":
            return f"collect_{_RESOURCE[o]}"
        if v == "beber":
            return "collect_drink"
        if v == "comer":
            return "eat_cow"
        if v == "dormir":
            return "wake_up"
        if v == "pelear":
            return "defeat_zombie"
        if v == "colocar":
            return f"place_{_PLACEABLE[o]}"
        if v == "fabricar":
            return f"make_{self.crafted_item}"
        raise ValueError(self)

    @property
    def crafted_item(self) -> str:
        return f"{_TOOL_MAT[self.material]}_{_TOOL[self.obj]}"

    @property
    def target(self) -> str | None:
        """Material/objeto del mundo al que hay que ir (para navegación scripteada)."""
        v = self.verb
        if v == "recolectar":
            return _RESOURCE_MAT[self.obj]
        if v == "beber":
            return "water"
        if v == "comer":
            return "cow"
        if v == "pelear":
            return "zombie"
        if v == "fabricar":
            return "table"
        return None

    @property
    def place_name(self) -> str:
        return _PLACEABLE[self.obj]

    @property
    def timeout(self) -> int:
        # Dormir dura hasta que la energía llega al máximo: necesita más pasos.
        return 300 if self.verb == "dormir" else 100


SKILLS: list[Skill] = (
    [Skill("recolectar", o) for o in _RESOURCE]
    + [Skill("beber"), Skill("comer", "vaca"), Skill("dormir"), Skill("pelear", "zombi")]
    + [Skill("colocar", o) for o in _PLACEABLE]
    + [Skill("fabricar", t, m) for t in _TOOL for m in _TOOL_MAT]
)
SKILL_NAMES = [str(s) for s in SKILLS]


def encode(skill: Skill) -> np.ndarray:
    """Skill -> goal_vec (float32, GOAL_DIM)."""
    vec = np.zeros(GOAL_DIM, np.float32)
    vec[VERBS.index(skill.verb)] = 1.0
    if skill.obj is not None:
        vec[len(VERBS) + OBJECTS.index(skill.obj)] = 1.0
    if skill.material is not None:
        vec[len(VERBS) + len(OBJECTS) + MATERIALS.index(skill.material)] = 1.0
    return vec


def succeeded(skill: Skill, prev_achievements: dict, achievements: dict) -> bool:
    key = skill.achievement
    return achievements.get(key, 0) > prev_achievements.get(key, 0)


def missing_requirements(skill: Skill, inventory: dict) -> dict[str, int]:
    """Ítems de inventario que faltan para poder ejecutar la habilidad.

    Sólo mira el inventario. Que haya una mesa cerca / un árbol a la vista lo
    resuelve la habilidad (caminando) o falla por timeout.
    """
    need: dict[str, int] = {}
    if skill.verb == "recolectar":
        need = dict(COLLECT[_RESOURCE_MAT[skill.obj]]["require"])
    elif skill.verb == "colocar":
        need = dict(PLACE[skill.place_name]["uses"])
    elif skill.verb == "fabricar":
        need = dict(MAKE[skill.crafted_item]["uses"])
    return {k: v - inventory.get(k, 0) for k, v in need.items() if inventory.get(k, 0) < v}


def is_feasible(skill: Skill, inventory: dict) -> bool:
    if skill.verb == "dormir":
        return inventory.get("energy", 0) < 9
    return not missing_requirements(skill, inventory)


def stations_needed(skill: Skill) -> list[str]:
    """Estaciones (table/furnace) que tienen que estar a 1 celda para fabricar."""
    return list(MAKE[skill.crafted_item]["nearby"]) if skill.verb == "fabricar" else []
