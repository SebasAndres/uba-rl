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

# Verbos cuyo logro / objetivo en el mapa no depende del objeto.
_FIXED_ACHIEVEMENT = {"beber": "collect_drink", "comer": "eat_cow", "dormir": "wake_up",
                      "pelear": "defeat_zombie"}
_FIXED_TARGET = {"beber": "water", "comer": "cow", "pelear": "zombie", "fabricar": "table"}

SLEEP_TIMEOUT = 300    # dormir dura hasta que la energía llega al máximo
DEFAULT_TIMEOUT = 100
MAX_VITAL = 9


@dataclass(frozen=True)
class Skill:
    """Una habilidad del catálogo, ej. ``Skill("fabricar", "pico", "madera")``.

    Attributes
    ----------
    verb : str
        Verbo (ver `VERBS`).
    obj : str or None
        Objeto sobre el que actúa (ver `OBJECTS`).
    material : str or None
        Material de la herramienta, sólo para "fabricar" (ver `MATERIALS`).
    """

    verb: str
    obj: str | None = None
    material: str | None = None

    def __str__(self) -> str:
        """Sintaxis que escribe el LLM, ej. "fabricar(pico, madera)"."""
        args = [a for a in (self.obj, self.material) if a]
        return f"{self.verb}({', '.join(args)})"

    # ---------------------------------------------------------------- crafter
    @property
    def achievement(self) -> str:
        """str: logro de Crafter cuyo contador sube cuando la habilidad se cumple.

        Se usan los contadores de info["achievements"] en lugar del inventario
        porque el inventario satura en 9 y se consume (ej. la madera al fabricar).
        """
        if self.verb in _FIXED_ACHIEVEMENT:
            return _FIXED_ACHIEVEMENT[self.verb]
        if self.verb == "recolectar":
            return f"collect_{_RESOURCE[self.obj]}"
        if self.verb == "colocar":
            return f"place_{_PLACEABLE[self.obj]}"
        if self.verb == "fabricar":
            return f"make_{self.crafted_item}"
        raise ValueError(self)

    @property
    def crafted_item(self) -> str:
        """str: ítem de Crafter que produce "fabricar", ej. "wood_pickaxe"."""
        return f"{_TOOL_MAT[self.material]}_{_TOOL[self.obj]}"

    @property
    def target(self) -> str | None:
        """str or None: material/objeto del mundo al que hay que ir (navegación scripteada)."""
        if self.verb == "recolectar":
            return _RESOURCE_MAT[self.obj]
        return _FIXED_TARGET.get(self.verb)

    @property
    def place_name(self) -> str:
        """str: nombre interno de lo que coloca "colocar", ej. "table"."""
        return _PLACEABLE[self.obj]

    @property
    def timeout(self) -> int:
        """int: pasos primitivos máximos para cumplir la habilidad."""
        return SLEEP_TIMEOUT if self.verb == "dormir" else DEFAULT_TIMEOUT


# SALIDA DEL LLM PLANIFICADOR: una de estas 19 habilidades por turno, escrita con
# la sintaxis de `str(skill)` en la línea "Habilidad:" (ver planner/translator.py
# para el formato y planner/parser.py para cómo se lee). Es también la ENTRADA
# del agente de habilidades, codificada con `encode` como goal_vec.
SKILLS: list[Skill] = [
    Skill("recolectar", "madera"),        # 0
    Skill("recolectar", "piedra"),        # 1  requiere pico de madera
    Skill("recolectar", "carbon"),        # 2  requiere pico de madera
    Skill("recolectar", "hierro"),        # 3  requiere pico de piedra
    Skill("recolectar", "diamante"),      # 4  requiere pico de hierro
    Skill("recolectar", "brote"),         # 5  del pasto, 10% por golpe
    Skill("beber"),                       # 6
    Skill("comer", "vaca"),               # 7
    Skill("dormir"),                      # 8
    Skill("pelear", "zombi"),             # 9
    Skill("colocar", "mesa"),             # 10
    Skill("colocar", "horno"),            # 11
    Skill("colocar", "piedra"),           # 12
    Skill("fabricar", "pico", "madera"),  # 13 cerca de una mesa
    Skill("fabricar", "pico", "piedra"),  # 14 cerca de una mesa
    Skill("fabricar", "pico", "hierro"),  # 15 cerca de mesa y horno
    Skill("fabricar", "espada", "madera"),  # 16 cerca de una mesa
    Skill("fabricar", "espada", "piedra"),  # 17 cerca de una mesa
    Skill("fabricar", "espada", "hierro"),  # 18 cerca de mesa y horno
]
assert len(set(SKILLS)) == len(SKILLS), "Habilidad repetida en SKILLS"
SKILL_NAMES = [str(s) for s in SKILLS]


def encode(skill: Skill) -> np.ndarray:
    """Codifica una habilidad como goal_vec (entrada del agente PPO).

    Parameters
    ----------
    skill : Skill
        Habilidad a codificar.

    Returns
    -------
    np.ndarray
        Vector float32 de largo GOAL_DIM: [one-hot verbo | one-hot objeto | one-hot material].
    """
    vec = np.zeros(GOAL_DIM, np.float32)
    vec[VERBS.index(skill.verb)] = 1.0
    if skill.obj is not None:
        vec[len(VERBS) + OBJECTS.index(skill.obj)] = 1.0
    if skill.material is not None:
        vec[len(VERBS) + len(OBJECTS) + MATERIALS.index(skill.material)] = 1.0
    return vec


def time_left(skill: Skill, step: int) -> np.ndarray:
    """Fracción del timeout que le queda a la habilidad (entrada del agente PPO).

    Se normaliza por `skill.timeout` para que tenga el mismo significado en
    todas las habilidades aunque tengan timeouts distintos.

    Parameters
    ----------
    skill : Skill
        Habilidad en ejecución.
    step : int
        Acciones primitivas ya tomadas en esta habilidad.

    Returns
    -------
    np.ndarray
        Vector float32 de largo 1 con valor en [0, 1] (1 = recién empieza).
    """
    return np.array([max(0.0, 1.0 - step / skill.timeout)], np.float32)


def succeeded(skill: Skill, prev_achievements: dict, achievements: dict) -> bool:
    """Indica si la habilidad se cumplió entre dos instantes.

    Parameters
    ----------
    skill : Skill
        Habilidad en ejecución.
    prev_achievements : dict
        Contadores de logros al empezar la habilidad.
    achievements : dict
        Contadores de logros actuales.

    Returns
    -------
    bool
        True si subió el contador del logro asociado a la habilidad.
    """
    key = skill.achievement
    return achievements.get(key, 0) > prev_achievements.get(key, 0)


def missing_requirements(skill: Skill, inventory: dict) -> dict[str, int]:
    """Ítems de inventario que faltan para poder ejecutar la habilidad.

    Sólo mira el inventario. Que haya una mesa cerca / un árbol a la vista lo
    resuelve la habilidad (caminando) o falla por timeout.

    Parameters
    ----------
    skill : Skill
        Habilidad a chequear.
    inventory : dict
        Inventario actual (info["inventory"]).

    Returns
    -------
    dict of str to int
        Cantidad faltante por ítem; vacío si no falta nada.
    """
    need: dict[str, int] = {}
    if skill.verb == "recolectar":
        need = dict(COLLECT[_RESOURCE_MAT[skill.obj]]["require"])
    elif skill.verb == "colocar":
        need = dict(PLACE[skill.place_name]["uses"])
    elif skill.verb == "fabricar":
        need = dict(MAKE[skill.crafted_item]["uses"])
    return {item: amount - inventory.get(item, 0)
            for item, amount in need.items() if inventory.get(item, 0) < amount}


def is_feasible(skill: Skill, inventory: dict) -> bool:
    """Indica si la habilidad se puede intentar con el inventario actual.

    Parameters
    ----------
    skill : Skill
        Habilidad a chequear.
    inventory : dict
        Inventario actual (info["inventory"]).

    Returns
    -------
    bool
        True si no falta ningún ítem (dormir sólo si la energía no está al máximo).
    """
    if skill.verb == "dormir":
        return inventory.get("energy", 0) < MAX_VITAL
    return not missing_requirements(skill, inventory)


def stations_needed(skill: Skill) -> list[str]:
    """Estaciones (table/furnace) que tienen que estar a 1 celda para fabricar.

    Parameters
    ----------
    skill : Skill
        Habilidad a chequear.

    Returns
    -------
    list of str
        Estaciones necesarias; vacío si la habilidad no es "fabricar".
    """
    if skill.verb != "fabricar":
        return []
    return list(MAKE[skill.crafted_item]["nearby"])
