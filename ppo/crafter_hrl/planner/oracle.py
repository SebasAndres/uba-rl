"""Oráculo scripteado: resuelve el árbol tecnológico de Crafter con reglas.

Usos:
  - baseline / demo sin LLM,
  - generar el dataset de SFT (training/fase2_sft/generar_dataset.py),
  - opcional: recompensa de prerrequisitos (+0.2) en PPO.

Se re-evalúa en cada turno mirando el inventario actual, así que se recupera
solo si una habilidad falla. Lee los costos de crafter.constants.
"""

from ..env.constants import COLLECT, ES, MAKE, PLACE
from ..env.perception import is_near, visible
from ..missions.missions import Mission
from ..skills.catalog import Skill
from .base import PlannerOutput
from .translator import format_response

Decision = tuple[Skill, str]  # (habilidad, razonamiento)

_ITEM_TO_RESOURCE = {"wood": "madera", "stone": "piedra", "coal": "carbon", "iron": "hierro",
                     "diamond": "diamante", "sapling": "brote"}
_MAT_ES = {"wood": "madera", "stone": "piedra", "iron": "hierro"}
_TOOL_ES = {"pickaxe": "pico", "sword": "espada"}
_PLACEABLE_ES = {"table": "mesa", "furnace": "horno", "stone": "piedra"}
_RESOURCE_MATERIAL = {"wood": "tree", "stone": "stone", "coal": "coal", "iron": "iron",
                      "diamond": "diamond", "sapling": "grass"}
_PLACE_PREFIX = "place_"


class OraclePlanner:
    """Planificador por reglas: supervivencia primero, después el árbol tecnológico.

    Parameters
    ----------
    station_radius : int, optional
        "Tengo una mesa/horno cerca" si está a <= este radio.
    low_vital : int, optional
        Umbral de agua/comida para priorizar beber/comer.
    threat_dist : int, optional
        Pelea si hay un zombi a <= esta distancia.
    """

    name = "oracle"

    def __init__(self, station_radius: int = 6, low_vital: int = 3, threat_dist: int = 3) -> None:
        self.station_radius = station_radius
        self.low_vital = low_vital
        self.threat_dist = threat_dist

    def reset(self, mission: Mission) -> None:
        """Guarda la misión del episodio.

        Parameters
        ----------
        mission : Mission
            Misión a resolver.
        """
        self.mission = mission

    def plan(self, text: str, info: dict) -> PlannerOutput:
        """Devuelve el plan del turno en el mismo formato que el LLM.

        Parameters
        ----------
        text : str
            Estado en texto (no se usa: el oráculo lee `info`).
        info : dict
            `info` de Crafter.

        Returns
        -------
        PlannerOutput
            Respuesta "Razonamiento: ...\\nHabilidad: ..." y la habilidad.
        """
        skill, reasoning = self.next_skill(info)
        return PlannerOutput(format_response(reasoning, str(skill)), skill)

    # ---------------------------------------------------------------- lógica
    def next_skill(self, info: dict) -> Decision:
        """Elige la próxima habilidad: primero sobrevivir, después avanzar la misión.

        Parameters
        ----------
        info : dict
            `info` de Crafter.

        Returns
        -------
        Decision
            Habilidad y razonamiento en castellano.
        """
        survival = self._survival(info)
        if survival is not None:
            return survival
        if self.mission.key == "base":
            return self._base(info)
        inv = info["inventory"]
        for kind, name, amount in self.mission.goals:
            have = (inv if kind == "item" else info["achievements"]).get(name, 0)
            if have < amount:
                return self._resolve(kind, name, info)
        # Goals cumplidos pero no el chequeo compuesto: misión sin regla en el oráculo.
        return Skill("recolectar", "madera"), "No sé cómo avanzar, junto madera."

    def _survival(self, info: dict) -> Decision | None:
        """Reglas de supervivencia: pelear, beber, comer, dormir.

        Parameters
        ----------
        info : dict
            `info` de Crafter.

        Returns
        -------
        Decision or None
            La acción de supervivencia más urgente, o None si no hace falta ninguna.
        """
        inv, sem, pos = info["inventory"], info["semantic"], info["player_pos"]
        zombies = [s for s in visible(sem, pos) if s.name == "zombie" and s.dist <= self.threat_dist]
        if zombies:
            return Skill("pelear", "zombi"), f"Hay un zombi a {zombies[0].dist} casillas, lo enfrento."
        if inv["drink"] <= self.low_vital:
            return Skill("beber"), "Tengo poca agua, primero tomo."
        if inv["food"] <= self.low_vital:
            return Skill("comer", "vaca"), "Tengo hambre, busco una vaca."
        if inv["energy"] <= self.low_vital - 1:
            return Skill("dormir"), "Estoy sin energía, duermo."
        return None

    def _base(self, info: dict) -> Decision:
        """Base = mesa y horno juntos. Primero juntar TODO, después colocar uno al lado del otro.

        Parameters
        ----------
        info : dict
            `info` de Crafter.

        Returns
        -------
        Decision
            Próxima habilidad para armar la base.
        """
        inv, sem, pos = info["inventory"], info["semantic"], info["player_pos"]
        wood_needed = PLACE["table"]["uses"]["wood"]
        stone_needed = PLACE["furnace"]["uses"]["stone"]
        if is_near(sem, pos, "table", 2):
            if inv["stone"] >= stone_needed:
                return Skill("colocar", "horno"), "Tengo la mesa al lado, pongo el horno."
            return self._resolve("item", "stone", info, why="Me falta piedra para el horno.")
        if inv["stone"] < stone_needed:
            return self._resolve("item", "stone", info, why="Junto piedra para el horno antes de armar la base.")
        if inv["wood"] < wood_needed:
            return self._resolve("item", "wood", info, why="Junto madera para la mesa de la base.")
        return Skill("colocar", "mesa"), "Ya tengo todo: pongo la mesa y al lado el horno."

    def _first_missing(self, requirements: dict[str, int], info: dict) -> Decision | None:
        """Resuelve el primer ítem de `requirements` que falte en el inventario.

        Parameters
        ----------
        requirements : dict of str to int
            Cantidad necesaria por ítem.
        info : dict
            `info` de Crafter.

        Returns
        -------
        Decision or None
            Habilidad para conseguir el primer faltante, o None si no falta nada.
        """
        for item, amount in requirements.items():
            if info["inventory"][item] < amount:
                return self._resolve("item", item, info)
        return None

    def _resolve(self, kind: str, name: str, info: dict, why: str | None = None) -> Decision:
        """Devuelve la próxima habilidad para conseguir `name` (recursivo sobre prerrequisitos).

        Parameters
        ----------
        kind : str
            "item" o "achievement".
        name : str
            Ítem o logro de Crafter (ej. "stone_pickaxe", "place_table").
        info : dict
            `info` de Crafter.
        why : str or None, optional
            Razonamiento a usar si se termina recolectando un recurso.

        Returns
        -------
        Decision
            Habilidad y razonamiento.

        Raises
        ------
        ValueError
            Si el oráculo no tiene regla para `name`.
        """
        if kind == "achievement" and name.startswith(_PLACE_PREFIX):
            return self._resolve_place(name[len(_PLACE_PREFIX):], info)
        if name in _ITEM_TO_RESOURCE:
            return self._resolve_resource(name, info, why)
        if name in MAKE:
            return self._resolve_tool(name, info)
        raise ValueError(f"El oráculo no sabe conseguir {kind}:{name}")

    def _resolve_place(self, obj: str, info: dict) -> Decision:
        """Colocar `obj` (table/furnace/stone), juntando antes los materiales.

        Parameters
        ----------
        obj : str
            Nombre interno de lo que se coloca.
        info : dict
            `info` de Crafter.

        Returns
        -------
        Decision
            Habilidad y razonamiento.
        """
        missing = self._first_missing(PLACE[obj]["uses"], info)
        if missing is not None:
            return missing
        return Skill("colocar", _PLACEABLE_ES[obj]), f"Tengo los materiales para colocar {ES[obj]}."

    def _resolve_resource(self, item: str, info: dict, why: str | None) -> Decision:
        """Recolectar un recurso básico, consiguiendo antes la herramienta necesaria.

        Parameters
        ----------
        item : str
            Recurso (wood, stone, coal, ...).
        info : dict
            `info` de Crafter.
        why : str or None
            Razonamiento a usar; por defecto "Me falta <recurso>.".

        Returns
        -------
        Decision
            Habilidad y razonamiento.
        """
        missing = self._first_missing(COLLECT[_RESOURCE_MATERIAL[item]]["require"], info)
        if missing is not None:
            return missing
        return Skill("recolectar", _ITEM_TO_RESOURCE[item]), why or f"Me falta {ES[item]}."

    def _resolve_tool(self, item: str, info: dict) -> Decision:
        """Fabricar una herramienta: estaciones cerca, después materiales, después fabricar.

        Parameters
        ----------
        item : str
            Herramienta (ej. "stone_pickaxe").
        info : dict
            `info` de Crafter.

        Returns
        -------
        Decision
            Habilidad y razonamiento.
        """
        sem, pos = info["semantic"], info["player_pos"]
        for station in MAKE[item]["nearby"]:
            if not is_near(sem, pos, station, self.station_radius):
                return self._resolve("achievement", f"{_PLACE_PREFIX}{station}", info)
        missing = self._first_missing(MAKE[item]["uses"], info)
        if missing is not None:
            return missing
        material, tool = item.split("_")
        skill = Skill("fabricar", _TOOL_ES[tool], _MAT_ES[material])
        return skill, f"Tengo todo para fabricar {ES[item]}."
