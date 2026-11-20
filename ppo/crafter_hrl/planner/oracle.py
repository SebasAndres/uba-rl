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

_ITEM_TO_RESOURCE = {"wood": "madera", "stone": "piedra", "coal": "carbon", "iron": "hierro",
                     "diamond": "diamante", "sapling": "brote"}
_MAT_ES = {"wood": "madera", "stone": "piedra", "iron": "hierro"}
_RESOURCE_MATERIAL = {"wood": "tree", "stone": "stone", "coal": "coal", "iron": "iron",
                      "diamond": "diamond", "sapling": "grass"}


class OraclePlanner:
    name = "oracle"

    def __init__(self, station_radius: int = 6, low_vital: int = 3, threat_dist: int = 3):
        self.station_radius = station_radius  # "tengo una mesa cerca" si está a <= este radio
        self.low_vital = low_vital
        self.threat_dist = threat_dist        # pelear si hay un zombi a <= esta distancia

    def reset(self, mission: Mission) -> None:
        self.mission = mission

    def plan(self, text: str, info: dict) -> PlannerOutput:
        skill, why = self.next_skill(info)
        return PlannerOutput(format_response(why, str(skill)), skill)

    # ---------------------------------------------------------------- lógica
    def next_skill(self, info: dict) -> tuple[Skill, str]:
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
        if self.mission.key == "base":
            return self._base(info)
        for kind, name, n in self.mission.goals:
            have = (inv if kind == "item" else info["achievements"]).get(name, 0)
            if have < n:
                return self._resolve(kind, name, info)
        # Goals cumplidos pero no el chequeo compuesto: misión sin regla en el oráculo.
        return Skill("recolectar", "madera"), "No sé cómo avanzar, junto madera."

    def _base(self, info: dict) -> tuple[Skill, str]:
        """Base = mesa y horno juntos. Primero juntar TODO, después colocar uno al lado del otro."""
        inv, sem, pos = info["inventory"], info["semantic"], info["player_pos"]
        wood, stone = PLACE["table"]["uses"]["wood"], PLACE["furnace"]["uses"]["stone"]
        if is_near(sem, pos, "table", 2):
            if inv["stone"] >= stone:
                return Skill("colocar", "horno"), "Tengo la mesa al lado, pongo el horno."
            return self._resolve("item", "stone", info, why="Me falta piedra para el horno.")
        if inv["stone"] < stone:
            return self._resolve("item", "stone", info, why="Junto piedra para el horno antes de armar la base.")
        if inv["wood"] < wood:
            return self._resolve("item", "wood", info, why="Junto madera para la mesa de la base.")
        return Skill("colocar", "mesa"), "Ya tengo todo: pongo la mesa y al lado el horno."

    def _resolve(self, kind: str, name: str, info: dict, why: str | None = None) -> tuple[Skill, str]:
        """Devuelve la próxima habilidad para conseguir `name` (recursivo sobre prerequisitos)."""
        inv, sem, pos = info["inventory"], info["semantic"], info["player_pos"]
        near = lambda station: is_near(sem, pos, station, self.station_radius)  # noqa: E731

        if kind == "achievement" and name.startswith("place_"):
            obj = name[len("place_"):]
            for item, n in PLACE[obj]["uses"].items():
                if inv[item] < n:
                    return self._resolve("item", item, info)
            es = {"table": "mesa", "furnace": "horno", "stone": "piedra"}[obj]
            return Skill("colocar", es), f"Tengo los materiales para colocar {ES[obj]}."

        if name in _ITEM_TO_RESOURCE:  # recurso básico
            for tool, n in COLLECT[_RESOURCE_MATERIAL[name]]["require"].items():
                if inv[tool] < n:
                    return self._resolve("item", tool, info)
            res = _ITEM_TO_RESOURCE[name]
            return Skill("recolectar", res), why or f"Me falta {ES[name]}."

        if name in MAKE:  # herramienta
            for station in MAKE[name]["nearby"]:
                if not near(station):
                    return self._resolve("achievement", f"place_{station}", info)
            for item, n in MAKE[name]["uses"].items():
                if inv[item] < n:
                    return self._resolve("item", item, info)
            mat, tool = name.split("_")
            skill = Skill("fabricar", "pico" if tool == "pickaxe" else "espada", _MAT_ES[mat])
            return skill, f"Tengo todo para fabricar {ES[name]}."

        raise ValueError(f"El oráculo no sabe conseguir {kind}:{name}")
