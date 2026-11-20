from dataclasses import dataclass
from typing import Protocol

from ..missions.missions import Mission
from ..skills.catalog import Skill


@dataclass
class PlannerOutput:
    text: str                 # respuesta cruda (razonamiento + habilidad)
    skill: Skill | None       # None si el parser no encontró una habilidad válida


class Planner(Protocol):
    name: str

    def reset(self, mission: Mission) -> None: ...

    def plan(self, text: str, info: dict) -> PlannerOutput:
        """`text` es el estado traducido; `info` sólo lo usan los planners scripteados."""
        ...
