"""Interfaz común de los planificadores de nivel alto (LLM u oráculo)."""

from dataclasses import dataclass
from typing import Protocol

from ..missions.missions import Mission
from ..skills.catalog import Skill


@dataclass
class PlannerOutput:
    """Plan de un turno.

    Attributes
    ----------
    text : str
        Respuesta cruda (razonamiento + habilidad).
    skill : Skill or None
        Habilidad parseada; None si el parser no encontró una válida.
    """

    text: str
    skill: Skill | None


class Planner(Protocol):
    """Planificador de nivel alto: elige qué habilidad ejecutar en cada turno."""

    name: str

    def reset(self, mission: Mission) -> None:
        """Prepara al planificador para una misión nueva.

        Parameters
        ----------
        mission : Mission
            Misión del episodio.
        """
        ...

    def plan(self, text: str, info: dict) -> PlannerOutput:
        """Devuelve el plan del turno.

        Parameters
        ----------
        text : str
            Estado traducido a texto (la entrada del LLM).
        info : dict
            `info` crudo de Crafter; sólo lo usan los planificadores scripteados.

        Returns
        -------
        PlannerOutput
            Texto de la respuesta y habilidad elegida.
        """
        ...
