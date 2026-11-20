"""Traductor estado -> texto (input del LLM en cada turno).

Formato (ver el zoom del turno 4 en el diagrama):

    Misión: Fabricá una espada de piedra.
    Estado: vida 9/9, comida 8/9, agua 7/9, energía 9/9
    Inventario: madera 2, pico de madera 1
    Cerca: árbol (N, 1), piedra (E, 3), agua (SO, 4)
    Historial: recolectar(madera) ✓, colocar(mesa) ✓, recolectar(piedra) ✗
    Habilidades: recolectar(madera), ..., fabricar(espada, hierro)

TODO(B): iterar este formato (qué incluir, cuánto historial, si conviene
listar sólo las habilidades factibles, etc.). Lo que importa es que sea
EXACTAMENTE el mismo texto en el dataset de SFT, en PPO y en evaluación:
todos llaman a `state_to_text`.
"""

from dataclasses import dataclass

from ..env.constants import ES, VITALS
from ..env.perception import closest_per_type, visible
from ..skills.catalog import SKILLS, is_feasible

SYSTEM_PROMPT = (
    "Sos el planificador de un agente en Crafter (un Minecraft 2D). En cada turno "
    "elegís UNA habilidad para que el agente la ejecute. Respondé exactamente con dos "
    "líneas:\n"
    "Razonamiento: <una oración>\n"
    "Habilidad: <una habilidad de la lista, con la sintaxis exacta>"
)


@dataclass
class HistoryEntry:
    skill: str          # texto tal cual lo pidió el LLM (o "inválida: ...")
    success: bool

    def __str__(self) -> str:
        return f"{self.skill} {'✓' if self.success else '✗'}"


def state_to_text(info: dict, mission_text: str, history: list[HistoryEntry],
                  max_history: int = 8, list_skills: str = "factibles") -> str:
    """list_skills: "factibles" (sólo las posibles con el inventario actual) o "todas".

    Con "factibles" el LLM casi no puede pedir algo inválido y el prompt es más
    corto; con "todas" tiene que aprender los requisitos (más difícil, más castigo -0.5).
    """
    inv = info["inventory"]
    vitals = ", ".join(f"{ES[k]} {inv[k]}/9" for k in VITALS)
    items = [f"{ES[k]} {v}" for k, v in inv.items() if k not in VITALS and v > 0]
    near = closest_per_type(visible(info["semantic"], info["player_pos"]))
    near_txt = [f"{ES[s.name]} ({s.direction}, {s.dist})" for s in near.values()]
    hist = [str(h) for h in history[-max_history:]]
    skills = [str(s) for s in SKILLS if list_skills == "todas" or is_feasible(s, inv)]
    return "\n".join([
        f"Misión: {mission_text}",
        f"Estado: {vitals}",
        f"Inventario: {', '.join(items) or 'vacío'}",
        f"Cerca: {', '.join(near_txt) or 'nada a la vista'}",
        f"Historial: {', '.join(hist) or 'ninguno'}",
        f"Habilidades: {', '.join(skills)}",
    ])


def format_response(reasoning: str, skill: str) -> str:
    """Formato de salida esperado del LLM (lo usa el oráculo para generar el SFT)."""
    return f"Razonamiento: {reasoning}\nHabilidad: {skill}"
