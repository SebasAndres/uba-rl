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
from ..skills.catalog import MAX_VITAL, SKILLS, Skill, is_feasible, missing_requirements, stations_needed

# SALIDA DEL LLM: dos líneas con estos prefijos. La habilidad tiene que ser una
# de `skills.catalog.SKILLS`, escrita como `str(skill)`. Los usan el prompt de
# abajo, `format_response` (respuestas del oráculo para el SFT) y `parser.parse_skill`.
REASONING_PREFIX = "Razonamiento:"
SKILL_PREFIX = "Habilidad:"


def _recipe(skill: Skill) -> str | None:
    """Requisitos de una habilidad en texto, ej. "usa madera 1, piedra 1; mesa cerca".

    Parameters
    ----------
    skill : Skill
        Habilidad del catálogo.

    Returns
    -------
    str or None
        Requisitos (herramienta para recolectar, materiales que se consumen al
        colocar/fabricar, estaciones cerca), o None si no necesita nada.
    """
    need = missing_requirements(skill, {})  # inventario vacío = todos los requisitos
    parts = []
    if need:
        verb = "requiere" if skill.verb == "recolectar" else "usa"
        parts.append(f"{verb} " + ", ".join(f"{ES[item]} {n}" for item, n in need.items()))
    stations = stations_needed(skill)
    if stations:
        parts.append(" y ".join(ES[s] for s in stations) + " cerca")
    return "; ".join(parts) or None


def recipes_text() -> str:
    """Tabla de requisitos de todas las habilidades (va en el system prompt).

    Se arma desde `crafter.constants` vía el catálogo: sin ella el LLM no puede
    deducir planes para misiones que no vio en el SFT (ej. pico de hierro).

    Returns
    -------
    str
        Una línea "- habilidad: requisitos" por habilidad que tenga requisitos.
    """
    lines = [f"- {skill}: {recipe}" for skill in SKILLS if (recipe := _recipe(skill))]
    return "\n".join(lines)


SYSTEM_PROMPT = (
    "Sos el planificador de un agente en Crafter (un Minecraft 2D). En cada turno "
    "elegís UNA habilidad para que el agente la ejecute.\n"
    "Cómo leer el estado:\n"
    "- Cerca: dirección (N, S, E, O) y distancia en casillas de lo más cercano de cada tipo.\n"
    "- Historial: tus últimas habilidades. ✓ = se cumplió; ✗ = no se cumplió (se acabó el "
    "tiempo); \"[interrumpida: daño]\" = se cortó porque te lastimaron; \"inválida\" = no era "
    "una habilidad de la lista o faltaban materiales.\n"
    "- Habilidades: la lista de habilidades que podés pedir, con su sintaxis.\n"
    "Requisitos (\"requiere\" = herramienta que tenés que tener; \"usa\" = materiales que se "
    "gastan; \"cerca\" = tiene que estar a tu lado):\n"
    f"{recipes_text()}\n"
    "Si te falta algo, conseguí primero lo que falta (puede encadenarse: para recolectar "
    "hierro necesitás pico de piedra, que a su vez usa piedra, etc.).\n"
    "Respondé exactamente con dos líneas:\n"
    f"{REASONING_PREFIX} <una oración>\n"
    f"{SKILL_PREFIX} <una habilidad de la lista, con la sintaxis exacta>"
)


@dataclass
class HistoryEntry:
    """Una habilidad ya intentada en el episodio.

    Attributes
    ----------
    skill : str
        Texto tal cual lo pidió el LLM (o "inválida: ...").
    success : bool
        Si se cumplió.
    """

    skill: str
    success: bool

    def __str__(self) -> str:
        """Ej. "recolectar(madera) ✓"."""
        return f"{self.skill} {'✓' if self.success else '✗'}"


def _join(parts: list[str], empty: str) -> str:
    """Une con comas, o devuelve `empty` si no hay nada.

    Parameters
    ----------
    parts : list of str
        Elementos a unir.
    empty : str
        Texto a usar si la lista está vacía.

    Returns
    -------
    str
        Elementos separados por ", " o `empty`.
    """
    return ", ".join(parts) or empty


def state_to_text(info: dict, mission_text: str, history: list[HistoryEntry],
                  max_history: int = 8, list_skills: str = "factibles") -> str:
    """Traduce el estado del juego al texto que lee el LLM.

    Parameters
    ----------
    info : dict
        `info` de Crafter.
    mission_text : str
        Enunciado de la misión.
    history : list of HistoryEntry
        Habilidades intentadas en el episodio.
    max_history : int, optional
        Cuántas entradas del historial (las últimas) se muestran.
    list_skills : str, optional
        "factibles" (sólo las posibles con el inventario actual) o "todas".
        Con "factibles" el LLM casi no puede pedir algo inválido y el prompt es
        más corto; con "todas" tiene que aprender los requisitos (más difícil,
        más castigo -0.5).

    Returns
    -------
    str
        Texto de 6 líneas: misión, estado, inventario, cerca, historial y habilidades.
    """
    inv = info["inventory"]
    vitals = [f"{ES[k]} {inv[k]}/{MAX_VITAL}" for k in VITALS]
    items = [f"{ES[k]} {v}" for k, v in inv.items() if k not in VITALS and v > 0]
    nearby = closest_per_type(visible(info["semantic"], info["player_pos"]))
    nearby_txt = [f"{ES[s.name]} ({s.direction}, {s.dist})" for s in nearby.values()]
    recent = [str(h) for h in history[-max_history:]]
    skills = [str(s) for s in SKILLS if list_skills == "todas" or is_feasible(s, inv)]
    return "\n".join([
        f"Misión: {mission_text}",
        f"Estado: {', '.join(vitals)}",
        f"Inventario: {_join(items, 'vacío')}",
        f"Cerca: {_join(nearby_txt, 'nada a la vista')}",
        f"Historial: {_join(recent, 'ninguno')}",
        f"Habilidades: {', '.join(skills)}",
    ])


def format_response(reasoning: str, skill: str) -> str:
    """Formato de salida esperado del LLM (lo usa el oráculo para generar el SFT).

    Parameters
    ----------
    reasoning : str
        Una oración de razonamiento.
    skill : str
        Habilidad con la sintaxis del catálogo.

    Returns
    -------
    str
        "Razonamiento: ...\\nHabilidad: ...".
    """
    return f"{REASONING_PREFIX} {reasoning}\n{SKILL_PREFIX} {skill}"
