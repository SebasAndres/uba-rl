"""Parser: texto del LLM -> Skill (o None si es inválido).

Tolerante a tildes, mayúsculas y espacios: "Fabricar(Pico,  Madera)" y
"recolectar(carbón)" son válidos. Toma la ÚLTIMA habilidad válida que aparezca
después de "Habilidad:" (o en todo el texto si no está ese prefijo).
"""

import re
import unicodedata

from ..skills.catalog import SKILLS, Skill

_SKILL_PREFIX = "habilidad:"
_BY_NAME = {str(s).replace(" ", ""): s for s in SKILLS}  # "fabricar(pico,madera)" -> Skill
_CALL = re.compile(r"([a-z]+)\s*\(([^)]*)\)")             # verbo(arg1, arg2)


def _normalize(text: str) -> str:
    """Pasa a minúsculas y saca tildes.

    Parameters
    ----------
    text : str
        Texto original.

    Returns
    -------
    str
        Texto normalizado (ej. "Carbón" -> "carbon").
    """
    decomposed = unicodedata.normalize("NFKD", text.lower())
    return "".join(c for c in decomposed if not unicodedata.combining(c))


def _canonical_call(verb: str, args: str) -> str:
    """Arma la clave de búsqueda de una llamada, sin espacios.

    Parameters
    ----------
    verb : str
        Verbo de la llamada.
    args : str
        Argumentos crudos entre paréntesis.

    Returns
    -------
    str
        Ej. ``"fabricar(pico,madera)"``.
    """
    clean_args = [a.strip() for a in args.split(",") if a.strip()]
    return f"{verb}({','.join(clean_args)})"


def parse_skill(text: str) -> Skill | None:
    """Extrae la habilidad pedida en la respuesta del LLM.

    Parameters
    ----------
    text : str
        Respuesta cruda del LLM.

    Returns
    -------
    Skill or None
        La última habilidad válida encontrada, o None si no hay ninguna.
    """
    norm = _normalize(text)
    if _SKILL_PREFIX in norm:
        norm = norm.rsplit(_SKILL_PREFIX, 1)[1]
    found = None
    for verb, args in _CALL.findall(norm):
        found = _BY_NAME.get(_canonical_call(verb, args), found)
    return found
