"""Parser: texto del LLM -> Skill (o None si es inválido).

Tolerante a tildes, mayúsculas y espacios: "Fabricar(Pico,  Madera)" y
"recolectar(carbón)" son válidos. Toma la ÚLTIMA habilidad válida que aparezca
después de "Habilidad:" (o en todo el texto si no está ese prefijo).
"""

import re
import unicodedata

from ..skills.catalog import SKILLS, Skill

_BY_NAME = {str(s).replace(" ", ""): s for s in SKILLS}
_CALL = re.compile(r"([a-z]+)\s*\(([^)]*)\)")


def _normalize(text: str) -> str:
    text = unicodedata.normalize("NFKD", text.lower())
    return "".join(c for c in text if not unicodedata.combining(c))


def parse_skill(text: str) -> Skill | None:
    norm = _normalize(text)
    if "habilidad:" in norm:
        norm = norm.rsplit("habilidad:", 1)[1]
    found = None
    for verb, args in _CALL.findall(norm):
        key = f"{verb}({','.join(a.strip() for a in args.split(',') if a.strip())})"
        found = _BY_NAME.get(key, found)
    return found
