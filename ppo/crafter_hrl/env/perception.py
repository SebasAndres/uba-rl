"""Lectura del mapa semántico: qué hay cerca, en qué dirección y cómo llegar.

Lo usan el traductor (estado -> texto), las habilidades scripteadas y el oráculo.
`semantic` es info["semantic"] (array [x, y] de ids) y `pos` es info["player_pos"].
"""

from collections import deque
from dataclasses import dataclass

import numpy as np

from .constants import ID_TO_NAME, MAT_ID, OBJ_ID, VIEW_RADIUS_X, VIEW_RADIUS_Y, WALKABLE

# Lo que nunca vale la pena reportar al LLM.
_IGNORED = {"grass", "sand", "path", "player", None}
_WALKABLE_IDS = np.array([MAT_ID[m] for m in WALKABLE])


@dataclass(frozen=True)
class Seen:
    name: str          # nombre interno (tree, cow, ...)
    dx: int
    dy: int

    @property
    def dist(self) -> int:
        return abs(self.dx) + abs(self.dy)

    @property
    def direction(self) -> str:
        """Dirección cardinal en castellano (N, S, E, O, NE, ...)."""
        ns = "N" if self.dy < 0 else "S" if self.dy > 0 else ""
        eo = "E" if self.dx > 0 else "O" if self.dx < 0 else ""
        return ns + eo or "aquí"


def name_to_ids(name: str) -> list[int]:
    if name in MAT_ID:
        return [MAT_ID[name]]
    if name in OBJ_ID:
        return [OBJ_ID[name]]
    raise KeyError(name)


def visible(semantic: np.ndarray, pos, rx: int = VIEW_RADIUS_X, ry: int = VIEW_RADIUS_Y) -> list[Seen]:
    """Todo lo interesante dentro de la ventana visible, ordenado por distancia."""
    x0, y0 = int(pos[0]), int(pos[1])
    out = []
    for x in range(max(0, x0 - rx), min(semantic.shape[0], x0 + rx + 1)):
        for y in range(max(0, y0 - ry), min(semantic.shape[1], y0 + ry + 1)):
            name = ID_TO_NAME.get(int(semantic[x, y]))
            if name in _IGNORED:
                continue
            out.append(Seen(name, x - x0, y - y0))
    return sorted(out, key=lambda s: (s.dist, s.name))


def closest_per_type(seen: list[Seen]) -> dict[str, Seen]:
    best: dict[str, Seen] = {}
    for s in seen:  # ya viene ordenado por distancia
        best.setdefault(s.name, s)
    return best


def is_near(semantic: np.ndarray, pos, name: str, radius: int) -> bool:
    """¿Hay `name` a distancia de Chebyshev <= radius? (Crafter usa radius=1 para fabricar)."""
    x0, y0 = int(pos[0]), int(pos[1])
    window = semantic[max(0, x0 - radius): x0 + radius + 1, max(0, y0 - radius): y0 + radius + 1]
    return bool(np.isin(window, name_to_ids(name)).any())


def find_path_to_adjacent(semantic: np.ndarray, pos, target_ids, max_radius: int | None = None):
    """BFS sobre celdas caminables hasta quedar AL LADO de una celda objetivo.

    Devuelve (camino, celda_objetivo) donde camino es la lista de pasos (dx, dy)
    a dar, o (None, None) si no hay objetivo alcanzable dentro de max_radius.
    Los objetos (vacas, zombis) bloquean el paso; la lava se evita.
    """
    w, h = semantic.shape
    start = (int(pos[0]), int(pos[1]))
    target_mask = np.isin(semantic, target_ids)
    walk_mask = np.isin(semantic, _WALKABLE_IDS)
    dirs = ((-1, 0), (1, 0), (0, -1), (0, 1))
    prev = {start: None}
    queue = deque([start])
    while queue:
        cx, cy = queue.popleft()
        for dx, dy in dirs:
            nx, ny = cx + dx, cy + dy
            if not (0 <= nx < w and 0 <= ny < h):
                continue
            if target_mask[nx, ny]:
                path, node = [], (cx, cy)
                while prev[node] is not None:
                    px, py = prev[node]
                    path.append((node[0] - px, node[1] - py))
                    node = prev[node]
                return path[::-1], (nx, ny)
            if (nx, ny) in prev or not walk_mask[nx, ny]:
                continue
            if max_radius is not None and abs(nx - start[0]) + abs(ny - start[1]) > max_radius:
                continue
            prev[(nx, ny)] = (cx, cy)
            queue.append((nx, ny))
    return None, None
